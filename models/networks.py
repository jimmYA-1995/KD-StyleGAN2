from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

from .blocks import DBlock
from .layers import *
from .utils import (
    normalize_2nd_moment,
    build_pose_encoder,
)
from torch_utils.misc import assert_shape


class MappingNetwork(nn.Module):
    def __init__(
        self,
        z_dim: int,                      # Latent vector (Z) dimensionality.
        c_dim: int,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim: int,                      # Disentangled latent (W) dimensionality.
        num_layers: int = 8,             # Number of mapping layers.
        embed_dim: int = 512,            # Dimensionality of embbeding layers. Force to zero when num_classes = 1
        layer_dim: int = 512,            # Dimensionality of intermediate layers.
        lrmul: float = 0.01,             # Learning rate multiplier for the mapping layers.
        activation='lrelu',              # activation for each layer except embedding(which is linear)
        normalize_latents: bool = True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
    ) -> nn.Module:
        assert z_dim or c_dim
        super(MappingNetwork, self).__init__()
        # TODO: track W_avg & truncation cutoff

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.normalize_latents = normalize_latents

        if c_dim > 0:
            # with bias, no learning rate multiplier
            self.embed = DenseLayer(c_dim, embed_dim)
        else:
            embed_dim = 0

        fc = []
        in_dim = z_dim + embed_dim
        for idx in range(num_layers):
            out_dim = w_dim if idx == num_layers - 1 else layer_dim
            fc.append(DenseLayer(in_dim, out_dim, lrmul=lrmul, activation=activation))
            in_dim = out_dim
        self.fc = nn.Sequential(*fc)

    def forward(self, z, c, broadcast=None, normalize_z=True):
        # Normalize, Embed & Concat if needed
        x = None
        if self.z_dim > 0:
            assert z is not None
            assert z.shape[1] == self.z_dim
            x = z
            if normalize_z:
                x = normalize_2nd_moment(z)

        if self.c_dim > 0:
            assert c is not None
            assert c.shape[1] == self.c_dim
            y = normalize_2nd_moment(self.embed(c))  # .to(torch.float32)
            x = torch.cat([x, y], dim=1) if x is not None else y

        x = self.fc(x)

        if broadcast:
            return x.unsqueeze(1).repeat(1, broadcast, 1)

        return x


class SynthesisNetwork(nn.Module):
    def __init__(
        self,
        w_dim: int,                        # Disentangled latent (W) dimensionality.
        img_resolution: int,               # Output resolution
        img_channels: int = 3,             # Number of output color channels.
        bottom_res: int = 4,               # Resolution of bottom layer
        pose: bool = False,                # Whether to build Pose Encoder
        const: bool = False,               # Whether to create const inputs
        pose_encoder_kwargs: dict = {},    # Kwargs for Pose Encoder when using pose encoder
        channel_base: int = 32768,         # Overall multiplier for the number of channels.
        channel_max: int = 512,            # Maximum number of channels in any layer.
        resample_filter=[1, 3, 3, 1],      # Low-pass filter to apply when resampling activations. None = no filtering.
    ):
        assert img_resolution & (img_resolution - 1) == 0
        assert pose or const
        assert bottom_res & (bottom_res - 1) == 0 and bottom_res < img_resolution
        super(SynthesisNetwork, self).__init__()

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.bottom_res = bottom_res
        self.pose = pose
        self.num_blocks = int(np.log2(img_resolution // bottom_res)) + 1
        self.block_resolutions = [bottom_res * 2 ** i for i in range(self.num_blocks)]
        self.channel_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}

        # bottom
        if pose:
            self.pose_encoder = build_pose_encoder(bottom_res, self.channel_dict[bottom_res], **pose_encoder_kwargs)
        if const:
            self.const = nn.Parameter(torch.randn([self.channel_dict[bottom_res], bottom_res, bottom_res]))

        for res in self.block_resolutions:
            out_channels = self.channel_dict[res]

            if res > bottom_res:
                in_channels = self.channel_dict[res // 2]
                self.add_module(f'b{res}_convUp', SynthesisLayer(in_channels, out_channels, w_dim, res, up=2, resample_filter=resample_filter))

            self.add_module(f'b{res}_conv', SynthesisLayer(out_channels, out_channels, w_dim, res, resample_filter=resample_filter))
            self.add_module(f'b{res}_trgb', ToRGB(out_channels, img_channels, w_dim, resample_filter=resample_filter))

    def forward(self, ws, pose=None, return_feat_res=None, **layer_kwargs):
        assert return_feat_res is None or isinstance(return_feat_res, list)
        if pose is not None:
            assert self.pose
            assert pose.shape[0] == ws.shape[0], f"{pose.shape[0]} v.s {ws.shape[0]}"
            btm_features = self.pose_encoder(pose)
        else:
            btm_features = self.const.unsqueeze(0).repeat(ws.shape[0], 1, 1, 1)

        x = btm_features
        img = None
        feats = {}
        for res in self.block_resolutions:
            res_log2 = int(np.log2(res))
            if res > self.bottom_res:
                x = getattr(self, f'b{res}_convUp')(x, ws[:, res_log2 * 2 - 5], **layer_kwargs)

            x = getattr(self, f'b{res}_conv')(x, ws[:, res_log2 * 2 - 4], **layer_kwargs)
            if return_feat_res is not None and res in return_feat_res:
                feats[res] = x

            img = getattr(self, f'b{res}_trgb')(x, ws[:, res_log2 * 2 - 3], skip=img)

        return img, feats


class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int,                        # Input latent(Z) dimension.
        w_dim: int,                        # Disentangled latent(W) dimension.
        classes: List[str],                # List of class name
        img_resolution: int,               # Output resolution.
        mode: str = 'split',               # split teacher & student
        mapping_kwargs: dict = {},         # Arguments for MappingNetwork.
        synthesis_kwargs: dict = {},       # Arguments for SynthesisNetwork.
    ):
        assert len(classes) == 2
        assert mode in ['split', 'joint']
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.classes = classes
        self.img_resolution = img_resolution
        self.mode = mode

        self.num_layers = int(np.log2(img_resolution)) * 2 - 2

        synthesis_args = (w_dim, img_resolution)

        if mode == 'joint':
            synthesis1 = SynthesisNetwork(*synthesis_args, const=True, **synthesis_kwargs)
            mapping1 = MappingNetwork(z_dim=z_dim, c_dim=2, w_dim=w_dim, **mapping_kwargs)
            self.mapping = nn.ModuleDict([[c, mapping1] for c in classes])
            self.synthesis = nn.ModuleDict([[c, synthesis1] for c in classes])
        else:
            assert len(classes) == 2, "Only support 2 classes"
            # 1: teacher Network, 2: student Network
            synthesis1 = SynthesisNetwork(*synthesis_args, const=True, **synthesis_kwargs)
            synthesis2 = SynthesisNetwork(*synthesis_args, const=True, **synthesis_kwargs)
            # self.heatmap_shape = synthesis2.pose_encoder.heatmap_shape
            mapping1 = MappingNetwork(z_dim=z_dim, c_dim=0, w_dim=w_dim, **mapping_kwargs)
            mapping2 = MappingNetwork(z_dim=z_dim, c_dim=0, w_dim=w_dim, **mapping_kwargs)
            self.mapping = nn.ModuleDict([[classes[0], mapping1], [classes[1], mapping2]])
            self.synthesis = nn.ModuleDict([[classes[0], synthesis1], [classes[1], synthesis2]])

        self.img_channels = synthesis1.img_channels
        self.block_resolutions = synthesis1.block_resolutions
        self.channel_dict = synthesis1.channel_dict

    def forward(self, z, pose, return_dlatent=False, **synthesis_kwargs) -> List[Dict[str, torch.Tensor]]:
        # TODO enable style mixing training
        assert z.shape[1] == self.z_dim
        z = normalize_2nd_moment(z.to(torch.float32))

        if self.mode == 'joint':
            c = torch.eye(len(self.classes), device=z.device).unsqueeze(1).repeat(1, z.shape[0], 1).flatten(0, 1)
            z = z.repeat(2, 1)
            ws = self.mapping[self.classes[0]](z, c, broadcast=self.num_layers, normalize_z=False).chunk(2)
            ws = {class_name: w for class_name, w in zip(self.classes, ws)}
        else:  # split
            ws = {}
            for class_name, mapping in self.mapping.items():
                ws[class_name] = mapping(z, broadcast=self.num_layers, normalize_z=False)

        img = {}
        for i, (class_name, synthesis) in enumerate(self.synthesis.items()):
            p = pose if i > 0 else None
            img[class_name], _ = synthesis(ws[class_name], pose=p)

        if return_dlatent:
            return img, _, ws
        return img, _

    @torch.no_grad()
    def inference(self, z, pose, target_class):
        """ forward only through target class"""
        assert z.shape[1] == self.z_dim
        assert target_class in self.classes
        c = None
        if self.mode == 'joint':
            c = torch.eye(len(self.classes), device=z.device)[self.classes.index(target_class)][None, ...].repeat(z.shape[0], 1)
        w = self.mapping[target_class](z, c=c, broadcast=self.num_layers)
        img, _ = self.synthesis[target_class](w, pose=pose)
        return img


class Discriminator(nn.Module):
    """ Discriminator with 2 branch on high resolution and merge on `branch_res` """
    def __init__(
        self,
        img_resolution: int,
        c_dim: int = 0,                            # Conditioning label (C) dimensionality.
        img_channels: List[int] = [3],             # img_channel for each class. It will create a branch for each class
        branch_res: int = 64,                      # separate classes until res
        top_res: int = 4,
        channel_base: int = 32768,
        channel_max: int = 512,
        cmap_dim: int = None,                      # Dimensionality of mapped conditioning label, None = default.
        mbstd_group_size: int = 4,
        mbstd_num_features: int = 1,
        resample_filter: List[int] = [1, 3, 3, 1],
        mapping_kwargs: Dict = {}
    ):
        assert top_res < branch_res <= img_resolution
        super(Discriminator, self).__init__()
        mbstd_num_channels = 1
        self.c_dim = c_dim
        self.input_shape = [None, sum(img_channels), img_resolution, img_resolution]
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.branch_res = branch_res
        self.top_res = top_res
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features
        self.resolution_log2 = int(np.log2(img_resolution))
        self.block_resolutions = [2 ** i for i in range(self.resolution_log2, int(np.log2(top_res)), -1)]
        channel_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [top_res]}

        if cmap_dim is None:
            cmap_dim = channel_dict[top_res]
        if c_dim == 0:
            # disable conditional Discriminator
            cmap_dim = 0
        else:
            assert len(img_channels) == 1, "conditional branch D. is invalid"
        self.cmap_dim = cmap_dim

        for res in self.block_resolutions:
            if res == img_resolution:
                for i, img_channel in enumerate(img_channels):
                    self.add_module(f'frgb_branch{i}', Conv2dLayer(img_channel, channel_dict[res], kernel_size=1))

            if res == branch_res and len(img_channels) > 1:
                merge_layer = Conv2dLayer(
                    channel_dict[res] * len(img_channels),
                    channel_dict[res],
                    kernel_size=1,
                    use_bias=False,
                    activation='lrelu',
                    resample_filter=resample_filter
                )
                self.add_module(f'b{res}_merge', merge_layer)

            if res > branch_res:
                for i in range(len(self.img_channels)):
                    self.add_module(f'b{res}_branch{i}', DBlock(channel_dict[res], channel_dict[res // 2]))
            else:
                self.add_module(f'b{res}', DBlock(channel_dict[res], channel_dict[res // 2]))

        # output layer
        if c_dim > 0:
            # mapping_kwargs.update(dict(w_avg=None))
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, **mapping_kwargs)

        self.conv_out = Conv2dLayer(channel_dict[top_res] + mbstd_num_channels, channel_dict[top_res], activation='lrelu')
        self.fc = DenseLayer(channel_dict[top_res] * top_res * top_res, channel_dict[top_res], activation='lrelu')
        self.joint_head = DenseLayer(channel_dict[top_res], 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, img, c=None):
        assert_shape(img, self.input_shape)
        imgs = torch.split(img, self.img_channels, dim=1)

        x = cmap = None
        for res in self.block_resolutions:
            if res == self.img_resolution:
                branches = [getattr(self, f'frgb_branch{i}')(imgs[i]) for i in range(len(self.img_channels))]
            if res == self.branch_res:
                x = branches[0] if len(self.img_channels) == 1 else getattr(self, f'b{res}_merge')(torch.cat(branches, dim=1))

            if res > self.branch_res:
                branches = [getattr(self, f'b{res}_branch{i}')(branches[i]) for i in range(len(self.img_channels))]
            else:
                assert x is not None
                x = getattr(self, f'b{res}')(x)

        if self.c_dim > 0:
            cmap = self.mapping(None, c)

        # Top res : 4x4
        x = minibatch_stddev_layer(x, self.mbstd_group_size, self.mbstd_num_features)
        x = self.conv_out(x)
        x = self.fc(x.flatten(1))
        x = self.joint_head(x)

        if cmap is not None:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return x  # flaot32
