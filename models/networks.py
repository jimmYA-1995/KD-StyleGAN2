import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import List, Dict

from .blocks import DBlock
from .layers import *
from .utils import (
    normalize_2nd_moment,
    build_pose_encoder,
)


class MappingNetwork(nn.Module):
    def __init__(
        self,
        z_dim: int,                      # Latent vector (Z) dimensionality.
        w_dim: int,                      # Disentangled latent (W) dimensionality.
        num_classes: int,                # Number of classes.
        num_layers: int = 8,             # Number of mapping layers.
        embed_dim: int = 512,            # Dimensionality of embbeding layers. Force to zero when num_classes = 1
        layer_dim: int = 512,            # Dimensionality of intermediate layers.
        lrmul: float = 0.01,             # Learning rate multiplier for the mapping layers.
        activation='lrelu',              # activation for each layer except embedding(which is linear)
        normalize_latents: bool = True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
    ) -> nn.Module:
        super(MappingNetwork, self).__init__()
        # TODO: track W_avg & truncation cutoff

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.normalize_latents = normalize_latents

        if num_classes > 1:
            # with bias, no learning rate multiplier
            self.embed = DenseLayer(num_classes, embed_dim)
        else:
            embed_dim = 0

        fc = []
        in_dim = z_dim + embed_dim
        for idx in range(num_layers):
            out_dim = w_dim if idx == num_layers - 1 else layer_dim
            fc.append(DenseLayer(in_dim, out_dim, lrmul=lrmul, activation=activation))
            in_dim = out_dim
        self.fc = nn.Sequential(*fc)

    def forward(self, z, c=None, broadcast=None, normalize_z=True):
        # Normalize, Embed & Concat if needed
        x = None
        if self.z_dim > 0:
            assert z.shape[1] == self.z_dim
            x = z
            if normalize_z:
                x = normalize_2nd_moment(z)

        if self.num_classes > 1:
            assert z.shape[0] == c.shape[0], f"{z.shape} v.s {c.shape}"
            assert c.shape[1] == self.num_classes
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
        return_feat_res: int = 64,         # the max resolution of features to return
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
        self.return_feat_res = return_feat_res
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

    def forward(self, ws, pose=None, on_forward=False, **layer_kwargs):
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
            if res <= self.return_feat_res and not on_forward:
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
        freeze_teacher: bool = False,      # whether to freeze teacher network
        attn_res: int = 64,                # the max resolution applied by attention
        mapping_kwargs: dict = {},         # Arguments for MappingNetwork.
        synthesis_kwargs: dict = {},       # Arguments for SynthesisNetwork.
    ):
        assert len(classes) == 2
        assert mode in ['split', 'joint']
        super(Generator, self).__init__()
        if freeze_teacher:
            assert mode == 'split', "Only supprt split mode for freeze teacher Net now"

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.classes = classes
        self.img_resolution = img_resolution
        self.mode = mode
        self.freeze_teacher = freeze_teacher
        self.attn_res = attn_res

        self.num_layers = int(np.log2(img_resolution)) * 2 - 2

        mapping_args = (z_dim, w_dim)
        synthesis_args = (w_dim, img_resolution)
        synthesis_kwargs['return_feat_res'] = attn_res
        if mode == 'joint':
            synthesis1 = SynthesisNetwork(*synthesis_args, pose=True, const=True, **synthesis_kwargs)
            self.heatmap_shape = synthesis1.pose_encoder.heatmap_shape
            mapping1 = MappingNetwork(*mapping_args, num_classes=2, **mapping_kwargs)
            self.mapping = nn.ModuleDict([[classes[0], mapping1], [classes[1], mapping1]])
            self.synthesis = nn.ModuleDict([[classes[0], synthesis1], [classes[1], synthesis1]])
        else:
            # 1: teacher Network, 2: student Network
            synthesis1 = SynthesisNetwork(*synthesis_args, const=True, **synthesis_kwargs)
            synthesis2 = SynthesisNetwork(*synthesis_args, pose=True, **synthesis_kwargs)
            self.heatmap_shape = synthesis2.pose_encoder.heatmap_shape
            mapping1 = MappingNetwork(*mapping_args, num_classes=1, **mapping_kwargs)
            mapping2 = MappingNetwork(*mapping_args, num_classes=1, **mapping_kwargs)
            self.mapping = nn.ModuleDict([[classes[0], mapping1], [classes[1], mapping2]])
            self.synthesis = nn.ModuleDict([[classes[0], synthesis1], [classes[1], synthesis2]])

        self.img_channels = synthesis1.img_channels
        self.block_resolutions = synthesis1.block_resolutions
        self.channel_dict = synthesis1.channel_dict
        for res in self.block_resolutions:
            if res > attn_res:
                break
            in_channels = self.channel_dict[res]
            self.add_module(f'{res}_atten', MultiHeadAttention('relu', 4, in_channels, in_channels // 8))  # TODO support FAVOR+

    def forward(self, z, pose, return_dlatent=False, on_forward=False, **synthesis_kwargs) -> List[Dict[str, torch.Tensor]]:
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

        img, feats = {}, {}
        for class_name, synthesis, p in zip(self.classes, self.synthesis.values(), (None, pose)):
            img[class_name], feats[class_name] = synthesis(ws[class_name], pose=p, on_forward=on_forward, **synthesis_kwargs)

        if not on_forward:
            # attention
            ref_c, c = self.classes
            feats['atten'] = {}
            for res in feats[ref_c].keys():
                feats['atten'][res] = getattr(self, f'{res}_atten')(feats[ref_c][res], feats[c][res], None)  # TODO support FAVOR+

        if return_dlatent:
            return img, feats, ws
        return img, feats

    def inference(self, z, pose):
        """ forward only through target class(i.e. human) """
        assert z.shape[1] == self.z_dim
        c = None
        if self.mode == 'joint':
            c = torch.eye(len(self.classes))[-1:].repeat(z.shape[0], 1)  # target class is last
        w = self.mapping[self.classes[1]](z, c=c, broadcast=self.num_layers)
        img, _ = self.synthesis[self.classes[1]](w, pose=pose)
        return img

    def requires_grad_with_freeze_(self, requires_grad: bool) -> None:
        for name, para in self.named_parameters():
            if self.freeze_teacher and self.classes[0] in name:
                para.requires_grad_(False)
            else:
                para.requires_grad_(requires_grad)


class Discriminator(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_resolution: int,
        img_channels: int = 3,
        top_res: int = 4,
        channel_base: int = 32768,
        channel_max: int = 512,
        mbstd_group_size: int = 4,
        mbstd_num_features: int = 1,
        resample_filter: List[int] = [1, 3, 3, 1],
    ):
        assert top_res < img_resolution
        assert isinstance(num_classes, int) and num_classes >= 1
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        mbstd_num_channels = 1
        self.img_channels = img_channels
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features
        self.resolution_log2 = int(np.log2(img_resolution))
        self.block_resolutions = [2 ** i for i in range(self.resolution_log2, int(np.log2(top_res)), -1)]
        channel_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [top_res]}

        self.frgb = Conv2dLayer(self.img_channels, channel_dict[img_resolution], kernel_size=1)
        self.blocks = nn.ModuleList()
        for res in self.block_resolutions:
            self.add_module(f'b{res}', DBlock(channel_dict[res], channel_dict[res // 2], resample_filter=resample_filter))

        # output layer
        self.conv_out = Conv2dLayer(channel_dict[top_res] + mbstd_num_channels, channel_dict[top_res], activation='lrelu')
        self.dense_out = DenseLayer(channel_dict[top_res] * top_res * top_res, channel_dict[top_res], activation='lrelu')
        self.label_out = DenseLayer(channel_dict[top_res], num_classes)

    def forward(self, img, c=None):
        assert img.shape[1] == self.img_channels, f"(D) channel unmatched. {img.shape[1]} v.s. {self.img_channels}"

        x = self.frgb(img)
        for res in self.block_resolutions:
            x = getattr(self, f'b{res}')(x)

        # TODO FRGB if skip
        x = minibatch_stddev_layer(x, self.mbstd_group_size, self.mbstd_num_features)
        x = self.conv_out(x)
        x = x.view(x.shape[0], -1)
        x = self.dense_out(x)
        out = self.label_out(x)
        if c is not None:
            out = torch.mean(out * c, dim=1, keepdims=True)

        return out
