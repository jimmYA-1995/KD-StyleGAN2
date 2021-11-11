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
            y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
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

    def forward(self, ws, pose=None, out_resolution=None, **layer_kwargs):
        if pose is not None:
            assert self.pose
            assert pose.shape[0] == ws.shape[0], f"{pose.shape[0]} v.s {ws.shape[0]}"
            btm_features = self.pose_encoder(pose)
        else:
            btm_features = self.const.unsqueeze(0).repeat(ws.shape[0], 1, 1, 1)

        x = btm_features
        img = None
        for res in self.block_resolutions:
            res_log2 = int(np.log2(res))
            if res > self.bottom_res:
                x = getattr(self, f'b{res}_convUp')(x, ws[:, res_log2 * 2 - 5], **layer_kwargs)

            x = getattr(self, f'b{res}_conv')(x, ws[:, res_log2 * 2 - 4], **layer_kwargs)

            img = getattr(self, f'b{res}_trgb')(x, ws[:, res_log2 * 2 - 3], skip=img)
            if out_resolution is not None and res == out_resolution:
                break

        return img


class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int,                          # Input latent(Z) dimension.
        w_dim: int,                          # Disentangled latent(W) dimension.
        num_classes: int,                    # The number of class: target + patches
        branch_resolutions: Dict[str, int],  # output resolution for each branch
        mapping_kwargs: dict = {},           # Arguments for MappingNetwork.
        synthesis_kwargs: dict = {},         # Arguments for SynthesisNetwork.
    ):
        super(Generator, self).__init__()
        self.branches_res = branch_resolutions
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_classes = num_classes
        self.num_layers = int(np.log2(branch_resolutions['target'])) * 2 - 2

        self.synthesis = SynthesisNetwork(w_dim, branch_resolutions['target'], const=True, **synthesis_kwargs)
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=self.num_classes, w_dim=w_dim, **mapping_kwargs)
        self.img_channels = self.synthesis.img_channels
        self.block_resolutions = self.synthesis.block_resolutions
        self.channel_dict = self.synthesis.channel_dict

    def forward(self, z, c, return_dlatent=False, **synthesis_kwargs) -> List[Dict[str, torch.Tensor]]:
        assert z.shape[1] == self.z_dim
        z = normalize_2nd_moment(z.to(torch.float32))
        zs = z.repeat(len(self.branches_res), 1)

        fixed_c = torch.eye(self.num_classes, device=z.device)[:1].unsqueeze(1).repeat(1, z.shape[0], 1).flatten(0, 1)
        cs = torch.cat([fixed_c, c], dim=0)
        ws = self.mapping(zs, cs, broadcast=self.num_layers, normalize_z=False).chunk(len(self.branches_res))
        ws = {branch_name: w for branch_name, w in zip(self.branches_res.keys(), ws)}

        img = {}
        for branch_name, res in self.branches_res.items():
            img[branch_name] = self.synthesis(ws[branch_name], out_resolution=res)

        if return_dlatent:
            return img, ws
        return img

    @torch.no_grad()
    def inference(self, z, c):
        assert z.shape[1] == self.z_dim

        w = self.mapping(z, c=c, broadcast=self.num_layers)
        img = self.synthesis(w)
        return img


class Discriminator(nn.Module):
    """ Discriminator with 2 data flow (by implementing target branch & patch branch):
        One branch deal with target image(e.g. deepfashion) which usually has higher resolutions;
        The other branch deal with patch which is expected belong to part of target images.
        And we use projection to condition which patch is feeded(discrete)
     """
    def __init__(
        self,
        img_resolutions: List[int],                # img resolutions: [target, patch]
        c_dim: int,                                # Conditioning label (C) dimensionality.
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
        assert top_res < branch_res
        assert c_dim != 0, "Current implementation requires conditional setting"
        assert len(img_resolutions) == 2, "Please give two img_resolutiosn info: One is for target, the other is for patch"
        assert any(branch_res <= res for res in img_resolutions)
        super(Discriminator, self).__init__()
        self.img_resolutions = img_resolutions
        self.img_channel = 3
        self.branch_res = branch_res
        self.top_res = top_res
        self.c_dim = c_dim
        mbstd_num_channels = 1
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features
        self.resolution_log2 = int(np.log2(img_resolutions[0]))
        self.block_resolutions = [2 ** i for i in range(self.resolution_log2, int(np.log2(top_res)), -1)]
        channel_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [top_res]}

        self.cmap_dim = channel_dict[top_res]

        # from RGB
        for i, res in enumerate(img_resolutions):
            self.add_module(f'frgb_branch{i}', Conv2dLayer(self.img_channel, channel_dict[res], kernel_size=1))

        for res in self.block_resolutions:

            if res == branch_res:
                merge_layer = Conv2dLayer(
                    channel_dict[res] * len(img_resolutions),
                    channel_dict[res],
                    kernel_size=1,
                    use_bias=False,
                    activation='lrelu',
                    resample_filter=resample_filter
                )
                self.add_module(f'b{res}_merge', merge_layer)

            if res > branch_res:
                for i in range(len(img_resolutions)):
                    if res <= img_resolutions[i]:
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

    def forward(self, target, patch, c: torch.Tensor = None):
        inputs = [target, patch]
        branches = []
        for i, (res, img) in enumerate(zip(self.img_resolutions, inputs)):
            assert_shape(img, [None, 3, res, res])
            branches.append(getattr(self, f'frgb_branch{i}')(img))

        x = cmap = None
        for res in self.block_resolutions:
            if res == self.branch_res:
                assert_shape(branches[0], branches[1].shape)
                x = getattr(self, f'b{res}_merge')(torch.cat(branches, dim=1))

            if res > self.branch_res:
                for i in range(len(branches)):
                    if branches[i].shape[2] == res:
                        branches[i] = getattr(self, f'b{res}_branch{i}')(branches[i])
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
