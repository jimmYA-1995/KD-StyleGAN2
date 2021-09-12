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
        normalize_latents: bool = True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
    ) -> nn.Module:
        super(MappingNetwork, self).__init__()
        # TODO: track W_avg & truncation cutoff

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim if num_classes > 1 else 0
        self.layer_dim = layer_dim
        self.normalize_latents = normalize_latents

        if num_classes > 1:
            # with bias, no learning rate multiplier
            self.embed = DenseLayer(num_classes, embed_dim, activation='linear')

        fc = []
        in_dim = z_dim + embed_dim
        for idx in range(num_layers):
            out_dim = w_dim if idx == num_layers - 1 else layer_dim
            fc.append(DenseLayer(in_dim, out_dim, lrmul=lrmul))
            in_dim = out_dim
        self.fc = nn.Sequential(*fc)

    def forward(self, z, c, broadcast=None, normalize_z=True):
        # Normalize, Embed & Concat if needed
        x = None
        if self.z_dim > 0:
            assert z.shape[1] == self.z_dim
            x = z
            if normalize_z:
                x = normalize_2nd_moment(z)

        if self.num_classes > 1:
            assert z.shape[0] == c.shape[0]
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
        pose_on: bool = False,             # Whether to use PoseEncoder
        pose_encoder_kwargs: dict = {},    # Kwargs for Pose Encoder when using pose encoder
        channel_base: int = 32768,         # Overall multiplier for the number of channels.
        channel_max: int = 512,            # Maximum number of channels in any layer.
        architecture='skip',               # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter=[1, 3, 3, 1],      # Low-pass filter to apply when resampling activations. None = no filtering.
    ):
        assert img_resolution & (img_resolution - 1) == 0
        assert bottom_res & (bottom_res - 1) == 0 and bottom_res < img_resolution
        super(SynthesisNetwork, self).__init__()

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.bottom_res = bottom_res
        self.pose_on = pose_on
        num_blocks = int(np.log2(img_resolution // bottom_res)) + 1
        self.block_resolutions = [bottom_res * 2 ** i for i in range(num_blocks)]
        self.channel_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}

        # bottom
        if pose_on:
            self.bottom = build_pose_encoder(bottom_res, self.channel_dict[bottom_res], **pose_encoder_kwargs)
            assert len(self.bottom.out_channel_splits) == 2
        else:
            self.bottom = nn.Parameter(torch.randn([self.channel_dict[bottom_res], bottom_res, bottom_res]))
        assert pose_on  # ####################################

        for res in self.block_resolutions:
            out_channels = self.channel_dict[res]

            if res > bottom_res:
                in_channels = self.channel_dict[res // 2]
                self.add_module(f'b{res}_convUp', SynthesisLayer(in_channels, out_channels, w_dim, res, up=2, resample_filter=resample_filter))
            else:
                self.add_module(f'b{res}_split', SplitSynthesisLayer(self.bottom.out_channel_splits, out_channels, w_dim, res, resample_filter=resample_filter))
            self.add_module(f'b{res}_conv', SynthesisLayer(out_channels, out_channels, w_dim, res, resample_filter=resample_filter))
            self.add_module(f'b{res}_trgb', ToRGB(out_channels, img_channels, w_dim, resample_filter=resample_filter))

    def forward(self, ws, pose=None, **layer_kwargs):
        if self.pose_on:
            assert pose is not None and all(pose.shape[0] == w.shape[0] for w in ws)
            btm_features = self.bottom(pose)
        else:
            btm_features = self.bottom.unsqueeze(0).repeat(ws.shape[0], 1, 1, 1)

        imgf = imgh = None
        for res in self.block_resolutions:
            res_log2 = int(np.log2(res))
            if res == self.bottom_res:
                face, human = getattr(self, f'b{res}_split')(btm_features, [w[:, 0] for w in ws], **layer_kwargs)
            else:
                convUp = getattr(self, f'b{res}_convUp')
                face = convUp(face, ws[0][:, res_log2 * 2 - 5], **layer_kwargs)
                human = convUp(human, ws[1][:, res_log2 * 2 - 5], **layer_kwargs)

            conv = getattr(self, f'b{res}_conv')
            face = conv(face, ws[0][:, res_log2 * 2 - 4], **layer_kwargs)
            human = conv(human, ws[1][:, res_log2 * 2 - 4], **layer_kwargs)
            imgf = getattr(self, f'b{res}_trgb')(face, ws[0][:, res_log2 * 2 - 3], skip=imgf)
            imgh = getattr(self, f'b{res}_trgb')(human, ws[1][:, res_log2 * 2 - 3], skip=imgh)

        return imgf, imgh


class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int,                        # Input latent(Z) dimension.
        w_dim: int,                        # Disentangled latent(W) dimension.
        classes: List[str],                # List of class name
        img_resolution: int,               # Output resolution.
        img_channels: int = 3,             # Number of output color channels.
        mapping_kwargs: dict = {},         # Arguments for MappingNetwork.
        synthesis_kwargs: dict = {},       # Arguments for SynthesisNetwork.
    ):
        assert len(classes) > 0
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.classes = classes
        self.img_channels = img_channels
        self.img_resolution = img_resolution

        self.num_layers = int(np.log2(img_resolution)) * 2 - 2
        self.synthesis = SynthesisNetwork(w_dim, img_resolution, img_channels=self.img_channels, **synthesis_kwargs)

        self.mapping = MappingNetwork(z_dim, w_dim, len(classes), **mapping_kwargs)

    def forward(self, z, c=None, pose=None, return_dlatent=False, **synthesis_kwargs):
        # TODO enable style mixing training
        assert z.shape[1] == self.z_dim
        z = normalize_2nd_moment(z.to(torch.float32))

        if self.synthesis.pose_on:
            ws = self.get_all_dlatents(z)  # List[torch.Tensor]
        else:
            assert c is not None
            ws = self.mapping(z, c, broadcast=self.num_layers, normalize_z=False)

        face, human = self.synthesis(ws, pose=pose, **synthesis_kwargs)
        imgs = torch.cat([face, human], dim=1)

        if return_dlatent:
            return imgs, ws
        return imgs

    def get_all_dlatents(self, z) -> List[torch.Tensor]:
        """ Get dlatents for all classes at one forward pass. """

        ws = []
        cs = torch.eye(len(self.classes), device=z.device).unsqueeze(1).repeat(1, z.shape[0], 1)
        for c in cs:
            w = self.mapping(z, c, broadcast=self.num_layers, normalize_z=False)
            ws.append(w)

        return ws


class Discriminator(nn.Module):
    def __init__(
        self,
        num_classes,
        img_resolution,
        img_channels=3,
        fmap_base=16 << 10,
        fmap_decay=1.0,
        fmap_min=1,
        fmap_max=512,
        mbstd_group_size=4,
        mbstd_num_features=1,
        resample_kernel=[1, 3, 3, 1],
    ):
        assert isinstance(num_classes, int) and num_classes >= 1
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        mbstd_num_channels = 1
        self.img_channels = img_channels
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features
        self.resolution_log2 = int(np.log2(img_resolution))

        def nf(stage):
            scaled = int(fmap_base / (2.0 ** (stage * fmap_decay)))
            return np.clip(scaled, fmap_min, fmap_max)

        self.frgb = Conv2dLayer(self.img_channels, nf(self.resolution_log2 - 1), kernel_size=1)

        self.blocks = nn.ModuleList()
        for res in range(self.resolution_log2, 2, -1):
            self.blocks.append(DBlock(nf(res - 1), nf(res - 2)))

        # output layer
        self.conv_out = Conv2dLayer(nf(1) + mbstd_num_channels, nf(1))
        self.dense_out = DenseLayer(512 * 4 * 4, nf(0))
        self.label_out = DenseLayer(nf(0), num_classes)

    def forward(self, img, labels_in=None):
        assert img.shape[1] == self.img_channels, f"(D) channel unmatched. {img.shape[1]} v.s. {self.img_channels}"

        x = self.frgb(img)
        for block in self.blocks:
            x = block(x)

        # TODO: FRGB if skip
        x = minibatch_stddev_layer(x, self.mbstd_group_size, self.mbstd_num_features)
        x = self.conv_out(x)
        x = x.view(x.shape[0], -1)
        x = self.dense_out(x)
        out = self.label_out(x)
        if labels_in is not None:
            out = torch.mean(out * labels_in, dim=1, keepdims=True)

        return out
