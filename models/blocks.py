import numpy as np
import torch
import torch.nn as nn

from .layers import Conv2dLayer
from .utils import register
from torch_utils.ops import upfirdn2d


@register("content")
class ContentEncoder(nn.Module):
    def __init__(self, res_in, res_out, nf_in=3, max_nf=512):
        assert (res_in / res_out) % 2 == 0
        super(ContentEncoder, self).__init__()
        num_layer = int(np.log2(res_in / res_out))
        self.num_layer = num_layer

        res = res_in
        for i in range(num_layer):
            nf_out = min(2 ** (i + 7), max_nf)  # to get half channels from synthesis layers
            res = res // 2
            setattr(self, f'b{res}', Conv2dLayer(nf_in, nf_out, mode='down'))
            nf_in = nf_out
        assert res == res_out, res

    def forward(self, x):
        res = x.shape[2]
        outs = {}
        for i in range(self.num_layer):
            res = res // 2
            conv_down = getattr(self, f'b{res}')
            x = conv_down(x)
            outs[f'b{res}'] = x

        return outs


@register("pose")
class HeatmapEncoder(nn.Module):
    """ Wrapper of PoseEncoder which separate face & body part

    """
    def __init__(self, out_dim, out_channels, max_channels=512, heatmap_size=256, in_channels=17):
        assert heatmap_size > out_dim
        assert (out_dim & (out_dim - 1)) == 0
        assert (heatmap_size & (heatmap_size - 1)) == 0
        super(HeatmapEncoder, self).__init__()
        self.heatmap_shape = (in_channels, heatmap_size, heatmap_size)
        self.in_channels = in_channels
        self.num_blocks = int(np.log2(heatmap_size // out_dim))
        self.out_channels = {f"b{i}": min(32 * (2 ** i), max_channels) for i in range(1, self.num_blocks)}
        self.out_channels[f"b{self.num_blocks}"] = out_channels

        for i in range(1, self.num_blocks + 1):
            out_channels = self.out_channels[f'b{i}']
            self.add_module(f"b{i}", DBlock(in_channels, out_channels))
            in_channels = out_channels

    def forward(self, heatmaps):
        assert heatmaps.shape[1] == self.in_channels, heatmaps.shape
        x = heatmaps
        for i in range(1, self.num_blocks + 1):
            x = getattr(self, f"b{i}")(x)

        return x


@register("pose")
class HeatmapSplitsEncoder(nn.Module):
    """ Wrapper of PoseEncoder which separate face & body part

    """
    def __init__(self, out_dim, out_channels, max_channels=512, heatmap_size=256, channel_splits=[5, 12]):
        assert heatmap_size > out_dim
        assert (out_dim & (out_dim - 1)) == 0
        assert (heatmap_size & (heatmap_size - 1)) == 0
        super(HeatmapSplitsEncoder, self).__init__()
        self.in_channel_splits = channel_splits
        self.num_blocks = int(np.log2(heatmap_size // out_dim))
        self.out_channels = {f"b{i}": min(32 * (2 ** i), max_channels) for i in range(1, self.num_blocks)}
        self.out_channels[f"b{self.num_blocks}"] = out_channels

        cur_channels = self.in_channel_splits
        for i in range(1, self.num_blocks + 1):
            remain_channels = out_channels = self.out_channels[f'b{i}']
            tmp_channels = []
            for j, in_ch in enumerate(cur_channels, 1):
                if j == len(cur_channels):
                    out_ch = remain_channels
                else:
                    out_ch = int(out_channels * in_ch / sum(cur_channels))
                    out_ch = min(out_ch, remain_channels)
                    remain_channels -= out_ch

                tmp_channels.append(out_ch)
                self.add_module(f"part{i}-b{j}", DBlock(in_ch, out_ch))
            cur_channels = tmp_channels
        self.out_channel_splits = cur_channels

    @property
    def num_splits(self):
        return len(self.in_channel_splits)

    def forward(self, heatmaps):
        assert heatmaps.shape[1] == sum(self.in_channel_splits), heatmaps.shape
        xs = torch.split(heatmaps, self.in_channel_splits, dim=1)
        for i in range(1, self.num_blocks + 1):
            xs = [getattr(self, f"part{i}-b{j}")(x) for j, x in enumerate(xs, 1)]

        return torch.cat(xs, dim=1)


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu', resample_filter=[1, 3, 3, 1]):
        super(DBlock, self).__init__()
        self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, use_bias=False)

        self.conv = Conv2dLayer(in_channels, in_channels, activation=activation, resample_filter=resample_filter)
        self.conv_down = Conv2dLayer(in_channels, out_channels, mode='down', activation=activation, resample_filter=resample_filter)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

    def forward(self, x):
        skip = x

        x = self.conv(x)
        x = self.conv_down(x)
        skip = upfirdn2d.downsample2d(x=skip, f=self.resample_filter, down=2)
        skip = self.skip(skip)

        return x + skip
