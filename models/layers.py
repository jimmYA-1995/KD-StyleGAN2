from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils.ops import bias_act, conv2d_resample, upfirdn2d


AVAILABLE_ACTIVATIONS = bias_act.activation_funcs.keys()


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bias: bool = True,
        bias_init: int = 0,
        activation: str = 'linear',
        lrmul: int = 1,  # learning rate multiplier
    ) -> nn.Module:
        assert activation in AVAILABLE_ACTIVATIONS
        super(DenseLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.activation = activation

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lrmul))
        self.weight_gain = lrmul / np.sqrt(in_dim)
        if use_bias:
            self.bias = nn.Parameter(torch.full([out_dim], np.float32(bias_init)))
            self.bias_gain = lrmul

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_dim}, {self.out_dim}, bias={self.use_bias}, act={self.activation})')

    def forward(self, x):
        w = (self.weight * self.weight_gain).to(x.dtype)
        b = (self.bias * self.bias_gain).to(x.dtype) if self.use_bias else None

        if self.activation == 'linear' and self.use_bias:
            return torch.addmm(b.unsqueeze(0), x, w.t())

        x = F.linear(x, w)
        return bias_act.bias_act(x, b.to(x.dtype), act=self.activation)


class Conv2dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        use_bias=True,
        activation='linear',
        mode=None,
        resample_filter=[1, 3, 3, 1],
    ):
        assert activation in AVAILABLE_ACTIVATIONS
        assert mode in ['up', 'down', None]
        super(Conv2dLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.activation = activation
        self.mode = mode
        self.up = 2 if self.mode == "up" else 1
        self.down = 2 if self.mode == "down" else 1
        self.padding = kernel_size // 2

        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.weight = nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros([out_channels]))

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.out_channels}, {self.in_channels}, {self.kernel_size}, mode={self.mode})')

    def forward(self, x):
        weight = self.weight * self.weight_gain
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=self.resample_filter,
                                            up=self.up, down=self.down, padding=self.padding, flip_weight=(self.up == 1))

        b = self.bias.to(x.dtype) if self.use_bias else None
        return bias_act.bias_act(x, b, act=self.activation)


def modulated_conv2d(x, weight, styles, up=1, down=1, padding=0, resample_filter=None, demodulate=True, flip_weight=True):
    batch_size = x.shape[0]
    out_channels, in_channels, kw, kh = weight.shape
    assert x.shape[1] == weight.shape[1]
    assert styles.shape == (batch_size, in_channels)

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(weight.shape[1:].numel()) / weight.norm(float('inf'), dim=[1, 2, 3], keepdim=True))  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)  # max_I

    w = weight.unsqueeze(0) * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoef = torch.rsqrt(torch.sum(w ** 2, dim=[2, 3, 4]) + 1e-8)  # [NO]
        w = w * dcoef.reshape(batch_size, -1, 1, 1, 1)

    # group convolution for per-instance convolution
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, *weight.shape[1:])
    x = conv2d_resample.conv2d_resample(
        x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)

    return x.reshape(batch_size, -1, *x.shape[2:])


def split_modulated_conv2d(x, weight, style_list, channel_split, up=1, down=1, padding=0, resample_filter=None, demodulate=True, flip_weight=True):
    batch_size = x.shape[0]
    assert x.shape[1] == weight.shape[1]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(weight.shape[1:].numel()) / weight.norm(float('inf'), dim=[1, 2, 3], keepdim=True))  # max_Ikk
        style_list = [styles / styles.norm(float('inf'), dim=1, keepdim=True) for styles in style_list]  # max_I

    weights = torch.split(weight, channel_split, dim=1)
    assert all([styles.shape == (batch_size, weight.shape[1]) for styles, weight in zip(style_list, weights)])

    ws = []
    for w, styles in zip(weights, style_list):
        w = w.unsqueeze(0) * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOI'kk]
        # TODO: to decide demodulate on whole channels or on each splits
        # If demodulate on whole channels, face part
        if demodulate:
            dcoef = torch.rsqrt(torch.sum(w ** 2, dim=[2, 3, 4]) + 1e-8)  # [NO]
            w = w * dcoef.reshape(batch_size, -1, 1, 1, 1)
        ws.append(w)
    w = torch.cat(ws, dim=2)

    # group convolution for per-instance convolution [1, NI, h, w] + [NO, I, k, k] -> [1, NO, h, w]
    face_part = channel_split[0]
    face = x[:, :face_part].reshape(1, -1, *x.shape[2:])
    human = x.view(1, -1, *x.shape[2:])
    w = w.reshape(-1, *weight.shape[1:]).to(x.dtype)

    conv_kwargs = dict(f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)

    face = conv2d_resample.conv2d_resample(x=face, w=w[:, :face_part], **conv_kwargs)
    human = conv2d_resample.conv2d_resample(x=human, w=w, **conv_kwargs)
    face = face.reshape(batch_size, -1, *x.shape[2:])
    human = human.reshape(batch_size, -1, *x.shape[2:])

    return face, human


class SynthesisLayer(nn.Module):
    """ Layer capsulates modulate convolution layer,
        nonlinearilty and noise layer.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        resolution,
        kernel_size=3,
        up=1,
        resample_filter=[1, 3, 3, 1],    # Low-pass filter to apply when resampling activations.
        use_noise=True,
        activation='lrelu',
        conv_clamp=None                  # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert activation in AVAILABLE_ACTIVATIONS
        super(SynthesisLayer, self).__init__()
        self.use_noise = use_noise
        self.activation = activation
        self.w_dim = w_dim
        self.up = up
        self.padding = kernel_size // 2
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

        in_resolution = resolution // up
        self.input_shape = (None, in_channels, in_resolution, in_resolution)
        self.output_shape = (None, out_channels, resolution, resolution)

        self.affine = DenseLayer(w_dim, in_channels, bias_init=1)  # init. bias to 1 instead of 0
        # Doesn't apply weight_gain(equalized learning rate) for sysnthesis layer
        # because the weight of this layer always be demodulate (?)
        self.weight = nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))  # already contiguous
        self.bias = nn.Parameter(torch.zeros([out_channels]))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))  # only used by G_ema
            self.noise_strength = nn.Parameter(torch.zeros([]))

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.input_shape} -> {self.output_shape}, act={self.activation}, noise={self.use_noise})')

    def forward(self, x, w, noise_mode='random', gain=1):
        assert noise_mode in ['random', 'const']
        assert x.shape[1:] == self.input_shape[1:]

        styles = self.affine(w)

        flip_weight = (self.up == 1)  #
        x = modulated_conv2d(x, self.weight, styles, up=self.up, padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight)

        if self.use_noise:
            noise = self.noise_const if noise_mode == 'const' else torch.randn([x.shape[0], 1, *self.output_shape[2:]], device=x.device)
            noise = noise * self.noise_strength
            x = x.add_(noise)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        return bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)


class SplitSynthesisLayer(nn.Module):
    """ Layer capsulates modulate convolution layer,
        nonlinearilty and noise layer.
    """
    def __init__(
        self,
        in_channel_splits: List[int],
        out_channels,
        w_dim,
        resolution,
        kernel_size=3,
        up=1,
        resample_filter=[1, 3, 3, 1],    # Low-pass filter to apply when resampling activations.
        use_noise=True,
        activation='lrelu',
        conv_clamp=None                  # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert activation in AVAILABLE_ACTIVATIONS
        super(SplitSynthesisLayer, self).__init__()
        in_channels = sum(in_channel_splits)
        self.in_channel_splits = in_channel_splits
        self.use_noise = use_noise
        self.activation = activation
        self.w_dim = w_dim
        self.up = up
        self.padding = kernel_size // 2
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

        in_resolution = resolution // up
        self.input_shape = (None, in_channels, in_resolution, in_resolution)
        self.output_shape = (None, out_channels, resolution, resolution)

        self.affines = nn.ModuleList()
        for in_ch in in_channel_splits:
            self.affines.append(DenseLayer(w_dim, in_ch, bias_init=1))  # init. bias to 1 instead of 0)

        # Doesn't apply weight_gain(equalized learning rate) for sysnthesis layer
        # because the weight of this layer always be demodulate (?)
        self.weight = nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))  # already contiguous
        self.bias = nn.Parameter(torch.zeros([out_channels]))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        if use_noise:
            self.register_buffer('noise_const', torch.randn([1, 1, resolution, resolution]))  # only used by G_ema
            self.noise_strength = nn.Parameter(torch.zeros([1]))

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.input_shape} -> {self.output_shape}, act={self.activation}, noise={self.use_noise})')

    def forward(self, x, ws, noise_mode='random', gain=1):
        assert noise_mode in ['random', 'const']
        assert x.shape[1:] == self.input_shape[1:], f"{x.shape} v.s {self.input_shape} ({self.in_channel_splits})"
        assert len(ws) == len(self.in_channel_splits)

        style_list = [affine(w) for affine, w in zip(self.affines, ws)]

        flip_weight = (self.up == 1)  #
        face, human = split_modulated_conv2d(x, self.weight, style_list, self.in_channel_splits, up=self.up, padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight)

        if self.use_noise:
            noise = self.noise_const if noise_mode == 'const' else torch.randn([x.shape[0], 1, *self.output_shape[2:]], device=x.device)
            noise = noise * self.noise_strength
            # TODO: which one to add noise?
            face = face.add_(noise)
            human = human.add_(noise)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        face = bias_act.bias_act(face, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        human = bias_act.bias_act(human, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return face, human


class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, resample_filter=[1, 3, 3, 1], conv_clamp=None):
        super(ToRGB, self).__init__()
        self.affine = DenseLayer(w_dim, in_channels, bias_init=1)
        self.weight = nn.Parameter(torch.randn([out_channels, in_channels, 1, 1]))
        self.bias = nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

    def forward(self, x, w, skip=None):
        styles = self.affine(w) * self.weight_gain
        weight = self.weight
        x = modulated_conv2d(x, weight, styles, demodulate=False)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)

        if skip is not None:
            skip = upfirdn2d.upsample2d(x=skip, f=self.resample_filter, up=2)
            x = x + skip

        return x


def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    b, c, h, w = x.shape
    group_size = np.minimum(group_size, b)
    if b % group_size != 0:
        group_size = b
    y = x.reshape(group_size, -1, num_new_features, c // num_new_features, h, w).float()
    y = y - y.mean(dim=0, keepdims=True)
    y = torch.sqrt(torch.mean(y**2, dim=0) + 1e-8)
    y = torch.mean(y, dim=[2, 3, 4], keepdims=True).squeeze(2)
    y = y.type(x.dtype)
    y = y.repeat(group_size, 1, h, w)
    return torch.cat([x, y], dim=1)


def valid_feature_type(feature_type):
    bool1 = feature_type in ['relu', 'elu+1', 'sqr', 'favor+']
    bool2 = feature_type.startswith('favor+') and feature_type.split(
        '_')[1].isdigit()
    return bool1 or bool2


class MultiHeadAttention(nn.Module):
    """Explicit multihead bidirectional attention."""

    def __init__(self, feature_type, n_heads, in_dim, hidden_dim):
        assert valid_feature_type(feature_type)
        super(MultiHeadAttention, self).__init__()

        self._feature_type = feature_type
        self._n_heads = n_heads
        self._hidden_dim = hidden_dim

        self.q_map = nn.Conv2d(in_dim, hidden_dim, 1)
        self.k_map = nn.Conv2d(in_dim, hidden_dim, 1)

    def forward(self, x, y, rfs):
        qs, ks, vs = self._get_queries_keys_values(x, y, rfs)
        R = torch.einsum('bsij,bsik->bsijk', ks, vs)  # [Batch, Seq, n_heads, feat_dim], [Batch, Seq, n_heads, head_dim]
        num = torch.einsum('bSijk,bsij->bsik', R, qs)

        s = ks.sum(dim=1, keepdim=True)
        den = torch.einsum('bsij,bsij->bsi', s, qs)

        outputs = num / (den[Ellipsis, None] + 1e-16)
        outputs = outputs.reshape(x.shape)

        return outputs

    def _get_queries_keys_values(self, x, y, rfs):
        queries = self.q_map(x)
        queries = queries.permute(0, 2, 3, 1).reshape([x.shape[0], x.shape[2] * x.shape[3], self._n_heads, -1])

        keys = self.k_map(y)
        keys = keys.permute(0, 2, 3, 1).reshape([y.shape[0], y.shape[2] * y.shape[3], self._n_heads, -1])

        values = y
        values = values.permute(0, 2, 3, 1).reshape([y.shape[0], y.shape[2] * y.shape[3], self._n_heads, -1])

        if self._feature_type == 'relu':
            queries = nn.functional.relu(queries)
            keys = nn.functional.relu(keys)
        elif self._feature_type == 'elu+1':
            queries = nn.functional.elu(queries) + 1
            keys = nn.functional.elu(keys) + 1
        elif self._feature_type == 'sqr':
            queries = queries**2
            keys = keys**2
        elif self._feature_type == 'abs':
            queries = torch.abs(queries)
            keys = torch.abs(keys)
        else:

            head_dim = self._hidden_dim // self._n_heads

            queries = queries * np.power(head_dim, -0.25)
            queries = torch.einsum('ijkl,klm->ijkm', queries, rfs) - (queries**2).sum(3, keepdim=True) / 2
            queries = torch.exp(queries)

            keys = keys * np.power(head_dim, -0.25)
            keys = torch.einsum('ijkl,klm->ijkm', keys, rfs) - (keys**2).sum(3, keepdim=True) / 2
            keys = torch.exp(keys)

        return queries, keys, values

    def sample_rfs(self, device):

        if not self._feature_type.startswith('favor+'):
            return None

        if self._feature_type == 'favor+':
            factor = 1
        else:
            splitted = self._feature_type.split('_')
            factor = int(splitted[1])

        head_dim = self._hidden_dim // self._n_heads

        rfs = [[
            _sample_orth_matrix(head_dim, device)[None, Ellipsis] for _ in range(factor)
        ] for _ in range(self._n_heads)]
        rfs = [torch.cat(x, 2) for x in rfs]
        rfs = torch.cat(rfs, 0)
        rfs = rfs * np.sqrt(head_dim)

        return rfs


def _sample_orth_matrix(size, device):
    """Samples orthogonal matrix to reduce variance for random features."""
    subspace = torch.randn(size, size, device=device)
    subspace = torch.tril(subspace)
    subspace = subspace / torch.sqrt((subspace**2).sum(0, keepdim=True))

    S = torch.triu(subspace.T.mm(subspace)) - 0.5 * torch.eye(
        subspace.shape[1], device=device)

    result = torch.eye(
        subspace.shape[0], device=device) - subspace.mm(torch.inverse(S)).mm(
            subspace.T)

    return result
