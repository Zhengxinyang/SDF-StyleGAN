from functools import partial
from math import log2
from network.custom_layer import Blur3d
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from utils.utils import leaky_relu, exists
from utils.feture_interpolation import feature_interpolation_trilinear
from torch_utils.ops import bias_act


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = lr_mul / np.sqrt(in_dim)
        self.lr_mul = lr_mul

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.scale
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.lr_mul != 1:
                b = b * self.lr_mul

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


def normalize_2nd_moment(x, dim=1, eps=1e-6):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul=0.1):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(EqualLinear(
                emb, emb, lr_mul=lr_mul, activation='lrelu'))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = normalize_2nd_moment(x)
        return self.net(x)


class MLP_Net(nn.Module):
    def __init__(self, mlp_network_depth=0, feature_dim=16, latent_dim=128, eps=1e-6):
        super().__init__()

        layers = [nn.Linear(feature_dim, latent_dim), leaky_relu()]
        for i in range(mlp_network_depth):
            layers.extend([nn.Linear(latent_dim, latent_dim), leaky_relu()])
        layers.append(nn.Linear(latent_dim, 1))
        self.mapping = nn.Sequential(*layers)
        self.bound = 1 + eps

    def forward(self, points, feature_volume, points_in_batch: bool = False):
        feature = feature_interpolation_trilinear(
            points=points, feature_volume=feature_volume, points_in_batch=points_in_batch)
        feature = self.mapping(feature.transpose(dim0=1, dim1=2))
        return feature.transpose(dim0=1, dim1=2)


class Conv3DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps=1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(
            (out_chan, in_chan, kernel, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(
            self.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, d, h, w = x.shape
        w1 = y[:, None, :, None, None, None]
        w2 = self.weight[None, :, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            div = torch.rsqrt(
                (weights ** 2).sum(dim=(2, 3, 4, 5), keepdim=True) + self.eps)
            weights = weights * div

        x = x.reshape(1, -1, d, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(
            h, self.kernel, self.dilation, self.stride)
        x = F.conv3d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, d, h, w)
        return x


class FeatureBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, feature_dim=64):
        super().__init__()
        self.input_channel = input_channel

        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = feature_dim
        self.conv = Conv3DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            Blur3d()
        ) if upsample else None

    def forward(self, x, prev_feature, istyle):
        style = self.to_style(istyle)

        x = self.conv(x, style)

        if exists(prev_feature):
            x = x + prev_feature

        if exists(self.upsample):
            x = self.upsample(x)

        return x


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample=True, upsample_feature=True,
                 feature_dim=64, use_noise=True):
        super().__init__()
        self.use_noise = use_noise
        self.upsample = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv3DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv3DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_feature = FeatureBlock(
            latent_dim, filters, upsample_feature, feature_dim)

    def forward(self, x, prev_feature, istyle, inoise):
        if exists(self.upsample):
            x = self.upsample(x)
        if self.use_noise:
            inoise = torch.randn(inoise.shape[0], x.shape[2], x.shape[3],
                                 x.shape[4], inoise.shape[-1], device=inoise.device)
        else:
            inoise = torch.zeros(inoise.shape[0], x.shape[2], x.shape[3],
                                 x.shape[4], inoise.shape[-1], device=inoise.device)

        noise1 = self.to_noise1(inoise).permute((0, 4, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 4, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        feature = self.to_feature(x, prev_feature, istyle)
        return x, feature


class Generator(nn.Module):
    def __init__(self, vol_size, latent_dim, network_capacity=16, fmap_max=512,
                 feature_dim=16, use_noise=True):
        super().__init__()
        self.vol_size = vol_size
        self.latent_dim = latent_dim
        self.use_noise = use_noise
        self.num_layers = int(log2(vol_size) - 1)

        filters = [network_capacity * (4 ** (i + 1))
                   for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.initial_block = nn.Parameter(
            torch.randn((1, init_channels, 4, 4, 4)))

        self.initial_conv = nn.Conv3d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_feature=not_last,
                feature_dim=feature_dim,
                use_noise=use_noise
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        x = self.initial_block.expand(batch_size, -1, -1, -1, -1)
        feature = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        for style, block in zip(styles, self.blocks):
            x, feature = block(x, feature, style, input_noise)
        return feature
