from functools import partial
import math
from math import log2
from utils.utils import leaky_relu, exists
from network.custom_layer import Flatten, Blur3d
from torch import nn


class vol_DiscriminatorBlock(nn.Module):
  def __init__(self, input_channels, filters, downsample=True):
    super().__init__()
    self.conv_res = nn.Conv3d(input_channels, filters, 1,
                              stride=(2 if downsample else 1))

    self.net = nn.Sequential(
        nn.Conv3d(input_channels, filters, 3, padding=1),
        leaky_relu(),
        nn.Conv3d(filters, filters, 3, padding=1),
        leaky_relu()
    )

    self.downsample = nn.Sequential(
        Blur3d(),
        nn.Conv3d(filters, filters, 3, padding=1, stride=2)
    ) if downsample else None

  def forward(self, x):
    res = self.conv_res(x)
    x = self.net(x)
    if exists(self.downsample):
      x = self.downsample(x)
    return (x + res) * (1 / math.sqrt(2))


class Discriminator_3D(nn.Module):
  def __init__(self, volume_size, fmap_max=512, network_capacity=16, use_feature=False, feature_dim=4):
    super().__init__()
    num_layers = int(log2(volume_size) - 1)
    self.volume_size = volume_size
    num_init_filters = feature_dim if use_feature else 1  
    filters = [num_init_filters] + \
              [network_capacity * num_init_filters * (4 ** i)
               for i in range(num_layers + 1)]

    set_fmap_max = partial(min, fmap_max)
    filters = list(map(set_fmap_max, filters))
    chan_in_out = list(zip(filters[:-1], filters[1:]))

    blocks = []
    for _, (in_chan, out_chan) in enumerate(chan_in_out):
      block = vol_DiscriminatorBlock(in_chan, out_chan)
      blocks.append(block)

    self.blocks = nn.ModuleList(blocks)
    self.flatten = Flatten()
    self.to_logit = nn.Linear(filters[-1], 1)

  def forward(self, x):
    for block in self.blocks:
      x = block(x)
    x = self.flatten(x)
    x = self.to_logit(x)
    return x.squeeze()
