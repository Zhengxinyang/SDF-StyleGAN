import torch
from torch import nn
from kornia.filters import filter3d
from utils.utils import exists


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Blur3d(nn.Module):
    def __init__(self, dowm=None):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, None, :] * \
            f[None, None, :, None] * f[None, :, None, None]
        return filter3d(x, f, normalized=True)
