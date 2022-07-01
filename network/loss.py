import torch
import torch.nn.functional as F
import numpy as np


def gradient_penalty(D, real, fake, rank):
    b, *ws = real.shape
    # print(*ws)
    initial_shape = [1] * len(ws)
    epsilon = torch.rand((b, *initial_shape)).repeat(1, *ws).cuda(rank)
    interpolated = real * epsilon + fake * (1 - epsilon)
    mixed_score = D(interpolated)
    mixed_grad = torch.autograd.grad(inputs=interpolated, outputs=mixed_score.sum(),
                                     create_graph=True, only_inputs=True)[0]
    mixed_grad = mixed_grad.view(mixed_grad.shape[0], -1)
    return torch.mean((mixed_grad.norm(2, dim=1) - 1) ** 2)


def simple_gradient_penalty(D, real, fake, rank):
    b, *ws = real.shape
    score = D(real)
    r1_grads = torch.autograd.grad(inputs=real, outputs=score.sum(),
                                   create_graph=True, only_inputs=True)[0]
    if len(ws) == 4:
        return torch.mean(r1_grads.square().sum([1, 2, 3, 4]))
    elif len(ws) == 3:
        return torch.mean(r1_grads.square().sum([1, 2, 3]))
    elif len(ws) == 1:
        return torch.mean(r1_grads.square().sum([1]))
    else:
        raise IndexError


def gen_stylegan_loss(fake, real):
    return torch.mean(F.softplus(-fake))


def dis_stylegan_loss(real, fake):
    return torch.mean(F.softplus(-real) + F.softplus(fake))


def calc_pl_lengths_3d(styles, voxels):
    device = voxels.device
    pl_noise = torch.randn(voxels.shape, device=device) / \
        np.sqrt(voxels.shape[2] * voxels.shape[3] * voxels.shape[4])
    outputs = (voxels * pl_noise).sum()
    pl_grads = torch.autograd.grad(outputs=outputs, inputs=styles,
                                   grad_outputs=torch.ones(
                                       outputs.shape, device=device),
                                   create_graph=True, only_inputs=True)[0]
    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()
