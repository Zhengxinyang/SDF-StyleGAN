import torch
import torch.nn.functional as F
# import numpy as np


def xyz2index(points, resolution: int = 32, padding_size: int = 3, ext_scale: float = 0.01):
    unit_len = (2+2*ext_scale) / (resolution - 1)
    index = ((points + (1+ext_scale)) / unit_len) + padding_size
    return index


def feature_interpolation_trilinear_singel_batch(points, feature_volume):
    points = points.to(feature_volume.device)
    voxel_size = feature_volume.shape[-1]

    padding_size = 2
    feature_volume = F.pad(feature_volume, (padding_size,)*6, mode='replicate')

    index = xyz2index(points=points, resolution=voxel_size,
                      padding_size=padding_size)
    index0 = index.to(torch.long)
    index1 = index0 + 1
    x0, y0, z0 = index0[:, 0], index0[:, 1], index0[:, 2]
    x1, y1, z1 = index1[:, 0], index1[:, 1], index1[:, 2]

    x, y, z = index[:, 0] - x0, index[:, 1] - y0, index[:, 2] - z0
    result = (feature_volume[:, :, x0, y0, z0]*(1-x)*(1-y)*(1-z) +
              feature_volume[:, :, x1, y0, z0]*x*(1-y)*(1-z) +
              feature_volume[:, :, x0, y1, z0]*(1-x)*y*(1-z) +
              feature_volume[:, :, x0, y0, z1]*(1-x)*(1-y)*z +
              feature_volume[:, :, x1, y0, z1]*x*(1-y)*z +
              feature_volume[:, :, x0, y1, z1]*(1-x)*y*z +
              feature_volume[:, :, x1, y1, z0]*x*y*(1-z) +
              feature_volume[:, :, x1, y1, z1]*x*y*z)
    return result


def feature_interpolation_trilinear(points, feature_volume, points_in_batch: bool = False):
    if points_in_batch:
        batch_size = feature_volume.shape[0]
        channel = feature_volume.shape[1]
        n = points.shape[1]
        result = torch.zeros(batch_size, channel, n, device=points.device)
        for i in range(batch_size):
            result[i] = feature_interpolation_trilinear_singel_batch(
                points=points[i], feature_volume=feature_volume[i, :, :, :, :].unsqueeze(dim=0))
    else:
        result = feature_interpolation_trilinear_singel_batch(
            points=points, feature_volume=feature_volume)
    return result
