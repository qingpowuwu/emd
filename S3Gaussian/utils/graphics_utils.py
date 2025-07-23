#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
from typing import Tuple
from torch import Tensor
from typing import List, Union

def contract(
    x: Tensor,
    aabb: Tensor,
    ord: Union[str, int] = None,
) -> Tensor:
    """
    Contract the input tensor to the unit cube using piecewise projective function.
    """
    # similar to the one in MeRF paper
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)  # 0~1
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    x = torch.where(mag < 1, x, (2 - 1 / mag) * (x / mag))
    x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    return x

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    # 用于从 w2c 中提取的 R 和 t，得到 平移和缩放后的 c2w
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def get_rays(
    x: Tensor, y: Tensor, c2w: Tensor, intrinsic: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        x: the horizontal coordinates of the pixels, shape: (num_rays,)
        y: the vertical coordinates of the pixels, shape: (num_rays,)
        c2w: the camera-to-world matrices, shape: (num_cams, 4, 4)
        intrinsic: the camera intrinsic matrices, shape: (num_cams, 3, 3)
    Returns:
        origins: the ray origins, shape: (num_rays, 3)
        viewdirs: the ray directions, shape: (num_rays, 3)
        direction_norm: the norm of the ray directions, shape: (num_rays, 1)
    """
    if len(intrinsic.shape) == 2:
        intrinsic = intrinsic[None, :, :]
    if len(c2w.shape) == 2:
        c2w = c2w[None, :, :]
    camera_dirs = torch.nn.functional.pad(
        torch.stack(
            [
                (x - intrinsic[:, 0, 2] + 0.5) / intrinsic[:, 0, 0],
                (y - intrinsic[:, 1, 2] + 0.5) / intrinsic[:, 1, 1],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [num_rays, 3]
    # rotate the camera rays w.r.t. the camera pose
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    # TODO: not sure if we still need direction_norm
    direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
    # normalize the ray directions
    viewdirs = directions / (direction_norm + 1e-8)
    return origins, viewdirs, direction_norm




def apply_rotation(q1, q2):
    """
    Applies a rotation to a quaternion.
    
    Parameters:
    q1 (Tensor): The original quaternion.
    q2 (Tensor): The rotation quaternion to be applied.
    
    Returns:
    Tensor: The resulting quaternion after applying the rotation.
    """
    # Extract components for readability
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    # Compute the product of the two quaternions
    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    # Combine the components into a new quaternion tensor
    q3 = torch.tensor([w3, x3, y3, z3])

    # Normalize the resulting quaternion
    q3_normalized = q3 / torch.norm(q3)
    
    return q3_normalized


def batch_quaternion_multiply(q1, q2):
    """
    Multiply batches of quaternions.
    
    Args:
    - q1 (torch.Tensor): A tensor of shape [N, 4] representing the first batch of quaternions.
    - q2 (torch.Tensor): A tensor of shape [N, 4] representing the second batch of quaternions.
    
    Returns:
    - torch.Tensor: The resulting batch of quaternions after applying the rotation.
    """
    # Calculate the product of each quaternion in the batch
    w = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    x = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    y = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    z = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]

    # Combine into new quaternions
    q3 = torch.stack((w, x, y, z), dim=1)
    
    # Normalize the quaternions
    norm_q3 = q3 / torch.norm(q3, dim=1, keepdim=True)
    
    return norm_q3

def get_rays(H, W, K, R, T, perturb=False):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    
    if perturb:
        perturb_i = np.random.rand(H, W)
        perturb_j = np.random.rand(H, W)
        xy1 = np.stack([i + perturb_i, j + perturb_j, np.ones_like(i)], axis=2)
    else:
        xy1 = np.stack([i + 0.5, j + 0.5, np.ones_like(i)], axis=2)
    
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def get_rays_torch(H, W, K, R, T, perturb=False):
    # calculate the camera origin
    rays_o = -torch.matmul(R.T, T).squeeze()
    # calculate the world coodinates of pixels
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=K.device),
                       torch.arange(H, dtype=torch.float32, device=K.device),
                       indexing='xy')

    if perturb:
        perturb_i = torch.rand(H, W, device=K.device)
        perturb_j = torch.rand(H, W, device=K.device)
        xy1 = torch.stack([i + perturb_i, j + perturb_j, torch.ones_like(i)], dim=2)
    else:
        xy1 = torch.stack([i + 0.5, j + 0.5, torch.ones_like(i)], dim=2)
    
    pixel_camera = torch.matmul(xy1, torch.inverse(K).T)
    pixel_world = torch.matmul(pixel_camera - T.squeeze(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o.unsqueeze(0).unsqueeze(0)
    rays_d = rays_d / torch.norm(rays_d, dim=2, keepdim=True)
    rays_o = rays_o.expand_as(rays_d)
    return rays_o, rays_d