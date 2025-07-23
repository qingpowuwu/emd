import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


''' Dense 3D grid
'''
class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, **kwargs):
        super(DenseGrid, self).__init__()
        # 定义grid 每个点的通道数
        self.channels = channels
        # 定义格点在每个维度上的尺寸（高度、宽度、深度）
        self.world_size = world_size

        
        # 初始化为全1的参数化格点
        self.grid = nn.Parameter(torch.ones([1, channels, *world_size]))

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        # if self.channels == 1:
            # out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.ones([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))
    def set_aabb(self, xyz_max, xyz_min):
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size}'
