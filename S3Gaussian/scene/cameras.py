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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 # for waymo
                 sky_mask = None, depth_map = None, semantic_mask = None, instance_mask = None,
                 num_panoptic_objects = 0,
                 sam_mask = None, dynamic_mask = None, feat_map = None, objects = None, intrinsic = None, c2w = None, 
                 time = None, cam_no = None, time_diff = None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.cam_no = cam_no
        self.time_diff = time_diff
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        # w2c： waymo_world --> opencv_cam 
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # proj : opencv_cam to 0-1-NDC
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # w2c + c2pixel : X_world * full_proj_transform = pixel
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        # for waymo
        self.sky_mask = sky_mask
        self.depth_map = depth_map
        self.semantic_mask = semantic_mask
        self.instance_mask = instance_mask
        self.num_panoptic_objects = num_panoptic_objects
        self.sam_mask = sam_mask
        self.dynamic_mask = dynamic_mask
        self.feat_map = feat_map
        # grouping
        self.intrinsic = intrinsic.float().cuda()
        self.c2w = c2w

