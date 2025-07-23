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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments.gaussian_options import BaseOptions
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.nn import functional as F
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : BaseOptions, gaussians : GaussianModel):
        """
        :param path: Path to colmap scene main folder.
        """
        self.args = args
        self.model_path = args.model_path
        self.duration = args.total_num_frames

        self.gaussians = gaussians

        scene_info = sceneLoadTypeCallbacks["Waymo"](
                                args.source_path, args.white_background, args.eval,
                                load_sky_mask = args.load_sky_mask, #False, 
                                load_intrinsic = args.load_intrinsic, #False,
                                load_c2w = args.load_c2w, #False,
                                load_dynamic_mask = args.load_dynamic_mask, #False,
                                load_feat_map = args.load_feat_map, #False,
                                start_time = args.start_time, #0,
                                end_time = args.end_time, # 100,
                                num_pts = args.num_pts,
                                save_occ_grid = args.save_occ_grid,
                                occ_voxel_size = args.occ_voxel_size,
                                recompute_occ_grid = args.recompute_occ_grid,
                                stride = args.stride,
                                original_start_time = args.original_start_time,
                                load_dense_depth = args.load_dense_depth,
                                is_negative_time = args.is_negative_time,
                            )

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)
        print("Loading Full Cameras")
        self.full_cameras = cameraList_from_camInfos(scene_info.full_cameras, args)

        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        self.gaussians.aabb = scene_info.cam_frustum_aabb
        self.gaussians.aabb_tensor = torch.tensor(scene_info.cam_frustum_aabb, dtype=torch.float32).cuda()
        self.gaussians.nerf_normalization = scene_info.nerf_normalization
        self.gaussians.img_width = scene_info.train_cameras[0].width
        self.gaussians.img_height = scene_info.train_cameras[0].height
        if scene_info.occ_grid is not None:
            self.gaussians.occ_grid = torch.tensor(scene_info.occ_grid, dtype=torch.bool).cuda() 
        else:
            self.gaussians.occ_grid = scene_info.occ_grid
        self.gaussians.occ_voxel_size = args.occ_voxel_size
        if hasattr(self.gaussians, '_deformation'):
            self.gaussians._deformation.deformation_net.set_aabb(scene_info.cam_frustum_aabb[1],
                                                scene_info.cam_frustum_aabb[0])
        # check occ
        #import numpy as np
        #voxel_coords = np.floor((self.gaussians._xyz.cpu().detach().numpy() - scene_info.cam_frustum_aabb[0]) / args.occ_voxel_size).astype(int)
        #occ = scene_info.occ_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
        #occ_mask = self.gaussians.get_gs_mask_in_occGrid()
        #assert all(occ == occ_mask), 'occ should be equal to occ_mask'

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            
            # if save_spilt:
            #     pc_dynamic_path = os.path.join(point_cloud_path,"point_cloud_dynamic.ply")
            #     pc_static_path = os.path.join(point_cloud_path,"point_cloud_static.ply")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)

    def save_gridgs(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}_grid".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, is_shuffle=False):
        if is_shuffle:
            shuffled_cameras = self.train_cameras.copy()
            random.shuffle(shuffled_cameras)
            return shuffled_cameras
        return self.train_cameras.copy()

    def getTestCameras(self):
        return self.test_cameras.copy()
    
    def getFullCameras(self):
        return self.full_cameras.copy()