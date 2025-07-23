import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply, contract
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
from arguments.gaussian_options import BaseOptions
from scene.encodings import HashEncoder
# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args: BaseOptions=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.is_use_hash = args.is_use_hash

        if self.is_use_hash:
            self.grid = HashEncoder(
                n_input_dims=args.hash_n_input_dims,
                n_levels=args.hash_n_levels,
                n_features_per_level=args.hash_n_features_per_level,
                base_resolution=args.hash_base_resolution,
                max_resolution=args.hash_max_resolution,
                log2_hashmap_size=args.hash_log2_hashmap_size,
                verbose=False,
            )
        else:
            self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)

        self.min_embeddings = args.min_embeddings
        self.max_embeddings = args.max_embeddings
        self.temporal_embedding_dim = args.temporal_embedding_dim
        self.gaussian_embedding_dim = args.gaussian_embedding_dim
        self.c2f_temporal_iter = args.c2f_temporal_iter
        if args.zero_temporal:
            self.weight = torch.nn.Parameter(torch.zeros(self.max_embeddings, self.temporal_embedding_dim))
        else:
            self.weight = torch.nn.Parameter(torch.normal(0., 0.01/np.sqrt(self.temporal_embedding_dim),size=(self.max_embeddings, self.temporal_embedding_dim)))

        self.args = args
        # self.args.empty_voxel=True
        if args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        if not self.args.no_time_offset:
            self.time_offset = torch.nn.Parameter(torch.zeros((3, 1)))  # hard coded the upper limit of the num cameras (adjust as necessary)
        
        self.ratio=0
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(self.args.posebase_pe)]))
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb

    def contract_points(
        self,
        positions,
    ):
        """
        contract [-inf, inf] points to the range [0, 1] for hash encoding

        Returns:
            normed_positions: [..., 3] in [0, 1]
        """
        normed_positions = contract(positions, self.aabb, ord=float("inf"))
        selector = (
            ((normed_positions > 0.0) & (normed_positions < 1.0))
            .all(dim=-1)
            .to(positions)
        )
        normed_positions = normed_positions * selector.unsqueeze(-1)
        return normed_positions

    def set_aabb(self, xyz_max, xyz_min):
        if self.is_use_hash:
            self.aabb = torch.tensor([xyz_min, xyz_max], dtype=torch.float32).cuda().view(-1, 6)
            print("Deformation Hashmap Set aabb", self.aabb)
            pass
        else:
            print("Deformation Net Set aabb",xyz_min, xyz_max)
            self.grid.set_aabb(xyz_min, xyz_max)
            if self.args.empty_voxel:
                self.empty_voxel.set_aabb(xyz_min, xyz_max)

    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            if self.is_use_hash:
                grid_out_dim = self.grid.n_output_dims + (self.grid.feat_dim)*2 
            else:
                grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            if self.is_use_hash:
                grid_out_dim = self.grid.n_output_dims
            else:
                grid_out_dim = self.grid.feat_dim

        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            # self.feature_out_f = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
            if not self.args.no_temporal_embedding_dim:
                temporal_embedding_dim = self.temporal_embedding_dim
            else:
                temporal_embedding_dim = 0
            
            if not self.args.no_gaussian_embedding_dim:
                gaussian_embedding_dim = self.gaussian_embedding_dim
            else:
                gaussian_embedding_dim = 0
            
            if not self.args.no_coarse_hexplane_features:
                hexplane_dim = mlp_out_dim + grid_out_dim
            else:
                hexplane_dim = 0
            self.feature_out = [nn.Linear(hexplane_dim + temporal_embedding_dim + gaussian_embedding_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))

        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

        if not self.args.no_fine_deform:
            if self.no_grid:
                self.feature_out_f = [nn.Linear(4,self.W)]
            else:
                # self.feature_out_f = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
                if not self.args.no_temporal_embedding_dim:
                    temporal_embedding_dim = self.temporal_embedding_dim
                else:
                    temporal_embedding_dim = 0
                
                if not self.args.no_gaussian_embedding_dim:
                    gaussian_embedding_dim = self.gaussian_embedding_dim
                else:
                    gaussian_embedding_dim = 0
                
                if not self.args.no_fine_hexplane_features:
                    hexplane_dim = mlp_out_dim + grid_out_dim
                else:
                    hexplane_dim = 0
                self.feature_out_f = [nn.Linear(hexplane_dim + temporal_embedding_dim + gaussian_embedding_dim, self.W)]
            
            for i in range(self.D-1):
                self.feature_out_f.append(nn.ReLU())
                self.feature_out_f.append(nn.Linear(self.W,self.W))
            self.feature_out_f = nn.Sequential(*self.feature_out_f)
            self.pos_deform_f = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
            self.scales_deform_f = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
            self.rotations_deform_f = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
            self.opacity_deform_f = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
            self.shs_deform_f = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

        if self.args.feat_head:
            semantic_feature_dim = 64
            feature_mlp_layer_width = 64
            feature_embedding_dim = 3
            self.dino_head = nn.Sequential(
                nn.Linear(semantic_feature_dim, feature_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(feature_mlp_layer_width, feature_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(feature_mlp_layer_width, feature_embedding_dim),
            )

    def query_hexplane(self, rays_pts_emb, time_emb):

        # 这里是 hexplane的forward 得到 feature [N, 128]
        if self.is_use_hash:
            normalized_positions = self.contract_points(rays_pts_emb[:,:3])
            grid_feature = self.grid(torch.cat([normalized_positions, time_emb[:,:1]],-1))
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])

        if self.grid_pe > 1:
            grid_feature = poc_fre(grid_feature, self.grid_pe)
        hidden = torch.cat([grid_feature],-1) 
        return hidden

    @property
    def get_empty_ratio(self):
        return self.ratio

    def int_lininterp(self, t, init_val, final_val, until):
        return int(init_val + (final_val - init_val) * min(max(t, 0), until) / until)

    def get_temporal_embed(self, t, current_num_embeddings, align_corners=True):
        emb_resized = F.interpolate(self.weight[None,None,...], 
                                 size=(current_num_embeddings, self.temporal_embedding_dim), 
                                 mode='bilinear', align_corners=True)
        N, _ = t.shape
        t = t[0,0]

        fdim = self.temporal_embedding_dim
        grid = torch.cat([torch.arange(fdim).cuda().unsqueeze(-1)/(fdim-1), torch.ones(fdim,1).cuda() * t, ], dim=-1)[None,None,...]
        grid = (grid - 0.5) * 2

        emb = F.grid_sample(emb_resized, grid, align_corners=align_corners, mode='bilinear', padding_mode='reflection')
        emb = emb.repeat(1,1,N,1).squeeze()
        return emb

    def query_time(self, hidden_hexplane, time_emb, embeddings=None, \
                   iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30):
        feature_list = []

        if use_coarse_temporal_embedding and not self.args.no_coarse_hexplane_features:
            feature_list.append(hidden_hexplane)
        
        if not use_coarse_temporal_embedding and not self.args.no_fine_hexplane_features:
            feature_list.append(hidden_hexplane)

        if not self.args.no_temporal_embedding_dim:
            t = time_emb[:,:1]
            if use_coarse_temporal_embedding:
                h = self.get_temporal_embed(t, num_down_emb)
            else:
                if self.args.no_c2f_temporal_embedding:
                    h = self.get_temporal_embed(t, self.max_embeddings)
                else:
                    if iter is None:
                        iter = self.c2f_temporal_iter
                    h = self.get_temporal_embed(t, self.int_lininterp(iter, num_down_emb, self.max_embeddings, self.c2f_temporal_iter))
            feature_list.append(h)
        
        if not self.args.no_gaussian_embedding_dim:
            if embeddings is not None:
                feature_list.append(embeddings)
        
        h = torch.cat(feature_list, dim=-1)
        h = feature_out(h)   # [N,64]
        return h

    def get_feature(self, rays_pts_emb, time_emb, embeddings, \
                    iter, feature_out, use_coarse_temporal_embedding, num_down_emb, time_diff=1.0, is_train=False):
        hidden_hexplane = self.query_hexplane(rays_pts_emb, time_emb)
        hidden = self.query_time(hidden_hexplane, time_emb, embeddings, \
                                iter=iter, feature_out=feature_out, use_coarse_temporal_embedding=use_coarse_temporal_embedding, num_down_emb=num_down_emb)
        
        if self.args.aggregate_feature:
            if is_train:
                noise = torch.rand_like(rays_pts_emb)[..., 0:1]
            else:
                noise = torch.ones_like(rays_pts_emb)[..., 0:1]
            if self.args.aggregate_time_warp:
                forward_warped_time_emb = torch.clamp(
                    time_emb + time_diff * noise, 0, 1.0
                )
                backward_warped_time_emb = torch.clamp(
                    time_emb - time_diff * noise, 0, 1.0
                )
            else:
                forward_warped_time_emb = time_emb
                backward_warped_time_emb = time_emb
            
            if self.args.aggregate_space_warp:
                if use_coarse_temporal_embedding:
                    dx = self.forward_dynamic_xyz(hidden, self.pos_deform)
                else:
                    dx = self.forward_dynamic_xyz(hidden, self.pos_deform_f)
                forward_warped_rays_pts_emb = rays_pts_emb[:,:3] + dx
                backward_warped_rays_pts_emb = rays_pts_emb[:,:3] - dx
                forward_warped_rays_pts_emb = poc_fre(forward_warped_rays_pts_emb, self.pos_poc)
                backward_warped_rays_pts_emb = poc_fre(backward_warped_rays_pts_emb, self.pos_poc)
            else:
                forward_warped_rays_pts_emb = rays_pts_emb
                backward_warped_rays_pts_emb = rays_pts_emb

            hidden_hexplane_forward = self.query_hexplane(forward_warped_rays_pts_emb, forward_warped_time_emb)
            hidden_hexplane_backward = self.query_hexplane(backward_warped_rays_pts_emb, backward_warped_time_emb)
            hidden_forward = self.query_time(hidden_hexplane_forward, forward_warped_time_emb, embeddings, \
                                            iter=iter, feature_out=feature_out, use_coarse_temporal_embedding=use_coarse_temporal_embedding, num_down_emb=num_down_emb)
            hidden_backward = self.query_time(hidden_hexplane_backward, backward_warped_time_emb, embeddings, \
                                            iter=iter, feature_out=feature_out, use_coarse_temporal_embedding=use_coarse_temporal_embedding, num_down_emb=num_down_emb)
            hidden = 0.5*hidden + 0.25*hidden_forward + 0.25*hidden_backward
        return hidden

    def forward(self, rays_pts_emb, \
                time_emb=None, embeddings=None, is_coarse=True, iter=None, \
                num_down_emb_c=30, num_down_emb_f=30, apply_deform=True, time_diff=1.0, is_train=False):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            if apply_deform:
                if is_coarse:
                    hidden = self.get_feature(rays_pts_emb, time_emb, embeddings, \
                                            iter, self.feature_out, use_coarse_temporal_embedding=True, num_down_emb=num_down_emb_c, time_diff=time_diff, is_train=is_train)
                    ddict = self.forward_dynamic(hidden, \
                                                    self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform, self.shs_deform)
                else:
                    hidden = self.get_feature(rays_pts_emb, time_emb, embeddings, \
                                            iter, self.feature_out_f, use_coarse_temporal_embedding=False, num_down_emb=num_down_emb_f, time_diff=time_diff, is_train=is_train)
                    ddict = self.forward_dynamic(hidden, \
                                                    self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.shs_deform_f)
            else:
                # pts = rays_pts_emb[:,:3]
                # scales = scales_emb[:,:3]
                # rotations = rotations_emb[:,:4]
                # opacity = opacity[:,:1]
                # shs = shs_emb
                ddict = None
            # return pts, scales, rotations, opacity, shs, ddict
            return ddict

    def forward_time_offset(self, time_emb, cam_no):
        if self.args.no_time_offset:
            return time_emb
        return time_emb + self.time_offset[cam_no]

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx

    def forward_dynamic_xyz(self, hidden, pos_deform):
        dx = pos_deform(hidden)
        return dx

    def forward_dynamic(self, hidden, 
                        pos_deform, scales_deform, rotations_deform, opacity_deform, shs_deform):

        dx = None
        if not self.args.no_dx:
            dx = pos_deform(hidden) # [N, 3]

        ds = None
        if not self.args.no_ds :
            ds = scales_deform(hidden)
        
        dr = None
        if not self.args.no_dr :
            dr = rotations_deform(hidden)
            # rotations = torch.zeros_like(rotations_emb[:,:4])
            # if self.args.apply_rotation:
            #     rotations = batch_quaternion_multiply(rotations_emb, dr)
            # else:
            #     rotations = rotations_emb[:,:4] + dr

        do = None
        if not self.args.no_do :
            do = opacity_deform(hidden) 
            # opacity = torch.zeros_like(opacity_emb[:,:1])
            # opacity = opacity_emb[:,:1]*mask + do

        dshs = None
        if not self.args.no_dshs:
        #     shs = shs_emb
        #     dshs = None
        # else:
            dshs = shs_deform(hidden).reshape([hidden.shape[0],16,3])
            # shs = torch.zeros_like(shs_emb)
            # shs = shs_emb*mask.unsqueeze(-1) + dshs

        feat = None
        if self.args.feat_head:
            feat = self.dino_head(hidden)
        
        ddict = {
            "dx": dx,
            "ds": ds,
            "dr": dr,
            "do": do,
            "dshs": dshs,
            "feat": feat
        }
        return ddict

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
class deform_network(nn.Module):
    def __init__(self, args: BaseOptions) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_output = args.timenet_output
        self.temporal_embedding_dim = args.temporal_embedding_dim
        self.gaussian_embedding_dim = args.gaussian_embedding_dim
        self.c2f_temporal_iter = args.c2f_temporal_iter
        self.min_embeddings = args.min_embeddings
        self.no_coarse_deform = args.no_coarse_deform
        self.no_fine_deform = args.no_fine_deform
        grid_pe = args.grid_pe
        self.args = args
        
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb

    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    # def forward_static(self, points):
    #     points = self.deformation_net(points)
    #     return points

    def apply_deform(self, point, scales=None, rotations=None, opacity=None, shs=None, ddict_c=None, ddict_f=None):
        point_final = point.clone() if self.args.direct_add_dx or self.args.no_dx else torch.zeros_like(point)
        scales_final = scales.clone() if self.args.direct_add_ds or self.args.no_ds else torch.zeros_like(scales)
        rotations_final = rotations.clone() if self.args.direct_add_dr or self.args.no_dr else torch.zeros_like(rotations)
        opacity_final = opacity.clone() if self.args.direct_add_do or self.args.no_do else torch.zeros_like(opacity)
        shs_final = shs.clone() if self.args.direct_add_dshs or self.args.no_dshs else torch.zeros_like(shs)

        if not self.args.no_dx and self.args.apply_final_dx:
            if not self.args.no_coarse_deform:
                point_final = point_final + ddict_c["dx"]
            
            if not self.args.no_fine_deform:
                point_final = point_final + ddict_f["dx"]

        if not self.args.no_ds :
            if not self.args.no_coarse_deform:
                scales_final = scales_final + ddict_c["ds"]
            
            if not self.args.no_fine_deform:
                scales_final = scales_final + ddict_f["ds"]
        
        if not self.args.no_dr :
            if not self.args.no_coarse_deform:
                rotations_final = batch_quaternion_multiply(rotations_final, ddict_c["dr"])
            
            if not self.args.no_fine_deform:
                rotations_final = batch_quaternion_multiply(rotations_final, ddict_f["dr"])

        if not self.args.no_do :
            if not self.args.no_coarse_deform:
                opacity_final = opacity_final + ddict_c["do"]
            
            if not self.args.no_fine_deform:
                opacity_final = opacity_final + ddict_f["do"]

        if not self.args.no_dshs:
            if not self.args.no_coarse_deform:
                shs_final = shs_final + ddict_c["dshs"]
            
            if not self.args.no_fine_deform:
                shs_final = shs_final + ddict_f["dshs"]

        return point_final, scales_final, rotations_final, opacity_final, shs_final


    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, \
                embeddings=None, iter=None, cam_no=None, time_diff=None, is_train=None):
        point_emb = poc_fre(point, self.pos_poc)
        times_sel = self.deformation_net.forward_time_offset(times_sel, cam_no)

        ddict_c = self.deformation_net(
                                        point_emb,
                                        # scales_emb,
                                        # rotations_emb,
                                        # opacity,
                                        # shs,
                                        times_sel,
                                        embeddings,
                                        is_coarse=True,
                                        iter=iter,
                                        num_down_emb_c=self.min_embeddings,
                                        apply_deform=not self.no_coarse_deform,
                                        time_diff=time_diff,
                                        is_train=is_train)
        if not self.no_coarse_deform:
            if self.args.apply_coarse_dx:
                point_emb = poc_fre(point + ddict_c["dx"], self.pos_poc)

        ddict_f = self.deformation_net(
                                        point_emb,
                                        # scales_emb,
                                        # rotations_emb,
                                        # opacity,
                                        # shs,
                                        times_sel,
                                        embeddings,
                                        is_coarse=False,
                                        iter=iter,
                                        num_down_emb_f=self.min_embeddings,
                                        apply_deform=not self.no_fine_deform,
                                        time_diff=time_diff,
                                        is_train=is_train)

        ddict = {
            "coarse": ddict_c,
            "fine": ddict_f
        }
        point_final, scales_final, rotations_final, opacity_final, shs_final = self.apply_deform(point, scales, rotations, opacity, shs, ddict_c, ddict_f)
        return point_final, scales_final, rotations_final, opacity_final, shs_final, ddict
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters()
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb