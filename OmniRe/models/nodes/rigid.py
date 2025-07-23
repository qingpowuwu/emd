from typing import Dict, List, Tuple
import logging
import random

import torch
import torch.nn as nn
from torch.nn import Parameter
import open3d as o3d
import numpy as np
from models.modules import ConditionalDeformNetwork
from models.gaussians.basics import *
from models.gaussians.vanilla import VanillaGaussians

logger = logging.getLogger()

class RigidNodes(VanillaGaussians):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gaussian_embedding_dim = self.ctrl_cfg.gaussian_embedding_dim
        self.temporal_embedding_dim = self.ctrl_cfg.temporal_embedding_dim
        self.no_gaussian_embedding_dim = self.ctrl_cfg.no_gaussian_embedding_dim
        self.no_temporal_embedding_dim = self.ctrl_cfg.no_temporal_embedding_dim
        self.no_coarse_deform = self.ctrl_cfg.no_coarse_deform
        self.no_fine_deform = self.ctrl_cfg.no_fine_deform
        self.no_c2f_temporal_embedding = self.ctrl_cfg.no_c2f_temporal_embedding
        self.min_embeddings = self.ctrl_cfg.min_embeddings
        self.max_embeddings = self.ctrl_cfg.max_embeddings
        self.c2f_temporal_iter = self.ctrl_cfg.c2f_temporal_iter
        self.no_apply_embed_shs = self.ctrl_cfg.no_apply_embed_shs
        self.no_apply_embed_track = self.ctrl_cfg.no_apply_embed_track
    
    @property
    def num_instances(self):
        return self.instances_fv.shape[1]
    @property
    def num_frames(self):
        return self.instances_fv.shape[0]
    
    def get_pts_valid_mask(self):
        """
        get the mask for valid points
        """
        return self.instances_fv[self.cur_frame][self.point_ids[..., 0]]
    
    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id
    def register_normalized_timestamps(self, normalized_timestamps: int):
        self.normalized_timestamps = normalized_timestamps
        
    def create_from_pcd(self, instance_pts_dict: Dict[str, torch.Tensor]) -> None:
        """
        instance_pts_dict: {
            id in dataset: {
                "class_name": str,
                "pts": torch.Tensor, (N, 3)
                "colors": torch.Tensor, (N, 3)
                "poses": torch.Tensor, (num_frame, 4, 4)
                "size": torch.Tensor, (3, )
                "frame_info": torch.Tensor, (num_frame)
                "num_pts": int,
            },
        }
        """
        # collect all instances
        init_means = []
        init_colors = []
        init_embeddings = []
        instances_pose = []
        instances_size = []
        instances_fv = []
        point_ids = []

        self.prev_num_pts = []
        self.neighbor_weight = []
        self.neighbor_indices = []
        self.weight = []
        for id_in_model, (id_in_dataset, v) in enumerate(instance_pts_dict.items()):
            init_means.append(v["pts"])
            init_colors.append(v["colors"])
            init_embeddings.append(torch.zeros(v["num_pts"], self.gaussian_embedding_dim))
            instances_pose.append(v["poses"].unsqueeze(1))
            instances_size.append(v["size"])
            instances_fv.append(v["frame_info"].unsqueeze(1))
            point_ids.append(torch.full((v["num_pts"], 1), id_in_model, dtype=torch.long))
            self.prev_num_pts.append(0)
            self.neighbor_weight.append(None)
            self.neighbor_indices.append(None)
            self.weight.append(torch.normal(0., 0.01/np.sqrt(self.temporal_embedding_dim),size=(self.max_embeddings, self.temporal_embedding_dim)))
        init_means = torch.cat(init_means, dim=0).to(self.device) # (N, 3)
        init_colors = torch.cat(init_colors, dim=0).to(self.device) # (N, 3)
        init_embeddings = torch.cat(init_embeddings, dim=0).to(self.device) # (N, 4)
        self.weight = torch.stack(self.weight, dim=0).to(self.device) # (N, 4)
        instances_pose = torch.cat(instances_pose, dim=1).to(self.device) # (num_frame, num_instances, 4, 4)
        self.instances_size = torch.stack(instances_size).to(self.device) # (num_instances, 3)
        self.instances_fv = torch.cat(instances_fv, dim=1).to(self.device) # (num_frame, num_instances)
        self.point_ids = torch.cat(point_ids, dim=0).to(self.device)
        instances_quats = self.get_instances_quats(instances_pose)
        instances_trans = instances_pose[..., :3, 3]
        
        # initialize the means, scales, quats, and colors
        self._means = Parameter(init_means)
        self._embeddings = Parameter(init_embeddings)
        self.weight = Parameter(self.weight)

        if not self.no_apply_embed_track:
            self.track_rot_c = nn.Linear(self.gaussian_embedding_dim + self.temporal_embedding_dim, 1).cuda()
            self.track_rot_f = nn.Linear(self.gaussian_embedding_dim + self.temporal_embedding_dim, 1).cuda()
            self.track_trans_c = nn.Linear(self.gaussian_embedding_dim + self.temporal_embedding_dim, 3).cuda()
            self.track_trans_f = nn.Linear(self.gaussian_embedding_dim + self.temporal_embedding_dim, 3).cuda()
            # init weights as 0
            self.track_rot_c.weight.data.fill_(0)
            self.track_rot_c.bias.data.fill_(0)
            self.track_rot_f.weight.data.fill_(0)
            self.track_rot_f.bias.data.fill_(0)

            self.track_trans_f.weight.data.fill_(0)
            self.track_trans_f.bias.data.fill_(0)
            self.track_trans_c.weight.data.fill_(0)
            self.track_trans_c.bias.data.fill_(0)

        distances, _ = k_nearest_sklearn(self._means.data, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        avg_dist = avg_dist.clamp(0.002, 100)
        self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))
        dim_sh = num_sh_bases(self.sh_degree)
        
        # pose refinement
        self.instances_quats = Parameter(self.quat_act(instances_quats)) # (num_frame, num_instances, 4)
        self.instances_trans = Parameter(instances_trans)              # (num_frame, num_instances, 3)

        fused_color = RGB2SH(init_colors) # float range [0, 1] 
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        self._opacities = Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1, device=self.device)))

    def int_lininterp(self, t, init_val, final_val, until):
        return int(init_val + (final_val - init_val) * min(max(t, 0), until) / until)

    def get_temporal_embed(self, t, current_num_embeddings, align_corners=True, weight=None):

        emb_resized = F.interpolate(weight[None,None,...], 
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

    def query_time(self, time_emb, iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30, mean=False, embeddings=None, weight=None):
        feature_list = []

        if not self.no_temporal_embedding_dim:
            t = time_emb[:,:1]
            if use_coarse_temporal_embedding:
                h = self.get_temporal_embed(t, num_down_emb, weight=weight)
            else:
                if self.no_c2f_temporal_embedding:
                    h = self.get_temporal_embed(t, self.max_embeddings, weight=weight)
                else:
                    if iter is None:
                        iter = self.c2f_temporal_iter
                    h = self.get_temporal_embed(t, self.int_lininterp(iter, num_down_emb, self.max_embeddings, self.c2f_temporal_iter), weight=weight)
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            feature_list.append(h)
        
        if not self.no_gaussian_embedding_dim:
            if mean:
                embeddings = torch.mean(embeddings, dim=0, keepdim=True)

            feature_list.append(embeddings)
        
        h = torch.cat(feature_list, dim=-1)
        h = feature_out(h)   # [N,64]
        return h
    
    def query_coarse_to_fine(self, time_emb, iter=None, feature_out_c=None, feature_out_f=None, mean=False, embeddings=None, weight=None):
        h_c = None
        if not self.no_coarse_deform:
            h_c = self.query_time(time_emb, iter=iter, feature_out=feature_out_c, use_coarse_temporal_embedding=True, mean=mean, embeddings=embeddings, weight=weight)
        h_f = None
        if not self.no_fine_deform:
            h_f = self.query_time(time_emb, iter=iter, feature_out=feature_out_f, use_coarse_temporal_embedding=False, mean=mean, embeddings=embeddings, weight=weight)
        return h_c, h_f

    def embedding_track_rot_offset(self, frame=0, start_frame=0, end_frame=1, embeddings=None, weight=None):
        normalized_frame = (frame - start_frame) / (end_frame - start_frame)
        time_emb = torch.tensor([[normalized_frame]]).float().cuda()

        def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """
            Multiply two quaternions.
            Usual torch rules for broadcasting apply.

            Args:
                a: Quaternions as tensor of shape (..., 4), real part first.
                b: Quaternions as tensor of shape (..., 4), real part first.

            Returns:
                The product of a and b, a tensor of quaternions shape (..., 4).
            """
            aw, ax, ay, az = torch.unbind(a, -1)
            bw, bx, by, bz = torch.unbind(b, -1)
            ow = aw * bw - ax * bx - ay * by - az * bz
            ox = aw * bx + ax * bw + ay * bz - az * by
            oy = aw * by - ax * bz + ay * bw + az * bx
            oz = aw * bz + ax * by - ay * bx + az * bw
            return torch.stack((ow, ox, oy, oz), -1)

        track_rot_c, track_rot_f = self.query_coarse_to_fine(time_emb, iter=self.step, feature_out_c=self.track_rot_c, feature_out_f=self.track_rot_f, mean=True, embeddings=embeddings, weight=weight)
        opt_rot_c = torch.zeros((4), device="cuda")
        opt_rot_f = torch.zeros((4), device="cuda")

        opt_rot_c[0] = torch.cos(track_rot_c)
        opt_rot_c[3] = torch.sin(track_rot_c)
        opt_rot_f[0] = torch.cos(track_rot_f)
        opt_rot_f[3] = torch.sin(track_rot_f)

        track_rot = quaternion_raw_multiply(opt_rot_c.unsqueeze(0), opt_rot_f.unsqueeze(0)).squeeze(0)

        return track_rot

    def embedding_track_trans_offset(self, frame=0, start_frame=0, end_frame=1, embeddings=None, weight=None):
        normalized_frame = (frame - start_frame) / (end_frame - start_frame)
        time_emb = torch.tensor([[normalized_frame]]).float().cuda()
        track_trans_c, track_trans_f = self.query_coarse_to_fine(time_emb, iter=self.step, feature_out_c=self.track_trans_c, feature_out_f=self.track_trans_f, mean=True, embeddings=embeddings, weight=weight)
        track_trans = track_trans_c + track_trans_f
        track_trans = track_trans.squeeze(0)
        return track_trans

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix+"ins_rotation"] = [self.instances_quats]
        param_groups[self.class_prefix+"ins_translation"] = [self.instances_trans]
        param_groups[self.class_prefix+"embeddings"] = [self._embeddings]
        param_groups[self.class_prefix+"weight"] = [self.weight]
        if not self.no_apply_embed_track:
            param_groups[self.class_prefix+"track_rot_c"] = self.track_rot_c.parameters()
            param_groups[self.class_prefix+"track_rot_f"] = self.track_rot_f.parameters()
            param_groups[self.class_prefix+"track_trans_c"] = self.track_trans_c.parameters()
            param_groups[self.class_prefix+"track_trans_f"] = self.track_trans_f.parameters()
        return param_groups
    
    def get_instances_quats(self, instances_pose: torch.Tensor) -> torch.Tensor:
        """
        Convert the pose to quaternion for all frames and instances
        """
        num_frames = instances_pose.shape[0]
        num_instances = instances_pose.shape[1]
        quats = torch.zeros(num_frames*num_instances, 4, device=self.device)
        
        poses = instances_pose[..., :3, :3].view(-1, 3, 3)
        valid_mask = self.instances_fv.view(-1)
        _quats = matrix_to_quaternion(poses[valid_mask])
        _quats = self.quat_act(_quats)
        
        quats[valid_mask] = _quats
        quats[~valid_mask, 0] = 1.0
        return quats.reshape(num_frames, num_instances, 4)

    def refinement_after(self, step: int, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = (
                self.step < self.ctrl_cfg.stop_split_at
                and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
            )
            # split & duplicate
            print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
            if do_densification:
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                
                splits = (
                    self.get_scaling.max(dim=-1).values > \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.ctrl_cfg.n_split_samples
                (
                    split_means,
                    split_feature_dc,
                    split_feature_rest,
                    split_opacities,
                    split_scales,
                    split_quats,
                    split_ids,
                    split_embeddings,
                ) = self.split_gaussians(splits, nsamps)

                dups = (
                    self.get_scaling.max(dim=-1).values <= \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                dups &= high_grads
                (
                    dup_means,
                    dup_feature_dc,
                    dup_feature_rest,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                    dup_ids,
                    dup_embeddings,
                ) = self.dup_gaussians(dups)
                
                self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                # self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))
                self._features_dc = Parameter(torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                self._features_rest = Parameter(torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                self._opacities = Parameter(torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))
                self.point_ids = torch.cat([self.point_ids, split_ids, dup_ids], dim=0)
                self._embeddings = Parameter(torch.cat([self._embeddings.detach(), split_embeddings, dup_embeddings], dim=0))
                
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                    dim=0,
                )
                
                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                param_groups[self.class_prefix+"embeddings"] = [self._embeddings]
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                param_groups[self.class_prefix+"embeddings"] = [self._embeddings]
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)

            # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
            if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_gaussian_param_groups()
                param_groups[self.class_prefix+"embeddings"] = [self._embeddings]
                remove_from_optim(optimizer, deleted_mask, param_groups)
            print(f"Class {self.class_prefix} left points: {self.num_points}")

            # reset opacity
            if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                    # we align to original repo of gaussians spalting
                reset_value = torch.min(self.get_opacity.data,
                                        torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                self._opacities.data = torch.logit(reset_value)
                # reset the exp of optimizer
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix+"opacity":
                        old_params = group["params"][0]
                        param_state = optimizer.state[old_params]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.ctrl_cfg.cull_out_of_bound:
            culls = culls | self.get_out_of_bound_mask()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > 
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self._embeddings = Parameter(self._embeddings[~culls].detach())
        self.point_ids = self.point_ids[~culls]

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int = 2) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        # step 6, sample new ids
        new_ids = self.point_ids[split_mask].repeat(samps, 1)
        new_embeddings = self._embeddings[split_mask].repeat(samps, 1)
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats, new_ids, new_embeddings

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        dup_ids = self.point_ids[dup_mask]
        dup_embeddings = self._embeddings[dup_mask]
        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats, dup_ids, dup_embeddings

    def get_out_of_bound_mask(self):
        """
        This function checks if the gaussians are out of instance boxes
        """
        # get the instance boxes
        per_pts_size = self.instances_size[self.point_ids[..., 0]]
        instance_pts = self._means
        
        mask = (instance_pts.abs() > per_pts_size / 2).any(dim=-1)
        return mask

    def transform_means(self, means: torch.Tensor) -> torch.Tensor:
        """
        transform the means of instances to world space
        according to the pose at the current frame
        """
        assert means.shape[0] == self.point_ids.shape[0], \
            "its a bug here, we need to pass the mask for points_ids"
        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            # use the previous and next frame to interpolate the pose
            _quats_prev_frame = self.instances_quats[self.cur_frame - 1]
            _quats_next_frame = self.instances_quats[self.cur_frame + 1]
            _quats_cur_frame = self.instances_quats[self.cur_frame]
            interpolated_quats = interpolate_quats(_quats_prev_frame, _quats_next_frame)
            
            inter_valid_mask = self.instances_fv[self.cur_frame - 1] & self.instances_fv[self.cur_frame + 1]
            quats_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_quats, _quats_cur_frame
            )
        else:
            quats_cur_frame = self.instances_quats[self.cur_frame] # (num_instances, 4)
        rot_cur_frame = quat_to_rotmat(
            self.quat_act(quats_cur_frame)
        )                                                          # (num_instances, 3, 3)
        rot_per_pts = rot_cur_frame[self.point_ids[..., 0]]        # (num_points, 3, 3)
        
        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_ins_trans = self.instances_trans[self.cur_frame - 1]
            _next_ins_trans = self.instances_trans[self.cur_frame + 1]
            _cur_ins_trans = self.instances_trans[self.cur_frame]
            interpolated_trans = (_prev_ins_trans + _next_ins_trans) * 0.5
            
            inter_valid_mask = self.instances_fv[self.cur_frame - 1] & self.instances_fv[self.cur_frame + 1]
            trans_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_trans, _cur_ins_trans
            )
        else:
            trans_cur_frame = self.instances_trans[self.cur_frame] # (num_instances, 3)
        trans_cur_frame = trans_cur_frame.clone()
        for ins_id in range(self.num_instances):
            track_trans_offset = self.embedding_track_trans_offset(
                frame=self.cur_frame,
                start_frame=0,
                end_frame=self.num_frames - 1,
                embeddings=self._embeddings[(self.point_ids == ins_id).squeeze(1), :],
                weight=self.weight[ins_id]
            )
            if track_trans_offset.isnan().any():
                continue
            trans_cur_frame[ins_id] = trans_cur_frame[ins_id] + track_trans_offset

        trans_per_pts = trans_cur_frame[self.point_ids[..., 0]]
        
        # transform the means to world space
        means = torch.bmm(
            rot_per_pts, means.unsqueeze(-1)
        ).squeeze(-1) + trans_per_pts
        return means

    def transform_quats(self, quats: torch.Tensor) -> torch.Tensor:
        """
        transform the quats of instances to world space
        according to the pose at the current frame
        """
        assert quats.shape[0] == self.point_ids.shape[0], \
            "its a bug here, we need to pass the mask for points_ids"
        global_quats_cur_frame = self.instances_quats[self.cur_frame]
        global_quats_cur_frame = global_quats_cur_frame.clone()
        # 针对每个实例单独应用 embedding_track_rot_offset 方法
        for ins_id in range(self.num_instances):
            rot_offset = self.embedding_track_rot_offset(
                frame=self.cur_frame,
                start_frame=0,
                end_frame=self.num_frames - 1,
                embeddings=self._embeddings[(self.point_ids == ins_id).squeeze(1), :],
                weight=self.weight[ins_id]
            )
            # 将旋转修正应用到 global_quats_cur_frame
            if rot_offset.isnan().any():
                continue
            global_quats_cur_frame_ = global_quats_cur_frame[ins_id].clone()
            global_quats_cur_frame[ins_id] = quat_mult(global_quats_cur_frame_, rot_offset)

        global_quats_per_pts = global_quats_cur_frame[self.point_ids[..., 0]]
            
        global_quats_per_pts = self.quat_act(global_quats_per_pts)
        _quats = self.quat_act(quats)
        return quat_mult(global_quats_per_pts, _quats)

    def get_gaussians(self, cam: dataclass_camera) -> Dict[str, torch.Tensor]:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        # NOTE: hack here, need to consider a gaussian filter for efficient rendering
        
        world_means = self.transform_means(self._means)
        world_quats = self.transform_quats(self._quats)
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
        
        valid_mask = self.get_pts_valid_mask()
            
        activated_opacities = self.get_opacity * valid_mask.float().unsqueeze(-1)
        activated_scales = self.get_scaling
        activated_rotations = self.quat_act(world_quats)
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
        
        self._gs_cache = {
            "_scales": activated_scales[filter_mask],
        }
        return gs_dict

    def get_instance_activated_gs_dict(self, ins_id: int) -> Dict[str, torch.Tensor]:
        pts_mask = self.point_ids[..., 0] == ins_id
        if pts_mask.sum() < 100:
            return None
        local_means = self._means[pts_mask]
        activated_opacities = torch.sigmoid(self._opacities[pts_mask])
        activated_scales = torch.exp(self._scales[pts_mask])
        activated_local_rotations = self.quat_act(self._quats[pts_mask])
        gaussian_dict = {
            "means": local_means,
            "opacities": activated_opacities,            
            "scales": activated_scales,
            "quats": activated_local_rotations,
            "sh_dcs": self._features_dc[pts_mask],
            "sh_rests": self._features_rest[pts_mask],
            "ids": self.point_ids[pts_mask],
        }
        return gaussian_dict
    
    def compute_reg_loss(self) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_reg_loss()
        scaling_reg = self.reg_cfg.get("scaling_reg", None)
        if scaling_reg is not None:
            w = scaling_reg.w
            precentile = scaling_reg.precentile
            stop_after = scaling_reg.stop_after
            start_after = scaling_reg.start_after
            
            if self.step < stop_after and self.step > start_after and w > 0:
                scale_prod = self._gs_cache["_scales"].prod(dim=-1)
                p = torch.kthvalue(scale_prod, int(scale_prod.shape[0] * precentile)).values
                # penalize the scales that are too large
                loss_dict["scaling_percentile_reg"] = torch.relu(scale_prod - p).mean() * w

        # temporal smooth regularization
        temporal_smooth_reg = self.reg_cfg.get("temporal_smooth_reg", None)
        if temporal_smooth_reg is not None:
            instance_mask = self.instances_fv[self.cur_frame]
            if instance_mask.sum() > 0:
                trans_cfg = temporal_smooth_reg.get("trans", None)
                if trans_cfg is not None:
                    fi_interval = random.randint(1, trans_cfg.smooth_range)
                    if self.cur_frame >= fi_interval and self.cur_frame < self.num_frames - fi_interval:
                        valid_mask = (
                            self.instances_fv[self.cur_frame - fi_interval] & \
                            self.instances_fv[self.cur_frame + fi_interval] & \
                            self.instances_fv[self.cur_frame]
                        )
                        if valid_mask.sum() > 0:
                            cur_trans = self.instances_trans[self.cur_frame]
                            pre_trans = self.instances_trans[self.cur_frame - fi_interval].data
                            next_trans = self.instances_trans[self.cur_frame + fi_interval].data
                            loss = (next_trans[valid_mask] + pre_trans[valid_mask] - 2 * cur_trans[valid_mask]).abs().mean()
                            loss_dict["trans_temporal_smooth"] = loss * trans_cfg.w
        def weighted_l2_loss_v2(x, y, w):
            return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()

        def o3d_knn(pts, num_knn):
            indices = []
            sq_dists = []
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            for p in pcd.points:
                [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
                indices.append(i[1:])
                sq_dists.append(d[1:])
            return np.array(sq_dists), np.array(indices)

        embedding_reg = self.reg_cfg.get("embedding_reg", None)
        try:
            if embedding_reg is not None:
                reg_loss = 0
                for i in range(self.num_instances):
                    if self.prev_num_pts[i] != self._means[(self.point_ids == i).squeeze(1)].shape[0] or self.neighbor_weight[i] is None or self.neighbor_indices[i] is None:
                        neighbor_sq_dist, neighbor_indices = o3d_knn(self._means[(self.point_ids == i).squeeze(1)].detach().cpu().numpy(), 20)
                        neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
                        neighbor_indices = torch.tensor(neighbor_indices).cuda().long().contiguous()
                        self.neighbor_weight[i] = torch.tensor(neighbor_weight).cuda().float().contiguous()
                        self.neighbor_indices[i] = neighbor_indices
                        self.prev_num_pts[i] = self._means[(self.point_ids == i).squeeze(1)].shape[0]

                    emb = self._embeddings[(self.point_ids == i).squeeze(1), :][:,None,:].repeat(1,20,1)
                    emb_knn = self._embeddings[(self.point_ids == i).squeeze(1)][self.neighbor_indices[i]]
                    reg_loss = 1.0 * weighted_l2_loss_v2(emb, emb_knn, self.neighbor_weight[i])
                loss_dict["embedding_reg"] = reg_loss
        except:
            pass

        return loss_dict

    def state_dict(self) -> Dict:
        state_dict = super().state_dict()
        state_dict.update({
            "points_ids": self.point_ids,
            "instances_size": self.instances_size,
            "instances_fv": self.instances_fv,
            "embeddings": self._embeddings,
            "_weight": self.weight,
            "track_rot_c": self.track_rot_c.state_dict(),
            "track_rot_f": self.track_rot_f.state_dict(),
            "track_trans_c": self.track_trans_c.state_dict(),
            "track_trans_f": self.track_trans_f.state_dict(),
        })
        return state_dict
    
    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        self.point_ids = state_dict.pop("points_ids")
        self.instances_size = state_dict.pop("instances_size")
        self.instances_fv = state_dict.pop("instances_fv")
        self._embeddings = Parameter(state_dict.pop("embeddings"))
        self.weight = Parameter(state_dict.pop("_weight"))

        self.instances_trans = Parameter(
            torch.zeros(self.num_frames, self.num_instances, 3, device=self.device)
        )
        self.instances_quats = Parameter(
            torch.zeros(self.num_frames, self.num_instances, 4, device=self.device)
        )
        self.track_rot_c = nn.Linear(self.gaussian_embedding_dim + self.temporal_embedding_dim, 1).cuda()
        self.track_rot_f = nn.Linear(self.gaussian_embedding_dim + self.temporal_embedding_dim, 1).cuda()
        self.track_trans_c = nn.Linear(self.gaussian_embedding_dim + self.temporal_embedding_dim, 3).cuda()
        self.track_trans_f = nn.Linear(self.gaussian_embedding_dim + self.temporal_embedding_dim, 3).cuda()
        self.track_rot_c.load_state_dict(state_dict.pop("track_rot_c"))
        self.track_rot_f.load_state_dict(state_dict.pop("track_rot_f"))
        self.track_trans_c.load_state_dict(state_dict.pop("track_trans_c"))
        self.track_trans_f.load_state_dict(state_dict.pop("track_trans_f"))
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg
    
    # editting functions
    def remove_instances(self, remove_id_list: List[int]) -> None:
        """
        remove instances from the model
        
        Args:
            remove_id_list: list of instance ids to be removed
        """
        for ins_ids in remove_id_list:
            mask = ~(self.point_ids[..., 0] == ins_ids)
            self._means = Parameter(self._means[mask])
            self._scales = Parameter(self._scales[mask])
            self._quats = Parameter(self._quats[mask])
            self._features_dc = Parameter(self._features_dc[mask])
            self._features_rest = Parameter(self._features_rest[mask])
            self._opacities = Parameter(self._opacities[mask])
            self._embeddings = Parameter(self._embeddings[mask])
            self.point_ids = self.point_ids[mask]
        
    def collect_gaussians_from_ids(self, ids: List[int]) -> Dict:
        gaussian_dict = {}
        for id in ids:
            if id not in gaussian_dict:
                instance_raw_dict = {
                    "_means": self._means[self.point_ids[..., 0] == id],
                    "_scales": self._scales[self.point_ids[..., 0] == id],
                    "_quats": self._quats[self.point_ids[..., 0] == id],
                    "_features_dc": self._features_dc[self.point_ids[..., 0] == id],
                    "_features_rest": self._features_rest[self.point_ids[..., 0] == id],
                    "_opacities": self._opacities[self.point_ids[..., 0] == id],
                    "point_ids": self.point_ids[self.point_ids[..., 0] == id],
                    "embeddings": self._embeddings[self.point_ids[..., 0] == id],
                }
                gaussian_dict[id] = instance_raw_dict
        return gaussian_dict

    def replace_instances(self, replace_dict: Dict[int, int]) -> None:
        """
        replace instances from the model
        
        Args:
            replace_dict: {
                ins_id(to be replaced): ins_id(replace with)
                ...
            }
        """
        new_gaussians_dict = self.collect_gaussians_from_ids(replace_dict.values())
        for ins_id, new_id in replace_dict.items():
            self.remove_instances([ins_id])
            new_gaussian = new_gaussians_dict[new_id]
            self._means = Parameter(torch.cat([self._means, new_gaussian["_means"]], dim=0))
            self._scales = Parameter(torch.cat([self._scales, new_gaussian["_scales"]], dim=0))
            self._quats = Parameter(torch.cat([self._quats, new_gaussian["_quats"]], dim=0))
            self._features_dc = Parameter(torch.cat([self._features_dc, new_gaussian["_features_dc"]], dim=0))
            self._features_rest = Parameter(torch.cat([self._features_rest, new_gaussian["_features_rest"]], dim=0))
            self._opacities = Parameter(torch.cat([self._opacities, new_gaussian["_opacities"]], dim=0))
            self._embeddings = Parameter(torch.cat([self._embeddings, new_gaussian["embeddings"]], dim=0))
            # keeps original point ids
            self.point_ids = torch.cat([self.point_ids, torch.full_like(new_gaussian["point_ids"], ins_id)], dim=0)
    
    def export_gaussians_to_ply(self, alpha_thresh: float, instance_id: List[int] = None) -> Dict[str, torch.Tensor]:
        pts_mask = self.point_ids[..., 0] == instance_id
        
        means = self._means[pts_mask]
        direct_color = self.colors[pts_mask]
        
        activated_opacities = self.get_opacity[pts_mask]
        mask = activated_opacities.squeeze() > alpha_thresh
        return {
            "positions": means[mask],
            "colors": direct_color[mask],
        }