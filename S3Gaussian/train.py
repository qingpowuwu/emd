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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, compute_depth
from gaussian_renderer import render
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from utils.scene_utils import render_training_image
from utils.extra_utils import o3d_knn, weighted_l2_loss_v2

import numpy as np
import time
import json
from utils.params_utils import merge_hparams
from utils.video_utils import render_pixels, save_videos
from arguments.gaussian_options import auto_argparse_from_class, BaseOptions
from torch.utils.tensorboard import SummaryWriter
   
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

@torch.no_grad()
def do_evaluation(
    args: BaseOptions,
    scene: Scene,
    gaussians: GaussianModel,
    render_full,
    step: int = 0,
):

    eval_dir = os.path.join(args.model_path,"eval")
    os.makedirs(eval_dir,exist_ok=True)
    viewpoint_stack_full = scene.getFullCameras()
    viewpoint_stack_test = scene.getTestCameras()
    viewpoint_stack_train = scene.getTrainCameras()
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    if len(viewpoint_stack_test) != 0:
        print("Evaluating Test Set Pixels...")
        render_results = render_pixels(
            args,
            viewpoint_stack_test,
            gaussians,
            bg,
            step=step,
            compute_metrics=True,
            debug=args.debug_test
        )
        eval_dict = {}
        for k, v in render_results["metrics"].items():
            eval_dict[f"pixel_metrics/test/{k}"] = v
                
        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/test_videos", exist_ok=True)
        
        test_metrics_file = f"{eval_dir}/metrics/{step}_images_test_{current_time}.json"
        with open(test_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {test_metrics_file}")

        video_output_pth = f"{eval_dir}/test_videos/{step}.mp4"

        vis_frame_dict = save_videos(
            render_results["renders"],
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_test)//3),
            # keys=render_keys,
            keys=render_results["renders"].keys(),
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )

        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
    if len(viewpoint_stack_train) != 0 and len(viewpoint_stack_test) != 0:
        print("Evaluating train Set Pixels...")
        render_results = render_pixels(
            args,
            viewpoint_stack_train,
            gaussians,
            bg,
            step=step,
            compute_metrics=True,
            debug=args.debug_test
        )
        eval_dict = {}
        for k, v in render_results["metrics"].items():
            eval_dict[f"pixel_metrics/train/{k}"] = v
                
        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/train_videos", exist_ok=True)
        
        train_metrics_file = f"{eval_dir}/metrics/{step}_images_train.json"
        with open(train_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {train_metrics_file}")

        video_output_pth = f"{eval_dir}/train_videos/{step}.mp4"

        vis_frame_dict = save_videos(
            render_results["renders"],
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_train)//3),
            # keys=render_keys,
            keys=render_results["renders"].keys(),
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )

        del render_results
        torch.cuda.empty_cache()

    if render_full:
        print("Evaluating Full Set...")
        render_results = render_pixels(
            args,
            viewpoint_stack_full,
            gaussians,
            bg,
            step=step,
            compute_metrics=True,
            debug=args.debug_test
        )
        eval_dict = {}
        for k, v in render_results["metrics"].items():
            eval_dict[f"pixel_metrics/full/{k}"] = v
                
        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/full_videos", exist_ok=True)

        test_metrics_file = f"{eval_dir}/metrics/{step}_images_full_{current_time}.json"
        with open(test_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {test_metrics_file}")

        # if render_video_postfix is None:
        video_output_pth = f"{eval_dir}/full_videos/{step}.mp4"
        vis_frame_dict = save_videos(
            render_results["renders"],
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_full)//3),
            # keys=render_keys,
            keys=render_results["renders"].keys(),
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )
        
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()

def scene_reconstruction(args: BaseOptions, gaussians: GaussianModel, scene: Scene, tb_writer):

    gaussians.training_setup(args)
    stage = "coarse"
    first_iter = 1
    if args.load_checkpoint:
        (model_params, first_iter) = torch.load(args.load_checkpoint, map_location="cuda")
        gaussians.restore(model_params, args)
    final_iter = args.coarse_iterations + args.iterations + 1
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    prev_num_pts = 0
    viewpoint_stack = None
    for iteration in progress_bar:
        if iteration == args.coarse_iterations:
            stage = "fine"
            if args.freeze_static:
                gaussians._xyz.requires_grad = False
                gaussians._scaling.requires_grad = False
                gaussians._rotation.requires_grad = False
            print("Switching to fine stage")

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(is_shuffle=True)
        viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))

        if iteration == args.debug_from:
            args.debug = True
        render_pkg = render(args, viewpoint_cam, gaussians, background, \
                            stage=stage, return_dx=True, render_feat = True if ('fine' in stage and args.feat_head) else False, iter=iteration, is_train=True)

        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        depth = render_pkg["depth"]
        weight = render_pkg["weight"]

        gt_image = viewpoint_cam.original_image.cuda()
        gt_depth = viewpoint_cam.depth_map.cuda()
        sky_mask = viewpoint_cam.sky_mask.cuda().bool() if args.load_sky_mask else None

        if sky_mask is not None:
            mask = ~sky_mask
        else:
            mask = torch.ones_like(gt_image)

        Ll1 = l1_loss(image, gt_image)
        psnr_training = psnr(image, gt_image).mean().double()
        # norm        
        loss = Ll1

        dx_loss = 0.0 * loss
        dshs_loss = 0.0 * loss
        tv_loss = 0.0 * loss
        loss_feat = 0.0 * loss
        reg_loss = 0.0 * loss
        opacity_sparse_loss = 0.0 * loss
        if 'fine' in stage:
            ddict = render_pkg["ddict"]
            if not args.no_dx and args.lambda_dx !=0:
                dx_loss_f = dx_loss_c = dx_loss_f2c = 0.0
                if not args.no_coarse_deform:
                    dx_abs_c = torch.abs(ddict['coarse']['dx'])
                    dx_loss_c = torch.mean(dx_abs_c) * args.lambda_dx
                
                if not args.no_fine_deform:
                    dx_abs_f = torch.abs(ddict['fine']['dx'])
                    dx_loss_f = torch.mean(dx_abs_f) * args.lambda_dx
                
                if not args.no_fine_deform and not args.no_coarse_deform and args.lambda_f2c != 0:
                    dx_abs_f2c = torch.abs(ddict['fine']['dx'] - ddict['coarse']['dx'])
                    dx_loss_f2c = torch.mean(dx_abs_f2c) * args.lambda_f2c
                
                dx_loss = dx_loss_c + dx_loss_f + dx_loss_f2c
                loss += dx_loss

            if not args.no_ds and args.lambda_ds != 0:
                ds_loss_f = ds_loss_c = 0.0
                if not args.no_coarse_deform:
                    ds_abs_c = torch.abs(ddict['coarse']['ds'])
                    ds_loss_c = torch.mean(ds_abs_c) * args.lambda_ds

                if not args.no_fine_deform:
                    ds_abs_f = torch.abs(ddict['fine']['ds'])
                    ds_loss_f = torch.mean(ds_abs_f) * args.lambda_ds
                
                ds_loss = ds_loss_c + ds_loss_f
                loss += ds_loss
            
            if not args.no_dr and args.lambda_dr != 0:
                dr_loss_f = dr_loss_c = 0.0
                if not args.no_coarse_deform:
                    dr_abs_c = torch.abs(ddict['coarse']['dr'])
                    dr_loss_c = torch.mean(dr_abs_c) * args.lambda_dr
                
                if not args.no_fine_deform:
                    dr_abs_f = torch.abs(ddict['fine']['dr'])
                    dr_loss_f = torch.mean(dr_abs_f) * args.lambda_dr
                
                dr_loss = dr_loss_c + dr_loss_f
                loss += dr_loss
            
            if not args.no_do and args.lambda_do != 0:
                do_loss_f = do_loss_c = 0.0
                if not args.no_coarse_deform:
                    do_abs_c = torch.abs(ddict['coarse']['do'])
                    do_loss_c = torch.mean(do_abs_c) * args.lambda_do
                
                if not args.no_fine_deform:
                    do_abs_f = torch.abs(ddict['fine']['do'])
                    do_loss_f = torch.mean(do_abs_f) * args.lambda_do
                
                do_loss = do_loss_c + do_loss_f
                loss += do_loss

            if not args.no_dshs and args.lambda_dshs != 0:
                dshs_loss_c = dshs_loss_f = 0.0
                if not args.no_coarse_deform:
                    dshs_abs_c = torch.abs(ddict['coarse']['dshs'])
                    dshs_loss_c = torch.mean(dshs_abs_c) * args.lambda_dshs
                
                if not args.no_fine_deform:
                    dshs_abs_f = torch.abs(ddict['fine']['dshs'])
                    dshs_loss_f = torch.mean(dshs_abs_f) * args.lambda_dshs

                dshs_loss = dshs_loss_c + dshs_loss_f
                loss += dshs_loss

            if args.time_smoothness_weight != 0:
                # tv_loss = 0
                tv_loss = gaussians.compute_regulation(args.time_smoothness_weight, args.l1_time_planes, args.plane_tv_weight)
                loss += tv_loss
                
            if args.feat_head:
                gt_feat = viewpoint_cam.feat_map.permute(2,0,1).to('cuda')
                loss_feat_c = loss_feat_f = 0.0
                if not args.no_coarse_deform:
                    feat_c = render_pkg['feat_c']
                    loss_feat_c = l2_loss(feat_c, gt_feat) * args.lambda_feat
                
                if not args.no_fine_deform:
                    feat_f = render_pkg['feat_f']
                    loss_feat_f = l2_loss(feat_f, gt_feat) * args.lambda_feat
                loss_feat = loss_feat_c + loss_feat_f
                loss += loss_feat

            if args.lambda_reg >= 0:
                if prev_num_pts != gaussians._xyz.shape[0]:
                    neighbor_sq_dist, neighbor_indices = o3d_knn(gaussians._xyz.detach().cpu().numpy(), 20)
                    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
                    neighbor_indices = torch.tensor(neighbor_indices).cuda().long().contiguous()
                    neighbor_weight = torch.tensor(neighbor_weight).cuda().float().contiguous()
                    prev_num_pts = gaussians._xyz.shape[0]
                
                emb = gaussians._embedding[:,None,:].repeat(1,20,1)
                emb_knn = gaussians._embedding[neighbor_indices]
                reg_loss = args.lambda_reg * weighted_l2_loss_v2(emb, emb_knn, neighbor_weight)
                loss += reg_loss
            
            if args.lambda_opacity_sparse > 0:
                opacity = gaussians.get_opacity
                opacity = opacity.clamp(1e-6, 1-1e-6)
                log_opacity = opacity * torch.log(opacity)
                log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
                opacity_sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()
                opacity_sparse_loss = opacity_sparse_loss * args.lambda_opacity_sparse
                loss += opacity_sparse_loss

        depth_loss = 0.0 * loss
        if args.lambda_depth != 0:
            depth_loss = compute_depth("l2", depth * mask, gt_depth * mask) * args.lambda_depth
            loss += depth_loss
        
        ssim_loss = 0.0 * loss
        if args.lambda_dssim != 0:
            ssim_loss = ssim(image, gt_image)
            ssim_loss = args.lambda_dssim * (1.0-ssim_loss)
            loss += ssim_loss

        sky_loss = 0.0 * loss
        if args.lambda_sky > 0 and sky_mask is not None and sky_mask.sum() > 0:
            weight = torch.clamp(weight, min=1e-6, max=1.-1e-6)
            sky_loss = torch.where(sky_mask, -torch.log(1 - weight), -torch.log(weight)).mean()
            loss += args.lambda_sky * sky_loss

        # breakpoint()
        loss.backward()

        viewspace_point_tensor_grad = viewspace_point_tensor.grad

        print_dict = {
            "Loss": f"{loss.item():.{4}f}",
            "Ll1": f"{Ll1.item():.{4}f}",
            "Ldepth": f"{depth_loss.item():.{4}f}",
            "Lssim": f"{ssim_loss.item():.{4}f}",
            "Lsky": f"{sky_loss.item():.{4}f}",
            "Lfeat": f"{loss_feat.item():.{4}f}",
            "psnr": f"{psnr_training:.{4}f}",
            "point":f"{gaussians._xyz.shape[0]}",
        }

        progress_bar.set_postfix(print_dict)
        log_dict = {
            "Loss": loss.item(),
            "Ll1": Ll1.item(),
            "Ldepth": depth_loss.item(),
            "Lssim": ssim_loss.item(),
            "Lsky": sky_loss.item(),
            "Lreg": reg_loss.item(),
            "Ldx": dx_loss.item(),
            "Ldshs": dshs_loss.item(),
            "Ltv": tv_loss.item(),
            "Lfeat": loss_feat.item(),
            "Lopacity_sparse": opacity_sparse_loss.item(),
            "psnr": psnr_training,
            "point":gaussians._xyz.shape[0],
        }
        for log_key, log_value in log_dict.items():
            tb_writer.add_scalar(log_key, log_value, iteration)

        with torch.no_grad():
            if iteration % args.visualize_iterations == 0:
                    render_training_image(args, scene, gaussians, [viewpoint_cam], render, background, iteration, stage)

            if iteration < args.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = args.opacity_threshold_coarse
                    densify_threshold = args.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = args.opacity_threshold_fine_init - iteration*(args.opacity_threshold_fine_init - args.opacity_threshold_fine_after)/(args.densify_until_iter)  
                    densify_threshold = args.densify_grad_threshold_fine_init - iteration*(args.densify_grad_threshold_fine_init - args.densify_grad_threshold_after)/(args.densify_until_iter )  

                if  iteration > args.densify_from_iter and iteration % args.densification_interval == 0 and gaussians.get_xyz.shape[0]<args.max_num_pts:
                    size_threshold = 20 if iteration > args.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > args.pruning_from_iter and iteration % args.pruning_interval == 0 : # and gaussians.get_xyz.shape[0]>200000
                    size_threshold = 20 if iteration > args.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                    
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                do_evaluation(
                    args,
                    scene,
                    gaussians,
                    render_full=True,
                    step=iteration,
                )


def prepare_output_and_logger():    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(args.model_path)
    return tb_writer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    torch.cuda.empty_cache()
    setup_seed(6666)
    base_options  = BaseOptions()
    parser = auto_argparse_from_class(base_options)
    args = parser.parse_args()

    args.checkpoint_iterations.append(args.coarse_iterations + args.iterations)
    print("Working dir: " + args.model_path)
    tb_writer = prepare_output_and_logger()        

    gaussians = GaussianModel(args)

    scene = Scene(args, gaussians)
    
    if not args.eval_only:
        scene_reconstruction(args, gaussians, scene, tb_writer)
    else:
        if args.load_checkpoint:
            (model_params, first_iter) = torch.load(args.load_checkpoint, map_location="cuda")
            gaussians.restore(model_params, args)
        do_evaluation(
            args,
            scene,
            gaussians,
            render_full=True,
            step=args.iterations,
        )
