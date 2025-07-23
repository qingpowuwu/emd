import os
from typing import Callable, Dict, List
import open3d as o3d
import imageio
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from lpipsPyTorch import lpips
from torch import Tensor
from tqdm import tqdm, trange
from gaussian_renderer import render

from utils.image_utils import psnr
from utils.visualization_tools import (
    resize_five_views,
    scene_flow_to_rgb,
    to8b,
    visualize_depth,
)

depth_visualizer = lambda frame, opacity: visualize_depth(
    frame,
    opacity,
    lo=4.0,
    hi=120,
    depth_curve_fn=lambda x: -np.log(x + 1e-6),
)
flow_visualizer = (
    lambda frame: scene_flow_to_rgb(
        frame,
        background="bright",
        flow_max_radius=1.0,
    )
    .cpu()
    .numpy()
)
get_numpy: Callable[[Tensor], np.ndarray] = lambda x: x.squeeze().cpu().numpy()
non_zero_mean: Callable[[Tensor], float] = (
    lambda x: sum(x) / len(x) if len(x) > 0 else -1
)

def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    rins = colors[fg_mask][s[:, 0] < m, 0]
    gins = colors[fg_mask][s[:, 1] < m, 1]
    bins = colors[fg_mask][s[:, 2] < m, 2]

    rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
    rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)

def render_pixels(
    args,
    viewpoint_stack,
    gaussians,
    bg,
    step: int = 0,
    compute_metrics: bool = True,
    return_decomposition: bool = True,
    debug:bool = False
):
    """
    Render pixel-related outputs from a model.

    Args:
        ....skip obvious args
        compute_metrics (bool, optional): Whether to compute metrics. Defaults to False.
        vis_indices (Optional[List[int]], optional): Indices to visualize. Defaults to None.
        return_decomposition (bool, optional): Whether to visualize the static-dynamic decomposition. Defaults to True.
    """
    # set up render function
    render_results = render_func(
        args,
        viewpoint_stack,
        gaussians,
        bg,
        step,
        compute_metrics=compute_metrics,
        return_decomposition=return_decomposition,
        debug = debug
    )
    if compute_metrics:
        num_samples = len(viewpoint_stack)
        print(f"Eval over {num_samples} images:")
        print(f"\tPSNR: {render_results['metrics']['psnr']:.4f}")
        print(f"\tSSIM: {render_results['metrics']['ssim']:.4f}")
        print(f"\tLPIPS: {render_results['metrics']['lpips']:.4f}")
        # print(f"\tFeature PSNR: {render_results['feat_psnr']:.4f}")
        print(f"\tMasked PSNR: {render_results['metrics']['masked_psnr']:.4f}")
        print(f"\tMasked SSIM: {render_results['metrics']['masked_ssim']:.4f}")
        # print(f"\tMasked Feature PSNR: {render_results['masked_feat_psnr']:.4f}")

    return render_results


def render_func(
    args,
    viewpoint_stack,
    gaussians,
    bg,
    step: int = 0,
    compute_metrics: bool = False,
    return_decomposition:bool = False,
    num_cams: int = 3,
    debug: bool = False,
    save_seperate_pcd = False
):
    """
    Renders a dataset utilizing a specified render function.
    For efficiency and space-saving reasons, this function doesn't store the original features; instead, it keeps
    the colors reduced via PCA.
    TODO: clean up this function

    Parameters:
        dataset: Dataset to render.
        render_func: Callable function used for rendering the dataset.
        compute_metrics: Optional; if True, the function will compute and return metrics. Default is False.
    """
    # rgbs
    rgbs, gt_rgbs = [], []
    static_rgbs, dynamic_rgbs = [], []

    depths, normals, opacities = [], [], []

    ddict_renders = {}


    if compute_metrics:
        psnrs, ssim_scores= [], []
        masked_psnrs, masked_ssims = [], []
        lpipss = []

    depth_max = -np.inf
    with torch.no_grad():
        for i in tqdm(range(len(viewpoint_stack)), desc=f"rendering full data", dynamic_ncols=True):
            viewpoint_cam = viewpoint_stack[i]
            render_pkg = render(args, viewpoint_cam, gaussians, bg, return_decomposition = return_decomposition, return_dx=True)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            # ------------- rgb ------------- #
            rgb = image
            gt_rgb = viewpoint_cam.original_image.cuda()

            rgbs.append(get_numpy(rgb.permute(1, 2, 0)))
            gt_rgbs.append(get_numpy(gt_rgb.permute(1, 2, 0)))
            
            # ------------- depth ------------- #
            depth = render_pkg["depth"]
            depth_np = depth.permute(1, 2, 0).cpu().numpy()
            if depth_np.max() > depth_max:
                depth_max = depth_np.max()
            # depth_np /= depth_np.max()
            depths.append(depth_np)
            # depth_np = np.repeat(depth_np, 3, axis=2)
            
            # ------------- normal ------------- #
            normal = render_pkg["normal"]
            normal_np = normal.permute(1, 2, 0).cpu().numpy()
            normal_np = (normal_np + 1) / 2
            normals.append(normal_np)

            # ------------- opacities ------------- #
            opacity = render_pkg["weight"]
            opacity_np = opacity.permute(1, 2, 0).cpu().numpy()
            opacities.append(opacity_np)
            
            # for decomposition
            if return_decomposition:
                ddict_render = render_pkg['ddict_render']
                
                for k, v in ddict_render.items():
                    if k not in list(ddict_renders.keys()):
                        ddict_renders[k] = {}
                    for kk, vv in v.items():
                        if kk not in list(ddict_renders[k].keys()):
                            ddict_renders[k][kk] = []
                        if "normal" in kk:
                            vv = (vv + 1) / 2
                        if "depth" in kk:
                            # vv = vv / vv.max()
                            vv = vv
                        if "dx" not in kk:
                            vv = vv.permute(1, 2, 0).cpu().numpy()
                        ddict_renders[k][kk].append(vv)

            if compute_metrics:
                psnrs.append(psnr(rgb, gt_rgb).mean().double().item())
                # ssim_scores.append(ssim(rgb, gt_rgb).mean().item())
                ssim_scores.append(
                    ssim(
                        get_numpy(rgb),
                        get_numpy(gt_rgb),
                        data_range=1.0,
                        channel_axis=0,
                    )
                )
                lpipss.append(torch.tensor(lpips(rgb, gt_rgb,net_type='alex')).mean().item())
                
                dynamic_mask = get_numpy(viewpoint_cam.dynamic_mask).astype(bool)
                if dynamic_mask.sum() > 0:
                    rgb_d = rgb.permute(1, 2, 0)[dynamic_mask]
                    rgb_d = rgb_d.permute(1, 0)
                    gt_rgb_d = gt_rgb.permute(1, 2, 0)[dynamic_mask]
                    gt_rgb_d = gt_rgb_d.permute(1, 0)

                    masked_psnrs.append(
                    psnr(rgb_d, gt_rgb_d).mean().double().item()
                    )
                    masked_ssims.append(
                        ssim(
                            get_numpy(rgb.permute(1, 2, 0)),
                            get_numpy(gt_rgb.permute(1, 2, 0)),
                            data_range=1.0,
                            channel_axis=-1,
                            full=True,
                        )[1][dynamic_mask].mean()
                    )

        depths = [depth / depth_max for depth in depths]
        if return_decomposition:
            for k, v in ddict_renders.items():
                for kk, vv in v.items():
                    if "depth" in kk:
                        ddict_renders[k][kk] = [depth / depth_max for depth in vv]

        def get_flow(dx_list):
            forward_flows, backward_flows = [], []
            bf_color_first = []
            ff_color_last = []
            for t in range(len(dx_list)): # 防止越界
                if t < len(dx_list)-num_cams:
                    # forward_flow_t 归一化一下
                    camera_move = torch.from_numpy(viewpoint_stack[t+num_cams].T - viewpoint_stack[t].T).cuda().float()
                    # forward_flow_t = dx_list[t + num_cams] - dx_list[t] - camera_move
                    forward_flow_t = dx_list[t + num_cams] - dx_list[t]
                    ff_color = (forward_flow_t.abs() / forward_flow_t.abs().max())

                    # ff_color = flow_visualizer(forward_flow_t)
                    # ff_color = torch.from_numpy(ff_color).to("cuda") 
                    # if debug:
                    #     ff_color = (ff_color - torch.min(ff_color)) / (torch.max(ff_color) - torch.min(ff_color) + 1e-6)  # 归一化，避免除零错误

                    # 创建 Open3D 点云对象
                    # xyz = gaussians._xyz.cpu().detach().numpy()
                    # colors = (forward_flow_t.abs() / forward_flow_t.abs().max()).cpu().detach().numpy()
                    # # colors = ff_color.cpu().detach().numpy()
                    # point_cloud = o3d.geometry.PointCloud()
                    # point_cloud.points = o3d.utility.Vector3dVector(xyz)
                    # point_cloud.colors = o3d.utility.Vector3dVector(colors)
                    # o3d.io.write_point_cloud('forward_flows_point_cloud_%03d.ply'%(t), point_cloud)

                    if t == len(dx_list)-num_cams-1 or t == len(dx_list)-num_cams-2 or t == len(dx_list)-num_cams-3: 
                        ff_color_last.append(ff_color)              
                    render_pkg2 = render(args, viewpoint_stack[t], gaussians, bg, override_color=ff_color)
                    ff_map = render_pkg2['render'].permute(1, 2, 0).cpu().numpy()

                    forward_flows.append(ff_map)
                
                # 同时处理 backward flow，除第一个时刻外
                if t > num_cams-1:
                    camera_move = torch.from_numpy(viewpoint_stack[t].T - viewpoint_stack[t-num_cams].T).cuda().float()
                    # backward_flow_t = dx_list[t] - dx_list[t - num_cams] - camera_move
                    backward_flow_t = dx_list[t] - dx_list[t - num_cams]

                    bf_color = (backward_flow_t.abs() / backward_flow_t.abs().max())

                    # bf_color = flow_visualizer(backward_flow_t)
                    # bf_color = torch.from_numpy(bf_color).to("cuda") 
                    # if debug:
                    #     bf_color = (bf_color - torch.min(bf_color)) / (torch.max(bf_color) - torch.min(bf_color) + 1e-6)  # 归一化，避免除零错误

                    # # 创建 Open3D 点云对象
                    # xyz = gaussians._xyz.cpu().detach().numpy()
                    # colors = bf_color.cpu().detach().numpy()
                    # point_cloud = o3d.geometry.PointCloud()
                    # point_cloud.points = o3d.utility.Vector3dVector(xyz)
                    # point_cloud.colors = o3d.utility.Vector3dVector(colors)
                    # o3d.io.write_point_cloud('backward_flows_point_cloud_%03d.ply'%(t), point_cloud)

                    if t == num_cams or t == num_cams+1 or t == num_cams+2: 
                        bf_color_first.append(bf_color)                 
                    # viewpoint_cam 要变化
                    render_pkg2 = render(args, viewpoint_stack[t], gaussians, bg, override_color=bf_color)
                    bf_map = render_pkg2['render'].permute(1, 2, 0).cpu().numpy()

                    backward_flows.append(bf_map)

            for i, bf_color in enumerate(bf_color_first):
                render_pkg3 = render(args, viewpoint_stack[i], gaussians, bg, override_color=bf_color)            
                bf_map_first = render_pkg3['render'].permute(1, 2, 0).cpu().numpy()       
                # 对于 backward flow 的第一个时刻，复制第一个计算的 forward flow
                backward_flows.insert(i, bf_map_first)

            for i, ff_color in enumerate(ff_color_last):
                # 对于 forward flow 的最后一个时刻，复制最后一个计算的 backward flow
                render_pkg4 = render(args, viewpoint_stack[len(viewpoint_stack)-num_cams+i], gaussians, bg, override_color=ff_color)            
                ff_map_last = render_pkg4['render'].permute(1, 2, 0).cpu().numpy()       
                forward_flows.append(ff_map_last)           
            return forward_flows, backward_flows


    results_dict = {
        "metrics": {},
        "renders": {},
    }
    results_dict["metrics"]["psnr"] = non_zero_mean(psnrs) if compute_metrics else -1
    results_dict["metrics"]["ssim"] = non_zero_mean(ssim_scores) if compute_metrics else -1
    results_dict["metrics"]["lpips"] = non_zero_mean(lpipss) if compute_metrics else -1
    results_dict["metrics"]["masked_psnr"] = non_zero_mean(masked_psnrs) if compute_metrics else -1
    results_dict["metrics"]["masked_ssim"] = non_zero_mean(masked_ssims) if compute_metrics else -1

    results_dict["renders"]["rgbs"] = rgbs
    results_dict["renders"]["depths"] = depths
    results_dict["renders"]["normals"] = normals
    results_dict["renders"]["opacities"] = opacities
    results_dict["renders"]["gt_rgbs"] = gt_rgbs
    if return_decomposition:
        for k, v in ddict_renders.items():
            for kk, vv in v.items():
                if kk == "dx":
                    forward_flows, backward_flows = get_flow(vv)
                    results_dict["renders"][f"{k}_forward_flows"] = forward_flows
                    results_dict["renders"][f"{k}_backward_flows"] = backward_flows
                else:
                    results_dict["renders"][f"{k}_{kk}"] = vv

    return results_dict


def save_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_seperate_video: bool = False,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if save_seperate_video:
        return_frame = save_seperate_videos(
            render_results,
            save_pth,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    else:
        return_frame = save_concatenated_videos(
            render_results,
            save_pth,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    return return_frame


def save_concatenated_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if num_timestamps == 1:  # it's an image
        writer = imageio.get_writer(save_pth, mode="I")
        return_frame_id = 0
    else:
        return_frame_id = num_timestamps // 2
        writer = imageio.get_writer(save_pth, mode="I", fps=fps)
    for i in trange(num_timestamps, desc="saving video", dynamic_ncols=True):
        merged_list = []
        for key in keys:
            if key == "sky_masks":
                frames = render_results["opacities"][i * num_cams : (i + 1) * num_cams]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif key == "sky_masks":
                frames = [
                    1 - np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            # elif "depth" in key:

            #     frames = [
            #         depth_visualizer(frame, opacity)
            #         for frame, opacity in zip(frames, opacities)
            #     ]
            frames = resize_five_views(frames)
            frames = np.concatenate(frames, axis=1)
            merged_list.append(frames)
        merged_frame = to8b(np.concatenate(merged_list, axis=0))
        if i == return_frame_id:
            return_frame = merged_frame
        writer.append_data(merged_frame)
    writer.close()
    if verbose:
        print(f"saved video to {save_pth}")
    del render_results
    return {"concatenated_frame": return_frame}


def save_seperate_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    fps: int = 10,
    verbose: bool = False,
    save_images: bool = False,
):
    return_frame_id = num_timestamps // 2
    return_frame_dict = {}
    for key in keys:
        tmp_save_pth = save_pth.replace(".mp4", f"_{key}.mp4")
        tmp_save_pth = tmp_save_pth.replace(".png", f"_{key}.png")
        if num_timestamps == 1:  # it's an image
            writer = imageio.get_writer(tmp_save_pth, mode="I")
        else:
            writer = imageio.get_writer(tmp_save_pth, mode="I", fps=fps)
        if key not in render_results or len(render_results[key]) == 0:
            continue
        for i in range(num_timestamps):
            if key == "sky_masks":
                frames = render_results["opacities"][i * num_cams : (i + 1) * num_cams]
            else:
                # 这里取3个，得到3个视角
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif key == "sky_masks":
                frames = [
                    1 - np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            # elif "depth" in key:
            #     opacities = render_results[key.replace("depths", "opacities")][
            #         i * num_cams : (i + 1) * num_cams
            #     ]
            #     frames = [
            #         depth_visualizer(frame, opacity)
            #         for frame, opacity in zip(frames, opacities)
            #     ]
            frames = resize_five_views(frames)
            if save_images:
                if i == 0:
                    os.makedirs(tmp_save_pth.replace(".mp4", ""), exist_ok=True)
                for j, frame in enumerate(frames):
                    imageio.imwrite(
                        tmp_save_pth.replace(".mp4", f"_{i*3 + j:03d}.png"),
                        to8b(frame),
                    )
            frames = to8b(np.concatenate(frames, axis=1))
            writer.append_data(frames) # [H,W,3]
            if i == return_frame_id:
                return_frame_dict[key] = frames
        # close the writer
        writer.close()
        del writer
        if verbose:
            print(f"saved video to {tmp_save_pth}")
    del render_results
    return return_frame_dict
