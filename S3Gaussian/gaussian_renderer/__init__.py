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
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from arguments.gaussian_options import BaseOptions

def pre_compute_colors(shs_final, pc : GaussianModel, viewpoint_camera):
    shs_view = shs_final.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    return colors_precomp

def render(args: BaseOptions, viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, \
           stage="fine", return_decomposition=False, return_dx=False, render_feat=False, iter=None, is_train=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=args.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if args.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:
        embeddings = pc.get_embedding
        cam_no = viewpoint_camera.cam_no
        time_diff = viewpoint_camera.time_diff
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, ddict = pc._deformation(means3D, scales, 
                                                                                        rotations, opacity, shs,
                                                                                        time, embeddings, iter, cam_no, time_diff, is_train=is_train)
    else:
        raise NotImplementedError
    
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp_final = None
    if override_color is None:
        if args.convert_SHs_python:
            colors_precomp_final = pre_compute_colors(shs_final, pc, viewpoint_camera)
        else:
            pass
    else:
        colors_precomp_final = override_color

    if "fine" in stage:
        scales = pc.scaling_activation(scales)
        rotations = pc.rotation_activation(rotations)
        opacity = pc.opacity_activation(opacity)
        if args.combine_dynamic_static:
            opacity_dynamic = opacity_final.clone()
            if colors_precomp_final is not None:
                colors_precomp_dynamic = colors_precomp_final.clone()
                shs_dynamic = None
            else:
                colors_precomp_dynamic = None
                shs_dynamic = shs_final.clone()
            dynamic_ratio = opacity_final / (opacity_final + opacity)
            static_ratio = opacity / (opacity_final + opacity)
            opacity_final = opacity_final + opacity
            if colors_precomp_final is not None:
                if override_color is None:
                    colors_precomp_static = pre_compute_colors(shs, pc, viewpoint_camera)
                else:
                    colors_precomp_static = override_color
                colors_precomp_final = colors_precomp_final * dynamic_ratio + colors_precomp_static * static_ratio
            else:
                shs_final_shape = shs_final.shape[1:]
                shs_final = shs_final.view(shs_final.shape[0], -1) * dynamic_ratio + shs.view(shs.shape[0], -1) * static_ratio
                shs_final = shs_final.view(-1, *shs_final_shape)
        
            

    if colors_precomp_final is not None:
        shs_final = None

    rendered_image, depth, normal, weight, radii, _ = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp_final,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3Ds_precomp = cov3D_precomp,
        extra_attrs = None
    )


    result_dict = {}
    
    result_dict.update({
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "depth":depth,
        "weight":weight,
        "normal":normal
    })

    if render_feat and "fine" in stage:
        if not args.no_coarse_deform:
            rendered_feat_c, _, _, _, _, _ = rasterizer(
                means3D = means3D_final,
                means2D = means2D,
                shs = None,
                colors_precomp = ddict["coarse"]["feat"], # [N,3]
                opacities = opacity_final,
                scales = scales_final,
                rotations = rotations_final,
                cov3Ds_precomp = cov3D_precomp,
                extra_attrs = None
            )
        else:
            rendered_feat_c = None

        if not args.no_fine_deform:
            rendered_feat_f, _, _, _, _, _ = rasterizer(
                means3D = means3D_final,
                means2D = means2D,
                shs = None,
                colors_precomp = ddict["fine"]["feat"], # [N,3]
                opacities = opacity_final,
                scales = scales_final,
                rotations = rotations_final,
                cov3Ds_precomp = cov3D_precomp,
                extra_attrs = None
            )
        else:
            rendered_feat_f = None
        
        result_dict.update({"feat_c": rendered_feat_c, "feat_f": rendered_feat_f})

    if return_decomposition and stage == "fine":
        result_dict['ddict_render'] = {}
        def render_dx(dx, do, is_static=True):
            dx_abs = torch.abs(dx) # [N,3]
            # do_abs = torch.abs(do)
            do_abs = torch.sigmoid(do)
            if args.combine_dynamic_static:
                if is_static:
                    rendered_image_d, depth_d, normal_d, weight_d, radii_d, _ = rasterizer(
                        means3D = means3D,
                        means2D = means2D,
                        shs = shs if colors_precomp_static is None else None,
                        colors_precomp = colors_precomp_static if colors_precomp_static is not None else None,
                        opacities = opacity,
                        scales = scales,
                        rotations = rotations,
                        cov3Ds_precomp = cov3D_precomp if cov3D_precomp is not None else None,
                        extra_attrs = None
                    ) 
                else:
                    rendered_image_d, depth_d, normal_d, weight_d, radii_d, _ = rasterizer(
                        means3D = means3D_final,
                        means2D = means2D,
                        shs = shs_dynamic,
                        colors_precomp = colors_precomp_dynamic,
                        opacities = opacity_dynamic,
                        scales = scales_final,
                        rotations = rotations_final,
                        cov3Ds_precomp = cov3D_precomp if cov3D_precomp is not None else None,
                        extra_attrs = None
                    )
            else:

                # Calculating Euclidean distance for each point
                
                distances = torch.norm(dx_abs, dim=1)
                # distances = torch.norm(do_abs, dim=1)
                # Finding the indices of the top 20% points with the largest distances
                top_20_percent_count = int(len(distances) * 0.005)
                _, top_indices = torch.topk(distances, top_20_percent_count)
                dynamic_mask = torch.zeros_like(distances, dtype=torch.bool)
                dynamic_mask[top_indices] = True


                rendered_image_d, depth_d, normal_d, weight_d, radii_d, _ = rasterizer(
                    means3D = means3D_final[dynamic_mask],
                    means2D = means2D[dynamic_mask],
                    shs = shs_final[dynamic_mask] if shs_final is not None else None,
                    colors_precomp = colors_precomp_final[dynamic_mask] if colors_precomp_final is not None else None, # [N,3]
                    opacities = opacity_final[dynamic_mask],
                    # opacities = do_abs[dynamic_mask],
                    scales = scales_final[dynamic_mask],
                    rotations = rotations_final[dynamic_mask],
                    cov3Ds_precomp = cov3D_precomp[dynamic_mask] if cov3D_precomp is not None else None,
                    extra_attrs = None
                )

            dx_max = dx_abs.max(dim=0, keepdim=True)[0]
            colors_precomp_dx = dx_abs / dx_max


            color_dx, _, _, _, _, _ = rasterizer(
                means3D = means3D_final,
                means2D = means2D,
                shs = None,
                colors_precomp = colors_precomp_dx, # [N,3]
                opacities = opacity_final,
                scales = scales_final,
                rotations = rotations_final,
                cov3Ds_precomp = cov3D_precomp if cov3D_precomp is not None else None,
                extra_attrs = None
            )

            return {
                "render": rendered_image_d,
                "depth": depth_d,
                "color" : color_dx,
                "weight": weight_d,
                "normal": normal_d,
                "dx": dx,
                # "render_s": rendered_image_s,
                # "depth_s":depth_s,
                # "visibility_filter_s" : radii_s > 0,
            }
        if not args.no_coarse_deform:
            result_dict['ddict_render']['coarse_render'] = render_dx(ddict['coarse']['dx'], ddict['coarse']['do'], is_static=False)
        
        if not args.no_fine_deform:
            result_dict['ddict_render']['fine_render'] = render_dx(ddict['fine']['dx'], ddict['fine']['do'], is_static=True)
        
        if not args.no_coarse_deform and not args.no_fine_deform:
            result_dict['ddict_render']['coarse_fine_render'] = render_dx(ddict['coarse']['dx'] - ddict['fine']['dx'], ddict['coarse']['do']-ddict['coarse']['do'])

    if return_dx and "fine" in stage:
        result_dict.update({"ddict": ddict})

    sky_color = pc._sky_model(viewpoint_camera, acc=weight, is_train=is_train)
    result_dict["render"] = result_dict["render"] * result_dict["weight"] + sky_color * (1 - result_dict["weight"])
    result_dict.update({"sky_color": sky_color})

    return result_dict