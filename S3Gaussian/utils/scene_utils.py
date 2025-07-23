import torch
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
from arguments.gaussian_options import BaseOptions
import numpy as np

@torch.no_grad()
def get_defromation_render_results(render_pkg):
    image = render_pkg["render"]
    depth = render_pkg["depth"]
    color = render_pkg["color"]
    weight = render_pkg["weight"]
    normal = render_pkg["normal"]
    image_np = image.permute(1, 2, 0).cpu().numpy()
    color_np = color.permute(1, 2, 0).cpu().numpy()
    depth_np = depth.permute(1, 2, 0).cpu().numpy()
    depth_np /= depth_np.max()
    depth_np = np.repeat(depth_np, 3, axis=2)
    weight_np = weight.permute(1, 2, 0).cpu().numpy()
    # weight_np /= weight_np.max()
    weight_np = np.repeat(weight_np, 3, axis=2)
    normal_np = normal.permute(1, 2, 0).cpu().numpy()
    normal_np = (normal_np + 1) / 2
    image_np = np.concatenate((color_np, image_np, depth_np, weight_np, normal_np), axis=1)
    return image_np
    

@torch.no_grad()
def render_training_image(args: BaseOptions, scene, gaussians, viewpoints, render_func, background, iteration, stage="fine"):
    def render(gaussians, viewpoint, path, scaling):
        render_pkg = render_func(args, viewpoint, gaussians, background, \
                            stage=stage, return_dx=True, render_feat = True, return_decomposition=True, iter=iteration)
        
        label1 = f"iter:{iteration}"

        image = render_pkg["render"]
        depth = render_pkg["depth"]
        weight = render_pkg["weight"]
        normal = render_pkg["normal"]
        gt_np = viewpoint.original_image.permute(1,2,0).cpu().numpy()
        image_np = image.permute(1, 2, 0).cpu().numpy()  # 转换通道顺序为 (H, W, 3)
        depth_np = depth.permute(1, 2, 0).cpu().numpy()
        depth_np /= depth_np.max()
        depth_np = np.repeat(depth_np, 3, axis=2)
        weight_np = weight.permute(1, 2, 0).cpu().numpy()
        # weight_np /= weight_np.max()
        weight_np = np.repeat(weight_np, 3, axis=2)
        normal_np = normal.permute(1, 2, 0).cpu().numpy()
        normal_np = (normal_np + 1) / 2
        image_np = np.concatenate((gt_np, image_np, depth_np, weight_np, normal_np), axis=1)
        
        if not args.no_fine_deform and stage == "fine":
            image_fine_np = get_defromation_render_results(render_pkg['ddict_render']["fine_render"])
            image_np = np.concatenate((image_np, image_fine_np), axis=0)
        
        if not args.no_coarse_deform and stage == "fine":
            image_coarse_np = get_defromation_render_results(render_pkg['ddict_render']["coarse_render"])
            image_np = np.concatenate((image_np, image_coarse_np), axis=0)
        
        if not args.no_fine_deform and not args.no_coarse_deform and stage == "fine":
            image_coarse_fine_np = get_defromation_render_results(render_pkg['ddict_render']["coarse_fine_render"])
            image_np = np.concatenate((image_np, image_coarse_fine_np), axis=0)
        
        image_with_labels = Image.fromarray((np.clip(image_np,0,1) * 255).astype('uint8'))  # 转换为8位图像
        # 创建PIL图像对象的副本以绘制标签
        draw1 = ImageDraw.Draw(image_with_labels)

        # 选择字体和字体大小
        font = ImageFont.truetype('./utils/TIMES.TTF', size=40)  # 请将路径替换为您选择的字体文件路径

        # 选择文本颜色
        text_color = (255, 0, 0)  # 白色

        # 选择标签的位置（左上角坐标）
        label1_position = (10, 10)

        # 在图像上添加标签
        draw1.text(label1_position, label1, fill=text_color, font=font)
        
        image_with_labels.save(path)
    render_base_path = os.path.join(scene.model_path, f"render")
    # point_cloud_path = os.path.join(render_base_path,"pointclouds")
    image_path = os.path.join(render_base_path,"images")
    if not os.path.exists(os.path.join(scene.model_path, f"render")):
        os.makedirs(render_base_path)
    # if not os.path.exists(point_cloud_path):
    #     os.makedirs(point_cloud_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    
    # point_save_path = os.path.join(point_cloud_path,f"{iteration}.jpg")
    for idx in range(len(viewpoints)):
        image_save_path = os.path.join(image_path,f"{iteration}_{idx}.jpg")
        render(gaussians,viewpoints[idx],image_save_path, scaling = 1)

        # pc_mask = gaussians.get_opacity
        # pc_mask = pc_mask > 0.1
        # xyz = gaussians.get_xyz.detach()[pc_mask.squeeze()].cpu().permute(1,0).numpy()
        # visualize_and_save_point_cloud(xyz, viewpoints[idx].R, viewpoints[idx].T, point_save_path)

    # 如果需要，您可以将PIL图像转换回PyTorch张量
    # return image
    # image_with_labels_tensor = torch.tensor(image_with_labels, dtype=torch.float32).permute(2, 0, 1) / 255.0
def visualize_and_save_point_cloud(point_cloud, R, T, filename):
    # 创建3D散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    R = R.T
    # 应用旋转和平移变换
    T = -R.dot(T)
    transformed_point_cloud = np.dot(R, point_cloud) + T.reshape(-1, 1)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(transformed_point_cloud.T)  # 转置点云数据以匹配Open3D的格式
    # transformed_point_cloud[2,:] = -transformed_point_cloud[2,:]
    # 可视化点云
    ax.scatter(transformed_point_cloud[0], transformed_point_cloud[1], transformed_point_cloud[2], c='g', marker='o')
    ax.axis("off")
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # 保存渲染结果为图片
    plt.savefig(filename)

