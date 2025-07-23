"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
import os, sys
from PIL import Image
import glob
import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '..'))
cwd = os.getcwd()
print(cwd)
from Utils.utils import str2bool, AverageMeter, depth_read 
import Models
import Datasets
from Datasets.My_Dataset import My_Dataset
from PIL import ImageOps
import matplotlib.pyplot as plt
import time
import cv2

#Training setttings
parser = argparse.ArgumentParser(description='KITTI Depth Completion Task TEST')
parser.add_argument('--dataset', type=str, default='kitti', choices = Datasets.allowed_datasets(), help='dataset to work with')
parser.add_argument('--mod', type=str, default='mod', choices = Models.allowed_models(), help='Model for use')
parser.add_argument('--no_cuda', action='store_true', help='no gpu usage')
parser.add_argument('--input_type', type=str, default='rgb', help='use rgb for rgbdepth')
# Data augmentation settings
parser.add_argument('--crop_w', type=int, default=960, help='width of image after cropping')
parser.add_argument('--crop_h', type=int, default=540, help='height of image after cropping')

# Paths settings
parser.add_argument('--save_path', type= str, default='../Saved/best', help='save path')
parser.add_argument('--data_path', type=str, required=True, help='path to desired datasets')
parser.add_argument('--out_dir', type=str)

# Cudnn
parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True, default=True, help="cudnn optimization active")
parser.add_argument('--multi', type=str2bool, nargs='?', const=True, default=False, help="use multiple gpus")
parser.add_argument('--normal', type=str2bool, nargs='?', const=True, default=False, help="Normalize input")
parser.add_argument('--max_depth', type=float, default=100.0, help="maximum depth of input")
parser.add_argument('--sparse_val', type=float, default=0.0, help="encode sparse values with 0")
parser.add_argument('--num_samples', default=0, type=int, help='number of samples')


def main():
    global args
    global dataset
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = args.cudnn
    # import pdb; pdb.set_trace()
    best_file_name = glob.glob(os.path.join(args.save_path, 'model_best*'))[0]

    # save_root = os.path.join(os.path.dirname(best_file_name), 'results')
    # if not os.path.isdir(save_root):
    #     os.makedirs(save_root)

    print("==========\nArgs:{}\n==========".format(args))
    # INIT
    print("Init model: '{}'".format(args.mod))
    channels_in = 1 if args.input_type == 'depth' else 4
    model = Models.define_model(mod=args.mod, in_channels=channels_in)
    print("Number of parameters in model {} is {:.3f}M".format(args.mod.upper(), sum(tensor.numel() for tensor in model.parameters())/1e6))
    if not args.no_cuda:
        # Load on gpu before passing params to optimizer
        if not args.multi:
            model = model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    if os.path.isfile(best_file_name):
        print("=> loading checkpoint '{}'".format(best_file_name))
        checkpoint = torch.load(best_file_name)
        model.load_state_dict(checkpoint['state_dict'])
        lowest_loss = checkpoint['loss']
        best_epoch = checkpoint['best epoch']
        print('Lowest RMSE for selection validation set was {:.4f} in epoch {}'.format(lowest_loss, best_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(best_file_name))
        return

    if not args.no_cuda:
        model = model.cuda()
    print("Initializing dataset {}".format(args.dataset))
    # dataset = Datasets.define_dataset(args.dataset, args.data_path, args.input_type)
    # dataset.prepare_dataset()
    dataset = My_Dataset(args.data_path)
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    depth_norm = transforms.Normalize(mean=[14.97/args.max_depth], std=[11.15/args.max_depth])
    model.eval()
    print("===> Start testing")
    total_time = []

    with torch.no_grad():
        # for i, (img, rgb, gt) in tqdm.tqdm(enumerate(zip(dataset.selected_paths['lidar_in'],
        #                                    dataset.selected_paths['img'], dataset.selected_paths['gt']))):
        for i, (img, rgb) in tqdm.tqdm(enumerate(dataset)):
            raw_path = os.path.join(img)

            sparse_depth = np.load(raw_path, allow_pickle=True).item()
            valid_depth_mask_map = sparse_depth["mask"]
            depth_map = np.zeros([640, 960], dtype=np.float64)
            # depth_map = np.zeros([1280, 1920], dtype=np.float64)
            depth_map[valid_depth_mask_map] = sparse_depth["value"]
            # raw_pil = Image.fromarray(depth_map)
            # raw_pil = Image.open(raw_path)
            # gt_path = os.path.join(gt)
            # gt_pil = Image.open(gt)
            # assert raw_pil.size == (960, 640)

            # crop = 640-args.crop_h
            # raw_pil_crop = raw_pil.crop((0, crop, 960, 640))
            # gt_pil_crop = gt_pil.crop((0, crop, 960, 640))

            # raw = depth_read(raw_pil_crop, args.sparse_val)
            depth_map_ori = depth_map.max()
            # raw = np.expand_dims(depth_map/depth_map_ori, axis=2)
            raw = np.expand_dims(depth_map, axis=2)

            raw = to_tensor(raw).float()
            # gt = depth_read(gt_pil_crop, args.sparse_val)
            # gt = to_tensor(gt).float()
            valid_mask = (raw > 0).detach().float()

            input = torch.unsqueeze(raw, 0).cuda()
            # gt = torch.unsqueeze(gt, 0).cuda()

            if args.normal:
                # Put in {0-1} range and then normalize
                input = input/args.max_depth
                # input = depth_norm(input)

            if args.input_type == 'rgb':
                rgb_path = os.path.join(rgb)
                rgb_pil = Image.open(rgb_path)
                # resize to 640 x 960
                rgb_pil = rgb_pil.resize((960, 640))
                # rgb_pil = rgb_pil.resize((1920, 1280))
                # assert rgb_pil.size == (960, 640)
                # rgb_pil_crop = rgb_pil.crop((0, crop, 960, 640))
                # rgb = to_tensor(rgb_pil_crop).float()
                rgb = to_tensor(rgb_pil).float()
                rgb = torch.unsqueeze(rgb, 0).cuda()

                # if not args.normal:
                rgb = rgb*255.0
                input = torch.cat((input, rgb), 1)
                # input = input[:,:,100:-100,:]

            output, _, _, _ = model(input)

            if args.normal:
                output = output*args.max_depth
            output = output[0][0:1].cpu()
            output = torch.clamp(output, min=0)

            dense_depth = raw_path.replace('sparse_depth', 'dense_depth')
            os.makedirs(os.path.dirname(dense_depth), exist_ok=True)
            np.save(dense_depth, output.squeeze(0).numpy())

            # for visulization
            # output_visual = ((output / output.max()) * 255)

            # mi = np.min(depth_map[depth_map>0]) # get minimum positive depth (ignore background)
            # ma = np.max(depth_map)
            # depth_map_normalize = (depth_map-mi)/(ma-mi+1e-8) # normalize to 0~1
            # depth_map_normalize = (255*depth_map_normalize).astype(np.uint8)
            
            # depth_map_color = cv2.applyColorMap(depth_map_normalize.astype(np.uint8), cv2.COLORMAP_JET)
            # depth_on_img = rgb[0].permute(1, 2, 0).cpu().numpy()
            # depth_on_img = depth_on_img[:,:,::-1]
            # depth_on_img[depth_map > 0] = depth_map_color[depth_map > 0]
            # cv2.imwrite("%010d_depth_on_img.png"% i, depth_on_img) 

            # output_depth_map_color = cv2.applyColorMap(output_visual[0].cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
            # cv2.imwrite("%010d_output_depth_map_color.png"% i, output_depth_map_color)

            # breakpoint()

    print('num imgs: ', i + 1)


if __name__ == '__main__':
    main()
