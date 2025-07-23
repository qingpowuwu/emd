# EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting

## News
- **[2025/7/23]** EMD Code on OmniRe and S3Gaussian release.
- **[2025/6/26]** EMD is accepted by ICCV2025! Code will be released soon.

#### [Project Page](https://qingpowuwu.github.io/emdgaussian.github.io/) | [**arXiv Paper**](https://arxiv.org/abs/2411.15582)

## Overview

![overview](./static/images/2-pipeline.png)

Photorealistic reconstruction of street scenes is essential for developing real-world simulators in autonomous driving. While recent methods based on 3D/4D Gaussian Splatting (GS) have demonstrated promising results, they still encounter challenges in complex street scenes due to the unpredictable motion of dynamic objects. Current methods typically decompose street scenes into static and dynamic objects , learning the Gaussians in either a supervised manner (e.g., w/ 3D bounding-box) or a self-supervised manner (e.g., w/o 3D bounding-box). However, these approaches do not effectively model the motions of dynamic objects (e.g., the motion speed of pedestrians is clearly different from that of vehicles), resulting in suboptimal scene decomposition. To address this, we propose Explicit Motion Decomposition (EMD), which models the motions of dynamic objects by introducing learnable motion embeddings to the Gaussians, enhancing the decomposition in street scenes. The proposed EMD is a plug-and-play approach applicable to various baseline methods. We also propose tailored training strategies to apply EMD to both supervised and self-supervised baselines. Through comprehensive experimentation, we illustrate the effectiveness of our approach with various established baselines.

## EMD + OmniRe

Please follow the [OmniRe installation guide](https://github.com/ziyc/drivestudio?tab=readme-ov-file#-installation) to set up the conda environment and prepare the required datasets. You will also need to install the third-party `smplx` repository and download the `smpl_models` into the correct directory. 

To start training, run the following commands:

```bash
cd OmniRe
bash train.sh <GPU_ID> <SCENE_ID>
```

- `<GPU_ID>`: Specifies the index of the GPU to use.
- `<SCENE_ID>`: Selects the Waymo scene for training. Available options are: `23`, `114`, `327`, `621`, `703`, `172`, `552`, `788`.

The main modifications in EMD are made to `models/nodes/smpl.py` and `models/nodes/rigid.py`, with corresponding parameters added to the configuration files.

## EMD + S3Gaussian

Please follow the [S3Gaussian installation guide](https://github.com/nnanhuang/S3Gaussian?tab=readme-ov-file#environmental-setups) to set up the conda environment and prepare the required datasets (32 dynamic scenes from Waymo).

To enhance the geometry in S3Gaussian, we densify the sparse depth maps obtained from LiDAR, generating dense depth maps for improved scene representation. Follow [S-NeRF](https://github.com/fudan-zvg/S-NeRF), we use Sparse-Depth-Completion to convert sparse depth map into dense depth maps. Please refer to the "submodules/Sparse-Depth-Completion/inference_dynamic.sh" for more details. 

We also provide a sample [scene 016](https://drive.google.com/file/d/1BXEJXrUFyV6mg8nQDstFABDcz70sco4Y/view?usp=sharing) in Waymo dynamic subset, which includes dense depth maps. Please download it from Google Cloud and unzip it into the correct folder. 

Next, we replace the default rasterizer with one capable of rendering depth maps, using [diff_gauss](https://github.com/slothfulxtx/diff-gaussian-rasterization). Please refer to the installation instructions in the diff_gauss repository.

Finally, to start training, run the following commands:

```bash
cd S3Gaussian
# For novel view synthesis with a stride of 10
bash scripts/dynamic/run_dynamic_nvs.sh <GPU_ID> <SCENE_ID>

# For reconstruction using all available views
bash scripts/dynamic/run_dynamic_recon.sh <GPU_ID> <SCENE_ID>
```

- `<GPU_ID>`: The index of the GPU to use.
- `<SCENE_ID>`: The Waymo dynamic scene ID for training. Available options are:  
  `16, 21, 22, 25, 31, 34, 35, 49, 53, 80, 84, 86, 89, 94, 96, 102, 111, 222, 323, 382, 402, 427, 438, 546, 581, 592, 620, 640, 700, 754, 795, 796`


## Citation

If you find our work useful for your research, please consider citing:

```bibtex
@inproceedings{wei2025emd,
  title={EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting},
  author  = {Wei, Xiaobao and Wuwu, Qingpo and Zhao, Zhongyu and Wu, Zhuangzhe and Huang, Nan and Lu, Ming and Ma, Ningning and Zhang, Shanghang},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  year={2025}
}
