#!/bin/bash

scene_no=$(printf "%03d" $2)
echo "scene_no: $scene_no"
data_dir="./data/processed/dynamic32/training/$scene_no"

# Get the current date and time in the desired format
DATE=$(date +"%Y_%m_%d_%H_%M_%S")
project=recon
output_root="./work_dirs/dynamic/$scene_no"_"$project"  "$DATE"

# 获取子目录的basename
model_name=$(basename "$data_dir")

# 使用basename来修改model_path
model_path="$output_root"

# 执行相同的命令，只修改-s和--model_path参数
CUDA_VISIBLE_DEVICES=$1 python train.py \
    --source_path "$data_dir" \
    --model_path "$model_path" \
    --end_time 49 \
    --load_dense_depth \
    --load_sky_mask \
    --load_intrinsic \
    --no_ds \
    --no_dr \
    --no_fine_hexplane_features \