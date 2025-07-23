export PYTHONPATH=$(pwd)

in_scene_no=$2
timestamp=$(date +"%Y%m%d_%H%M%S")
output_root="outputs"
project="waymo_examples"
expname="scene_${in_scene_no}-${timestamp}"
scene_idx=${in_scene_no}

declare -A start_timesteps
declare -A end_timesteps

start_timesteps=(
    [23]=0
    [114]=0
    [327]=0
    [621]=0
    [703]=0
    [172]=30
    [552]=0
    [788]=130
)

end_timesteps=(
    [23]=150
    [114]=150
    [327]=150
    [621]=140
    [703]=170
    [172]=180
    [552]=140
    [788]=-1
)

start_timestep=${start_timesteps[$scene_idx]}
end_timestep=${end_timesteps[$scene_idx]}

export CUDA_VISIBLE_DEVICES=$1
python tools/train.py \
    --config_file configs/paper_legacy/omnire.yaml \
    --output_root ${output_root} \
    --project ${project} \
    --run_name ${expname} \
    dataset=waymo/3cams_examples \
    data.scene_idx=${scene_idx} \
    data.start_timestep=${start_timestep} \
    data.end_timestep=${end_timestep}
