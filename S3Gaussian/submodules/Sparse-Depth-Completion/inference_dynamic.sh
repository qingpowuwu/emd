dynamic_scene_no=(
    16
    21
    22
    25
    31
    34
    35
    49
    53
    80
    84
    86
    89
    94
    96
    102
    111
    222
    323
    382
    402
    427
    438
    546
    581
    592
    620
    640
    700
    754
    795
    796
)


for scene_no in ${dynamic_scene_no[@]}; do
    echo "Processing scene: $scene_no"
    CUDA_VISIBLE_DEVICES=0 python Test/test.py --save_path "Saved"  --data_path "../../data/processed/dynamic32/training/$(printf "%03d" $scene_no)"
done
