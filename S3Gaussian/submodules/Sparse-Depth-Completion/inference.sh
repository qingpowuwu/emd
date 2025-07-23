scene_no=3
CUDA_VISIBLE_DEVICES=1 python Test/test.py --save_path "Saved"  --data_path "/lustre/xiaobao.wei/gaussianSim/s3gaussianpp/processed/static32/training/$(printf "%03d" $scene_no)" --out_dir ./"output"
