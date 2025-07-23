import argparse

def auto_argparse_from_class(cls_instance):
    parser = argparse.ArgumentParser(description="Auto argparse from class")
    
    for attribute, value in vars(cls_instance).items():
        if isinstance(value, bool):
            parser.add_argument(f'--{attribute}', action='store_true' if not value else 'store_false',
                                help=f"Flag for {attribute}, default is {value}")
        elif isinstance(value, list):
            parser.add_argument(f'--{attribute}', type=type(value[0]), nargs='+', default=value,
                                help=f"List for {attribute}, default is {value}")
        else:
            parser.add_argument(f'--{attribute}', type=type(value), default=value,
                                help=f"Argument for {attribute}, default is {value}")

    return parser

class BaseOptions(object):
    def __init__(self) -> None:

        super().__init__()

        self.debug_test = False
        self.debug = False
        self.debug_from = -1
        self.sh_degree = 3
        self.source_path = ""
        self.model_path = ""
        self.images = "images"
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.eval_only = False
        self.load_checkpoint = ""

        # Dataset
        self.stride = 0
        # visual
        self.render_process=True
        self.start_time = 0 # now hard-coded
        self.end_time = 19
        self.original_start_time = 0 # now hard-coded
        self.num_pts = 1500000
        self.max_num_pts = 2000000
        # mask loading options
        self.load_sky_mask = False
        self.load_dynamic_mask = True
        self.load_feat_map = True
        # waymo
        self.load_intrinsic = False
        self.load_c2w = False
        self.load_dense_depth = False
        self.is_negative_time = False
        # occ grid
        self.save_occ_grid = True
        # self.occ_voxel_size = 2.0
        self.occ_voxel_size = 0.5
        self.recompute_occ_grid = True 

        # renderer
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        
        # train
        self.coarse_iterations = 5000
        self.iterations = 50_000
        self.checkpoint_iterations = [55000]
        self.visualize_iterations = 1000

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = self.iterations + self.coarse_iterations

        self.deformation_lr_init = 0.000016
        self.deformation_lr_final = 0.0000016
        self.deformation_lr_delay_mult = 0.01
        self.deformation_lr_max_steps = self.iterations + self.coarse_iterations
        self.offsets_lr = 0.00002
        self.grid_lr_init = 0.00016
        self.grid_lr_final = 0.000016

        self.feature_lr = 0.0025
        self.feature_lr_div_factor = 20.0
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001

        self.sky_cube_map_lr_init = 0.01
        self.sky_cube_map_lr_final = 0.0001
        self.sky_cube_map_max_steps = self.iterations + self.coarse_iterations
        
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.5
        self.lambda_reg = 1.0
        self.lambda_dx = 0.001
        self.lambda_ds = 0.001
        self.lambda_dr = 0.001
        self.lambda_do = 0.001
        self.lambda_dshs = 0.001
        self.lambda_f2c = 0.0
        self.lambda_feat = 0.001
        self.lambda_sky = 0.05
        self.lambda_opacity_sparse=0.0
        self.plane_tv_weight = 0.0001
        self.time_smoothness_weight = 0.0
        self.l1_time_planes = 0.0001
        # densify
        self.percent_dense = 0.01
        self.densification_interval = 100   # 100
        self.opacity_reset_interval = 3000
        self.pruning_interval = 100
        self.pruning_from_iter = 500
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002

        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        self.reset_opacity_ratio = 0.

        # gaussian embedding
        self.net_width = 64
        self.defor_depth = 1
        self.timebase_pe = 4
        self.posebase_pe = 10
        self.scale_rotation_pe = 2
        self.opacity_pe = 2
        self.timenet_width = 64
        self.timenet_output = 32
        self.bounds = 1.6
        self.grid_pe=0
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]
                            }

        self.hash_n_input_dims = 4
        self.hash_n_levels = 10
        self.hash_n_features_per_level = 4
        self.hash_base_resolution = 32
        self.hash_max_resolution = 8192
        self.hash_log2_hashmap_size = 18
        self.aggregate_feature = False
        self.aggregate_time_warp = True
        self.aggregate_space_warp = True
        self.is_use_hash = False

        self.multires = [1, 2, 4, 8]
        self.empty_voxel = False
        self.static_mlp = False

        # sky model
        self.sky_resolution = 1024
        self.sky_white_background = True
        self.feat_head=True
        self.min_embeddings = 30
        self.max_embeddings = 150
        self.total_num_frames = 300
        self.temporal_embedding_dim=32
        self.gaussian_embedding_dim=4
        self.c2f_temporal_iter = 20000 + self.coarse_iterations
        self.zero_temporal = False
        self.deform_from_iter = self.coarse_iterations
        self.use_anneal = True
        self.use_coarse_temporal_embedding = False
        self.no_c2f_temporal_embedding = False
        self.no_coarse_deform = False
        self.no_fine_deform = False
        self.no_temporal_embedding_dim=False
        self.no_gaussian_embedding_dim=False
        self.no_fine_hexplane_features=False
        self.no_coarse_hexplane_features=False
        self.no_time_offset=False
        self.no_dx=False
        self.apply_coarse_dx = True
        self.apply_final_dx = True
        self.no_grid=False
        self.no_ds=False 
        self.no_dr=False
        self.no_do=False
        self.no_dshs=False
        self.direct_add_dx=True
        self.direct_add_ds=True
        self.direct_add_dr=True
        self.direct_add_do=True
        self.direct_add_dshs=True
        self.combine_dynamic_static=False
        self.freeze_static = False

