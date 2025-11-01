### point cloud auto-encoder, with skeleton regularization
import torch
import torch.nn as nn
from flemme.encoder import PointNet2Encoder, PointTrans2Encoder, \
            PointMamba2Encoder
from .skeleton import SkeletonSDFDecoder
from flemme.logger import get_logger

### skeleton-regularized point cloud auto-encoder
logger = get_logger("skeleton_encoder")

class SparsePointCNNEncoder(PointNet2Encoder):
    def __init__(self, point_dim = 3,
                 projection_channel = 64,
                 num_fps_points = [1024, 512, 256, 64],
                 num_neighbors_k = 32,
                 neighbor_radius = [0.1, 0.2, 0.4, 0.8], 
                 fps_feature_channels = [128, 256, 512, 1024], 
                 num_blocks = 2,
                 num_scales = 2,
                 use_xyz = True,
                 sorted_query = False,
                 knn_query = False,
                 dense_channels = [1024],
                 building_block = 'dense', 
                 normalization = 'group', num_norm_groups = 8, 
                 activation = 'lrelu', dropout = 0., 
                 final_concat = False,
                 pos_embedding = False,
                 channel_attention = None,
                 voxel_resolutions = [],
                 voxel_conv_kernel_size = 3,
                 with_se = False,
                 coordinate_normalize = True,
                 condition_channel = 0,
                 condition_injection = 'gate_bias',
                 standardize_latents = True,
                 **kwargs):
        super().__init__(point_dim=point_dim, 
                projection_channel = projection_channel,
                num_fps_points = num_fps_points,
                num_neighbors_k=num_neighbors_k,
                neighbor_radius = neighbor_radius,
                fps_feature_channels = fps_feature_channels,
                num_blocks = num_blocks,
                num_scales = num_scales,
                use_xyz = use_xyz,
                sorted_query = sorted_query,
                knn_query = knn_query,
                dense_channels = dense_channels,
                building_block = building_block,
                activation = activation, 
                dropout = dropout,
                normalization = normalization, 
                num_norm_groups = num_norm_groups,  
                vector_embedding = False, 
                is_point2decoder = False,
                return_xyz = True, 
                final_concat = final_concat,
                pos_embedding=pos_embedding,
                channel_attention = channel_attention,
                voxel_resolutions = voxel_resolutions,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize,
                condition_channel = condition_channel,
                condition_injection = condition_injection)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.num_latent_points = self.num_fps_points[-1]
        self.standardize_latents = standardize_latents
        if self.standardize_latents:
            ## without activation
            self.normalize = nn.Sequential(nn.Linear(dense_channels[-1], dense_channels[-1]),
                        nn.LayerNorm((self.num_latent_points, dense_channels[-1]), 
                        elementwise_affine = False, bias = False))
        self.out_channel += 3
    def forward(self, x, c = None):
        x, sp = super().forward(x, c = c)
        ### without activation
        if self.standardize_latents:
            x = self.normalize(x)
        return torch.cat((sp, x), dim = -1)

class SparsePointTransEncoder(PointTrans2Encoder):
    def __init__(self, point_dim = 3,
                projection_channel = 64,
                num_fps_points = [1024, 512, 256, 64],
                num_neighbors_k = 32,
                neighbor_radius = [0.1, 0.2, 0.4, 0.8], 
                fps_feature_channels = [128, 256, 512, 1024], 
                num_blocks = 2,
                num_scales = 2,
                use_xyz = True,
                sorted_query = False,
                knn_query = False,
                dense_channels = [1024],
                building_block = 'dense', 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                num_heads = 4, d_k = None, 
                qkv_bias = True, qk_scale = None, 
                atten_dropout = None, 
                mlp_hidden_ratios=[4.0, 4.0], 
                long_range_modeling = False,
                final_concat = False,
                pos_embedding = False,
                channel_attention = None,
                voxel_resolutions = [],
                voxel_conv_kernel_size = 3,
                with_se = False,
                coordinate_normalize = True,
                condition_channel = 0,
                condition_injection = 'gate_bias',
                standardize_latents = True,
                **kwargs):
        super().__init__(point_dim=point_dim, 
                projection_channel = projection_channel,
                num_fps_points = num_fps_points,
                num_neighbors_k=num_neighbors_k,
                neighbor_radius = neighbor_radius,
                fps_feature_channels = fps_feature_channels,
                num_blocks = num_blocks,
                num_scales = num_scales,
                use_xyz = use_xyz,
                sorted_query = sorted_query,
                knn_query = knn_query,
                dense_channels = dense_channels,
                building_block = building_block,
                num_heads = num_heads,
                d_k = d_k, 
                qkv_bias = qkv_bias, 
                qk_scale = qk_scale, 
                atten_dropout = atten_dropout, 
                mlp_hidden_ratios=mlp_hidden_ratios, 
                long_range_modeling = long_range_modeling,
                activation = activation, 
                dropout = dropout,
                normalization = normalization, 
                num_norm_groups = num_norm_groups,  
                vector_embedding = False, 
                is_point2decoder = False,
                return_xyz = True,
                final_concat = final_concat,
                pos_embedding= pos_embedding,
                channel_attention = channel_attention,
                voxel_resolutions=voxel_resolutions,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize,
                condition_channel = condition_channel,
                condition_injection = condition_injection)        
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.num_latent_points = self.num_fps_points[-1]
        self.standardize_latents = standardize_latents
        if self.standardize_latents:
            ## without activation
            self.normalize = nn.Sequential(nn.Linear(dense_channels[-1], dense_channels[-1]),
                        nn.LayerNorm((self.num_latent_points, dense_channels[-1]), 
                        elementwise_affine = False, bias = False))
        self.out_channel += 3
    def forward(self, x, c = None):
        x, sp = super().forward(x, c = c)
        ### without activation
        if self.standardize_latents:
            x = self.normalize(x)
        return torch.cat((sp, x), dim = -1)

            

class SparsePointMambaEncoder(PointMamba2Encoder):
    def __init__(self, point_dim = 3,
                projection_channel = 64,
                num_fps_points = [1024, 512, 256, 64],
                num_neighbors_k = 32,
                neighbor_radius = [0.1, 0.2, 0.4, 0.8], 
                fps_feature_channels = [128, 256, 512, 1024], 
                num_blocks = 2,
                num_scales = 2,
                use_xyz = True,
                sorted_query = False,
                knn_query = False,
                dense_channels = [1024],
                building_block = 'dense', 
                scan_strategies = None,
                flip_scan = False,
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                state_channel = 64, 
                conv_kernel_size = 4, inner_factor = 2.0,  
                head_channel = 64,
                conv_bias=True, bias=False,
                learnable_init_states = True, chunk_size=256,
                dt_min=0.001, A_init_range=(1, 16),
                dt_max=0.1, dt_init_floor=1e-4, 
                dt_rank = None, dt_scale = 1.0,
                mlp_hidden_ratios=[4.0, 4.0], 
                long_range_modeling = False,
                final_concat = False,
                pos_embedding = False,
                channel_attention = None,
                voxel_resolutions = [],
                voxel_conv_kernel_size = 3,
                with_se = False,
                coordinate_normalize = True,
                condition_channel = 0,
                condition_injection = 'gate_bias',
                standardize_latents = True,
                **kwargs):
        super().__init__(point_dim=point_dim, 
                projection_channel = projection_channel,
                num_fps_points = num_fps_points,
                num_neighbors_k=num_neighbors_k,
                neighbor_radius = neighbor_radius,
                fps_feature_channels = fps_feature_channels,
                num_blocks = num_blocks,
                num_scales = num_scales,
                use_xyz = use_xyz,
                sorted_query = sorted_query,
                knn_query = knn_query,
                dense_channels = dense_channels,
                building_block = building_block,
                activation = activation, 
                dropout = dropout,
                normalization = normalization, 
                num_norm_groups = num_norm_groups,  
                vector_embedding = False, 
                is_point2decoder = False,
                final_concat = final_concat,
                pos_embedding=pos_embedding,
                return_xyz = True,
                channel_attention = channel_attention,
                voxel_resolutions=voxel_resolutions,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                state_channel = state_channel, 
                conv_kernel_size = conv_kernel_size, 
                inner_factor = inner_factor,  
                head_channel = head_channel,
                conv_bias=conv_bias, bias=bias,
                learnable_init_states = learnable_init_states, 
                chunk_size=chunk_size,
                dt_min=dt_min, A_init_range=A_init_range,
                dt_max=dt_max, dt_init_floor=dt_init_floor, 
                dt_rank = dt_rank, dt_scale = dt_scale,
                mlp_hidden_ratios=mlp_hidden_ratios, 
                scan_strategies = scan_strategies,
                flip_scan = flip_scan,
                long_range_modeling = long_range_modeling,
                coordinate_normalize = coordinate_normalize,
                condition_channel = condition_channel,
                condition_injection = condition_injection)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.num_latent_points = self.num_fps_points[-1]
        self.standardize_latents = standardize_latents
        if self.standardize_latents:
            ## without activation
            self.normalize = nn.Sequential(nn.Linear(dense_channels[-1], dense_channels[-1]),
                        nn.LayerNorm((self.num_latent_points, dense_channels[-1]), 
                        elementwise_affine = False, bias = False))
        self.out_channel += 3
    def forward(self, x, c = None):
        x, sp = super().forward(x, c = c)
        ### without activation
        if self.standardize_latents:
            x = self.normalize(x)
        return torch.cat((sp, x), dim = -1)



class SparsePointSDFDecoder(SkeletonSDFDecoder):
    def __init__(self, point_dim=1,
                ### embedding  
                projection_channel = 256, 
                latent_channel = 256,
                latent_injection = 'cross_atten',
                num_latent_points = 256,
                num_blocks = 2,
                building_block = 'dense', seq_feature_channels = [], 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                standardize_latents = True,
                **kwargs):
        super().__init__(point_dim = point_dim,
            latent_channel = latent_channel,
            projection_channel=projection_channel,
            latent_injection = latent_injection,
            num_blocks = num_blocks,
            building_block = building_block,
            seq_feature_channels = seq_feature_channels,
            normalization = normalization,
            num_norm_groups = num_norm_groups,
            activation = activation,
            dropout = dropout,
            num_latent_points = num_latent_points,
            condition_channel = condition_channel,
            condition_injection = condition_injection,
            standardize_latents = standardize_latents,
            with_radius = False,
            )