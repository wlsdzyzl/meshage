### point cloud auto-encoder, with skeleton regularization
import torch
import torch.nn as nn
from flemme.block import QueryAndGroup, MultipleBuildingBlocks, \
                    GroupSeqModelingLayer, channel_transfer, channel_recover
from flemme.encoder import PointNetEncoder, PointNet2Encoder, \
            PointTransEncoder, PointTrans2Encoder, \
            PointMambaEncoder, PointMamba2Encoder
from meshage.sknet import SkeletonNet
from flemme.logger import get_logger
from functools import partial
from knn_cuda import KNN
### skeleton-regularized point cloud auto-encoder
logger = get_logger("sksp_encoder")

class SKSPCNNEncoder(PointNet2Encoder):
    def __init__(self, point_dim = 3,
                 point_num = 2560,
                 projection_channel = 64,
                 num_fps_points = [1024, 512, 256, 64],
                 num_neighbors_k = 32,
                 neighbor_radius = [0.1, 0.2, 0.4, 0.8], 
                 fps_feature_channels = [128, 256, 512, 1024], 
                 num_blocks = 2,
                 num_scales = 2,
                 use_xyz = True,
                 sorted_query = False,
                 knn_query = 'feature',
                 dense_channels = [1024],
                 building_block = 'dense', 
                 normalization = 'group', num_norm_groups = 8, 
                 activation = 'lrelu', dropout = 0., 
                 final_concat = False,
                 pos_embedding = False,
                 channel_attention = None,
                 voxel_resolutions = [],
                 voxel_attens = [],
                 voxel_conv_kernel_size = 3,
                 with_se = False,
                 coordinate_normalize = True,
                 condition_channel = 0,
                 condition_injection = 'gate_bias',
                 local_feature_channels = [64, 64, 128, 256], 
                 num_neighbors_k_self = 0,
                 num_neighbors_k_cross = 32,
                 skeleton_net_config = {},
                 num_skeleton_points = 256,
                 with_radius = False,
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
                voxel_attens = voxel_attens,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize,
                condition_channel = condition_channel,
                condition_injection = condition_injection)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.pos_embed_channel = 3
        if pos_embedding:
            self.pos_embed_channel = projection_channel
        self.sk_layer = PointNetEncoder(point_dim=point_dim + with_radius, 
                time_channel = 0,
                time_injection = 'gate_bias',
                projection_channel = projection_channel,
                local_feature_channels = local_feature_channels, 
                num_blocks = num_blocks,
                num_neighbors_k = num_neighbors_k_self,
                building_block = building_block,
                dense_channels = dense_channels,
                normalization = normalization,
                num_norm_groups = num_norm_groups,
                activation = activation, dropout = dropout, 
                vector_embedding = False, 
                channel_attention = channel_attention,
                voxel_resolutions=voxel_resolutions,
                voxel_attens=voxel_attens,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize,
                condition_channel = condition_channel,
                condition_injection = condition_injection,
                condition_first = False)
        is_seq = self.BuildingBlock.func.is_sequence_modeling()
        if is_seq:
            self.sk_layer.meshage = nn.ModuleList([MultipleBuildingBlocks(in_channel = lfc + self.pos_embed_channel,
                                out_channel = lfc,
                                n = num_blocks,
                                BuildingBlock = partial(
                                    GroupSeqModelingLayer,
                                    BuildingBlock = self.BuildingBlock),)
                            for lfc in local_feature_channels ])
        else:
            self.sk_layer.meshage = nn.ModuleList([MultipleBuildingBlocks(in_channel = lfc + self.pos_embed_channel,
                                out_channel = lfc,
                                n = num_blocks,
                                BuildingBlock = self.BuildingBlock) 
                            for lfc in local_feature_channels ])
        assert self.out_channel == self.sk_layer.out_channel, \
            "skeleton latents should have an identical number of channels with sparse-point latents."
        self.point_num = point_num
        if not skeleton_net_config is None:
            logger.info('Using online skeletonization.')
            skeleton_net_config['point_num'] = point_num
            self.skeletonize = SkeletonNet(skeleton_net_config)
            self.num_skeleton_points = self.skeletonize.skp_num
        else:
            logger.info('Pre-computed skeleton is needed.')
            self.skeletonize = None
            self.num_skeleton_points = num_skeleton_points
            if with_radius:
                self.knn = KNN(k = 4, transpose_mode=True)
        ## 4 dim: xyzr
        self.query_and_group = QueryAndGroup(num_neighbors_k_cross, knn_query=knn_query)
        self.out_channel += (3 + with_radius)
        self.with_radius = with_radius
        self.standardize_latents = standardize_latents
        self.num_latent_points = self.num_fps_points[-1] + self.num_skeleton_points

        self.standardize_latents = standardize_latents
        if self.standardize_latents:
            ## without activation
            self.normalize = nn.Sequential(nn.Linear(dense_channels[-1], dense_channels[-1]),
                        nn.LayerNorm((self.num_latent_points, dense_channels[-1]), 
                        elementwise_affine = False, bias = False))

    def group_surface_to_skeleton(self, layer_id, pos, pos_emb, sk, sk_emb, sff, skf, c):
        sff_trans = channel_recover(sff)
        skf_trans = channel_recover(skf)
        grouped_sff = channel_transfer(self.query_and_group(pos, pos_emb, sk, sk_emb, sff_trans, skf_trans))
        ## intergrate skf into grouped surface feature
        sk_emb_feature = torch.cat((sk_emb - sk_emb, skf), dim = -1)
        grouped_sff = torch.cat((grouped_sff, sk_emb_feature.unsqueeze(2)), dim = 2)
        grouped_sff = self.sk_layer.meshage[layer_id](grouped_sff, c = c)
        ### (B, N, K, C) -> (B, N, C), N: num of skeletal points
        skf = grouped_sff.max(dim = 2)[0]
        return skf
    ## extract skeleton feature
    def extract_skf(self, x, c = None, ske = None):
        sf_res, sk_res = [], []

        pos = x[..., :3]
        if self.skeletonize:
            if not ske is None:
                logger.warning('This model extract skeleton online, the input skeleton will be ignored.')
            ske_res = self.skeletonize(pos)
            sk, r = ske_res['recon_skeleton'], ske_res['radius']
        else:
            assert not ske is None, 'Precomputed skeleton is needed.'
            sk = ske
            if ske.shape[-1] == 4:
                sk, r = ske[...,0:3], ske[..., 3:4]
            elif self.with_radius:
                dist, _ = self.knn(pos, sk)
                r = dist.mean(dim = -1, keepdim = True)
        assert sk.shape[1] == self.num_skeleton_points, \
            'The number of points in the input skeleton does not match the model.'
        pos_emb, sk_emb = pos, sk
        
        if hasattr(self, 'pos_embed'):
            pos_emb = self.pos_embed(pos_emb)
            sk_emb = self.pos_embed(sk_emb)
        
        sff = self.point_proj(x)
        if self.with_radius:
            skf = self.sk_layer.point_proj(torch.cat((sk, r), dim = -1))
        else:
            skf = self.sk_layer.point_proj(sk)
        ## create shared lf, vlf and ca for skeleton and surface?
        for lid, lf in enumerate(self.sk_layer.lf[:-1]):
            vsff, vskf = sff, skf
            sff, skf = lf(sff, c = c), lf(skf, c = c)

            if hasattr(self.sk_layer, 'vlf'):
                sff = sff + self.sk_layer.vlf[lid](vsff, pos, c = c)
                skf = skf + self.sk_layer.vlf[lid](vskf, sk, c = c)
            if hasattr(self.sk_layer, 'ca'):
                sff = self.sk_layer.ca[lid](sff)
                skf = self.sk_layer.ca[lid](sff)
            ## group skeleton feature from neighbor surface points
            skf = self.group_surface_to_skeleton(lid, pos, pos_emb, sk, sk_emb, sff, skf, c)
            sf_res.append(sff)
            sk_res.append(skf)
            
        vsff = torch.concat(sf_res, dim = -1)
        vskf = torch.concat(sk_res, dim = -1)
        sff, skf = self.sk_layer.lf[-1](vsff, c = c), self.sk_layer.lf[-1](vskf, c = c)
        # print(sk.mean(), r.mean(), pos_emb.mean(), sk_emb.mean(), sff.mean(), skf.mean())
        if hasattr(self.sk_layer, 'vlf'):
            sff = sff + self.sk_layer.vlf[-1](vsff, pos, c = c)
            skf = skf + self.sk_layer.vlf[-1](vskf, sk, c = c)
        if hasattr(self.sk_layer, 'ca'):
            sff = self.sk_layer.ca[-1](sff)
            skf = self.sk_layer.ca[-1](skf)
        skf = self.group_surface_to_skeleton(-1, pos, pos_emb, sk, sk_emb, sff, skf, c)
        
        ### concate global embedding and local point feautre
        x1 = skf.max(dim = 1, keepdim = True)[0]
        x2 = skf.mean(dim = 1, keepdim = True)
        x = torch.concat((x1, x2), dim = -1)
        ## vector embedding should always be false
        x = x.repeat(1, skf.shape[1], 1)
        x = torch.concat([x, skf], dim=-1)

        ## compute embedding vectors
        x = self.sk_layer.dense(x, c = c)
        return x, sk, r
    def forward(self, x, c = None, ske = None):
        spx, sp = super().forward(x, c = c)
        skx, sk, r = self.extract_skf(x, c = c, ske = ske)
        x = torch.cat((spx, skx), dim = 1)
        pk = torch.cat((sp, sk), dim = 1)

        ### without activation
        if self.standardize_latents:
            x = self.normalize(x)
        if self.with_radius:
            r = torch.cat((torch.zeros((spx.shape[0], spx.shape[1], 1),
                                       dtype = spx.dtype, device = spx.device), r), 
                                       dim = 1)
            return torch.cat((pk, r, x), dim = -1)
        else:
            return torch.cat((pk, x), dim = -1)

class SKSPTransEncoder(PointTrans2Encoder):
    def __init__(self, point_dim = 3,
                point_num = 2560,
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
                voxel_attens = [],
                voxel_conv_kernel_size = 3,
                with_se = False,
                coordinate_normalize = True,
                condition_channel = 0,
                condition_injection = 'gate_bias',
                local_feature_channels = [64, 64, 128, 256], 
                num_neighbors_k_self = 0,
                num_neighbors_k_cross = 32,
                skeleton_net_config = {},
                num_skeleton_points = 256,
                with_radius = False,
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
                voxel_attens=voxel_attens,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize,
                condition_channel = condition_channel,
                condition_injection = condition_injection)        
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.pos_embed_channel = 3
        if pos_embedding:
            self.pos_embed_channel = projection_channel
        self.sk_layer = PointTransEncoder(point_dim=point_dim + with_radius, 
                time_channel = 0,
                time_injection = 'gate_bias',
                projection_channel = projection_channel,
                local_feature_channels = local_feature_channels, 
                num_blocks = num_blocks,
                num_neighbors_k = num_neighbors_k_self,
                building_block = building_block,
                dense_channels = dense_channels,
                normalization = normalization,
                num_norm_groups = num_norm_groups,
                activation = activation, dropout = dropout, 
                num_heads = num_heads, d_k = d_k, 
                qkv_bias = qkv_bias, 
                qk_scale = qk_scale, 
                atten_dropout = atten_dropout, 
                mlp_hidden_ratios=mlp_hidden_ratios, 
                vector_embedding = False, 
                channel_attention = channel_attention,
                voxel_resolutions=voxel_resolutions,
                voxel_attens=voxel_attens,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize,
                condition_channel = condition_channel,
                condition_injection = condition_injection,
                condition_first = False)
        is_seq = self.BuildingBlock.func.is_sequence_modeling()
        if is_seq:
            self.sk_layer.meshage = nn.ModuleList([MultipleBuildingBlocks(in_channel = lfc + self.pos_embed_channel,
                                out_channel = lfc,
                                n = num_blocks,
                                BuildingBlock = partial(
                                    GroupSeqModelingLayer,
                                    BuildingBlock = self.BuildingBlock),)
                            for lfc in local_feature_channels ])
        else:
            self.sk_layer.meshage = nn.ModuleList([MultipleBuildingBlocks(in_channel = lfc + self.pos_embed_channel,
                                out_channel = lfc,
                                n = num_blocks,
                                BuildingBlock = self.BuildingBlock) 
                            for lfc in local_feature_channels ])
        assert self.out_channel == self.sk_layer.out_channel, \
            "skeleton latents should have an identical number of channels with sparse-point latents."
        self.point_num = point_num
        if not skeleton_net_config is None:
            logger.info('Using online skeletonization.')
            skeleton_net_config['point_num'] = point_num
            self.skeletonize = SkeletonNet(skeleton_net_config)
            self.num_skeleton_points = self.skeletonize.skp_num
        else:
            logger.info('Pre-computed skeleton is needed.')
            self.skeletonize = None
            self.num_skeleton_points = num_skeleton_points
            if with_radius:
                self.knn = KNN(k = 4, transpose_mode=True)
        ## 4 dim: xyzr
        self.query_and_group = QueryAndGroup(num_neighbors_k_cross, knn_query=knn_query)
        self.out_channel += (3 + with_radius)
        self.with_radius = with_radius
        self.standardize_latents = standardize_latents
        self.num_latent_points = self.num_fps_points[-1] + self.num_skeleton_points

        self.standardize_latents = standardize_latents
        if self.standardize_latents:
            ## without activation
            self.normalize = nn.Sequential(nn.Linear(dense_channels[-1], dense_channels[-1]),
                        nn.LayerNorm((self.num_latent_points, dense_channels[-1]), 
                        elementwise_affine = False, bias = False))

    def group_surface_to_skeleton(self, layer_id, pos, pos_emb, sk, sk_emb, sff, skf, c):
        sff_trans = channel_recover(sff)
        skf_trans = channel_recover(skf)
        grouped_sff = channel_transfer(self.query_and_group(pos, pos_emb, sk, sk_emb, sff_trans, skf_trans))
        ## intergrate skf into grouped surface feature
        sk_emb_feature = torch.cat((sk_emb - sk_emb, skf), dim = -1)
        grouped_sff = torch.cat((grouped_sff, sk_emb_feature.unsqueeze(2)), dim = 2)
        grouped_sff = self.sk_layer.meshage[layer_id](grouped_sff, c = c)
        ### (B, N, K, C) -> (B, N, C), N: num of skeletal points
        skf = grouped_sff.max(dim = 2)[0]
        return skf
    ## extract skeleton feature
    def extract_skf(self, x, c = None, ske = None):
        sf_res, sk_res = [], []

        pos = x[..., :3]
        if self.skeletonize:
            if not ske is None:
                logger.warning('This model extract skeleton online, the input skeleton will be ignored.')
            ske_res = self.skeletonize(pos)
            sk, r = ske_res['recon_skeleton'], ske_res['radius']
        else:
            assert not ske is None, 'Precomputed skeleton is needed.'
            sk = ske
            if ske.shape[-1] == 4:
                sk, r = ske[...,0:3], ske[..., 3:4]
            elif self.with_radius:
                dist, _ = self.knn(pos, sk)
                r = dist.mean(dim = -1, keepdim = True)
        assert sk.shape[1] == self.num_skeleton_points, \
            'The number of points in the input skeleton does not match the model.'
        pos_emb, sk_emb = pos, sk
        
        if hasattr(self, 'pos_embed'):
            pos_emb = self.pos_embed(pos_emb)
            sk_emb = self.pos_embed(sk_emb)
        
        sff = self.point_proj(x)
        if self.with_radius:
            skf = self.sk_layer.point_proj(torch.cat((sk, r), dim = -1))
        else:
            skf = self.sk_layer.point_proj(sk)
        ## create shared lf, vlf and ca for skeleton and surface?
        for lid, lf in enumerate(self.sk_layer.lf[:-1]):
            vsff, vskf = sff, skf
            sff, skf = lf(sff, c = c), lf(skf, c = c)

            if hasattr(self.sk_layer, 'vlf'):
                sff = sff + self.sk_layer.vlf[lid](vsff, pos, c = c)
                skf = skf + self.sk_layer.vlf[lid](vskf, sk, c = c)
            if hasattr(self.sk_layer, 'ca'):
                sff = self.sk_layer.ca[lid](sff)
                skf = self.sk_layer.ca[lid](sff)
            ## group skeleton feature from neighbor surface points
            skf = self.group_surface_to_skeleton(lid, pos, pos_emb, sk, sk_emb, sff, skf, c)
            sf_res.append(sff)
            sk_res.append(skf)
            
        vsff = torch.concat(sf_res, dim = -1)
        vskf = torch.concat(sk_res, dim = -1)
        sff, skf = self.sk_layer.lf[-1](vsff, c = c), self.sk_layer.lf[-1](vskf, c = c)
        # print(sk.mean(), r.mean(), pos_emb.mean(), sk_emb.mean(), sff.mean(), skf.mean())
        if hasattr(self.sk_layer, 'vlf'):
            sff = sff + self.sk_layer.vlf[-1](vsff, pos, c = c)
            skf = skf + self.sk_layer.vlf[-1](vskf, sk, c = c)
        if hasattr(self.sk_layer, 'ca'):
            sff = self.sk_layer.ca[-1](sff)
            skf = self.sk_layer.ca[-1](skf)
        skf = self.group_surface_to_skeleton(-1, pos, pos_emb, sk, sk_emb, sff, skf, c)
        
        ### concate global embedding and local point feautre
        x1 = skf.max(dim = 1, keepdim = True)[0]
        x2 = skf.mean(dim = 1, keepdim = True)
        x = torch.concat((x1, x2), dim = -1)
        ## vector embedding should always be false
        x = x.repeat(1, skf.shape[1], 1)
        x = torch.concat([x, skf], dim=-1)

        ## compute embedding vectors
        x = self.sk_layer.dense(x, c = c)
        return x, sk, r
    def forward(self, x, c = None, ske = None):
        spx, sp = super().forward(x, c = c)
        skx, sk, r = self.extract_skf(x, c = c, ske = ske)
        x = torch.cat((spx, skx), dim = 1)
        pk = torch.cat((sp, sk), dim = 1)

        ### without activation
        if self.standardize_latents:
            x = self.normalize(x)
        if self.with_radius:
            r = torch.cat((torch.zeros((spx.shape[0], spx.shape[1], 1),
                                       dtype = spx.dtype, device = spx.device), r), 
                                       dim = 1)
            return torch.cat((pk, r, x), dim = -1)
        else:
            return torch.cat((pk, x), dim = -1)
            

class SKSPMambaEncoder(PointMamba2Encoder):
    def __init__(self, point_dim = 3,
                point_num = 2560,
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
                voxel_attens = [],
                voxel_conv_kernel_size = 3,
                with_se = False,
                coordinate_normalize = True,
                condition_channel = 0,
                condition_injection = 'gate_bias',
                local_feature_channels = [64, 64, 128, 256], 
                num_neighbors_k_self = 0,
                num_neighbors_k_cross = 32,
                skeleton_net_config = {},
                num_skeleton_points = 256,
                with_radius = False,
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
                voxel_attens=voxel_attens,
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
        self.pos_embed_channel = 3
        if pos_embedding:
            self.pos_embed_channel = projection_channel
        self.sk_layer = PointMambaEncoder(point_dim=point_dim + with_radius, 
                time_channel = 0,
                time_injection = 'gate_bias',
                projection_channel = projection_channel,
                local_feature_channels = local_feature_channels, 
                num_blocks = num_blocks,
                num_neighbors_k = num_neighbors_k_self,
                building_block = building_block,
                dense_channels = dense_channels,
                normalization = normalization,
                num_norm_groups = num_norm_groups,
                activation = activation, dropout = dropout, 
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
                vector_embedding = False, 
                channel_attention = channel_attention,
                voxel_resolutions=voxel_resolutions,
                voxel_attens=voxel_attens,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize,
                condition_channel = condition_channel,
                condition_injection = condition_injection,
                condition_first = False)
        is_seq = self.BuildingBlock.func.is_sequence_modeling()
        if is_seq:
            self.sk_layer.meshage = nn.ModuleList([MultipleBuildingBlocks(in_channel = lfc + self.pos_embed_channel,
                                out_channel = lfc,
                                n = num_blocks,
                                BuildingBlock = partial(
                                    GroupSeqModelingLayer,
                                    BuildingBlock = self.BuildingBlock),)
                            for lfc in local_feature_channels ])
        else:
            self.sk_layer.meshage = nn.ModuleList([MultipleBuildingBlocks(in_channel = lfc + self.pos_embed_channel,
                                out_channel = lfc,
                                n = num_blocks,
                                BuildingBlock = self.BuildingBlock) 
                            for lfc in local_feature_channels ])
        assert self.out_channel == self.sk_layer.out_channel, \
            "skeleton latents should have an identical number of channels with sparse-point latents."
        self.point_num = point_num
        if not skeleton_net_config is None:
            logger.info('Using online skeletonization.')
            skeleton_net_config['point_num'] = point_num
            self.skeletonize = SkeletonNet(skeleton_net_config)
            self.num_skeleton_points = self.skeletonize.skp_num
        else:
            logger.info('Pre-computed skeleton is needed.')
            self.skeletonize = None
            self.num_skeleton_points = num_skeleton_points
            if with_radius:
                self.knn = KNN(k = 4, transpose_mode=True)
        ## 4 dim: xyzr
        self.query_and_group = QueryAndGroup(num_neighbors_k_cross, knn_query=knn_query)
        self.out_channel += (3 + with_radius)
        self.with_radius = with_radius
        self.standardize_latents = standardize_latents
        self.num_latent_points = self.num_fps_points[-1] + self.num_skeleton_points

        self.standardize_latents = standardize_latents
        if self.standardize_latents:
            ## without activation
            self.normalize = nn.Sequential(nn.Linear(dense_channels[-1], dense_channels[-1]),
                        nn.LayerNorm((self.num_latent_points, dense_channels[-1]), 
                        elementwise_affine = False, bias = False))

    def group_surface_to_skeleton(self, layer_id, pos, pos_emb, sk, sk_emb, sff, skf, c):
        sff_trans = channel_recover(sff)
        skf_trans = channel_recover(skf)
        grouped_sff = channel_transfer(self.query_and_group(pos, pos_emb, sk, sk_emb, sff_trans, skf_trans))
        ## intergrate skf into grouped surface feature
        sk_emb_feature = torch.cat((sk_emb - sk_emb, skf), dim = -1)
        grouped_sff = torch.cat((grouped_sff, sk_emb_feature.unsqueeze(2)), dim = 2)
        grouped_sff = self.sk_layer.meshage[layer_id](grouped_sff, c = c)
        ### (B, N, K, C) -> (B, N, C), N: num of skeletal points
        skf = grouped_sff.max(dim = 2)[0]
        return skf
    ## extract skeleton feature
    def extract_skf(self, x, c = None, ske = None):
        sf_res, sk_res = [], []

        pos = x[..., :3]
        if self.skeletonize:
            if not ske is None:
                logger.warning('This model extract skeleton online, the input skeleton will be ignored.')
            ske_res = self.skeletonize(pos)
            sk, r = ske_res['recon_skeleton'], ske_res['radius']
        else:
            assert not ske is None, 'Precomputed skeleton is needed.'
            sk = ske
            if ske.shape[-1] == 4:
                sk, r = ske[...,0:3], ske[..., 3:4]
            elif self.with_radius:
                dist, _ = self.knn(pos, sk)
                r = dist.mean(dim = -1, keepdim = True)
        assert sk.shape[1] == self.num_skeleton_points, \
            'The number of points in the input skeleton does not match the model.'
        pos_emb, sk_emb = pos, sk
        
        if hasattr(self, 'pos_embed'):
            pos_emb = self.pos_embed(pos_emb)
            sk_emb = self.pos_embed(sk_emb)
        
        sff = self.point_proj(x)
        if self.with_radius:
            skf = self.sk_layer.point_proj(torch.cat((sk, r), dim = -1))
        else:
            skf = self.sk_layer.point_proj(sk)
        ## create shared lf, vlf and ca for skeleton and surface?
        for lid, lf in enumerate(self.sk_layer.lf[:-1]):
            vsff, vskf = sff, skf
            sff, skf = lf(sff, c = c), lf(skf, c = c)

            if hasattr(self.sk_layer, 'vlf'):
                sff = sff + self.sk_layer.vlf[lid](vsff, pos, c = c)
                skf = skf + self.sk_layer.vlf[lid](vskf, sk, c = c)
            if hasattr(self.sk_layer, 'ca'):
                sff = self.sk_layer.ca[lid](sff)
                skf = self.sk_layer.ca[lid](sff)
            ## group skeleton feature from neighbor surface points
            skf = self.group_surface_to_skeleton(lid, pos, pos_emb, sk, sk_emb, sff, skf, c)
            sf_res.append(sff)
            sk_res.append(skf)
            
        vsff = torch.concat(sf_res, dim = -1)
        vskf = torch.concat(sk_res, dim = -1)
        sff, skf = self.sk_layer.lf[-1](vsff, c = c), self.sk_layer.lf[-1](vskf, c = c)
        # print(sk.mean(), r.mean(), pos_emb.mean(), sk_emb.mean(), sff.mean(), skf.mean())
        if hasattr(self.sk_layer, 'vlf'):
            sff = sff + self.sk_layer.vlf[-1](vsff, pos, c = c)
            skf = skf + self.sk_layer.vlf[-1](vskf, sk, c = c)
        if hasattr(self.sk_layer, 'ca'):
            sff = self.sk_layer.ca[-1](sff)
            skf = self.sk_layer.ca[-1](skf)
        skf = self.group_surface_to_skeleton(-1, pos, pos_emb, sk, sk_emb, sff, skf, c)
        
        ### concate global embedding and local point feautre
        x1 = skf.max(dim = 1, keepdim = True)[0]
        x2 = skf.mean(dim = 1, keepdim = True)
        x = torch.concat((x1, x2), dim = -1)
        ## vector embedding should always be false
        x = x.repeat(1, skf.shape[1], 1)
        x = torch.concat([x, skf], dim=-1)

        ## compute embedding vectors
        x = self.sk_layer.dense(x, c = c)
        return x, sk, r
    def forward(self, x, c = None, ske = None):
        spx, sp = super().forward(x, c = c)
        skx, sk, r = self.extract_skf(x, c = c, ske = ske)
        x = torch.cat((spx, skx), dim = 1)
        pk = torch.cat((sp, sk), dim = 1)

        ### without activation
        if self.standardize_latents:
            x = self.normalize(x)
        if self.with_radius:
            r = torch.cat((torch.zeros((spx.shape[0], spx.shape[1], 1),
                                       dtype = spx.dtype, device = spx.device), r), 
                                       dim = 1)
            return torch.cat((pk, r, x), dim = -1)
        else:
            return torch.cat((pk, x), dim = -1)