### point cloud auto-encoder, with skeleton regularization
import torch
import torch.nn as nn
from flemme.block import QueryAndGroup, MultipleBuildingBlocks, \
                    GroupSeqModelingLayer, get_building_block, \
                    channel_transfer, channel_recover, \
                    LocalGraphLayer, PositionEmbeddingBlock, \
                    ScaleShiftBlock
from flemme.encoder import PointEncoder, SeqNetDecoder
from flemme.logger import get_logger
from meshage.sknet import SkeletonNet
from functools import partial
from knn_cuda import KNN
### skeleton-regularized point cloud auto-encoder
logger = get_logger("skeleton_encoder")

class SkeletonEncoder(PointEncoder):
    def __init__(self, point_dim,
                 point_num,
                 projection_channel,
                 num_neighbors_k_self,
                 num_neighbors_k_cross, 
                 local_feature_channels,
                 voxel_resolutions,
                 voxel_attens,
                 voxel_conv_kernel_size,
                 num_blocks,
                 dense_channels,
                 activation, dropout,
                 normalization, num_norm_groups,  
                 channel_attention,
                 with_se,
                 coordinate_normalize,
                 condition_channel,
                 condition_injection,
                 skeleton_net_config,
                 num_skeleton_points,
                 pos_embedding,
                 with_radius,
                 standardize_latents,
                 knn_query,
                 **kwargs):
        super().__init__(point_dim=point_dim, 
                time_channel = 0,
                time_injection = 'gate_bias',
                projection_channel = projection_channel,
                local_feature_channels = local_feature_channels, 
                num_blocks = num_blocks,
                num_neighbors_k = num_neighbors_k_self,
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
        self.pos_embed_channel = 3
        if pos_embedding:
            self.pos_embed_channel = projection_channel
            if pos_embedding == 'sin':
                self.pos_embed = PositionEmbeddingBlock(in_channel = 3, 
                    out_channel = projection_channel,
                    activation = activation)
                logger.info("Using sinusoidal point cloud positional embedding.")
            else:
                self.pos_embed = nn.Linear(3, projection_channel)
                logger.info("Using point cloud positional embedding.")
        self.point_num = point_num
        if not skeleton_net_config is None:
            logger.info('Using online skeletonization.')
            skeleton_net_config['point_num'] = point_num
            self.skeletonize = SkeletonNet(skeleton_net_config)
            self.num_latent_points = self.skeletonize.skp_num
        else:
            logger.info('Pre-computed skeleton is needed.')
            self.skeletonize = None
            self.num_latent_points = num_skeleton_points
            if with_radius:
                self.knn = KNN(k = 4, transpose_mode=True)
        ## 4 dim: xyzr
        
        self.sk_proj = nn.Linear(3 + with_radius, projection_channel)
        self.query_and_group = QueryAndGroup(num_neighbors_k_cross, knn_query=knn_query)
        self.out_channel += (3 + with_radius)
        self.with_radius = with_radius
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
        grouped_sff = self.meshage[layer_id](grouped_sff, c = c)
        ### (B, N, K, C) -> (B, N, C), N: num of skeletal points
        skf = grouped_sff.max(dim = 2)[0]
        return skf
    def forward(self, x, c = None, ske = None):
        if self.lf is None:
            raise NotImplementedError
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
        assert sk.shape[1] == self.num_latent_points, \
            'The number of points in the input skeleton does not match the model.'
        pos_emb, sk_emb = pos, sk
        
        if hasattr(self, 'pos_embed'):
            pos_emb = self.pos_embed(pos_emb)
            sk_emb = self.pos_embed(sk_emb)
        
        sff = self.point_proj(x)
        if self.with_radius:
            skf = self.sk_proj(torch.cat((sk, r), dim = -1))
        else:
            skf = self.sk_proj(sk)
        ## create shared lf, vlf and ca for skeleton and surface?
        for lid, lf in enumerate(self.lf[:-1]):
            vsff, vskf = sff, skf
            sff, skf = lf(sff, c = c), lf(skf, c = c)

            if hasattr(self, 'vlf'):
                sff = sff + self.vlf[lid](vsff, pos, c = c)
                skf = skf + self.vlf[lid](vskf, sk, c = c)
            if hasattr(self, 'ca'):
                sff = self.ca[lid](sff)
                skf = self.ca[lid](sff)
            ## group skeleton feature from neighbor surface points
            skf = self.group_surface_to_skeleton(lid, pos, pos_emb, sk, sk_emb, sff, skf, c)
            sf_res.append(sff)
            sk_res.append(skf)
            
        vsff = torch.concat(sf_res, dim = -1)
        vskf = torch.concat(sk_res, dim = -1)
        sff, skf = self.lf[-1](vsff, c = c), self.lf[-1](vskf, c = c)
        # print(sk.mean(), r.mean(), pos_emb.mean(), sk_emb.mean(), sff.mean(), skf.mean())
        if hasattr(self, 'vlf'):
            sff = sff + self.vlf[-1](vsff, pos, c = c)
            skf = skf + self.vlf[-1](vskf, sk, c = c)
        if hasattr(self, 'ca'):
            sff = self.ca[-1](sff)
            skf = self.ca[-1](skf)
        skf = self.group_surface_to_skeleton(-1, pos, pos_emb, sk, sk_emb, sff, skf, c)
        
        ### concate global embedding and local point feautre
        ### is this necessary?
        x1 = skf.max(dim = 1, keepdim = True)[0]
        x2 = skf.mean(dim = 1, keepdim = True)
        x = torch.concat((x1, x2), dim = -1)
        ## vector embedding should always be false
        x = x.repeat(1, skf.shape[1], 1)
        x = torch.concat([x, skf], dim=-1)

        ## compute embedding vectors
        x = self.dense(x, c = c)
        ### without activation
        if self.standardize_latents:
            x = self.normalize(x)
        if self.with_radius:
            return torch.cat((sk, r, x), dim = -1)
        else:
            return torch.cat((sk, x), dim = -1)

class SkeletonCNNEncoder(SkeletonEncoder):
    def __init__(self, point_dim=3, 
                 point_num = 2560,
                 projection_channel = 64,
                 num_neighbors_k_self = 0, 
                 num_neighbors_k_cross = 32,
                 ### point-voxel cnn
                 local_feature_channels = [64, 64, 128, 256], 
                 voxel_resolutions = [],
                 voxel_attens = [],
                 num_blocks = 2,
                 dense_channels = [256, 256],
                 building_block = 'dense', 
                 normalization = 'group', num_norm_groups = 8, 
                 activation = 'lrelu', dropout = 0., 
                 channel_attention = None, 
                 voxel_conv_kernel_size = 3,
                 with_se = False,
                 coordinate_normalize = True,
                 condition_channel = 0,
                 condition_injection = 'gate_bias',
                 skeleton_net_config = {},
                 num_skeleton_points = 256,
                 pos_embedding = True,
                 with_radius = False,
                 standardize_latents = True,
                 knn_query = 'feature',
                 **kwargs):
        super().__init__(point_dim=point_dim, 
                point_num=point_num,
                projection_channel = projection_channel,
                num_neighbors_k_self=num_neighbors_k_self,
                num_neighbors_k_cross=num_neighbors_k_cross, 
                local_feature_channels = local_feature_channels, 
                num_blocks = num_blocks,
                dense_channels = dense_channels,
                normalization = normalization,
                num_norm_groups = num_norm_groups,
                activation = activation, dropout = dropout, 
                channel_attention = channel_attention,
                voxel_resolutions=voxel_resolutions,
                voxel_attens=voxel_attens,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize,
                condition_channel = condition_channel,
                condition_injection = condition_injection,
                skeleton_net_config = skeleton_net_config,
                num_skeleton_points = num_skeleton_points,
                pos_embedding = pos_embedding,
                with_radius = with_radius,
                standardize_latents = standardize_latents,
                knn_query = knn_query)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))

        self.BuildingBlock = get_building_block(building_block, 
                                        activation=activation, 
                                        norm = normalization, 
                                        num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection)
        
        if self.num_neighbors_k > 0:
            lf_sequence = [LocalGraphLayer(k = self.num_neighbors_k, 
                                            in_channel = self.lf_path[i],
                                            out_channel = self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,
                                            num_blocks = self.num_blocks,
                                            ) for i in range(len(self.lf_path) - 2) ]
        ## local feature, similar to pointnet
        else:  
            lf_sequence = [MultipleBuildingBlocks(in_channel=self.lf_path[i], 
                                            out_channel=self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,
                                            num_blocks = self.num_blocks,
                                            channel_dim=-1) for i in range(len(self.lf_path) - 2) ]
        
        lf_sequence.append(self.BuildingBlock(in_channel=sum(self.lf_path[1:-1]), 
                                        out_channel=self.lf_path[-1]))
        self.lf = nn.ModuleList(lf_sequence)
        ## feature aggregating after group
        is_seq = self.BuildingBlock.func.is_sequence_modeling()
        if is_seq:
            self.meshage = nn.ModuleList([MultipleBuildingBlocks(in_channel = lfc + self.pos_embed_channel,
                                out_channel = lfc,
                                n = num_blocks,
                                BuildingBlock = partial(
                                    GroupSeqModelingLayer,
                                    BuildingBlock = self.BuildingBlock),)
                            for lfc in local_feature_channels ])
        else:
            self.meshage = nn.ModuleList([MultipleBuildingBlocks(in_channel = lfc + self.pos_embed_channel,
                                out_channel = lfc,
                                n = num_blocks,
                                BuildingBlock = self.BuildingBlock) 
                            for lfc in local_feature_channels ])
            

class SkeletonTransEncoder(SkeletonEncoder):
    def __init__(self, point_dim=3, 
                    point_num=2560,
                    projection_channel = 64,
                    num_neighbors_k_self=0,
                    num_neighbors_k_cross=32, 
                    local_feature_channels = [64, 64, 128, 256], 
                    voxel_resolutions = [],
                    voxel_attens = [],
                    num_blocks = 2,
                    dense_channels = [256, 256],
                    building_block = 'pct_sa', 
                    normalization = 'group', num_norm_groups = 8, 
                    activation = 'lrelu', dropout = 0., num_heads = 4, d_k = None, 
                    qkv_bias = True, qk_scale = None, atten_dropout = None, 
                    mlp_hidden_ratios=[4.0, 4.0], 
                    channel_attention = None,
                    voxel_conv_kernel_size = 3,
                    with_se = False,
                    coordinate_normalize = True,
                    condition_channel = 0,
                    condition_injection = 'gate_bias',
                    pos_embedding = True,
                    skeleton_net_config = {},
                    num_skeleton_points = 256,
                    with_radius = False,
                    standardize_latents = True,
                    knn_query = 'feature',
                    **kwargs):
        super().__init__(point_dim=point_dim, 
                point_num=point_num,
                projection_channel = projection_channel,
                num_neighbors_k_self=num_neighbors_k_self,
                num_neighbors_k_cross=num_neighbors_k_cross, 
                local_feature_channels = local_feature_channels, 
                num_blocks = num_blocks,
                dense_channels = dense_channels,
                normalization = normalization,
                num_norm_groups = num_norm_groups,
                activation = activation, dropout = dropout, 
                channel_attention = channel_attention,
                voxel_resolutions=voxel_resolutions,
                voxel_attens=voxel_attens,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize,
                condition_channel = condition_channel,
                condition_injection = condition_injection,
                skeleton_net_config = skeleton_net_config,
                num_skeleton_points = num_skeleton_points,
                pos_embedding = pos_embedding,
                with_radius = with_radius,
                standardize_latents = standardize_latents,
                knn_query = knn_query)
        
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))

        self.BuildingBlock = get_building_block(building_block, 
                                        activation=activation, 
                                        norm = normalization, num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout,
                                        mlp_hidden_ratios = mlp_hidden_ratios,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection)
        
        if self.num_neighbors_k > 0:
            lf_sequence = [LocalGraphLayer(k = self.num_neighbors_k, 
                                            in_channel = self.lf_path[i],
                                            out_channel = self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,
                                            num_blocks = self.num_blocks,
                                            ) for i in range(len(self.lf_path) - 2) ]
        ## local feature, similar to pointnet
        else:  
            lf_sequence = [MultipleBuildingBlocks(in_channel=self.lf_path[i], 
                                            out_channel=self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,
                                            num_blocks = self.num_blocks,
                                            channel_dim=-1) for i in range(len(self.lf_path) - 2) ]
        
        lf_sequence.append(self.BuildingBlock(in_channel=sum(self.lf_path[1:-1]), 
                                        out_channel=self.lf_path[-1]))
        self.lf = nn.ModuleList(lf_sequence)
        ## feature aggregating after group
        is_seq = self.BuildingBlock.func.is_sequence_modeling()
        if is_seq:
            self.meshage = nn.ModuleList([MultipleBuildingBlocks(in_channel = lfc + self.pos_embed_channel,
                                out_channel = lfc,
                                n = num_blocks,
                                BuildingBlock = partial(
                                    GroupSeqModelingLayer,
                                    BuildingBlock = self.BuildingBlock),)
                            for lfc in local_feature_channels ])
        else:
            self.meshage = nn.ModuleList([MultipleBuildingBlocks(in_channel = lfc + self.pos_embed_channel,
                                out_channel = lfc,
                                n = num_blocks,
                                BuildingBlock = self.BuildingBlock) 
                            for lfc in local_feature_channels ])
            

class SkeletonMambaEncoder(SkeletonEncoder):
    def __init__(self, point_dim=3, 
                point_num=2560,
                projection_channel = 64,
                num_neighbors_k_self=0,
                num_neighbors_k_cross=32, 
                local_feature_channels = [64, 64, 128, 256], 
                voxel_resolutions = [],
                voxel_attens = [],
                num_blocks = 2,
                dense_channels = [256, 256],
                building_block = 'pmamba', 
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
                channel_attention = None,
                voxel_conv_kernel_size = 3,
                with_se = False,
                coordinate_normalize = True,
                condition_channel = 0,
                condition_injection = 'gate_bias',
                pos_embedding = True,
                skeleton_net_config = {},
                num_skeleton_points = 256,
                with_radius = False,
                standardize_latents = True,
                knn_query = 'feature',
                **kwargs):
        super().__init__(point_dim=point_dim, 
                point_num=point_num,
                projection_channel = projection_channel,
                num_neighbors_k_self=num_neighbors_k_self,
                num_neighbors_k_cross=num_neighbors_k_cross, 
                local_feature_channels = local_feature_channels, 
                num_blocks = num_blocks,
                dense_channels = dense_channels,
                normalization = normalization,
                num_norm_groups = num_norm_groups,
                activation = activation, dropout = dropout, 
                channel_attention = channel_attention,
                voxel_resolutions=voxel_resolutions,
                voxel_attens=voxel_attens,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize,
                condition_channel = condition_channel,
                condition_injection = condition_injection,
                skeleton_net_config = skeleton_net_config,
                num_skeleton_points = num_skeleton_points,
                pos_embedding = pos_embedding,
                with_radius = with_radius,
                standardize_latents = standardize_latents,
                knn_query = knn_query)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))

        self.BuildingBlock = get_building_block(building_block, 
                                        activation=activation, 
                                        norm = normalization, 
                                        num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
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
                                        mlp_hidden_ratios = mlp_hidden_ratios,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection)
        
        if self.num_neighbors_k > 0:
            lf_sequence = [LocalGraphLayer(k = self.num_neighbors_k, 
                                            in_channel = self.lf_path[i],
                                            out_channel = self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,
                                            num_blocks = self.num_blocks,
                                            ) for i in range(len(self.lf_path) - 2) ]
        ## local feature, similar to pointnet
        else:  
            lf_sequence = [MultipleBuildingBlocks(in_channel=self.lf_path[i], 
                                            out_channel=self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,
                                            num_blocks = self.num_blocks,
                                            channel_dim=-1) for i in range(len(self.lf_path) - 2) ]
        
        lf_sequence.append(self.BuildingBlock(in_channel=sum(self.lf_path[1:-1]), 
                                        out_channel=self.lf_path[-1]))
        self.lf = nn.ModuleList(lf_sequence)
        ## feature aggregating after group
        is_seq = self.BuildingBlock.func.is_sequence_modeling()
        if is_seq:
            self.meshage = nn.ModuleList([MultipleBuildingBlocks(in_channel = lfc + self.pos_embed_channel,
                                out_channel = lfc,
                                n = num_blocks,
                                BuildingBlock = partial(
                                    GroupSeqModelingLayer,
                                    BuildingBlock = self.BuildingBlock),)
                            for lfc in local_feature_channels ])
        else:
            self.meshage = nn.ModuleList([MultipleBuildingBlocks(in_channel = lfc + self.pos_embed_channel,
                                out_channel = lfc,
                                n = num_blocks,
                                BuildingBlock = self.BuildingBlock) 
                            for lfc in local_feature_channels ])



class SkeletonSDFDecoder(SeqNetDecoder):
    def __init__(self, point_dim=1,
                ### embedding  
                projection_channel = 256, 
                latent_channel = 16,
                latent_projection_channel = None,
                latent_injection = 'cross_atten',
                num_latent_points = 256,
                num_blocks = 2,
                building_block = 'dense', seq_feature_channels = [], 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                with_radius = False,
                standardize_latents = True,
                self_atten_for_latent = False,
                ## embed latent attention into building blocks
                embed_la = True,
                **kwargs):
        latent_projection_channel = latent_projection_channel or latent_channel
        super().__init__(point_dim = point_dim,
            latent_channel = projection_channel,
            time_channel = latent_projection_channel,
            time_injection = latent_injection,
            num_blocks = num_blocks,
            building_block = building_block,
            seq_feature_channels = seq_feature_channels,
            normalization = normalization,
            num_norm_groups = num_norm_groups,
            activation = activation,
            dropout = dropout,
            condition_channel = condition_channel,
            condition_injection = condition_injection)
        # self.out_point_dim = point_dim
        if not latent_channel == latent_projection_channel:
            self.latent_point_proj = nn.Linear(latent_channel,
                latent_projection_channel)
        self.coord_proj = PositionEmbeddingBlock(in_channel = 3, 
                    out_channel = projection_channel,
                    activation = activation)
        self.with_radius = with_radius
        self.standardize_latents = standardize_latents
        if standardize_latents:
            self.scale_shift = ScaleShiftBlock((num_latent_points, latent_channel - 3 - with_radius))
        if self_atten_for_latent:
            logger.info('Apply self-attention on latent points.')
            SABlock = get_building_block('pct_sa',                                        
                                        activation=activation, 
                                        norm = normalization, 
                                        num_norm_groups = num_norm_groups, 
                                        dropout = dropout,)
            sequence = [MultipleBuildingBlocks(n = self.num_blocks, 
                                    BuildingBlock=SABlock,
                                    in_channel=latent_projection_channel, 
                                    out_channel=latent_projection_channel) 
                                for i in range(len(self.seq_path) - 1) ]
            self.latent_attens = nn.ModuleList(sequence)
        self.embed_la = embed_la
        
    ## latent: B * N * C, coordinate: B * M * 3, return B * M * 1 (sdf)
    def forward(self, latent, coordinate, c = None):
        ### use cross attention to compute coordinate and local feature
        if self.standardize_latents:
            latent = torch.concat((latent[..., :3 + self.with_radius], 
                self.scale_shift(latent[..., 3 + self.with_radius:])), dim = -1)
        if hasattr(self, 'latent_point_proj'):
            latent = self.latent_point_proj(latent)
        if hasattr(self, 'latent_attens') and not self.embed_la:
            for la in self.latent_attens:
                latent = la(latent)
        coord_feature = self.coord_proj(coordinate)
        for sid in range(len(self.seq)):
            if hasattr(self, 'latent_attens') and self.embed_la:
                latent = self.latent_attens[sid](latent)
            coord_feature = self.seq[sid](coord_feature, t = latent)
        # res = super().forward(coord_feature, t = latent, c = c)
        return self.latent_proj(coord_feature)

