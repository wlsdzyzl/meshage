### point cloud auto-encoder, with skeleton regularization
import torch
import torch.nn as nn
from flemme.block import QueryAndGroup, MultipleBuildingBlocks, \
                    GroupSeqModelingLayer, MultiLayerPerceptionBlock,\
                    channel_transfer, channel_recover
from flemme.encoder import PointNetEncoder, PointTransEncoder, \
        PointMambaEncoder, SeqNetDecoder, \
        SeqTransDecoder, SeqMambaDecoder, \
        supported_buildingblocks_for_encoder, \
        supported_point_encoders,\
        supported_encoders, create_encoder

from flemme.logger import get_logger

from .sknet import SkeletonNet
from functools import partial

### skeleton-regularized point cloud auto-encoder
logger = get_logger("skeleton_encoder")

class SkeletonEncoder(nn.Module):
    def __init__(self, 
                 projection_channel,
                 num_blocks, 
                 pos_embedding,
                 skeleton_net_config,
                 dense_channels,
                 skeleton_channel_scaling,
                 normalization,
                 num_norm_groups,
                 activation):
        super().__init__()
        self.pos_embed_channel = 3
        if pos_embedding:
            self.pos_embed = nn.Linear(3, projection_channel)
            self.pos_embed_channel = projection_channel
            logger.info("Using point cloud positional embedding.")
        self.skeletonize = SkeletonNet(skeleton_net_config)
        self.k = self.skeletonize.skp_neighbor_num
        self.query_and_group = QueryAndGroup(self.k, knn_query='xyz')
        self.sk_encoder = None
        self.sf_encoder = None
        self.msg = None
        self.sk_dense_channels = [int(skeleton_channel_scaling * f) for f in dense_channels]
        self.out_channel = self.sk_dense_channels[-1] * 2
        self.global_mlps = MultiLayerPerceptionBlock(in_channel = self.out_channel * 2, 
                                    out_channel = self.out_channel, n = num_blocks,
                                    norm = normalization, 
                                    num_norm_groups = num_norm_groups,
                                    activation = activation)
        self.local_mlps = MultiLayerPerceptionBlock(in_channel = self.out_channel, 
                                    out_channel = self.out_channel, n = num_blocks,
                                    norm = normalization, 
                                    num_norm_groups = num_norm_groups,
                                    activation = activation) 
    def __str__(self):
        _str = f'skeleton_encoder: {self.sk_encoder.__str__()}'
        _str += f'surface_encoder: {self.sf_encoder.__str__()}'
        _str += f'Local Dense layers: {self.out_channel }->{self.out_channel}\n'
        _str += f'Global Dense layers: {self.out_channel * 2}->{self.out_channel}\n'
        
        return _str 


    def forward(self, x, c = None):

        pos = x[..., :3]
        ske_res = self.skeletonize(pos)
        
        sk, r = ske_res['recon_skeleton'], ske_res['radius']
        pos_emb, sk_emb = pos, sk

        ## surface features
        sff = self.sf_encoder.forward(x, c = c)
        ## skeleton features
        skf = self.sk_encoder(torch.concat((sk, r), dim = -1), c = c)

        if hasattr(self, 'pos_embed'):
            pos_emb = self.pos_embed(pos_emb)
            sk_emb = self.pos_embed(sk_emb)
        ## group surface features using skeletal points and skeletal features
        # print(pos.shape, sk.shape, pos_emb.shape, sk_emb.shape)
        sff_trans = channel_recover(sff)
        grouped_sff = channel_transfer(self.query_and_group(pos, pos_emb, sk, sk_emb, sff_trans))
        sk_emb_feature = torch.cat((sk_emb - sk_emb, skf), dim = -1)
        grouped_sff = torch.cat((grouped_sff, sk_emb_feature.unsqueeze(2)), dim = 2)
        grouped_sff = self.msg(grouped_sff)
        grouped_sff_max = grouped_sff.max(dim = 2)[0]
        grouped_sff_avg = grouped_sff.mean(dim = 2)
        # skf = torch.cat( (skf, grouped_sff), dim = -1)
        skf = torch.cat((grouped_sff_max, grouped_sff_avg), dim = -1)

        skf_max = skf.max(dim=1, keepdim = True)[0]
        skf_avg = skf.mean(dim = 1, keepdim = True)
        global_skf = torch.cat((skf_max, skf_avg), dim = -1)


        global_features = self.global_mlps(global_skf)
        local_features = self.local_mlps(skf)
        
        return local_features, global_features, sk, r

class SkeletonCNNEncoder(SkeletonEncoder):
    def __init__(self, point_dim=3, 
                 projection_channel = 64,
                 time_channel = 0,
                 num_neighbors_k=0, 
                 ### point-voxel cnn
                 local_feature_channels = [64, 64, 128, 256], 
                 skeleton_channel_scaling = 1.0,
                 ### scaling: number of nearest neighbor 
                 skeleton_nnn_scaling = 0.5,
                 voxel_resolutions = [],
                 num_blocks = 2,
                 dense_channels = [256, 256],
                 building_block = 'dense', 
                 normalization = 'group', num_norm_groups = 8, 
                 activation = 'lrelu', dropout = 0., 
                 channel_attention = None, 
                 time_injection = 'gate_bias', 
                 voxel_conv_kernel_size = 3,
                 with_se = False,
                 coordinate_normalize = True,
                 pos_embedding = True,
                 hidden_channels = [512, 512],
                 skeleton_net_config = {},
                 **kwargs):
        super().__init__(projection_channel = projection_channel,
                 num_blocks = num_blocks,
                 pos_embedding = pos_embedding,
                 skeleton_net_config = skeleton_net_config,
                 dense_channels = dense_channels,
                 normalization=normalization,
                 num_norm_groups=num_norm_groups,
                 activation=activation,
                 skeleton_channel_scaling = skeleton_channel_scaling,
                 )
                 
        
        ## encode skeleton feature from skeletal points
        ## point dim = point dim + radius dim
        self.sf_encoder = PointNetEncoder(point_dim=point_dim, 
                projection_channel = projection_channel,
                time_channel = time_channel,
                num_neighbors_k=num_neighbors_k, 
                local_feature_channels = local_feature_channels, 
                num_blocks = num_blocks,
                building_block = building_block,
                dense_channels = dense_channels,
                normalization = normalization,
                num_norm_groups = num_norm_groups,
                activation = activation, dropout = dropout, 
                ## father z_count is 1, 
                ## because they are just used to compute local point feature for surface and sle;etpm
                z_count = 1, vector_embedding = False, 
                channel_attention = channel_attention,
                time_injection=time_injection,
                voxel_resolutions=voxel_resolutions,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize)
        
        self.sk_encoder = PointNetEncoder(point_dim=point_dim + 1, 
                projection_channel = projection_channel,
                time_channel = time_channel,
                num_neighbors_k= int(num_neighbors_k * skeleton_nnn_scaling), 
                local_feature_channels = [int(f * skeleton_channel_scaling) 
                                            for f in local_feature_channels], 
                num_blocks = num_blocks,
                building_block = building_block,
                dense_channels = self.sk_dense_channels,
                normalization = normalization,
                num_norm_groups = num_norm_groups,
                activation = activation, dropout = dropout, 
                vector_embedding = False, 
                channel_attention = channel_attention,
                time_injection=time_injection,
                voxel_resolutions=voxel_resolutions,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize)

        self.BuildingBlock = self.sf_encoder.BuildingBlock
        ## feature aggregating
        is_seq = self.BuildingBlock.func.is_sequence_modeling()
        if is_seq:
            self.msg = MultipleBuildingBlocks(in_channel = self.sf_encoder.out_channel + self.pos_embed_channel,
                            out_channel = self.sk_encoder.out_channel,
                            n = num_blocks,
                            hidden_channels = hidden_channels,
                            BuildingBlock = partial(
                                GroupSeqModelingLayer,
                                BuildingBlock = self.BuildingBlock),
                            )
        else:
            self.msg = MultipleBuildingBlocks(in_channel = self.sf_encoder.out_channel + self.pos_embed_channel,
                            out_channel = self.sk_encoder.out_channel,
                            hidden_channels = hidden_channels,
                            n = num_blocks,
                            BuildingBlock = self.BuildingBlock)
            

class SkeletonTransEncoder(SkeletonEncoder):
    def __init__(self, point_dim=3, 
                 projection_channel = 64,
                 time_channel = 0,
                 num_neighbors_k=0, 
                 ### point-voxel cnn
                 local_feature_channels = [64, 64, 128, 256], 
                 skeleton_channel_scaling = 1.0,
                 ### scaling: number of nearest neighbor 
                 skeleton_nnn_scaling = 0.5,
                 voxel_resolutions = [],
                 num_blocks = 2,
                 dense_channels = [256, 256],
                 building_block = 'dense', 
                 normalization = 'group', num_norm_groups = 8, 
                 activation = 'lrelu', dropout = 0., 
                 num_heads = 4, d_k = None, 
                 qkv_bias = True, qk_scale = None, atten_dropout = None, 
                 residual_attention = False, skip_connection = True,
                 channel_attention = None, 
                 time_injection = 'gate_bias', 
                 voxel_conv_kernel_size = 3,
                 with_se = False,
                 coordinate_normalize = True,
                 pos_embedding = True,
                 hidden_channels = [512, 512],
                 skeleton_net_config = {},
                 **kwargs):
        super().__init__(projection_channel = projection_channel,
                 num_blocks = num_blocks,
                 pos_embedding = pos_embedding,
                 skeleton_net_config = skeleton_net_config,
                 dense_channels = dense_channels,
                 normalization=normalization,
                 num_norm_groups=num_norm_groups,
                 activation=activation,
                 skeleton_channel_scaling = skeleton_channel_scaling,)
        
        ## encode skeleton feature from skeletal points
        ## point dim = point dim + radius dim
        self.sf_encoder = PointTransEncoder(point_dim=point_dim, 
                projection_channel = projection_channel,
                time_channel = time_channel,
                num_neighbors_k=num_neighbors_k, 
                local_feature_channels = local_feature_channels, 
                num_blocks = num_blocks,
                building_block = building_block,
                dense_channels = dense_channels,
                normalization = normalization,
                num_norm_groups = num_norm_groups,
                activation = activation, dropout = dropout, 
                num_heads = num_heads, d_k = d_k, 
                qkv_bias = qkv_bias, qk_scale = qk_scale, 
                atten_dropout = atten_dropout, 
                residual_attention = residual_attention, 
                skip_connection = skip_connection,
                vector_embedding = False, 
                channel_attention = channel_attention,
                time_injection=time_injection,
                voxel_resolutions=voxel_resolutions,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize)
        
        self.sk_encoder = PointTransEncoder(point_dim=point_dim + 1, 
                projection_channel = projection_channel,
                time_channel = time_channel,
                num_neighbors_k= int(num_neighbors_k * skeleton_nnn_scaling), 
                local_feature_channels = [int(f * skeleton_channel_scaling) 
                                            for f in local_feature_channels], 
                num_blocks = num_blocks,
                building_block = building_block,
                dense_channels = self.sk_dense_channels,
                normalization = normalization,
                num_norm_groups = num_norm_groups,
                activation = activation, dropout = dropout, 
                num_heads = num_heads, d_k = d_k, 
                qkv_bias = qkv_bias, qk_scale = qk_scale, 
                atten_dropout = atten_dropout, 
                residual_attention = residual_attention, 
                skip_connection = skip_connection,
                vector_embedding = False, 
                channel_attention = channel_attention,
                time_injection=time_injection,
                voxel_resolutions=voxel_resolutions,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize)

        self.BuildingBlock = self.sf_encoder.BuildingBlock
        ## feature aggregating
        is_seq = self.BuildingBlock.func.is_sequence_modeling()
        if is_seq:
            self.msg = MultipleBuildingBlocks(in_channel = self.sf_encoder.out_channel + self.pos_embed_channel,
                            out_channel = self.sk_encoder.out_channel,
                            n = num_blocks,
                            hidden_channels = hidden_channels,
                            BuildingBlock = partial(
                                GroupSeqModelingLayer,
                                BuildingBlock = self.BuildingBlock),
                            )
        else:
            self.msg = MultipleBuildingBlocks(in_channel = self.sf_encoder.out_channel + self.pos_embed_channel,
                            out_channel = self.sk_encoder.out_channel,
                            hidden_channels = hidden_channels,
                            n = num_blocks,
                            BuildingBlock = self.BuildingBlock)
            

class SkeletonMambaEncoder(SkeletonEncoder):
    def __init__(self, point_dim=3, 
                projection_channel = 64,
                time_channel = 0,
                num_neighbors_k=0, 
                ### point-voxel cnn
                local_feature_channels = [64, 64, 128, 256], 
                skeleton_channel_scaling = 1.0,
                ### scaling: number of nearest neighbor 
                skeleton_nnn_scaling = 0.5,
                voxel_resolutions = [],
                num_blocks = 2,
                dense_channels = [256, 256],
                building_block = 'dense', 
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
                channel_attention = None, 
                time_injection = 'gate_bias', 
                voxel_conv_kernel_size = 3,
                with_se = False,
                coordinate_normalize = True,
                pos_embedding = True,
                hidden_channels = [512, 512],
                skeleton_net_config = {},
                **kwargs):
        super().__init__(projection_channel = projection_channel,
                 num_blocks = num_blocks,
                 pos_embedding = pos_embedding,
                 skeleton_net_config = skeleton_net_config,
                 dense_channels = dense_channels,
                 normalization=normalization,
                 num_norm_groups=num_norm_groups,
                 activation=activation,
                 skeleton_channel_scaling = skeleton_channel_scaling,)
        
        ## encode skeleton feature from skeletal points
        ## point dim = point dim + radius dim
        self.sf_encoder = PointMambaEncoder(point_dim=point_dim, 
                projection_channel = projection_channel,
                time_channel = time_channel,
                num_neighbors_k=num_neighbors_k, 
                local_feature_channels = local_feature_channels, 
                num_blocks = num_blocks,
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
                vector_embedding = False, 
                channel_attention = channel_attention,
                time_injection=time_injection,
                voxel_resolutions=voxel_resolutions,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize)
        
        self.sk_encoder = PointMambaEncoder(point_dim=point_dim + 1, 
                projection_channel = projection_channel,
                time_channel = time_channel,
                num_neighbors_k= int(num_neighbors_k * skeleton_nnn_scaling), 
                local_feature_channels = [int(f * skeleton_channel_scaling) 
                                            for f in local_feature_channels], 
                num_blocks = num_blocks,
                building_block = building_block,
                dense_channels = self.sk_dense_channels,
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
                vector_embedding = False, 
                channel_attention = channel_attention,
                time_injection=time_injection,
                voxel_resolutions=voxel_resolutions,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize)

        self.BuildingBlock = self.sf_encoder.BuildingBlock
        ## feature aggregating
        is_seq = self.BuildingBlock.func.is_sequence_modeling()
        if is_seq:
            self.msg = MultipleBuildingBlocks(in_channel = self.sf_encoder.out_channel + self.pos_embed_channel,
                            out_channel = self.sk_encoder.out_channel,
                            n = num_blocks,
                            hidden_channels = hidden_channels,
                            BuildingBlock = partial(
                                GroupSeqModelingLayer,
                                BuildingBlock = self.BuildingBlock),
                            )
        else:
            self.msg = MultipleBuildingBlocks(in_channel = self.sf_encoder.out_channel + self.pos_embed_channel,
                            out_channel = self.sk_encoder.out_channel,
                            hidden_channels = hidden_channels,
                            n = num_blocks,
                            BuildingBlock = self.BuildingBlock)


class SkeletonCNNDecoder(SeqNetDecoder):
    def __init__(self, point_dim=3, in_channel = 256, 
                time_injection = 'gate_bias',
                num_neighbors_k = 4,
                num_blocks = 2,
                building_block = 'dense', seq_feature_channels = [], 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__(point_dim = point_dim,
            in_channel = in_channel // num_neighbors_k,
            time_channel = in_channel,
            time_injection = time_injection,
            num_blocks = num_blocks,
            building_block = building_block,
            seq_feature_channels = seq_feature_channels,
            normalization = normalization,
            num_norm_groups = num_norm_groups,
            activation = activation,
            dropout = dropout,
            condition_channel = condition_channel,
            condition_injection = condition_injection,
            condition_first = condition_first,
            )
        # self.out_point_dim = point_dim
        self.num_neighbors_k = num_neighbors_k
    def forward(self, local_feature, global_feature, c = None):
        B, N, _ = local_feature.shape
        local_feature = local_feature.reshape(B, N * self.num_neighbors_k, -1)
        res = super().forward(local_feature, t = global_feature, c = c)
        return res

class SkeletonTransDecoder(SeqTransDecoder):
    def __init__(self, point_dim=3, in_channel = 256, 
                time_injection = 'gate_bias',
                num_neighbors_k = 4,
                num_blocks = 2,
                building_block = 'pct_sa', seq_feature_channels = [], 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                num_heads = 4, d_k = None, 
                qkv_bias = True, qk_scale = None, atten_dropout = None, 
                mlp_hidden_ratios=[4.0, 4.0], 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__(point_dim = point_dim,
            in_channel = in_channel // num_neighbors_k,
            time_channel = in_channel,
            time_injection = time_injection,
            num_blocks = num_blocks,
            building_block = building_block,
            seq_feature_channels = seq_feature_channels,
            normalization = normalization,
            num_norm_groups = num_norm_groups,
            activation = activation,
            dropout = dropout, 
            num_heads = num_heads, d_k = d_k, 
            qkv_bias = qkv_bias, qk_scale = qk_scale, 
            atten_dropout = atten_dropout, 
            mlp_hidden_ratios=mlp_hidden_ratios, 
            condition_channel = condition_channel,
            condition_injection = condition_injection,
            condition_first = condition_first,
            )

        # self.out_point_dim = point_dim
        self.num_neighbors_k = num_neighbors_k
    def forward(self, local_feature, global_feature, c = None):
        B, N, _ = local_feature.shape
        local_feature = local_feature.reshape(B, N * self.num_neighbors_k, -1)
        res = super().forward(local_feature, t = global_feature, c = c)
        return res

class SkeletonMambaDecoder(SeqMambaDecoder):
    def __init__(self, point_dim=3, in_channel = 256, 
                time_injection = 'gate_bias',
                num_blocks = 2,
                num_neighbors_k = 4,
                building_block = 'pmamba', seq_feature_channels = [], 
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
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__(point_dim = point_dim,
            in_channel = in_channel // num_neighbors_k,
            time_channel = in_channel,
            time_injection = time_injection,
            num_blocks = num_blocks,
            building_block = building_block,
            seq_feature_channels = seq_feature_channels,
            normalization = normalization,
            num_norm_groups = num_norm_groups,
            activation = activation,
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
            mlp_hidden_ratios=mlp_hidden_ratios, 
            condition_channel = condition_channel,
            condition_injection = condition_injection,
            condition_first = condition_first,
            )

        # self.out_point_dim = point_dim
        self.num_neighbors_k = num_neighbors_k
    def forward(self, local_feature, global_feature, c = None):
        B, N, _ = local_feature.shape
        local_feature = local_feature.reshape(B, N * self.num_neighbors_k, -1)
        res = super().forward(local_feature, t = global_feature, c = c)
        return res
    
supported_skeleton_encoders = {}
supported_skeleton_encoders['SkCNN'] = (SkeletonCNNEncoder, SkeletonCNNDecoder)
supported_skeleton_encoders['SkTrans'] = (SkeletonTransEncoder, SkeletonTransDecoder)
supported_skeleton_encoders['SkMamba'] = (SkeletonMambaEncoder, SkeletonMambaDecoder)
supported_point_encoders.update(supported_skeleton_encoders)
supported_encoders.update(supported_skeleton_encoders)

supported_buildingblocks_for_encoder['SkCNN'] = ['dense', 'double_dense', 'res_dense', 'fc', 'double_fc', 'res_fc']
supported_buildingblocks_for_encoder['SkTrans'] = ['pct_sa', 'pct_oa']
supported_buildingblocks_for_encoder['SkMamba'] = ['pmamba', 'pmamba2']