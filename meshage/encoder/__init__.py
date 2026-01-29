from .skeleton import SkeletonSDFDecoder, SkeletonCNNEncoder, \
    SkeletonTransEncoder, SkeletonMambaEncoder
from .sparse_point import SparsePointSDFDecoder, SparsePointCNNEncoder, \
    SparsePointTransEncoder, SparsePointMambaEncoder
from .sksp import SKSPCNNEncoder, SKSPTransEncoder, SKSPMambaEncoder
from flemme.utils import DataForm
from flemme.logger import get_logger

### skeleton-regularized point cloud auto-encoder
logger = get_logger("meshage_encoder")

supported_meshage_encoders = {}
supported_meshage_encoders['SKSDFCNN'] = (SkeletonCNNEncoder, SkeletonSDFDecoder)
supported_meshage_encoders['SKSDFTrans'] = (SkeletonTransEncoder, SkeletonSDFDecoder)
supported_meshage_encoders['SKSDFMamba'] = (SkeletonMambaEncoder, SkeletonSDFDecoder)
supported_meshage_encoders['SPSDFCNN'] = (SparsePointCNNEncoder, SparsePointSDFDecoder)
supported_meshage_encoders['SPSDFTrans'] = (SparsePointTransEncoder, SparsePointSDFDecoder)
supported_meshage_encoders['SPSDFMamba'] = (SparsePointMambaEncoder, SparsePointSDFDecoder)
supported_meshage_encoders['SKSPSDFCNN'] = (SKSPCNNEncoder, SkeletonSDFDecoder)
supported_meshage_encoders['SKSPSDFTrans'] = (SKSPTransEncoder, SkeletonSDFDecoder)
supported_meshage_encoders['SKSPSDFMamba'] = (SKSPMambaEncoder, SkeletonSDFDecoder)

supported_flemme_encoders = ['SeqNet', 'SeqTrans', 'SeqMamba']

supported_buildingblocks_for_encoder = {}
supported_buildingblocks_for_encoder['SKSDFCNN'] = ['dense', 'double_dense', 'res_dense', 'fc', 'double_fc', 'res_fc']
supported_buildingblocks_for_encoder['SKSDFTrans'] = ['pct_sa', 'pct_oa']
supported_buildingblocks_for_encoder['SKSDFMamba'] = ['pmamba', 'pmamba2']

supported_buildingblocks_for_encoder['SPSDFCNN'] = ['dense', 'double_dense', 'res_dense', 'fc', 'double_fc', 'res_fc']
supported_buildingblocks_for_encoder['SPSDFTrans'] = ['pct_sa', 'pct_oa']
supported_buildingblocks_for_encoder['SPSDFMamba'] = ['pmamba', 'pmamba2']

supported_buildingblocks_for_encoder['SKSPSDFCNN'] = ['dense', 'double_dense', 'res_dense', 'fc', 'double_fc', 'res_fc']
supported_buildingblocks_for_encoder['SKSPSDFTrans'] = ['pct_sa', 'pct_oa']
supported_buildingblocks_for_encoder['SKSPSDFMamba'] = ['pmamba', 'pmamba2']

def create_meshage_encoder(encoder_config, return_encoder = True, return_decoder = True):
        encoder_name = encoder_config.pop('name')
        if not encoder_name in supported_meshage_encoders:
            logger.error(f'Unsupported encoder: {encoder_name}, should be one of {supported_meshage_encoders.keys()}')
            exit(1)        
        Encoder, Decoder = supported_meshage_encoders[encoder_name]
        building_block = encoder_config.pop('building_block', 'single')
        ### all encoders correspond to the same decoder whose building block could be different from encoder.
        ### therefore we especially assign a building block for decoder during the reconstruction.
        decoder_building_block = encoder_config.pop('decoder_building_block', 'dense')
        if not building_block in supported_buildingblocks_for_encoder[encoder_name]:
            ### pointnet decoder doesn't need building block
            logger.error(f'Unsupported building block \'{building_block}\' for encoder {encoder_name}, please use one of {supported_buildingblocks_for_encoder[encoder_name]}.')
            exit(1)
        if not decoder_building_block in supported_buildingblocks_for_encoder['SKSDFCNN']:
            ### pointnet decoder doesn't need building block
            logger.error(f'Unsupported building block \'{decoder_building_block}\' for decoder, please use one of {supported_buildingblocks_for_encoder["SKSDFCNN"]}.')
            exit(1)
        in_channel = encoder_config.pop('in_channel', 3)
        out_channel = encoder_config.pop('out_channel', 1)

        encoder, decoder = None, None
        logger.info('Model is constructed for point cloud.')
        #### point cloud encoder
        point_num = encoder_config.pop('point_num', 2560)
        voxel_resolutions = encoder_config.pop('voxel_resolutions', [])
        voxel_attens = encoder_config.pop('voxel_attens', None)
        dense_channels = encoder_config.pop('dense_channels', [256, 256])
        assert type(voxel_resolutions) == list, "voxel_resolutions should be a list."
        assert type(dense_channels) == list, "dense_channels should be a list."
        if not isinstance(voxel_attens, list): 
            voxel_attens = [voxel_attens,] * len(voxel_resolutions)
        if not return_encoder:
            latent_channel = encoder_config.pop('latent_channel', None)
            num_latent_points = encoder_config.pop('num_latent_points', 256)
            assert latent_channel is not None, "Input channel of decoder (latent_channel) is not specified."
        ## 0: without using local graph
        local_feature_channels = encoder_config.pop('local_feature_channels', [64, 128, 256])  
        assert isinstance(local_feature_channels, list), 'feature channels should be a list.'
        seq_feature_channels = encoder_config.pop('seq_feature_channels', [512, 512, 512])
        assert isinstance(seq_feature_channels, list), 'feature channels should be a list.'
        if return_encoder:
            encoder = Encoder(point_dim=in_channel, 
                            point_num=point_num,
                            local_feature_channels=local_feature_channels, 
                            dense_channels=dense_channels, 
                            building_block=building_block,
                            voxel_resolutions = voxel_resolutions,
                            voxel_attens = voxel_attens,
                            **encoder_config)
            latent_channel = encoder.out_channel
            num_latent_points = encoder.num_latent_points
            encoder.data_form = DataForm.PCD
            encoder.channel_dim = -1
            encoder.feature_channel_dim = -1
            encoder.point_num = point_num
        if return_decoder:
            decoder = Decoder(point_dim=out_channel, 
                                        latent_channel = latent_channel,
                                        num_latent_points = num_latent_points,
                                        seq_feature_channels = seq_feature_channels,
                                        building_block=decoder_building_block,
                                        **encoder_config)
        return encoder, decoder