from vcg.sknet import SkeletonNet, LearnableSkeletonNet
from vcg.sk_dpm import SkeletonDPM
from vcg.sk_ae import SkeletonSDF
from vcg.encoder import create_skeleton_encoder
from flemme.model import create_model as _create_model, supported_models
from functools import partial

supported_skeleton_models = {
    #### base model
    'SKNet': SkeletonNet,
    'LSKNet': LearnableSkeletonNet,
    'SKSDF': SkeletonSDF
    }

supported_models.update(supported_skeleton_models)
create_model = partial(_create_model, create_encoder_fn = create_skeleton_encoder)
