from vcg.sknet import SkeletonNet, LearnableSkeletonNet
from vcg.sk_dpm import SkeletonDPM
from vcg.sk_ae import SkeletonAE, SkeletonVAE
# from vcg.pcd_ae import TubeVAE as TuVAE
# from vcg.pcd_ldm import TubeLDPM as TuLDPM, TubeLDIM as TuLDIM 


from flemme.logger import get_logger
from flemme.utils import load_config

logger = get_logger("create_model")

supported_models = {
    #### base model
    'SKNet': SkeletonNet,
    'LSKNet': LearnableSkeletonNet,
    'SkDPM': SkeletonDPM,
    'SkAE': SkeletonAE,
    'SkVAE': SkeletonVAE
    }
    # 'TuVAE': TuVAE,
    # 'TuLDPM': TuLDPM,
    # 'TuLDIM': TuLDIM}
def create_model(model_config):
    tmpl_path = model_config.pop('template_path', None)
    if tmpl_path is not None:
        logger.info('creating model from template ...')
        model_config = load_config(tmpl_path).get('model')
        return create_model(model_config)
    
    logger.info('creating model from specific configuration ...')
    
    model_name = model_config.pop('name', 'SKNet')
    model_class = None
    if model_name in supported_models:
        model_class = supported_models[model_name]
    else:
        logger.error(f'Unsupported model class: {model_name}')
        exit(1)
    return model_class(model_config)
    ### compute loss between skeleton and surface
