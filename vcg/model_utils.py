from vcg.sknet import SkeletonNet #, LearnableSkeletonNet
from vcg.sdf_model import SDFModel
from vcg.encoder import create_vcg_encoder
from vcg.utils import save_sdf2mesh, save_occ2mesh
from vcg.config import truncated_value, train_truncate_scaling, use_occupancy
from flemme.model import create_model as _create_model, EDM, LDM, supported_ae_models
from flemme.logger import get_logger
from flemme.trainer import save_data as _save_data
from flemme.utils import save_npy, DataForm
import torch
import numpy as np

logger = get_logger('model.utils')
## if we want to train pcd or image, 
## make sure that the image size from data loader and image size from the model parameters are identical
device = "cuda" if torch.cuda.is_available() else "cpu"
supported_vcg_models = {
    #### base model
    'SKNet': SkeletonNet,
    # 'LSKNet': LearnableSkeletonNet,
    'SDF': SDFModel}
supported_flemme_models = ['Base', 'EDM', 'LDM']
supported_ae_models.append('SDF')
def process_input(t):
    x, ske, c, coord, sdf, p = None, None, None, None, None, None
    if len(t) == 2:
        x, p = t
    if len(t) == 3:
        x, c, p = t
    if len(t) == 5:
        x, ske, coord, sdf, p = t 
    if len(t) == 6:
        x, ske, c, coord, sdf, p = t
    return x, ske, c, coord, sdf, p

def compute_loss(model, x, coord, sdf, c, **kwargs):
    ## pointsdf with label
    if model.is_conditional and model.is_supervised:
        losses, res = model.compute_loss(x, coord = coord, y = sdf, c = c, **kwargs)
    ## edm model with condition
    elif model.is_conditional:
        losses, res = model.compute_loss(x, c = c, **kwargs)
    ## pointsdf
    elif model.is_supervised:
        losses, res = model.compute_loss(x, coord = coord, y = sdf, **kwargs)
    ## skeletonnet and edm
    else:
        losses, res = model.compute_loss(x, **kwargs)
    return losses, res
def forward_pass(model, x, coord, c, **kwargs):
    ## pointsdf with label
    if model.is_conditional and model.is_supervised:
        res = model(x, coord = coord, c = c, **kwargs)
    ## edm model with condition
    elif model.is_conditional:
        res = model(x, c = c, **kwargs)
    ## pointsdf
    elif model.is_supervised:
        res = model(x, coord = coord, **kwargs)
    ## skeletonnet and edm
    else:
        res = model(x, **kwargs)
    return res


def create_model(model_config):
    model_name = model_config.get('name', 'Base')
    if model_name in supported_vcg_models:
        return _create_model(model_config, 
                             supported_underlying_models = supported_vcg_models,
                             create_encoder_fn = create_vcg_encoder)
    elif model_name in supported_flemme_models:
        return _create_model(model_config, create_model_fn = create_model)
    else:
        logger.error(f'Unsupported model class: {model_name}, should be one of {list(supported_vcg_models.keys()) + supported_flemme_models}')
        exit(1)
    
def train_run(model, t, only_forward = False):
    processed_input = process_input(t)
    x, ske, c, coord, sdf, _ = processed_input
    x = x.to(device) 
    if ske is not None:
        ske = ske.to(device)
    if c is not None: 
        c = c.to(device)
    if coord is not None:
        coord = coord.to(device)
    if sdf is not None:
        sdf = sdf.to(device)

    if not x.shape[1:] == tuple(model.get_input_shape()):
        logger.error("Inconsistent sample shape between data and model: {} and {}".format(x.shape[1:], tuple(model.get_input_shape())))
        exit(1)  
    res = {'input': x, }
    if model.is_supervised:
        res['target'] = sdf
    if model.is_conditional:
        res['condition'] = c
    ### here we want to generate raw image
    if only_forward:
        ## model.forward
        tmp_res = forward_pass(model, x, coord, c, ske = ske)
        res.update(tmp_res)
        return res 
    else:
        ## model.compute_loss
        losses, tmp_res = compute_loss(model, x, coord, sdf, c, ske = ske)
        res.update(tmp_res)
        return losses, res

def test_run(model, t):
    processed_input = process_input(t)
    x, ske, c, coord, sdf, path = processed_input
    x = x.to(device) 
    if ske is not None:
        ske = ske.to(device)
    if c is not None: 
        c = c.to(device)
    if coord is not None:
        coord = coord.to(device)
    if sdf is not None:
        sdf = sdf.to(device)
        
    if not x.shape[1:] == tuple(model.get_input_shape()):
        logger.error("Inconsistent sample shape between data and model: {} and {}".format(x.shape[1:], tuple(model.get_input_shape())))
        exit(1)  
    res = forward_pass(model, x, coord, c, ske = ske)
    res['input'] = x
    res['path'] = path
    if model.is_supervised:
        res['target'] = sdf
    if model.is_conditional:
        res['condition'] = c
    return res 

## save sdf to sdf volume
def save_data(output, data_form, output_path):  
    if output.shape[-1] == 1:
        output = output.squeeze(-1)
        length = int(np.cbrt(output.shape[0]))
        output = output.reshape((length, length, length))
        if use_occupancy:
            output = output > 0
            save_occ2mesh(output_path+'.ply', output)
        else:
            output = output * truncated_value * train_truncate_scaling
            save_sdf2mesh(output_path+'.ply', output)
    else:
        _save_data(output, data_form, output_path)