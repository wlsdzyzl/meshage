from vcg.sknet import SkeletonNet #, LearnableSkeletonNet
from vcg.sk_sdf import SkeletonSDF
from vcg.encoder import create_skeleton_encoder
from vcg.utils import save_sdf
from flemme.model import create_model as _create_model, EDM, LDM
from flemme.logger import get_logger
from flemme.trainer import save_data as _save_data
import torch

logger = get_logger('model.utils')
## if we want to train pcd or image, 
## make sure that the image size from data loader and image size from the model parameters are identical
device = "cuda" if torch.cuda.is_available() else "cpu"
supported_skeleton_models = {
    #### base model
    'SKNet': SkeletonNet,
    # 'LSKNet': LearnableSkeletonNet,
    'SKSDF': SkeletonSDF}
supported_flemme_models = {
    ### edm: Elucidating the Design Space of Diffusion-Based Generative Models
    'EDM': EDM,
    #### latent diffusion model: diffusion model with a pre-trained auto-encoder
    'LDM': LDM,
    }
def process_input(t):
    x, c, coord, sdf, p = None, None, None, None, None
    if len(t) == 2:
        x, p = t
    if len(t) == 3:
        x, c, p = t
    if len(t) == 4:
        x, coord, sdf, p = t 
    if len(t) == 5:
        x, c, coord, sdf, p = t
    ## patch
    # if len(t) == 5:
    #     x, y, c, si, p = t
    return x, c, coord, sdf, p

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
    if model_config['name'] in supported_skeleton_models:
        return _create_model(model_config, 
                             supported_models = supported_skeleton_models,
                             create_encoder_fn = create_skeleton_encoder)
    elif model_config['name'] in supported_flemme_models:
        return _create_model(model_config, 
                             supported_models = supported_flemme_models)
    else:
        logger.error("Model {} is not supported.".format(model_config['name']))
        exit(1)
    
def train_run(model, t, only_forward = False):
    processed_input = process_input(t)
    x, c, coord, sdf, _ = processed_input
    x = x.to(device) 
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
        tmp_res = forward_pass(model, x, coord, c)
        res.update(tmp_res)
        return res 
    else:
        ## model.compute_loss
        losses, tmp_res = compute_loss(model, x, coord, sdf, c)
        res.update(tmp_res)
        return losses, res

def test_run(model, t):
    processed_input = process_input(t)
    x, c, coord, sdf, _ = processed_input
    x = x.to(device) 
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
    tmp_res = forward_pass(model, x, coord, c)
    res.update(tmp_res)
    return res 

## save sdf to sdf volume
def save_data(output, data_form, output_path):    
    if output.shape[-1] == 1:
        output = output.squeeze(-1)
        save_sdf(output_path+'.npy', output)
    else:
        _save_data(output, data_form, output_path)