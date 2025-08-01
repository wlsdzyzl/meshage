### point cloud auto-encoder, with skeleton regularization
import torch
import torch.nn as nn
import torch.nn.functional as F
from flemme.logger import get_logger
from .loss import Sphere
from flemme.model.ldm import create_diff_model

from vcg.utils import radius_inv_normalize, radius_normalize
logger = get_logger("sk_dpm")



class SkeletonDPM(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        diff_config = model_config.pop('diffusion', None)
        assert diff_config is not None, 'SkeletonDPM needs a diffusion model to generate latents.'            
        self.diff_model, self.diff_model_name = create_diff_model(diff_config)

        sphere_config = model_config.pop('sphere', {})
        self.sphere = Sphere(**sphere_config)
        self.is_conditional = self.diff_model.is_conditional
        self.is_generative = True
        self.is_supervised = False
        self.data_form = self.diff_model.data_form
    def forward(self, x, c = None):
        
        ### normalize radius (larger than 0) to range [-1, 1]
        sk, r = x[..., :3], radius_inv_normalize(x[..., 3:])
        r = radius_normalize(r)
        x = torch.cat((sk, r), dim = -1)
        # print(x.shape)
        res = self.diff_model(x, c = c)
        recon_sk, recon_r = res['recon_dpm'][..., 0:3], res['recon_dpm'][..., 3:]
        recon_r = radius_inv_normalize(recon_r)
        
        # res['recon_radius'] = recon_r
        # print(recon_sk.shape, recon_r.shape)
        res['recon_dpm_sphere'] = self.sphere.get_batch_sphere_points(recon_sk, recon_r)
        res['input_sphere'] = self.sphere.get_batch_sphere_points(sk, r)
        return res
    @property
    def device(self):
        return self.diff_model.device
    ### to visualize the skeleton
    @torch.no_grad()
    def sample(self, xt, c = None, return_processing = False, **kwargs):
        xt = self.diff_model.sample(xt, c = c, return_processing = return_processing, **kwargs)
        return xt[..., 0:3], radius_inv_normalize(xt[..., 3:])
    def __str__(self):
        _str = "********************* SkeletonDiffusion *********************\n{}"\
            .format(self.diff_model.__str__())
        return _str
    def compute_loss(self, x0, c = None):
        res = self.forward(x0, c)
        return self.diff_model.compute_loss(x0, res = res)
    def get_loss_name(self):
        return self.diff_model.get_loss_name()
    def get_input_shape(self):
        return self.diff_model.get_input_shape()
    def get_output_shape(self):
        return self.diff_model.get_output_shape()
    def get_latent_shape(self):
        return self.diff_model.get_latent_shape()
    


