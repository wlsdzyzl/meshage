### point cloud auto-encoder, with skeleton regularization
import torch
import torch.nn as nn
from flemme.model import AE
from flemme.logger import get_logger
from flemme.utils import onehot_to_label
# from flemme.model.distribution import GaussianDistribution as Gaussian
from .encoder import create_skeleton_encoder
### skeleton-regularized point cloud auto-encoder
logger = get_logger("sk_sdf")

class SkeletonSDF(AE):
    def __init__(self, model_config, create_encoder_fn=create_skeleton_encoder):
        super().__init__(model_config, create_encoder_fn)
        self.is_supervised = True
    # def encode(self, x, c = None):
    #     x, c = self.parse_encoder_condition(x, c)
    #     return self.encoder(x, c = c)
    def decode(self, z, coord, c = None):
        z, c = self.parse_decoder_condition(z, c)
        return self.decoder(z, coord, c = c)
    def forward(self, x, coord, c=None):
        z = self.encode(x, c = c)
        res = self.decode(z, coord, c = c)
        return {'sdf':res, 'latent':z}
    def compute_loss(self, x, coord, y, c = None, res = None):
        if res is None:
            res = self.forward(x, coord, c)   
        losses = []
        for loss, weight in zip(self.recon_losses, self.recon_loss_weights):
            losses.append(loss(res['sdf'], y) * weight) 
        return losses, res

