### point cloud auto-encoder, with skeleton regularization
import torch
import torch.nn as nn
from flemme.loss import get_loss
from flemme.model import AE, VAE
from flemme.logger import get_logger
from flemme.model.distribution import GaussianDistribution as Gaussian
from .encoder import create_encoder
# from .sknet import SkeletonNet
# from functools import partial
### skeleton-regularized point cloud auto-encoder
logger = get_logger("sk_ae")

class SkeletonAE(AE):
    def __init__(self, model_config):
        super().__init__(model_config)
    def encode(self, x, c=None):
        return self.encoder(x, c = c)
    def decode(self, lf, gf, c = None):
        return self.decoder(lf, gf, c = c)
    def forward(self, x, c=None):
        lf, gf, sk, r = self.encode(x, c = c)
        res = self.decode(lf, gf, c = c)

        return {'recon':res, 'local_latent':lf, 
                'global_latent': gf,
                'skeleton': sk, 'radius': r}
    def compute_loss(self, x, c = None, res = None):
        if res is None:
            res = self.forward(x, c)   
        losses = []
        ### target is skeleton and radius
        # target = torch.concat((res['skeleton'], res['radius']), dim = -1)
        for loss, weight in zip(self.recon_losses, self.recon_loss_weights):
            losses.append(loss(res['recon'], x) * weight) 
        return losses, res
    def get_latent_shape(self):    
        return [self.encoder.point_num, self.encoder.out_channel,], [self.encoder.out_channel]
class SkeletonVAE(SkeletonAE):
    def __init__(self, model_config):
        super().__init__(model_config)
        distr_loss_config = model_config.get('distribution_loss', {'name':'KL'})
        distr_loss_config['reduction'] = self.loss_reduction
        self.distr_loss_name = distr_loss_config.get('name')
        self.distr_loss_weight = distr_loss_config.pop('weight', 1.0)
        self.distr_loss = get_loss(distr_loss_config, self.data_form)
        self.is_generative = True
        latent_channel = self.encoder.out_channel
        self.mean_layer = nn.Linear(latent_channel, latent_channel,)
        self.logvar_layer = nn.Linear(latent_channel, latent_channel,)
        self.mean_layer_gf = nn.Linear(latent_channel, latent_channel,)
        self.logvar_layer_gf = nn.Linear(latent_channel, latent_channel,)
    def encode(self, x, c=None):
        try:
            lf, gf, sk, r = super().encode(x, c = c)
            mean_lf, logvar_lf = self.mean_layer(lf), self.logvar_layer(lf)
            mean_gf, logvar_gf = self.mean_layer_gf(gf), self.logvar_layer_gf(gf)
            gauss_lf = Gaussian(mean = mean_lf, logvar = logvar_lf)
            gauss_gf = Gaussian(mean = mean_gf, logvar = logvar_gf)
        except Exception as e:
            logger.error(f'Parsing mean and logvar failed: {e}')
            exit(1)
        return gauss_lf, gauss_gf, sk, r
    def decode(self, lf, gf, c = None):
        return super().decode(lf, gf, c = c)
    def forward(self, x, c=None):
        gauss_lf, gauss_gf, sk, r = self.encode(x, c = c)
        lf = gauss_lf.sample()
        gf = gauss_gf.sample()
        res = self.decode(lf, gf, c = c)
        return {'recon':res, 'local_latent':lf, 
                'global_latent': gf,
                'local_gaussian': gauss_lf,
                'global_gaussian': gauss_gf,
                'skeleton': sk, 'radius': r}
    def get_loss_name(self):
         return ['local_' + self.distr_loss_name, 'global_' + self.distr_loss_name] + self.recon_loss_names
    def compute_loss(self, x, c = None, res = None):
        if res is None:
            res = self.forward(x, c)   
        ## compute the KL Divergence between N(mean, var) and N(0, 1)
        distr_loss_lf = self.distr_loss(res['local_gaussian']) * self.distr_loss_weight
        distr_loss_gf = self.distr_loss(res['global_gaussian']) * self.distr_loss_weight
        ## compute the reconstruction loss
        recon_losses = []
        # target = torch.concat((res['skeleton'], res['radius']), dim = -1)
        for loss, weight in zip(self.recon_losses, self.recon_loss_weights):
            recon_losses.append(loss(res['recon'], x) * weight) 
        # print([distr_loss_lf, distr_loss_gf, ] + recon_losses)
        return [distr_loss_lf, distr_loss_gf, ] + recon_losses, res
