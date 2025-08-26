### point cloud auto-encoder, with skeleton regularization
import torch
import torch.nn as nn
from flemme.model import AE
from flemme.logger import get_logger
from flemme.block import gather_features
from knn_cuda import KNN
from vcg.utils import truncated_value
# from flemme.model.distribution import GaussianDistribution as Gaussian
from .encoder import create_skeleton_encoder
### skeleton-regularized point cloud auto-encoder
logger = get_logger("sk_sdf")

class SkeletonSDF(AE):
    def __init__(self, model_config, create_encoder_fn=create_skeleton_encoder):
        super().__init__(model_config, create_encoder_fn)
        self.is_supervised = True
        ### should only be used for test
        self.coordinate_sampling_ratio = model_config.pop('coordinate_sampling_ratio', 1.0)
        assert self.coordinate_sampling_ratio > 0, "Coordinate sampling ratio must be positive."
        if self.coordinate_sampling_ratio < 1.0:
            self.knn = KNN(k = 1, transpose_mode=True)
    def decode(self, z, coord, c = None):
        z, c = self.parse_decoder_condition(z, c)
        if self.coordinate_sampling_ratio < 1.0:
            ske = z[..., :3]
            dist, _ = self.knn(ske, coord)
            dist = dist.squeeze(-1)
            _, idx = torch.topk(dist, k=int(self.coordinate_sampling_ratio * coord.shape[1]), dim=-1, largest = False)
            sampled_coord = gather_features(coord, index = idx, channel_dim = -1, gather_dim = 1)
            sampled_sdf = self.decoder(z, sampled_coord, c = c)
            reconed_sdf = - torch.ones((coord.shape[0], coord.shape[1], 1), device=coord.device) * truncated_value * 2
            reconed_sdf.scatter_(1, idx.unsqueeze(-1), sampled_sdf)
            return reconed_sdf, sampled_sdf, idx
        else:
            sdf = self.decoder(z, coord, c = c)
            return sdf
    def forward(self, x, coord, c=None):
        z = self.encode(x, c = c)
        res = self.decode(z, coord, c = c)
        if self.coordinate_sampling_ratio < 1.0:
            return {'recon': res[0], 
                    'latent':z, 
                    'sampled_sdf': res[1],
                    'sampled_index': res[2]}
        else:
            return {'recon': res, 'latent': z}
    def compute_loss(self, x, coord, y, c = None, res = None):
        if res is None:
            res = self.forward(x, coord, c)   
        losses = []
        pred = res['recon']
        # assert self.coordinate_sampling_ratio == 1.0, 'Coordinate sampling should only be used for test.'
        if self.coordinate_sampling_ratio < 1.0:
            y = gather_features(y, index = res['sampled_index'], channel_dim = -1, gather_dim = 1)
            pred = res['sampled_sdf']
        for loss, weight in zip(self.recon_losses, self.recon_loss_weights):
            losses.append(loss(pred, y) * weight) 
        return losses, res

