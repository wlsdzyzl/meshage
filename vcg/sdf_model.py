import torch
import torch.nn as nn
from flemme.model import AE
from flemme.logger import get_logger
from flemme.block import gather_features
from knn_cuda import KNN
# from flemme.model.distribution import GaussianDistribution as Gaussian
from .encoder import create_vcg_encoder
from .utils import resolution2coord
from .config import truncate_sdf, truncated_value, train_truncate_scaling, use_occupancy
import numpy as np
logger = get_logger("sdf_model")

class SDFModel(AE):
    def __init__(self, model_config, create_encoder_fn=create_vcg_encoder):
        super().__init__(model_config, create_encoder_fn)
        self.is_supervised = True
        ### should only be used for test
        self.coordinate_sampling_ratio = model_config.pop('coordinate_sampling_ratio', 1.0)
        assert self.coordinate_sampling_ratio > 0, "Coordinate sampling ratio must be positive."
        if self.coordinate_sampling_ratio < 1.0:
            self.knn = KNN(k = 1, transpose_mode=True)
        self.resolution = model_config.pop('resolution', 0.022)
        self.skeleton_latent = hasattr(self.encoder, "skeletonize")
        self.latent_point_constraints = model_config.pop('latent_point_constraints', False)
        if self.latent_point_constraints:
            if self.skeleton_latent:
                assert self.with_radius, \
                    "with_radius need to be set as true for skeleton point constraints."
            self.latent_point_inter_num = model_config.pop('latent_point_inter_num', 0)
            self.latent_point_knn = KNN(k = self.latent_point_inter_num + 1, transpose_mode = True)
    def decode(self, z, coord = None, c = None):
        z, c = self.parse_decoder_condition(z, c)
        if coord is None:
            coord = torch.from_numpy(np.stack((resolution2coord(self.resolution)[0], ) * z.shape[0]).astype(np.float32)).to(z.device)
        if self.coordinate_sampling_ratio < 1.0:
            ske = z[..., :3]
            dist, _ = self.knn(ske, coord)
            dist = dist.squeeze(-1)
            _, idx = torch.topk(dist, k=int(self.coordinate_sampling_ratio * coord.shape[1]), dim=-1, largest = False)
            sampled_coord = gather_features(coord, index = idx, channel_dim = -1, gather_dim = 1)
            sampled_sdf = self.decoder(z, sampled_coord, c = c)
            reconed_sdf = - torch.ones((coord.shape[0], coord.shape[1], 1), device=coord.device)
            reconed_sdf.scatter_(1, idx.unsqueeze(-1), sampled_sdf)
            # return reconed_sdf, sampled_sdf, idx
            return reconed_sdf
        else:
            ske_inter_idx = None
            if self.latent_point_constraints:
                ske = z[..., :3]
                if self.latent_point_inter_num > 0:
                    _, ske_inter_idx = self.latent_point_knn(ske, ske)
                    ske_inter_idx = ske_inter_idx[..., 1:]
                    inter_skes = tuple(0.5 * ske + 0.5 * gather_features(ske, index = ske_inter_idx[..., i], channel_dim = -1, gather_dim = 1) for i in range(self.latent_point_inter_num))
                    ske = torch.cat((ske, ) + inter_skes, dim = 1)
                coord = torch.cat((coord, ske), dim = 1)
            sdf = self.decoder(z, coord, c = c)
            if ske_inter_idx is None:
                return sdf
            return sdf, ske_inter_idx
    def forward(self, x, coord, c=None):
        z = self.encode(x, c = c)
        sdf = self.decode(z, coord, c = c)
        if type(sdf) == tuple:
            sdf, ske_inter_idx = sdf
            return {'recon': sdf, 'latent': z, 'ske_inter_idx': ske_inter_idx, 'latent_points': z[..., :3]}
        return {'recon': sdf, 'latent': z, 'latent_points': z[..., :3]}
    def compute_loss(self, x, coord, y, c = None, res = None):
        if res is None:
            res = self.forward(x, coord, c)   
        losses = []
        assert self.coordinate_sampling_ratio == 1.0, 'Coordinate sampling should only be used for test.'
        if self.latent_point_constraints:
            if self.skeleton_latent:
                r = res['latent'][..., 3:4]
                if self.latent_point_inter_num > 0:
                    ske_inter_idx = res['ske_inter_idx']
                    inter_rs = tuple(0.5 * r + 0.5 * gather_features(r, index = ske_inter_idx[..., i], channel_dim = -1, gather_dim = 1) for i in range(self.latent_point_inter_num))
                    r = torch.cat((r, ) + inter_rs, dim = 1)
                r = r / self.encoder.skeletonize.radius_scaling
                if truncate_sdf:
                    r = r / (truncated_value * train_truncate_scaling)
                y = torch.concat((y, r), dim = 1)
            else:
                zero_sdf = torch.zeros((y.shape[0], res['recon'].shape[1] - y.shape[1], 1), 
                                       dtype = y.dtype, device = y.device)
                y = torch.concat((y, zero_sdf), dim = 1)
        
        if use_occupancy:
            y = (y >= 0).float()
        for loss, weight in zip(self.recon_losses, self.recon_loss_weights):
            losses.append(loss(res['recon'], y) * weight) 
        return losses, res

