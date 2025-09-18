### point cloud auto-encoder, with skeleton regularization
import torch
import torch.nn as nn
from flemme.model import AE
from flemme.logger import get_logger
from flemme.block import gather_features
from knn_cuda import KNN
# from flemme.model.distribution import GaussianDistribution as Gaussian
from .encoder import create_skeleton_encoder
from .utils import resolution2coord
from .config import truncate_sdf, truncated_value, train_truncate_scaling, use_occupancy
import numpy as np
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
        self.resolution = model_config.pop('resolution', 0.022)
        self.skeleton_constraints = model_config.pop('skeleton_constraints', False)
        if self.skeleton_constraints:
            self.skeleton_inter_num = model_config.pop('skeleton_inter_num', 0)
            self.skeleton_knn = KNN(k = self.skeleton_inter_num + 1, transpose_mode = True)
    def decode(self, z, coord = None, c = None):
        z, c = self.parse_decoder_condition(z, c)
        if coord is None:
            coord = torch.from_numpy(np.stack((resolution2coord(self.resolution)[0], ) * z.shape[0])).to(z.device)
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
            if self.skeleton_constraints:
                ske = z[..., :3]
                if self.skeleton_inter_num > 0:
                    _, ske_inter_idx = self.skeleton_knn(ske, ske)
                    ske_inter_idx = ske_inter_idx[..., 1:]
                    inter_skes = tuple(0.5 * ske + 0.5 * gather_features(ske, index = ske_inter_idx[..., i], channel_dim = -1, gather_dim = 1) for i in range(self.skeleton_inter_num))
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
            return {'recon': sdf, 'latent': z, 'ske_inter_idx': ske_inter_idx, 'skeleton': z[..., :3]}
        return {'recon': sdf, 'latent': z, 'skeleton': z[..., :3]}
    def compute_loss(self, x, coord, y, c = None, res = None):
        if res is None:
            res = self.forward(x, coord, c)   
        losses = []
        assert self.coordinate_sampling_ratio == 1.0, 'Coordinate sampling should only be used for test.'
        if self.skeleton_constraints:
            r = res['latent'][..., 3:4]
            if self.skeleton_inter_num > 0:
                ske_inter_idx = res['ske_inter_idx']
                inter_rs = tuple(0.5 * r + 0.5 * gather_features(r, index = ske_inter_idx[..., i], channel_dim = -1, gather_dim = 1) for i in range(self.skeleton_inter_num))
                r = torch.cat((r, ) + inter_rs, dim = 1)
            r = r / self.encoder.skeletonize.radius_scaling
            if truncate_sdf:
                r = r / (truncated_value * train_truncate_scaling)
            y = torch.concat((y, r), dim = 1)
        
        if use_occupancy:
            y = (y >= 0).float()
        # if self.coordinate_sampling_ratio < 1.0:
        #     y = gather_features(y, index = res['sampled_index'], channel_dim = -1, gather_dim = 1)
        #     pred = res['sampled_sdf']
        for loss, weight in zip(self.recon_losses, self.recon_loss_weights):
            losses.append(loss(res['recon'], y) * weight) 
        return losses, res

