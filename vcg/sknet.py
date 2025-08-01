import torch
import torch.nn as nn
import torch.nn.functional as F
from flemme.model import AE
from flemme.loss import SinkhornLoss
from flemme.logger import get_logger
from knn_cuda import KNN
from flemme.block.pcd_utils import furthest_point_sample, gather_features
from flemme.utils import batch_normalize
from .loss import Sphere, RadiusConsistencyLoss as RCLoss, SingleTopoTreeLoss as TTLoss
from .block import SkeletonizationBlock

logger = get_logger("vcg.sknet")

    
## skeleton extraction from local graph
### not learnable
class SkeletonNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.skp_num = model_config.pop('skp_num', 256)
        self.isk_num = model_config.pop('isk_num', 2)
        self.point_num = model_config.pop('point_num', 4096)
        self.skp_neighbor_num = model_config.pop('skp_neighbor_num', 32)
        self.dbscan_eps = model_config.pop('dbscan_eps', 0.75)
        self.dbscan_min_sample_num = model_config.pop('dbscan_min_sample_num', 1)
        self.sklnz = SkeletonizationBlock(num_neighbor=self.skp_neighbor_num,
                                dbscan_eps = self.dbscan_eps,
                                dbscan_min_sample_num = self.dbscan_min_sample_num)
        self.skp_neighbor_num_2nd = int(self.skp_neighbor_num * self.skp_num * 4/ self.point_num )
        # print(self.skp_neighbor_num, self.skp_neighbor_num_2nd)
        self.sklnz2 = SkeletonizationBlock(num_neighbor=self.skp_neighbor_num_2nd)
        sphere_config = model_config.pop('sphere', {})
        self.sphere = Sphere(**sphere_config)
        self.radius_scaling = model_config.pop('radius_scaling', 1.5)
        self.is_conditional = False 
        self.is_supervised = False 
        self.is_generative = False
    def forward(self, xyz, **kwargs):
 
        ### first skeletonization
        sample_ids = furthest_point_sample(xyz, self.skp_num * 4)
        centers = gather_features(xyz, index = sample_ids, 
            channel_dim = -1, gather_dim = 1)
        for _ in range(self.isk_num):
            centers, radius = self.sklnz(centers, xyz)
               

        ### second skeletonization       
        xyz = centers  
        sample_ids = furthest_point_sample(xyz, self.skp_num)
        centers = gather_features(xyz, index = sample_ids, 
            channel_dim = -1, gather_dim = 1)
        radius = gather_features(radius, index = sample_ids, 
            channel_dim = -1, gather_dim = 1)
        for _ in range(self.isk_num):
            centers, _radius = self.sklnz2(centers, xyz)  
        radius = radius + _radius

        radius = radius * self.radius_scaling
        sphere_points = self.sphere.get_batch_sphere_points(centers, radius)
        res = {
            'recon_skeleton': centers,
            'recon_sphere': sphere_points,
            'radius': radius
        }
        return res
    
# class SkeletonGraphNet(SkeletonNet):
    
## skeleton extraction from local graph
class LearnableSkeletonNet(AE):
    def __init__(self, model_config):
        self.skp_num = model_config.pop('skp_num', 256)
        self.isk_num = model_config.pop('isk_num', 2)
        self.skp_neighbor_num = model_config.pop('skp_neighbor_num', 32)
        self.dbscan_eps = model_config.pop('dbscan_eps', 0.75)
        self.dbscan_min_sample_num = model_config.pop('dbscan_min_sample_num', 1)
        self.ncc = model_config.pop('num_connected_components', 1)
        sphere_config = model_config.pop('sphere', {})
        self.sphere = Sphere(**sphere_config)
        self.radius_scaling = model_config.pop('radius_scaling', 1.5)
        super().__init__(model_config)
        self.is_supervised = False
        self.knn = KNN(k = int(self.skp_neighbor_num), transpose_mode=True)
        self.sklnz = SkeletonizationBlock(num_neighbor=self.skp_neighbor_num,
                                dbscan_eps = self.dbscan_eps,
                                dbscan_min_sample_num = self.dbscan_min_sample_num)

        rc_config = model_config.get('radius_consistency_loss', {})
        rc_config['reduction'] = self.loss_reduction
        self.rc_weight = rc_config.pop('weight', 0.1)
        self.rc_loss = RCLoss(**rc_config)
        

        tt_config = model_config.get('topology_tree_loss', {})
        tt_config['reduction'] = self.loss_reduction
        # tt_config['num_connected_components'] = self.ncc
        self.tt_weight = tt_config.pop('weight', 0.1)
        self.tt_loss = TTLoss(**tt_config)
        

    def forward(self, xyz, c = None, **kwargs):

        ### get a good initialization
        ncc_xyzs = torch.chunk(xyz, dim = 1, chunks = self.ncc)
        res = {}
        ncc_nxyzs = []
        ncc_means = []
        ncc_scalings = []
        ncc_weights = []
        ncc_centers = []
        ncc_init_centers = []
        ncc_radius = []
        ncc_sphere_points = []
        
        for nxyz in ncc_xyzs:
            nxyz, (nmean, nscaling) = batch_normalize(nxyz, channel_dim = -1, return_transform = True)
            dists, _ = self.knn(nxyz, nxyz)
            nxyz_weights = F.softmax(dists.mean(dim = -1), dim = -1)
            

            sample_ids = furthest_point_sample(nxyz, self.skp_num // self.ncc)
            centers = gather_features(nxyz, index = sample_ids, 
                channel_dim = -1, gather_dim = 1)
            for _ in range(self.isk_num):
                centers, radius = self.sklnz(centers, nxyz)
            
            init_centers = centers
            
            centers = super().forward(centers, c = c)['recon']
            
            radius = radius.detach()
            radius = radius * self.radius_scaling
            sphere_points = self.sphere.get_batch_sphere_points(centers, radius)

            ncc_nxyzs.append(nxyz)
            ncc_means.append(nmean)
            ncc_scalings.append(nscaling)
            ncc_weights.append(nxyz_weights)
            ncc_centers.append(centers)
            ncc_radius.append(radius)
            ncc_sphere_points.append(sphere_points)
            ncc_init_centers.append(init_centers)
        # print(len(ncc_init_centers), len(ncc_means) )
        res = {'init_skeleton': torch.cat([ c / s + m \
                    for c, m, s in zip(ncc_init_centers, ncc_means, ncc_scalings)], dim = 1),
            'recon_skeleton': torch.cat([ c / s + m \
                    for c, m, s in zip(ncc_centers, ncc_means, ncc_scalings)], dim = 1),
            'recon_sphere': torch.cat([ c / s + m \
                    for c, m, s in zip(ncc_sphere_points, ncc_means, ncc_scalings)], dim = 1),
            'radius': torch.cat([ c / s \
                    for c, s in zip(ncc_radius, ncc_scalings)], dim = 1),
            'ncc_point_weights': ncc_weights,
            'ncc_nxyzs': ncc_nxyzs,
            'ncc_recon_skeletons': ncc_centers,
            'ncc_recon_spheres': ncc_sphere_points,
            'ncc_radius': ncc_radius
            }
        return res

    def get_loss_name(self):
        return self.recon_loss_names + ['radius_consistency_loss', 'topo_tree_loss']

    def compute_loss(self, x, c = None, res = None, pretrain = False, **kwargs):
        if res is None:
            res = self.forward(x, c = c)
        all_losses = []
        for ni in range(self.ncc):
            losses = []
            for loss, weight in zip(self.recon_losses, self.recon_loss_weights):
                if isinstance(loss, SinkhornLoss):
                    losses.append(loss(res['ncc_recon_spheres'][ni], res['ncc_nxyzs'][ni], y_weight = res['ncc_point_weights'][ni]) * weight )    
                else:
                    losses.append(loss(res['ncc_recon_spheres'][ni], res['ncc_nxyzs'][ni]) * weight)
            ### losses after pretraining
            if not pretrain:
                rc_loss = self.rc_loss(centers = res['ncc_recon_skeletons'][ni], 
                                    radius = res['ncc_radius'][ni],
                                    xyz = res['ncc_nxyzs'][ni], 
                                    sphere_points = res['ncc_recon_spheres'][ni]) * self.rc_weight 
                tt_loss = self.tt_loss(res['ncc_recon_skeletons'][ni]) * self.tt_weight 
                losses = losses + [rc_loss, tt_loss]
            all_losses.append(losses)

        for li in range(len(all_losses[0])):
            for ni in range(1, self.ncc):
                all_losses[0][li] += all_losses[ni][li]
        all_losses = [ l / self.ncc for l in all_losses[0]]
        return all_losses, res
    