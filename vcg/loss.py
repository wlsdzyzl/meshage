import torch
import torch.nn as nn

from knn_cuda import KNN
from flemme.block.pcd_utils import grouping_operation
from flemme.block import channel_transfer, channel_recover
from flemme.encoder.point.sphere3d import icosphere, uvsphere
from flemme.logger import get_logger
from topologylayer.nn import AlphaLayer, BarcodePolyFeature
from flemme.loss.ext_modules import ChamferDistance
from functools import partial
from flemme.utils import DataForm
from flemme.loss import get_loss as get_flemme_loss


logger = get_logger("vcg.loss")
class Sphere:
    def __init__(self, dim = 3, sphere_n = 16, 
        method = 'online_random', 
        **kwargs):
        self.dim = dim
        self.sphere_n = sphere_n
        if self.dim == 2:
            if method == 'online_random':
                self.local_coor = None
            else:
                # uniform
                step = 2 * np.pi / sphere_n
                theta_vec =  np.arange(step / 2, 2 * np.pi, step)
                self.local_coor = torch.from_numpy(np.array([np.cos(theta_vec),np.sin(theta_vec)]).T)
        elif self.dim == 3:
            # icosphere
            if method == 'icosphere':
                self.local_coor = torch.from_numpy(icosphere(sphere_n))
            elif method == 'uvsphere':
                self.local_coor = torch.from_numpy(uvsphere(sphere_n) )
            elif method == 'online_random':
                self.local_coor = None
            else:
                logger.error("Unsupported method for sphere sampling, should be one of ['uvsphere', 'icosphere', 'online_random'].")
                exit(1)
            logger.info(f'Using \'{method}\' to sample sphere points.')
        if self.local_coor is not None:
            self.sphere_n = self.local_coor.shape[0]
        logger.info(f'The number of sphere points is {self.sphere_n}')
      
    def get_batch_sphere_points(self, skeleton, radius, is_surface = True):
        # skeleton: B, N_k, D
        # radius: B, N_k, 1
        # surface: B, N, D
        local_coor = self.local_coor
        if local_coor is None:
            local_coor = torch.randn(self.sphere_n, self.dim, device = skeleton.device).type(skeleton.dtype)
            local_coor =  local_coor / local_coor.norm(dim = -1, keepdim = True)
        else:
          local_coor = local_coor.to(skeleton.device).type(skeleton.dtype)
        if not is_surface:
            local_coor = local_coor * torch.rand(self.sphere_n, 1)
        B, skeleton_n = skeleton.shape[0], skeleton.shape[1]
        ### term1: construct distance between sphere points to surface 
        sphere_points = local_coor.repeat(B, skeleton_n, 1) * radius.repeat_interleave(self.sphere_n, dim = 1)\
                    + skeleton.repeat_interleave(self.sphere_n, dim = 1)
        return sphere_points


### based on eigen value
class EigenRatioPerPoints(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        self.k = k
        self.knn = KNN(k = self.k, transpose_mode=True)
    def forward(self, x):
        # x: B N 3
        # idx: B N k
        x = x[..., :3]
        B, N, _ = x.shape
        _, idx = self.knn(x, x)
        # neighbors: B, N, k, 3
        neighbors = channel_transfer(grouping_operation(channel_recover(x), idx.int()))
        neighbor_mean = torch.mean(neighbors, dim = 2, keepdim = True)
        # B*N*k, 3, 1
        diff = (neighbors - neighbor_mean).reshape(-1, 3).unsqueeze(-1)
        # cov: B*N*K, 3, 3
        cov = torch.bmm(diff, diff.transpose(1, 2).contiguous())
        # conv: B, N, 3, 3
        cov = cov.reshape(B, N, self.k, 3, 3).mean(dim=2)
        evalues = torch.linalg.eigvalsh(cov)
        lambda0 = evalues[..., 2]
        lambda1 = evalues[..., 1]
        # ratio: B, N
        ratio = lambda0 / lambda1
        return ratio
    
class MSEEigenRatioLoss(nn.Module):
    def __init__(self, k = 16, reduction = 'mean'):
        super().__init__()
        self.eigen_ratio = EigenRatioPerPoints(k)
        self.mse = partial(nn.functional.mse_loss, reduction = 'none')
        self.reduction = reduction
    def forward(self, x, y):
        er1 = self.eigen_ratio(x)
        er2 = self.eigen_ratio(y)
        # print(er1, er2)
        res = self.mse(er1, er2)
        res = res.mean(dim = -1)
        if self.reduction == 'mean':
            return res.mean()
        elif self.reduction == 'sum':
            return res.sum()
        return res

class ChamferEigenRatioLoss(nn.Module):
    def __init__(self, k = 16, reduction = 'mean', extended = False):
        super().__init__()
        self.eigen_ratio = EigenRatioPerPoints(k)
        self.chamfer = ChamferDistance()
        self.mse = partial(nn.functional.mse_loss, reduction = 'none')
        self.reduction = reduction
        self.extended = extended
    def forward(self, x, y):
        # (B, N) (B, m)
        dist1, dist2, idx1, idx2 = self.chamfer(x, y, return_idx=True)
        er1 = self.eigen_ratio(x)
        er2 = self.eigen_ratio(y)

        corr_er1 = torch.gather(er2, dim = -1, index = idx1.long())
        corr_er2 = torch.gather(er1, dim = -1, index = idx2.long())
        er_d1 = self.mse(er1, corr_er1)
        er_d2 = self.mse(er2, corr_er2)
        er_d1, er_d2 = er_d1.mean(dim = -1, keepdim = True), er_d2.mean(dim = -1, keepdim = True)
        if self.extended:
            dist, _ = torch.max(torch.cat([er_d1, er_d2], dim = -1), dim = -1)
        else:
            dist = ((er_d1 + er_d2) * 0.5).squeeze(-1)
        if self.reduction == 'mean':
            return dist.mean()
        elif self.reduction == 'sum':
            return dist.sum()
        return dist
    
class RadiusConsistencyLoss(nn.Module):
    def __init__(self, reduction = 'mean',
                 is_surface = False):
        super().__init__()
        self.reduction = reduction
        self.is_surface = is_surface
    def forward(self, centers, radius, xyz, sphere_points = None):

        # distance_matrix: B N K
        B, NC, _ = centers.shape
        distance_matrix = torch.cdist(xyz, centers)
        cd_p_to_c, cd_p_to_c_idx = torch.min(distance_matrix, dim = -1, keepdim = True)
        cd_c_to_p = torch.min(distance_matrix, dim = 1)[0]
        radius_from_p = torch.gather(radius, dim = 1, index = cd_p_to_c_idx)

        if self.is_surface:
            radius_difference_c_to_p = (radius - cd_c_to_p.unsqueeze(-1)).squeeze(-1).abs().mean(dim = 1)
            radius_difference_p_to_c = (radius_from_p - cd_p_to_c).squeeze(-1).abs().mean(dim = 1)
        else:
            radius_difference_p_to_c = (radius_from_p - cd_p_to_c).squeeze(-1)
            radius_difference_p_to_c[radius_difference_p_to_c > 0] = 0
            radius_difference_p_to_c = radius_difference_p_to_c.abs().mean(dim = 1)
            distance_matrix_sphere_to_sample = torch.cdist(xyz, sphere_points)
            cd_sp_to_p = torch.min(distance_matrix_sphere_to_sample, dim = 1)[0]
            max_cd_sp_to_p = torch.max(cd_sp_to_p.reshape(B, NC, -1), dim = -1)[0]            
            radius_difference_c_to_p = max_cd_sp_to_p.abs().mean(dim = 1) 

        res = radius_difference_c_to_p + radius_difference_p_to_c

        if self.reduction == "mean":
            res = res.mean()
        if self.reduction == 'sum':
            res = res.sum()
        return res

class SingleTopoTreeLoss(nn.Module):
    def __init__(self, reduction = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.layer = AlphaLayer(maxdim=1)
        self.func = BarcodePolyFeature(0,2,0)

    def forward(self, r):
        res = torch.stack([self.func(self.layer(ri)) for ri in r])
        if self.reduction == "mean":
            res = res.mean()
        if self.reduction == 'sum':
            res = res.sum()
        return res


class MultipleTopoTreeLoss(nn.Module):
    def __init__(self, reduction = 'mean', num_connected_components = 2):
        super().__init__()
        self.reduction = reduction
        self.layer = AlphaLayer(maxdim=1)
        self.func = BarcodePolyFeature(0,2,0)
        self.ncc = num_connected_components
    def forward(self, r):
        nr = torch.chunk(r, dim = 1, chunks = self.ncc)
        n_res = []
        for tmp_r in nr: 
            tmp_r = batch_normalize(tmp_r, channel_dim = -1)
            res = torch.stack([self.func(self.layer(ri)) for ri in tmp_r])
            if self.reduction == "mean":
                res = res.mean()
            if self.reduction == 'sum':
                res = res.sum()
            n_res.append(res)
        return sum(n_res) / len(n_res)

# class IsoForceLoss(nn.Module):
#     def __init__(self, k = 2, reduction = 'mean'):
#         super().__init__()
#         self.k = k
#         self.knn = KNN(k = self.k, transpose_mode=True)
#         self.reduction = reduction
#     def forward(self, x):
#         # x: B N 3
#         # idx: B N k
#         _, idx = self.knn(x, x)
#         # neighbors: B, N, k, 3
#         neighbors = channel_transfer(grouping_operation(channel_recover(x), idx.int()))
#         neighbors_sub_center = neighbors - x.unsqueeze(-2)
#         sum_directions = neighbors_sub_center.sum(dim = -2)
#         res = sum_directions.norm(p = 2, dim = -1).mean(dim = -1)
        
#         if self.reduction == 'mean':
#             res = res.mean()
#         if self.reduction == 'sum':
#             res = res.sum()
#         return res
# class RadiusSwellingLoss(nn.Module):
#     def __init__(self, reduction = 'mean'):
#         super().__init__()
#         self.reduction = reduction
#     def forward(self, r):
#         res = r.squeeze(-1).mean(dim=1)
#         if self.reduction == "mean":
#             res = res.mean()
#         if self.reduction == 'sum':
#             res = res.sum()
#         return -res
def get_loss(loss_config):
    loss_name = loss_config.pop('name', None)
    if loss_name == 'Radius':
        return RadiusConsistencyLoss(**loss_config)
    elif loss_name == 'TopoTree':
        return SingleTopoTreeLoss(**loss_config)
    elif loss_name == 'MultipleTopoTree':
        return MultipleTopoTreeLoss(**loss_config)
    elif loss_name == 'ChamferER':
        return ChamferEigenRatioLoss(**loss_config)
    elif loss_name == 'MSEER':
        return MSEEigenRatioLoss(**loss_config)
    else:
        loss_config['name'] = loss_name
        return get_flemme_loss(loss_config, data_form = DataForm.PCD)
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(16, 1024, 3).to(device)
    y = torch.randn(16, 1024, 3).to(device)
    eloss = MSEEigenRatioLoss()
    closs = ChamferEigenRatioLoss()
    print(eloss(x, y))
    print(closs(x, y))