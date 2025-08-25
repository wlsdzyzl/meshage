import torch
import torch.nn as nn

from flemme.logger import get_logger
from knn_cuda import KNN
from flemme.block.pcd_utils import grouping_operation
from flemme.block import channel_transfer, channel_recover


import numpy as np
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed

logger = get_logger("vcg.sknet")

def dbscan_single_cloud_np(points_np, eps=0.05, min_samples=10):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_np)
    return clustering.labels_

def batch_dbscan(batch_points, eps=0.05, min_samples=10, n_jobs=-1):
    """
    Args:
        batch_points: torch.Tensor [B, N, 3]
    Returns:
        cluster_labels: torch.LongTensor [B, N] (label -1 = noise)
    """
    B, N, _ = batch_points.shape
    batch_np = batch_points.detach().cpu().numpy()
    results = Parallel(n_jobs=n_jobs)(delayed(dbscan_single_cloud_np)(
        batch_np[i], eps, min_samples
        ) for i in range(B))
    results = np.array(results)

    cluster_labels = torch.tensor(results, dtype=torch.long, device=batch_points.device)  # [B, N]
    return cluster_labels

class SkeletonizationBlock(nn.Module):
    def __init__(self, num_neighbor = 32, dbscan_eps = 0.75, dbscan_min_sample_num = 1):
        super().__init__()
        self.num_neighbor = num_neighbor
        self.knn = KNN(k = num_neighbor, transpose_mode=True)
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_sample_num = dbscan_min_sample_num
    def forward(self, centers, xyz):
        ### knn query
        B, N, _ = centers.shape
        dist, idx = self.knn(xyz, centers)
        # if dist[..., 0].max()>0.1:
            # exit(1)
        # neighbors: B, N, k, 3
        neighbors = channel_transfer(grouping_operation(channel_recover(xyz), idx.int()))
        idx_mask = torch.ones_like(dist)
        if self.dbscan_eps > 0:
            ## B N, K+1, 3 -> (B*N, k+1, 3)
            neighbors_with_centers = torch.concat((neighbors, centers.unsqueeze(-2)), 
                                                  dim = -2).reshape(-1, self.num_neighbor+1, 3)
            # use DB scans clustering to exclude outliers
            # (B*N, k+1)
            cluster_labels = batch_dbscan(neighbors_with_centers, 
                                          eps = self.dbscan_eps, 
                                          min_samples = self.dbscan_min_sample_num)
            # print(cluster_labels.max(dim = -1)[0])
            # (B*N, 1)
            correct_labels = cluster_labels[:, -1].unsqueeze(-1)
            idx_mask = (cluster_labels == correct_labels).reshape(B, N, self.num_neighbor+1)[..., :-1].float()

        ### use eigen decomposition to compute the radius
        neighbor_weights = idx_mask.unsqueeze(-1)
        # B, N, K, 1
        neighbor_weights = neighbor_weights / (neighbor_weights.sum(dim = -2, keepdim = True) + 1e-7) 
        if False:
            neighbor_mean = (neighbors * neighbor_weights).sum(dim = -2, keepdim = True)
            # print(neighbor_weights, neighbors)
            diff = (neighbors - neighbor_mean).reshape(-1, 3).unsqueeze(-1)
            # cov: B*N*K, 3, 3
            cov = torch.bmm(diff, diff.transpose(1, 2).contiguous())
            # cov: B, N, K, 3, 3
            cov = cov.reshape(B, N, self.num_neighbor, 3, 3)
            cov_weights = neighbor_weights.unsqueeze(-1)
            # cov: B, N, 3, 3
            cov = (cov_weights * cov).sum(dim = -3)
            evalues = torch.linalg.eigvalsh(cov)
            # lambda0 = evalues[..., 2]
            lambda1 = evalues[..., 1]
            # lambda2 = evalues[..., 0]
            radius = (lambda1**0.5).unsqueeze(-1)

            idx_mask = ((dist - 3 * radius) <= 0).float() * idx_mask
            neighbor_weights = idx_mask.unsqueeze(-1)
            neighbor_weights = neighbor_weights / (neighbor_weights.sum(dim = -2, keepdim = True) + 1e-7) 
        new_centers = (neighbors * neighbor_weights).sum(dim = -2)
        radius = (dist.unsqueeze(-1) * neighbor_weights).sum(dim = -2)
        return new_centers, radius
    