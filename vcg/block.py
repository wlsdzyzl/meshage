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
        return new_centers, radius
    
class SkeletonizingAndGroupingLayer(nn.Module):
    # in_channel: in_channel of input features
    # out_channel: out_channel of output featuress
    def __init__(self, in_channel, out_channels, 
            num_fps_points, k, 
            BuildingBlock, radius = 0.1, num_blocks = 2, 
            hidden_channels = None, use_xyz = True, 
            sorted_query = False,
            knn_query = False,
            pos_embedding_channel = 3):
        super().__init__()
        self.in_channel = in_channel
        self.out_channels = out_channels or self.in_channel
        self.num_fps_points = num_fps_points
        self.k = k
        self.radius = radius

        if not isinstance(self.out_channels, list):
            self.out_channels = [self.out_channels, ]
        if not isinstance(self.k, list):
            self.k = [self.k, ] * len(self.out_channels)
        if not isinstance(self.radius, list):
            self.radius = [self.radius, ]* len(self.out_channels)
        if not isinstance(num_blocks, list):
            num_blocks = [num_blocks, ]* len(self.out_channels)
        if not (isinstance(hidden_channels, list) and \
            len(hidden_channels) > 0 and  \
            isinstance(hidden_channels[0], list)):
            hidden_channels = [hidden_channels, ] * len(self.out_channels)

        assert len(self.out_channels) == len(self.k) and \
            len(self.out_channels) == len(self.radius) and \
            len(self.out_channels) == len(num_blocks) and \
            len(self.out_channels) == len(hidden_channels), 'The numbers of scales inferred from different parameters are not identical.'
        
        if use_xyz:
                self.in_channel += pos_embedding_channel
        ### real out_channel
        self.out_channel = sum(self.out_channels)
        is_seq = BuildingBlock.func.is_sequence_modeling()        
        
        self.groupers = []

        self.bb = nn.ModuleList()
        for sid in range(len(self.out_channels)):
            self.groupers.append(QueryAndGroup(self.k[sid], 
                        radius = self.radius[sid], 
                        use_xyz=use_xyz, 
                        sorted_query = sorted_query,
                        knn_query = knn_query) if self.num_fps_points > 0 
                        else GroupAll(use_xyz))
            if not is_seq:
                self.bb.append(MultipleBuildingBlocks(in_channel = self.in_channel, 
                        out_channel = self.out_channels[sid], 
                        hidden_channels = hidden_channels[sid],
                        n = num_blocks[sid],
                        BuildingBlock = BuildingBlock))
            else:
                self.bb.append(MultipleBuildingBlocks(in_channel = self.in_channel,
                        out_channel = self.out_channels[sid],
                        n = num_blocks[sid],
                        hidden_channels = hidden_channels[sid],
                        BuildingBlock = partial(
                            GroupSeqModelingLayer,
                            BuildingBlock = BuildingBlock),
                        ))
    def forward(self, xyz, xyz_embed, features = None, t = None):
        r"""
        Parameters
        ----------
        xyz : (B, N, 3) tensor of the xyz coordinates of the features
        features: (B, N, C_in) tensor of the descriptors of the the features

        Returns
        -------
        centers: # (B, M, 3)
        center_features : torch.Tensor
            (B, M, C_out) tensor of the center_features descriptors
        """
        centers = None
        center_embed = None
        features_trans = channel_recover(features) if features is not None else None
        center_features_trans = None
        sample_ids = None
        if self.num_fps_points > 0:
            sample_ids = furthest_point_sample(xyz, self.num_fps_points)
            centers = gather_features(xyz, index = sample_ids, 
                channel_dim = -1, gather_dim = 1)
            center_embed = gather_features(xyz_embed, index = sample_ids, 
                channel_dim = -1, gather_dim = 1)
            
            if features_trans is not None:
                center_features_trans = gather_features(features_trans, 
                    index = sample_ids, channel_dim = 1,
                    gather_dim = -1)
        
        center_feature_list = []
        for gid in range(len(self.groupers)):
            ## (B, N, 3), (B, M, 3), (B, N, C_in) -> (B, C_in (+3), M, k) 
            grouped_features_trans = self.groupers[gid](
                xyz, xyz_embed, centers, center_embed, features_trans, center_features_trans
            ) 
            ## (B, C (+3), M, k) -> (B, M, k, C (+3))
            grouped_features = channel_transfer(grouped_features_trans)
            ## (B, M, k, C (+3)) -> (B, M, k, C_out)
            center_features = self.bb[gid](grouped_features, t)  
            # (B, M, out_channel)
            center_features = center_features.max(dim=2)[0]
            center_feature_list.append(center_features)
        
        return centers, center_embed, torch.cat(center_feature_list, dim = -1), sample_ids