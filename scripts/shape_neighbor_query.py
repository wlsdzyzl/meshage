import numpy as np
from flemme.metrics import CD, EMD
from flemme.logger import get_logger
from flemme.utils import load_ply, topk, normalize
from tqdm import tqdm
import argparse
import glob
import os

logger = get_logger("script::shape_neighbor_query")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Searching neighbors for query shapes from target shapes.")
    parser.add_argument("--query_dir", type=str, required=True, help="Path to the query shapes directory.")
    parser.add_argument("--target_dir", type=str, required=True, help="Path to the target shapes directory.")
    parser.add_argument("--output_file", type=str, default="shape_neigbors.txt", help="Path to the text that saves the query shapes and neighbors.")
    parser.add_argument("--distance", type=str, default="CD", help="Distance metric to use for neighbor search.")
    parser.add_argument("--k", type=int, default=1, help="Number of nearest neighbors to find.")
    parser.add_argument("--fixed_points", type=int, default=-1, help="Number of points to sample from each shape.")
    parser.add_argument("--normalize", action='store_true', help="Normalize the query and target point clouds.")
    args = parser.parse_args()

    if args.distance.lower() not in ['cd', 'emd']:
        logger.error("Distance metric must be either 'CD' or 'EMD'.")
        exit(1)
    dist = CD() if args.distance == 'CD' else EMD()
    query_files = sorted(glob.glob(os.path.join(args.query_dir, "*.ply")))
    target_files = sorted(glob.glob(os.path.join(args.target_dir, "*.ply")))
    logger.info(f"Found {len(query_files)} query shapes and {len(target_files)} target shapes.")

    ### distance matrix
    query_shapes = [load_ply(f) for f in query_files]
    target_shapes = [load_ply(f) for f in target_files]
    if args.normalize:
        query_shapes = [normalize(p, channel_dim = -1) for p in query_shapes]
        target_shapes = [normalize(p, channel_dim = -1) for p in target_shapes]
    if args.fixed_points > 0:
        from flemme.augment.pcd_transforms import FixedPoints
        fixed_points_transform = FixedPoints(num=args.fixed_points, method='qfps')
        query_shapes = [fixed_points_transform(s) for s in query_shapes]
        target_shapes = [fixed_points_transform(s) for s in target_shapes]
        logger.info(f"Sampled {args.fixed_points} points from each shape.")
    dist_matrix = np.ones((len(query_shapes), len(target_shapes))) * np.inf
    
    for i, q_shape in tqdm(enumerate(query_shapes), total=len(query_shapes), desc="Computing distance matrix"):
        distances = []
        for j, t_shape in enumerate(target_shapes):
            tmp_dist = dist(q_shape, t_shape)
            dist_matrix[i, j] = tmp_dist

    neighbor_dist, neighbor_index = topk(-dist_matrix, k=args.k, axis=1, sorted=True)
    with open(args.output_file, 'w') as f:
        for i, q_file in enumerate(query_files):
            neighbors = [target_files[idx] for idx in neighbor_index[i]]
            f.write(f"{q_file} " + " ".join(neighbors) + "\n")
    logger.info(f"Saved neighbors to {args.output_file}")

