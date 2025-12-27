import numpy as np
from flemme.utils import load_ply, mkdirs, save_ply
from flemme.logger import get_logger
from scipy.spatial import KDTree
from tqdm import tqdm
import argparse
import glob
import os

logger = get_logger("script::standardize_pcds")

def standardize_points(point_clouds, followed_skeletons):
    max_bb = 0.0
    max_radius = []
    
    for i in tqdm(range(len(point_clouds)), desc = "computing_radius"):
        ## move to center
        center = 0.5 * (point_clouds[i].max(axis=0, keepdims = True) + point_clouds[i].min(axis=0, keepdims = True))
        point_clouds[i] -= center
        followed_skeletons[i] -= center

        surf_tree = KDTree(point_clouds[i])
        dist, idx = surf_tree.query(followed_skeletons[i], k = 1, workers=-1)
        max_radius.append(dist.max())

    mean_max_radius = sum(max_radius) / len(max_radius)
    max_bbs = []
    ## scale samples to have similar max radius
    for i in range(len(point_clouds)):        
        separate_scaling = (mean_max_radius +  np.random.uniform(0, 0.025)) / max_radius[i]
        point_clouds[i] *= separate_scaling
        if followed_skeletons:
            followed_skeletons[i] *= separate_scaling
        tmp_max_bb = (point_clouds[i].max(axis=0) - point_clouds[i].min(axis=0)).max()  
        max_bbs.append(tmp_max_bb)

    histogram, bin_edges = np.histogram(max_bbs, bins = 20)
    sum_hist = len(point_clouds)
    for i in range(len(histogram)-1, -1, -1):
        sum_hist -= histogram[i]
        ### 95% of the samples should be within the bounding box and we leave 5% for outliers
        if sum_hist < 0.95 * len(point_clouds):
            max_bb = bin_edges[i+1]
            break
    # print(histogram, bin_edges, max_bb)
    global_scaling = 2.0 / max_bb

    for i in range(len(point_clouds)):
        point_clouds[i] *= global_scaling
        if followed_skeletons:
            followed_skeletons[i] *= global_scaling
# python standardize_pcds.py --skeleton_dir /media/wlsdzyzl/DATA/datasets/pcd/imageCAS/vessel_diffusion/output/skeleton/ --surface_dir /media/wlsdzyzl/DATA/datasets/pcd/imageCAS/vessel_diffusion/output/surface/ --output_dir /media/wlsdzyzl/DATA/datasets/pcd/imageCAS/vessel_diffusion/output_standardized/ > standardized_both.out
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize point clouds and skeletons.")
    parser.add_argument("--skeleton_dir", type=str, required=True, help="Path to the skeleton files.")
    parser.add_argument("--surface_dir", type=str, required=True, help="Path to the surface points files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output files.")
    args = parser.parse_args()

    surface_path_list = sorted(glob.glob(os.path.join(args.surface_dir, "*.ply")))
    skeleton_path_list = [ sf_path.replace(args.surface_dir, args.skeleton_dir)  for sf_path in surface_path_list]
    assert len(skeleton_path_list) == len(surface_path_list), "Number of skeleton and surface files must match."
    
    output_surface_dir = os.path.join(args.output_dir, 'surface/')
    output_skeleton_dir = os.path.join(args.output_dir, 'skeleton/')
    mkdirs(output_surface_dir)
    mkdirs(output_skeleton_dir)
    surfs = [load_ply(p) for p in surface_path_list]
    skels = [load_ply(p) for p in skeleton_path_list]
    standardize_points(surfs, skels)
    for i, p in enumerate(surface_path_list):
        op_surf_path = p.replace(args.surface_dir, output_surface_dir)
        op_skel_path = p.replace(args.surface_dir, output_skeleton_dir)
        logger.info(f'Saving to {op_surf_path} and {op_skel_path}')
        save_ply(op_surf_path, surfs[i])
        save_ply(op_skel_path, skels[i])