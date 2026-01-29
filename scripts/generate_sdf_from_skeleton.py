import numpy as np
from scipy.spatial import KDTree
from flemme.utils import load_ply, normalize, rmdirs, mkdirs, save_ply, save_npy
from flemme.logger import get_logger
from scipy.ndimage import uniform_filter
import argparse
import glob
import os
from meshage.utils import save_sdf2mesh, truncated_value, resolution2coord
logger = get_logger("script::generate_sdf_from_skeleton")
### generate sdf from skeleton and surface points for volume (-1, -1, -1) to (1, 1, 1)
def generate_sdf(skeleton, surface, resolution=0.01, k = 5):
    """
    Generate a signed distance field (SDF) from a skeleton and surface points.

    Parameters:
    - skeleton: A list of points representing the skeleton.
    - surface: A list of points representing the surface.
    - resolution: The resolution of the SDF grid.

    Returns:
    - sdf: A numpy array representing the signed distance field.
    """
    grid_points, length = resolution2coord(resolution)
    surf_tree = KDTree(surface)
    ske_tree = KDTree(skeleton)
    ## surface: N x 3, skeleton: M x 3, grid_points: L x 3
    ## dist: L * k, idx: L * k
    dist, idx = surf_tree.query(grid_points, k = k, workers=-1)
    # closest_surface = surface[idx]
    ### query multiple neighbors to get a more stable distance
    dist = np.mean(dist, axis=1)
    # closest_surface = (surface[idx[..., 0]] + surface[idx[..., 1]] + surface[idx[..., 2]] + surface[idx[..., 3]] + surface[idx[..., 4]]) / 5
    
    closest_surface = np.take(surface, axis=0, indices = idx).mean(axis = 1)
    _, idx = ske_tree.query(closest_surface, k = k, workers=-1)
    closest_skeleton = np.take(skeleton, axis=0, indices = idx).mean(axis = 1)
    sign = np.sign(np.einsum("bi,bi->b", grid_points - closest_surface,  closest_skeleton - closest_surface))
    sdf = (sign * dist).reshape(length, length, length)
    return sdf.astype(np.float32)
def process(skeleton_path, surface_path, sdf_path, mesh_path, 
            normalization = True,
            resolution=0.01, 
            sdf_smoothing=True, 
            k = 5):
    skeleton = load_ply(skeleton_path)
    surface = load_ply(surface_path)
    if len(surface) < k or len(skeleton) < k:
        logger.info(f"Skip empty surface: {surface_path}")
        return
    
    if normalization:
        surface, (center, scaling) = normalize(surface, channel_dim = -1, return_transform=True)
        skeleton = normalize(skeleton, channel_dim = -1, center = center, scaling = scaling)
        save_ply(surface_path[:-4]+'_normalized.ply', surface)
    logger.info(f"generating sdf which will be saved to {sdf_path}")
    sdf = generate_sdf(skeleton, surface, resolution=resolution, k = k)
    save_npy(sdf_path, sdf)
    logger.info(f"extracting mesh which will be saved to {mesh_path}")
    ### smooth sdf
    if sdf_smoothing:
        truncated_indices = np.abs(sdf) < truncated_value
        sdf[truncated_indices] = uniform_filter(sdf, size = 3)[truncated_indices]
        save_npy(sdf_path[:-4]+'.smooth.npy', sdf)
    save_sdf2mesh(mesh_path, sdf)
### python generate_sdf_from_skeleton.py --surface_dir /media/wlsdzyzl/DATA/datasets/pcd/imageCAS/output_lr/surface --skeleton_dir /media/wlsdzyzl/DATA/datasets/pcd/imageCAS/output_lr/skeleton --sdf_dir /media/wlsdzyzl/DATA/datasets/pcd/imageCAS/output_lr/sdf --recon_mesh_dir /media/wlsdzyzl/DATA/datasets/pcd/imageCAS/output_lr/mesh_from_sdf --sdf_smoothing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SDF from skeleton and surface points.")
    parser.add_argument("--skeleton_dir", type=str, required=True, help="Path to the skeleton files.")
    parser.add_argument("--surface_dir", type=str, required=True, help="Path to the surface points files.")
    parser.add_argument("--sdf_dir", type=str, required=True, help="Path to save the generated SDF.")
    parser.add_argument("--recon_mesh_dir", type=str, required=True, help="Path to save the generated mesh.")
    parser.add_argument("--resolution", type=float, default=0.01, help="Resolution of the SDF grid.")
    parser.add_argument("--num_nn", type=int, default=5, help="Number of nearest neighbors.")
    parser.add_argument("--normalization", action='store_true', help="Apply normalization to the surface.")
    parser.add_argument("--sdf_smoothing", action='store_true', help="Apply smoothing to the SDF.")
    parser.add_argument("--resume", action='store_true', help="Resume generating from the last file.")
    args = parser.parse_args()
    surface_path_list = sorted(glob.glob(os.path.join(args.surface_dir, "*.ply")))
    skeleton_path_list = [ sf_path.replace(args.surface_dir, args.skeleton_dir)  for sf_path in surface_path_list]
    
    if not args.resume:
        rmdirs(args.sdf_dir)
        rmdirs(args.recon_mesh_dir)
    mkdirs(args.sdf_dir)
    mkdirs(args.recon_mesh_dir)

    assert len(skeleton_path_list) == len(surface_path_list), "Number of skeleton and surface files must match."
    for sk_path, surf_path in zip(skeleton_path_list, surface_path_list):
        logger.info(f'Processing skeleton: {sk_path} and surface: {surf_path}')
        base_name = os.path.basename(sk_path).replace(".ply", "")
        sdf_path = os.path.join(args.sdf_dir, f"{base_name}_sdf.npy")
        mesh_path = os.path.join(args.recon_mesh_dir, f"{base_name}_mesh.ply")
        if args.resume and os.path.exists(mesh_path):
            logger.info(f'Skip processed file: {surf_path}')
            continue
        process(sk_path, surf_path, sdf_path, 
                    mesh_path = mesh_path, 
                    normalization=args.normalization,
                    resolution=args.resolution, 
                    sdf_smoothing=args.sdf_smoothing,
                    k = args.num_nn)