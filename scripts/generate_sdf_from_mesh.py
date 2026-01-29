import numpy as np
from scipy.spatial import KDTree
from flemme.utils import normalize, rmdirs, mkdirs, load_mesh, save_ply, save_npy
from flemme.logger import get_logger
from scipy.ndimage import uniform_filter
import argparse
import glob
import os
from meshage.utils import save_sdf2mesh, resolution2coord
from meshage.config import truncated_value
from trimesh.smoothing import filter_laplacian

logger = get_logger("script::generate_sdf_from_mesh")
### generate sdf from skeleton and surface points for volume (-1, -1, -1) to (1, 1, 1)
# max_degree = 70
def generate_sdf(surface, normals, resolution=0.01):
    """
    Generate a signed distance field (SDF) from surface points and normals.

    Parameters:
    - surface: A list of points representing the surface.
    - normals: A list of normal vectors.
    - resolution: The resolution of the SDF grid.

    Returns:
    - sdf: A numpy array representing the signed distance field.
    """ 
    grid_points, length = resolution2coord(resolution)
    surf_tree = KDTree(surface)
    ## surface: N x 3, skeleton: M x 3, grid_points: L x 3
    ## dist: L, idx: L
    dist, idx = surf_tree.query(grid_points, k = 1, workers=-1)
    closest_surface = surface[idx]
    closest_surface_normal = normals[idx]
    dir_vec = (grid_points - closest_surface)
    dot_prod = np.einsum("bi,bi->b", dir_vec,  closest_surface_normal)
    sign = -np.sign(dot_prod)
    # sign[np.abs(dist) > 0.1] = 0
    # sign[ np.abs(dot_prod) < math.cos(math.radians(max_degree))] = 1.0
    sdf = (sign * dist).reshape(length, length, length)
    return sdf.astype(np.float32)

def process(mesh_path, sdf_path, recon_mesh_path, resolution=0.01, 
    mesh_smoothing=False, sdf_smoothing = False, 
    normalization = False, skip = False):
    try:
        mesh = load_mesh(mesh_path, clean = True)
    except Exception as e:
        logger.error(f"Skip broken mesh: {mesh_path}")
        logger.error(f'exception: {e}')
        return
    if len(mesh.vertices) == 0:
        logger.info(f"Skip empty mesh: {mesh_path}")
        return
    mesh = max(mesh.split(only_watertight = False), key=lambda item: item.vertices.shape[0])
    if mesh_smoothing:
        mesh = filter_laplacian(mesh, lamb=0.5, iterations=5)
    if not skip:
        surface, normals = mesh.vertices, mesh.vertex_normals
        if normalization:
            surface = normalize(surface, channel_dim = -1)
        save_ply(mesh_path[:-4]+'.ply', surface, faces = mesh.faces)
        logger.info(f"generating sdf which will be saved to {sdf_path}")
        sdf = generate_sdf(surface, normals, resolution=resolution)
        save_npy(sdf_path, sdf)
        ### smooth sdf
        if sdf_smoothing:
            truncated_indices = np.abs(sdf) < truncated_value
            sdf[truncated_indices] = uniform_filter(sdf, size = 3)[truncated_indices]
            save_npy(sdf_path[:-4]+'.smooth.npy', sdf)
        logger.info(f"extracting mesh which will be saved to {recon_mesh_path}")
        save_sdf2mesh(recon_mesh_path, sdf)
    return mesh.is_watertight
### python generate_sdf_from_mesh.py --mesh_dir /media/wlsdzyzl/DATA/datasets/pcd/MedShapeNet/bladder --sdf_dir /media/wlsdzyzl/DATA/datasets/pcd/MedShapeNet_SDF/bladder --recon_mesh_dir /media/wlsdzyzl/DATA/datasets/pcd/MedShapeNet_Mesh/bladder --smoothing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SDF from mesh files.")
    parser.add_argument("--mesh_dir", type=str, required=True, help="Path to the mesh files.")
    parser.add_argument("--mesh_suffix", type=str, default='.ply', help="Suffix of input mesh.")
    parser.add_argument("--sdf_dir", type=str, required=True, help="Path to save the generated SDF.")
    parser.add_argument("--recon_mesh_dir", type=str, required=True, help="Path to save the generated mesh.")
    parser.add_argument("--resolution", type=float, default=0.01, help="Resolution of the SDF grid.")
    parser.add_argument("--sdf_smoothing", action='store_true', help="Apply smoothing to the SDF.")
    parser.add_argument("--mesh_smoothing", action='store_true', help="Apply smoothing to the input mesh.")
    parser.add_argument("--normalization", action='store_true', help="Apply normalization to the surface.")
    parser.add_argument("--resume", action='store_true', help="Resume generating from the last file.")
    parser.add_argument("--record_watertight", action='store_true', help="Record the watertight attribute (Saved to recon_mesh_dir/non-watertight-meshes.txt).")
    args = parser.parse_args()
    mesh_path_list = sorted(glob.glob(os.path.join(args.mesh_dir, f"*{args.mesh_suffix}")))
    if not args.resume:
        rmdirs(args.sdf_dir)
        rmdirs(args.recon_mesh_dir)
    mkdirs(args.sdf_dir)
    mkdirs(args.recon_mesh_dir)
    non_watertight_meshes = []
    for mesh_path in mesh_path_list:
        base_name = os.path.basename(mesh_path).rsplit('.')[0]
        sdf_path = os.path.join(args.sdf_dir, f"{base_name}_sdf.npy")
        recon_mesh_path = os.path.join(args.recon_mesh_dir, f"{base_name}_mesh.ply")
        skip = args.resume and os.path.exists(recon_mesh_path)
        if skip: 
            if args.record_watertight: 
                logger.info(f'Skip processed file {mesh_path}, only record the water-tight attribute.')
            else:
                logger.info(f'Skip processed file {mesh_path}.')
                continue
        is_watertight = process(mesh_path, sdf_path, 
                    recon_mesh_path = recon_mesh_path, 
                    resolution=args.resolution, 
                    mesh_smoothing=args.mesh_smoothing,
                    sdf_smoothing=args.sdf_smoothing,
                    normalization = args.normalization,
                    skip = skip)
        if not is_watertight and args.record_watertight:
            logger.info(f'Found non-watertight mesh: {mesh_path}.')
            non_watertight_meshes.append(mesh_path)
    file_path = os.path.join(args.recon_mesh_dir, 'non-watertight-meshes.txt')

    if len(non_watertight_meshes) > 0:
        with open(file_path, "w") as file:
            for item in non_watertight_meshes:
                file.write(item + "\n")