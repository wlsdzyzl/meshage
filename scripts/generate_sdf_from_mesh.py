import numpy as np
from scipy.spatial import KDTree
from flemme.utils import normalize, rmdirs, mkdirs, load_mesh, save_ply, save_npy, index_points
from flemme.logger import get_logger
from scipy.ndimage import uniform_filter
import argparse
import glob
import os
from meshage.utils import save_sdf2mesh, resolution2coord
from meshage.config import truncated_value
from trimesh.smoothing import filter_laplacian
import trimesh
logger = get_logger("script::generate_sdf_from_mesh")
### generate sdf from skeleton and surface points for volume (-1, -1, -1) to (1, 1, 1)
# max_degree = 70
def point_in_triangle(p, v0, v1, v2, eps=1e-8):
    """
    p, v0, v1, v2: (N, 3)
    return: (N,) bool
    """
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p  = p  - v0

    dot00 = np.sum(v0v2 * v0v2, axis=-1)
    dot01 = np.sum(v0v2 * v0v1, axis=-1)
    dot02 = np.sum(v0v2 * v0p,  axis=-1)
    dot11 = np.sum(v0v1 * v0v1, axis=-1)
    dot12 = np.sum(v0v1 * v0p,  axis=-1)

    denom = dot00 * dot11 - dot01 * dot01 + eps

    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom

    return (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps)

def generate_sdf_from_faces(mesh, resolution=0.01):
    """
    Generate a signed distance field (SDF) from vertices points and normals.

    Parameters:
    - mesh: input mesh (with vertices and faces)
    - resolution: The resolution of the SDF grid.

    Returns:
    - sdf: A numpy array representing the signed distance field.
    """ 
    vertices = mesh.vertices
    grid_points, length = resolution2coord(resolution)
    
    ### use trimesh to compute the closest distance,, too slow!!
    # closest_points, dist, triangle_id = mesh.nearest.on_surface(grid_points)
    # face_normals = mesh.face_normals[triangle_id] 
    # dir_vec = (grid_points - closest_points)

    surf_tree = KDTree(mesh.vertices)
    point_dist, idx = surf_tree.query(grid_points, k = 1, workers=-1)
    ### signed vertex dist
    dir_vec = grid_points - mesh.vertices[idx]
    dot_prod = np.einsum("bi,bi->b", dir_vec, mesh.vertex_normals[idx])
    sign = -np.sign(dot_prod)
    signed_point_dist = sign * point_dist

    ### signed face dist
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]

    d = -np.sum(mesh.face_normals * v0, axis=1) 

    vertex_first_faces = mesh.vertex_faces[:, :1]
    vertex_repeat_faces = np.repeat(vertex_first_faces, mesh.vertex_faces.shape[1], axis = 1)
    vertex_faces = mesh.vertex_faces.copy()
    ## without -1 padding
    vertex_faces[vertex_faces < 0] = vertex_repeat_faces[vertex_faces < 0]

    neighbor_faces = vertex_faces[idx]
    # m * n * 3
    nf_normals = mesh.face_normals[neighbor_faces.view()].reshape(neighbor_faces.shape[0], neighbor_faces.shape[1], -1)
    # m * n
    nf_d = d[neighbor_faces.view()].reshape(neighbor_faces.shape[0], neighbor_faces.shape[1])

    signed_face_dist = np.einsum("abi,abi->ab", np.repeat(grid_points[:, None, ...], vertex_faces.shape[1], axis = 1),  nf_normals) + nf_d

    ## check if the projected points are in the triangles
    projected_p = grid_points[:, None, ...] - signed_face_dist[..., None] * nf_normals
    nf_v0 = v0[neighbor_faces.view()].reshape(neighbor_faces.shape[0], neighbor_faces.shape[1], -1)
    nf_v1 = v1[neighbor_faces.view()].reshape(neighbor_faces.shape[0], neighbor_faces.shape[1], -1)
    nf_v2 = v2[neighbor_faces.view()].reshape(neighbor_faces.shape[0], neighbor_faces.shape[1], -1)

    in_triangles = point_in_triangle(projected_p, nf_v0, nf_v1, nf_v2)

    signed_face_dist[np.logical_not(point_in_triangle)] = 1e8

    min_idx = np.argmin(np.abs(signed_face_dist), axis = 1)
    # row = np.arange(signed_face_dist.shape[0])
    # signed_face_dist = signed_face_dist[row, min_idx]
    signed_face_dist = -index_points(signed_face_dist, min_idx)

    ### check which dists is better
    sdf = signed_face_dist.copy()
    point_dist_filter = np.abs(signed_face_dist) > np.abs(signed_point_dist)
    sdf[point_dist_filter] = signed_point_dist[point_dist_filter]
    sdf = sdf.reshape(length, length, length)
    return sdf.astype(np.float32)

def generate_sdf_from_points(surface, normals, resolution=0.01):
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
    normalization = False, skip = False, num_resample_points = 0,
    num_components=1, from_points = True):
    try:
        mesh = load_mesh(mesh_path, clean = True)
    except Exception as e:
        logger.error(f"Skip broken mesh: {mesh_path}")
        logger.error(f'exception: {e}')
        return
    if len(mesh.vertices) == 0:
        logger.info(f"Skip empty mesh: {mesh_path}")
        return
    if num_components > 0:
        mesh_list = sorted(mesh.split(only_watertight = False), key=lambda item: item.vertices.shape[0], reverse=True)[:num_components]
        mesh = trimesh.util.concatenate(mesh_list)
    if mesh_smoothing:
        mesh = filter_laplacian(mesh, lamb=0.5, iterations=5)
    if not skip:
        if normalization:
            mesh.vertices = normalize(mesh.vertices, channel_dim = -1)
        save_ply(mesh_path[:-4]+'.ply', mesh.vertices, faces = mesh.faces)
        if from_points:
            if num_resample_points > 0:
                surface, face_idx = mesh.sample(num_resample_points, return_index=True)
                normals = mesh.face_normals[face_idx]
            else:
                surface, normals = mesh.vertices, mesh.vertex_normals
            logger.info(f"generating sdf which will be saved to {sdf_path}")
            sdf = generate_sdf_from_points(surface, normals, resolution=resolution)
        else:
            sdf = generate_sdf_from_faces(mesh, resolution=resolution)
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
    parser.add_argument("--num_resample_points", type=int, default=0, help="Resample points on the triangles of surface.")
    parser.add_argument("--from_faces", action='store_true', help="Compute SDF using faces (when the model has very sparse points).")
    parser.add_argument("--num_components", type=int, default=1, help="Only keep n connected components of the mesh. By default, we only keep one connected component.")

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
                    skip = skip, 
                    num_resample_points = args.num_resample_points,
                    num_components=args.num_components, 
                    from_points = not args.from_faces
                    )
        if not is_watertight and args.record_watertight:
            logger.info(f'Found non-watertight mesh: {mesh_path}.')
            non_watertight_meshes.append(mesh_path)
    file_path = os.path.join(args.recon_mesh_dir, 'non-watertight-meshes.txt')

    if len(non_watertight_meshes) > 0:
        with open(file_path, "w") as file:
            for item in non_watertight_meshes:
                file.write(item + "\n")