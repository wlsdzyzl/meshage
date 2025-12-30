import numpy as np
import torch
import open3d as o3d
import trimesh
from scipy.spatial import KDTree
from flemme.utils import *
from flemme.logger import get_logger
import glob
from copy import deepcopy
logger = get_logger("script::generate_skeleton_from_mesh")
# ============================================================
#  Get SDF using prebuilt scene
# ============================================================
def get_o3d_mesh(trimesh_mesh, color=[1, 0, 0]):
    cur_mesh_o3 = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(trimesh_mesh.vertices),
                                            triangles=o3d.utility.Vector3iVector(trimesh_mesh.faces))

    cur_mesh_o3.compute_vertex_normals()
    cur_mesh_o3.paint_uniform_color(color)

    return cur_mesh_o3

def get_sdfs(query_sample, trimesh_mesh):
    tst_o3d = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(trimesh_mesh.vertices),
                                            triangles=o3d.utility.Vector3iVector(trimesh_mesh.faces))

    tst_o3d.compute_vertex_normals()

    sample_o3d = o3d.core.Tensor(query_sample, dtype=o3d.core.Dtype.Float32)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(tst_o3d)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    # Compute distance of the query point from the surface
    signed_distance = scene.compute_signed_distance(sample_o3d)

    return signed_distance.numpy()


# ============================================================
#  Helper: Surface initialization (you must confirm your logic)
# ============================================================

def get_surface_init(mesh, surface_sample, num_candidates=500, sampling_scale=0.0005):
    gt_mesh = deepcopy(mesh)
    mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(get_o3d_mesh(gt_mesh))
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_legacy)

    skel = surface_sample.copy()[:, None, :]
    skel = skel + sampling_scale * np.random.randn(len(skel), num_candidates, 3)
    flat_skel = skel.reshape(-1, 3)
    flat_sdfs = get_sdfs(flat_skel, mesh)
    sdfs = flat_sdfs.reshape(*skel.shape[:2])
    min_inds = np.argmin(sdfs, axis=1)
    surface_init = skel[range(len(skel)), min_inds]

    return surface_init

def get_init_skel(mesh, dir_angle=np.pi / 6, num_samples=50, init_type='mesh_verts', num_init_samples=50000):

    if init_type == 'mesh_verts':
        vertex_normals = mesh.vertex_normals
        verts = mesh.vertices
    elif init_type == 'random':
        verts, vert_faces = mesh.sample(num_init_samples, return_index=True)
        vertex_normals = mesh.face_normals[vert_faces]
    elif init_type == 'combined':
        assert num_init_samples > len(mesh.vertices), "num_init_samples needs to be > len(mesh.vertices)"
        residual_num = num_init_samples - len(mesh.vertices)
        verts, vert_faces = mesh.sample(residual_num, return_index=True)
        vertex_normals = mesh.face_normals[vert_faces]
        verts = np.concatenate((mesh.vertices, verts), axis=0)
        vertex_normals = np.concatenate((mesh.vertex_normals, vertex_normals), axis=0)

    theta = np.cos(dir_angle)
    z = torch.rand(num_samples, 1) * (1 - theta) + theta
    angle = 2 * torch.pi * torch.rand(num_samples, 1)
    pts = torch.hstack((torch.sqrt(1 - z ** 2) * torch.cos(angle), torch.sqrt(1 - z ** 2) * torch.sin(angle), z))

    source_vec = torch.FloatTensor([0, 0, 1]).reshape(1, -1)
    target_vec = torch.FloatTensor(np.array(-vertex_normals))

    vs = torch.linalg.cross(source_vec, target_vec)
    ss = torch.linalg.norm(vs, axis=-1)
    cc = torch.sum(target_vec * source_vec, axis=-1)
    kmat = torch.zeros(len(vs), 3, 3)
    kmat[:, 0, 1] = -vs[:, 2]
    kmat[:, 0, 2] = vs[:, 1]
    kmat[:, 1, 0] = vs[:, 2]
    kmat[:, 1, 2] = -vs[:, 0]
    kmat[:, 2, 0] = -vs[:, 1]
    kmat[:, 2, 1] = vs[:, 0]
    rrs = torch.eye(3)[None, :, :] + kmat + kmat.bmm(kmat) * ((1 - cc[:, None, None]) / (ss[:, None, None] ** 2 + 1e-8))

    all_rotated_dirs = torch.bmm(rrs, pts[None, :, :].repeat(len(rrs), 1, 1).permute(0, 2, 1))
    all_rotated_dirs = all_rotated_dirs.permute(0, 2, 1)

    mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(get_o3d_mesh(mesh))
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_legacy)

    surface_init = get_surface_init(mesh, verts)
    tiled_verts = np.tile(surface_init[:, np.newaxis, :], (1, num_samples, 1))
    rays = np.concatenate((tiled_verts, all_rotated_dirs), axis=-1)
    rays = rays.reshape(-1, 6)
    rays = o3d.core.Tensor(list(rays),
                           dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)
    lsd_ray_dists = ans['t_hit'].numpy().reshape((len(verts), num_samples, 1))

    pad_val = np.median(lsd_ray_dists[lsd_ray_dists != np.inf])
    lsd_ray_dists[lsd_ray_dists == np.inf] = pad_val
    lsds = np.nanmean(lsd_ray_dists, axis=1)
    init_skel = verts - 0.5 * lsds * vertex_normals

    return init_skel, lsds, verts


def get_min_sdf_skel(mesh, num_iter=50, lsds_mult=0.6, init_type='mesh_verts', num_init_samples=50000):
    init_skel, lsds, verts = get_init_skel(mesh, init_type=init_type, num_init_samples=num_init_samples)
    skel = init_skel.copy()
    alpha = 0.1
    skel_points = torch.FloatTensor(skel.copy()).type(torch.float64)
    inds_to_check = np.array(range(len(skel_points)))

    for i in range(num_iter):
        cur_skel = skel_points[inds_to_check]
        grad_input = cur_skel.clone().detach().requires_grad_(True)
        k_neighb = 20
        query_pts = skel
        neighbs = cur_skel[:, None, :] + torch.clip(0.003 * torch.randn(len(cur_skel), k_neighb, 3), max=0.01)
        nn_sdfs = get_sdfs(neighbs.reshape(-1, 3).detach().cpu().numpy(), mesh)
        nn_sdfs = torch.FloatTensor(nn_sdfs.reshape(-1, k_neighb))

        dists = grad_input[:, None, :] - neighbs
        dists = torch.linalg.norm(dists, axis=-1)
        weights = torch.exp(-dists ** 2 / 0.002)  # OLD 0.002 IMPORTANT
        weights = weights / weights.sum(axis=1, keepdims=True)
        weighted_sdfs = nn_sdfs * weights
        weighted_sdfs = weighted_sdfs.sum(axis=1)
        loss = weighted_sdfs.sum()
        loss.backward()

        fin_grad = grad_input.grad
        del grad_input
        cur_skel_new = cur_skel - alpha * fin_grad  # alpha*mean_grads
        norms = np.linalg.norm(cur_skel_new - cur_skel, axis=1)

        disps = skel_points.numpy() - verts
        disps = np.linalg.norm(disps, axis=1)
        if i < 10:
            lsds_mask = disps < 1e3
        else:
            lsds_mask = disps < lsds_mult * lsds[:, 0]
        #print(lsds_mask.sum())
        check_mask = lsds_mask[inds_to_check]
        skel_points[inds_to_check[check_mask]] = cur_skel_new[check_mask]
        inds_to_check = inds_to_check[check_mask]
        if len(inds_to_check) == 0:
            break

    return skel_points.numpy()
# ============================================================
#  Main execution example
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Skeleton from mesh files.")
    parser.add_argument("--mesh_dir", type=str, required=True, help="Path to the mesh files.")
    parser.add_argument("--mesh_suffix", type=str, default='.ply', help="Suffix of input mesh.")
    parser.add_argument("--ske_dir", type=str, required=True, help="Path to save the generated Skeleton.")
    parser.add_argument("--resume", action='store_true', help="Resume generating from the last file.")
    args = parser.parse_args()
    mesh_path_list = sorted(glob.glob(os.path.join(args.mesh_dir, f"*{args.mesh_suffix}")))
    if not args.resume:
        rmdirs(args.ske_dir)
    mkdirs(args.ske_dir)

    for mesh_path in mesh_path_list:
        base_name = os.path.basename(mesh_path).rsplit('.')[0]
        ske_path = os.path.join(args.ske_dir, f"{base_name}_ske.ply")
        skip = args.resume and os.path.exists(ske_path)
        if skip: 
            continue
        else:
            logger.info(f'Saving to {ske_path}.')
            mesh = load_mesh(mesh_path, True, True)
            ske = get_min_sdf_skel(mesh, num_iter=50)
            save_ply(ske_path, ske)


