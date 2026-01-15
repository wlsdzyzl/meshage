import numpy as np
from flemme.utils import save_ply
import mcubes
from vcg.config import *

def load_skeleton(sk_path):
    sk = np.load(sk_path)
    r = None
    assert sk.shape[1] == 3 or sk.shape[1] == 4, \
        "Shape of skeleton should be (N, 3) or (N, 4) (plus a radius dimension)."
    if sk.shape[1] == 4:
        r = sk[:, 3:]
        sk = sk[:, 0:3]
    return sk, r
def save_skeleton(sk, r, sk_path):
    if not r is None:
        sk = np.concatenate((sk, r), axis=1)
    assert sk.shape[1] == 3 or sk.shape[1] == 4, \
        "Shape of skeleton should be (N, 3) or (N, 4) (plus a radius dimension)."
    np.save(sk_path, sk)

def radius_normalize(data):
    return (data - center_r) * scaling_r

def radius_inv_normalize(data):
    return data / scaling_r + center_r

def save_sdf2mesh(save_path, sdf, threshold=0.0):
    vertices, triangles = mcubes.marching_cubes(sdf, isovalue = threshold, 
        truncated_value = truncated_value)
    vertices = (vertices + 0.5) / sdf.shape[0] * space_length - space_length / 2.0
    save_ply(save_path, vertices, faces = triangles)

def save_occ2mesh(save_path, sdf):
    vertices, triangles = mcubes.marching_cubes(sdf, isovalue = 0.5)
    vertices = (vertices + 0.5) / sdf.shape[0] * space_length - space_length / 2.0
    save_ply(save_path, vertices, faces = triangles)

def resolution2coord(resolution):
    length = int(space_length / resolution)
    x, y, z = ((np.mgrid[:length, :length, :length] - length / 2 + 0.5 ) * resolution)
    coord = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return coord, length
def save_valid_sdf_to_points(save_path, sdf):
    resolution = space_length / sdf.shape[0]
    coord = resolution2coord(resolution)[0]
    sdf = sdf.flatten()
    sampled_idx = np.logical_and(sdf >-1, sdf < 1) 
    sampled_coord = coord[sampled_idx]
    sampled_sdf = sdf[sampled_idx]
    save_ply(save_path, (sampled_coord, sampled_sdf))