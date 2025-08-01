import numpy as np
from flemme.utils import save_ply
import mcubes
min_r = 0.007
max_r = 0.12
center_r = (max_r - min_r) / 2.0
scaling_r = 2.0 / (max_r - min_r)
space_length = 2.2
truncated_value = 0.1
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
    vertices, triangles = mcubes.marching_cubes(sdf, isovalue = threshold, truncated_value = truncated_value)
    vertices = (vertices + 0.5) / sdf.shape[0] * space_length - space_length / 2.0
    save_ply(save_path, vertices, faces = triangles)