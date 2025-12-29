import numpy as np
from skimage.transform import resize
from flemme.utils import rmdirs, mkdirs, save_npy
from flemme.logger import get_logger
from scipy.ndimage import uniform_filter
import math
import argparse
import glob
import os
from vcg.utils import save_sdf2mesh
from vcg.config import space_length, truncated_value


logger = get_logger("script::generate_sdf_from_sdf")
### generate sdf from skeleton and surface points for volume (-1, -1, -1) to (1, 1, 1)
# max_degree = 70
def generate_sdf(sdf, resolution=0.01):
    """
    Generate a signed distance field (SDF) from a high-resolution sdf.
    """
    ### 
    length = int(space_length / resolution)
    sdf = resize(sdf, (length, length, length), anti_aliasing = False, order = 3)
    return sdf.astype(np.float32)

def process(input_sdf_path, output_sdf_path, 
        recon_mesh_path, resolution=0.01, sdf_smoothing = False):
    input_sdf = np.load(input_sdf_path)
    logger.info(f"generating sdf which will be saved to {output_sdf_path}")
    output_sdf = generate_sdf(input_sdf, resolution=resolution)
    save_npy(output_sdf_path, output_sdf)
    ### smooth input sdf and resize
    if sdf_smoothing:
        truncated_indices = np.abs(input_sdf) < truncated_value
        input_sdf[truncated_indices] = uniform_filter(input_sdf, size = 3)[truncated_indices]
        output_sdf = generate_sdf(input_sdf, resolution=resolution)
        save_npy(sdf_path[:-4]+'.smooth.npy', output_sdf)
    logger.info(f"extracting mesh which will be saved to {recon_mesh_path}")
    save_sdf2mesh(recon_mesh_path, output_sdf)
### python generate_sdf_from_sdf.py --input_dir /media/wlsdzyzl/DATA/datasets/sdf/MedShapeNet/bladder --sdf_dir /media/wlsdzyzl/DATA/datasets/pcd/MedShapeNet_SDF/bladder --recon_mesh_dir /media/wlsdzyzl/DATA/datasets/pcd/MedShapeNet_Mesh/bladder --smoothing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SDF from sdf files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input sdf files.")
    parser.add_argument("--sdf_dir", type=str, required=True, help="Path to save the generated SDF.")
    parser.add_argument("--recon_mesh_dir", type=str, required=True, help="Path to save the generated mesh.")
    parser.add_argument("--resolution", type=float, default=0.01, help="Resolution of the SDF grid.")
    parser.add_argument("--resume", action='store_true', help="Resume generating from the last file.")
    parser.add_argument("--sdf_smoothing", action='store_true', help="Apply smoothing to the SDF.")

    args = parser.parse_args()
    input_path_list = sorted(glob.glob(os.path.join(args.input_dir, f"*.npy")))
    if not args.resume:
        rmdirs(args.sdf_dir)
        rmdirs(args.recon_mesh_dir)
    mkdirs(args.sdf_dir)
    mkdirs(args.recon_mesh_dir)
    for input_path in input_path_list:
        base_name = os.path.basename(input_path).rsplit('.')[0]
        base_name = base_name.replace('_sdf', '')
        sdf_path = os.path.join(args.sdf_dir, f"{base_name}_sdf.npy")
        recon_mesh_path = os.path.join(args.recon_mesh_dir, f"{base_name}_mesh.ply")
        skip = args.resume and os.path.exists(recon_mesh_path)
        if skip: 
            logger.info(f'Skip processed file {input_path}.')
            continue
        process(input_path, sdf_path, 
                recon_mesh_path = recon_mesh_path, 
                resolution=args.resolution,
                sdf_smoothing = args.sdf_smoothing)