import os
import numpy as np
import argparse
from flemme.logger import get_logger

logger = get_logger("script::to_float32")

def convert_to_float32(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".npy"):
                npy_path = os.path.join(dirpath, filename)
                try:
                    arr = np.load(npy_path, allow_pickle=True)
                    arr = arr.astype(np.float32)
                    np.save(npy_path, arr)
                    logger.info(f"Converted: {npy_path} -> {npy_path}")
                except Exception as e:
                    logger.error(f"Failed to convert {npy_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer all npy files (float64) to npy files (float32).")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to root directory.")
    args = parser.parse_args()
    convert_to_float32(args.root_dir)
