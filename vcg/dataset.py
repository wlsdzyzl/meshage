from flemme.utils import load_pcd, get_random_state, set_random_state, rreplace
import glob
import numpy as np
import os
from torch.utils.data import Dataset
from flemme.logger import get_logger
from flemme.dataset import pcd_dataset_dict, create_loader
from flemme.augment.pcd_transforms import FixedPoints, Normalize
from flemme.utils import load_npy
# from functools import partial
from vcg.utils import radius_normalize
logger = get_logger('coronary_dataset')

class CoronaryDataset(Dataset):
    def __init__(self, data_path, data_transform = None, mode = 'train', left_dir = 'left',
                 right_dir = 'right',
                 data_suffix = '.ply', 
                 normalization = 'minmax', 
                 fixed_points = 4096, 
                 sample_method = 'qfps', **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.left_path_list = sorted(glob.glob(os.path.join(data_path + '/' + left_dir,  "*" + data_suffix)))
        self.right_path_list = [rreplace(ppath, left_dir, right_dir, 1) for ppath in self.left_path_list]
        self.data_path = data_path
        self.mode = mode
        self.data_transform = data_transform
        assert normalization in ['minmax', 'mean'], 'normalization should be one of ["minmax", "mean"]'
        self.normalize = Normalize(method = normalization)
        self.fixed = FixedPoints(num = fixed_points // 2, method = sample_method)
    def __len__(self):
        return len(self.left_path_list)

    def __getitem__(self, index):
        """Get the pcds"""
        left_path = self.left_path_list[index]
        left_pcd = load_pcd(left_path)

        right_path = self.right_path_list[index]
        right_pcd = load_pcd(right_path)

        if self.data_transform:
            n_state, t_state = get_random_state()
            left_pcd = self.data_transform(left_pcd)
            set_random_state(n_state, t_state)
            right_pcd = self.data_transform(right_pcd)
        left_pcd = self.fixed(left_pcd)
        right_pcd = self.fixed(right_pcd)
        pcd = np.concatenate((left_pcd, right_pcd), axis=0) 
        pcd = self.normalize(pcd)
        return pcd, 0, left_path
    
pcd_dataset_dict['Coronary'] = CoronaryDataset