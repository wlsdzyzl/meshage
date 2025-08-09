from flemme.utils import load_pcd, get_random_state, set_random_state, rreplace
import glob
import numpy as np
import os
from torch.utils.data import Dataset
from flemme.logger import get_logger
from flemme.dataset import pcd_dataset_dict, create_loader
from flemme.augment.pcd_transforms import FixedPoints, Normalize
from flemme.utils import load_npy
from vcg.utils import resolution2coord
# from functools import partial
from vcg.utils import radius_normalize
logger = get_logger('coronary_dataset')

class PcdSDFDataset(PcdDataset):
    def __init__(self, data_path, data_transform = None, sdf_transform = None, mode = 'train', data_dir = 'pcd', 
                 sdf_dir = 'target', data_suffix = '.ply', sdf_suffix='.ply', resolution = 0.01, **kwargs):
        super().__init__(data_path = data_path, data_transform = data_transform, mode = mode,
            data_dir = data_dir, data_suffix = data_suffix)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.sdf_path_list = [rreplace(rreplace(ppath, data_suffix, sdf_suffix, 1), data_dir, sdf_dir, 1) for ppath in self.pcd_path_list]
        self.sdf_transform = sdf_transform
        self.coord = resolution2coord(resolution)[0]
    def __getitem__(self, index):
        """Get the pcds"""
        pcd = load_pcd(self.pcd_path_list[index])
        target = load_npy(self.sdf_path_list[index])
        if self.data_transform:
            n_state, t_state = get_random_state()
            pcd = self.data_transform(pcd)
            set_random_state(n_state, t_state)
            target = self.sdf_transform(target)
            # save_pcd(pcd_path+'.transformed.ply', pcd.numpy())
        return pcd, target, self.pcd_path_list[index]
    
class PcdReconWithClassLabelDataset(Dataset):
    def __init__(self, data_path, 
                 data_transform = None, 
                 sdf_transform = None,
                 class_label_transform = None, 
                 mode = 'train', 
                 data_dir = 'partial', 
                 sdf_dir = 'target', 
                 data_suffix = '.ply', 
                 sdf_suffix='.ply', 
                 cls_label = {},
                 pre_shuffle = True,
                 **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
        self.mode = mode
        self.data_transform = data_transform
        self.class_label_transform = class_label_transform
        self.sdf_transform = sdf_transform
        self.pcd_path_list = []
        self.sdf_path_list = []
        self.class_labels = []

        class_dirs = list(cls_label.keys())

        for cls_dir in class_dirs:
            sub_path_list = sorted(glob.glob(os.path.join(data_path, cls_dir, data_dir,  "*" + data_suffix)))
            self.pcd_path_list = self.pcd_path_list + sub_path_list
            sub_sdf_path_list = [rreplace(rreplace(s, data_dir, sdf_dir, 1), data_suffix, sdf_suffix, 1) for s in sub_path_list]
            self.sdf_path_list = self.sdf_path_list + sub_sdf_path_list
            assert cls_dir in cls_label, f'Unknowk class: {cls_dir}'
            self.class_labels = self.class_labels + [cls_label[cls_dir], ] * len(sub_path_list)
        if pre_shuffle:
            shuffled_index = np.arange(len(self.pcd_path_list))
            np.random.shuffle(shuffled_index)
            self.pcd_path_list = [self.pcd_path_list[i] for i in shuffled_index]
            self.sdf_path_list = [self.sdf_path_list[i] for i in shuffled_index]
            self.class_labels = [self.class_labels[i] for i in shuffled_index]

    def __len__(self):
        return len(self.pcd_path_list)

    def __getitem__(self, index):
        """Get the pcds"""
        pcd = load_pcd(self.pcd_path_list[index])
        target = load_pcd(self.sdf_path_list[index])
        cls_label = self.class_labels[index]
        if self.data_transform:
            n_state, t_state = get_random_state()
            pcd = self.data_transform(pcd)
            set_random_state(n_state, t_state)
            target = self.sdf_transform(target)
            if self.class_label_transform:
                cls_label = self.class_label_transform(cls_label)
            # save_pcd(pcd_path+'.transformed.ply', pcd.numpy())
        return pcd, target, cls_label, self.pcd_path_list[index]
