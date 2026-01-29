import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob
from flemme.utils import load_pcd, load_npy, \
    get_random_state, set_random_state, rreplace, \
    contains_one_of, load_config
from flemme.logger import get_logger
from flemme.dataset import pcd_dataset_dict, \
    create_loader as _create_loader
from meshage.utils import resolution2coord
from meshage.config import truncate_sdf, truncated_value, \
                    train_truncate_scaling
from functools import partial
### Point Cloud with SDF
### target is sdf
logger = get_logger('pcd_sdf_dataset')

class PcdSDFDataset(Dataset):
    def __init__(self, data_path, data_transform = None, 
                target_transform = None, mode = 'train', data_dir = 'raw', 
                target_dir = 'sdf', data_suffix = '.ply', 
                target_suffix='.npy', resolution = 0.01, 
                skeleton_dir = None, skeleton_suffix='.ply',
                skeleton_transform = None,
                filter_file = None, truncate_prob = 1.0, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        
        self.data_path = data_path
        self.mode = mode
        self.data_transform = data_transform
        self.pcd_path_list = sorted(glob.glob(os.path.join(data_path + '/' + data_dir,  "*" + data_suffix)))
        if filter_file is not None:
            with open(filter_file, 'r') as file:
                filter_list = [line.strip() for line in file.readlines()]
            self.pcd_path_list = [ p for p in self.pcd_path_list if contains_one_of(p, filter_list)]
        if target_dir is None:
            assert self.mode == 'test', "Only in test mode, target_dir can be None."
            self.target_path_list = None
        else:
            self.target_path_list = [rreplace(rreplace(ppath, data_suffix, target_suffix, 1), data_dir, target_dir, 1) for ppath in self.pcd_path_list]
        self.skeleton_path_list = None
        if not skeleton_dir is None:
            self.skeleton_path_list = [rreplace(rreplace(ppath, data_suffix, skeleton_suffix, 1), data_dir, skeleton_dir, 1) for ppath in self.pcd_path_list]
        self.target_transform = target_transform
        self.skeleton_transform = skeleton_transform
        self.coord = resolution2coord(resolution)[0]
        self.truncate_prob = truncate_prob
    def __len__(self):
        return len(self.pcd_path_list)
    def __getitem__(self, index):
        """Get the pcds"""
        
        pcd = load_pcd(self.pcd_path_list[index])
        if not self.target_path_list is None:
            sdf = load_npy(self.target_path_list[index]).flatten()
            assert len(sdf) == len(self.coord), f"Coordinates ({len(self.coord)}) and sdf ({len(sdf)}) are not matched."
        else:
            sdf = None

        if not self.skeleton_path_list is None:
            ske = load_pcd(self.skeleton_path_list[index])
        else:
            ske = None
        ### pose transformation (translation and rotation) is not allowed
        ### only perform fixed_points transform
        if self.data_transform:

            n_state, t_state = get_random_state()
            pcd = self.data_transform(pcd)
            if self.target_transform:
                if self.target_transform.fixed_points:
                    coord = self.coord
                    if truncate_sdf:
                        if np.random.rand() < self.truncate_prob:
                            sdf_filter = (np.abs(sdf) <= truncated_value * train_truncate_scaling)
                            coord = coord[sdf_filter]
                            sdf = sdf[sdf_filter] 
                        sdf = sdf / (truncated_value * train_truncate_scaling)
                    negative_sdf = sdf[sdf <= 0] 
                    positive_sdf = sdf[sdf > 0]
                    negative_coord = coord[sdf <= 0]
                    positive_coord = coord[sdf > 0]

                    set_random_state(n_state, t_state)
                    negative_sdf = self.target_transform((negative_sdf, negative_coord)).unsqueeze(-1)
                    positive_sdf = self.target_transform((positive_sdf, positive_coord)).unsqueeze(-1)

                    set_random_state(n_state, t_state)
                    negative_coord = self.target_transform(negative_coord)
                    positive_coord = self.target_transform(positive_coord)
                    
                    sdf = torch.concat((negative_sdf, positive_sdf), dim = 0)
                    coord = torch.concat((negative_coord, positive_coord), dim = 0)
                    shuffle_indices = torch.randperm(sdf.shape[0])
                    sdf = sdf[shuffle_indices]
                    coord = coord[shuffle_indices]
                else:
                    set_random_state(n_state, t_state)
                    sdf = self.target_transform(sdf).unsqueeze(-1)
                    set_random_state(n_state, t_state)
                    coord = self.target_transform(self.coord)
            else:
                coord = torch.from_numpy(self.coord).float()
            if not ske is None and self.skeleton_transform:
                set_random_state(n_state, t_state)
                ske = self.skeleton_transform(ske)
        return pcd, ske, coord, sdf, self.pcd_path_list[index]

    
class PcdSDFWithClassLabelDataset(Dataset):
    def __init__(self, data_path, 
                data_transform = None, 
                target_transform = None,
                class_label_transform = None, 
                mode = 'train', 
                data_dir = 'raw', 
                target_dir = 'sdf', 
                data_suffix = '.ply', 
                target_suffix='.ply', 
                cls_label = {},
                pre_shuffle = True,
                resolution = 0.01, 
                filter_file = None, 
                truncate_prob = 1.0,
                skeleton_dir = None, 
                skeleton_suffix='.ply',
                skeleton_transform = None,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
        self.mode = mode
        self.data_transform = data_transform
        self.class_label_transform = class_label_transform
        self.target_transform = target_transform
        self.skeleton_transform = skeleton_transform
        self.pcd_path_list = []
        self.class_labels = []
        if target_dir is None:
            assert self.mode == 'test', "Only in test mode, target_dir can be None."
            self.target_path_list = None
        else:
            self.target_path_list = []
        if skeleton_dir is None:
            self.skeleton_path_list = None
        else:
            self.skeleton_path_list = []

        class_dirs = list(cls_label.keys())
        filter_dict = None
        if filter_file is not None:
            filter_dict = load_config(filter_file)

        for cls_dir in class_dirs:
            sub_path_list = sorted(glob.glob(os.path.join(data_path, data_dir, cls_dir, "*" + data_suffix)))
            if filter_dict:
                filter_list = filter_dict[cls_dir]
                sub_path_list = [p for p in sub_path_list if contains_one_of(p, filter_list)]
            self.pcd_path_list = self.pcd_path_list + sub_path_list
            if not self.target_path_list is None:
                sub_target_path_list = [rreplace(rreplace(s, data_dir, target_dir, 1), data_suffix, target_suffix, 1) for s in sub_path_list]
                self.target_path_list = self.target_path_list + sub_target_path_list
            if not self.skeleton_path_list is None:
                sub_skeleton_path_list = [rreplace(rreplace(s, data_dir, skeleton_dir, 1), data_suffix, skeleton_suffix, 1) for s in sub_path_list]
                self.skeleton_path_list = self.skeleton_path_list + sub_skeleton_path_list
            assert cls_dir in cls_label, f'Unknowk class: {cls_dir}'
            self.class_labels = self.class_labels + [cls_label[cls_dir], ] * len(sub_path_list)

        if pre_shuffle:
            shuffled_index = np.arange(len(self.pcd_path_list))
            np.random.shuffle(shuffled_index)
            self.pcd_path_list = [self.pcd_path_list[i] for i in shuffled_index]
            if not self.target_path_list is None:
                self.target_path_list = [self.target_path_list[i] for i in shuffled_index]
            if not self.skeleton_path_list is None:
                self.skeleton_path_list = [self.skeleton_path_list[i] for i in shuffled_index]
            self.class_labels = [self.class_labels[i] for i in shuffled_index]
        self.coord = resolution2coord(resolution)[0]
        self.truncate_prob = truncate_prob
    def __len__(self):
        return len(self.pcd_path_list)

    def __getitem__(self, index):
        """Get the pcds"""
        pcd = load_pcd(self.pcd_path_list[index])
        if not self.target_path_list is None:
            sdf = load_npy(self.target_path_list[index]).flatten()
            assert len(sdf) == len(self.coord), f"Coordinates ({len(self.coord)}) and sdf ({len(sdf)}) are not matched."
        else:
            sdf = None
        if not self.skeleton_path_list is None:
            ske = load_pcd(self.skeleton_path_list[index])
        else:
            ske = None
        cls_label = self.class_labels[index]

        ### pose transformation (translation and rotation) is not allowed
        ### only perform fixed_points transform
        if self.data_transform:

            n_state, t_state = get_random_state()
            pcd = self.data_transform(pcd)

            set_random_state(n_state, t_state)
            cls_label = self.class_label_transform(cls_label)
            if self.target_transform:
                if self.target_transform.fixed_points:
                    coord = self.coord
                    if truncate_sdf:
                        if np.random.rand() < self.truncate_prob:
                            sdf_filter = (np.abs(sdf) <= truncated_value * train_truncate_scaling)
                            coord = coord[sdf_filter]
                            sdf = sdf[sdf_filter] 
                        sdf = sdf / (truncated_value * train_truncate_scaling)
                    negative_sdf = sdf[sdf <= 0] 
                    positive_sdf = sdf[sdf > 0]
                    negative_coord = coord[sdf <= 0]
                    positive_coord = coord[sdf > 0]

                    set_random_state(n_state, t_state)
                    negative_sdf = self.target_transform((negative_sdf, negative_coord)).unsqueeze(-1)
                    positive_sdf = self.target_transform((positive_sdf, positive_coord)).unsqueeze(-1)

                    set_random_state(n_state, t_state)
                    negative_coord = self.target_transform(negative_coord)
                    positive_coord = self.target_transform(positive_coord)
                    
                    sdf = torch.concat((negative_sdf, positive_sdf), dim = 0)
                    coord = torch.concat((negative_coord, positive_coord), dim = 0)
                    shuffle_indices = torch.randperm(sdf.shape[0])
                    sdf = sdf[shuffle_indices]
                    coord = coord[shuffle_indices]
                else:
                    set_random_state(n_state, t_state)
                    sdf = self.target_transform(sdf).unsqueeze(-1)
                    set_random_state(n_state, t_state)
                    coord = self.target_transform(self.coord)
            else:
                coord = torch.from_numpy(self.coord).float()
            if not ske is None and self.skeleton_transform:
                set_random_state(n_state, t_state)
                ske = self.skeleton_transform(ske)        
        return pcd, ske, cls_label, coord, sdf, self.pcd_path_list[index]

pcd_dataset_dict['PcdSDFDataset'] = PcdSDFDataset
pcd_dataset_dict['PcdSDFWithClassLabelDataset'] = PcdSDFWithClassLabelDataset

create_loader = partial(_create_loader, custom_attributes=['skeleton'])
