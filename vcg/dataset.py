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
    create_loader
from vcg.utils import resolution2coord

### Point Cloud with SDF
### target is sdf
logger = get_logger('pcd_sdf_dataset')

class PcdSDFDataset(Dataset):
    def __init__(self, data_path, data_transform = None, 
                target_transform = None, mode = 'train', data_dir = 'pcd', 
                target_dir = 'sdf', data_suffix = '.ply', 
                target_suffix='.npy', resolution = 0.01, 
                filter_file = None, truncate_sdf = True, **kwargs):
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
        self.target_path_list = [rreplace(rreplace(ppath, data_suffix, target_suffix, 1), data_dir, target_dir, 1) for ppath in self.pcd_path_list]
        self.target_transform = target_transform
        self.coord = resolution2coord(resolution)[0]
    def __len__(self):
        return len(self.pcd_path_list)
    def __getitem__(self, index):
        """Get the pcds"""
        
        pcd = load_pcd(self.pcd_path_list[index])
        sdf = load_npy(self.target_path_list[index]).flatten()
        assert len(sdf) == len(self.coord), f"Coordinates ({len(self.coord)}) and sdf ({len(sdf)}) are not matched."


        ### pose transformation (translation and rotation) is not allowed
        ### only perform fixed_points transform
        if self.data_transform:

            n_state, t_state = get_random_state()
            pcd = self.data_transform(pcd)

            if self.target_transform.fixed_points:
                negative_sdf = sdf[sdf <= 0] 
                positive_sdf = sdf[sdf > 0]
                # print('neg / pos:', len(negative_sdf), len(positive_sdf))
                # print('neg / pos:', len(negative_sdf), len(positive_sdf))
                negative_coord = self.coord[sdf <= 0]
                positive_coord = self.coord[sdf > 0]

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
        return pcd, coord, sdf, self.pcd_path_list[index]
    
class PcdSDFWithClassLabelDataset(Dataset):
    def __init__(self, data_path, 
                 data_transform = None, 
                 target_transform = None,
                 class_label_transform = None, 
                 mode = 'train', 
                 data_dir = 'partial', 
                 target_dir = 'target', 
                 data_suffix = '.ply', 
                 target_suffix='.ply', 
                 cls_label = {},
                 pre_shuffle = True,
                 resolution = 0.01, 
                 filter_file = None, 
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
        self.pcd_path_list = []
        self.target_path_list = []
        self.class_labels = []

        class_dirs = list(cls_label.keys())
        filter_dict = None
        if filter_file is not None:
            filter_dict = load_config(filter_file)
        for cls_dir in class_dirs:
            sub_path_list = sorted(glob.glob(os.path.join(data_path, data_dir, cls_dir,  "*" + data_suffix)))
            if filter_dict:
                filter_list = filter_dict[cls_dir]
                sub_path_list = [p for p in sub_path_list if contains_one_of(p, filter_list)]
            self.pcd_path_list = self.pcd_path_list + sub_path_list
            sub_target_path_list = [rreplace(rreplace(s, data_dir, target_dir, 1), data_suffix, target_suffix, 1) for s in sub_path_list]
            self.target_path_list = self.target_path_list + sub_target_path_list
            assert cls_dir in cls_label, f'Unknowk class: {cls_dir}'
            self.class_labels = self.class_labels + [cls_label[cls_dir], ] * len(sub_path_list)

        if pre_shuffle:
            shuffled_index = np.arange(len(self.pcd_path_list))
            np.random.shuffle(shuffled_index)
            self.pcd_path_list = [self.pcd_path_list[i] for i in shuffled_index]
            self.target_path_list = [self.target_path_list[i] for i in shuffled_index]
            self.class_labels = [self.class_labels[i] for i in shuffled_index]
        self.coord = resolution2coord(resolution)[0]
    def __len__(self):
        return len(self.pcd_path_list)

    def __getitem__(self, index):
        """Get the pcds"""
        pcd = load_pcd(self.pcd_path_list[index])
        sdf = load_npy(self.target_path_list[index]).flatten()
        assert len(sdf) == len(self.coord), f"Coordinates ({len(self.coord)}) and sdf ({len(sdf)}) are not matched."
        cls_label = self.class_labels[index]

        ### pose transformation (translation and rotation) is not allowed
        ### only perform fixed_points transform
        if self.data_transform:

            n_state, t_state = get_random_state()
            pcd = self.data_transform(pcd)

            set_random_state(n_state, t_state)
            cls_label = self.class_label_transform(cls_label)

            if self.target_transform.fixed_points:
                negative_sdf = sdf[sdf <= 0] 
                positive_sdf = sdf[sdf > 0]
                negative_coord = self.coord[sdf <= 0]
                positive_coord = self.coord[sdf > 0]

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

        return pcd, cls_label, coord, sdf, self.pcd_path_list[index]

pcd_dataset_dict['PcdSDFDataset'] = PcdSDFDataset
pcd_dataset_dict['PcdSDFWithClassLabelDataset'] = PcdSDFWithClassLabelDataset
