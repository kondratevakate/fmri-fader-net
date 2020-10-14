import os
import copy
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from utils import *

from transforms import *


# ABIDE-{I, II}
class ABIDEMRI(data.Dataset):
    """
    Arguments:
        path: path to data folder
        labels_path: path to file with targets and additional information
        target: column of targets df with target to predict. If None, loads images only
        encode_target: if True, encode target with LabelEncoder
        load_online (bool): if True, load mri images online. Else, preload everything during initialization
    """
    
    def __init__(self, paths, labels_path, target=None, encode_target=False, load_online=False, 
                 sub_path="ses-2", mri_type="sMRI", mri_file_suffix="", transform=None,
                 use_sources=[], 
                 sources_boundaries={
                     "all" : [(0, 0, 0,), (152, 152, 152,)]
                 },
                 sources_scales={
                     "all" : 1.
                 },
                 clip=False,
                 start_pos=None, seq_len=None,
                 domain_target=None# "SOURCE"
                ):
        self.mri_paths = {
            "participant_id" : [],
            "path" : [],
        }
        
        self.paths = paths if type(paths) is list else [paths]
        self.labels = pd.read_csv(labels_path)
        self.target, self.domain_target = self.set_target(target, encode_target, domain_target)
        self.load_online = load_online
        
        self.mri_type = mri_type
        if self.mri_type == "sMRI":
            self.type = "anat" 
        elif self.mri_type == "fMRI":
            self.type = "func"
        else:
            self.type = None
#             raise ValueError("Select sMRI or fMRI mri type.")
        self.mri_file_suffix = mri_file_suffix
    
        self.use_sources = use_sources
        self.sources_boundaries = sources_boundaries
        self.sources_scales = sources_scales
        self.clip = clip
        self.start_pos = start_pos
        self.seq_len = seq_len
        self.transform = transform
        
        for path_to_folder in self.paths:
            for patient_folder_name in tqdm(os.listdir(path_to_folder)):
                if 'sub-' in patient_folder_name and os.path.isdir(path_to_folder + patient_folder_name):

                    if self.type is not None and self.type in os.listdir(os.path.join(path_to_folder, patient_folder_name, sub_path)):
                        temp_path = os.path.join(path_to_folder, patient_folder_name, sub_path, self.type)
                    elif self.type is None:
                        temp_path = os.path.join(path_to_folder, patient_folder_name, sub_path)
                    else:
                        continue

                    for filename in os.listdir(temp_path):
                        if self.mri_file_suffix in filename:
                            self.mri_paths["participant_id"].append(patient_folder_name)
                            full_path = os.path.join(temp_path, filename)
                            self.mri_paths["path"].append(full_path)
                            
        self.mri_paths = pd.DataFrame(self.mri_paths)
        self.labels = self.labels.merge(self.mri_paths, on="participant_id")
        self.mri_files = self.labels["path"].tolist()
        
        if not self.load_online:
            self.mri_files = [self.get_image(index, self.start_pos, self.seq_len) for index in tqdm(range(len(self.mri_files)))]

        # update self.img_shape (and other params ?)
#         self.output_img_shape = self[0].shape[1:4]
        
    
    def set_target(self, target=None, encode_target=False, domain_target=None):
        self.target, self.domain_target = None, None
        if target is not None:
            self.target = self.labels[target].copy()
            if self.use_sources:
                # зануляем таргеты для объектов из неинтересных нам источников
                null_idx = ~self.labels["SOURCE"].isin(self.use_sources)
                self.target[null_idx] = np.nan
            if encode_target:
                enc = LabelEncoder()
                idx = self.target.notnull()
                self.target[idx] = enc.fit_transform(self.target[idx])
            if domain_target is not None:
                self.domain_target = self.labels[domain_target].copy()
                if self.use_sources:
                    # зануляем таргеты для объектов из неинтересных нам источников
                    null_idx = ~self.labels["SOURCE"].isin(self.use_sources)
                    self.domain_target[null_idx] = np.nan
                self.domain_enc = LabelEncoder()
                idx = self.domain_target.notnull()
                self.domain_target[idx] = self.domain_enc.fit_transform(self.domain_target[idx])
        return self.target, self.domain_target
            
    def reshape_image(self, mri_img, coord_min, img_shape):
        if self.mri_type == "sMRI":
            return mri_img[coord_min[0]:coord_min[0] + img_shape[0],
                           coord_min[1]:coord_min[1] + img_shape[1],
                           coord_min[2]:coord_min[2] + img_shape[2]].reshape((1,) + img_shape)
        if self.mri_type == "fMRI":
            seq_len = mri_img.shape[-1]
            return mri_img[coord_min[0]:coord_min[0] + img_shape[0],
                           coord_min[1]:coord_min[1] + img_shape[1],
                           coord_min[2]:coord_min[2] + img_shape[2], :].reshape((1,) + img_shape + (seq_len,))
        
    def get_image(self, index, start_pos=None, seq_len=None):
        
        def load_mri(mri_file):
            if "nii" in mri_file:
                img = load_nii_to_array(mri_file)
            else:
                img = np.load(mri_file)
            return img
        
        mri_file = self.mri_files[index]
        s = self.labels["SOURCE"][index]
        if self.use_sources and s not in self.use_sources:
            return None
        
        if s in self.sources_boundaries:
            coord_min, img_shape = self.sources_boundaries[s]
        else:
            coord_min, img_shape = self.sources_boundaries["all"]
        if s in self.sources_scales:
            scale = self.sources_scales[s]
        else:
            scale = self.sources_scales["all"] 
        
        img = load_mri(mri_file)      
        # check for padding
        cur_shape = np.array(img[..., 0].shape) if self.mri_type == "fMRI" else np.array(img.shape)
        req_shape = np.array(coord_min) + np.array(img_shape)
        padding = np.maximum(req_shape - cur_shape, 0)
        img = Pad(tuple(padding), img_type=self.mri_type)(img[np.newaxis, :])[0]
        # reshape
        img = self.reshape_image(img, coord_min, img_shape)
        if self.clip:
            img = np.clip(img, 0., scale)
        img /= scale
        
        if self.mri_type == "sMRI":
            return img
        
        if self.mri_type == "fMRI":
            if seq_len is None:
                seq_len = img.shape[-1]
            # what if seq_len == 0 ?
            if start_pos is None:
                start_pos = np.random.choice(img.shape[-1] - seq_len)
            if seq_len == 1:
                img = img[:, :, :, :, start_pos]
            else:
                img = img[:, :, :, :, start_pos:start_pos + seq_len]
                img = img.transpose((4, 0, 1, 2, 3))
            return img
    
    def __getitem__(self, index):
        img = self.get_image(index, self.start_pos, self.seq_len) if self.load_online else self.mri_files[index]
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target is None:
            item = img
        else:
            item = [img, self.target[index]]
            if self.domain_target is not None:
                item += [self.domain_target[index]]
        return item
    
    def __len__(self):
        return len(self.mri_files)


# CPAC
class CPACMRI(data.Dataset):
    """
    Arguments:
        path: path to data folder
        labels_path: path to file with targets and additional information
        target: column of targets df with target to predict. If None, loads images only
        encode_target: if True, encode target with LabelEncoder
        load_online (bool): if True, load mri images online. Else, preload everything during initialization
        
        # CPAC - weare only supposed to find only fMRI files.
        
    """
    
    def __init__(self, paths, labels_path, target=None, encode_target=False, load_online=False, 
                 mri_type="fMRI", mri_file_suffix="", 
                 get_patient_id=lambda p: "sub-" + p.split("_")[-3],
                 transform=None,
                 use_sources=[], 
                 sources_boundaries={
                     "all" : [(0, 0, 0,), (152, 152, 152,)]
                 },
                 sources_scales={
                     "all" : 1.
                 },
                 clip=False,
                 start_pos=None, seq_len=None,
                 domain_target=None# "SOURCE"
                ):
        self.mri_paths = {
            "participant_id" : [],
            "path" : [],
        }
        
        self.paths = paths if type(paths) is list else [paths]
        self.labels = pd.read_csv(labels_path)
        self.target, self.domain_target = self.set_target(target, encode_target, domain_target)
        self.load_online = load_online
        
        self.mri_type = mri_type
        if self.mri_type == "sMRI":
            self.type = "anat" 
        elif self.mri_type == "fMRI":
            self.type = "func"

        self.mri_file_suffix = mri_file_suffix
        self.get_patient_id = get_patient_id
    
        self.use_sources = use_sources
        self.sources_boundaries = sources_boundaries
        self.sources_scales = sources_scales
        self.clip = clip
        self.start_pos = start_pos
        self.seq_len = seq_len
        self.transform = transform
        
        for path_to_folder in self.paths:
            for filename in tqdm(os.listdir(path_to_folder)):
                if self.mri_file_suffix in filename:
                    patient_id = self.get_patient_id(filename)
                    self.mri_paths["participant_id"].append(patient_id)
                    full_path = os.path.join(path_to_folder, filename)
                    self.mri_paths["path"].append(full_path)
                            
        self.mri_paths = pd.DataFrame(self.mri_paths)
        self.labels = self.labels.merge(self.mri_paths, on="participant_id")
        self.mri_files = self.labels["path"].tolist()
        
        if not self.load_online:
            self.mri_files = [self.get_image(index, self.start_pos, self.seq_len) for index in tqdm(range(len(self.mri_files)))]

        # update self.img_shape (and other params ?)
#         self.output_img_shape = self[0].shape[1:4]
        
    
    def set_target(self, target=None, encode_target=False, domain_target=None):
        self.target, self.domain_target = None, None
        if target is not None:
            self.target = self.labels[target].copy()
            if self.use_sources:
                # зануляем таргеты для объектов из неинтересных нам источников
                null_idx = ~self.labels["SOURCE"].isin(self.use_sources)
                self.target[null_idx] = np.nan
            if encode_target:
                enc = LabelEncoder()
                idx = self.target.notnull()
                self.target[idx] = enc.fit_transform(self.target[idx])
            if domain_target is not None:
                self.domain_target = self.labels[domain_target].copy()
                if self.use_sources:
                    # зануляем таргеты для объектов из неинтересных нам источников
                    null_idx = ~self.labels["SOURCE"].isin(self.use_sources)
                    self.domain_target[null_idx] = np.nan
                self.domain_enc = LabelEncoder()
                idx = self.domain_target.notnull()
                self.domain_target[idx] = self.domain_enc.fit_transform(self.domain_target[idx])
        return self.target, self.domain_target
            
    def reshape_image(self, mri_img, coord_min, img_shape):
        if self.mri_type == "sMRI":
            return mri_img[coord_min[0]:coord_min[0] + img_shape[0],
                           coord_min[1]:coord_min[1] + img_shape[1],
                           coord_min[2]:coord_min[2] + img_shape[2]].reshape((1,) + img_shape)
        if self.mri_type == "fMRI":
            seq_len = mri_img.shape[-1]
            return mri_img[coord_min[0]:coord_min[0] + img_shape[0],
                           coord_min[1]:coord_min[1] + img_shape[1],
                           coord_min[2]:coord_min[2] + img_shape[2], :].reshape((1,) + img_shape + (seq_len,))
        
    def get_image(self, index, start_pos=None, seq_len=None):
        
        def load_mri(mri_file):
            if "nii" in mri_file:
                img = load_nii_to_array(mri_file)
            else:
                img = np.load(mri_file)
            return img
        
        mri_file = self.mri_files[index]
        s = self.labels["SOURCE"][index]
        if self.use_sources and s not in self.use_sources:
            return None
        
        if s in self.sources_boundaries:
            coord_min, img_shape = self.sources_boundaries[s]
        else:
            coord_min, img_shape = self.sources_boundaries["all"]
        if s in self.sources_scales:
            scale = self.sources_scales[s]
        else:
            scale = self.sources_scales["all"] 
        
        img = load_mri(mri_file)      
        # check for padding
        cur_shape = np.array(img[..., 0].shape) if self.mri_type == "fMRI" else np.array(img.shape)
        req_shape = np.array(coord_min) + np.array(img_shape)
        padding = np.maximum(req_shape - cur_shape, 0)
        img = Pad(tuple(padding), img_type=self.mri_type)(img[np.newaxis, :])[0]
        # reshape
        img = self.reshape_image(img, coord_min, img_shape)
        if self.clip:
            img = np.clip(img, -scale, scale)
        img /= scale
        
        if self.mri_type == "sMRI":
            return img
        
        if self.mri_type == "fMRI":
            if seq_len is None:
                seq_len = img.shape[-1]
            # what if seq_len == 0 ?
            if start_pos is None:
                start_pos = np.random.choice(img.shape[-1] - seq_len)
            if seq_len == 1:
                img = img[:, :, :, :, start_pos]
            else:
                img = img[:, :, :, :, start_pos:start_pos + seq_len]
                img = img.transpose((4, 0, 1, 2, 3))
            return img
    
    def __getitem__(self, index):
        img = self.get_image(index, self.start_pos, self.seq_len) if self.load_online else self.mri_files[index]
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target is None:
            item = img
        else:
            item = [img, self.target[index]]
            if self.domain_target is not None:
                item += [self.domain_target[index]]
        return item
    
    def __len__(self):
        return len(self.mri_files)

# таргеты возвращаются в том же порядке, с _теми же_ индексами
# но, видимо, в процессе обучения мы будем брать только те индексы, которые соответствуют нотналл позициям здесь
# те для списка с данными (упорядоченного в том же порядке, что и общий список индексов)
# ничего не меняется в зависимости от задачи, 
# варьируется только то, какое _подмножество индексов_ мы используем для получения данных для обучения