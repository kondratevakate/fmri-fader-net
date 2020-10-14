import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from utils import *


class ToTensor(object):
    def __call__(self, img):
        return torch.FloatTensor(img)
    
class Tanh(object):
    def __call__(self, img):
        return np.tanh(img)
    

def get_absmax(dataset):
    absmax = 0.
    for img in tqdm(dataset):
        if dataset.target is not None:
            img = torch.FloatTensor(img[0])
        else:
            img = torch.FloatTensor(img)
        absmax = max(absmax, img.abs().max().item())
        del img
    return absmax

class AbsMaxScale(object):
    def __init__(self, absmax):
        self.absmax = absmax
        
    def __call__(self, img):
        return img / self.absmax


class Pad(object):
    """
    Pads image with predefined padding size 
    Applicable: sMRI, fMRI-slice, fMRI
    
    Args:
    --- img_type - which type of image is transformed. 
        Options:
        - sMRI - structural image OR fMRI slice
        - fMRI - fMRI sequence (4D, time-axis last)
        - fMRI_tr - fMRI sequence transposed (4D, time-axis first)
    """
    def __init__(self, padding=(0, 0, 0), value=0, img_type="sMRI"):
        self.padding = padding
        self.value = np.float64(value)
        self.img_type = img_type
    
    def __call__(self, img):
        if self.padding == (0, 0, 0):
                return img
        
        img_shape = img.shape
        if self.img_type == "sMRI":
            # img_shape - C, x, y, z (C = 1)
            padded_shape = np.array(img_shape)
            padded_shape[1:] += np.array(self.padding) * 2
            padded_img = np.full(padded_shape, self.value)
            padded_img[:, 
                       self.padding[0]:self.padding[0] + img_shape[1], 
                       self.padding[1]:self.padding[1] + img_shape[2], 
                       self.padding[2]:self.padding[2] + img_shape[3]] = img
        elif self.img_type == "fMRI":
            # img_shape - C, x, y, z, t (C = 1)
            padded_shape = np.array(img_shape)
            padded_shape[1:4] += np.array(self.padding) * 2
            padded_img = np.full(padded_shape, self.value)
            padded_img[:, 
                       self.padding[0]:self.padding[0] + img_shape[1], 
                       self.padding[1]:self.padding[1] + img_shape[2], 
                       self.padding[2]:self.padding[2] + img_shape[3], :] = img
        elif self.img_type == "fMRI_tr":
            # img_shape = t, C, x, y, z (C = 1)
            padded_shape = np.array(img_shape)
            padded_shape[2:] += np.array(self.padding) * 2
            padded_img = np.full(padded_shape, self.value)
            padded_img[:, :,
                       self.padding[0]:self.padding[0] + img_shape[2], 
                       self.padding[1]:self.padding[1] + img_shape[3], 
                       self.padding[2]:self.padding[2] + img_shape[4]] = img
            
        return padded_img
