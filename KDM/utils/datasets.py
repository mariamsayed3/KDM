# -*- coding: utf-8 -*-
"""
Created on 18/08/2020 7:41 pm

@author: Soan Duong, UOW
"""
import cv2
# Standard library imports
import numpy as np
from numpy import loadtxt
import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from spectral import *
#from . import tiff
import os

# Third party imports
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Local application imports


class HSIDataset(Dataset):
    def __init__(self, root_dir, txt_files, classes=[0, 1, 2, 3, 4], n_cutoff_imgs=None, dataset=None):
        """
        :param root_dir: root directory to the dataset folder, e.g ../02-Data/UOW-HSI/
        :param txt_files: text files contain filenames of BMP (ground-truth) files
                               The hsi file has the same filename but with the extension '.raw'
        :param classes: list of classes in the segmentation problem (classes must start from 0)
                        e.g. [0, 1, 2, 3, 4, 5]
        :param n_cutoff_imgs: maximum number of used images in each text file
        """
        super(HSIDataset, self).__init__()
        self.classes = classes
        self.txt_files = txt_files
        self.root_dir = root_dir
        self.dataset = dataset

        if (not isinstance(n_cutoff_imgs, int)) or (n_cutoff_imgs <= 0):
            n_cutoff_imgs = None

        # Get filename of the training images stored in txt_files
        if isinstance(txt_files, str):
            txt_files = [txt_files]
        self.training_imgs = []
        for file in txt_files:
            imgs = list(loadtxt(file, dtype=str)[:n_cutoff_imgs])

            if len(self.training_imgs) == 0:
                self.training_imgs = imgs
            else:
                self.training_imgs = np.concatenate((self.training_imgs, imgs))

    def __len__(self):
        """
        :return: the size of the dataset, i.e. the number of hsi images
        """
        return len(self.training_imgs)

    def __getitem__(self, index):
        """
        Read the an image from the training images
        :param index: index of the image in the list of training images
        :return: hsi image and ground-truth images
                + x: input hsi image of size (n_bands, H, W)
                + y_seg: ground-truth segmentation image of size (H, W)
                + y_oht: one-hot coding of the ground-truth image of size (n_classes, H, W)
        """
        from utils.utils import hsi_read_data
        ## Get the paths of the data
        # Get the ground truth and raw files
        bmp_file = self.root_dir + self.training_imgs[index] + '_mask.npy'
        raw_file = self.root_dir + self.training_imgs[index] + '_raw.npy'
        if not os.path.exists(raw_file) or not os.path.exists(bmp_file):
            raise FileNotFoundError(f"File not found: {raw_file} or {bmp_file}")
        # Read the hsi image
        x = np.load(raw_file)
        #x = spectral.Image(x)
        if len(x.shape) != 3:
            raise ValueError(f"HSI image does not have 3 dimensions, got {x.shape}")
        x = np.moveaxis(x, [0, 1, 2], [1, 2, 0])    # of size (n_bands, H, W)
        x = np.float32(x)                           # convert the input data into float32 datatype
        # Read the ground-truth image
        y_seg = np.load(bmp_file)
        y_oht = convert_seg2onehot(y_seg, self.classes)                     # of size (n_classes, H, W)
        # Convert the images into Pytorch tensors
        x = torch.Tensor(x)                                 # of size (n_bands, H, W)
        y_seg = torch.as_tensor(y_seg, dtype=torch.long)    # of size (H, W)
        y_oht = torch.as_tensor(y_oht, dtype=torch.float32)	# of size (n_classes, H, W)
        return {'input': x, 'ground_truth_onehot': y_oht, 'ground_truth_seg': y_seg, 'name': self.training_imgs[index][:-4]}

    def get_palette(self,n, color_map_by_label):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3

        # Custom color by label
        for label, colors in color_map_by_label.items():
            palette[label * 3: label * 3 + 3] = colors
        return palette

    def save_pred(self, preds, sv_path, name):
        # print(name)
        # print(sv_path)
        import os
        color_map_by_label = {
            0: [119, 158, 203],
            1: [124, 252, 0],
            2: [155, 118, 83],
            3: [255, 0, 0],
            4: [213, 213, 215]
        }
        palette = self.get_palette(256, color_map_by_label)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = preds[i]
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            n = os.path.splitext(name)[0]
            save_img.save(os.path.join(sv_path, '{}.png'.format(n)))

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json

class SpatialTranscriptomicsDataset(Dataset):
    def __init__(self, img_paths, seg_paths=None, 
                 cutting=None, transform=None,
                 channels=None, outtype='3d', envi_type='img',
                 multi_class=1, classes=[1, 2, 3, 4, 5], ignore_index=0):  # Note: removed 0 from classes
        """
        Initialize Spatial Transcriptomics Dataset
        Args:
            classes: Active classes (without background/ignore class)
            ignore_index: Index to ignore in loss computation (default: 0)
        """
        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.classes = classes  # Only active classes [1, 2, 3, 4, 5]
        self.ignore_index = ignore_index
        self.transform = transform
        self.cutting = cutting
        self.channels = channels
        self.outtype = outtype
        self.envi_type = envi_type
        self.multi_class = multi_class
        
        # Include ignore index for one-hot encoding but not for active training
        self.all_classes = [ignore_index] + classes  # [0, 1, 2, 3, 4, 5]
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.seg_paths[index]
        
        # Load data
        mask = np.load(mask_path)[:32, :32]
        img = np.load(img_path)[:32, :32, :]  # Shape: [32, 32, 136]
        
        # Clean mask - keep only valid classes, set rest to ignore_index
        mask[mask == 190] = self.ignore_index  # Remove old ignore-mask
        valid = np.isin(mask, self.classes)
        mask[~valid] = self.ignore_index  # Set all non-valid to ignore_index
        mask = mask.astype(np.int64)
        
        # Process image: [32, 32, 136] -> [136, 32, 32]
        img = np.transpose(img, (2, 0, 1))  # Now: [136, 32, 32]
        img = img.astype(np.float32)
        
        # Create one-hot encoding for ALL classes (including ignore_index for compatibility)
        mask_onehot = np.zeros((len(self.all_classes), mask.shape[0], mask.shape[1]))
        for i, class_label in enumerate(self.all_classes):
            mask_onehot[i, mask == class_label] = 1
        
        # Convert to torch tensors
        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask).long()
        mask_onehot_tensor = torch.from_numpy(mask_onehot).float()
        
        # Get sample name
        sample_name = os.path.basename(img_path).replace('.npy', '')
        
        return {
            'input': img_tensor,                    # Shape: (136, 32, 32)
            'ground_truth_seg': mask_tensor.unsqueeze(0),  # Shape: (1, 32, 32)
            'ground_truth_onehot': mask_onehot_tensor,     # Shape: (6, 32, 32)
            'name': sample_name
        }
    
    def __len__(self):
        return len(self.img_paths)

    def save_pred(self, pred_tensor, sv_path, name):
        """Save prediction - same as HSI dataset"""
        # Convert tensor to numpy if needed
        if torch.is_tensor(pred_tensor):
            pred_np = pred_tensor.detach().cpu().numpy()
        else:
            pred_np = pred_tensor
            
        # Handle different input shapes
        if len(pred_np.shape) == 4:  # Batch dimension
            pred_np = pred_np[0]
            
        if len(pred_np.shape) == 3:  # Multi-class predictions
            pred_np = np.argmax(pred_np, axis=0)
            
        # Create color-coded visualization like HSI
        colors = ['black', 'green', 'blue', 'red', 'yellow', 'cyan']
        color_map_by_label = {
            0: [119, 158, 203],  # Background
            1: [124, 252, 0],    # Class 1
            2: [155, 118, 83],   # Class 2
            3: [255, 0, 0],      # Class 3
            4: [213, 213, 215],  # Class 4
            5: [0, 255, 255]     # Class 5
        }
        
        # Save as image
        os.makedirs(sv_path, exist_ok=True)
        save_img = Image.fromarray(pred_np.astype(np.uint8))
        
        # Apply color palette
        palette = []
        for i in range(256):
            if i in color_map_by_label:
                palette.extend(color_map_by_label[i])
            else:
                palette.extend([0, 0, 0])
        save_img.putpalette(palette)
        
        save_path = os.path.join(sv_path, f'{name}.png')
        save_img.save(save_path)


def get_spatial_transcriptomics_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create dataloaders for spatial transcriptomics data using the existing file structure
    Args:
        data_dir: Base directory containing the dataset divide JSON and data files
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load dataset split
    dataset_divide = os.path.join(data_dir, 'dataset_divide.json')
    with open(dataset_divide, 'r') as f:
        dataset_dict = json.load(f)
    
    # Get file lists for each split
    if 'train' in dataset_dict and 'val' in dataset_dict and 'test' in dataset_dict:
        train_files = dataset_dict['train']
        val_files = dataset_dict['val'] 
        test_files = dataset_dict['test']
    else:
        # Fallback: use folds
        train_files = dataset_dict.get('fold1', []) + dataset_dict.get('fold2', []) + dataset_dict.get('fold3', [])
        val_files = dataset_dict.get('fold4', [])
        test_files = dataset_dict.get('fold5', val_files)  # Use val as test if no separate test
    
    # Create file paths
    train_img_paths = [os.path.join(data_dir, f"gene_expre_matrix_{i}.npy") for i in train_files]
    train_mask_paths = [os.path.join(data_dir, f"label_matrix_{i}.npy") for i in train_files]
    
    val_img_paths = [os.path.join(data_dir, f"gene_expre_matrix_{i}.npy") for i in val_files]
    val_mask_paths = [os.path.join(data_dir, f"label_matrix_{i}.npy") for i in val_files]
    
    test_img_paths = [os.path.join(data_dir, f"gene_expre_matrix_{i}.npy") for i in test_files]
    test_mask_paths = [os.path.join(data_dir, f"label_matrix_{i}.npy") for i in test_files]
    
    # Create datasets
    train_dataset = SpatialTranscriptomicsDataset(
        train_img_paths, 
        train_mask_paths,
        outtype='3d'
    )
    
    val_dataset = SpatialTranscriptomicsDataset(
        val_img_paths,
        val_mask_paths, 
        outtype='3d'
    )
    
    test_dataset = SpatialTranscriptomicsDataset(
        test_img_paths,
        test_mask_paths,
        outtype='3d'
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")
    
    return train_loader, val_loader, test_loader


