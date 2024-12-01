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
from . import tiff

# Third party imports
import torch
from PIL import Image
from torch.utils.data import Dataset

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
        if self.dataset == 'brain':
            bmp_file = self.root_dir + self.training_imgs[index] + "/" + self.training_imgs[index] + '/gtMap.hdr'
            raw_file = self.root_dir + self.training_imgs[index] + "/" + self.training_imgs[index] + '/raw.hdr'
        elif self.dataset == 'pathology':
            bmp_file = self.root_dir + 'Mask/' + self.training_imgs[index] + '.png'
            raw_file = self.root_dir + 'MHSI/' + 'MHSI/' + self.training_imgs[index] + '.hdr'
        elif self.dataset == 'dental':
            bmp_file = self.root_dir + self.training_imgs[index] + '_masks.tif'
            raw_file = self.root_dir + self.training_imgs[index] + '.tif'
            print("understanding root")
            print(self.root_dir)
            print("training image")
            print(self.training_imgs)
            print("understanding bmp_file")
            print(bmp_file)
            print("understanding raw_file")
            print(raw_file)
        else:
            bmp_file = self.root_dir + self.training_imgs[index]
            raw_file = self.root_dir + self.training_imgs[index][:-4] + '.raw'

        assert False
        #import os
        #print(os.listdir('D:/Users/vmhp806/data/HSI/UOW-HSI'))
        # Read the hsi image
        if self.dataset == 'brain':
            x = envi.open(raw_file, image=raw_file[:-4])[:, :, :]
            chosen_channels = np.linspace(0, x.shape[2] - 1, num=300, dtype=int)
            new_x = [x[:, :, channel] for channel in chosen_channels]
            x = np.stack(new_x, axis=2)
        elif self.dataset == 'pathology':
            x = envi.open(raw_file, image=raw_file[:-4]+'.img')[:, :, :]
        elif self.dataset == 'dental':
            x, _, _, _ = tiff.read_stiff(raw_file)
            x = cv2.resize(x, (250, 250), interpolation=cv2.INTER_NEAREST)
            chosen_channels = np.linspace(0, x.shape[2]-1, num=51, dtype=int)
            new_x = [x[:, :, channel] for channel in chosen_channels]
            x = np.stack(new_x, axis=2)
        else:
            x, _ = hsi_read_data(raw_file)  # of size (H, W, n_bands)
        x = np.moveaxis(x, [0, 1, 2], [1, 2, 0])    # of size (n_bands, H, W)
        x = np.float32(x)                           # convert the input data into float32 datatype
        # x = x[:30,:50,:50]

        # Read the ground-truth image
        if self.dataset == 'brain':
            mask = envi.open(bmp_file, image=bmp_file[:-4])[:, :, 0]
            y_seg = np.squeeze(mask)
            y_seg = y_seg.astype(np.float32)
        elif self.dataset == 'pathology':
            bmp = (cv2.imread(bmp_file, 0) / 255).astype(np.uint8)
            y_seg = bmp
            # y_seg = np.array(bmp.getdata()).reshape(bmp.size[1], bmp.size[0]) / 255
        elif self.dataset == 'dental':
            masks = tiff.read_mtiff(bmp_file)
            y_seg = tiff.mtiff_to_2d_arr(masks)
            y_seg = cv2.resize(y_seg, (250, 250), interpolation=cv2.INTER_NEAREST)
        else:
            bmp = Image.open(bmp_file)
            y_seg = np.array(bmp.getdata()).reshape(bmp.size[1], bmp.size[0])   # of size (H, W)

        # y_seg = y_seg[:50,:50]
        #y_seg[y_seg == 4] = 0
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

def convert_seg2onehot(y_seg, classes):
    """
    Convert the segmentation image y into the one-hot code image
    :param y_seg: 2D array, a segmentation image with a size of H x W
    :param classes: list of classes in image y
    :return: one-hot code image of y with a size of n_classes x H x W
    """
    y_onehot = np.zeros((len(classes), y_seg.shape[0], y_seg.shape[1]))

    for k, class_label in enumerate(classes):
        y_onehot[k, :, :][y_seg == class_label] = 1

    return y_onehot


def convert_prob2seg(y_prob, classes):
    """
    Convert the class-probability image into the segmentation image
    :param y_prob: class-probability image with a size of n_classes x H x W
    :param classes: list of classes in image y
    :return: 2D array, a segmentation image with a size of H x W
    """
    y_class = np.argmax(y_prob, axis=0)
    y_seg = np.zeros((y_prob.shape[1], y_prob.shape[2]))

    for k, class_label in enumerate(classes):
        # Find indices in y_class whose pixels are k
        indx = np.where(y_class == k)
        if len(indx[0] > 0):
            y_seg[indx] = class_label

    return y_seg


if __name__ == "__main__":
    pc_dir = '/mnt/Windows/cv_projects/archive/'
    # pc_dir = 'U:/02-Data/UOW-HSI/'
    training_files = ['../data-3125/P2.txt']
    t = {}
    for im_file in training_files:
        img_files = list(loadtxt(im_file, dtype=str))
        for f in img_files:
            bmp_file = pc_dir + f
            bmp = Image.open(bmp_file)
            # y = np.array(bmp.getdata())
            y = np.array(bmp.getdata()).reshape(bmp.size[1], bmp.size[0])
            y_unique = np.unique(y)
            #print(f)
            if 0 in y_unique:
                print(im_file, '-',f)


    classes = [0, 1, 2, 3, 4]
    # Convert the segmentation image y into the one-hot code image
    y_gt = convert_seg2onehot(y, classes)

    # Convert the one-hot code image y into the segmentation image
    y_seg = convert_prob2seg(y_gt, classes)

    colors = ['black', 'green', 'blue', 'red', 'yellow']
    cmap = mpl.colors.ListedColormap(colors[np.min(bmp):])

    # Plot images
    fig = plt.figure(figsize=(9, 6))
    fig.subplots_adjust(bottom=0.5)
    ax = plt.subplot(321)
    im = ax.imshow(y, cmap=cmap)
    plt.title('Ground-truth image')
    fig.colorbar(im, ax=ax)

    ax = plt.subplot(322)
    im = ax.imshow(y_seg, cmap=cmap)
    plt.title('Ground-truth image')
    fig.colorbar(im, ax=ax)

    ax = plt.subplot(323)
    im = ax.imshow(y_gt[0, :, :])
    plt.title('Ground-truth 1st channel')
    fig.colorbar(im, ax=ax)

    ax = plt.subplot(324)
    im = ax.imshow(y_gt[1, :, :])
    plt.title('Ground-truth 2nd channel')
    fig.colorbar(im, ax=ax)

    ax = plt.subplot(325)
    im = ax.imshow(y_gt[2, :, :])
    plt.title('Ground-truth 3rd channel')
    fig.colorbar(im, ax=ax)

    ax = plt.subplot(326)
    im = ax.imshow(y_gt[3, :, :])
    plt.title('Ground-truth 4th channel')
    fig.colorbar(im, ax=ax)
    plt.show()

