# -*- coding: utf-8 -*-
"""
Created on 18/08/2020 9:32 am

@author: Soan Duong, UOW
"""
# Standard library imports
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

# Third party imports
import torch
from PIL import Image
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import torch.nn.functional as F
from utils.metrics import DynamicClassMetrics
# Local application imports
from tqdm import tqdm


def hsi_read_header(file_name, verbose=False):
    """
    Load the information of a hyperspectral image from a header file (.hdr)
    :param file_name: file path of the header hsi file
    :param verbose: bool value to display the result (defaul: False)
    :return: 5 params
        - n_samples: number of n_samples in the image (width)
        - lines: number of lines in the image (height)
        - bands: number of bands (wave lengths)
        - data_type: data type stored in the data file
        - wave_lengths: list of wave lengths used to acquired the data
    """
    # Open a file for reading
    f = open(file_name, 'r')

    # Read all the lines in the header file
    text = f.readlines()

    # Close the file
    f.close()

    n = 0
    while n < len(text):
        text_line = text[n].replace('\n', '')
        # Get number of samples (width)
        if 'samples' in text_line:
            n_samples = int(text_line.split(' ')[-1])

        # Get number of lines (height)
        if 'lines' in text_line:
            n_lines = int(text_line.split(' ')[-1])

        # Get number of bands/wave lengths
        if 'bands' in text_line and not(' bands' in text_line):
            n_bands = int(text_line.split(' ')[-1])

        # Get the data type
        if 'data type' in text_line:
            data_type = int(text_line.split(' ')[-1])

        # Get the wave length values
        if 'Wavelength' in text_line:
            wave_lengths = np.zeros(n_bands)
            for k in range(n_bands):
                n = n + 1
                text_line = text[n].replace(',\n', '').replace(' ', '')
                wave_lengths[k] = float(text_line)
            break
        n = n + 1

    # Convert the data_type into the string format
    if data_type == 1:
        data_type = 'int8'
    elif data_type == 2:
        data_type = 'int16'
    elif data_type == 3:
        data_type = 'int32'
    elif data_type == 4:
        data_type = 'float32'
    elif data_type == 5:
        data_type = 'double'
    elif data_type == 12:
        data_type = 'uint16'
    elif data_type == 13:
        data_type = 'uint32'
    elif data_type == 14:
        data_type = 'int64'
    elif data_type == 15:
        data_type = 'uint64'

    if verbose:     # display the outputs if it is necessary
        print('Image width = %d' % n_samples)
        print('Image height = %d' % n_lines)
        print('Bands = %d' % n_bands)
        print('Data type = %s' % data_type)

    return n_samples, n_lines, n_bands, data_type, wave_lengths


def hsi_read_data(file_name, sorted=True):
    """
    Read the image cube from the raw hyperspectral image file (.raw)
    :param file_name: file path of the raw hsi file
    :param sorted: bool value to sort the image cube in the ascending of wave_lengths
    :return: 2 params
        - img: image cube in the shape of [n_lines, n_samples, n_bands]
        - wave_lengths: list of wave lengths used to acquired the data
    """
    # Get the information from the header file
    hdr_file = file_name[:-4] + '.hdr'
    n_samples, n_lines, n_bands, data_type, wave_lengths = hsi_read_header(hdr_file)

    # Open the raw file
    f = open(file_name, 'rb')
    # Read the data in the raw file
    data = np.frombuffer(f.read(), dtype=data_type)

    # Close the file
    f.close()

    # Reshape the data into the 3D formar of lines x bands x samples]
    img = data.reshape([n_lines, n_bands, n_samples])

    # Permute the image into the correct format lines x samples x bands
    img = np.moveaxis(img, [0, 1, 2], [0, 2, 1])

    if sorted:
        # Get the sorted indices of wave_lengths in the ascending order
        indx = np.argsort(wave_lengths)

        # Get the sorted wave_lengths
        wave_lengths = wave_lengths[indx]

        # Sort the image cube in the ascending order of wave_lengths
        img = img[:, :, indx]

    return img, wave_lengths


def norm_inten(I, max_val=255):
    """
    Normalize intensities of I to the range of [0, max_val]
    :param I: ndarray
    :param max_val: maximum value of the normalized range, default = 255
    :return: normalized ndarray
    """
    I = I - np.min(I)
    I = (max_val/np.max(I)) * I

    return I


def hsi_img2rgb(img, wave_lengths=None):
    """
    Convert raw hsi image cube into a pseudo-color image
    :param img: 3D array of size H x W x bands
    :param wave_lengths: 1D array of wavelength bands
    :return: array of size H x W x 3, a pseudo-color image
    """
    # Get the indices of ascending-sorted wavelengths
    if wave_lengths is None:
        indx = list(range(img.shape[-1]))
    else:
        indx = np.argsort(wave_lengths)

    # Get the pseudo-red channel (slice of the longest wavelength)
    ind = indx[-1]
    r = norm_inten(img[:, :, ind])[..., np.newaxis]

    # Get the pseudo-green channel (slice of the median wavelength)
    ind = indx[len(indx)//2]
    g = norm_inten(img[:, :, ind])[..., np.newaxis]

    # Get the pseudo-blue channel (slice of the shortest wavelength)
    ind = indx[0]
    b = norm_inten(img[:, :, ind])[..., np.newaxis]

    # Concatenate the channels into a color image
    rgb_img = np.concatenate([r, g, b], axis=-1)

    return rgb_img.astype(np.uint8)

def visualize_att(dataloader, model, num_results = 3, sv_dir = '',
                                   labels={}):
    for id,label in labels.items():
        new_dir = os.path.join(sv_dir,'class_{}'.format(label))
        os.makedirs(new_dir,exist_ok=True)
    model.eval()
    max_count = 200
    with torch.no_grad():
        for index, batch in enumerate(tqdm(dataloader)):
            # 1. Get a minibatch data for training
            image, label, name = batch['input'], batch['ground_truth_seg'], batch['name']
            image = image.to('cuda')
            #image, label, _, name = batch
            name = name[0]
            h,w = image.size(2), image.size(3)
            atts, preds = model(image)
            #atts, preds = preds[:-num_results], preds[-num_results:]
            #print(len(a))
            #print(name)
            for clf_id in range(num_results-1):
                att = atts[clf_id]
                #print(torch.sum(att))
                att = F.interpolate(att,size=(h, w)).cpu().data.numpy()
                for id, label in labels.items():
                    new_dir = os.path.join(sv_dir, 'class_{}'.format(label))
                    att_ = att[:,id]
                    sv_path = os.path.join(new_dir, 'att_heatmap_{}_clf{}.jpg'.format(name,clf_id))
                    heatmap = cv2.normalize(att_[0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8U)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
                    if name == '000001' and label == 'creature':
                        print(clf_id)
                        print(sv_path)
                        #print(heatmap)
                        print(np.mean(heatmap))
                    cv2.imwrite(sv_path, heatmap)
                    del att_
            if index > max_count:
                return

# def show_visual_results(x, y_gt, y_pr, classes=[0, 1, 2, 3, 4],
#                         show_visual=0, comet=None, fig_name=""):
#     """
#     Show the pseudo-color, ground-truth, and output images
#     :param x: array of size (batchsize, n_bands, H, W)
#     :param y_gt: array of size (batchsize, 1, H, W)
#     :param y_pr: array of size (batchsize, n_classes, H, W)
#     :param classes: list of class labels
#     :param show_visual: boolean to display the figure or not
#     :param comet: comet logger object
#     :param fig_name: string as the figure name for the comet logger
#     :return:
#     """
#     # Select the first image in the batch to display
#     y_gt = y_gt[0, ...]                      # of size (H, W)
#     y_pr = y_pr[0, ...]                         # of size (n_classes, H, W)
#     x = x[0, ...]                               # of size (n_bands, H, W)
#     x = np.moveaxis(x, [0, 1, 2], [2, 0, 1])    # of size (H, W, n_bands)

#     # Convert the probability into the segmentation result image
#     y_pr = convert_prob2seg(y_pr, classes)

#     # Set figure to display
#     h = 2
#     if plt.fignum_exists(h):  # close if the figure existed
#         plt.close()
#     fig = plt.figure(figsize=(9, 2))
#     fig.subplots_adjust(bottom=0.5)

#     # Set colormap
#     colors = ['black', 'green', 'blue', 'red', 'yellow']
#     cmap_ygt = mpl.colors.ListedColormap(colors[np.int(np.min(y_gt)):np.int(np.max(y_gt)) + 1])
#     cmap_ypr = mpl.colors.ListedColormap(colors[np.int(np.min(y_pr)):np.int(np.max(y_pr)) + 1])
#     # print('Input classes = ', np.unique(y_gt))
#     # print('Output classes = ', np.unique(y_pr))

#     # Plot the pseudo-color image
#     plt.subplot(131)
#     plt.imshow(hsi_img2rgb(x))
#     plt.title('Pseudo-color image')

#     # Plot the ground-truth image
#     ax = plt.subplot(132)
#     im = ax.imshow(y_gt, cmap=cmap_ygt)
#     plt.title('Ground-truth image')
#     fig.colorbar(im, ax=ax)

#     # Plot the predicted segmentation image
#     ax = plt.subplot(133)
#     im = ax.imshow(y_pr, cmap=cmap_ypr)
#     plt.title('Predicted segmentation image')
#     fig.colorbar(im, ax=ax)
#     plt.savefig(fig_name)

#     # if show_visual:
#     #     plt.show()

#     if comet is not None:
#         comet.log_figure(figure_name=fig_name, figure=fig)


def create_exp_dir(path, visual_folder=False):
    if not os.path.exists(path):
        os.mkdir(path)
        if visual_folder is True:
            os.mkdir(path + '/visual')  # for visual results
    else:
        print("DIR already existed.")
    print('Experiment dir : {}'.format(path))


def prepare_device(n_gpu_use=1):
    """
    Setup GPU device if it is available, move the model into the configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There\'s no GPU available on this machine,"
            "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available "
            "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def get_experiment_dataloaders(cfg):
    """
    Get and return the train, validation, and test dataloaders for spatial transcriptomics.
    :param cfg: dict that contains the required settings for the dataloaders
    :return: train, validation, and test dataloaders
    """
    print("Loading Spatial Transcriptomics dataset...")
    print('dataset_dir:', cfg['dataset_dir'])
    print('classes:', cfg['classes'])
    print('batch_size:', cfg['batch_size'])
    print('num_workers:', cfg['num_workers'])
    
    # Import and use spatial transcriptomics dataloader
    from utils.datasets import get_spatial_transcriptomics_dataloaders
    return get_spatial_transcriptomics_dataloaders(
        data_dir=cfg['dataset_dir'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers']
    )

def init_obj(module_name, module_args, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.
    `object = config.init_obj('name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    assert all([k not in module_args for k in
                kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


def convert_prob2seg_dynamic(y_prob, available_classes, y_gt=None, mask=None, ignore_index=0):
    """
    Convert class-probability image into segmentation with dynamic classes
    :param y_prob: class-probability image with size n_classes x H x W
    :param available_classes: list of classes present in this sample (e.g., [1,2,3] or [3,4,5])
    :param y_gt: ground truth for masking (optional)
    :param mask: whether to apply masking
    :param ignore_index: index to ignore (default: 0)
    :return: 2D array, segmentation image with size H x W
    """
    if mask and ignore_index == 0:
        y_prob[0] = 0  # Zero out background probability
    
    y_class = np.argmax(y_prob, axis=0)
    
    # Initialize segmentation with ignore_index
    y_seg = np.full((y_prob.shape[1], y_prob.shape[2]), ignore_index, dtype=np.int8)
    
    # Map probability indices to actual class labels
    for prob_idx, class_label in enumerate(available_classes):
        if class_label != ignore_index:  # Skip ignore class
            indx = np.where(y_class == prob_idx)
            if len(indx[0]) > 0:
                y_seg[indx] = class_label
    
    # Apply ignore mask if provided
    if mask and y_gt is not None:
        y_seg[y_gt == ignore_index] = ignore_index
    
    return y_seg


def compute_confusion_matrix_dynamic(y_gt, y_pr, all_possible_classes=None, ignore_index=0):
    """
    Compute confusion matrix handling samples with different class sets
    :param y_gt: ground truth labels
    :param y_pr: predictions (probabilities or labels)
    :param all_possible_classes: all possible classes across dataset (e.g., [1,2,3,4,5])
    :param ignore_index: index to ignore
    :return: confusion matrix
    """
    if all_possible_classes is None:
        # Automatically determine all classes from ground truth
        all_classes = []
        for k in range(y_gt.shape[0]):
            unique_classes = np.unique(y_gt[k])
            all_classes.extend(unique_classes)
        all_possible_classes = sorted(list(set(all_classes)))
        # Remove ignore_index if present
        if ignore_index in all_possible_classes:
            all_possible_classes.remove(ignore_index)
    
    cm = np.zeros((len(all_possible_classes), len(all_possible_classes)), dtype=np.int64)
    
    for k in range(y_gt.shape[0]):
        # Get predictions and ground truth for this sample
        if len(y_pr[k].shape) > 2:  # If predictions are probabilities
            y_prk = np.argmax(y_pr[k], axis=0)
            # Get available classes for this sample
            available_classes = []
            for i in range(y_pr[k].shape[0]):
                if np.any(y_pr[k][i] > 0):  # Check if this class channel has any activation
                    available_classes.append(all_possible_classes[i] if i < len(all_possible_classes) else ignore_index)
        else:
            y_prk = y_pr[k]
            available_classes = all_possible_classes
        
        y_gtk = y_gt[k].squeeze() if len(y_gt[k].shape) > 2 else y_gt[k]
        
        # Create mask for valid pixels
        valid_mask = (y_gtk != ignore_index)
        
        if valid_mask.sum() == 0:
            continue
        
        # Apply mask
        y_prk_valid = y_prk[valid_mask]
        y_gtk_valid = y_gtk[valid_mask]
        
        # Use sklearn with explicit labels
        sample_cm = sklearn_confusion_matrix(
            y_gtk_valid.flatten(), 
            y_prk_valid.flatten(), 
            labels=all_possible_classes
        )
        
        cm += sample_cm
    
    return cm, all_possible_classes


def compute_eval_from_cm_robust(confusion_matrix, class_names=None):
    """
    Calculate performance metrics with robust handling of missing classes
    :param confusion_matrix: confusion matrix
    :param class_names: optional list of class names for better reporting
    :return: metrics dictionary
    """
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    
    # Calculate pixel accuracy
    pixel_acc = np.trace(confusion_matrix) / (np.sum(confusion_matrix) + eps)
    
    # Calculate per-class accuracy
    per_class_acc = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + eps)
    mean_acc = np.nanmean(per_class_acc)
    
    # Calculate IoU for each class
    true_positives = np.diag(confusion_matrix)
    false_positives = np.sum(confusion_matrix, axis=0) - true_positives
    false_negatives = np.sum(confusion_matrix, axis=1) - true_positives
    
    IoU_array = true_positives / (true_positives + false_positives + false_negatives + eps)
    mean_IoU = np.nanmean(IoU_array)
    
    # Calculate Dice coefficient
    dice_array = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives + eps)
    mean_dice = np.nanmean(dice_array)
    
    # Calculate F1 score
    precision = true_positives / (true_positives + false_positives + eps)
    recall = true_positives / (true_positives + false_negatives + eps)
    f1_array = 2 * (precision * recall) / (precision + recall + eps)
    mean_f1 = np.nanmean(f1_array)
    
    # Calculate Cohen's Kappa
    total_sum = np.sum(confusion_matrix)
    observed_accuracy = np.trace(confusion_matrix) / total_sum
    expected_accuracy = (np.sum(confusion_matrix, axis=0) / total_sum) @ (np.sum(confusion_matrix, axis=1) / total_sum)
    kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy + eps)
    
    # Create results dictionary
    results = {
        'pixel_accuracy': pixel_acc,
        'mean_accuracy': mean_acc,
        'per_class_accuracy': per_class_acc,
        'mean_IoU': mean_IoU,
        'IoU_per_class': IoU_array,
        'mean_dice': mean_dice,
        'dice_per_class': dice_array,
        'mean_f1': mean_f1,
        'f1_per_class': f1_array,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'kappa': kappa
    }
    
    # Add class names if provided
    if class_names is not None:
        for metric in ['per_class_accuracy', 'IoU_per_class', 'dice_per_class', 'f1_per_class', 'precision_per_class', 'recall_per_class']:
            results[f'{metric}_named'] = dict(zip(class_names, results[metric]))
    
    return results


def show_visual_results_dynamic(x, y_gt, y_pr, available_classes, all_classes=None,
                               show_visual=False, comet=None, fig_name="", ignore_index=0):
    """
    Show visualization with support for dynamic classes
    :param x: input image (batchsize, n_bands, H, W)
    :param y_gt: ground truth (batchsize, 1, H, W)
    :param y_pr: predictions (batchsize, n_classes, H, W)
    :param available_classes: classes available in this sample
    :param all_classes: all possible classes in dataset
    :param show_visual: whether to display
    :param comet: comet logger
    :param fig_name: figure name
    :param ignore_index: index to ignore
    :return: None
    """
    # Select first image in batch
    y_gt = y_gt[0, ...]
    y_pr = y_pr[0, ...]
    x = x[0, ...]
    x = np.moveaxis(x, [0, 1, 2], [2, 0, 1])
    
    # Convert probabilities to segmentation
    y_pr_seg = convert_prob2seg_dynamic(y_pr, available_classes, y_gt, mask=True, ignore_index=ignore_index)
    
    # Set up figure
    if plt.fignum_exists(1):
        plt.close()
    fig = plt.figure(figsize=(12, 4))
    fig.subplots_adjust(bottom=0.3)
    
    # Create colormap for all possible classes
    if all_classes is None:
        all_classes = sorted(list(set(np.unique(y_gt)) | set(available_classes)))
        if ignore_index in all_classes:
            all_classes.remove(ignore_index)
        all_classes = [ignore_index] + all_classes  # Add ignore_index at the beginning
    
    # Define colors (extend as needed)
    base_colors = ['black', 'green', 'blue', 'red', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'brown']
    colors = base_colors[:len(all_classes)]
    
    # Create colormaps
    cmap_gt = mpl.colors.ListedColormap(colors[:np.max(all_classes)+1])
    cmap_pr = mpl.colors.ListedColormap(colors[:np.max(all_classes)+1])
    
    # Plot pseudo-color image
    plt.subplot(131)
    plt.imshow(hsi_img2rgb(x))
    plt.title('Pseudo-color image')
    plt.axis('off')
    
    # Plot ground truth
    ax = plt.subplot(132)
    im = ax.imshow(y_gt, cmap=cmap_gt, vmin=0, vmax=max(all_classes))
    plt.title(f'Ground-truth\nClasses: {sorted(np.unique(y_gt))}')
    plt.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot prediction
    ax = plt.subplot(133)
    im = ax.imshow(y_pr_seg, cmap=cmap_pr, vmin=0, vmax=max(all_classes))
    plt.title(f'Prediction\nClasses: {sorted(np.unique(y_pr_seg))}')
    plt.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if fig_name:
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    
    if show_visual:
        plt.show()
    
    if comet is not None:
        comet.log_figure(figure_name=fig_name, figure=fig)

class TrainerWithDynamicClasses:
    """Training utilities for spatial transcriptomics with dynamic classes"""
    
    def __init__(self, model, optimizer, loss_fn, device='cuda', ignore_index=0):
        """
        Initialize trainer with dynamic class support
        
        Args:
            model: The neural network model to train
            optimizer: Optimizer for training
            loss_fn: Loss function (should support ignore_index)
            device: Device to run training on ('cuda' or 'cpu')
            ignore_index: Index to ignore in loss calculation (default: 0)
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.ignore_index = ignore_index
        
        # Metrics tracking
        self.train_metrics = DynamicClassMetrics(ignore_index=ignore_index)
        self.val_metrics = DynamicClassMetrics(ignore_index=ignore_index)
        
        # Class tracking across training
        self.global_class_tracker = {}
        
    def train_epoch(self, dataloader, epoch):
        """
        Train for one epoch with dynamic class tracking
        
        Args:
            dataloader: Training dataloader
            epoch: Current epoch number
            
        Returns:
            dict: Training metrics for this epoch
        """
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        pbar = tqdm(dataloader, ncols=80, desc=f'Training Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Get batch data
            inputs = batch['input'].to(self.device)
            targets = batch['ground_truth_seg'].to(self.device)
            sample_ids = batch.get('name', [f'batch_{batch_idx}_sample_{i}' for i in range(inputs.size(0))])
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Handle different model outputs (some return features + outputs)
            model_output = self.model(inputs)
            if isinstance(model_output, tuple):
                # If model returns (features, outputs) or (features, outputs_list)
                features, outputs = model_output
                # Take the last output if it's a list (for multi-head models)
                if isinstance(outputs, list):
                    outputs = outputs[-1]
            else:
                # Single output
                outputs = model_output
            
            # Compute loss
            if targets.dim() == 4:
                targets = targets.squeeze(1)  # Remove channel dimension if present
            
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update metrics with dynamic class tracking
            self.train_metrics.update(outputs, targets, sample_ids, probabilities=True)
            
            # Track classes for this epoch
            for i, sample_id in enumerate(sample_ids):
                sample_classes = set(np.unique(targets[i].cpu().numpy())) - {self.ignore_index}
                if sample_id not in self.global_class_tracker:
                    self.global_class_tracker[sample_id] = sample_classes
                else:
                    self.global_class_tracker[sample_id].update(sample_classes)
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Compute metrics for this epoch
        metrics = self.train_metrics.compute_metrics()
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def validate(self, dataloader):
        """
        Validate with dynamic class tracking
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        pbar = tqdm(dataloader, ncols=80, desc='Validating')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Get batch data
                inputs = batch['input'].to(self.device)
                targets = batch['ground_truth_seg'].to(self.device)
                sample_ids = batch.get('name', [f'val_batch_{batch_idx}_sample_{i}' for i in range(inputs.size(0))])
                
                # Forward pass
                model_output = self.model(inputs)
                if isinstance(model_output, tuple):
                    features, outputs = model_output
                    if isinstance(outputs, list):
                        outputs = outputs[-1]
                else:
                    outputs = model_output
                
                # Compute loss
                if targets.dim() == 4:
                    targets = targets.squeeze(1)
                    
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                
                # Update metrics
                self.val_metrics.update(outputs, targets, sample_ids, probabilities=True)
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Compute metrics for this epoch
        metrics = self.val_metrics.compute_metrics()
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def get_class_distribution_summary(self):
        """
        Get summary of class distribution across all samples
        
        Returns:
            dict: Summary of classes and their distribution
        """
        all_classes = set()
        class_frequency = {}
        
        for sample_id, classes in self.global_class_tracker.items():
            all_classes.update(classes)
            for cls in classes:
                class_frequency[cls] = class_frequency.get(cls, 0) + 1
        
        return {
            'all_classes': sorted(list(all_classes)),
            'class_frequency': class_frequency,
            'total_samples': len(self.global_class_tracker),
            'samples_per_class': {cls: count for cls, count in class_frequency.items()}
        }
    
    def save_checkpoint(self, epoch, best_metric, save_path):
        """
        Save training checkpoint
        
        Args:
            epoch: Current epoch
            best_metric: Best validation metric so far
            save_path: Path to save checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': best_metric,
            'class_tracker': self.global_class_tracker
        }
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load training checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            dict: Loaded checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_class_tracker = checkpoint.get('class_tracker', {})
        return checkpoint
    
    def get_learning_rate(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def set_learning_rate(self, lr):
        """Set learning rate for all parameter groups"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Usage example for dynamic class handling
class ClassTracker:
    """Helper class to track classes across samples and dataset"""
    
    def __init__(self, ignore_index=0):
        self.ignore_index = ignore_index
        self.all_classes = set()
        self.sample_classes = {}
    
    def update(self, sample_id, ground_truth, predictions=None):
        """Update class tracking with new sample"""
        # Extract classes from ground truth
        gt_classes = set(np.unique(ground_truth)) - {self.ignore_index}
        self.sample_classes[sample_id] = gt_classes
        self.all_classes.update(gt_classes)
        
        # Extract classes from predictions if provided
        if predictions is not None and len(predictions.shape) > 2:
            # For probability predictions, check which channels have activations
            pred_classes = set()
            for i in range(predictions.shape[0]):
                if np.any(predictions[i] > 0.1):  # Threshold for active channels
                    pred_classes.add(i + 1)  # Assuming classes start from 1
            self.all_classes.update(pred_classes)
    
    def get_all_classes(self):
        """Get sorted list of all classes"""
        return sorted(list(self.all_classes))
    
    def get_sample_classes(self, sample_id):
        """Get classes for specific sample"""
        return sorted(list(self.sample_classes.get(sample_id, set())))
    
def cohen_kappa_score(confusion):
    r"""Cohen's kappa: a statistic that measures inter-annotator agreement.

    This function computes Cohen's kappa [1]_, a score that expresses the level
    of agreement between two annotators on a classification problem. It is
    defined as

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement on the label
    assigned to any sample (the observed agreement ratio), and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly.
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels [2]_.

    Read more in the :ref:`User Guide <cohen_kappa>`.

    Parameters
    ----------
    weights : str, optional
        Weighting type to calculate the score. None means no weighted;
        "linear" means linear weighted; "quadratic" means quadratic weighted.


    Returns
    -------
    kappa : float
        The kappa statistic, which is a number between -1 and 1. The maximum
        value means complete agreement; zero or lower means chance agreement.
    """
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)
    w_mat = np.ones([n_classes, n_classes], dtype=int)
    w_mat.flat[:: n_classes + 1] = 0

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k

def test_running_time(dump_input, model, comet):
    model.eval()
    reps = 100
    dump_input = dump_input.cuda()
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((reps, 1))
    with torch.no_grad():
        for _ in range(50):
            model(dump_input)
        print("Start testing inference time...")
        for rep in range(reps):
            starter.record()
            _ = model(dump_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    #infer_time = total_time/reps
    infer_time = np.sum(timings) / reps
    infer_time /= 1000

    print("Inference time : {} seconds {} FPS".format(infer_time,1/infer_time))
    comet.log_metric(f'inference_time', infer_time)


def test_running_time_with_wrapper(dump_input, model, comet, n_heads):
    from models.sgr_wrapper import SGR_Wrapper
    from thop import profile, clever_format
    model.eval()

    infer_time = [0] * n_heads
    reps = 500
    dump_input = dump_input.cuda()
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for i in range(n_heads):
            sub_model = SGR_Wrapper(sgr_model=model.module, t_head=i+1).cuda()
            sub_model.eval()
            flops, params = profile(sub_model.to('cuda:0'), inputs=(dump_input.to('cuda:0'),), verbose=False)
            macs, params = clever_format([flops, params], "%.3f")
            print(f"Model {n_heads} FLOPS: {macs}, PARAMS: {params}")
            comet.log_other(f'Model {n_heads} trainable parameters', params)
            comet.log_other(f'Model {n_heads} Floating point operations per second (FLOPS)', flops)
            comet.log_other(f'Model {n_heads} Multiply accumulates per second (MACs)', macs)

            for _ in range(100):
                sub_model(dump_input)
            print(f"Start testing inference time for model {i+1}...")
            timings = np.zeros((reps, 1))
            for rep in range(reps):
                starter.record()
                _ = sub_model(dump_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
            infer_time[i] = np.sum(timings) / rep
            print(timings.mean())

    for i in range(n_heads):
        print("Inference time : {} seconds {} FPS".format(infer_time[i],1/(infer_time[i]/1000)))
        comet.log_metric(f'inference_time_{i}', infer_time[i])
        

# ------------------------------------------------------------------------------
# Main function for testing
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # file_name = '../../../02.data/hsi_data/200716_botanic_garden/200716_mp_0.raw'
    # file_name = '../../../02.data/hsi_data/200716_botanic_garden/200716_mp_26.raw'
    file_name = '../data/UOW-HSI/000426.raw'
    # file_name = '../data/UOW-HSI/000247.raw'
    # file_name = 'U:/02-Data/UOW-HSI/000426.raw'
    #
    # file_name = 'U:/02-Data/UOW-HSI/000399.raw'
    # file_name = 'U:/02-Data/UOW-HSI/000425.raw'
    # file_name = 'U:/02-Data/UOW-HSI/000570.raw'
    # file_name = 'U:/02-Data/UOW-HSI/000002.raw'
    bmp_file = file_name[:-4] + '.bmp'
    # Read hsi raw data
    img, wave_lengths = hsi_read_data(file_name)
    rgb = hsi_img2rgb(img)    # get pseudo-color image

    # Read ground-truth image
    bmp = Image.open(bmp_file)
    bmp = np.array(bmp.getdata()).reshape(bmp.size[1], bmp.size[0])

    colors = ['black', 'green', 'blue', 'red', 'yellow']
    cmap = mpl.colors.ListedColormap(colors[np.min(bmp):np.max(bmp)+1])
    fig = plt.figure(figsize=(3, 2))
    fig.subplots_adjust(bottom=0.5)
    plt.imshow(bmp, cmap=cmap)
    plt.show()
    # # Set colormap
    # colors = ['black', 'green', 'blue', 'red', 'yellow']
    # cmap = mpl.colors.ListedColormap(colors[np.min(bmp):np.max(bmp)+1])
    # # cmap.set_over('0.25')
    # # cmap.set_under('0.75')
    # bounds = [0, 1, 2, 3, 4]
    # labels = ['Other', 'Plant', 'Soil', 'Creature', 'Yellow']
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #
    # # Plot images
    # fig = plt.figure(figsize=(9, 2))
    # fig.subplots_adjust(bottom=0.5)
    # plt.subplot(131)
    # plt.imshow(rgb)
    # plt.title('Pseudo-color image')
    #
    # # Plot the ground-truth image
    # ax = plt.subplot(132)
    # im = ax.imshow(bmp, cmap=cmap)
    # plt.title('Ground-truth image')
    # fig.colorbar(im, ax=ax)
    #
    # # Plot the Float-converted ground-truth image
    # ax = plt.subplot(133)
    # im = ax.imshow(np.float32(bmp[np.newaxis, ...])[0, ...], cmap=cmap)
    # plt.title('Float-converted ground-truth image')
    # fig.colorbar(im, ax=ax)
    #
    # plt.show()
