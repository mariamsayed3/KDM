# -*- coding: utf-8 -*-
"""
Created on 20/08/2020 12:51 pm

@author: Soan Duong, UOW
"""
# Standard library imports
# Third party imports
import torch.nn as nn
import torch
import numpy as np
import sklearn.metrics  # Add this line

# Local application imports
from . import base
from . import functional as F
from .base import Activation
import torch.nn.functional as func



class DynamicClassMetrics:
    """Wrapper for handling metrics with dynamic classes across samples"""
    
    def __init__(self, ignore_index=0):
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset accumulated values"""
        self.all_targets = []
        self.all_predictions = []
        self.sample_classes = {}
        
    def update(self, predictions, targets, sample_ids=None, probabilities=False):
        """
        Update metrics with new batch
        :param predictions: Model predictions
        :param targets: Ground truth targets
        :param sample_ids: List of sample identifiers
        :param probabilities: Whether predictions are probabilities or class indices
        """
        if probabilities:
            # Convert probabilities to class predictions
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions
            
        # Store for later computation
        self.all_predictions.append(pred_classes.cpu())
        self.all_targets.append(targets.cpu())
        
        # Track classes per sample
        if sample_ids is not None:
            for i, sample_id in enumerate(sample_ids):
                if sample_id not in self.sample_classes:
                    self.sample_classes[sample_id] = set()
                
                # Get unique classes in this sample (excluding ignore_index)
                sample_targets = targets[i].cpu().numpy()
                sample_classes = set(np.unique(sample_targets)) - {self.ignore_index}
                self.sample_classes[sample_id].update(sample_classes)
    
    def compute_metrics(self, mask=True):
        """
        Compute all metrics across accumulated data
        :param mask: Whether to apply ignore_index masking
        :return: Dictionary of computed metrics
        """
        if not self.all_predictions:
            return {}
            
        # Concatenate all data
        all_preds = torch.cat(self.all_predictions, dim=0)
        all_targets = torch.cat(self.all_targets, dim=0)
        
        # Get all unique classes across dataset
        all_classes = set()
        for classes in self.sample_classes.values():
            all_classes.update(classes)
        all_classes = sorted(list(all_classes))
        
        if not all_classes:  # Fallback if no classes found
            all_classes = [1, 2, 3, 4, 5]
        
        # Convert to probability format for metric classes
        num_classes = max(all_classes) + 1
        preds_prob = torch.zeros(all_preds.shape[0], num_classes, *all_preds.shape[1:])
        preds_prob.scatter_(1, all_preds.unsqueeze(1).long(), 1)
        
        # Initialize metrics with dynamic classes
        metrics = {
            'accuracy': Accuracy(ignore_index=self.ignore_index),
            'mean_iou': MeanIoU(ignore_index=self.ignore_index, classes=all_classes),
            'macro_iou': MacroIoU(ignore_index=self.ignore_index, classes=all_classes),
            'dice': Dice(ignore_index=self.ignore_index, classes=all_classes),
        }
        
        # Compute metrics
        results = {}
        for name, metric in metrics.items():
            results[name] = metric(preds_prob, all_targets, mask=mask).item()
        
        # Add standalone metrics
        results['kappa'] = kappa(preds_prob, all_targets, mask=mask, ignore_index=self.ignore_index, classes=all_classes)
        results['average_accuracy'] = average_accuracy(preds_prob, all_targets, mask=mask, ignore_index=self.ignore_index, classes=all_classes)
        
        # Add per-class metrics
        results['per_class_iou'] = self._compute_per_class_iou(all_preds, all_targets, all_classes)
        results['per_class_dice'] = self._compute_per_class_dice(all_preds, all_targets, all_classes)
        
        return results
    
    def _compute_per_class_iou(self, predictions, targets, classes):
        """Compute IoU for each class separately"""
        iou_scores = {}
        for cls in classes:
            pred_mask = (predictions == cls)
            target_mask = (targets == cls)
            
            # Apply ignore mask
            valid_mask = (targets != self.ignore_index)
            pred_mask = pred_mask & valid_mask
            target_mask = target_mask & valid_mask
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union > 0:
                iou_scores[f'class_{cls}'] = (intersection / union).item()
            else:
                iou_scores[f'class_{cls}'] = 0.0
                
        return iou_scores
    
    def _compute_per_class_dice(self, predictions, targets, classes):
        """Compute Dice score for each class separately"""
        dice_scores = {}
        for cls in classes:
            pred_mask = (predictions == cls)
            target_mask = (targets == cls)
            
            # Apply ignore mask
            valid_mask = (targets != self.ignore_index)
            pred_mask = pred_mask & valid_mask
            target_mask = target_mask & valid_mask
            
            intersection = (pred_mask & target_mask).sum().float()
            total = pred_mask.sum() + target_mask.sum()
            
            if total > 0:
                dice_scores[f'class_{cls}'] = (2.0 * intersection / total).item()
            else:
                dice_scores[f'class_{cls}'] = 0.0
                
        return dice_scores
    
    def get_summary(self):
        """Get summary of classes found across samples"""
        return {
            'total_samples': len(self.sample_classes),
            'all_classes_found': sorted(list(set().union(*self.sample_classes.values()))),
            'classes_per_sample': {k: sorted(list(v)) for k, v in self.sample_classes.items()}
        }


class MeanIoU(base.Metric):
    __name__ = 'mean_iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, 
                 ignore_index=0, classes=[1, 2, 3, 4, 5], **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, y_pr, y_gt, mask=True):
        y_pr = self.activation(y_pr)
        
        # Apply ignore logic
        if mask:
            y_pr[:, self.ignore_index] = 0
        pred = torch.argmax(y_pr, dim=1)
        if mask:
            pred[y_gt == self.ignore_index] = self.ignore_index
            
        # Calculate IoU for each class
        ious = []
        for cls in self.classes:
            pred_mask = (pred == cls)
            gt_mask = (y_gt == cls)
            
            intersection = (pred_mask & gt_mask).sum().float()
            union = (pred_mask | gt_mask).sum().float()
            
            if union == 0:
                # If class doesn't exist in both pred and gt, skip it
                continue
            else:
                iou = intersection / union
            ious.append(iou)
        
        # Return mean IoU only for classes that exist
        return torch.tensor(ious).mean() if ious else torch.tensor(0.0)


class IoU(base.Metric):
    __name__ = 'iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, ignore_index=0, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(y_pr, y_gt)


def iou(pred, target, mask=True, ignore_index=0, classes=[1, 2, 3, 4, 5]):
    if mask:
        pred[:, ignore_index] = 0
    pred = torch.argmax(pred, dim=1)
    if mask:
        pred[target == ignore_index] = ignore_index
    
    # Create mask for valid pixels
    valid_mask = (target != ignore_index)
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    
    if len(pred_valid) == 0:
        return 0.0
    
    return sklearn.metrics.jaccard_score(
        target_valid.flatten().detach().cpu().numpy(),
        pred_valid.flatten().detach().cpu().numpy(),
        labels=classes,
        average='macro',
        zero_division=0
    )


class MacroIoU(base.Metric):
    __name__ = 'macro_iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, 
                 ignore_index=0, classes=[1, 2, 3, 4, 5], **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, y_pr, y_gt, mask=True):
        y_pr = self.activation(y_pr)
        if mask:
            y_pr[:, self.ignore_index] = 0
        pred = torch.argmax(y_pr, dim=1)
        if mask:
            pred[y_gt == self.ignore_index] = self.ignore_index
            
        # Create mask for valid pixels
        valid_mask = (y_gt != self.ignore_index)
        pred_valid = pred[valid_mask]
        gt_valid = y_gt[valid_mask]
        
        if len(pred_valid) == 0:
            return 0.0
        
        return sklearn.metrics.jaccard_score(
            gt_valid.flatten().detach().cpu().numpy(),
            pred_valid.flatten().detach().cpu().numpy(),
            labels=self.classes,
            average='macro',
            zero_division=0
        )


class Fscore(base.Metric):
    __name__ = 'f1_score'

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(y_pr, y_gt,
                         eps=self.eps,
                         beta=self.beta,
                         threshold=self.threshold,
                         ignore_channels=self.ignore_channels)


class Accuracy(base.Metric):
    __name__ = 'accuracy_score'

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, ignore_index=0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt, mask=True):
        y_pr = self.activation(y_pr)
        if mask:
            y_pr[:, self.ignore_index] = 0
        pred = torch.argmax(y_pr, dim=1)
        if mask:
            pred[y_gt == self.ignore_index] = self.ignore_index
        
        # Create mask for valid pixels
        valid_mask = (y_gt != self.ignore_index)
        pred_valid = pred[valid_mask]
        gt_valid = y_gt[valid_mask]
        
        if len(pred_valid) == 0:
            return 0.0
        
        return sklearn.metrics.accuracy_score(
            gt_valid.flatten().detach().cpu().numpy(),
            pred_valid.flatten().detach().cpu().numpy()
        )


class Recall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(y_pr, y_gt,
                        eps=self.eps,
                        threshold=self.threshold,
                        ignore_channels=self.ignore_channels)


class Precision(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(y_pr, y_gt,
                           eps=self.eps,
                           threshold=self.threshold,
                           ignore_channels=self.ignore_channels)


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class Dice(base.Metric):
    __name__ = 'dice_score'

    def __init__(self, smooth=1.0, threshold=0.5, activation=None, ignore_channels=None, 
                 ignore_index=0, classes=[1, 2, 3, 4, 5], **kwargs):
        super().__init__(**kwargs)
        self.smooth = 1.0
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, y_pr, y_gt, mask=True):
        y_pr = self.activation(y_pr)
        if mask:
            y_pr[:, self.ignore_index] = 0
        y_pr = torch.argmax(y_pr, dim=1)
        if mask:
            y_pr[y_gt == self.ignore_index] = self.ignore_index
            
        # Create mask for valid pixels
        valid_mask = (y_gt != self.ignore_index)
        pred_valid = y_pr[valid_mask]
        gt_valid = y_gt[valid_mask]
        
        if len(pred_valid) == 0:
            return 0.0
        
        return sklearn.metrics.f1_score(
            gt_valid.flatten().detach().cpu().numpy(),
            pred_valid.flatten().detach().cpu().numpy(),
            labels=self.classes,
            average='macro',
            zero_division=0
        )


# Standalone functions
def accuracy(pred, target, mask=True, ignore_index=0, classes=[1, 2, 3, 4, 5]):
    if mask:
        pred[:, ignore_index] = 0
    pred = torch.argmax(pred, dim=1)
    if mask:
        pred[target == ignore_index] = ignore_index
    
    # Create mask for valid pixels
    valid_mask = (target != ignore_index)
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    
    if len(pred_valid) == 0:
        return 0.0
    
    acc = sklearn.metrics.accuracy_score(
        target_valid.flatten().detach().cpu().numpy(),
        pred_valid.flatten().detach().cpu().numpy()
    )
    return acc


def dice_coeff(pred, target, mask=True, ignore_index=0, classes=[1, 2, 3, 4, 5]):
    if mask:
        pred[:, ignore_index] = 0
    pred = torch.argmax(pred, dim=1)
    if mask:
        pred[target == ignore_index] = ignore_index
    
    # Create mask for valid pixels
    valid_mask = (target != ignore_index)
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    
    if len(pred_valid) == 0:
        return 0.0
    
    # Use explicit labels parameter to handle missing classes
    dice = sklearn.metrics.f1_score(
        target_valid.flatten().detach().cpu().numpy(),
        pred_valid.flatten().detach().cpu().numpy(),
        labels=classes,  # Include all possible classes
        average='macro',
        zero_division=0  # Return 0 for missing classes
    )
    return dice


def average_accuracy(pred, target, mask=True, ignore_index=0, classes=[1, 2, 3, 4, 5]):
    if mask:
        pred[:, ignore_index] = 0
    pred = torch.argmax(pred, dim=1)
    if mask:
        pred[target == ignore_index] = ignore_index
    
    # Create mask for valid pixels
    valid_mask = (target != ignore_index)
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    
    if len(pred_valid) == 0:
        return 0.0
    
    # Use explicit labels and zero_division handling
    aa = sklearn.metrics.recall_score(
        target_valid.flatten().detach().cpu().numpy(),
        pred_valid.flatten().detach().cpu().numpy(),
        labels=classes,
        average='macro',
        zero_division=0  # Return 0 for missing classes
    )
    return aa


def kappa(pred, target, mask=True, ignore_index=0, classes=[1, 2, 3, 4, 5]):
    if mask:
        pred[:, ignore_index] = 0
    pred = torch.argmax(pred, dim=1)
    if mask:
        pred[target == ignore_index] = ignore_index
    
    # Create mask for valid pixels
    valid_mask = (target != ignore_index)
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    
    if len(pred_valid) == 0:
        return 0.0
    
    # For kappa, sklearn automatically handles missing classes
    kappa_score = sklearn.metrics.cohen_kappa_score(
        target_valid.flatten().detach().cpu().numpy(),
        pred_valid.flatten().detach().cpu().numpy(),
        labels=classes
    )
    return kappa_score


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input"""
    def __init__(self, weight=None, ignore_index=0, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = func.softmax(predict, dim=1)
        
        # Count only non-ignored classes
        valid_classes = 0
        
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss
                valid_classes += 1

        return total_loss / max(valid_classes, 1)  # Avoid division by zero