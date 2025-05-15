# -*- coding: utf-8 -*-
"""
Created on 23/11/2020 1:36 pm

@author: Soan Duong, Hieu Phan UOW
"""
# Standard library imports
# Third party imports
import torch
import torch.nn as nn

# Local application imports
from . import base
from . import functional as F
import torch.nn.functional as func
from .base import Activation


class SpatialTranscriptomicsLoss(nn.Module):
    """Combined loss for spatial transcriptomics with dynamic class support"""
    
    def __init__(self, 
                 ce_weight=1.0,
                 dice_weight=0.5,
                 focal_weight=0.3,
                 ignore_index=0,
                 focal_gamma=2.0,
                 class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ignore_index = ignore_index
        
        # Initialize loss components
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        
    def forward(self, predictions, targets, sample_info=None):
        """
        Forward pass with support for dynamic classes
        :param predictions: Model predictions (B, C, H, W)
        :param targets: Ground truth (B, H, W) or (B, 1, H, W)
        :param sample_info: Optional dict with sample-specific info
        :return: Combined loss
        """
        if targets.dim() == 4:
            targets = targets.squeeze(1)
            
        # Cross-entropy loss
        ce_loss = self.ce_loss(predictions, targets.long())
        
        # Create one-hot for dice loss (excluding ignore_index)
        num_classes = predictions.shape[1]
        targets_one_hot = torch.zeros_like(predictions)
        
        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index)
        
        # Fill one-hot encoding only for valid pixels
        for i in range(num_classes):
            if i != self.ignore_index:
                class_mask = (targets == i) & valid_mask
                targets_one_hot[:, i][class_mask] = 1.0
        
        # Dice loss
        dice_loss = self.dice_loss(predictions, targets_one_hot)
        
        # Focal loss (apply mask afterwards)
        focal_loss = self.focal_loss(predictions, targets.long())
        focal_loss = focal_loss * valid_mask.float()
        focal_loss = focal_loss.sum() / (valid_mask.sum() + 1e-8)
        
        # Combine losses
        total_loss = (self.ce_weight * ce_loss + 
                     self.dice_weight * dice_loss + 
                     self.focal_weight * focal_loss)
        
        return total_loss


class JaccardLoss(base.Loss):
    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)

        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_gt, y_pr):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(y_pr, y_gt,
                             eps=self.eps,
                             threshold=None,
                             ignore_channels=self.ignore_channels)


class DiceLoss(base.Loss):
    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)

        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_gt, y_pr):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(y_pr, y_gt,
                             beta=self.beta,
                             eps=self.eps,
                             threshold=None,
                             ignore_channels=self.ignore_channels)


class FocalLoss(nn.Module):

    def __init__(self, weight=None,
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = nn.functional.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(FSCELoss, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, inputs, *targets, weights=None, mask=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    if mask:
                        hasLabel = (target != 0).unsqueeze(1).long()
                        loss += weights[i] * self.criterion(inputs[i] * hasLabel, target)
                    else:
                        loss += weights[i] * self.criterion(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    if mask:
                        hasLabel = (target != 0).unsqueeze(1).long()
                        loss += weights[i] * self.criterion(inputs[i] * hasLabel, target)
                    else:
                        loss += weights[i] * self.criterion(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            if mask:
                hasLabel = (target != 0).unsqueeze(1).long()
                loss = self.criterion(inputs * hasLabel, target)
            else:
                loss = self.criterion(inputs, target)
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = func.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        score = score[-1]
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = func.interpolate(
                input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=3, ignore_label=0, weight=None, feat_w=0, resp_w=0):  # Changed default to 0
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.feat_w = feat_w
        self.resp_w = resp_w
        self.ignore_label = ignore_label  # Store ignore_label
        self.response_loss = KDPixelWiseCE(temperature=temperature, ignore_label=ignore_label)
        self.feature_loss = KDFeat(ignore_label=ignore_label)
        self.classification_loss = FSCELoss(ignore_label=ignore_label, weight=weight)

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = func.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()

    def forward(self, student_output, teacher_output, student_feat, teacher_feat, output):
        # Ensure teacher and student outputs have same size
        if student_output.size() != teacher_output.size():
            teacher_output = func.interpolate(teacher_output, size=(student_output.size(2), student_output.size(3)), mode='nearest')
        
        # Scale target to match output size
        output = self._scale_target(output, (student_output.size(2), student_output.size(3)))
        
        # Response distillation loss
        loss = self.resp_w * self.response_loss(student_output, teacher_output, output)
        
        # Feature distillation loss
        if student_feat.size() != teacher_feat.size():
            teacher_feat = func.interpolate(teacher_feat, (student_feat.size(-2), student_feat.size(-1)))
        loss += self.feat_w * self.feature_loss(
            torch.sum(student_feat, dim=1).unsqueeze(1), 
            torch.sum(teacher_feat, dim=1).unsqueeze(1),
            output
        )
        
        # Classification loss
        loss += self.classification_loss(student_output, output)
        return loss

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = func.interpolate(input=score, size=(h, w), mode='bilinear')
        pred = func.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

def SpatialSoftmax(feature):
    feature = feature.view(feature.shape[0], feature.shape[1], -1)
    softmax_attention = func.softmax(feature, dim=-1)
    return softmax_attention



class KDFeat(nn.Module):
    def __init__(self, ignore_label=0):
        super(KDFeat, self).__init__()
        self.criterion = nn.MSELoss()
        self.ignore_label = ignore_label

    def forward(self, f_S, f_T, target=None):
        """
        f_S - student feature map
        f_T - teacher feature map  
        target - ground truth for masking (optional)
        """
        # Ensure same size
        if f_S.size() != f_T.size():
            f_T = func.interpolate(f_T, size=(f_S.shape[2], f_S.shape[3]), mode='bilinear', align_corners=True)
        
        # Calculate attention maps
        f_S = torch.sum(f_S * f_S, dim=1, keepdim=True)
        f_S = SpatialSoftmax(f_S)
        f_T = torch.sum(f_T * f_T, dim=1, keepdim=True)
        f_T = SpatialSoftmax(f_T)
        
        # Apply ignore mask if target is provided
        if target is not None:
            if target.dim() == 4:
                target = target.squeeze(1)
            # Resize target to match feature size
            target_resized = func.interpolate(
                target.unsqueeze(1).float(), 
                size=(f_S.shape[2], f_S.shape[3]), 
                mode='nearest'
            ).squeeze(1).long()
            
            # Create mask for valid pixels
            valid_mask = (target_resized != self.ignore_label).unsqueeze(1).float()
            
            # Apply mask
            f_S = f_S * valid_mask
            f_T = f_T * valid_mask
        
        loss = self.criterion(f_S, f_T)
        return loss


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())  # OR operation
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class KDPixelWiseCE(nn.Module):
    def __init__(self, temperature=3, ignore_label=0):
        super(KDPixelWiseCE, self).__init__()
        self.T = temperature
        self.ignore_label = ignore_label

    def forward(self, preds_S, preds_T, target=None):
        preds_T.detach()
        assert preds_S.shape == preds_T.shape, f'the dim of teacher {preds_T.shape} != and student {preds_S.shape}'
        
        B, C, H, W = preds_S.shape
        
        # Create mask for valid pixels (not ignore_label)
        if target is not None:
            if target.dim() == 4:
                target = target.squeeze(1)
            valid_mask = (target != self.ignore_label).view(-1)
        else:
            valid_mask = torch.ones(B * H * W, device=preds_S.device, dtype=torch.bool)
        
        # Reshape predictions
        preds_S = preds_S.permute(0, 2, 3, 1).contiguous().view(-1, C)
        preds_T = preds_T.permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        # Apply mask
        preds_S = preds_S[valid_mask]
        preds_T = preds_T[valid_mask]
        
        if preds_S.numel() == 0:  # No valid pixels
            return torch.tensor(0.0, device=preds_S.device, requires_grad=True)
        
        # Apply temperature scaling
        preds_S = func.log_softmax(preds_S / self.T, dim=1)
        preds_T = func.softmax(preds_T / self.T, dim=1)
        preds_T = preds_T + 10 ** (-7)
        preds_T = torch.autograd.Variable(preds_T.data, requires_grad=False)
        
        # Calculate loss only on valid pixels
        loss = self.T * self.T * torch.sum(-preds_T * preds_S) / valid_mask.sum().float()
        return loss

class KDLoss(nn.Module):
    def __init__(self, temperature=3, ignore_label=0, weight=None, base_feat_w=0, base_resp_w=0, student_loss_w=1.0):
        super(KDLoss, self).__init__()
        self.base_feat_w = base_feat_w
        self.base_resp_w = base_resp_w
        self.feat_w = 0
        self.resp_w = 0
        self.student_loss_w = student_loss_w
        self.ignore_label = ignore_label
        self.response_loss = KDPixelWiseCE(temperature=temperature, ignore_label=ignore_label)
        self.feature_loss = KDFeat(ignore_label=ignore_label)
        self.classification_loss = FSCELoss(ignore_label=ignore_label, weight=weight)

    def update_kd_loss_params(self, iters, max_iters):
        self.feat_w = (iters / max_iters) * self.base_feat_w
        self.resp_w = (iters / max_iters) * self.base_resp_w

    def forward(self, f_results, o_results, targets, semi=False, mask=None, **kwargs):
        loss = 0
        num_blocks = len(f_results)
        for r in range(num_blocks):
            if r < len(f_results) - 1:
                f_prev, f_next = f_results[r], f_results[-1]
            else:
                f_prev, f_next = None, None
            o_prev, o_next = o_results[r], o_results[-1]
            loss += self.forward_one_session(f_prev, f_next, o_prev, o_next, targets, semi, **kwargs)
        loss += self.classification_loss(o_results[-1], targets, mask=mask)
        return loss

    def forward_one_session(self, f_prev, f_next, o_prev, o_next, targets, semi, **kwargs):
        # Resize to match
        o_prev = func.interpolate(o_prev, (o_next.size(-2), o_next.size(-1)))
        
        if semi:  # Semi-supervised training
            cls_loss = 0
        else:
            cls_loss = self.classification_loss(o_prev, targets)
        loss = self.student_loss_w * cls_loss
        
        if self.resp_w > 0 and o_next is not None and o_prev is not None:
            response_loss = self.resp_w * self.response_loss(o_prev, o_next, targets)
            loss += response_loss
        if self.feat_w > 0 and f_next is not None and f_prev is not None:
            feature_loss = self.feat_w * self.feature_loss(f_prev, f_next, targets)
            loss += feature_loss
        return loss

class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


# class NLLLoss2d(nn.NLLLoss2d, base.Loss):
#     pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass

class KDMLoss(nn.Module):
    """Multi-head Knowledge Distillation Loss for Spatial Transcriptomics"""
    def __init__(self, feat_w=0.5, resp_w=0.5, temperature=4.0, ignore_label=0):
        super(KDMLoss, self).__init__()
        self.feat_w = feat_w
        self.resp_w = resp_w
        self.temperature = temperature
        self.ignore_label = ignore_label
        
        # Loss components
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_out, teacher_out, student_feat, teacher_feat, target):
        """
        Args:
            student_out: Student predictions
            teacher_out: Teacher predictions  
            student_feat: Student features
            teacher_feat: Teacher features
            target: Ground truth labels
        """
        if target.dim() == 4:
            target = target.squeeze(1)
        
        # Hard target loss (ignore background)
        ce_loss = self.ce_loss(student_out, target)
        
        # Create mask for valid pixels
        valid_mask = (target != self.ignore_label).unsqueeze(1).float()
        
        # Soft target loss (KD loss) - only on valid pixels
        student_soft = func.log_softmax(student_out / self.temperature, dim=1)
        teacher_soft = func.softmax(teacher_out / self.temperature, dim=1)
        # student_soft = func.log_softmax(student_out / self.temperature, dim=1)
        # teacher_soft = func.softmax(teacher_out / self.temperature, dim=1)

        kl_per_pixel = func.kl_div(student_soft, teacher_soft, reduction='none').sum(dim=1)

        kl_per_pixel_masked = kl_per_pixel * valid_mask.float()
        valid_pixels = valid_mask.sum().float()
        
        # Apply mask to KD loss
        if valid_pixels > 0:
            kd_loss = (kl_per_pixel_masked.sum() / valid_pixels) * (self.temperature ** 2)
        else:
            kd_loss = torch.tensor(0.0, device=student_out.device, requires_grad=True)
        
        # Feature distillation loss - only on valid pixels
        if student_feat.size() != teacher_feat.size():
            teacher_feat = func.interpolate(teacher_feat, size=student_feat.shape[2:], mode='bilinear')
        feat_loss = self.mse_loss(student_feat * valid_mask, teacher_feat * valid_mask)
        
        # Combine losses
        total_loss = ce_loss + self.resp_w * kd_loss + self.feat_w * feat_loss
        
        return total_loss
