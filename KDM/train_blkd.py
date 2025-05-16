# -*- coding: utf-8 -*-
"""
Updated train_blkd.py for spatial transcriptomics with dynamic class handling
Block Knowledge Distillation for Spatial Transcriptomics
"""
# Standard library imports
import os
import time
import yaml
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Third party imports
from comet_ml import Experiment
import torch
from tqdm import tqdm
from thop import profile, clever_format
import torch.nn.functional as F
from models.hsi_net import HSINet, Res_SGR_Net, SpatialTranscriptomics_SGR_Net


# Local application imports
import utils.losses
import utils.metrics
import models.hsi_net
from models.hsi_net import HSINet, Res_SGR_Net, SpatialTranscriptomics_SGR_Net
from utils.metrics import DynamicClassMetrics
from utils.utils import (create_exp_dir, init_obj, test_running_time,
                        compute_confusion_matrix_dynamic, compute_eval_from_cm_robust,
                        ClassTracker, show_visual_results_dynamic)
from utils.datasets import get_spatial_transcriptomics_dataloaders


def main(cfg, comet):
    """
    Block Knowledge Distillation training for spatial transcriptomics with dynamic class handling
    """
    # Set random seeds for reproducibility
    init_seeds(cfg['seed'])
    device = torch.device('cuda:0')
    
    # Create teacher model (multi-head model like SGR_Net)
    teacher_params = cfg['teacher_params']
    teacher_model_class = getattr(models.hsi_net, teacher_params['name'])
    teacher_kwargs = {
        'n_bands': teacher_params['n_bands'],
        'classes': teacher_params['classes'],
        'nf_enc': teacher_params['nf_enc'],
        'nf_dec': teacher_params['nf_dec'],
        'do_batchnorm': teacher_params['do_batchnorm'],
        'max_norm_val': None
    }

    # Add multi-head specific parameters only if needed
    if 'SGR' in teacher_params['name'] or 'Res_' in teacher_params['name']:
        teacher_kwargs['n_heads'] = teacher_params.get('n_heads', 5)
        teacher_kwargs['encoder_name'] = teacher_params.get('encoder_name', None)

    teacher_model = teacher_model_class(**teacher_kwargs)
    
    # Create student model (single head)
    student_params = cfg['student_params']
    if ('name' not in student_params) or (student_params['name'] == 'HSINet'):
        student_model = HSINet(
            n_bands=student_params['n_bands'],
            classes=student_params['classes'],
            nf_enc=student_params['nf_enc'],
            nf_dec=student_params['nf_dec'],
            do_batchnorm=student_params['do_batchnorm'],
            max_norm_val=None
        )
    else:
        # Handle other student architectures
        student_model = getattr(models.hsi_net, student_params['name'])(
            n_bands=student_params['n_bands'],
            classes=student_params['classes'],
            nf_enc=student_params['nf_enc'],
            nf_dec=student_params['nf_dec'],
            do_batchnorm=student_params['do_batchnorm'],
            max_norm_val=None
        )
    
    # Get spatial transcriptomics dataloaders
    print("Loading spatial transcriptomics dataloaders...")
    train_loader, val_loader, test_loader = get_spatial_transcriptomics_dataloaders(
        data_dir=cfg['train_params']['dataset_dir'],
        batch_size=cfg['train_params']['batch_size'],
        num_workers=cfg['train_params']['num_workers']
    )
    
    # Initialize class tracker for dynamic class handling
    class_tracker = ClassTracker(ignore_index=0)
    
    # Log model parameters
    x = test_loader.dataset[0]['input'].unsqueeze(0).to(device)
    
    # Student model profiling
    flops, params = profile(student_model.to(device), inputs=(x,), verbose=False)
    macs, params = clever_format([flops, params], "%.3f")
    print(f"Student - FLOPS: {flops}, PARAMS: {params}")
    comet.log_other('Student model trainable parameters', params)
    comet.log_other('Student FLOPS', flops)
    comet.log_other('Student MACs', macs)
    
    # Teacher model profiling
    flops, params = profile(teacher_model.to(device), inputs=(x,), verbose=False)
    macs, params = clever_format([flops, params], "%.3f")
    print(f"Teacher - FLOPS: {flops}, PARAMS: {params}")
    comet.log_other('Teacher model trainable parameters', params)
    comet.log_other('Teacher FLOPS', flops)
    comet.log_other('Teacher MACs', macs)
    
    # Move student to GPU with DataParallel
    student_model = torch.nn.DataParallel(student_model).cuda()

    # Load pre-trained teacher BEFORE DataParallel wrapping
    saved = torch.load(teacher_params['pretrained_file'], map_location=device)

    # Clean the state dict of profiling artifacts and DataParallel prefixes
    cleaned_state_dict = {}
    for key, value in saved.items():
        # Remove DataParallel prefix if present
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        
        # Skip profiling artifacts
        if not (new_key.endswith('.total_ops') or new_key.endswith('.total_params')):
            cleaned_state_dict[new_key] = value

    # Load cleaned state dict into teacher model
    teacher_model.load_state_dict(cleaned_state_dict, strict=False)

    # NOW wrap teacher with DataParallel and move to GPU
    teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Set teacher to eval mode
    
    # Define optimizer
    print(f"Optimizer for student weights: {cfg['optimizer']['type']}")
    optimizer = init_obj(cfg['optimizer']['type'],
                        cfg['optimizer']['args'],
                        torch.optim, student_model.parameters())
    
    # Resume if needed
    best_performance = 0
    last_epoch = 0
    if cfg['resume']:
        model_state_file = os.path.join(cfg['train_params']['save_dir'], 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_performance = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            student_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")
    
    # Re-initialize optimizer after loading
    optimizer = init_obj(cfg['optimizer']['type'],
                        cfg['optimizer']['args'],
                        torch.optim, student_model.parameters())
    
    # Initialize metric and loss function
    metric = getattr(utils.metrics, cfg['metric'])(ignore_index=0, activation='softmax2d')
    
    # Block Knowledge Distillation loss
    loss_fn = getattr(utils.losses, cfg['loss'])(
        feat_w=student_params['feat_weight'],
        resp_w=student_params['resp_weight'],
        temperature=student_params.get('temperature', 3.0),
        ignore_label=0
    )
    
    # Training with dynamic classes
    training_blkd_with_dynamic_classes(cfg, optimizer, student_model, teacher_model, 
                                     train_loader, val_loader, loss_fn, metric, 
                                     device, last_epoch, best_performance, 
                                     class_tracker, comet)
    
    # Testing
    print("\nBlock knowledge distillation training completed. Testing the student model...")
    saved = torch.load(cfg['train_params']['save_dir'] + '/best_model.pth')
    student_model.load_state_dict(saved)
    
    testing_blkd_with_dynamic_classes(student_model, test_loader, metric, device, 
                                    comet, cfg['train_params']['save_dir'], 
                                    class_tracker)
    
    comet.log_asset(cfg['train_params']['save_dir'] + '/best_model.pth')
    test_running_time(test_loader.dataset[0]['input'].unsqueeze(0), student_model, comet)


def training_blkd_with_dynamic_classes(cfg, optimizer, student_model, teacher_model, 
                                     train_loader, val_loader, loss_fn, metric, 
                                     device, last_epoch, best_performance, 
                                     class_tracker, comet):
    """Block Knowledge Distillation training with dynamic class support"""
    train_cfg = cfg['train_params']
    n_epochs = train_cfg['n_epochs']
    n_networks = cfg['teacher_params']['n_heads']
    
    # Validate n_heads by testing teacher model
    with torch.no_grad():
        test_input = next(iter(train_loader))['input'][:1].to(device)
        teacher_test_output = teacher_model(test_input)
        
        if isinstance(teacher_test_output, tuple) and len(teacher_test_output) == 2:
            test_f, test_o = teacher_test_output
            if isinstance(test_o, list):
                actual_n_heads = len(test_o)
                print(f"INFO: Teacher model has {actual_n_heads} heads")
                if actual_n_heads != n_networks:
                    print(f"WARNING: Config specifies {n_networks} heads but teacher has {actual_n_heads}")
                    n_networks = actual_n_heads  # Use actual number of heads
            else:
                print(f"WARNING: Teacher model appears to be single-head")
                n_networks = 1
        else:
            print(f"WARNING: Teacher model output format unexpected, treating as single-head")
            n_networks = 1
    
    # Recalculate resets based on actual number of heads
    resets = max(1, n_epochs // n_networks)  # Ensure at least 1 epoch per head
    print(f"INFO: Using {n_networks} heads with {resets} epochs per head")
    
    not_improved_epochs = 0
    
    # Initialize dynamic metrics
    train_metrics = DynamicClassMetrics(ignore_index=0)
    val_metrics = DynamicClassMetrics(ignore_index=0)
    
    for epoch in range(last_epoch, n_epochs):
        # Determine which teacher head to use based on distillation mode
        distillation_mode = cfg.get('distillation_mode', 'block')  # Default to block KD
        
        if distillation_mode == 'single_head':
            # Single Head KD: Always use the specified head
            target_head = cfg.get('target_head', 4)  # Default to head 4 (best teacher)
            teacher_head_id = min(target_head, n_networks - 1)  # Ensure within bounds
            mode_description = f"Single Head KD (Head {teacher_head_id})"
            
        elif distillation_mode == 'block':
            # Block KD: Progressive head selection (ORIGINAL BEHAVIOR)
            teacher_head_id = min(epoch // resets, n_networks - 1)  # Ensure within bounds
            mode_description = f"Block KD (Progressive Head {teacher_head_id + 1}/{n_networks})"
            
        elif distillation_mode == 'best_head':
            # Always use the best head (last head, typically head 4)
            teacher_head_id = n_networks - 1
            mode_description = f"Best Head KD (Head {teacher_head_id})"
            
        elif distillation_mode == 'random':
            # Random Head KD: Randomly select a head each epoch
            import random
            teacher_head_id = random.randint(0, n_networks - 1)
            mode_description = f"Random Head KD (Head {teacher_head_id})"
            
        else:
            # Default to block KD if mode not recognized
            teacher_head_id = min(epoch // resets, n_networks - 1)
            mode_description = f"Block KD (Default - Head {teacher_head_id + 1}/{n_networks})"
        print(f"Warning: Unknown distillation_mode '{distillation_mode}', using block KD")
        print(f"\nTraining epoch {epoch + 1}/{n_epochs}")
        print(f"Mode: {mode_description}")
        print(f"Using teacher head {teacher_head_id + 1}/{n_networks} for distillation")
        print("-----------------------------------")
        # Reset metrics
        train_metrics.reset()
        val_metrics.reset()
        
        # Train epoch
        with comet.train():
            train_loss, train_performance = train_epoch_blkd(
                optimizer, student_model, teacher_model, teacher_head_id,
                train_loader, loss_fn, metric, device, class_tracker, train_metrics)
            
            # Compute comprehensive training metrics
            train_results = train_metrics.compute_metrics()
            train_results['loss'] = train_loss
            train_results['primary_metric'] = train_performance
            
            # Log to comet
            for key, value in train_results.items():
                comet.log_metric(f'train_{key}', value, epoch=epoch + 1)
            comet.log_metric('teacher_head_id', teacher_head_id, epoch=epoch + 1)
        
        # Validation epoch
        with comet.validate():
            val_loss, val_performance = val_epoch_blkd(
                student_model, teacher_model, teacher_head_id, val_loader, 
                loss_fn, metric, device, class_tracker, val_metrics)
            
            # Compute comprehensive validation metrics
            val_results = val_metrics.compute_metrics()
            val_results['loss'] = val_loss
            val_results['primary_metric'] = val_performance
            
            # Log to comet
            for key, value in val_results.items():
                comet.log_metric(f'val_{key}', value, epoch=epoch + 1)
        
        # Get class distribution summary
        class_summary = get_class_summary(train_metrics, val_metrics)
        print(f"Classes found: {class_summary.get('all_classes_train', [])} (train), "
              f"{class_summary.get('all_classes_val', [])} (val)")
        
        # Print summary
        print(f"\nSummary of epoch {epoch + 1}:")
        print(f"Train - Loss: {train_loss:.4f}, mIoU: {train_results['mean_iou']:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, mIoU: {val_results['mean_iou']:.4f} - Best: {best_performance:.4f}")
        
        # Save checkpoint
        print('=> saving checkpoint to {}'.format(train_cfg['save_dir'] + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch + 1,
            'best_mIoU': best_performance,
            'state_dict': student_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(train_cfg['save_dir'], 'checkpoint.pth.tar'))
        
        # Save best model
        current_performance = val_results['mean_iou']
        if current_performance > best_performance:
            print(f'Student model exceeds prev best score ({current_performance:.4f} > {best_performance:.4f}). Saving it now.')
            best_performance = current_performance
            torch.save(student_model.state_dict(), train_cfg['save_dir'] + '/best_model.pth')
            not_improved_epochs = 0
        else:
            not_improved_epochs += 1
            if not_improved_epochs > train_cfg['early_stop']:
                print(f"Stopping training early because it has not improved for {train_cfg['early_stop']} epochs.")
                break
        
        lr = get_lr(optimizer)
        print(f"Learning rate: {lr}")

def train_epoch_blkd(optimizer, student_model, teacher_model, teacher_head_id,
                    train_loader, loss_fn, metric, device, class_tracker, train_metrics):
    """Train one epoch with block knowledge distillation and dynamic classes"""
    student_model.train()
    teacher_model.eval()
    
    pbar = tqdm(train_loader, ncols=80, desc='Training')
    running_loss = 0
    running_performance = 0
    
    for step, minibatch in enumerate(pbar):
        optimizer.zero_grad()
        
        # Get batch data
        x = minibatch['input'].to(device)
        y_seg = minibatch['ground_truth_seg'].to(device)
        sample_ids = minibatch.get('name', [f'train_batch_{step}_sample_{i}' for i in range(x.size(0))])
        
        # Forward pass through student (returns single output)
        student_f, student_o = student_model(x)
        
        # Forward pass through teacher (returns multiple heads)
        with torch.no_grad():
            teacher_outputs = teacher_model(x)
            
            # Handle different teacher output formats
            if isinstance(teacher_outputs, tuple) and len(teacher_outputs) == 2:
                teacher_f_list, teacher_o_list = teacher_outputs
                
                # DEBUG: Print info about teacher outputs
                if step == 0:
                    print(f"DEBUG - Teacher outputs format: features={type(teacher_f_list)}, outputs={type(teacher_o_list)}")
                    if isinstance(teacher_f_list, list):
                        print(f"DEBUG - Number of feature heads: {len(teacher_f_list)}")
                    if isinstance(teacher_o_list, list):
                        print(f"DEBUG - Number of output heads: {len(teacher_o_list)}")
                    print(f"DEBUG - Requested teacher_head_id: {teacher_head_id}")
                
                # Handle feature selection
                if isinstance(teacher_f_list, list):
                    if teacher_head_id < len(teacher_f_list):
                        teacher_f = teacher_f_list[teacher_head_id]
                    else:
                        print(f"WARNING: teacher_head_id {teacher_head_id} >= len(teacher_f_list) {len(teacher_f_list)}")
                        teacher_f = teacher_f_list[-1]  # Use last head as fallback
                else:
                    teacher_f = teacher_f_list
                
                # Handle output selection
                if isinstance(teacher_o_list, list):
                    if teacher_head_id < len(teacher_o_list):
                        teacher_o = teacher_o_list[teacher_head_id]
                    else:
                        print(f"WARNING: teacher_head_id {teacher_head_id} >= len(teacher_o_list) {len(teacher_o_list)}")
                        teacher_o = teacher_o_list[-1]  # Use last head as fallback
                else:
                    teacher_o = teacher_o_list
            else:
                # Single head teacher model
                print(f"WARNING: Teacher model returned single output, treating as single-head model")
                teacher_f, teacher_o = teacher_outputs if isinstance(teacher_outputs, tuple) else (None, teacher_outputs)
                teacher_head_id = 0  # Reset to 0 for single head
        
        # Update class tracker
        for i, sample_id in enumerate(sample_ids):
            # Handle tensor shape properly
            if y_seg.dim() == 4:  # [B, C, H, W]
                mask_data = y_seg[i, 0].cpu().numpy()
            elif y_seg.dim() == 3:  # [B, H, W]
                mask_data = y_seg[i].cpu().numpy()
            else:
                mask_data = y_seg.cpu().numpy()
            class_tracker.update(sample_id, mask_data)
        
        # Debug feature shapes
        if step == 0:
            print(f"DEBUG - Student features shape: {student_f.shape}")
            print(f"DEBUG - Teacher features shape: {teacher_f.shape}")
            print(f"DEBUG - Student output shape: {student_o.shape}")
            print(f"DEBUG - Teacher output shape: {teacher_o.shape}")
        
        # Compute block knowledge distillation loss
        loss = loss_fn(student_o, teacher_o, student_f, teacher_f, y_seg)
        
        loss.backward()
        optimizer.step()
        
        # Update metrics
        train_metrics.update(student_o, y_seg, sample_ids, probabilities=True)
        
        # Compute primary metric
        performance = metric(student_o, y_seg)
        
        running_loss += loss.item()
        running_performance += performance.item()
        
        # Update progress bar
        result = f"Train loss: {loss:.4f}, Head: {teacher_head_id}"
        pbar.set_postfix_str(result)
    
    avg_loss = running_loss / len(train_loader)
    avg_performance = running_performance / len(train_loader)
    
    return avg_loss, avg_performance


def val_epoch_blkd(student_model, teacher_model, teacher_head_id, val_loader, 
                  loss_fn, metric, device, class_tracker, val_metrics):
    """Validate one epoch with block knowledge distillation and dynamic classes"""
    student_model.eval()
    teacher_model.eval()
    
    pbar = tqdm(val_loader, ncols=80, desc='Validating')
    running_loss = 0
    running_performance = 0
    
    with torch.no_grad():
        for step, minibatch in enumerate(pbar):
            # Get batch data
            x = minibatch['input'].to(device)
            y_seg = minibatch['ground_truth_seg'].to(device)
            sample_ids = minibatch.get('name', [f'val_batch_{step}_sample_{i}' for i in range(x.size(0))])
            
            # Forward pass through student
            student_f, student_o = student_model(x)
            
            # Forward pass through teacher
            teacher_outputs = teacher_model(x)
            
            # Handle different teacher output formats
            if isinstance(teacher_outputs, tuple) and len(teacher_outputs) == 2:
                teacher_f_list, teacher_o_list = teacher_outputs
                
                # Handle feature selection with bounds checking
                if isinstance(teacher_f_list, list):
                    if teacher_head_id < len(teacher_f_list):
                        teacher_f = teacher_f_list[teacher_head_id]
                    else:
                        teacher_f = teacher_f_list[-1]  # Use last head as fallback
                else:
                    teacher_f = teacher_f_list
                
                # Handle output selection with bounds checking
                if isinstance(teacher_o_list, list):
                    if teacher_head_id < len(teacher_o_list):
                        teacher_o = teacher_o_list[teacher_head_id]
                    else:
                        teacher_o = teacher_o_list[-1]  # Use last head as fallback
                else:
                    teacher_o = teacher_o_list
            else:
                # Single head teacher model
                teacher_f, teacher_o = teacher_outputs if isinstance(teacher_outputs, tuple) else (None, teacher_outputs)
            
            # Update class tracker
            for i, sample_id in enumerate(sample_ids):
                # Handle tensor shape properly
                if y_seg.dim() == 4:  # [B, C, H, W]
                    mask_data = y_seg[i, 0].cpu().numpy()
                elif y_seg.dim() == 3:  # [B, H, W]
                    mask_data = y_seg[i].cpu().numpy()
                else:
                    mask_data = y_seg.cpu().numpy()
                class_tracker.update(sample_id, mask_data)
            
            # Compute loss
            loss = loss_fn(student_o, teacher_o, student_f, teacher_f, y_seg)
            
            # Update metrics
            val_metrics.update(student_o, y_seg, sample_ids, probabilities=True)
            
            # Compute primary metric
            performance = metric(student_o, y_seg)
            
            running_loss += loss.item()
            running_performance += performance.item()
            
            # Update progress bar
            result = f"Val loss: {loss:.4f}, Head: {teacher_head_id}"
            pbar.set_postfix_str(result)
    
    avg_loss = running_loss / len(val_loader)
    avg_performance = running_performance / len(val_loader)
    
    return avg_loss, avg_performance
## old version without extra metrices 
# def testing_blkd_with_dynamic_classes(model, test_loader, metric, device, comet, 
#                                     save_dir, class_tracker):
#     """Test the block-distilled student model with dynamic class support"""
#     vis_path = f'{save_dir}/visual'
#     os.makedirs(vis_path, exist_ok=True)
    
#     # Initialize test metrics
#     test_metrics = DynamicClassMetrics(ignore_index=0)
    
#     model.eval()
#     pbar = tqdm(test_loader, ncols=80, desc='Testing')
    
#     all_predictions = []
#     all_targets = []
#     all_sample_ids = []
    
#     with torch.no_grad():
#         for step, minibatch in enumerate(pbar):
#             # Get batch data
#             x = minibatch['input'].to(device)
#             y_seg = minibatch['ground_truth_seg'].to(device) 
#             y_oht = minibatch['ground_truth_onehot'].to(device)
#             sample_id = minibatch.get('name', [f'test_{step}'])[0]
            
#             # Forward pass (student model has single output)
#             _, o = model(x)
            
#             # Update class tracker and metrics
#             # Handle batch processing properly
#             batch_size = y_seg.size(0)
#             for batch_idx in range(batch_size):
#                 if y_seg.dim() == 4:
#                     sample_mask = y_seg[batch_idx, 0].cpu().numpy()
#                 else:
#                     sample_mask = y_seg[batch_idx].cpu().numpy()
                
#                 batch_sample_id = f"{sample_id}_batch_{batch_idx}"
#                 class_tracker.update(batch_sample_id, sample_mask)
#             test_metrics.update(o, y_seg, [sample_id], probabilities=True)
            
#             # Store for final evaluation
#             all_predictions.append(o.cpu())
#             all_targets.append(y_seg.cpu())
#             all_sample_ids.append(sample_id)
            
#             # Save visualizations for first few samples
#             if step < 10:
#                 test_loader.dataset.save_pred(y_oht, sv_path=vis_path, name=f'{sample_id}_gt.png')
#                 test_loader.dataset.save_pred(o, sv_path=vis_path, name=f'{sample_id}_pred.png')
                
#                 # Save detailed visualization
#                 show_visual_results_dynamic(
#                     x=x.cpu().numpy(),
#                     y_gt=y_seg.cpu().numpy(),
#                     y_pr=o.cpu().numpy(),
#                     available_classes=class_tracker.get_sample_classes(sample_id),
#                     all_classes=class_tracker.get_all_classes(),
#                     show_visual=False,
#                     fig_name=f'{vis_path}/{sample_id}_detailed.png',
#                     ignore_index=0
#                 )
    
#     # Compute final metrics
#     final_metrics = test_metrics.compute_metrics()
#     class_summary = test_metrics.get_summary()
    
#     # Get all classes found
#     all_classes = class_tracker.get_all_classes()
    
#     # Compute confusion matrix
#     all_predictions = torch.cat(all_predictions, dim=0)
#     all_targets = torch.cat(all_targets, dim=0)
    
#     confusion_matrix, class_labels = compute_confusion_matrix_dynamic(
#         all_targets.numpy(),
#         all_predictions.numpy(),
#         all_possible_classes=all_classes,
#         ignore_index=0
#     )
    
#     # Compute comprehensive metrics
#     comprehensive_metrics = compute_eval_from_cm_robust(
#         confusion_matrix, 
#         class_names=[f'class_{c}' for c in class_labels]
#     )
    
#     # Print and log results
#     print("\n" + "="*50)
#     print("BLOCK KNOWLEDGE DISTILLATION TEST RESULTS")
#     print("="*50)
#     print(f"Mean IoU: {comprehensive_metrics['mean_IoU']:.4f}")
#     print(f"Pixel Accuracy: {comprehensive_metrics['pixel_accuracy']:.4f}")
#     print(f"Mean Accuracy: {comprehensive_metrics['mean_accuracy']:.4f}")
#     print(f"Dice Score: {comprehensive_metrics['mean_dice']:.4f}")
#     print(f"Kappa Score: {comprehensive_metrics['kappa']:.4f}")
#     print(f"Classes found: {all_classes}")
#     print(f"Total samples: {class_summary['total_samples']}")
    
#     # Log to comet
#     comet.log_metric('test_mean_iou', comprehensive_metrics['mean_IoU'])
#     comet.log_metric('test_pixel_accuracy', comprehensive_metrics['pixel_accuracy'])
#     comet.log_metric('test_mean_accuracy', comprehensive_metrics['mean_accuracy'])
#     comet.log_metric('test_dice', comprehensive_metrics['mean_dice'])
#     comet.log_metric('test_kappa', comprehensive_metrics['kappa'])
#     comet.log_metric('test_iou_array', comprehensive_metrics['IoU_per_class'])
    
#     # Log confusion matrix
#     comet.log_confusion_matrix(matrix=confusion_matrix, labels=[str(c) for c in class_labels])
    
#         # Before line 440, add:
#     import json

#     def convert_numpy_types(obj):
#             """Convert numpy types to Python native types for JSON serialization"""
#             if isinstance(obj, np.integer):
#                 return int(obj)
#             elif isinstance(obj, np.floating):
#                 return float(obj)
#             elif isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             elif isinstance(obj, dict):
#                 return {key: convert_numpy_types(value) for key, value in obj.items()}
#             elif isinstance(obj, list):
#                 return [convert_numpy_types(item) for item in obj]
#             elif isinstance(obj, set):
#                 return [convert_numpy_types(item) for item in obj]
#             else:
#                 return obj

#         # Then modify the save section:
#     results_summary = {
#             'metrics': convert_numpy_types(comprehensive_metrics),
#             'class_summary': convert_numpy_types(class_summary),
#             'all_classes': convert_numpy_types(all_classes),
#             'method': 'block_knowledge_distillation'
#         }
    
#     with open(f'{save_dir}/test_results.json', 'w') as f:
#         json.dump(results_summary, f, indent=2)
    
#     return comprehensive_metrics


def testing_blkd_with_dynamic_classes(model, test_loader, metric, device, comet, 
                                    save_dir, class_tracker):
    """Test the block-distilled student model with dynamic class support"""
    vis_path = f'{save_dir}/visual'
    os.makedirs(vis_path, exist_ok=True)
    
    # Initialize test metrics
    test_metrics = DynamicClassMetrics(ignore_index=0)
    
    model.eval()
    pbar = tqdm(test_loader, ncols=80, desc='Testing')
    
    all_predictions = []
    all_targets = []
    all_sample_ids = []
    
    with torch.no_grad():
        for step, minibatch in enumerate(pbar):
            # Get batch data
            x = minibatch['input'].to(device)
            y_seg = minibatch['ground_truth_seg'].to(device) 
            y_oht = minibatch['ground_truth_onehot'].to(device)
            sample_id = minibatch.get('name', [f'test_{step}'])[0]
            
            # Forward pass (student model has single output)
            _, o = model(x)
            
            # Update class tracker and metrics
            # Handle batch processing properly
            batch_size = y_seg.size(0)
            for batch_idx in range(batch_size):
                if y_seg.dim() == 4:
                    sample_mask = y_seg[batch_idx, 0].cpu().numpy()
                else:
                    sample_mask = y_seg[batch_idx].cpu().numpy()
                
                batch_sample_id = f"{sample_id}_batch_{batch_idx}"
                class_tracker.update(batch_sample_id, sample_mask)
            test_metrics.update(o, y_seg, [sample_id], probabilities=True)
            
            # Store for final evaluation
            all_predictions.append(o.cpu())
            all_targets.append(y_seg.cpu())
            all_sample_ids.append(sample_id)
            
            # Save visualizations for first few samples
            if step < 10:
                test_loader.dataset.save_pred(y_oht, sv_path=vis_path, name=f'{sample_id}_gt.png')
                test_loader.dataset.save_pred(o, sv_path=vis_path, name=f'{sample_id}_pred.png')
                
                # Save detailed visualization
                show_visual_results_dynamic(
                    x=x.cpu().numpy(),
                    y_gt=y_seg.cpu().numpy(),
                    y_pr=o.cpu().numpy(),
                    available_classes=class_tracker.get_sample_classes(sample_id),
                    all_classes=class_tracker.get_all_classes(),
                    show_visual=False,
                    fig_name=f'{vis_path}/{sample_id}_detailed.png',
                    ignore_index=0
                )
    
    # Compute final metrics
    final_metrics = test_metrics.compute_metrics()
    class_summary = test_metrics.get_summary()
    
    # Get all classes found
    all_classes = class_tracker.get_all_classes()
    
    # Compute confusion matrix
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    confusion_matrix, class_labels = compute_confusion_matrix_dynamic(
        all_targets.numpy(),
        all_predictions.numpy(),
        all_possible_classes=all_classes,
        ignore_index=0
    )
    
    # Compute comprehensive metrics (now includes micro/macro precision, recall, f1)
    comprehensive_metrics = compute_eval_from_cm_robust(
        confusion_matrix, 
        class_names=[f'class_{c}' for c in class_labels]
    )
    
    # Print and log results
    print("\n" + "="*50)
    print("BLOCK KNOWLEDGE DISTILLATION TEST RESULTS")
    print("="*50)
    print(f"Macro IoU: {comprehensive_metrics['macro_IoU']:.4f}")
    print(f"Pixel Accuracy (Micro): {comprehensive_metrics['pixel_accuracy']:.4f}")
    print(f"Macro Accuracy: {comprehensive_metrics['macro_accuracy']:.4f}")
    print(f"Macro Dice: {comprehensive_metrics['macro_dice']:.4f}")
    print(f"Micro Dice: {comprehensive_metrics['micro_dice']:.4f}")
    print(f"Kappa Score: {comprehensive_metrics['kappa']:.4f}")
    print("\nPrecision/Recall/F1 Scores:")
    print(f"  Macro - Precision: {comprehensive_metrics['macro_precision']:.4f}, "
          f"Recall: {comprehensive_metrics['macro_recall']:.4f}, F1: {comprehensive_metrics['macro_f1']:.4f}")
    print(f"  Micro - Precision: {comprehensive_metrics['micro_precision']:.4f}, "
          f"Recall: {comprehensive_metrics['micro_recall']:.4f}, F1: {comprehensive_metrics['micro_f1']:.4f}")
    print(f"Classes found: {all_classes}")
    print(f"Total samples: {class_summary['total_samples']}")
    
    # Log metrics to comet
    # Main metrics
    comet.log_metric('test_macro_iou', comprehensive_metrics['macro_IoU'])
    comet.log_metric('test_pixel_accuracy', comprehensive_metrics['pixel_accuracy'])
    comet.log_metric('test_macro_accuracy', comprehensive_metrics['macro_accuracy'])
    comet.log_metric('test_macro_dice', comprehensive_metrics['macro_dice'])
    comet.log_metric('test_micro_dice', comprehensive_metrics['micro_dice'])
    comet.log_metric('test_kappa', comprehensive_metrics['kappa'])
    
    # Precision, Recall, F1 metrics
    comet.log_metric('test_macro_precision', comprehensive_metrics['macro_precision'])
    comet.log_metric('test_macro_recall', comprehensive_metrics['macro_recall'])
    comet.log_metric('test_macro_f1', comprehensive_metrics['macro_f1'])
    comet.log_metric('test_micro_precision', comprehensive_metrics['micro_precision'])
    comet.log_metric('test_micro_recall', comprehensive_metrics['micro_recall'])
    comet.log_metric('test_micro_f1', comprehensive_metrics['micro_f1'])
    
    # Per-class metrics
    comet.log_metric('test_iou_per_class', comprehensive_metrics['IoU_per_class'])
    comet.log_metric('test_precision_per_class', comprehensive_metrics['precision_per_class'])
    comet.log_metric('test_recall_per_class', comprehensive_metrics['recall_per_class'])
    comet.log_metric('test_f1_per_class', comprehensive_metrics['f1_per_class'])
    
    # Log confusion matrix
    comet.log_confusion_matrix(matrix=confusion_matrix, labels=[str(c) for c in class_labels])
    
    # Save confusion matrix to file
    print(f"\nSaving confusion matrix to {save_dir}/confusion_matrix.npy")
    np.save(f'{save_dir}/confusion_matrix.npy', confusion_matrix)
    
    # Save confusion matrix as CSV for readability
    import pandas as pd
    cm_df = pd.DataFrame(confusion_matrix, 
                        index=[f'True_{c}' for c in class_labels],
                        columns=[f'Pred_{c}' for c in class_labels])
    cm_df.to_csv(f'{save_dir}/confusion_matrix.csv')
    print(f"Confusion matrix also saved as CSV: {save_dir}/confusion_matrix.csv")
    
    # Convert numpy types for JSON serialization
    import json

    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, set):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    # Create comprehensive results summary
    results_summary = {
        'metrics': convert_numpy_types(comprehensive_metrics),
        'class_summary': convert_numpy_types(class_summary),
        'all_classes': convert_numpy_types(all_classes),
        'confusion_matrix': convert_numpy_types(confusion_matrix),
        'class_labels': convert_numpy_types(class_labels),
        'method': 'block_knowledge_distillation'
    }
    
    # Save results to JSON
    with open(f'{save_dir}/test_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nAll results saved in {save_dir}/test_results.json")
    
    return comprehensive_metrics


def get_class_summary(train_metrics, val_metrics):
    """Get summary of classes found during training and validation"""
    train_summary = train_metrics.get_summary()
    val_summary = val_metrics.get_summary()
    
    return {
        'all_classes_train': train_summary.get('all_classes_found', []),
        'all_classes_val': val_summary.get('all_classes_found', []),
        'train_samples': train_summary.get('total_samples', 0),
        'val_samples': val_summary.get('total_samples', 0)
    }


def adjust_learning_rate(optimizer, base_lr, epoch, step_size):
    """Adjust learning rate with decay"""
    left_epochs = epoch % step_size
    lr = base_lr * (0.8 ** (left_epochs // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def init_seeds(seed):
    """Initialize random seeds for reproducibility"""
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Block Knowledge Distillation for Spatial Transcriptomics')
    args.add_argument('--config', default='configs/base.yml', type=str,
                      help='config file path (default: None)')
    args.add_argument('--debug', default=0, type=int,
                      help='debug mode? (default: 0)')
    cmd_args = args.parse_args()

    assert cmd_args.config is not None, "Please specify a config file"

    # Configuring comet-ml logger
    api_key_path = "./configs/comet-ml-key.txt"
    if os.path.isfile(api_key_path) and os.access(api_key_path, os.R_OK):
        with open(api_key_path, "r") as f:
            comet_key = f.read().strip()
    else:
        raise FileNotFoundError(
            'You need to create a textfile containing only the comet-ml api key. '
            'The full path should be ./configs/comet-ml-key.txt')

    comet = Experiment(api_key=comet_key,
                      project_name="spatial-transcriptomics-blkd",
                      workspace="hieuphan",
                      disabled=bool(cmd_args.debug),
                      auto_metric_logging=False)

    # Read experiment configurations
    with open(cmd_args.config) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Handle debug mode
    if cmd_args.debug == 1:
        print('DEBUG mode')
        save_dir = 'experiments/test-folder'
        create_exp_dir(save_dir, visual_folder=True)
    elif cfg['train_params']['save_dir'] == '':
        save_dir = f'experiments/{cfg["name"]}-blkd-{time.strftime("%Y%m%d-%H%M%S")}'
        create_exp_dir(save_dir, visual_folder=True)
    else:
        save_dir = f"experiments/{cfg['train_params']['save_dir']}"

    cfg['train_params']['save_dir'] = save_dir
    comet.set_name(f'{cfg["name"]}-blkd-{cfg["train_params"]["n_epochs"]}epochs')
    comet.log_asset(cmd_args.config)
    comet.add_tags(cfg.get('tags', []) + ['block_knowledge_distillation', 'spatial_transcriptomics'])

    main(cfg, comet)