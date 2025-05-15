# -*- coding: utf-8 -*-
"""
Updated train_teacher.py for spatial transcriptomics with dynamic class handling
"""
# Standard library imports
import os
import time
import pandas as pd
import yaml
import argparse
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
matplotlib.use('Agg')

# Third party imports
from comet_ml import Experiment
import torch
from tqdm import tqdm
from thop import profile, clever_format
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

# Local application imports
import utils.losses
import utils.metrics
import models.hsi_net
from utils.metrics import DynamicClassMetrics
from utils.utils import (create_exp_dir, test_running_time_with_wrapper, 
                        init_obj, compute_confusion_matrix_dynamic, 
                        compute_eval_from_cm_robust, ClassTracker,
                         TrainerWithDynamicClasses)
from utils.datasets import get_spatial_transcriptomics_dataloaders


def main(cfg, comet):
    """
    Set the network based on the configuration specified in the .yml file
    Enhanced for spatial transcriptomics with dynamic class handling
    """
    # Set random seeds for reproducibility
    init_seeds(cfg['seed'])
    
    device = torch.device('cuda:0')
    
    # Create the model
    m_params = cfg['model_params']

  # New code - handles both HSINet and SGR_Net variants
    model_class = getattr(models.hsi_net, m_params['name'])

    # Base parameters that all models accept
    model_kwargs = {
        'n_bands': m_params['n_bands'],
        'classes': m_params['classes'],
        'nf_enc': m_params['nf_enc'],
        'nf_dec': m_params['nf_dec'],
        'do_batchnorm': m_params['do_batchnorm'],
        'max_norm_val': None
    }

    # Add SGR_Net specific parameters only if needed
    if 'SGR' in m_params['name'] or 'Res_' in m_params['name']:
        model_kwargs['n_heads'] = m_params.get('n_heads', 3)
        model_kwargs['encoder_name'] = m_params.get('encoder_name', 'resnet50')

    # Create the model
    model = model_class(**model_kwargs)
    
    # Get dataloaders for spatial transcriptomics
    print("Loading spatial transcriptomics dataloaders...")
    train_loader, val_loader, test_loader = get_spatial_transcriptomics_dataloaders(
        data_dir=cfg['train_params']['dataset_dir'],
        batch_size=cfg['train_params']['batch_size'],
        num_workers=cfg['train_params']['num_workers']
    )
    
    # Initialize class tracker for dynamic class handling
    class_tracker = ClassTracker(ignore_index=0)
    
    # Log model parameters
    x = test_loader.dataset[0]['input'].unsqueeze(0)
    if 'cuda' in device.type:
        x = x.to('cuda:0')
    
    flops, params = profile(model.to('cuda:0'), inputs=(x.to('cuda:0'),), verbose=False)
    macs, params = clever_format([flops, params], "%.3f")
    print("FLOPS: {}, PARAMS: {}".format(flops, params))
    comet.log_other('Model trainable parameters', params)
    comet.log_other('Floating point operations per second (FLOPS)', flops)
    comet.log_other('Multiply accumulates per second (MACs)', macs)
    
    model = model.to(device)
    
    # Define optimizer
    print(f"Optimizer for model weights: {cfg['optimizer']['type']}")
    optimizer = init_obj(cfg['optimizer']['type'],
                        cfg['optimizer']['args'],
                        torch.optim, model.parameters())
    
    # Initialize loss function with spatial transcriptomics support
    if cfg['loss'] == 'SpatialTranscriptomicsLoss':
        loss_fn = utils.losses.SpatialTranscriptomicsLoss(
            ignore_index=0,
            ce_weight=m_params.get('ce_weight', 1.0),
            dice_weight=m_params.get('dice_weight', 0.5),
            focal_weight=m_params.get('focal_weight', 0.3)
        )
    else:
        loss_fn = getattr(utils.losses, cfg['loss'])(
            base_feat_w=m_params.get('feat_weight', 0.5),
            base_resp_w=m_params.get('resp_weight', 0.5),
            student_loss_w=m_params.get('student_weight', 1.0),
            ignore_label=0
        )
    
    # Initialize dynamic metrics
    metrics = DynamicClassMetrics(ignore_index=0)
    metric = getattr(utils.metrics, cfg['metric'])(ignore_index=0, activation='softmax2d')
    
    # Initialize trainer with dynamic classes
    trainer = TrainerWithDynamicClasses(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        ignore_index=0
    )
    
    # Resume if needed
    best_performance = 0
    last_epoch = 0
    if cfg['resume']:
        model_state_file = os.path.join(cfg['train_params']['save_dir'], 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_performance = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")
    
    # Training with dynamic class handling
    training_with_dynamic_classes(cfg['train_params'], trainer, train_loader, val_loader,
                                 loss_fn, metric, device, last_epoch, best_performance, 
                                 class_tracker, comet)
    
    # Testing
    print("\nThe training has completed. Testing the model now...")
    saved = torch.load(cfg['train_params']['save_dir'] + '/best_model.pth')
    model.load_state_dict(saved)

        # Get n_heads with proper default handling
    if 'SGR' in m_params['name'] or 'Res_' in m_params['name']:
        n_heads = m_params.get('n_heads', 3)
    else:
        n_heads = 1  # HSINet is single-head

    testing_with_dynamic_classes(model, test_loader, metric, device, 
                                n_heads, comet, 
                                cfg['train_params']['save_dir'], class_tracker)
    
    comet.log_asset(cfg['train_params']['save_dir'] + '/best_model.pth')


def training_with_dynamic_classes(train_cfg, trainer, train_loader, val_loader, loss_fn,
                                 metric, device, last_epoch, best_performance, 
                                 class_tracker, comet=None):
    """Enhanced training with dynamic class support"""
    n_epochs = train_cfg['n_epochs']
    not_improved_epochs = 0
    
    # Calculate iterations for KD loss scheduling
    epoch_iters = len(train_loader)
    max_iters = epoch_iters * n_epochs
    
    for epoch in range(last_epoch, n_epochs):
        print(f"\nTraining epoch {epoch + 1}/{n_epochs}")
        print("-----------------------------------")
        
        # Train epoch with dynamic classes
        with comet.train():
            train_metrics = trainer.train_epoch(train_loader, epoch)
            
            # Log training metrics
            for key, value in train_metrics.items():
                comet.log_metric(key, value, epoch=epoch + 1)
        
        # Validation epoch with dynamic classes
        with comet.validate():
            val_metrics = trainer.validate(val_loader)
            
            # Log validation metrics
            for key, value in val_metrics.items():
                comet.log_metric(key, value, epoch=epoch + 1)
        
        # Get class distribution summary
        class_summary = trainer.get_class_distribution_summary()
        print(f"Classes found so far: {class_summary['all_classes']}")
        print(f"Samples per class: {class_summary['samples_per_class']}")
        
        # Print summary
        print(f"\nSummary of epoch {epoch + 1}:")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, mIoU: {train_metrics['mean_iou']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, mIoU: {val_metrics['mean_iou']:.4f} - Best: {best_performance:.4f}")
        print(f"Pixel Acc: {val_metrics['accuracy']:.4f}, Dice: {val_metrics['dice']:.4f}, Kappa: {val_metrics['kappa']:.4f}")
        
        # Save checkpoint
        print('=> saving checkpoint to {}'.format(train_cfg['save_dir'] + 'checkpoint.pth.tar'))
        os.makedirs(train_cfg['save_dir'], exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'best_mIoU': best_performance,
            'state_dict': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, os.path.join(train_cfg['save_dir'], 'checkpoint.pth.tar'))
        
        # Save best model
        current_performance = val_metrics['mean_iou']
        if current_performance >= best_performance:
            print(f'Model exceeds prev best score ({current_performance:.4f} > {best_performance:.4f}). Saving it now.')
            best_performance = current_performance
            torch.save(trainer.model.state_dict(), train_cfg['save_dir'] + '/best_model.pth')
            not_improved_epochs = 0
        else:
            not_improved_epochs += 1
            if not_improved_epochs > train_cfg['early_stop']:
                print(f"Stopping training early because it has not improved for {train_cfg['early_stop']} epochs.")
                break
        
        # Update KD loss parameters if applicable
        if hasattr(loss_fn, 'update_kd_loss_params'):
            loss_fn.update_kd_loss_params(iters=epoch * epoch_iters, max_iters=max_iters)


def testing_with_dynamic_classes(model, test_loader, metric, device, n_heads, 
                                comet, save_dir, class_tracker):
    """Enhanced testing with dynamic class support for both single and multi-head models"""
    vis_path = f'{save_dir}/visual'
    os.makedirs(vis_path, exist_ok=True)
    
    # Initialize dynamic metrics for testing
    test_metrics = DynamicClassMetrics(ignore_index=0)
    
    model.eval()
    pbar = tqdm(test_loader, ncols=80, desc='Testing')
    
    # Storage for outputs - FIX: Correct initialization based on model type
    if n_heads > 1:
        all_predictions = [[] for _ in range(n_heads)]
    else:
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
            
            # DEBUG: Check what classes are in this sample
            if step < 3:
                unique_classes = torch.unique(y_seg).cpu().numpy()
                print(f"Sample {sample_id}: unique classes = {unique_classes}")
                print(f"Sample shape: {y_seg.shape}")
                print(f"Using {n_heads} heads evaluation")
            
            # Forward pass - FIX: Proper handling for multi-head models
            model_output = model(x)
            
            if n_heads > 1:  # Multi-head model
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    f_results, o_results = model_output
                    
                    # Interpolate each head output to match ground truth size
                    o_results = [F.interpolate(o, size=(y_seg.size(-2), y_seg.size(-1)), 
                                             mode='nearest') for o in o_results]
                    
                    # Store predictions for each head
                    for i in range(n_heads):
                        if i < len(o_results):  # Make sure head exists
                            all_predictions[i].append(o_results[i].cpu())
                else:
                    print(f"Warning: Expected tuple output for multi-head model, got {type(model_output)}")
                    # Fallback - treat as single output
                    o_results = [model_output]
                    for i in range(min(n_heads, len(o_results))):
                        all_predictions[i].append(o_results[i].cpu())
            else:  # Single-head model
                if isinstance(model_output, tuple):
                    f_results, o_results = model_output
                else:
                    o_results = model_output
                
                # Interpolate single output
                o_results = F.interpolate(o_results, size=(y_seg.size(-2), y_seg.size(-1)), 
                                        mode='nearest')
                all_predictions.append(o_results.cpu())
            
            # Update class tracker for each sample in batch
            batch_size = y_seg.size(0)
            for batch_idx in range(batch_size):
                if y_seg.dim() == 4:
                    sample_mask = y_seg[batch_idx, 0].cpu().numpy()
                else:
                    sample_mask = y_seg[batch_idx].cpu().numpy()
                
                batch_sample_id = f"{sample_id}_batch_{batch_idx}"
                class_tracker.update(batch_sample_id, sample_mask)
                
                if step < 3 and batch_idx < 2:
                    unique_in_sample = np.unique(sample_mask)
                    print(f"  Batch {batch_idx}: shape={sample_mask.shape}, unique={unique_in_sample}")
                    tracked_classes = class_tracker.get_sample_classes(batch_sample_id)
                    print(f"  Tracked classes for {batch_sample_id}: {tracked_classes}")
            
            # Store targets and sample IDs
            all_targets.append(y_seg.cpu())
            all_sample_ids.append(sample_id)
            
            # Save visualizations
            if step < 10:
                # Save ground truth
                if y_seg.dim() == 4:
                    gt_to_save = y_seg[0:1, 0:1]
                else:
                    gt_to_save = y_seg[0:1].unsqueeze(1)
                test_loader.dataset.save_pred(gt_to_save, sv_path=vis_path, name=f'{sample_id}_gt.png')
                
                # Save predictions
                if n_heads > 1:
                    for i in range(min(n_heads, len(o_results))):
                        test_loader.dataset.save_pred(o_results[i][0:1], sv_path=vis_path, 
                                                    name=f'{sample_id}_head_{i}.png')
                else:
                    test_loader.dataset.save_pred(o_results[0:1], sv_path=vis_path, 
                                                name=f'{sample_id}_pred.png')
    
    # Get all classes found during testing
    all_classes = class_tracker.get_all_classes()
    print(f"\nClasses found during testing: {all_classes}")
    print(f"Total samples tracked: {len(class_tracker.sample_classes)}")
    print(f"All classes in tracker: {class_tracker.all_classes}")
    
    # Show sample tracking info
    sample_ids = list(class_tracker.sample_classes.keys())[:5]
    for sid in sample_ids:
        classes_found = class_tracker.get_sample_classes(sid)
        print(f"{sid}: {classes_found}")
    
    # Fallback class handling
    if not all_classes or all_classes == [0]:
        print("Warning: Class tracker found no classes. Extracting from data...")
        all_classes_from_data = set()
        for targets in all_targets:
            unique_classes = torch.unique(targets).numpy()
            all_classes_from_data.update(unique_classes)
        all_classes_from_data.discard(0)
        all_classes = sorted(list(all_classes_from_data))
        if not all_classes:
            all_classes = [1, 2, 3, 4, 5]
        print(f"Using classes from data: {all_classes}")
    
    # Import evaluation functions
    from utils.utils import compute_confusion_matrix_dynamic, compute_eval_from_cm_robust
    
    # Evaluate based on model type
    results = {}
    
    if n_heads > 1:
        # Multi-head evaluation
        print(f"\n=== Evaluating Multi-Head Model ({n_heads} heads) ===")
        
        for i in range(n_heads):
            if len(all_predictions[i]) == 0:
                print(f"Warning: Head {i} has no predictions")
                continue
                
            print(f"\n--- Head {i} ---")
            
            # Concatenate predictions for this head
            head_predictions = torch.cat(all_predictions[i], dim=0)
            head_targets = torch.cat(all_targets, dim=0)
            
            # Compute metrics
            confusion_matrix, class_labels = compute_confusion_matrix_dynamic(
                head_targets.numpy(), 
                head_predictions.numpy(),
                all_possible_classes=all_classes,
                ignore_index=0
            )
            
            metrics = compute_eval_from_cm_robust(confusion_matrix, class_names=[f'class_{c}' for c in class_labels])
            
            # Display and log metrics
            print(f"Head {i} - mIoU: {metrics['mean_IoU']:.4f}")
            print(f"Head {i} - Pixel Acc: {metrics['pixel_accuracy']:.4f}")
            print(f"Head {i} - Dice: {metrics['mean_dice']:.4f}")
            print(f"Head {i} - Kappa: {metrics['kappa']:.4f}")
            
            comet.log_metric(f'test_mIoU_head_{i}', metrics['mean_IoU'])
            comet.log_metric(f'test_pixel_acc_head_{i}', metrics['pixel_accuracy'])
            comet.log_metric(f'test_dice_head_{i}', metrics['mean_dice'])
            comet.log_metric(f'test_kappa_head_{i}', metrics['kappa'])
            
            results[f'head_{i}'] = metrics
            
            # Log confusion matrix for this head
            comet.log_confusion_matrix(matrix=confusion_matrix, labels=[str(c) for c in class_labels],
                                     title=f'Confusion Matrix - Head {i}')
        
        # Log average performance across heads
        if results:
            avg_miou = np.mean([r['mean_IoU'] for r in results.values()])
            avg_pixel_acc = np.mean([r['pixel_accuracy'] for r in results.values()])
            avg_dice = np.mean([r['mean_dice'] for r in results.values()])
            
            print(f"\n=== Average Performance Across Heads ===")
            print(f"Average mIoU: {avg_miou:.4f}")
            print(f"Average Pixel Acc: {avg_pixel_acc:.4f}")
            print(f"Average Dice: {avg_dice:.4f}")
            
            comet.log_metric('test_avg_mIoU', avg_miou)
            comet.log_metric('test_avg_pixel_acc', avg_pixel_acc)
            comet.log_metric('test_avg_dice', avg_dice)
            
            results['average'] = {
                'mean_IoU': avg_miou,
                'pixel_accuracy': avg_pixel_acc,
                'mean_dice': avg_dice
            }
    
    else:
        # Single-head evaluation
        print(f"\n=== Evaluating Single Head Model ===")
        
        all_predictions_tensor = torch.cat(all_predictions, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        
        confusion_matrix, class_labels = compute_confusion_matrix_dynamic(
            all_targets_tensor.numpy(), 
            all_predictions_tensor.numpy(),
            all_possible_classes=all_classes,
            ignore_index=0
        )
        
        metrics = compute_eval_from_cm_robust(confusion_matrix, class_names=[f'class_{c}' for c in class_labels])
        
        print(f"mIoU: {metrics['mean_IoU']:.4f}")
        print(f"Pixel Acc: {metrics['pixel_accuracy']:.4f}")
        print(f"Dice: {metrics['mean_dice']:.4f}")
        print(f"Kappa: {metrics['kappa']:.4f}")
        
        comet.log_metric('test_mIoU', metrics['mean_IoU'])
        comet.log_metric('test_pixel_acc', metrics['pixel_accuracy'])
        comet.log_metric('test_dice', metrics['mean_dice'])
        comet.log_metric('test_kappa', metrics['kappa'])
        
        results['single_head'] = metrics
        comet.log_confusion_matrix(matrix=confusion_matrix, labels=[str(c) for c in class_labels])
    
    # Save results with numpy type conversion
    import json
    
    def convert_numpy_types(obj):
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
    
    summary_data = {
        'all_classes': convert_numpy_types(all_classes),
        'class_distribution': {
            'tracked_samples': len(class_tracker.sample_classes),
            'found_classes': convert_numpy_types(list(class_tracker.all_classes))
        },
        'test_results': convert_numpy_types(results),
        'model_type': 'multi_head' if n_heads > 1 else 'single_head',
        'n_heads': n_heads
    }
    
    with open(f'{save_dir}/test_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    return results


def init_seeds(seed):
    """Initialize random seeds for reproducibility"""
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Train teacher model for spatial transcriptomics')
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
                      project_name="spatial-transcriptomics-teacher",
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
        save_dir = f'experiments/{cfg["name"]}-teacher-{time.strftime("%Y%m%d-%H%M%S")}'
        create_exp_dir(save_dir, visual_folder=True)
    else:
        save_dir = f"experiments/{cfg['train_params']['save_dir']}"

    cfg['train_params']['save_dir'] = save_dir
    comet.set_name(f'{cfg["name"]}-teacher-{cfg["train_params"]["n_epochs"]}epochs')
    comet.log_asset(cmd_args.config)
    comet.add_tags(cfg.get('tags', []) + ['teacher', 'spatial_transcriptomics'])

    main(cfg, comet)