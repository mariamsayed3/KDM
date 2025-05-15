# -*- coding: utf-8 -*-
"""
Updated train_kd.py for spatial transcriptomics with dynamic class handling
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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Local application imports
import utils.losses
import utils.metrics
import models.hsi_net
from models.hsi_net import HSINet, Res_SGR_Net, SpatialTranscriptomics_SGR_Net

from utils.utils import (create_exp_dir, init_obj, test_running_time,
                        compute_confusion_matrix_dynamic, compute_eval_from_cm_robust,
                        ClassTracker, DynamicClassMetrics, show_visual_results_dynamic)
from utils.datasets import get_spatial_transcriptomics_dataloaders


def main(cfg, comet):
    """
    Knowledge distillation training for spatial transcriptomics with dynamic class handling
    """
    # Set random seeds for reproducibility
    init_seeds(cfg['seed'])
    device = torch.device('cuda:0')
    
    # Create teacher model
    teacher_params = cfg['teacher_params']
    teacher_model = getattr(models.hsi_net, teacher_params['name'])(
        n_bands=teacher_params['n_bands'],
        classes=teacher_params['classes'],
        nf_enc=teacher_params['nf_enc'],
        nf_dec=teacher_params['nf_dec'],
        do_batchnorm=teacher_params['do_batchnorm'],
        n_heads=teacher_params['n_heads'],
        max_norm_val=None,
        encoder_name=teacher_params['encoder_name']
    )
    
    # Create student model
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
        # Handle other architectures
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
    
    # Initialize class tracker
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
    
    # Move models to GPU and load teacher weights
    student_model = torch.nn.DataParallel(student_model).cuda()
    teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    
    # Load pre-trained teacher
    saved = torch.load(teacher_params['pretrained_file'])
    teacher_model.load_state_dict(saved)
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
    
    # Initialize metric and loss function
    metric = getattr(utils.metrics, cfg['metric'])(ignore_index=0, activation='softmax2d')
    
    # Knowledge distillation loss for spatial transcriptomics
    if cfg['loss'] == 'KDMLoss':
        loss_fn = utils.losses.KDMLoss(
            feat_w=student_params['feat_weight'],
            resp_w=student_params['resp_weight'],
            temperature=student_params.get('temperature', 4.0),
            ignore_label=0,
            adaptive_weights=True
        )
    else:
        loss_fn = getattr(utils.losses, cfg['loss'])(
            feat_w=student_params['feat_weight'],
            resp_w=student_params['resp_weight'],
            temperature=student_params.get('temperature', 4.0),
            ignore_label=0
        )
    
    # Training with dynamic classes
    training_kd_with_dynamic_classes(cfg, optimizer, student_model, teacher_model, 
                                   train_loader, val_loader, loss_fn, metric, 
                                   device, last_epoch, best_performance, 
                                   class_tracker, comet)
    
    # Testing
    print("\nKnowledge distillation training completed. Testing the student model...")
    saved = torch.load(cfg['train_params']['save_dir'] + '/best_model.pth')
    student_model.load_state_dict(saved)
    
    testing_kd_with_dynamic_classes(student_model, test_loader, metric, device, 
                                  comet, cfg['train_params']['save_dir'], 
                                  class_tracker)
    
    comet.log_asset(cfg['train_params']['save_dir'] + '/best_model.pth')
    test_running_time(test_loader.dataset[0]['input'].unsqueeze(0), student_model, comet)


def training_kd_with_dynamic_classes(cfg, optimizer, student_model, teacher_model, 
                                   train_loader, val_loader, loss_fn, metric, 
                                   device, last_epoch, best_performance, 
                                   class_tracker, comet):
    """Knowledge distillation training with dynamic class support"""
    train_cfg = cfg['train_params']
    n_epochs = train_cfg['n_epochs']
    n_networks = cfg['teacher_params']['n_heads']
    resets = n_epochs // n_networks
    not_improved_epochs = 0
    
    # Initialize dynamic metrics
    train_metrics = DynamicClassMetrics(ignore_index=0)
    val_metrics = DynamicClassMetrics(ignore_index=0)
    
    for epoch in range(last_epoch, n_epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer=optimizer, 
                           base_lr=cfg['optimizer']['args']['lr'], 
                           epoch=epoch, step_size=resets)
        
        model_id = epoch // resets
        
        print(f"\nTraining epoch {epoch + 1}/{n_epochs}")
        print(f"Distilling the {model_id + 1}-th teacher to the student")
        print("-----------------------------------")
        
        # Reset metrics
        train_metrics.reset()
        val_metrics.reset()
        
        # Train epoch
        with comet.train():
            train_loss, train_performance = train_epoch_kd(
                optimizer, student_model, teacher_model, model_id,
                train_loader, loss_fn, metric, device, class_tracker, train_metrics)
            
            # Compute comprehensive training metrics
            train_results = train_metrics.compute_metrics()
            train_results['loss'] = train_loss
            train_results['primary_metric'] = train_performance
            
            # Log to comet
            for key, value in train_results.items():
                comet.log_metric(f'train_{key}', value, epoch=epoch + 1)
        
        # Validation epoch
        with comet.validate():
            val_loss, val_performance = val_epoch_kd(
                student_model, teacher_model, model_id, val_loader, 
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


def train_epoch_kd(optimizer, student_model, teacher_model, model_id, 
                  train_loader, loss_fn, metric, device, class_tracker, train_metrics):
    """Train one epoch with knowledge distillation and dynamic classes"""
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
        
        # Forward pass through student
        student_f, student_o = student_model(x)
        
        # Forward pass through teacher (no gradients)
        with torch.no_grad():
            teacher_f, teacher_o = teacher_model(x)
        
        # Update class tracker
        for i, sample_id in enumerate(sample_ids):
            class_tracker.update(sample_id, y_seg[i].cpu().numpy())
        
        # Compute loss with dynamic class support
        if hasattr(loss_fn, 'forward') and 'sample_info' in loss_fn.forward.__code__.co_varnames:
            # For losses that support sample info
            sample_info = {'sample_ids': sample_ids}
            loss = loss_fn(student_o, teacher_o, student_f, teacher_f, y_seg, sample_info)
        else:
            # Standard KD loss
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
        result = f"Train loss: {loss:.4f}"
        pbar.set_postfix_str(result)
    
    avg_loss = running_loss / len(train_loader)
    avg_performance = running_performance / len(train_loader)
    
    return avg_loss, avg_performance


def val_epoch_kd(student_model, teacher_model, model_id, val_loader, 
                loss_fn, metric, device, class_tracker, val_metrics):
    """Validate one epoch with knowledge distillation and dynamic classes"""
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
            teacher_f, teacher_o = teacher_model(x)
            
            # Update class tracker
            for i, sample_id in enumerate(sample_ids):
                class_tracker.update(sample_id, y_seg[i].cpu().numpy())
            
            # Compute loss
            if hasattr(loss_fn, 'forward') and 'sample_info' in loss_fn.forward.__code__.co_varnames:
                sample_info = {'sample_ids': sample_ids}
                loss = loss_fn(student_o, teacher_o, student_f, teacher_f, y_seg, sample_info)
            else:
                loss = loss_fn(student_o, teacher_o, student_f, teacher_f, y_seg)
            
            # Update metrics
            val_metrics.update(student_o, y_seg, sample_ids, probabilities=True)
            
            # Compute primary metric
            performance = metric(student_o, y_seg)
            
            running_loss += loss.item()
            running_performance += performance.item()
            
            # Update progress bar
            result = f"Val loss: {loss:.4f}"
            pbar.set_postfix_str(result)
    
    avg_loss = running_loss / len(val_loader)
    avg_performance = running_performance / len(val_loader)
    
    return avg_loss, avg_performance


def testing_kd_with_dynamic_classes(model, test_loader, metric, device, comet, 
                                  save_dir, class_tracker):
    """Test the distilled student model with dynamic class support"""
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
            
            # Forward pass
            _, o = model(x)
            
            # Update class tracker and metrics
            class_tracker.update(sample_id, y_seg[0].cpu().numpy())
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
    
    # Compute comprehensive metrics
    comprehensive_metrics = compute_eval_from_cm_robust(
        confusion_matrix, 
        class_names=[f'class_{c}' for c in class_labels]
    )
    
    # Print and log results
    print("\n" + "="*50)
    print("KNOWLEDGE DISTILLATION TEST RESULTS")
    print("="*50)
    print(f"Mean IoU: {comprehensive_metrics['mean_IoU']:.4f}")
    print(f"Pixel Accuracy: {comprehensive_metrics['pixel_accuracy']:.4f}")
    print(f"Mean Accuracy: {comprehensive_metrics['mean_accuracy']:.4f}")
    print(f"Dice Score: {comprehensive_metrics['mean_dice']:.4f}")
    print(f"Kappa Score: {comprehensive_metrics['kappa']:.4f}")
    print(f"Classes found: {all_classes}")
    print(f"Total samples: {class_summary['total_samples']}")
    
    # Log to comet
    comet.log_metric('test_mean_iou', comprehensive_metrics['mean_IoU'])
    comet.log_metric('test_pixel_accuracy', comprehensive_metrics['pixel_accuracy'])
    comet.log_metric('test_mean_accuracy', comprehensive_metrics['mean_accuracy'])
    comet.log_metric('test_dice', comprehensive_metrics['mean_dice'])
    comet.log_metric('test_kappa', comprehensive_metrics['kappa'])
    comet.log_metric('test_iou_array', comprehensive_metrics['IoU_per_class'])
    
    # Log confusion matrix
    comet.log_confusion_matrix(matrix=confusion_matrix, labels=[str(c) for c in class_labels])
    
    # Save results
    import json
    results_summary = {
        'metrics': comprehensive_metrics,
        'class_summary': class_summary,
        'all_classes': all_classes
    }
    
    with open(f'{save_dir}/test_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
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


def adjust_learning_rate(optimizer, base_lr, epoch, step_size, cycle=25):
    """Adjust learning rate with decay"""
    left_epochs = epoch % step_size
    lr = base_lr * (0.8 ** (left_epochs // cycle))
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
    args = argparse.ArgumentParser(description='Knowledge Distillation for Spatial Transcriptomics')
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
                      project_name="spatial-transcriptomics-kd",
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
        save_dir = f'experiments/{cfg["name"]}-kd-{time.strftime("%Y%m%d-%H%M%S")}'
        create_exp_dir(save_dir, visual_folder=True)
    else:
        save_dir = f"experiments/{cfg['train_params']['save_dir']}"

    cfg['train_params']['save_dir'] = save_dir
    comet.set_name(f'{cfg["name"]}-kd-{cfg["train_params"]["n_epochs"]}epochs')
    comet.log_asset(cmd_args.config)
    comet.add_tags(cfg.get('tags', []) + ['knowledge_distillation', 'spatial_transcriptomics'])

    main(cfg, comet)