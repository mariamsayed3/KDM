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
from utils.utils import (create_exp_dir, test_running_time_with_wrapper, 
                        init_obj, compute_confusion_matrix_dynamic, 
                        compute_eval_from_cm_robust, ClassTracker,
                        DynamicClassMetrics, TrainerWithDynamicClasses)
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
    model = getattr(models.hsi_net, m_params['name'])(
        n_bands=m_params['n_bands'],
        classes=m_params['classes'],
        nf_enc=m_params['nf_enc'],
        nf_dec=m_params['nf_dec'],
        do_batchnorm=m_params['do_batchnorm'],
        n_heads=m_params['n_heads'],
        max_norm_val=None,
        encoder_name=m_params['encoder_name']
    )
    
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
    
    # Enhanced testing with dynamic classes
    testing_with_dynamic_classes(model, test_loader, metric, device, 
                                m_params['n_heads'], comet, 
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
    """Enhanced testing with dynamic class support"""
    vis_path = f'{save_dir}/visual'
    os.makedirs(vis_path, exist_ok=True)
    
    # Initialize dynamic metrics for testing
    test_metrics = DynamicClassMetrics(ignore_index=0)
    
    model.eval()
    pbar = tqdm(test_loader, ncols=80, desc='Testing')
    
    # Storage for multi-head outputs
    all_predictions = [[] for _ in range(n_heads)]
    all_targets = []
    all_sample_ids = []
    
    with torch.no_grad():
        for step, minibatch in enumerate(pbar):
            # Get batch data
            x = minibatch['input'].to(device)
            y_seg = minibatch['ground_truth_seg'].to(device)
            sample_id = minibatch.get('name', [f'test_{step}'])[0]
            
            # Forward pass
            f_results, o_results = model(x)
            
            # Interpolate outputs to match ground truth size
            o_results = [F.interpolate(o, size=(y_seg.size(-2), y_seg.size(-1)), 
                                     mode='nearest') for o in o_results]
            
            # Update class tracker
            class_tracker.update(sample_id, y_seg[0].cpu().numpy())
            
            # Store predictions and targets
            for i in range(n_heads):
                all_predictions[i].append(o_results[i].cpu())
            all_targets.append(y_seg.cpu())
            all_sample_ids.append(sample_id)
            
            # Save visualizations
            if step < 10:  # Save first 10 samples
                test_loader.dataset.save_pred(y_seg, sv_path=vis_path, name=f'{sample_id}_gt.png')
                for i in range(n_heads):
                    test_loader.dataset.save_pred(o_results[i], sv_path=vis_path, 
                                                name=f'{sample_id}_head_{i}.png')
    
    # Get all classes found during testing
    all_classes = class_tracker.get_all_classes()
    print(f"\nClasses found during testing: {all_classes}")
    
    # Evaluate each head
    results = {}
    for i in range(n_heads):
        print(f"\n=== Evaluating Head {i} ===")
        
        # Concatenate predictions for this head
        head_predictions = torch.cat(all_predictions[i], dim=0)
        head_targets = torch.cat(all_targets, dim=0)
        
        # Compute confusion matrix with dynamic classes
        confusion_matrix, class_labels = compute_confusion_matrix_dynamic(
            head_targets.numpy(), 
            head_predictions.numpy(),
            all_possible_classes=all_classes,
            ignore_index=0
        )
        
        # Compute comprehensive metrics
        metrics = compute_eval_from_cm_robust(confusion_matrix, class_names=[f'class_{c}' for c in class_labels])
        
        # Log metrics
        print(f"Head {i} - mIoU: {metrics['mean_IoU']:.4f}")
        print(f"Head {i} - Pixel Acc: {metrics['pixel_accuracy']:.4f}")
        print(f"Head {i} - Mean Acc: {metrics['mean_accuracy']:.4f}")
        print(f"Head {i} - Dice: {metrics['mean_dice']:.4f}")
        print(f"Head {i} - Kappa: {metrics['kappa']:.4f}")
        
        comet.log_metric(f'test_mIoU_head_{i}', metrics['mean_IoU'])
        comet.log_metric(f'test_pixel_acc_head_{i}', metrics['pixel_accuracy'])
        comet.log_metric(f'test_mean_acc_head_{i}', metrics['mean_accuracy'])
        comet.log_metric(f'test_dice_head_{i}', metrics['mean_dice'])
        comet.log_metric(f'test_kappa_head_{i}', metrics['kappa'])
        
        # Log per-class metrics
        print(f"IoU per class: {metrics['IoU_per_class']}")
        comet.log_metric(f'test_IoU_array_head_{i}', metrics['IoU_per_class'])
        
        # Log confusion matrix
        comet.log_confusion_matrix(matrix=confusion_matrix, labels=[str(c) for c in class_labels])
        
        # Save confusion matrix
        cm_df = pd.DataFrame(confusion_matrix, columns=class_labels, index=class_labels)
        cm_df.to_csv(f'{save_dir}/confusion_matrix_head_{i}.csv')
        
        # Store results
        results[f'head_{i}'] = metrics
    
    # Save class distribution summary
    class_summary = class_tracker.get_samples_by_class()
    print(f"\nFinal class distribution: {class_summary}")
    
    # Save summary to file
    summary_data = {
        'all_classes': all_classes,
        'class_distribution': class_summary,
        'test_results': results
    }
    
    import json
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