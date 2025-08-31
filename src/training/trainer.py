# src/training/trainer.py
"""
Enhanced training framework for CBAM-STN-TPS-YOLO
Comprehensive trainer with multi-GPU support, advanced optimization, and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
import json
import pickle
from tqdm.auto import tqdm
import wandb

# Internal imports
from .losses import YOLOLoss, calculate_loss, get_loss_weights_for_dataset
from .metrics import DetectionMetrics, AdvancedDetectionMetrics, ClassificationMetrics
from ..models import create_model, get_model_info
from ..utils.visualization import create_training_plots, save_prediction_samples
from ..utils.evaluation import ModelEvaluator
from ..data.dataset import create_dataloader

logger = logging.getLogger(__name__)

class BaseTrainer:
    """Base trainer class with common functionality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.start_epoch = 0
        self.best_metrics = {}
        self.training_history = defaultdict(list)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model
        self.model = self._create_model()
        
        # Setup optimization
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Setup metrics
        self.train_metrics = self._create_metrics()
        self.val_metrics = self._create_metrics()
        
        # Mixed precision training
        self.use_amp = config.get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 20),
            min_delta=config.get('early_stopping_min_delta', 1e-4),
            mode=config.get('early_stopping_mode', 'max')
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Wandb integration
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            self.setup_wandb()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('log_level', 'INFO')
        log_file = self.config.get('log_file', None)
        
        # Configure logger
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
    
    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        try:
            wandb.init(
                project=self.config.get('wandb_project', 'cbam-stn-tps-yolo'),
                name=self.config.get('experiment_name', 'experiment'),
                config=self.config,
                tags=self.config.get('tags', []),
                notes=self.config.get('notes', ''),
                resume=self.config.get('wandb_resume', False)
            )
            logger.info("✅ Wandb initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def _create_model(self) -> nn.Module:
        """Create and initialize model"""
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'CBAM-STN-TPS-YOLO')
        
        model = create_model(model_type, **model_config)
        model = model.to(self.device)
        
        # Initialize weights if specified
        if model_config.get('pretrained_weights'):
            self._load_pretrained_weights(model, model_config['pretrained_weights'])
        elif model_config.get('init_method'):
            self._initialize_weights(model, model_config['init_method'])
        
        # Model info
        model_info = get_model_info(model)
        logger.info(f"✅ Model created: {model_type}")
        logger.info(f"   Parameters: {model_info['total_params']:,}")
        logger.info(f"   Trainable: {model_info['trainable_params']:,}")
        logger.info(f"   Model size: {model_info['size_mb']:.2f} MB")
        
        return model
    
    def _load_pretrained_weights(self, model: nn.Module, weights_path: str):
        """Load pretrained weights"""
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle potential key mismatches
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())
            
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys
            
            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
            # Load with strict=False to handle partial loading
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"✅ Loaded pretrained weights from {weights_path}")
            
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            raise
    
    def _initialize_weights(self, model: nn.Module, init_method: str):
        """Initialize model weights"""
        def init_fn(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if init_method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_method == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_method == 'normal':
                    nn.init.normal_(m.weight, 0, 0.02)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        model.apply(init_fn)
        logger.info(f"✅ Initialized weights using {init_method} method")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'AdamW')
        lr = opt_config.get('lr', 1e-3)
        weight_decay = opt_config.get('weight_decay', 1e-4)
        
        # Parameter groups for different learning rates
        param_groups = self._create_parameter_groups()
        
        if opt_type == 'Adam':
            optimizer = optim.Adam(
                param_groups, 
                lr=lr, 
                weight_decay=weight_decay,
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'AdamW':
            optimizer = optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'SGD':
            optimizer = optim.SGD(
                param_groups,
                lr=lr,
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=weight_decay,
                nesterov=opt_config.get('nesterov', True)
            )
        elif opt_type == 'RMSprop':
            optimizer = optim.RMSprop(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                momentum=opt_config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
        
        logger.info(f"✅ Created {opt_type} optimizer with lr={lr}")
        return optimizer
    
    def _create_parameter_groups(self) -> List[Dict]:
        """Create parameter groups with different learning rates"""
        # Different learning rates for different components
        backbone_params = []
        attention_params = []
        detection_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'backbone' in name or 'feature' in name:
                backbone_params.append(param)
            elif 'cbam' in name or 'attention' in name:
                attention_params.append(param)
            elif 'head' in name or 'detect' in name or 'pred' in name:
                detection_params.append(param)
            else:
                other_params.append(param)
        
        # Base learning rate
        base_lr = self.config.get('optimizer', {}).get('lr', 1e-3)
        
        param_groups = [
            {
                'params': backbone_params,
                'lr': base_lr * self.config.get('backbone_lr_mult', 0.1),
                'name': 'backbone'
            },
            {
                'params': attention_params,
                'lr': base_lr * self.config.get('attention_lr_mult', 1.0),
                'name': 'attention'
            },
            {
                'params': detection_params,
                'lr': base_lr * self.config.get('detection_lr_mult', 1.0),
                'name': 'detection'
            },
            {
                'params': other_params,
                'lr': base_lr,
                'name': 'other'
            }
        ]
        
        # Filter out empty groups
        param_groups = [group for group in param_groups if len(group['params']) > 0]
        
        logger.info(f"Created {len(param_groups)} parameter groups")
        for group in param_groups:
            logger.info(f"  {group['name']}: {len(group['params'])} params, lr={group['lr']}")
        
        return param_groups
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        sched_config = self.config.get('scheduler', {})
        if not sched_config:
            return None
        
        sched_type = sched_config.get('type', 'CosineAnnealingLR')
        
        if sched_type == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get('T_max', self.config.get('epochs', 100)),
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_type == 'CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config.get('T_0', 10),
                T_mult=sched_config.get('T_mult', 2),
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_config.get('mode', 'max'),
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 10),
                verbose=True
            )
        elif sched_type == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=sched_config.get('milestones', [30, 60, 90]),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_type == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=sched_config.get('gamma', 0.95)
            )
        elif sched_type == 'OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=sched_config.get('max_lr', self.config.get('optimizer', {}).get('lr', 1e-3)),
                epochs=self.config.get('epochs', 100),
                steps_per_epoch=len(self.train_loader),
                pct_start=sched_config.get('pct_start', 0.3),
                anneal_strategy=sched_config.get('anneal_strategy', 'cos')
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")
        
        logger.info(f"✅ Created {sched_type} scheduler")
        return scheduler
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function"""
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'yolo')
        
        # Get dataset-specific weights if available
        dataset_type = self.config.get('dataset', {}).get('type', 'default')
        loss_weights = get_loss_weights_for_dataset(dataset_type)
        
        # Override with config weights
        loss_weights.update(loss_config.get('weights', {}))
        
        if loss_type.lower() == 'yolo':
            criterion = YOLOLoss(
                num_classes=self.config.get('num_classes', 3),
                **loss_weights,
                **loss_config.get('params', {})
            )
        else:
            criterion = calculate_loss(loss_type, **loss_config.get('params', {}))
        
        logger.info(f"✅ Created {loss_type} loss function")
        return criterion
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders"""
        data_config = self.config.get('data', {})
        
        train_loader = create_dataloader(
            data_config.get('train_path'),
            batch_size=self.config.get('batch_size', 16),
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            augmentations=data_config.get('train_augmentations', True),
            input_size=self.config.get('input_size', 640),
            num_classes=self.config.get('num_classes', 3),
            spectral_bands=self.config.get('spectral_bands', 4)
        )
        
        val_loader = create_dataloader(
            data_config.get('val_path'),
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            augmentations=False,
            input_size=self.config.get('input_size', 640),
            num_classes=self.config.get('num_classes', 3),
            spectral_bands=self.config.get('spectral_bands', 4)
        )
        
        logger.info(f"✅ Created data loaders:")
        logger.info(f"   Train: {len(train_loader)} batches, {len(train_loader.dataset)} samples")
        logger.info(f"   Val: {len(val_loader)} batches, {len(val_loader.dataset)} samples")
        
        return train_loader, val_loader
    
    def _create_metrics(self) -> Union[DetectionMetrics, AdvancedDetectionMetrics]:
        """Create metrics tracker"""
        metrics_config = self.config.get('metrics', {})
        
        if metrics_config.get('advanced', False):
            metrics = AdvancedDetectionMetrics(
                num_classes=self.config.get('num_classes', 3),
                iou_thresholds=metrics_config.get('iou_thresholds', [0.5]),
                conf_threshold=metrics_config.get('conf_threshold', 0.5),
                class_names=self.config.get('class_names'),
                track_per_image=metrics_config.get('track_per_image', False)
            )
        else:
            metrics = DetectionMetrics(
                num_classes=self.config.get('num_classes', 3),
                iou_thresholds=metrics_config.get('iou_thresholds', [0.5]),
                conf_threshold=metrics_config.get('conf_threshold', 0.5),
                class_names=self.config.get('class_names')
            )
        
        return metrics
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_loss = 0.0
        epoch_losses = defaultdict(float)
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.get("epochs", 100)}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = [{k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                       for k, v in target.items()} for target in targets]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    predictions = self.model(images)
                    losses = self.criterion(predictions, targets)
                
                # Backward pass
                total_loss = losses['total_loss'] if isinstance(losses, dict) else losses
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if self.config.get('grad_clip_norm'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip_norm']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                losses = self.criterion(predictions, targets)
                
                total_loss = losses['total_loss'] if isinstance(losses, dict) else losses
                total_loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip_norm']
                    )
                
                self.optimizer.step()
            
            # Update metrics
            if isinstance(losses, dict):
                for key, value in losses.items():
                    epoch_losses[key] += value.item()
                total_loss_val = losses['total_loss'].item()
            else:
                total_loss_val = losses.item()
                epoch_losses['total_loss'] += total_loss_val
            
            epoch_loss += total_loss_val
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss_val:.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log batch metrics to wandb
            if self.use_wandb and batch_idx % self.config.get('log_interval', 100) == 0:
                log_dict = {
                    'train/batch_loss': total_loss_val,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                }
                if isinstance(losses, dict):
                    for key, value in losses.items():
                        if key != 'total_loss':
                            log_dict[f'train/batch_{key}'] = value.item()
                
                wandb.log(log_dict)
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}
        avg_losses['total_loss'] = epoch_loss / num_batches
        
        # Step scheduler (if not ReduceLROnPlateau)
        if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        return avg_losses
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        epoch_loss = 0.0
        epoch_losses = defaultdict(float)
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Validation {epoch}')
            
            for batch_idx, (images, targets) in enumerate(pbar):
                images = images.to(self.device, non_blocking=True)
                targets = [{k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                           for k, v in target.items()} for target in targets]
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        predictions = self.model(images)
                        losses = self.criterion(predictions, targets)
                else:
                    predictions = self.model(images)
                    losses = self.criterion(predictions, targets)
                
                # Update losses
                if isinstance(losses, dict):
                    for key, value in losses.items():
                        epoch_losses[key] += value.item()
                    total_loss_val = losses['total_loss'].item()
                else:
                    total_loss_val = losses.item()
                    epoch_losses['total_loss'] += total_loss_val
                
                epoch_loss += total_loss_val
                
                # Update metrics (simplified for validation)
                # In practice, you'd decode predictions and compute detection metrics
                
                # Update progress bar
                pbar.set_postfix({
                    'val_loss': f'{total_loss_val:.4f}',
                    'avg_val_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
                })
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}
        avg_losses['total_loss'] = epoch_loss / num_batches
        
        # Compute detection metrics
        val_metrics = self.val_metrics.compute_metrics()
        
        # Step scheduler (if ReduceLROnPlateau)
        if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            monitor_metric = self.config.get('monitor_metric', 'mAP')
            if monitor_metric in val_metrics:
                self.scheduler.step(val_metrics[monitor_metric])
            else:
                self.scheduler.step(avg_losses['total_loss'])
        
        # Combine losses and metrics
        result = {**avg_losses}
        result.update({f'val_{k}': v for k, v in val_metrics.items() if isinstance(v, (int, float))})
        
        return result
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info("🚀 Starting training...")
        start_time = time.time()
        
        # Resume from checkpoint if specified
        if self.config.get('resume_from'):
            self.load_checkpoint(self.config['resume_from'])
        
        best_metric = float('-inf')
        best_epoch = 0
        
        for epoch in range(self.start_epoch, self.config.get('epochs', 100)):
            epoch_start_time = time.time()
            
            # Training
            train_losses = self.train_epoch(epoch)
            
            # Validation
            val_results = self.validate_epoch(epoch)
            
            # Update history
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(train_losses['total_loss'])
            self.training_history['val_loss'].append(val_results['total_loss'])
            
            for key, value in val_results.items():
                if key.startswith('val_') and isinstance(value, (int, float)):
                    self.training_history[key].append(value)
            
            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train/epoch_loss': train_losses['total_loss'],
                    'val/epoch_loss': val_results['total_loss'],
                    'time/epoch_time': time.time() - epoch_start_time
                }
                
                # Add component losses
                for key, value in train_losses.items():
                    if key != 'total_loss':
                        log_dict[f'train/{key}'] = value
                
                # Add validation metrics
                for key, value in val_results.items():
                    if isinstance(value, (int, float)):
                        log_dict[f'val/{key}'] = value
                
                wandb.log(log_dict)
            
            # Monitor metric for early stopping and checkpointing
            monitor_metric = self.config.get('monitor_metric', 'val_mAP')
            current_metric = val_results.get(monitor_metric, val_results.get('val_total_loss', 0))
            
            # Invert loss for maximization
            if 'loss' in monitor_metric:
                current_metric = -current_metric
            
            # Early stopping
            should_stop = self.early_stopping(current_metric)
            
            # Save best model
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                self.best_metrics = val_results.copy()
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"✅ New best model at epoch {epoch}: {monitor_metric}={current_metric:.4f}")
            
            # Regular checkpoint
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Log epoch summary
            logger.info(
                f"Epoch {epoch}/{self.config.get('epochs', 100)} - "
                f"Train Loss: {train_losses['total_loss']:.4f}, "
                f"Val Loss: {val_results['total_loss']:.4f}, "
                f"{monitor_metric}: {current_metric:.4f}, "
                f"Time: {time.time() - epoch_start_time:.2f}s"
            )
            
            # Early stopping
            if should_stop:
                logger.info(f"🛑 Early stopping at epoch {epoch}")
                break
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"✅ Training completed in {total_time / 3600:.2f} hours")
        logger.info(f"   Best epoch: {best_epoch}")
        logger.info(f"   Best {monitor_metric}: {best_metric:.4f}")
        
        # Final checkpoint
        self.save_checkpoint(epoch, is_best=False, final=True)
        
        # Create training plots
        if self.config.get('create_plots', True):
            self.create_training_plots()
        
        return {
            'best_metrics': self.best_metrics,
            'best_epoch': best_epoch,
            'training_history': dict(self.training_history),
            'total_time': total_time
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, final: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config,
            'training_history': dict(self.training_history),
            'best_metrics': self.best_metrics
        }
        
        # Save different types of checkpoints
        if final:
            checkpoint_path = self.checkpoint_dir / 'final_checkpoint.pth'
        elif is_best:
            checkpoint_path = self.checkpoint_dir / 'best_checkpoint.pth'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Also save model weights only for easier loading
        if is_best:
            model_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(self.model.state_dict(), model_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        if 'training_history' in checkpoint:
            self.training_history.update(checkpoint['training_history'])
        if 'best_metrics' in checkpoint:
            self.best_metrics = checkpoint['best_metrics']
        
        logger.info(f"✅ Resumed from epoch {self.start_epoch}")
    
    def create_training_plots(self):
        """Create training visualization plots"""
        try:
            from ..utils.visualization import create_training_plots
            
            plots_dir = Path(self.config.get('plots_dir', './plots'))
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            create_training_plots(
                self.training_history,
                save_dir=plots_dir,
                experiment_name=self.config.get('experiment_name', 'experiment')
            )
            
            logger.info(f"📊 Training plots saved to {plots_dir}")
            
        except ImportError:
            logger.warning("Visualization utilities not available, skipping plots")
        except Exception as e:
            logger.error(f"Failed to create training plots: {e}")

class CBAMSTNTPSYOLOTrainer(BaseTrainer):
    """Specialized trainer for CBAM-STN-TPS-YOLO model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Additional CBAM-STN-TPS-YOLO specific setup
        self.attention_loss_weight = config.get('attention_loss_weight', 0.1)
        self.stn_loss_weight = config.get('stn_loss_weight', 0.05)
        
        # Track component-specific metrics
        self.attention_metrics = {}
        self.stn_metrics = {}
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Enhanced training with component-specific losses"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_loss = 0.0
        epoch_losses = defaultdict(float)
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'CBAM-STN-TPS-YOLO Epoch {epoch}/{self.config.get("epochs", 100)}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = [{k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                       for k, v in target.items()} for target in targets]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    # Get model outputs including intermediate features
                    outputs = self.model(images, return_features=True)
                    predictions = outputs['predictions']
                    features = outputs.get('features', {})
                    
                    # Main detection loss
                    detection_losses = self.criterion(predictions, targets)
                    
                    # Additional component losses
                    component_losses = self._compute_component_losses(features, targets)
                    
                    # Combine losses
                    total_loss = self._combine_losses(detection_losses, component_losses)
                
                # Backward pass
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if self.config.get('grad_clip_norm'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip_norm']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Get model outputs including intermediate features
                outputs = self.model(images, return_features=True)
                predictions = outputs['predictions']
                features = outputs.get('features', {})
                
                # Main detection loss
                detection_losses = self.criterion(predictions, targets)
                
                # Additional component losses
                component_losses = self._compute_component_losses(features, targets)
                
                # Combine losses
                total_loss = self._combine_losses(detection_losses, component_losses)
                
                total_loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip_norm']
                    )
                
                self.optimizer.step()
            
            # Update losses
            epoch_loss += total_loss.item()
            epoch_losses['total_loss'] += total_loss.item()
            
            if isinstance(detection_losses, dict):
                for key, value in detection_losses.items():
                    epoch_losses[f'detection_{key}'] += value.item()
            
            for key, value in component_losses.items():
                epoch_losses[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}
        
        # Step scheduler
        if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        return avg_losses
    
    def _compute_component_losses(self, features: Dict[str, torch.Tensor], 
                                targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """Compute losses for CBAM and STN components"""
        component_losses = {}
        
        # CBAM attention loss (encourage attention on objects)
        if 'attention_maps' in features:
            attention_loss = self._compute_attention_loss(features['attention_maps'], targets)
            component_losses['attention_loss'] = attention_loss
        
        # STN transformation loss (encourage smooth transformations)
        if 'transformation_matrices' in features:
            stn_loss = self._compute_stn_loss(features['transformation_matrices'])
            component_losses['stn_loss'] = stn_loss
        
        # TPS control points loss (regularization)
        if 'tps_control_points' in features:
            tps_loss = self._compute_tps_loss(features['tps_control_points'])
            component_losses['tps_loss'] = tps_loss
        
        return component_losses
    
    def _compute_attention_loss(self, attention_maps: torch.Tensor, 
                              targets: List[Dict]) -> torch.Tensor:
        """Compute attention loss to encourage focus on objects"""
        # Simplified attention loss - encourage high attention on object regions
        attention_loss = torch.tensor(0.0, device=attention_maps.device)
        
        for i, target in enumerate(targets):
            if 'boxes' in target and len(target['boxes']) > 0:
                # Create attention mask from bounding boxes
                H, W = attention_maps.shape[-2:]
                mask = torch.zeros((H, W), device=attention_maps.device)
                
                boxes = target['boxes']  # Normalized coordinates
                for box in boxes:
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                    mask[y1:y2, x1:x2] = 1.0
                
                # Encourage high attention on object regions
                if attention_maps.dim() == 4:  # [B, C, H, W]
                    att_map = attention_maps[i].mean(dim=0)  # Average across channels
                else:
                    att_map = attention_maps[i]
                
                # Loss: negative attention on object regions
                attention_loss += -torch.mean(att_map * mask)
        
        return attention_loss / len(targets)
    
    def _compute_stn_loss(self, transformation_matrices: torch.Tensor) -> torch.Tensor:
        """Compute STN loss to encourage smooth transformations"""
        # Encourage transformations close to identity
        batch_size = transformation_matrices.shape[0]
        identity = torch.eye(2, 3, device=transformation_matrices.device)
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
        
        stn_loss = F.mse_loss(transformation_matrices, identity)
        return stn_loss
    
    def _compute_tps_loss(self, control_points: torch.Tensor) -> torch.Tensor:
        """Compute TPS loss for regularization"""
        # Encourage small deformations
        tps_loss = torch.mean(control_points ** 2)
        return tps_loss
    
    def _combine_losses(self, detection_losses: Union[Dict, torch.Tensor], 
                       component_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine all losses with appropriate weights"""
        if isinstance(detection_losses, dict):
            total_loss = detection_losses['total_loss']
        else:
            total_loss = detection_losses
        
        # Add component losses
        if 'attention_loss' in component_losses:
            total_loss += self.attention_loss_weight * component_losses['attention_loss']
        
        if 'stn_loss' in component_losses:
            total_loss += self.stn_loss_weight * component_losses['stn_loss']
        
        if 'tps_loss' in component_losses:
            total_loss += self.config.get('tps_loss_weight', 0.01) * component_losses['tps_loss']
        
        return total_loss

class MultiGPUTrainer(CBAMSTNTPSYOLOTrainer):
    """Multi-GPU distributed training support"""
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize distributed training
        self.setup_distributed()
        
        super().__init__(config)
        
        # Wrap model with DDP
        if self.config.get('distributed', False):
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=config.get('find_unused_parameters', False)
            )
    
    def setup_distributed(self):
        """Setup distributed training"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            
            # Initialize process group
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )
            
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            
            self.config['distributed'] = True
            logger.info(f"✅ Distributed training initialized: rank {self.rank}/{self.world_size}")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.config['distributed'] = False
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders with distributed samplers"""
        data_config = self.config.get('data', {})
        
        # Create datasets first
        from ..data.dataset import create_dataset
        
        train_dataset = create_dataset(
            data_config.get('train_path'),
            augmentations=data_config.get('train_augmentations', True),
            input_size=self.config.get('input_size', 640),
            num_classes=self.config.get('num_classes', 3),
            spectral_bands=self.config.get('spectral_bands', 4)
        )
        
        val_dataset = create_dataset(
            data_config.get('val_path'),
            augmentations=False,
            input_size=self.config.get('input_size', 640),
            num_classes=self.config.get('num_classes', 3),
            spectral_bands=self.config.get('spectral_bands', 4)
        )
        
        # Create samplers
        if self.config.get('distributed', False):
            train_sampler = DistributedSampler(
                train_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 16),
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 16),
            sampler=val_sampler,
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            drop_last=False
        )
        
        return train_loader, val_loader

class ResumableTrainer(MultiGPUTrainer):
    """Trainer with advanced resumption and fault tolerance"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Automatic checkpoint saving
        self.auto_save_interval = config.get('auto_save_interval', 5)  # epochs
        self.max_checkpoints = config.get('max_checkpoints', 5)
        
        # Fault tolerance
        self.max_retries = config.get('max_retries', 3)
        self.retry_count = 0
    
    def train(self) -> Dict[str, Any]:
        """Training with fault tolerance"""
        while self.retry_count < self.max_retries:
            try:
                return super().train()
            except Exception as e:
                self.retry_count += 1
                logger.error(f"Training failed (attempt {self.retry_count}/{self.max_retries}): {e}")
                
                if self.retry_count < self.max_retries:
                    logger.info("Attempting to resume from last checkpoint...")
                    self._recover_from_failure()
                else:
                    logger.error("Max retries exceeded, training failed")
                    raise
        
        return {}
    
    def _recover_from_failure(self):
        """Recover from training failure"""
        # Find latest checkpoint
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
            logger.info(f"Recovering from {latest_checkpoint}")
            self.load_checkpoint(str(latest_checkpoint))
        else:
            logger.warning("No checkpoints found, starting from scratch")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, final: bool = False):
        """Enhanced checkpoint saving with rotation"""
        super().save_checkpoint(epoch, is_best, final)
        
        # Cleanup old checkpoints
        if not is_best and not final:
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space"""
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if len(checkpoint_files) > self.max_checkpoints:
            # Sort by epoch number
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            
            # Remove oldest checkpoints
            for checkpoint in checkpoint_files[:-self.max_checkpoints]:
                checkpoint.unlink()
                logger.info(f"🗑️  Removed old checkpoint: {checkpoint}")

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        
        self.mode_worse = np.less if mode == 'max' else np.greater
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop"""
        if self.best_score is None:
            self.best_score = score
        elif self.mode_worse(score, self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        
        return False

# Factory functions
def create_trainer(config: Dict[str, Any], trainer_type: str = 'cbam_stn_tps_yolo') -> BaseTrainer:
    """Factory function to create trainers"""
    trainer_registry = {
        'base': BaseTrainer,
        'cbam_stn_tps_yolo': CBAMSTNTPSYOLOTrainer,
        'multi_gpu': MultiGPUTrainer,
        'resumable': ResumableTrainer
    }
    
    trainer_type = trainer_type.lower()
    if trainer_type not in trainer_registry:
        raise ValueError(f"Unknown trainer type: {trainer_type}")
    
    return trainer_registry[trainer_type](config)

if __name__ == "__main__":
    # Test trainer creation
    print("Testing trainer creation...")
    
    config = {
        'model': {'type': 'CBAM-STN-TPS-YOLO', 'num_classes': 3},
        'data': {'train_path': './data/train', 'val_path': './data/val'},
        'optimizer': {'type': 'AdamW', 'lr': 1e-3},
        'scheduler': {'type': 'CosineAnnealingLR', 'T_max': 100},
        'loss': {'type': 'yolo'},
        'epochs': 100,
        'batch_size': 16,
        'device': 'cpu'  # For testing
    }
    
    try:
        trainer = create_trainer(config, 'cbam_stn_tps_yolo')
        print("✅ Trainer created successfully")
        print(f"   Model: {type(trainer.model).__name__}")
        print(f"   Optimizer: {type(trainer.optimizer).__name__}")
        print(f"   Criterion: {type(trainer.criterion).__name__}")
        
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
    
    print("Trainer tests completed! 🎉")