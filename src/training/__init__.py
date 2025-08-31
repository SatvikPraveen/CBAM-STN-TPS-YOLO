# src/training/__init__.py
"""
Enhanced training infrastructure for CBAM-STN-TPS-YOLO
Comprehensive training, loss functions, and metrics for agricultural object detection
"""

import warnings
from pathlib import Path

# Core training components
from .trainer import (
    CBAMSTNTPSYOLOTrainer,
    MultiGPUTrainer,
    ResumableTrainer,
    create_trainer
)

from .losses import (
    YOLOLoss,
    CIoULoss,
    DistributedFocalLoss,
    ComboLoss,
    AdaptiveLoss,
    TverskyLoss,
    calculate_loss
)

from .metrics import (
    DetectionMetrics,
    AdvancedDetectionMetrics,
    ClassificationMetrics,
    non_max_suppression,
    calculate_iou,
    calculate_giou,
    calculate_diou,
    calculate_ciou,
    compute_ap,
    compute_precision_recall,
    compute_f1_score
)

# Training utilities
def get_available_optimizers():
    """Get list of available optimizers"""
    return ['Adam', 'AdamW', 'SGD', 'RMSprop', 'Ranger', 'LAMB']

def get_available_schedulers():
    """Get list of available learning rate schedulers"""
    return [
        'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau',
        'MultiStepLR', 'ExponentialLR', 'LinearLR', 'OneCycleLR'
    ]

def get_available_loss_functions():
    """Get list of available loss functions"""
    return ['YOLOLoss', 'CIoULoss', 'DistributedFocalLoss', 'ComboLoss', 'AdaptiveLoss', 'TverskyLoss']

def create_training_pipeline(config: dict, model_type: str = 'CBAM-STN-TPS-YOLO'):
    """Create complete training pipeline
    
    Args:
        config: Training configuration dictionary
        model_type: Type of model to train
        
    Returns:
        tuple: (trainer, data_loaders, optimizer, scheduler, criterion)
    """
    # Create trainer
    trainer = create_trainer(config, model_type)
    
    # Get components
    data_loaders = trainer.get_data_loaders()
    optimizer = trainer.get_optimizer()
    scheduler = trainer.get_scheduler()
    criterion = trainer.get_criterion()
    
    return trainer, data_loaders, optimizer, scheduler, criterion

def validate_training_setup(config: dict) -> bool:
    """Validate training configuration and setup"""
    from ..utils.config_validator import ConfigValidator
    
    try:
        validator = ConfigValidator()
        validator.validate_training_config(config)
        return True
    except Exception as e:
        warnings.warn(f"Training setup validation failed: {e}")
        return False

def estimate_training_time(config: dict, model_type: str = 'CBAM-STN-TPS-YOLO') -> dict:
    """Estimate training time and resource requirements"""
    import torch
    
    # Basic estimates (simplified)
    base_time_per_epoch = {
        'YOLO': 10,
        'CBAM-YOLO': 12,
        'STN-YOLO': 15,
        'STN-TPS-YOLO': 18,
        'CBAM-STN-YOLO': 20,
        'CBAM-STN-TPS-YOLO': 25
    }
    
    batch_size = config.get('batch_size', 16)
    epochs = config.get('epochs', 100)
    image_size = config.get('image_size', 640)
    
    # Adjust for batch size and image size
    base_time = base_time_per_epoch.get(model_type, 20)
    time_factor = (image_size / 640) ** 2 * (16 / batch_size)
    
    estimated_time_per_epoch = base_time * time_factor
    total_estimated_time = estimated_time_per_epoch * epochs
    
    # Memory estimation
    memory_per_sample = (image_size ** 2 * 4 * 4) / (1024 ** 2)  # MB per sample
    estimated_memory = memory_per_sample * batch_size * 2  # Factor of 2 for gradients
    
    return {
        'estimated_time_per_epoch_minutes': estimated_time_per_epoch,
        'total_estimated_time_hours': total_estimated_time / 60,
        'estimated_memory_mb': estimated_memory,
        'recommended_gpu_memory_gb': max(8, estimated_memory / 1024 * 1.5),
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

def setup_training_environment(config: dict) -> dict:
    """Setup optimal training environment"""
    import torch
    import os
    
    environment_info = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'mixed_precision_available': hasattr(torch.cuda.amp, 'GradScaler'),
        'distributed_available': torch.distributed.is_available(),
        'optimizations_applied': []
    }
    
    # Set optimal thread count for CPU
    if not torch.cuda.is_available():
        torch.set_num_threads(min(8, os.cpu_count()))
        environment_info['optimizations_applied'].append('CPU threads optimized')
    
    # Enable CUDNN optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        environment_info['optimizations_applied'].append('CUDNN optimizations enabled')
    
    # Set memory format for better performance
    if config.get('channels_last', False):
        environment_info['memory_format'] = 'channels_last'
        environment_info['optimizations_applied'].append('Channels last memory format')
    
    return environment_info

def create_training_summary(trainer, config: dict) -> dict:
    """Create comprehensive training summary"""
    
    summary = {
        'model_info': {
            'model_type': trainer.model_type,
            'total_parameters': sum(p.numel() for p in trainer.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() for p in trainer.model.parameters()) * 4 / (1024 * 1024)
        },
        'training_config': {
            'epochs': config.get('epochs', 100),
            'batch_size': config.get('batch_size', 16),
            'learning_rate': config.get('lr', 0.001),
            'optimizer': config.get('optimizer', 'Adam'),
            'scheduler': config.get('scheduler', 'CosineAnnealingLR'),
            'mixed_precision': config.get('mixed_precision', False)
        },
        'dataset_info': {
            'train_samples': len(trainer.train_loader.dataset) if hasattr(trainer, 'train_loader') else 0,
            'val_samples': len(trainer.val_loader.dataset) if hasattr(trainer, 'val_loader') else 0,
            'num_classes': config.get('num_classes', 3),
            'input_channels': config.get('input_channels', 4)
        }
    }
    
    return summary

def log_training_info(config: dict, model_type: str = 'CBAM-STN-TPS-YOLO'):
    """Log comprehensive training information"""
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Epochs: {config.get('epochs', 100)}")
    logger.info(f"Batch Size: {config.get('batch_size', 16)}")
    logger.info(f"Learning Rate: {config.get('lr', 0.001)}")
    logger.info(f"Image Size: {config.get('image_size', 640)}")
    logger.info(f"Mixed Precision: {config.get('mixed_precision', False)}")
    
    # Log estimates
    estimates = estimate_training_time(config, model_type)
    logger.info("\nTRAINING ESTIMATES")
    logger.info("-" * 40)
    logger.info(f"Time per epoch: {estimates['estimated_time_per_epoch_minutes']:.1f} minutes")
    logger.info(f"Total time: {estimates['total_estimated_time_hours']:.1f} hours")
    logger.info(f"Memory usage: {estimates['estimated_memory_mb']:.0f} MB")
    logger.info(f"Recommended GPU: {estimates['recommended_gpu_memory_gb']:.0f} GB")
    
    # Log environment
    env_info = setup_training_environment(config)
    logger.info(f"\nENVIRONMENT")
    logger.info("-" * 40)
    logger.info(f"Device: {env_info['device']}")
    logger.info(f"Optimizations: {', '.join(env_info['optimizations_applied'])}")
    logger.info("=" * 80)

# Export all public components
__all__ = [
    # Trainer classes
    'CBAMSTNTPSYOLOTrainer', 'MultiGPUTrainer', 'ResumableTrainer', 'create_trainer',
    
    # Loss functions
    'YOLOLoss', 'CIoULoss', 'DistributedFocalLoss', 'ComboLoss', 'AdaptiveLoss', 
    'TverskyLoss', 'calculate_loss',
    
    # Metrics
    'DetectionMetrics', 'AdvancedDetectionMetrics', 'ClassificationMetrics',
    'non_max_suppression', 'calculate_iou', 'calculate_giou', 'calculate_diou', 
    'calculate_ciou', 'compute_ap', 'compute_precision_recall', 'compute_f1_score',
    
    # Utility functions
    'get_available_optimizers', 'get_available_schedulers', 'get_available_loss_functions',
    'create_training_pipeline', 'validate_training_setup', 'estimate_training_time',
    'setup_training_environment', 'create_training_summary', 'log_training_info'
]