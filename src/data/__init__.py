"""
Data loading and preprocessing utilities for CBAM-STN-TPS-YOLO
Enhanced agricultural object detection with multi-spectral imaging support
"""

import warnings
from pathlib import Path

# Core data handling components
from .dataset import (
    PGPDataset, 
    MelonFlowerDataset, 
    GlobalWheatDataset,
    MultiSpectralDataset,
    BaseAgriculturalDataset,
    get_dataset,
    validate_dataset_structure
)

from .transforms import (
    Compose, 
    RandomRotation, 
    RandomShear, 
    RandomCrop,
    ColorJitter, 
    Normalize, 
    ToTensor,
    TestAugmentations,
    SpectralTransform,
    PseudoRGBConversion,
    GeometricTransform,
    create_train_transforms,
    create_val_transforms,
    create_test_transforms
)

from .preprocessing import (
    DatasetPreprocessor,
    create_data_loaders,
    collate_fn,
    preprocess_pgp_data,
    preprocess_melon_flower_data, 
    preprocess_global_wheat_data,
    DataPreprocessor,
    SpectralDataProcessor,
    AugmentationPipeline
)

# Utility functions
def get_available_datasets():
    """Get list of available dataset classes"""
    return {
        'PGP': PGPDataset,
        'MelonFlower': MelonFlowerDataset,
        'GlobalWheat': GlobalWheatDataset,
        'MultiSpectral': MultiSpectralDataset
    }

def create_agricultural_pipeline(
    data_dir, 
    dataset_type='PGP', 
    batch_size=16,
    image_size=640,
    multi_spectral=True,
    num_workers=4,
    augmentation_level='medium'
):
    """Create complete data pipeline for agricultural datasets
    
    Args:
        data_dir: Path to dataset directory
        dataset_type: Type of dataset ('PGP', 'MelonFlower', 'GlobalWheat')
        batch_size: Batch size for data loaders
        image_size: Target image size
        multi_spectral: Whether to use multi-spectral processing
        num_workers: Number of data loading workers
        augmentation_level: 'light', 'medium', 'heavy'
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset_info)
    """
    # Create transforms based on augmentation level
    if augmentation_level == 'light':
        train_transform = create_train_transforms(
            image_size=image_size,
            rotation_degrees=5,
            color_jitter_prob=0.3,
            geometric_prob=0.3
        )
    elif augmentation_level == 'medium':
        train_transform = create_train_transforms(
            image_size=image_size,
            rotation_degrees=10,
            color_jitter_prob=0.5,
            geometric_prob=0.5
        )
    else:  # heavy
        train_transform = create_train_transforms(
            image_size=image_size,
            rotation_degrees=15,
            color_jitter_prob=0.7,
            geometric_prob=0.7
        )
    
    val_transform = create_val_transforms(image_size=image_size)
    test_transform = create_test_transforms(image_size=image_size)
    
    # Create datasets
    dataset_class = get_available_datasets()[dataset_type]
    
    train_dataset = dataset_class(
        data_dir=data_dir,
        split='train',
        transform=train_transform,
        multi_spectral=multi_spectral,
        image_size=image_size
    )
    
    val_dataset = dataset_class(
        data_dir=data_dir,
        split='val',
        transform=val_transform,
        multi_spectral=multi_spectral,
        image_size=image_size
    )
    
    test_dataset = dataset_class(
        data_dir=data_dir,
        split='test',
        transform=test_transform,
        multi_spectral=multi_spectral,
        image_size=image_size
    )
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    dataset_info = {
        'num_classes': getattr(train_dataset, 'num_classes', 3),
        'class_names': getattr(train_dataset, 'class_names', ['cotton', 'rice', 'corn']),
        'input_channels': 4 if multi_spectral else 3,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }
    
    return train_loader, val_loader, test_loader, dataset_info

def validate_data_directory(data_dir, dataset_type='PGP'):
    """Validate dataset directory structure"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Check for required subdirectories
    required_splits = ['train', 'val']
    missing_splits = []
    
    for split in required_splits:
        split_path = data_path / split
        if not split_path.exists():
            # Try alternative structure
            split_path = data_path / 'images' / split
            if not split_path.exists():
                missing_splits.append(split)
    
    if missing_splits:
        warnings.warn(
            f"Missing splits: {missing_splits}. "
            f"Expected structure: {data_dir}/{{train,val}}/{{images,labels}}/"
        )
    
    return True

# Export all public components
__all__ = [
    # Dataset classes
    'PGPDataset', 'MelonFlowerDataset', 'GlobalWheatDataset', 'MultiSpectralDataset',
    'BaseAgriculturalDataset', 'get_dataset', 'validate_dataset_structure',
    
    # Transform classes
    'Compose', 'RandomRotation', 'RandomShear', 'RandomCrop',
    'ColorJitter', 'Normalize', 'ToTensor', 'TestAugmentations',
    'SpectralTransform', 'PseudoRGBConversion', 'GeometricTransform',
    
    # Transform creators
    'create_train_transforms', 'create_val_transforms', 'create_test_transforms',
    
    # Preprocessing utilities
    'DatasetPreprocessor', 'create_data_loaders', 'collate_fn',
    'preprocess_pgp_data', 'preprocess_melon_flower_data', 'preprocess_global_wheat_data',
    'DataPreprocessor', 'SpectralDataProcessor', 'AugmentationPipeline',
    
    # Pipeline utilities
    'get_available_datasets', 'create_agricultural_pipeline', 'validate_data_directory'
]