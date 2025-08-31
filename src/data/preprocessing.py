# src/data/preprocessing.py
import torch
from torch.utils.data import DataLoader
from .transforms import Compose, RandomRotation, RandomShear, ColorJitter, Normalize
from .dataset import PGPDataset, MelonFlowerDataset, GlobalWheatDataset, collate_fn
import os
import shutil
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import json
import yaml
import cv2
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Advanced data preprocessing for agricultural datasets"""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'processed'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocessing statistics
        self.stats = {
            'total_images': 0,
            'valid_images': 0,
            'corrupted_images': 0,
            'total_annotations': 0,
            'class_distribution': {},
            'image_sizes': [],
            'processing_errors': []
        }
    
    def analyze_dataset(self) -> Dict:
        """Analyze dataset structure and quality"""
        logger.info("Analyzing dataset structure...")
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.data_dir.rglob(ext)))
        
        self.stats['total_images'] = len(image_files)
        
        # Analyze image quality and properties
        for img_path in image_files:
            try:
                # Check if image can be loaded
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.stats['valid_images'] += 1
                    self.stats['image_sizes'].append(img.shape[:2])
                else:
                    self.stats['corrupted_images'] += 1
                    self.stats['processing_errors'].append(f"Cannot load: {img_path}")
            except Exception as e:
                self.stats['corrupted_images'] += 1
                self.stats['processing_errors'].append(f"Error loading {img_path}: {e}")
        
        # Analyze annotations
        self._analyze_annotations()
        
        # Generate summary
        summary = self._generate_analysis_summary()
        
        # Save analysis results
        analysis_file = self.output_dir / 'dataset_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        logger.info(f"Dataset analysis saved to {analysis_file}")
        return summary
    
    def _analyze_annotations(self):
        """Analyze annotation files and class distribution"""
        annotation_files = list(self.data_dir.rglob('*.json')) + list(self.data_dir.rglob('*.txt'))
        
        for ann_file in annotation_files:
            try:
                if ann_file.suffix == '.json':
                    self._analyze_json_annotations(ann_file)
                elif ann_file.suffix == '.txt' and 'classes' not in ann_file.stem:
                    self._analyze_txt_annotations(ann_file)
            except Exception as e:
                self.stats['processing_errors'].append(f"Annotation error {ann_file}: {e}")
    
    def _analyze_json_annotations(self, ann_file: Path):
        """Analyze JSON annotation file"""
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        if 'annotations' in data:
            # COCO format
            for ann in data['annotations']:
                cat_id = ann.get('category_id', 'unknown')
                self.stats['class_distribution'][str(cat_id)] = \
                    self.stats['class_distribution'].get(str(cat_id), 0) + 1
                self.stats['total_annotations'] += 1
    
    def _analyze_txt_annotations(self, ann_file: Path):
        """Analyze text annotation file"""
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = parts[0]
                    self.stats['class_distribution'][class_id] = \
                        self.stats['class_distribution'].get(class_id, 0) + 1
                    self.stats['total_annotations'] += 1
    
    def _generate_analysis_summary(self) -> Dict:
        """Generate human-readable analysis summary"""
        if self.stats['image_sizes']:
            sizes_array = np.array(self.stats['image_sizes'])
            height_stats = {
                'min': int(sizes_array[:, 0].min()),
                'max': int(sizes_array[:, 0].max()),
                'mean': int(sizes_array[:, 0].mean()),
                'std': int(sizes_array[:, 0].std())
            }
            width_stats = {
                'min': int(sizes_array[:, 1].min()),
                'max': int(sizes_array[:, 1].max()),
                'mean': int(sizes_array[:, 1].mean()),
                'std': int(sizes_array[:, 1].std())
            }
        else:
            height_stats = width_stats = {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        
        return {
            'total_images': self.stats['total_images'],
            'valid_images': self.stats['valid_images'],
            'corrupted_images': self.stats['corrupted_images'],
            'corruption_rate': self.stats['corrupted_images'] / max(self.stats['total_images'], 1),
            'total_annotations': self.stats['total_annotations'],
            'annotations_per_image': self.stats['total_annotations'] / max(self.stats['valid_images'], 1),
            'class_distribution': self.stats['class_distribution'],
            'image_dimensions': {
                'height': height_stats,
                'width': width_stats
            },
            'errors': self.stats['processing_errors']
        }
    
    def clean_dataset(self, remove_corrupted: bool = True, 
                     min_annotation_size: float = 0.001) -> Dict:
        """Clean dataset by removing corrupted images and invalid annotations"""
        logger.info("Cleaning dataset...")
        
        cleaned_stats = {
            'removed_images': 0,
            'removed_annotations': 0,
            'cleaned_files': []
        }
        
        if remove_corrupted:
            # Remove corrupted images
            for error in self.stats['processing_errors']:
                if 'Cannot load:' in error or 'Error loading' in error:
                    # Extract file path and remove
                    error_parts = error.split(': ')
                    if len(error_parts) > 1:
                        corrupted_file = Path(error_parts[1])
                        if corrupted_file.exists():
                            try:
                                corrupted_file.unlink()
                                cleaned_stats['removed_images'] += 1
                                cleaned_stats['cleaned_files'].append(str(corrupted_file))
                                logger.info(f"Removed corrupted image: {corrupted_file}")
                            except Exception as e:
                                logger.warning(f"Could not remove {corrupted_file}: {e}")
        
        # Clean annotations (remove boxes that are too small)
        annotation_files = list(self.data_dir.rglob('*.txt'))
        
        for ann_file in annotation_files:
            if 'classes' in ann_file.stem:
                continue
                
            try:
                cleaned_annotations = []
                with open(ann_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls, x, y, w, h = map(float, parts)
                            # Keep annotations above minimum size threshold
                            if w >= min_annotation_size and h >= min_annotation_size:
                                cleaned_annotations.append(line)
                            else:
                                cleaned_stats['removed_annotations'] += 1
                
                # Write back cleaned annotations
                if len(cleaned_annotations) != sum(1 for _ in open(ann_file)):
                    with open(ann_file, 'w') as f:
                        f.writelines(cleaned_annotations)
                    cleaned_stats['cleaned_files'].append(str(ann_file))
                    
            except Exception as e:
                logger.warning(f"Could not clean annotations in {ann_file}: {e}")
        
        logger.info(f"Dataset cleaning completed. Removed {cleaned_stats['removed_images']} images "
                   f"and {cleaned_stats['removed_annotations']} annotations")
        
        return cleaned_stats

class DatasetPreprocessor:
    """Legacy preprocessor for backward compatibility"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        logger.warning("DatasetPreprocessor is deprecated. Use DataPreprocessor instead.")
    
    def setup_pgp_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                         test_ratio: float = 0.15, seed: int = 42):
        """Setup PGP dataset with proper train/val/test splits"""
        return setup_dataset_splits(
            self.data_dir, 
            train_ratio=train_ratio,
            val_ratio=val_ratio, 
            test_ratio=test_ratio,
            seed=seed
        )
    
    def convert_to_yolo_format(self, annotation_file: str, output_dir: str, 
                              class_mapping: Dict = None):
        """Convert various annotation formats to YOLO format"""
        return convert_annotations_to_yolo(annotation_file, output_dir, class_mapping)

class SpectralDataProcessor:
    """Specialized processor for multi-spectral agricultural data"""
    
    def __init__(self, data_dir: str, spectral_bands: Dict[str, str] = None):
        self.data_dir = Path(data_dir)
        self.spectral_bands = spectral_bands or {
            'green': '580nm',
            'red': '660nm',
            'red_edge': '730nm',
            'nir': '820nm'
        }
    
    def process_multispectral_images(self, output_format: str = 'pseudo_rgb') -> Dict:
        """Process multi-spectral images into specified format"""
        logger.info(f"Processing multi-spectral images to {output_format}...")
        
        processed_count = 0
        error_count = 0
        
        # Find all image base names (without spectral suffixes)
        image_files = list(self.data_dir.rglob('*.tiff')) + list(self.data_dir.rglob('*.tif'))
        base_names = set()
        
        for img_file in image_files:
            # Extract base name by removing spectral band suffix
            base_name = str(img_file)
            for band_suffix in self.spectral_bands.values():
                if f'_{band_suffix}' in base_name:
                    base_name = base_name.replace(f'_{band_suffix}', '')
                    break
            base_names.add(base_name)
        
        output_dir = self.data_dir / f'processed_{output_format}'
        output_dir.mkdir(exist_ok=True)
        
        for base_name in base_names:
            try:
                # Load all spectral bands
                bands = {}
                for band_name, band_suffix in self.spectral_bands.items():
                    band_file = f'{base_name}_{band_suffix}.tiff'
                    if os.path.exists(band_file):
                        band_img = cv2.imread(band_file, cv2.IMREAD_ANYDEPTH)
                        if band_img is not None:
                            bands[band_name] = band_img
                
                if len(bands) >= 3:
                    if output_format == 'pseudo_rgb':
                        processed_img = self._create_pseudo_rgb(bands)
                    elif output_format == 'ndvi_enhanced':
                        processed_img = self._create_ndvi_enhanced(bands)
                    else:
                        processed_img = self._create_four_band(bands)
                    
                    # Save processed image
                    output_file = output_dir / f'{Path(base_name).stem}_processed.png'
                    cv2.imwrite(str(output_file), processed_img)
                    processed_count += 1
                
            except Exception as e:
                error_count += 1
                logger.warning(f"Error processing {base_name}: {e}")
        
        logger.info(f"Processed {processed_count} multi-spectral images, {error_count} errors")
        
        return {
            'processed_count': processed_count,
            'error_count': error_count,
            'output_directory': str(output_dir)
        }
    
    def _create_pseudo_rgb(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Create pseudo-RGB from spectral bands"""
        if 'red' in bands and 'red_edge' in bands and 'green' in bands:
            pseudo_rgb = np.stack([bands['red'], bands['red_edge'], bands['green']], axis=-1)
        elif 'red' in bands and 'green' in bands and 'nir' in bands:
            pseudo_rgb = np.stack([bands['red'], bands['green'], bands['nir']], axis=-1)
        else:
            # Use first 3 available bands
            available_bands = list(bands.values())[:3]
            pseudo_rgb = np.stack(available_bands, axis=-1)
        
        # Normalize to 0-255
        for i in range(3):
            band = pseudo_rgb[:, :, i].astype(np.float32)
            if band.max() > band.min():
                band = (band - band.min()) / (band.max() - band.min())
                pseudo_rgb[:, :, i] = (band * 255).astype(np.uint8)
            else:
                pseudo_rgb[:, :, i] = 0
        
        return pseudo_rgb
    
    def _create_ndvi_enhanced(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Create NDVI-enhanced image"""
        if 'red' in bands and 'nir' in bands:
            red = bands['red'].astype(np.float32)
            nir = bands['nir'].astype(np.float32)
            
            # Calculate NDVI
            ndvi = (nir - red) / (nir + red + 1e-8)
            
            # Normalize NDVI to 0-255
            ndvi_norm = ((ndvi + 1) / 2 * 255).astype(np.uint8)
            
            # Create enhanced RGB with NDVI as green channel
            if 'green' in bands:
                enhanced = np.stack([bands['red'], ndvi_norm, bands['green']], axis=-1)
            else:
                enhanced = np.stack([bands['red'], ndvi_norm, bands['red']], axis=-1)
            
            return enhanced
        else:
            return self._create_pseudo_rgb(bands)
    
    def _create_four_band(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Create 4-band image (RGBN)"""
        band_order = ['red', 'green', 'green', 'nir']  # Duplicate green for 4 channels
        if 'red_edge' in bands:
            band_order[2] = 'red_edge'
        
        four_band = []
        for band_name in band_order:
            if band_name in bands:
                four_band.append(bands[band_name])
            else:
                # Use zeros if band not available
                four_band.append(np.zeros_like(list(bands.values())[0]))
        
        return np.stack(four_band, axis=-1)

class AugmentationPipeline:
    """Pipeline for applying data augmentation strategies"""
    
    def __init__(self, strategy: str = 'medium'):
        self.strategy = strategy
        self.augmentation_configs = {
            'light': {
                'rotation_degrees': 5,
                'shear_degrees': 3,
                'crop_factor': 0.05,
                'color_jitter_prob': 0.3,
                'geometric_prob': 0.3
            },
            'medium': {
                'rotation_degrees': 10,
                'shear_degrees': 5,
                'crop_factor': 0.1,
                'color_jitter_prob': 0.5,
                'geometric_prob': 0.5
            },
            'heavy': {
                'rotation_degrees': 15,
                'shear_degrees': 8,
                'crop_factor': 0.15,
                'color_jitter_prob': 0.7,
                'geometric_prob': 0.7
            },
            'agricultural': {
                'rotation_degrees': 12,
                'shear_degrees': 6,
                'crop_factor': 0.12,
                'color_jitter_prob': 0.6,
                'geometric_prob': 0.6,
                'spectral_enhance': True
            }
        }
    
    def get_augmentation_config(self) -> Dict:
        """Get augmentation configuration for current strategy"""
        return self.augmentation_configs.get(self.strategy, self.augmentation_configs['medium'])
    
    def create_pipeline(self, image_size: int = 640):
        """Create augmentation pipeline based on strategy"""
        from .transforms import (create_train_transforms, create_val_transforms, 
                               create_test_transforms)
        
        config = self.get_augmentation_config()
        
        return {
            'train': create_train_transforms(
                image_size=image_size,
                rotation_degrees=config['rotation_degrees'],
                color_jitter_prob=config['color_jitter_prob'],
                geometric_prob=config['geometric_prob']
            ),
            'val': create_val_transforms(image_size=image_size),
            'test': create_test_transforms(image_size=image_size)
        }

# Utility functions for backward compatibility and common operations

def setup_dataset_splits(data_dir: str, train_ratio: float = 0.7, 
                        val_ratio: float = 0.15, test_ratio: float = 0.15,
                        seed: int = 42) -> Dict:
    """Setup dataset with train/val/test splits"""
    data_path = Path(data_dir)
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Create directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (data_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (data_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(data_path.glob(f'*{ext}')))
    
    if len(image_files) == 0:
        logger.warning(f"No images found in {data_dir}")
        return {'train': 0, 'val': 0, 'test': 0}
    
    # Split dataset
    np.random.seed(seed)
    train_files, temp_files = train_test_split(
        image_files, 
        test_size=(1 - train_ratio), 
        random_state=seed
    )
    
    val_files, test_files = train_test_split(
        temp_files, 
        test_size=(test_ratio / (val_ratio + test_ratio)), 
        random_state=seed
    )
    
    # Move files to appropriate directories
    file_splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split, files in file_splits.items():
        for img_file in files:
            # Copy image
            dst_img = data_path / split / 'images' / img_file.name
            if not dst_img.exists():
                shutil.copy2(img_file, dst_img)
            
            # Copy corresponding label if exists
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                dst_label = data_path / split / 'labels' / label_file.name
                if not dst_label.exists():
                    shutil.copy2(label_file, dst_label)
    
    split_info = {
        'train': len(train_files),
        'val': len(val_files),
        'test': len(test_files)
    }
    
    logger.info(f"Dataset split completed: {split_info}")
    return split_info

def convert_annotations_to_yolo(annotation_file: str, output_dir: str, 
                               class_mapping: Dict = None) -> Dict:
    """Convert various annotation formats to YOLO format"""
    annotation_path = Path(annotation_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    conversion_stats = {
        'converted_files': 0,
        'total_annotations': 0,
        'errors': []
    }
    
    try:
        if annotation_path.suffix == '.json':
            # Load JSON annotations
            with open(annotation_path, 'r') as f:
                data = json.load(f)
            
            # Process COCO format
            if 'images' in data and 'annotations' in data:
                conversion_stats = _convert_coco_to_yolo(data, output_path, class_mapping)
            else:
                logger.warning("Unknown JSON annotation format")
        
        elif annotation_path.suffix == '.csv':
            # Handle CSV format (like Global Wheat)
            conversion_stats = _convert_csv_to_yolo(annotation_path, output_path, class_mapping)
        
        else:
            logger.warning(f"Unsupported annotation format: {annotation_path.suffix}")
    
    except Exception as e:
        conversion_stats['errors'].append(str(e))
        logger.error(f"Conversion failed: {e}")
    
    return conversion_stats

def _convert_coco_to_yolo(coco_data: Dict, output_dir: Path, class_mapping: Dict) -> Dict:
    """Convert COCO format to YOLO format"""
    images = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    conversion_stats = {'converted_files': 0, 'total_annotations': 0, 'errors': []}
    
    # Create YOLO labels
    for img_id, img_info in images.items():
        try:
            label_file = output_dir / f"{Path(img_info['file_name']).stem}.txt"
            
            with open(label_file, 'w') as f:
                if img_id in image_annotations:
                    for ann in image_annotations[img_id]:
                        # Convert bbox from [x, y, w, h] to normalized [x_center, y_center, w, h]
                        bbox = ann['bbox']
                        img_w, img_h = img_info['width'], img_info['height']
                        
                        x_center = (bbox[0] + bbox[2] / 2) / img_w
                        y_center = (bbox[1] + bbox[3] / 2) / img_h
                        width = bbox[2] / img_w
                        height = bbox[3] / img_h
                        
                        # Map category ID
                        cat_id = ann['category_id']
                        if class_mapping:
                            cat_id = class_mapping.get(cat_id, cat_id)
                        
                        f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        conversion_stats['total_annotations'] += 1
            
            conversion_stats['converted_files'] += 1
            
        except Exception as e:
            conversion_stats['errors'].append(f"Error converting {img_info['file_name']}: {e}")
    
    return conversion_stats

def _convert_csv_to_yolo(csv_file: Path, output_dir: Path, class_mapping: Dict) -> Dict:
    """Convert CSV format to YOLO format"""
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        conversion_stats = {'converted_files': 0, 'total_annotations': 0, 'errors': []}
        
        # Group by image_id
        for image_id, group in df.groupby('image_id'):
            try:
                label_file = output_dir / f'{image_id}.txt'
                
                with open(label_file, 'w') as f:
                    for _, row in group.iterrows():
                        # Assuming standard image size or extract from data
                        img_w = row.get('image_width', 1024)
                        img_h = row.get('image_height', 1024)
                        
                        # Convert from absolute to normalized coordinates
                        x, y, w, h = row['x'], row['y'], row['width'], row['height']
                        
                        x_center = (x + w / 2) / img_w
                        y_center = (y + h / 2) / img_h
                        width = w / img_w
                        height = h / img_h
                        
                        # Default class or map from data
                        cls = 0
                        if 'class' in row:
                            cls = row['class']
                        if class_mapping and cls in class_mapping:
                            cls = class_mapping[cls]
                        
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        conversion_stats['total_annotations'] += 1
                
                conversion_stats['converted_files'] += 1
                
            except Exception as e:
                conversion_stats['errors'].append(f"Error converting {image_id}: {e}")
        
        return conversion_stats
        
    except ImportError:
        raise ImportError("pandas is required for CSV conversion")

def create_data_loaders(data_dir: str, batch_size: int = 16, num_workers: int = 4, 
                       image_size: int = 640, dataset_type: str = 'PGP',
                       augmentation_strategy: str = 'medium', **kwargs) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation with enhanced options"""
    
    # Create augmentation pipeline
    aug_pipeline = AugmentationPipeline(strategy=augmentation_strategy)
    transforms = aug_pipeline.create_pipeline(image_size=image_size)
    
    # Get dataset class
    dataset_classes = {
        'PGP': PGPDataset,
        'MelonFlower': MelonFlowerDataset,
        'GlobalWheat': GlobalWheatDataset
    }
    
    if dataset_type not in dataset_classes:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    dataset_class = dataset_classes[dataset_type]
    
    # Create datasets
    train_dataset = dataset_class(
        data_dir=data_dir,
        split='train',
        transform=transforms['train'],
        image_size=image_size,
        **kwargs
    )
    
    val_dataset = dataset_class(
        data_dir=data_dir,
        split='val', 
        transform=transforms['val'],
        image_size=image_size,
        **kwargs
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )
    
    logger.info(f"Created data loaders: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    return train_loader, val_loader

# Preprocessing functions for specific datasets
def preprocess_pgp_data(data_dir: str, output_dir: str = None, **kwargs) -> Dict:
    """Preprocess PGP dataset"""
    processor = DataPreprocessor(data_dir, output_dir)
    
    # Analyze dataset
    analysis = processor.analyze_dataset()
    
    # Clean dataset
    cleaning_stats = processor.clean_dataset()
    
    # Setup splits
    split_stats = setup_dataset_splits(data_dir, **kwargs)
    
    return {
        'analysis': analysis,
        'cleaning': cleaning_stats,
        'splits': split_stats
    }

def preprocess_melon_flower_data(data_dir: str, output_dir: str = None, **kwargs) -> Dict:
    """Preprocess MelonFlower dataset"""
    return preprocess_pgp_data(data_dir, output_dir, **kwargs)

def preprocess_global_wheat_data(data_dir: str, output_dir: str = None, **kwargs) -> Dict:
    """Preprocess Global Wheat dataset"""
    processor = DataPreprocessor(data_dir, output_dir)
    
    # Analyze dataset
    analysis = processor.analyze_dataset()
    
    # Convert CSV annotations to YOLO format if needed
    csv_files = list(Path(data_dir).glob('*.csv'))
    conversion_stats = {}
    
    for csv_file in csv_files:
        stats = convert_annotations_to_yolo(
            str(csv_file), 
            str(Path(data_dir) / 'labels'),
            class_mapping={1: 0}  # Map wheat class to 0
        )
        conversion_stats[csv_file.name] = stats
    
    # Setup splits
    split_stats = setup_dataset_splits(data_dir, **kwargs)
    
    return {
        'analysis': analysis,
        'conversion': conversion_stats,
        'splits': split_stats
    }

if __name__ == "__main__":
    # Test preprocessing functionality
    print("Testing data preprocessing...")
    
    # Test basic preprocessing
    try:
        test_dir = "data/test"
        Path(test_dir).mkdir(parents=True, exist_ok=True)
        
        processor = DataPreprocessor(test_dir)
        print("✅ DataPreprocessor created successfully")
        
        # Test augmentation pipeline
        aug_pipeline = AugmentationPipeline('medium')
        transforms = aug_pipeline.create_pipeline()
        print("✅ Augmentation pipeline created successfully")
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
    
    print("Preprocessing tests completed!")