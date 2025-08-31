# src/data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from PIL import Image
import random
import warnings
from typing import Dict, List, Tuple, Optional, Union
import glob

class BaseAgriculturalDataset(Dataset):
    """Base class for agricultural datasets with common functionality"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None, 
                 image_size: int = 640, class_names: List[str] = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.class_names = class_names or ['cotton', 'rice', 'corn']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        # Initialize annotations
        self.annotations = []
        self._load_dataset()
        
        if len(self.annotations) == 0:
            warnings.warn(f"No valid annotations found for {split} split in {data_dir}")
    
    def _load_dataset(self):
        """Override in subclasses to implement specific loading logic"""
        raise NotImplementedError("Subclasses must implement _load_dataset")
    
    def _validate_annotation(self, annotation: Dict) -> bool:
        """Validate annotation format"""
        required_keys = ['image', 'boxes']
        return all(key in annotation for key in required_keys)
    
    def _load_image_safe(self, image_path: Union[str, Path]) -> np.ndarray:
        """Safely load image with fallback"""
        try:
            if isinstance(image_path, str):
                image_path = Path(image_path)
            
            # Try to read image
            image = cv2.imread(str(image_path))
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Try with PIL as fallback
            image = Image.open(image_path).convert('RGB')
            return np.array(image)
            
        except Exception as e:
            warnings.warn(f"Failed to load image {image_path}: {e}")
            # Return dummy image
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Load image
        image_path = self.data_dir / annotation['image']
        image = self._load_image_safe(image_path)
        
        # Resize image
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert to tensor and normalize to [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Load and process targets
        boxes = annotation.get('boxes', [])
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 5), dtype=torch.float32)
        
        # Apply transformations
        if self.transform:
            image, boxes_tensor = self.transform(image, boxes_tensor)
        
        return image, boxes_tensor, annotation['image']

class PGPDataset(BaseAgriculturalDataset):
    """Plant Growth and Phenotyping Dataset with multi-spectral support"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None, 
                 multi_spectral: bool = True, image_size: int = 640,
                 class_names: List[str] = None):
        self.multi_spectral = multi_spectral
        self.spectral_bands = {
            'green': '580nm',
            'red': '660nm', 
            'red_edge': '730nm',
            'nir': '820nm'
        }
        
        # Default class names for PGP dataset
        if class_names is None:
            class_names = ['cotton', 'rice', 'corn', 'soybean', 'wheat']
        
        super().__init__(data_dir, split, transform, image_size, class_names)
    
    def _load_dataset(self):
        """Load PGP dataset annotations"""
        self.annotations = self._load_annotations()
        print(f"Loaded {len(self.annotations)} samples for {self.split} split")
    
    def _load_annotations(self):
        """Load dataset annotations from various formats"""
        annotations = []
        
        # Try different annotation formats
        annotation_formats = [
            f'{self.split}_annotations.json',
            f'{self.split}_annotations.txt',
            f'{self.split}.json',
            'annotations.json',
            f'{self.split}.txt'
        ]
        
        for fmt in annotation_formats:
            annotation_file = self.data_dir / fmt
            if annotation_file.exists():
                if fmt.endswith('.json'):
                    annotations = self._load_json_annotations(annotation_file)
                else:
                    annotations = self._load_txt_annotations(annotation_file)
                break
        
        if not annotations:
            # If no annotation file found, try to build from directory structure
            annotations = self._build_annotations_from_directory()
        
        return [ann for ann in annotations if self._validate_annotation(ann)]
    
    def _load_json_annotations(self, annotation_file: Path):
        """Load annotations from JSON format (COCO-style)"""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        annotations = []
        
        if 'images' in data and 'annotations' in data:
            # COCO format
            image_dict = {img['id']: img for img in data['images']}
            
            # Group annotations by image
            image_annotations = {}
            for ann in data['annotations']:
                image_id = ann['image_id']
                if image_id not in image_annotations:
                    image_annotations[image_id] = []
                image_annotations[image_id].append(ann)
            
            for image_id, image_info in image_dict.items():
                boxes = []
                if image_id in image_annotations:
                    for ann in image_annotations[image_id]:
                        # Convert bbox from [x, y, w, h] to normalized [x_center, y_center, w, h]
                        bbox = ann['bbox']
                        img_w, img_h = image_info['width'], image_info['height']
                        
                        x_center = (bbox[0] + bbox[2] / 2) / img_w
                        y_center = (bbox[1] + bbox[3] / 2) / img_h
                        width = bbox[2] / img_w
                        height = bbox[3] / img_h
                        
                        boxes.append([ann['category_id'], x_center, y_center, width, height])
                
                annotations.append({
                    'image': image_info['file_name'],
                    'boxes': boxes
                })
        else:
            # Simple format
            for item in data:
                if self._validate_annotation(item):
                    annotations.append(item)
        
        return annotations
    
    def _load_txt_annotations(self, annotation_file: Path):
        """Load annotations from text format (YOLO-style)"""
        annotations = []
        
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # image_path class x y w h
                    image_path = parts[0]
                    boxes = []
                    
                    # Parse multiple boxes for same image
                    i = 1
                    while i + 4 < len(parts):
                        try:
                            cls, x, y, w, h = map(float, parts[i:i+5])
                            boxes.append([int(cls), x, y, w, h])
                            i += 5
                        except ValueError:
                            break
                    
                    annotations.append({
                        'image': image_path,
                        'boxes': boxes
                    })
        
        return annotations
    
    def _build_annotations_from_directory(self):
        """Build annotations from directory structure when no annotation file exists"""
        annotations = []
        
        # Look for images in common formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        
        # Try different directory structures
        possible_dirs = [
            self.data_dir / 'images' / self.split,
            self.data_dir / self.split / 'images',
            self.data_dir / self.split,
            self.data_dir
        ]
        
        image_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                image_dir = dir_path
                break
        
        if image_dir is None:
            return []
        
        for ext in image_extensions:
            for img_path in image_dir.glob(f'*{ext}'):
                # Try to find corresponding label file
                label_paths = [
                    img_path.with_suffix('.txt'),  # Same directory
                    img_path.parent.parent / 'labels' / f'{img_path.stem}.txt',  # labels subdirectory
                    self.data_dir / 'labels' / self.split / f'{img_path.stem}.txt'  # split labels
                ]
                
                boxes = []
                for label_path in label_paths:
                    if label_path.exists():
                        boxes = self._load_yolo_label(label_path)
                        break
                
                annotations.append({
                    'image': str(img_path.relative_to(self.data_dir)),
                    'boxes': boxes
                })
        
        return annotations
    
    def _load_yolo_label(self, label_path: Path):
        """Load YOLO format label file"""
        boxes = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        boxes.append([int(cls), x, y, w, h])
        except Exception as e:
            warnings.warn(f"Failed to load label {label_path}: {e}")
        
        return boxes
    
    def _load_multispectral_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load and process multi-spectral image"""
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        full_path = self.data_dir / image_path
        
        if self.multi_spectral:
            # Try to load 4 spectral bands
            base_path = str(full_path).replace(full_path.suffix, '')
            
            bands = {}
            band_files = {
                'green': f'{base_path}_{self.spectral_bands["green"]}.tiff',
                'red': f'{base_path}_{self.spectral_bands["red"]}.tiff', 
                'red_edge': f'{base_path}_{self.spectral_bands["red_edge"]}.tiff',
                'nir': f'{base_path}_{self.spectral_bands["nir"]}.tiff'
            }
            
            # Try to load spectral bands
            for band_name, band_file in band_files.items():
                if os.path.exists(band_file):
                    try:
                        band_img = cv2.imread(band_file, cv2.IMREAD_ANYDEPTH)
                        if band_img is not None:
                            bands[band_name] = band_img
                    except Exception as e:
                        warnings.warn(f"Failed to load spectral band {band_file}: {e}")
            
            if len(bands) >= 3:
                # Create pseudo-RGB using available bands
                if 'red' in bands and 'red_edge' in bands and 'green' in bands:
                    pseudo_rgb = np.stack([bands['red'], bands['red_edge'], bands['green']], axis=-1)
                elif 'red' in bands and 'green' in bands and 'nir' in bands:
                    pseudo_rgb = np.stack([bands['red'], bands['green'], bands['nir']], axis=-1)
                else:
                    # Use first 3 available bands
                    available_bands = list(bands.values())[:3]
                    pseudo_rgb = np.stack(available_bands, axis=-1)
                
                # Normalize each band to 0-255
                for i in range(3):
                    band = pseudo_rgb[:, :, i].astype(np.float32)
                    if band.max() > band.min():
                        band = (band - band.min()) / (band.max() - band.min())
                        pseudo_rgb[:, :, i] = (band * 255).astype(np.uint8)
                    else:
                        pseudo_rgb[:, :, i] = 0
                
                return pseudo_rgb
        
        # Fallback to regular RGB image loading
        return self._load_image_safe(full_path)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Load image (with multi-spectral support)
        if self.multi_spectral:
            image = self._load_multispectral_image(annotation['image'])
        else:
            image = self._load_image_safe(self.data_dir / annotation['image'])
        
        # Resize image
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert to tensor and normalize to [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Add 4th channel for multi-spectral if needed
        if self.multi_spectral and image.shape[0] == 3:
            # Add NIR channel as weighted combination of RGB
            nir_channel = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            image = torch.cat([image, nir_channel.unsqueeze(0)], dim=0)
        
        # Load and process targets
        boxes = annotation.get('boxes', [])
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 5), dtype=torch.float32)
        
        # Apply transformations
        if self.transform:
            image, boxes_tensor = self.transform(image, boxes_tensor)
        
        return image, boxes_tensor, annotation['image']

class MelonFlowerDataset(BaseAgriculturalDataset):
    """MelonFlower dataset for small object detection"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None, 
                 image_size: int = 640, class_names: List[str] = None):
        # Default class names for MelonFlower dataset
        if class_names is None:
            class_names = ['melon', 'flower']
        
        super().__init__(data_dir, split, transform, image_size, class_names)
    
    def _load_dataset(self):
        """Load MelonFlower dataset annotations"""
        self.annotations = self._load_roboflow_annotations()
        print(f"Loaded {len(self.annotations)} samples for {self.split} split")
    
    def _load_roboflow_annotations(self):
        """Load Roboflow format annotations"""
        annotations = []
        
        # Roboflow structure: train/images, train/labels, valid/images, valid/labels
        possible_structures = [
            (self.data_dir / self.split / 'images', self.data_dir / self.split / 'labels'),
            (self.data_dir / 'images' / self.split, self.data_dir / 'labels' / self.split),
            (self.data_dir / self.split, self.data_dir / self.split.replace('images', 'labels'))
        ]
        
        image_dir, label_dir = None, None
        for img_dir, lbl_dir in possible_structures:
            if img_dir.exists():
                image_dir, label_dir = img_dir, lbl_dir
                break
        
        if image_dir is None:
            warnings.warn(f"Could not find image directory for {self.split} split")
            return []
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(ext))
        
        for img_file in image_files:
            label_file = label_dir / f'{img_file.stem}.txt'
            
            boxes = []
            if label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls, x, y, w, h = map(float, parts)
                                boxes.append([int(cls), x, y, w, h])
                except Exception as e:
                    warnings.warn(f"Failed to load label {label_file}: {e}")
            
            annotations.append({
                'image': str(img_file.relative_to(self.data_dir)),
                'boxes': boxes
            })
        
        return annotations

class GlobalWheatDataset(BaseAgriculturalDataset):
    """Global Wheat Detection Dataset"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None, 
                 image_size: int = 640, class_names: List[str] = None):
        # Default class names for Global Wheat dataset
        if class_names is None:
            class_names = ['wheat_head']
        
        super().__init__(data_dir, split, transform, image_size, class_names)
    
    def _load_dataset(self):
        """Load Global Wheat dataset annotations"""
        self.annotations = self._load_wheat_annotations()
        print(f"Loaded {len(self.annotations)} samples for {self.split} split")
    
    def _load_wheat_annotations(self):
        """Load Global Wheat dataset specific annotations"""
        annotations = []
        
        # Try to load from CSV format (original Global Wheat format)
        csv_files = [
            self.data_dir / f'{self.split}.csv',
            self.data_dir / 'train.csv',  # Fallback to train.csv
            self.data_dir / 'annotations.csv'
        ]
        
        csv_file = None
        for csv_path in csv_files:
            if csv_path.exists():
                csv_file = csv_path
                break
        
        if csv_file:
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                
                # Group by image_id
                for image_id, group in df.groupby('image_id'):
                    boxes = []
                    for _, row in group.iterrows():
                        # Convert from [x, y, w, h] to normalized [x_center, y_center, w, h]
                        # Note: Global Wheat uses absolute coordinates
                        x, y, w, h = row['x'], row['y'], row['width'], row['height']
                        
                        # Assuming standard image size of 1024x1024 for Global Wheat
                        img_w, img_h = 1024, 1024
                        
                        x_center = (x + w / 2) / img_w
                        y_center = (y + h / 2) / img_h
                        width = w / img_w
                        height = h / img_h
                        
                        boxes.append([0, x_center, y_center, width, height])  # Class 0 for wheat
                    
                    annotations.append({
                        'image': f'{image_id}.jpg',
                        'boxes': boxes
                    })
            except ImportError:
                warnings.warn("pandas not available for CSV loading, falling back to directory structure")
                annotations = self._build_annotations_from_directory()
            except Exception as e:
                warnings.warn(f"Failed to load CSV annotations: {e}")
                annotations = self._build_annotations_from_directory()
        else:
            # Fallback to directory structure
            annotations = self._build_annotations_from_directory()
        
        return annotations

class MultiSpectralDataset(PGPDataset):
    """Specialized dataset for multi-spectral agricultural imaging"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None,
                 image_size: int = 640, spectral_bands: Dict[str, str] = None,
                 class_names: List[str] = None):
        
        # Custom spectral band configuration
        if spectral_bands:
            self.spectral_bands = spectral_bands
        
        super().__init__(data_dir, split, transform, True, image_size, class_names)
    
    def set_spectral_bands(self, band_config: Dict[str, str]):
        """Update spectral band configuration"""
        self.spectral_bands = band_config

# Utility functions
def get_dataset(dataset_type: str, **kwargs) -> BaseAgriculturalDataset:
    """Factory function to get dataset by type"""
    dataset_map = {
        'PGP': PGPDataset,
        'MelonFlower': MelonFlowerDataset,
        'GlobalWheat': GlobalWheatDataset,
        'MultiSpectral': MultiSpectralDataset
    }
    
    if dataset_type not in dataset_map:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(dataset_map.keys())}")
    
    return dataset_map[dataset_type](**kwargs)

def validate_dataset_structure(data_dir: str, dataset_type: str = 'PGP') -> bool:
    """Validate dataset directory structure"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Dataset directory does not exist: {data_dir}")
    
    # Check for basic structure
    has_images = False
    has_annotations = False
    
    # Look for image directories
    image_paths = [
        data_path / 'train' / 'images',
        data_path / 'images' / 'train',
        data_path / 'train',
        data_path / 'images'
    ]
    
    for img_path in image_paths:
        if img_path.exists():
            image_files = list(img_path.glob('*.jpg')) + list(img_path.glob('*.png'))
            if len(image_files) > 0:
                has_images = True
                break
    
    # Look for annotation files
    annotation_paths = [
        data_path / 'train_annotations.json',
        data_path / 'annotations.json',
        data_path / 'train.csv',
        data_path / 'train' / 'labels',
        data_path / 'labels' / 'train'
    ]
    
    for ann_path in annotation_paths:
        if ann_path.exists():
            has_annotations = True
            break
    
    if not has_images:
        warnings.warn(f"No image files found in {data_dir}")
    
    if not has_annotations:
        warnings.warn(f"No annotation files found in {data_dir}")
    
    return has_images and has_annotations

# Collate function for variable-sized annotations
def collate_fn(batch):
    """Custom collate function for variable number of targets"""
    images, targets, paths = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Handle variable number of targets
    if all(len(t) == 0 for t in targets):
        # All empty targets
        targets_padded = torch.zeros((len(batch), 1, 5))
    else:
        max_targets = max(len(t) for t in targets if len(t) > 0)
        targets_padded = torch.zeros((len(batch), max_targets, 5))
        
        for i, target in enumerate(targets):
            if len(target) > 0:
                targets_padded[i, :len(target)] = target
    
    return images, targets_padded, list(paths)

# Quick dataset creation function
def create_data_loaders(data_dir: str, dataset_type: str = 'PGP', 
                       batch_size: int = 16, num_workers: int = 4,
                       image_size: int = 640, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    from .transforms import create_train_transforms, create_val_transforms
    
    # Create transforms
    train_transform = create_train_transforms(image_size=image_size)
    val_transform = create_val_transforms(image_size=image_size)
    
    # Create datasets
    train_dataset = get_dataset(
        dataset_type=dataset_type,
        data_dir=data_dir,
        split='train',
        transform=train_transform,
        image_size=image_size,
        **kwargs
    )
    
    val_dataset = get_dataset(
        dataset_type=dataset_type,
        data_dir=data_dir,
        split='val',
        transform=val_transform,
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
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset functionality...")
    
    # Test PGP dataset creation
    try:
        dataset = PGPDataset(
            data_dir='data/PGP',
            split='train',
            image_size=512,
            multi_spectral=True
        )
        print(f"✅ PGP Dataset: {len(dataset)} samples")
        
        # Test one sample
        if len(dataset) > 0:
            image, targets, path = dataset[0]
            print(f"✅ Sample shape: {image.shape}, targets: {targets.shape}")
    except Exception as e:
        print(f"❌ PGP Dataset test failed: {e}")
    
    # Test data loader creation
    try:
        train_loader, val_loader = create_data_loaders(
            data_dir='data/PGP',
            dataset_type='PGP',
            batch_size=4,
            num_workers=0  # For testing
        )
        print(f"✅ Data loaders created successfully")
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
    
    print("Dataset tests completed!")