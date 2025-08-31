# src/data/transforms.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random
from torchvision import transforms
from typing import Tuple, List, Optional, Union
import warnings

class Compose:
    """Compose multiple transforms"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            image, boxes = transform(image, boxes)
        return image, boxes
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor:
    """Convert numpy array to tensor"""
    
    def __call__(self, image: np.ndarray, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(image, np.ndarray):
            # Handle different input formats
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            # Convert HWC to CHW format
            if len(image.shape) == 3:
                image = torch.from_numpy(image).permute(2, 0, 1)
            else:
                image = torch.from_numpy(image).unsqueeze(0)
        
        return image, boxes

class RandomRotation:
    """Random rotation augmentation with proper bounding box transformation"""
    
    def __init__(self, degrees: float = 10, probability: float = 0.5):
        self.degrees = degrees
        self.probability = probability
    
    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.probability:
            angle = random.uniform(-self.degrees, self.degrees)
            image, boxes = self._rotate_image_and_boxes(image, boxes, angle)
        
        return image, boxes
    
    def _rotate_image_and_boxes(self, image: torch.Tensor, boxes: torch.Tensor, angle: float):
        """Rotate image and transform bounding boxes"""
        # Convert tensor to numpy for rotation
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        h, w = image_np.shape[:2]
        
        # Rotation matrix
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation to image
        if len(image_np.shape) == 3:
            rotated_image = cv2.warpAffine(image_np, rotation_matrix, (w, h), 
                                         borderMode=cv2.BORDER_REFLECT)
        else:
            rotated_image = cv2.warpAffine(image_np, rotation_matrix, (w, h), 
                                         borderMode=cv2.BORDER_REFLECT)
            rotated_image = np.expand_dims(rotated_image, axis=-1)
        
        # Convert back to tensor
        image = torch.from_numpy(rotated_image).permute(2, 0, 1).float() / 255.0
        
        # Transform bounding boxes
        if len(boxes) > 0:
            boxes = self._rotate_boxes(boxes, rotation_matrix, w, h)
        
        return image, boxes
    
    def _rotate_boxes(self, boxes: torch.Tensor, rotation_matrix: np.ndarray, w: int, h: int):
        """Rotate bounding boxes using rotation matrix"""
        if len(boxes) == 0:
            return boxes
        
        # Convert normalized coordinates to absolute
        boxes_abs = boxes.clone()
        boxes_abs[:, 1] *= w  # x_center
        boxes_abs[:, 2] *= h  # y_center
        boxes_abs[:, 3] *= w  # width
        boxes_abs[:, 4] *= h  # height
        
        # Convert to corner format
        x1 = boxes_abs[:, 1] - boxes_abs[:, 3] / 2
        y1 = boxes_abs[:, 2] - boxes_abs[:, 4] / 2
        x2 = boxes_abs[:, 1] + boxes_abs[:, 3] / 2
        y2 = boxes_abs[:, 2] + boxes_abs[:, 4] / 2
        
        # Create corner points matrix
        corners = torch.stack([
            torch.stack([x1, y1, torch.ones_like(x1)], dim=1),
            torch.stack([x2, y1, torch.ones_like(x2)], dim=1),
            torch.stack([x1, y2, torch.ones_like(x1)], dim=1),
            torch.stack([x2, y2, torch.ones_like(x2)], dim=1)
        ], dim=1)  # [N, 4, 3]
        
        # Apply rotation
        rot_tensor = torch.from_numpy(rotation_matrix).float()
        rotated_corners = torch.matmul(corners, rot_tensor.t())  # [N, 4, 2]
        
        # Get new bounding boxes from rotated corners
        x_coords = rotated_corners[:, :, 0]
        y_coords = rotated_corners[:, :, 1]
        
        new_x1 = torch.min(x_coords, dim=1)[0]
        new_y1 = torch.min(y_coords, dim=1)[0]
        new_x2 = torch.max(x_coords, dim=1)[0]
        new_y2 = torch.max(y_coords, dim=1)[0]
        
        # Convert back to center format and normalize
        new_x_center = (new_x1 + new_x2) / 2 / w
        new_y_center = (new_y1 + new_y2) / 2 / h
        new_width = (new_x2 - new_x1) / w
        new_height = (new_y2 - new_y1) / h
        
        # Clamp to image bounds and filter invalid boxes
        new_x_center = torch.clamp(new_x_center, 0, 1)
        new_y_center = torch.clamp(new_y_center, 0, 1)
        new_width = torch.clamp(new_width, 0, 1)
        new_height = torch.clamp(new_height, 0, 1)
        
        # Filter out boxes that are too small
        valid_mask = (new_width > 0.01) & (new_height > 0.01)
        
        # Update boxes
        valid_boxes = boxes[valid_mask].clone()
        if len(valid_boxes) > 0:
            valid_boxes[:, 1] = new_x_center[valid_mask]
            valid_boxes[:, 2] = new_y_center[valid_mask]
            valid_boxes[:, 3] = new_width[valid_mask]
            valid_boxes[:, 4] = new_height[valid_mask]
        
        return valid_boxes

class RandomShear:
    """Random shear augmentation with proper bounding box transformation"""
    
    def __init__(self, shear_x: float = 10, shear_y: float = 10, probability: float = 0.5):
        self.shear_x = shear_x
        self.shear_y = shear_y
        self.probability = probability
    
    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.probability:
            shear_x = random.uniform(-self.shear_x, self.shear_x)
            shear_y = random.uniform(-self.shear_y, self.shear_y)
            image, boxes = self._shear_image_and_boxes(image, boxes, shear_x, shear_y)
        
        return image, boxes
    
    def _shear_image_and_boxes(self, image: torch.Tensor, boxes: torch.Tensor, 
                              shear_x: float, shear_y: float):
        """Apply shear transformation to image and boxes"""
        # Convert to numpy for OpenCV operations
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        h, w = image_np.shape[:2]
        
        # Shear matrix
        shear_matrix = np.array([
            [1, np.tan(np.radians(shear_x)), 0],
            [np.tan(np.radians(shear_y)), 1, 0]
        ], dtype=np.float32)
        
        # Apply shear to image
        if len(image_np.shape) == 3:
            sheared_image = cv2.warpAffine(image_np, shear_matrix, (w, h),
                                         borderMode=cv2.BORDER_REFLECT)
        else:
            sheared_image = cv2.warpAffine(image_np, shear_matrix, (w, h),
                                         borderMode=cv2.BORDER_REFLECT)
            sheared_image = np.expand_dims(sheared_image, axis=-1)
        
        # Convert back to tensor
        image = torch.from_numpy(sheared_image).permute(2, 0, 1).float() / 255.0
        
        # Transform bounding boxes
        if len(boxes) > 0:
            boxes = self._shear_boxes(boxes, shear_matrix, w, h)
        
        return image, boxes
    
    def _shear_boxes(self, boxes: torch.Tensor, shear_matrix: np.ndarray, w: int, h: int):
        """Apply shear transformation to bounding boxes"""
        if len(boxes) == 0:
            return boxes
        
        # Similar to rotation, but with shear matrix
        boxes_abs = boxes.clone()
        boxes_abs[:, 1] *= w  # x_center
        boxes_abs[:, 2] *= h  # y_center
        boxes_abs[:, 3] *= w  # width
        boxes_abs[:, 4] *= h  # height
        
        # Convert to corner format
        x1 = boxes_abs[:, 1] - boxes_abs[:, 3] / 2
        y1 = boxes_abs[:, 2] - boxes_abs[:, 4] / 2
        x2 = boxes_abs[:, 1] + boxes_abs[:, 3] / 2
        y2 = boxes_abs[:, 2] + boxes_abs[:, 4] / 2
        
        # Create corner points
        corners = torch.stack([
            torch.stack([x1, y1, torch.ones_like(x1)], dim=1),
            torch.stack([x2, y1, torch.ones_like(x2)], dim=1),
            torch.stack([x1, y2, torch.ones_like(x1)], dim=1),
            torch.stack([x2, y2, torch.ones_like(x2)], dim=1)
        ], dim=1)  # [N, 4, 3]
        
        # Apply shear
        shear_tensor = torch.from_numpy(shear_matrix).float()
        sheared_corners = torch.matmul(corners, shear_tensor.t())  # [N, 4, 2]
        
        # Get new bounding boxes
        x_coords = sheared_corners[:, :, 0]
        y_coords = sheared_corners[:, :, 1]
        
        new_x1 = torch.min(x_coords, dim=1)[0]
        new_y1 = torch.min(y_coords, dim=1)[0]
        new_x2 = torch.max(x_coords, dim=1)[0]
        new_y2 = torch.max(y_coords, dim=1)[0]
        
        # Convert back to center format and normalize
        new_x_center = torch.clamp((new_x1 + new_x2) / 2 / w, 0, 1)
        new_y_center = torch.clamp((new_y1 + new_y2) / 2 / h, 0, 1)
        new_width = torch.clamp((new_x2 - new_x1) / w, 0, 1)
        new_height = torch.clamp((new_y2 - new_y1) / h, 0, 1)
        
        # Filter valid boxes
        valid_mask = (new_width > 0.01) & (new_height > 0.01)
        valid_boxes = boxes[valid_mask].clone()
        
        if len(valid_boxes) > 0:
            valid_boxes[:, 1] = new_x_center[valid_mask]
            valid_boxes[:, 2] = new_y_center[valid_mask]
            valid_boxes[:, 3] = new_width[valid_mask]
            valid_boxes[:, 4] = new_height[valid_mask]
        
        return valid_boxes

class RandomCrop:
    """Random crop augmentation with bounding box adjustment"""
    
    def __init__(self, crop_factor: float = 0.15, probability: float = 0.5):
        self.crop_factor = crop_factor
        self.probability = probability
    
    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.probability:
            image, boxes = self._random_crop(image, boxes)
        
        return image, boxes
    
    def _random_crop(self, image: torch.Tensor, boxes: torch.Tensor):
        """Apply random crop to image and adjust boxes"""
        c, h, w = image.shape
        
        # Calculate crop parameters
        crop_factor = random.uniform(0.05, self.crop_factor)
        crop_h = int(h * (1 - crop_factor))
        crop_w = int(w * (1 - crop_factor))
        
        # Random crop position
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        
        # Crop image
        cropped_image = image[:, top:top+crop_h, left:left+crop_w]
        
        # Resize back to original size
        image = F.interpolate(
            cropped_image.unsqueeze(0), 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Adjust bounding boxes
        if len(boxes) > 0:
            boxes = self._crop_boxes(boxes, left, top, crop_w, crop_h, w, h)
        
        return image, boxes
    
    def _crop_boxes(self, boxes: torch.Tensor, left: int, top: int, 
                   crop_w: int, crop_h: int, orig_w: int, orig_h: int):
        """Adjust bounding boxes for crop operation"""
        if len(boxes) == 0:
            return boxes
        
        # Convert to absolute coordinates in original image
        abs_boxes = boxes.clone()
        abs_boxes[:, 1] *= orig_w  # x_center
        abs_boxes[:, 2] *= orig_h  # y_center
        abs_boxes[:, 3] *= orig_w  # width
        abs_boxes[:, 4] *= orig_h  # height
        
        # Adjust for crop offset
        abs_boxes[:, 1] -= left
        abs_boxes[:, 2] -= top
        
        # Scale for resize back to original size
        scale_x = orig_w / crop_w
        scale_y = orig_h / crop_h
        
        abs_boxes[:, 1] *= scale_x
        abs_boxes[:, 2] *= scale_y
        abs_boxes[:, 3] *= scale_x
        abs_boxes[:, 4] *= scale_y
        
        # Convert back to normalized coordinates
        boxes[:, 1] = abs_boxes[:, 1] / orig_w
        boxes[:, 2] = abs_boxes[:, 2] / orig_h
        boxes[:, 3] = abs_boxes[:, 3] / orig_w
        boxes[:, 4] = abs_boxes[:, 4] / orig_h
        
        # Filter boxes that are still within bounds and have reasonable size
        x_center = boxes[:, 1]
        y_center = boxes[:, 2]
        width = boxes[:, 3]
        height = boxes[:, 4]
        
        valid_mask = (
            (x_center >= 0) & (x_center <= 1) &
            (y_center >= 0) & (y_center <= 1) &
            (width > 0.01) & (height > 0.01)  # Minimum size threshold
        )
        
        return boxes[valid_mask]

class ColorJitter:
    """Enhanced color jittering for agricultural data"""
    
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, 
                 saturation: float = 0.2, hue: float = 0.1, probability: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.probability = probability
        
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.probability:
            # Handle multi-spectral images (4 channels)
            if image.shape[0] == 4:
                # Apply color jitter only to RGB channels
                rgb_image = image[:3]
                nir_channel = image[3:4]
                
                # Convert to PIL for torchvision transform
                rgb_pil = transforms.ToPILImage()(rgb_image)
                rgb_jittered = self.color_jitter(rgb_pil)
                rgb_tensor = transforms.ToTensor()(rgb_jittered)
                
                # Combine back with NIR channel
                image = torch.cat([rgb_tensor, nir_channel], dim=0)
            else:
                # Standard RGB processing
                image_pil = transforms.ToPILImage()(image)
                image = transforms.ToTensor()(self.color_jitter(image_pil))
        
        return image, boxes

class Normalize:
    """Enhanced normalization with support for multi-spectral data"""
    
    def __init__(self, mean: List[float] = None, std: List[float] = None):
        # Default values for RGB + NIR
        if mean is None:
            mean = [0.485, 0.456, 0.406, 0.5]  # RGB + NIR
        if std is None:
            std = [0.229, 0.224, 0.225, 0.25]  # RGB + NIR
        
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Adjust normalization based on number of channels
        num_channels = image.shape[0]
        mean = self.mean[:num_channels].view(num_channels, 1, 1)
        std = self.std[:num_channels].view(num_channels, 1, 1)
        
        image = (image - mean) / std
        return image, boxes

class SpectralTransform:
    """Transform for multi-spectral agricultural data"""
    
    def __init__(self, enhance_vegetation: bool = True, probability: float = 0.5):
        self.enhance_vegetation = enhance_vegetation
        self.probability = probability
    
    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.probability and image.shape[0] >= 4:
            if self.enhance_vegetation:
                # Enhance vegetation using NDVI-like calculation
                red = image[0]  # Red channel
                nir = image[3]  # NIR channel
                
                # Calculate NDVI-like index
                ndvi = (nir - red) / (nir + red + 1e-8)
                
                # Enhance vegetation areas
                vegetation_mask = ndvi > 0.2
                if vegetation_mask.any():
                    # Slightly enhance NIR in vegetation areas
                    image[3][vegetation_mask] = torch.clamp(
                        image[3][vegetation_mask] * 1.1, 0, 1
                    )
        
        return image, boxes

class PseudoRGBConversion:
    """Convert multi-spectral to pseudo-RGB for visualization"""
    
    def __init__(self, band_mapping: dict = None):
        # Default mapping: R=NIR, G=Red, B=Green
        self.band_mapping = band_mapping or {'R': 3, 'G': 0, 'B': 1}
    
    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.shape[0] >= 4:
            # Create pseudo-RGB
            pseudo_rgb = torch.stack([
                image[self.band_mapping['R']],
                image[self.band_mapping['G']],
                image[self.band_mapping['B']]
            ], dim=0)
            
            return pseudo_rgb, boxes
        
        return image, boxes

class GeometricTransform:
    """Combined geometric transformations"""
    
    def __init__(self, rotation_prob: float = 0.3, shear_prob: float = 0.3, 
                 crop_prob: float = 0.3):
        self.transforms = []
        
        if rotation_prob > 0:
            self.transforms.append(RandomRotation(degrees=10, probability=rotation_prob))
        if shear_prob > 0:
            self.transforms.append(RandomShear(shear_x=5, shear_y=5, probability=shear_prob))
        if crop_prob > 0:
            self.transforms.append(RandomCrop(crop_factor=0.1, probability=crop_prob))
    
    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply one random geometric transform
        if self.transforms:
            transform = random.choice(self.transforms)
            image, boxes = transform(image, boxes)
        
        return image, boxes

class TestAugmentations:
    """Test-time augmentations for model evaluation"""
    
    def __init__(self, rotation_deg: float = 10, shear_deg: float = 10, 
                 crop_factor: float = 0.15):
        self.transforms = {
            'no_aug': lambda x, b: (x, b),
            'rotation': RandomRotation(rotation_deg, probability=1.0),
            'shear': RandomShear(shear_deg, shear_deg, probability=1.0),
            'crop': RandomCrop(crop_factor, probability=1.0),
            'rotation_shear': Compose([
                RandomRotation(rotation_deg, probability=1.0),
                RandomShear(shear_deg, shear_deg, probability=1.0)
            ]),
            'all': Compose([
                RandomRotation(rotation_deg, probability=1.0),
                RandomShear(shear_deg, shear_deg, probability=1.0),
                RandomCrop(crop_factor, probability=1.0)
            ])
        }
    
    def get_transform(self, aug_type: str):
        """Get specific augmentation transform"""
        return self.transforms.get(aug_type, self.transforms['no_aug'])
    
    def get_all_transforms(self):
        """Get all available transforms"""
        return self.transforms

# Factory functions for creating transform pipelines
def create_train_transforms(image_size: int = 640, rotation_degrees: float = 10,
                          color_jitter_prob: float = 0.5, geometric_prob: float = 0.5):
    """Create training transform pipeline"""
    return Compose([
        RandomRotation(degrees=rotation_degrees, probability=geometric_prob),
        RandomShear(shear_x=rotation_degrees//2, shear_y=rotation_degrees//2, 
                   probability=geometric_prob),
        RandomCrop(crop_factor=0.15, probability=geometric_prob),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1,
                   probability=color_jitter_prob),
        SpectralTransform(enhance_vegetation=True, probability=0.3),
        Normalize()
    ])

def create_val_transforms(image_size: int = 640):
    """Create validation transform pipeline (no augmentation)"""
    return Compose([
        Normalize()
    ])

def create_test_transforms(image_size: int = 640):
    """Create test transform pipeline"""
    return Compose([
        Normalize()
    ])

def create_inference_transforms(image_size: int = 640, multi_spectral: bool = True):
    """Create inference transform pipeline"""
    transforms_list = []
    
    if multi_spectral:
        transforms_list.append(SpectralTransform(enhance_vegetation=False, probability=0))
    
    transforms_list.append(Normalize())
    
    return Compose(transforms_list)

# Utility functions
def validate_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Validate and clean bounding boxes"""
    if len(boxes) == 0:
        return boxes
    
    # Ensure coordinates are within [0, 1] range
    boxes[:, 1:] = torch.clamp(boxes[:, 1:], 0, 1)
    
    # Filter out boxes that are too small
    min_size = 0.01
    valid_mask = (boxes[:, 3] > min_size) & (boxes[:, 4] > min_size)
    
    return boxes[valid_mask]

def visualize_transforms(dataset, transform, num_samples: int = 4):
    """Visualize the effect of transforms on dataset samples"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
        
        for i in range(num_samples):
            # Get original sample
            image, boxes, path = dataset[i]
            
            # Apply transform
            if transform:
                image_transformed, boxes_transformed = transform(image.clone(), boxes.clone())
            else:
                image_transformed, boxes_transformed = image, boxes
            
            # Convert to display format
            orig_img = image.permute(1, 2, 0).numpy()
            trans_img = image_transformed.permute(1, 2, 0).numpy()
            
            # Handle multi-spectral images (take first 3 channels)
            if orig_img.shape[2] > 3:
                orig_img = orig_img[:, :, :3]
                trans_img = trans_img[:, :, :3]
            
            # Normalize for display
            orig_img = np.clip(orig_img, 0, 1)
            trans_img = np.clip(trans_img, 0, 1)
            
            # Plot original
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Plot transformed
            axes[1, i].imshow(trans_img)
            axes[1, i].set_title(f'Transformed {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available for visualization")
    except Exception as e:
        print(f"Visualization error: {e}")

if __name__ == "__main__":
    # Test transforms
    print("Testing data transforms...")
    
    # Create dummy data
    dummy_image = torch.rand(3, 512, 512)  # RGB image
    dummy_boxes = torch.tensor([
        [0, 0.5, 0.5, 0.2, 0.2],  # class, x_center, y_center, width, height
        [1, 0.3, 0.3, 0.1, 0.1]
    ])
    
    # Test individual transforms
    transforms_to_test = [
        ('RandomRotation', RandomRotation(10)),
        ('RandomShear', RandomShear(5, 5)),
        ('RandomCrop', RandomCrop(0.1)),
        ('ColorJitter', ColorJitter(0.2, 0.2, 0.2, 0.1)),
        ('Normalize', Normalize())
    ]
    
    for name, transform in transforms_to_test:
        try:
            transformed_image, transformed_boxes = transform(dummy_image.clone(), dummy_boxes.clone())
            print(f"✅ {name}: Image {dummy_image.shape} -> {transformed_image.shape}, "
                  f"Boxes {dummy_boxes.shape} -> {transformed_boxes.shape}")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    # Test transform pipelines
    print("\nTesting transform pipelines...")
    
    pipelines = [
        ('Training', create_train_transforms()),
        ('Validation', create_val_transforms()),
        ('Test', create_test_transforms())
    ]
    
    for name, pipeline in pipelines:
        try:
            transformed_image, transformed_boxes = pipeline(dummy_image.clone(), dummy_boxes.clone())
            print(f"✅ {name} pipeline: Image {dummy_image.shape} -> {transformed_image.shape}, "
                  f"Boxes {dummy_boxes.shape} -> {transformed_boxes.shape}")
        except Exception as e:
            print(f"❌ {name} pipeline failed: {e}")
    
    # Test multi-spectral support
    print("\nTesting multi-spectral transforms...")
    
    dummy_multispectral = torch.rand(4, 512, 512)  # RGBN image
    
    try:
        # Test spectral transform
        spectral_transform = SpectralTransform()
        transformed_ms, _ = spectral_transform(dummy_multispectral.clone(), dummy_boxes.clone())
        print(f"✅ SpectralTransform: {dummy_multispectral.shape} -> {transformed_ms.shape}")
        
        # Test pseudo-RGB conversion
        pseudo_rgb_transform = PseudoRGBConversion()
        pseudo_rgb, _ = pseudo_rgb_transform(dummy_multispectral.clone(), dummy_boxes.clone())
        print(f"✅ PseudoRGBConversion: {dummy_multispectral.shape} -> {pseudo_rgb.shape}")
        
        # Test normalization with 4 channels
        normalize_4ch = Normalize()
        normalized_ms, _ = normalize_4ch(dummy_multispectral.clone(), dummy_boxes.clone())
        print(f"✅ Multi-spectral Normalize: {dummy_multispectral.shape} -> {normalized_ms.shape}")
        
    except Exception as e:
        print(f"❌ Multi-spectral transform failed: {e}")
    
    print("Transform tests completed!")