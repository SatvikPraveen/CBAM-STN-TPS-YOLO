# src/inference/predict.py
"""
Enhanced inference and prediction utilities for CBAM-STN-TPS-YOLO
Comprehensive inference pipeline with ONNX export, batch processing, and visualization
"""

import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import onnx
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Internal imports
from ..models import create_model
from ..utils.visualization import Visualizer
from ..training.metrics import non_max_suppression, calculate_iou
from ..data.transforms import get_test_transforms

logger = logging.getLogger(__name__)

class ModelPredictor:
    """Enhanced model inference for agricultural object detection"""
    
    def __init__(self, model_path: str, model_type: str = 'CBAM-STN-TPS-YOLO', 
                 class_names: List[str] = None, device: str = None,
                 config: Optional[Dict] = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model to load
            class_names: List of class names for predictions
            device: Device to run inference on
            config: Optional model configuration
        """
        self.model_type = model_type
        self.class_names = class_names or ['Cotton', 'Rice', 'Corn']
        self.num_classes = len(self.class_names)
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_path = Path(model_path)
        
        # Load model configuration and weights
        self.config = self._load_config(config)
        self.model = self._load_model()
        
        # Initialize components
        self.visualizer = Visualizer(class_names=self.class_names)
        self.transforms = get_test_transforms(
            input_size=self.config.get('input_size', 640),
            spectral_bands=self.config.get('spectral_bands', 4)
        )
        
        # Inference settings
        self.conf_threshold = 0.5
        self.nms_threshold = 0.45
        self.input_size = self.config.get('input_size', 640)
        self.spectral_bands = self.config.get('spectral_bands', 4)
        
        logger.info(f"✅ Model loaded: {model_type} on {self.device}")
        logger.info(f"   Classes: {self.class_names}")
        logger.info(f"   Input size: {self.input_size}")
        logger.info(f"   Spectral bands: {self.spectral_bands}")
    
    def _load_config(self, config: Optional[Dict]) -> Dict:
        """Load model configuration"""
        if config:
            return config
        
        # Try to load config from checkpoint
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            if 'config' in checkpoint:
                return checkpoint['config']
        except Exception as e:
            logger.warning(f"Could not load config from checkpoint: {e}")
        
        # Default configuration
        return {
            'input_size': 640,
            'spectral_bands': 4,
            'num_classes': self.num_classes,
            'num_control_points': 20,
            'backbone_type': 'darknet53'
        }
    
    def _load_model(self) -> torch.nn.Module:
        """Load trained model from checkpoint"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create model
            model = create_model(
                model_type=self.model_type,
                num_classes=self.config['num_classes'],
                input_channels=self.config.get('spectral_bands', 4),
                num_control_points=self.config.get('num_control_points', 20),
                backbone_type=self.config.get('backbone_type', 'darknet53')
            )
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DataParallel wrapper
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()
            
            logger.info("✅ Model weights loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_image(self, image_path: Union[str, Path], 
                     conf_threshold: float = None,
                     nms_threshold: float = None,
                     save_result: bool = True,
                     output_dir: str = None,
                     return_features: bool = False) -> Dict:
        """
        Predict on a single image
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold
            save_result: Whether to save visualization
            output_dir: Directory to save results
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing predictions and metadata
        """
        conf_threshold = conf_threshold or self.conf_threshold
        nms_threshold = nms_threshold or self.nms_threshold
        
        image_path = Path(image_path)
        logger.info(f"Predicting on {image_path}")
        
        # Load and preprocess image
        image, original_image = self._load_and_preprocess_image(image_path)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            if return_features:
                outputs = self.model(image.unsqueeze(0), return_features=True)
                predictions = outputs['predictions']
                features = outputs.get('features', {})
            else:
                predictions = self.model(image.unsqueeze(0))
                features = {}
        
        inference_time = time.time() - start_time
        
        # Post-process outputs
        detections = self._post_process_outputs(
            predictions, conf_threshold, nms_threshold
        )
        
        # Prepare result
        result = {
            'image_path': str(image_path),
            'detections': detections,
            'num_detections': len(detections),
            'inference_time': inference_time,
            'model_type': self.model_type,
            'confidence_threshold': conf_threshold,
            'nms_threshold': nms_threshold
        }
        
        if return_features:
            result['features'] = features
        
        # Visualize and save results
        if save_result:
            output_dir = Path(output_dir) if output_dir else image_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result_path = output_dir / f"result_{image_path.stem}.jpg"
            
            # Create visualization
            self._save_prediction_visualization(
                original_image, detections, result_path, 
                inference_time, features if return_features else None
            )
            
            result['result_path'] = str(result_path)
            logger.info(f"Result saved: {result_path}")
        
        logger.info(f"Found {len(detections)} detections in {inference_time:.3f}s")
        return result
    
    def predict_batch(self, image_dir: Union[str, Path], 
                     output_dir: Union[str, Path],
                     conf_threshold: float = None,
                     nms_threshold: float = None,
                     max_images: int = None) -> List[Dict]:
        """
        Predict on a batch of images
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save results
            conf_threshold: Confidence threshold
            nms_threshold: NMS threshold
            max_images: Maximum number of images to process
            
        Returns:
            List of prediction results
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
            image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Found {len(image_files)} images in {image_dir}")
        
        results = []
        total_detections = 0
        total_time = 0
        
        for i, img_path in enumerate(image_files):
            try:
                logger.info(f"Processing {i+1}/{len(image_files)}: {img_path.name}")
                
                result = self.predict_image(
                    img_path, 
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold,
                    save_result=True,
                    output_dir=output_dir
                )
                
                results.append(result)
                total_detections += result['num_detections']
                total_time += result['inference_time']
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        # Save batch summary
        summary = {
            'total_images': len(results),
            'total_detections': total_detections,
            'average_detections_per_image': total_detections / len(results) if results else 0,
            'total_inference_time': total_time,
            'average_inference_time': total_time / len(results) if results else 0,
            'images_per_second': len(results) / total_time if total_time > 0 else 0
        }
        
        summary_path = output_dir / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Batch processing completed:")
        logger.info(f"   Processed: {len(results)} images")
        logger.info(f"   Total detections: {total_detections}")
        logger.info(f"   Average time: {summary['average_inference_time']:.3f}s per image")
        logger.info(f"   Summary saved: {summary_path}")
        
        return results
    
    def predict_video(self, video_path: Union[str, Path],
                     output_path: Union[str, Path],
                     conf_threshold: float = None,
                     nms_threshold: float = None,
                     skip_frames: int = 1) -> Dict:
        """
        Predict on video frames
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            conf_threshold: Confidence threshold
            nms_threshold: NMS threshold
            skip_frames: Process every nth frame
            
        Returns:
            Video processing summary
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        processed_frames = 0
        total_detections = 0
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames == 0:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Preprocess frame
                input_tensor = self._preprocess_frame(frame_rgb)
                
                # Inference
                with torch.no_grad():
                    predictions = self.model(input_tensor.unsqueeze(0))
                
                # Post-process
                detections = self._post_process_outputs(
                    predictions, conf_threshold or self.conf_threshold,
                    nms_threshold or self.nms_threshold
                )
                
                # Draw detections on frame
                frame = self._draw_detections_on_frame(frame, detections, width, height)
                total_detections += len(detections)
                processed_frames += 1
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        summary = {
            'input_video': str(video_path),
            'output_video': str(output_path),
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'total_detections': total_detections,
            'average_detections_per_frame': total_detections / processed_frames if processed_frames > 0 else 0
        }
        
        logger.info(f"✅ Video processing completed: {output_path}")
        return summary
    
    def export_onnx(self, output_path: Union[str, Path], 
                   input_size: Tuple[int, int] = None,
                   opset_version: int = 11,
                   simplify: bool = True) -> str:
        """
        Export model to ONNX format
        
        Args:
            output_path: Path to save ONNX model
            input_size: Input image size (H, W)
            opset_version: ONNX opset version
            simplify: Whether to simplify the model
            
        Returns:
            Path to exported model
        """
        output_path = Path(output_path)
        input_size = input_size or (self.input_size, self.input_size)
        
        logger.info(f"Exporting model to ONNX: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(
            1, self.spectral_bands, input_size[0], input_size[1],
            device=self.device
        )
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Simplify if requested
        if simplify:
            try:
                import onnxsim
                model_onnx = onnx.load(str(output_path))
                model_onnx, check = onnxsim.simplify(model_onnx)
                onnx.save(model_onnx, str(output_path))
                logger.info("✅ ONNX model simplified")
            except ImportError:
                logger.warning("onnxsim not available, skipping simplification")
        
        logger.info(f"✅ ONNX export completed: {output_path}")
        return str(output_path)
    
    def benchmark_model(self, num_runs: int = 100, 
                       input_size: Tuple[int, int] = None,
                       warmup_runs: int = 10) -> Dict:
        """
        Benchmark model performance
        
        Args:
            num_runs: Number of inference runs
            input_size: Input image size
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results
        """
        input_size = input_size or (self.input_size, self.input_size)
        
        logger.info(f"Benchmarking model performance...")
        logger.info(f"  Runs: {num_runs}, Warmup: {warmup_runs}")
        logger.info(f"  Input size: {input_size}")
        
        # Create dummy input
        dummy_input = torch.randn(
            1, self.spectral_bands, input_size[0], input_size[1],
            device=self.device
        )
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(dummy_input)
        
        # Synchronize GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        results = {
            'num_runs': num_runs,
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'median_time': float(np.median(times)),
            'fps': 1.0 / np.mean(times),
            'device': str(self.device),
            'input_size': input_size,
            'model_type': self.model_type
        }
        
        logger.info(f"✅ Benchmark completed:")
        logger.info(f"  Average time: {results['mean_time']:.4f}s ± {results['std_time']:.4f}s")
        logger.info(f"  FPS: {results['fps']:.2f}")
        logger.info(f"  Min/Max: {results['min_time']:.4f}s / {results['max_time']:.4f}s")
        
        return results
    
    def _load_and_preprocess_image(self, image_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
        """Load and preprocess image for inference"""
        # Load image
        if self.spectral_bands == 4:
            # Load multi-spectral image (TIFF with 4 bands)
            try:
                image = Image.open(image_path)
                if hasattr(image, 'n_frames') and image.n_frames >= 4:
                    # Multi-frame TIFF
                    bands = []
                    for i in range(4):
                        image.seek(i)
                        bands.append(np.array(image))
                    image_array = np.stack(bands, axis=-1)
                else:
                    # Regular image - duplicate channels or use RGB + create NIR
                    image_array = np.array(image)
                    if len(image_array.shape) == 2:
                        image_array = np.stack([image_array] * 4, axis=-1)
                    elif image_array.shape[-1] == 3:
                        # Add NIR channel (simple approximation)
                        nir = np.mean(image_array, axis=-1, keepdims=True)
                        image_array = np.concatenate([image_array, nir], axis=-1)
            except Exception:
                # Fallback to regular image loading
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Create 4-channel version
                nir = np.mean(image, axis=-1, keepdims=True)
                image_array = np.concatenate([image, nir], axis=-1)
        else:
            # Regular RGB image
            image = cv2.imread(str(image_path))
            image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_image = image_array.copy()
        
        # Resize to model input size
        image_array = cv2.resize(image_array, (self.input_size, self.input_size))
        
        # Convert to tensor
        if self.transforms:
            # Use transforms if available
            image_tensor = self.transforms(image_array)
        else:
            # Manual preprocessing
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            
            # Normalize
            if self.spectral_bands == 4:
                # Multi-spectral normalization
                mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(4, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225, 0.2]).view(4, 1, 1)
            else:
                # RGB normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            image_tensor = (image_tensor - mean) / std
        
        return image_tensor.to(self.device), original_image
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess video frame"""
        # Resize frame
        frame = cv2.resize(frame, (self.input_size, self.input_size))
        
        # Handle spectral bands
        if self.spectral_bands == 4 and frame.shape[-1] == 3:
            # Add NIR channel approximation
            nir = np.mean(frame, axis=-1, keepdims=True)
            frame = np.concatenate([frame, nir], axis=-1)
        elif self.spectral_bands == 3 and frame.shape[-1] == 4:
            # Use only RGB channels
            frame = frame[:, :, :3]
        
        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        if self.spectral_bands == 4:
            mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(4, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225, 0.2]).view(4, 1, 1)
        else:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        frame_tensor = (frame_tensor - mean) / std
        return frame_tensor.to(self.device)
    
    def _post_process_outputs(self, predictions: torch.Tensor, 
                            conf_threshold: float, nms_threshold: float) -> List[Dict]:
        """Post-process model outputs to get final detections"""
        detections = []
        
        # Handle different model output formats
        if isinstance(predictions, (list, tuple)):
            # Multiple scale outputs (YOLO-style)
            all_boxes = []
            all_scores = []
            all_classes = []
            
            for pred in predictions:
                # Decode each scale's predictions
                boxes, scores, classes = self._decode_yolo_output(pred, conf_threshold)
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)
            
            if all_boxes:
                # Concatenate all predictions
                boxes = torch.cat(all_boxes, dim=0)
                scores = torch.cat(all_scores, dim=0)
                classes = torch.cat(all_classes, dim=0)
                
                # Apply NMS
                keep_indices = non_max_suppression(boxes, scores, nms_threshold)
                
                for idx in keep_indices:
                    detections.append({
                        'bbox': boxes[idx].cpu().numpy(),
                        'confidence': scores[idx].item(),
                        'class': int(classes[idx].item()),
                        'class_name': self.class_names[int(classes[idx].item())]
                    })
        
        else:
            # Single tensor output
            boxes, scores, classes = self._decode_yolo_output(predictions, conf_threshold)
            
            if len(boxes) > 0:
                # Apply NMS
                keep_indices = non_max_suppression(boxes, scores, nms_threshold)
                
                for idx in keep_indices:
                    detections.append({
                        'bbox': boxes[idx].cpu().numpy(),
                        'confidence': scores[idx].item(),
                        'class': int(classes[idx].item()),
                        'class_name': self.class_names[int(classes[idx].item())]
                    })
        
        return detections
    
    def _decode_yolo_output(self, prediction: torch.Tensor, 
                          conf_threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode YOLO output to boxes, scores, and classes"""
        # Simplified YOLO decoding - should be adapted to your specific model
        batch_size, num_anchors, height, width, num_attrs = prediction.shape
        
        # Extract components
        xy = torch.sigmoid(prediction[..., :2])  # Center coordinates
        wh = prediction[..., 2:4]  # Width and height
        conf = torch.sigmoid(prediction[..., 4])  # Objectness confidence
        class_probs = torch.softmax(prediction[..., 5:], dim=-1)  # Class probabilities
        
        # Generate grid
        grid_x, grid_y = torch.meshgrid(
            torch.arange(width, device=prediction.device),
            torch.arange(height, device=prediction.device),
            indexing='xy'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        
        # Decode boxes
        xy = (xy + grid.unsqueeze(0).unsqueeze(0)) / torch.tensor([width, height], device=prediction.device)
        wh = torch.exp(wh)  # Should be normalized by anchors in real implementation
        
        # Convert to corner format
        x1 = xy[..., 0] - wh[..., 0] / 2
        y1 = xy[..., 1] - wh[..., 1] / 2
        x2 = xy[..., 0] + wh[..., 0] / 2
        y2 = xy[..., 1] + wh[..., 1] / 2
        
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        
        # Get class predictions
        class_conf, class_pred = torch.max(class_probs, dim=-1)
        total_conf = conf * class_conf
        
        # Filter by confidence
        conf_mask = total_conf > conf_threshold
        
        # Flatten and filter
        boxes = boxes[conf_mask]
        scores = total_conf[conf_mask]
        classes = class_pred[conf_mask]
        
        return boxes, scores, classes
    
    def _save_prediction_visualization(self, image: np.ndarray, detections: List[Dict],
                                     save_path: Path, inference_time: float,
                                     features: Optional[Dict] = None):
        """Save visualization of predictions"""
        fig, axes = plt.subplots(1, 2 if features else 1, figsize=(15 if features else 8, 6))
        if not features:
            axes = [axes]
        
        # Main prediction visualization
        ax = axes[0]
        if len(image.shape) == 3 and image.shape[-1] >= 3:
            display_image = image[:, :, :3]  # Use RGB channels
        else:
            display_image = image
        
        ax.imshow(display_image)
        ax.set_title(f'Detections: {len(detections)} objects\nInference: {inference_time:.3f}s')
        ax.axis('off')
        
        # Draw bounding boxes
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Scale bbox to image size
            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox
            x1 *= w
            y1 *= h
            x2 *= w
            y2 *= h
            
            # Draw rectangle
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                x1, y1 - 5, f'{class_name}: {confidence:.2f}',
                fontsize=10, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
            )
        
        # Feature visualization (if available)
        if features and len(axes) > 1:
            ax = axes[1]
            
            if 'attention_maps' in features:
                # Show attention map
                attention = features['attention_maps'][0].mean(dim=0).cpu().numpy()
                im = ax.imshow(attention, cmap='hot', alpha=0.7)
                ax.set_title('Attention Map (CBAM)')
                plt.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, 'No attention maps available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Features')
            
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _draw_detections_on_frame(self, frame: np.ndarray, detections: List[Dict],
                                frame_width: int, frame_height: int) -> np.ndarray:
        """Draw detections on video frame"""
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Scale bbox to frame size
            x1, y1, x2, y2 = bbox
            x1 = int(x1 * frame_width)
            y1 = int(y1 * frame_height)
            x2 = int(x2 * frame_width)
            y2 = int(y2 * frame_height)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f'{class_name}: {confidence:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def set_thresholds(self, conf_threshold: float = None, nms_threshold: float = None):
        """Set detection thresholds"""
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if nms_threshold is not None:
            self.nms_threshold = nms_threshold
        
        logger.info(f"Updated thresholds: conf={self.conf_threshold}, nms={self.nms_threshold}")

class ONNXPredictor:
    """ONNX-based predictor for faster inference"""
    
    def __init__(self, onnx_path: str, class_names: List[str] = None, 
                 providers: List[str] = None):
        """
        Initialize ONNX predictor
        
        Args:
            onnx_path: Path to ONNX model
            class_names: List of class names
            providers: ONNX Runtime providers
        """
        self.onnx_path = Path(onnx_path)
        self.class_names = class_names or ['Cotton', 'Rice', 'Corn']
        
        # Set providers
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Load ONNX model
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        logger.info(f"✅ ONNX model loaded: {onnx_path}")
        logger.info(f"   Input shape: {self.input_shape}")
        logger.info(f"   Providers: {self.session.get_providers()}")
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict using ONNX model"""
        # Preprocess image to match input shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)  # Add batch dimension
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: image})
        return outputs[0]

# CLI interface
def main():
    """Command line interface for model prediction"""
    parser = argparse.ArgumentParser(description='CBAM-STN-TPS-YOLO Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image/directory/video path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory path')
    parser.add_argument('--model_type', type=str, default='CBAM-STN-TPS-YOLO',
                       help='Model type')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.45,
                       help='NMS threshold')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory of images')
    parser.add_argument('--video', action='store_true',
                       help='Process video file')
    parser.add_argument('--export_onnx', type=str, default=None,
                       help='Export model to ONNX format')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create predictor
    predictor = ModelPredictor(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device
    )
    
    # Set thresholds
    predictor.set_thresholds(args.conf_threshold, args.nms_threshold)
    
    # Export ONNX if requested
    if args.export_onnx:
        predictor.export_onnx(args.export_onnx)
        return
    
    # Benchmark if requested
    if args.benchmark:
        results = predictor.benchmark_model()
        print(json.dumps(results, indent=2))
        return
    
    # Run prediction
    if args.video:
        # Video processing
        predictor.predict_video(args.input, args.output, 
                               args.conf_threshold, args.nms_threshold)
    elif args.batch:
        # Batch processing
        predictor.predict_batch(args.input, args.output,
                               args.conf_threshold, args.nms_threshold)
    else:
        # Single image
        result = predictor.predict_image(args.input, args.conf_threshold,
                                       args.nms_threshold, save_result=True,
                                       output_dir=args.output)
        print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()