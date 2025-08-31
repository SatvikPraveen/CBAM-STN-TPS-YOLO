# src/inference/__init__.py
"""
Inference and prediction utilities for CBAM-STN-TPS-YOLO
Comprehensive inference pipeline with multiple deployment options
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# Core inference components
from .predict import ModelPredictor, ONNXPredictor

logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"

# Available prediction backends
AVAILABLE_BACKENDS = ['pytorch', 'onnx']

# Supported input formats
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']

def create_predictor(model_path: Union[str, Path], 
                    backend: str = 'pytorch',
                    model_type: str = 'CBAM-STN-TPS-YOLO',
                    class_names: Optional[List[str]] = None,
                    device: Optional[str] = None,
                    **kwargs) -> Union[ModelPredictor, ONNXPredictor]:
    """
    Factory function to create predictors
    
    Args:
        model_path: Path to model file (.pth for PyTorch, .onnx for ONNX)
        backend: Prediction backend ('pytorch' or 'onnx')
        model_type: Type of model architecture
        class_names: List of class names for predictions
        device: Device to run inference on
        **kwargs: Additional arguments for predictor initialization
        
    Returns:
        Predictor instance
        
    Raises:
        ValueError: If backend is not supported
        FileNotFoundError: If model file doesn't exist
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    backend = backend.lower()
    if backend not in AVAILABLE_BACKENDS:
        raise ValueError(f"Backend '{backend}' not supported. Available: {AVAILABLE_BACKENDS}")
    
    # Default class names for agricultural detection
    if class_names is None:
        class_names = ['Cotton', 'Rice', 'Corn', 'Wheat', 'Soybean']
    
    logger.info(f"Creating {backend} predictor with model: {model_path}")
    
    if backend == 'pytorch':
        return ModelPredictor(
            model_path=str(model_path),
            model_type=model_type,
            class_names=class_names,
            device=device,
            **kwargs
        )
    
    elif backend == 'onnx':
        if model_path.suffix.lower() != '.onnx':
            logger.warning(f"Expected .onnx file for ONNX backend, got: {model_path.suffix}")
        
        return ONNXPredictor(
            onnx_path=str(model_path),
            class_names=class_names,
            **kwargs
        )

def batch_predict(predictor: Union[ModelPredictor, ONNXPredictor],
                 input_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 file_pattern: str = '*',
                 max_files: Optional[int] = None,
                 **kwargs) -> Dict:
    """
    Convenience function for batch prediction
    
    Args:
        predictor: Predictor instance
        input_dir: Directory containing input files
        output_dir: Directory to save results
        file_pattern: File pattern to match
        max_files: Maximum number of files to process
        **kwargs: Additional arguments for prediction
        
    Returns:
        Batch processing summary
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    logger.info(f"Starting batch prediction:")
    logger.info(f"  Input: {input_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Pattern: {file_pattern}")
    
    if hasattr(predictor, 'predict_batch'):
        # Use predictor's batch method if available
        return predictor.predict_batch(
            input_dir, output_dir, 
            max_images=max_files, **kwargs
        )
    else:
        # Fallback to individual predictions
        image_files = list(input_dir.glob(file_pattern))
        if max_files:
            image_files = image_files[:max_files]
        
        results = []
        for img_path in image_files:
            if img_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                try:
                    if hasattr(predictor, 'predict_image'):
                        result = predictor.predict_image(
                            img_path, save_result=True,
                            output_dir=output_dir, **kwargs
                        )
                    else:
                        # Basic prediction for ONNX
                        import cv2
                        image = cv2.imread(str(img_path))
                        prediction = predictor.predict(image)
                        result = {'image_path': str(img_path), 'prediction': prediction}
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
        
        return {
            'total_processed': len(results),
            'results': results
        }

def convert_model_to_onnx(pytorch_model_path: Union[str, Path],
                         onnx_output_path: Union[str, Path],
                         model_type: str = 'CBAM-STN-TPS-YOLO',
                         input_size: tuple = (640, 640),
                         spectral_bands: int = 4,
                         **kwargs) -> str:
    """
    Convert PyTorch model to ONNX format
    
    Args:
        pytorch_model_path: Path to PyTorch model checkpoint
        onnx_output_path: Path to save ONNX model
        model_type: Type of model architecture
        input_size: Input image size (H, W)
        spectral_bands: Number of input channels
        **kwargs: Additional arguments for ONNX export
        
    Returns:
        Path to exported ONNX model
    """
    logger.info(f"Converting model to ONNX:")
    logger.info(f"  Input: {pytorch_model_path}")
    logger.info(f"  Output: {onnx_output_path}")
    
    # Create predictor
    predictor = create_predictor(
        pytorch_model_path, 
        backend='pytorch',
        model_type=model_type
    )
    
    # Export to ONNX
    return predictor.export_onnx(
        onnx_output_path,
        input_size=input_size,
        **kwargs
    )

def benchmark_inference(model_path: Union[str, Path],
                       backend: str = 'pytorch',
                       model_type: str = 'CBAM-STN-TPS-YOLO',
                       num_runs: int = 100,
                       **kwargs) -> Dict:
    """
    Benchmark model inference performance
    
    Args:
        model_path: Path to model file
        backend: Inference backend
        model_type: Model architecture type
        num_runs: Number of benchmark runs
        **kwargs: Additional arguments
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking {backend} inference with {num_runs} runs")
    
    predictor = create_predictor(
        model_path,
        backend=backend,
        model_type=model_type,
        **kwargs
    )
    
    if hasattr(predictor, 'benchmark_model'):
        return predictor.benchmark_model(num_runs=num_runs)
    else:
        logger.warning(f"Benchmark not available for {backend} backend")
        return {}

def get_model_info(model_path: Union[str, Path]) -> Dict:
    """
    Get information about a trained model
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        Model information dictionary
    """
    import torch
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if model_path.suffix.lower() == '.onnx':
        # ONNX model info
        import onnx
        model = onnx.load(str(model_path))
        
        return {
            'model_format': 'ONNX',
            'model_path': str(model_path),
            'model_size_mb': model_path.stat().st_size / (1024 * 1024),
            'opset_version': model.opset_import[0].version,
            'input_shape': [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim],
            'output_shape': [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
        }
    
    else:
        # PyTorch model info
        checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            'model_format': 'PyTorch',
            'model_path': str(model_path),
            'model_size_mb': model_path.stat().st_size / (1024 * 1024)
        }
        
        # Extract configuration if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            info.update({
                'model_type': config.get('model', {}).get('type', 'Unknown'),
                'num_classes': config.get('num_classes', 'Unknown'),
                'input_size': config.get('input_size', 'Unknown'),
                'spectral_bands': config.get('spectral_bands', 'Unknown')
            })
        
        # Extract training info if available
        if 'epoch' in checkpoint:
            info['trained_epochs'] = checkpoint['epoch']
        
        if 'best_metrics' in checkpoint:
            info['best_metrics'] = checkpoint['best_metrics']
        
        # Model parameters info
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            total_params = sum(p.numel() for p in state_dict.values())
            info['total_parameters'] = total_params
            info['parameter_size_mb'] = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return info

def validate_input_format(input_path: Union[str, Path], 
                         input_type: str = 'auto') -> str:
    """
    Validate and determine input format
    
    Args:
        input_path: Path to input file/directory
        input_type: Expected input type ('auto', 'image', 'video', 'directory')
        
    Returns:
        Detected input type
        
    Raises:
        ValueError: If input format is not supported
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    if input_path.is_dir():
        detected_type = 'directory'
    elif input_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
        detected_type = 'image'
    elif input_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
        detected_type = 'video'
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")
    
    if input_type != 'auto' and input_type != detected_type:
        raise ValueError(f"Expected {input_type}, but detected {detected_type}")
    
    logger.info(f"Input type detected: {detected_type}")
    return detected_type

def get_available_models() -> Dict[str, List[str]]:
    """
    Get information about available model architectures
    
    Returns:
        Dictionary of available models and their variants
    """
    return {
        'YOLO_variants': [
            'YOLO',
            'STN-YOLO', 
            'STN-TPS-YOLO',
            'CBAM-STN-YOLO',
            'CBAM-STN-TPS-YOLO'
        ],
        'supported_backends': AVAILABLE_BACKENDS,
        'supported_image_formats': SUPPORTED_IMAGE_FORMATS,
        'supported_video_formats': SUPPORTED_VIDEO_FORMATS,
        'default_classes': [
            'Cotton', 'Rice', 'Corn', 'Wheat', 'Soybean',
            'Flower', 'Fruit', 'Leaf', 'Stem', 'Root'
        ]
    }

def setup_inference_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """
    Setup logging for inference module
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    import logging
    
    # Configure logger
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    logger.info(f"Inference logging setup complete (level: {log_level})")

# Export all public components
__all__ = [
    # Core classes
    'ModelPredictor',
    'ONNXPredictor',
    
    # Factory functions
    'create_predictor',
    
    # Utility functions
    'batch_predict',
    'convert_model_to_onnx',
    'benchmark_inference',
    'get_model_info',
    'validate_input_format',
    'get_available_models',
    'setup_inference_logging',
    
    # Constants
    'AVAILABLE_BACKENDS',
    'SUPPORTED_IMAGE_FORMATS',
    'SUPPORTED_VIDEO_FORMATS'
]

# Module initialization
logger.info(f"CBAM-STN-TPS-YOLO Inference Module v{__version__} loaded")
logger.info(f"Available backends: {AVAILABLE_BACKENDS}")