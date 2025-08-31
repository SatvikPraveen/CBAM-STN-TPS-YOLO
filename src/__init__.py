"""
CBAM-STN-TPS-YOLO: Enhancing Agricultural Object Detection
Authors: Satvik Praveen, Yoonsung Jung
Institution: Texas A&M University

A comprehensive deep learning framework combining Convolutional Block Attention Module (CBAM),
Spatial Transformer Networks (STN), Thin Plate Splines (TPS), and YOLO for enhanced
agricultural object detection in challenging conditions.
"""

__version__ = "1.0.0"
__authors__ = ["Satvik Praveen", "Yoonsung Jung"]
__email__ = ["satvikpraveen_164@tamu.edu", "yojung@tamu.edu"]
__institution__ = "Texas A&M University"
__description__ = "Enhanced Agricultural Object Detection with CBAM-STN-TPS-YOLO"
__url__ = "https://github.com/your-username/CBAM-STN-TPS-YOLO"

import sys
import warnings
from pathlib import Path

# Add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

# Core Dependencies Check
try:
    import torch
    import torchvision
    import numpy as np
    import cv2
    import yaml
except ImportError as e:
    raise ImportError(
        f"Missing required dependency: {e.name}. "
        "Please install all requirements with: pip install -r requirements.txt"
    )

# Version compatibility checks
def check_compatibility():
    """Check for compatible versions of key dependencies"""
    torch_version = torch.__version__
    if not torch_version.startswith(('1.9', '1.10', '1.11', '1.12', '1.13', '2.0', '2.1')):
        warnings.warn(
            f"PyTorch version {torch_version} may not be fully compatible. "
            "Recommended: PyTorch 1.9+ or 2.0+"
        )

check_compatibility()

# Model imports with error handling
try:
    from .models.cbam import CBAM, ChannelAttention, SpatialAttention
    from .models.stn_tps import STN_TPS, TPS_SpatialTransformerNetwork
    from .models.yolo_backbone import YOLOBackbone, ConvBlock, CSPDarknet53
    from .models.cbam_stn_tps_yolo import (
        CBAM_STN_TPS_YOLO, 
        create_model,
        ModelVariants
    )
    from .models.detection_head import DetectionHead, YOLOHead
except ImportError as e:
    warnings.warn(f"Could not import models: {e}")
    # Provide fallback empty classes to prevent complete failure
    class CBAM_STN_TPS_YOLO: pass
    class ModelVariants: pass

# Data pipeline imports
try:
    from .data.dataset import (
        PGPDataset, 
        MelonFlowerDataset, 
        GlobalWheatDataset,
        MultiSpectralDataset,
        get_dataset
    )
    from .data.transforms import (
        Compose,
        RandomRotation,
        RandomShear,
        ColorJitter,
        Normalize,
        ToTensor,
        SpectralTransform,
        PseudoRGBConversion
    )
    from .data.preprocessing import (
        preprocess_pgp_data,
        preprocess_melon_flower_data,
        preprocess_global_wheat_data,
        create_data_loaders,
        DataPreprocessor
    )
except ImportError as e:
    warnings.warn(f"Could not import data modules: {e}")

# Training infrastructure imports
try:
    from .training.trainer import CBAMSTNTPSYOLOTrainer
    from .training.losses import (
        YOLOLoss,
        FocalLoss,
        IoULoss,
        CombinedLoss,
        compute_loss
    )
    from .training.metrics import (
        DetectionMetrics,
        compute_ap,
        compute_precision_recall,
        compute_f1_score
    )
except ImportError as e:
    warnings.warn(f"Could not import training modules: {e}")

# Utility imports
try:
    from .utils.visualization import (
        Visualizer,
        plot_training_curves,
        visualize_predictions,
        plot_attention_maps,
        visualize_transformations
    )
    from .utils.evaluation import (
        ModelEvaluator,
        evaluate_model,
        benchmark_models
    )
    from .utils.config_validator import (
        ConfigValidator,
        load_and_validate_config
    )
except ImportError as e:
    warnings.warn(f"Could not import utility modules: {e}")

# Inference imports
try:
    from .inference.predict import (
        ModelPredictor,
        load_model,
        predict_single_image,
        predict_batch
    )
except ImportError as e:
    warnings.warn(f"Could not import inference modules: {e}")

# Experimental framework imports
try:
    import sys
    from pathlib import Path
    # Add experiments directory to path
    experiments_path = Path(__file__).parent.parent / "experiments"
    if experiments_path.exists():
        sys.path.insert(0, str(experiments_path))
        
    from experiments.run_experiments import (
        run_full_experiment,
        run_ablation_study,
        compare_models
    )
    from experiments.statistical_analysis import (
        perform_statistical_analysis,
        compute_statistical_significance,
        generate_comparison_report
    )
    from experiments.ablation_study import (
        AblationStudy,
        run_component_ablation,
        analyze_component_importance
    )
except ImportError as e:
    warnings.warn(f"Could not import experimental modules: {e}")

# Public API definition
__all__ = [
    # Version info
    '__version__', '__authors__', '__email__', '__institution__',
    
    # Core models
    'CBAM_STN_TPS_YOLO', 'CBAM', 'STN_TPS', 'YOLOBackbone', 'DetectionHead',
    'create_model', 'ModelVariants',
    
    # Data handling
    'PGPDataset', 'MelonFlowerDataset', 'GlobalWheatDataset', 'MultiSpectralDataset',
    'get_dataset', 'create_data_loaders', 'DataPreprocessor',
    
    # Transforms
    'Compose', 'RandomRotation', 'RandomShear', 'ColorJitter', 'Normalize',
    'ToTensor', 'SpectralTransform', 'PseudoRGBConversion',
    
    # Training
    'CBAMSTNTPSYOLOTrainer', 'YOLOLoss', 'FocalLoss', 'IoULoss', 'CombinedLoss',
    'DetectionMetrics', 'compute_ap', 'compute_precision_recall', 'compute_f1_score',
    
    # Utilities
    'Visualizer', 'ModelEvaluator', 'ConfigValidator',
    'plot_training_curves', 'visualize_predictions', 'evaluate_model',
    'load_and_validate_config',
    
    # Inference
    'ModelPredictor', 'load_model', 'predict_single_image', 'predict_batch',
    
    # Experiments
    'run_full_experiment', 'run_ablation_study', 'compare_models',
    'perform_statistical_analysis', 'AblationStudy',
    
    # Preprocessing functions
    'preprocess_pgp_data', 'preprocess_melon_flower_data', 'preprocess_global_wheat_data',
]

# Convenience functions for quick setup
def get_default_config():
    """Get default configuration for CBAM-STN-TPS-YOLO"""
    return {
        'model_type': 'CBAM-STN-TPS-YOLO',
        'num_classes': 3,
        'input_channels': 4,  # Multi-spectral agricultural data
        'image_size': 640,
        'batch_size': 16,
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 0.0005,
        'use_mixed_precision': True,
        'use_wandb': True,
        'early_stopping_patience': 10,
        'cbam_reduction_ratio': 16,
        'num_control_points': 20,
        'tps_regularization': 0.1
    }

def create_default_model(num_classes=3, input_channels=4, pretrained=False):
    """Create a default CBAM-STN-TPS-YOLO model with standard agricultural settings"""
    config = get_default_config()
    config['num_classes'] = num_classes
    config['input_channels'] = input_channels
    
    try:
        return create_model(config, model_type='CBAM-STN-TPS-YOLO', pretrained=pretrained)
    except NameError:
        raise ImportError("Could not create model. Please check that all dependencies are installed.")

def setup_experiment(config_path=None, experiment_name="default"):
    """Setup a complete experiment with all necessary components"""
    if config_path is None:
        config = get_default_config()
    else:
        try:
            config = load_and_validate_config(config_path)
        except NameError:
            raise ImportError("Could not load config validator. Please check imports.")
    
    # Create model
    model = create_default_model(
        num_classes=config['num_classes'],
        input_channels=config['input_channels']
    )
    
    # Setup trainer
    try:
        trainer = CBAMSTNTPSYOLOTrainer(config, model_type=config['model_type'])
        return model, trainer, config
    except NameError:
        raise ImportError("Could not create trainer. Please check that training modules are available.")

# Package information
def print_package_info():
    """Print comprehensive package information"""
    print(f"""
CBAM-STN-TPS-YOLO v{__version__}
{'='*50}
Authors: {', '.join(__authors__)}
Institution: {__institution__}
Contact: {', '.join(__email__)}

This package implements an enhanced agricultural object detection system
combining Convolutional Block Attention Module (CBAM), Spatial Transformer
Networks (STN), Thin Plate Splines (TPS), and YOLO for robust detection
in challenging agricultural environments.

Key Features:
- Multi-spectral agricultural data support (4-band imaging)
- Geometric transformation invariance via STN-TPS
- Enhanced feature attention with CBAM
- Comprehensive experimental framework
- Production-ready inference pipeline
- Statistical significance testing

Supported Datasets:
- Plant Genome Project (PGP)
- MelonFlower Dataset  
- Global Wheat Detection Dataset

For documentation and examples, visit: {__url__}
    """)

# Development utilities
def check_installation():
    """Check if the package is properly installed and configured"""
    issues = []
    
    # Check core dependencies
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA is not available - GPU acceleration disabled")
    except ImportError:
        issues.append("PyTorch is not installed")
    
    try:
        import cv2
    except ImportError:
        issues.append("OpenCV is not installed")
    
    try:
        import wandb
    except ImportError:
        issues.append("Weights & Biases (wandb) is not installed - experiment tracking disabled")
    
    # Check model imports
    try:
        model = create_default_model()
        print("✅ Model creation successful")
    except Exception as e:
        issues.append(f"Model creation failed: {e}")
    
    if issues:
        print("⚠️  Installation Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRun: pip install -r requirements.txt to resolve dependencies")
    else:
        print("✅ Installation check passed - all components are working correctly")
    
    return len(issues) == 0

# Add convenience shortcuts
def quick_train(data_dir, num_classes=3, epochs=50, model_type='CBAM-STN-TPS-YOLO'):
    """Quick training setup for rapid experimentation"""
    config = get_default_config()
    config.update({
        'data_dir': data_dir,
        'num_classes': num_classes,
        'epochs': epochs,
        'model_type': model_type
    })
    
    model, trainer, config = setup_experiment()
    return trainer.train()

# Version check on import
if __name__ == "__main__":
    print_package_info()
    check_installation()