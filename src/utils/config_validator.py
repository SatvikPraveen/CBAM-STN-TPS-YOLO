import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Advanced configuration validator for CBAM-STN-TPS-YOLO experiments"""
    
    def __init__(self):
        self.required_fields = [
            'data_dir', 'num_classes', 'input_channels', 
            'image_size', 'batch_size', 'epochs', 'lr'
        ]
        
        self.optional_fields = [
            'weight_decay', 'momentum', 'scheduler', 'optimizer',
            'save_dir', 'experiment_name', 'seed', 'device',
            'num_workers', 'pin_memory', 'mixed_precision'
        ]
        
        self.model_specific_fields = {
            'STN': ['stn_loc_hidden_dim'],
            'TPS': ['num_control_points', 'tps_reg_lambda'],
            'CBAM': ['cbam_reduction_ratio'],
            'YOLO': ['num_anchors', 'anchor_scales']
        }
        
        self.value_ranges = {
            'batch_size': (1, 256),
            'lr': (1e-6, 1.0),
            'epochs': (1, 10000),
            'image_size': [224, 256, 320, 384, 416, 512, 608, 640, 768, 896, 1024],
            'num_classes': (1, 1000),
            'input_channels': (1, 10),
            'weight_decay': (0, 1.0),
            'momentum': (0, 1.0),
            'num_control_points': (4, 50),
            'tps_reg_lambda': (1e-6, 10.0),
            'cbam_reduction_ratio': [2, 4, 8, 16, 32, 64],
            'stn_loc_hidden_dim': (32, 1024)
        }
    
    def validate_training_config(self, config: Dict[str, Any]) -> bool:
        """Comprehensive training configuration validation"""
        errors = []
        warnings_list = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate data types and ranges
        self._validate_field_types_and_ranges(config, errors, warnings_list)
        
        # Validate paths
        self._validate_paths(config, errors, warnings_list)
        
        # Validate compatibility
        self._validate_compatibility(config, errors, warnings_list)
        
        # Report warnings
        for warning in warnings_list:
            warnings.warn(warning)
            logger.warning(warning)
        
        # Raise errors if any
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Training configuration validation passed")
        return True
    
    def validate_model_config(self, config: Dict[str, Any], model_type: str) -> bool:
        """Validate model-specific configuration"""
        errors = []
        warnings_list = []
        
        # Validate model type
        valid_model_types = [
            'YOLO', 'STN-YOLO', 'STN-TPS-YOLO', 
            'CBAM-YOLO', 'CBAM-STN-YOLO', 'CBAM-STN-TPS-YOLO'
        ]
        
        if model_type not in valid_model_types:
            errors.append(f"Unknown model type: {model_type}. Valid types: {valid_model_types}")
        
        # Validate component-specific requirements
        self._validate_component_requirements(config, model_type, errors)
        
        # Validate component compatibility
        self._validate_component_compatibility(config, model_type, warnings_list)
        
        # Report warnings
        for warning in warnings_list:
            warnings.warn(warning)
            logger.warning(warning)
        
        if errors:
            error_msg = "Model configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Model configuration validation passed for {model_type}")
        return True
    
    def _validate_field_types_and_ranges(self, config: Dict[str, Any], errors: List[str], warnings_list: List[str]):
        """Validate field types and value ranges"""
        
        # Integer fields
        int_fields = ['batch_size', 'epochs', 'num_classes', 'input_channels', 'image_size', 'num_workers']
        for field in int_fields:
            if field in config:
                if not isinstance(config[field], int):
                    errors.append(f"{field} must be an integer, got {type(config[field])}")
                elif field in self.value_ranges:
                    self._validate_range(field, config[field], errors, warnings_list)
        
        # Float fields
        float_fields = ['lr', 'weight_decay', 'momentum', 'tps_reg_lambda']
        for field in float_fields:
            if field in config:
                if not isinstance(config[field], (int, float)):
                    errors.append(f"{field} must be a number, got {type(config[field])}")
                elif field in self.value_ranges:
                    self._validate_range(field, config[field], errors, warnings_list)
        
        # Boolean fields
        bool_fields = ['mixed_precision', 'pin_memory', 'use_wandb', 'early_stopping']
        for field in bool_fields:
            if field in config and not isinstance(config[field], bool):
                errors.append(f"{field} must be a boolean, got {type(config[field])}")
        
        # String fields
        str_fields = ['optimizer', 'scheduler', 'data_dir', 'save_dir', 'experiment_name']
        for field in str_fields:
            if field in config and not isinstance(config[field], str):
                errors.append(f"{field} must be a string, got {type(config[field])}")
        
        # List fields
        list_fields = ['class_names', 'augmentation_types']
        for field in list_fields:
            if field in config and not isinstance(config[field], list):
                errors.append(f"{field} must be a list, got {type(config[field])}")
    
    def _validate_range(self, field: str, value: Union[int, float], errors: List[str], warnings_list: List[str]):
        """Validate value is within acceptable range"""
        range_spec = self.value_ranges[field]
        
        if isinstance(range_spec, tuple):
            min_val, max_val = range_spec
            if not (min_val <= value <= max_val):
                errors.append(f"{field} must be between {min_val} and {max_val}, got {value}")
        elif isinstance(range_spec, list):
            if value not in range_spec:
                if field == 'image_size':
                    # Find closest valid size
                    closest = min(range_spec, key=lambda x: abs(x - value))
                    warnings_list.append(f"{field}={value} not in recommended sizes {range_spec}. Closest: {closest}")
                else:
                    errors.append(f"{field} must be one of {range_spec}, got {value}")
    
    def _validate_paths(self, config: Dict[str, Any], errors: List[str], warnings_list: List[str]):
        """Validate file and directory paths"""
        
        # Data directory
        if 'data_dir' in config:
            data_path = Path(config['data_dir'])
            if not data_path.exists():
                errors.append(f"data_dir does not exist: {config['data_dir']}")
            elif not data_path.is_dir():
                errors.append(f"data_dir is not a directory: {config['data_dir']}")
            else:
                # Check for expected subdirectories
                expected_subdirs = ['train', 'val']
                missing_subdirs = []
                
                for subdir in expected_subdirs:
                    if not (data_path / subdir).exists():
                        # Try alternative structure
                        if not (data_path / 'images' / subdir).exists():
                            missing_subdirs.append(subdir)
                
                if missing_subdirs:
                    warnings_list.append(
                        f"Missing expected subdirectories in data_dir: {missing_subdirs}. "
                        f"Expected structure: data_dir/{{train,val}}/{{images,labels}}/"
                    )
        
        # Save directory
        if 'save_dir' in config:
            save_path = Path(config['save_dir'])
            try:
                save_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create save_dir: {config['save_dir']} - {e}")
        
        # Pretrained weights
        if 'pretrained_weights' in config and config['pretrained_weights']:
            weights_path = Path(config['pretrained_weights'])
            if not weights_path.exists():
                errors.append(f"Pretrained weights file not found: {config['pretrained_weights']}")
    
    def _validate_compatibility(self, config: Dict[str, Any], errors: List[str], warnings_list: List[str]):
        """Validate configuration compatibility"""
        
        # Batch size vs GPU memory
        if 'batch_size' in config and 'image_size' in config:
            batch_size = config['batch_size']
            image_size = config['image_size']
            
            # Rough memory estimation (MB)
            estimated_memory = batch_size * image_size * image_size * 4 * 3 / (1024 * 1024)
            
            if estimated_memory > 8000:  # 8GB threshold
                warnings_list.append(
                    f"Large memory usage estimated: {estimated_memory:.0f}MB "
                    f"(batch_size={batch_size}, image_size={image_size}). "
                    f"Consider reducing batch_size or image_size if you encounter OOM errors."
                )
        
        # Learning rate vs optimizer
        if 'lr' in config and 'optimizer' in config:
            lr = config['lr']
            optimizer = config['optimizer'].lower()
            
            if optimizer == 'adam' and lr > 0.01:
                warnings_list.append(f"High learning rate ({lr}) for Adam optimizer. Consider lr < 0.01")
            elif optimizer == 'sgd' and lr < 0.001:
                warnings_list.append(f"Low learning rate ({lr}) for SGD optimizer. Consider lr > 0.001")
        
        # Mixed precision compatibility
        if config.get('mixed_precision', False):
            try:
                import torch
                if not torch.cuda.is_available():
                    warnings_list.append("Mixed precision enabled but CUDA not available")
            except ImportError:
                pass
    
    def _validate_component_requirements(self, config: Dict[str, Any], model_type: str, errors: List[str]):
        """Validate component-specific requirements"""
        
        # TPS requirements
        if 'TPS' in model_type:
            if 'num_control_points' not in config:
                errors.append("TPS models require 'num_control_points' parameter")
            elif config['num_control_points'] < 4:
                errors.append("num_control_points must be at least 4 for TPS")
            
            if 'tps_reg_lambda' not in config:
                config['tps_reg_lambda'] = 0.1  # Default value
                logger.info("Added default tps_reg_lambda=0.1")
        
        # CBAM requirements
        if 'CBAM' in model_type:
            if 'cbam_reduction_ratio' not in config:
                config['cbam_reduction_ratio'] = 16  # Default value
                logger.info("Added default cbam_reduction_ratio=16")
        
        # STN requirements
        if 'STN' in model_type:
            if 'stn_loc_hidden_dim' not in config:
                config['stn_loc_hidden_dim'] = 128  # Default value
                logger.info("Added default stn_loc_hidden_dim=128")
    
    def _validate_component_compatibility(self, config: Dict[str, Any], model_type: str, warnings_list: List[str]):
        """Validate component compatibility"""
        
        # TPS + high resolution warning
        if 'TPS' in model_type and config.get('image_size', 512) > 640:
            warnings_list.append(
                "TPS with high resolution images may be computationally expensive. "
                "Consider reducing image_size or num_control_points."
            )
        
        # CBAM + large batch size
        if 'CBAM' in model_type and config.get('batch_size', 16) > 32:
            warnings_list.append(
                "CBAM with large batch sizes may increase memory usage significantly."
            )

def validate_experiment_config(config: Dict[str, Any], model_type: str = None) -> bool:
    """Validate complete experiment configuration"""
    validator = ConfigValidator()
    
    # Validate training config
    validator.validate_training_config(config)
    
    # Validate model config if model_type provided
    if model_type:
        validator.validate_model_config(config, model_type)
    
    return True

def load_and_validate_config(config_path: str, model_type: str = None) -> Dict[str, Any]:
    """Load and validate configuration file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load config based on file extension
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # Validate config
    validate_experiment_config(config, model_type)
    
    logger.info(f"Configuration loaded and validated from: {config_path}")
    return config

def create_config_template(model_type: str = 'CBAM-STN-TPS-YOLO', 
                          save_path: str = None) -> Dict[str, Any]:
    """Create configuration template for specified model type"""
    
    # Base template
    config_template = {
        # Data configuration
        'data_dir': 'data/PGP',
        'num_classes': 3,
        'input_channels': 4,  # Multi-spectral
        'image_size': 640,
        'class_names': ['cotton', 'rice', 'corn'],
        
        # Training configuration
        'batch_size': 16,
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 0.0005,
        'momentum': 0.9,
        'optimizer': 'Adam',
        'scheduler': 'CosineAnnealingLR',
        
        # Model configuration
        'model_type': model_type,
        'pretrained': True,
        
        # Training options
        'mixed_precision': True,
        'num_workers': 4,
        'pin_memory': True,
        'early_stopping': True,
        'early_stopping_patience': 10,
        
        # Logging and saving
        'save_dir': 'experiments',
        'experiment_name': f'{model_type.lower()}_experiment',
        'use_wandb': True,
        'save_best_only': True,
        
        # Reproducibility
        'seed': 42,
        'deterministic': True
    }
    
    # Add model-specific parameters
    if 'TPS' in model_type:
        config_template.update({
            'num_control_points': 20,
            'tps_reg_lambda': 0.1
        })
    
    if 'CBAM' in model_type:
        config_template.update({
            'cbam_reduction_ratio': 16
        })
    
    if 'STN' in model_type:
        config_template.update({
            'stn_loc_hidden_dim': 128
        })
    
    # Add YOLO-specific parameters
    if 'YOLO' in model_type:
        config_template.update({
            'num_anchors': 3,
            'anchor_scales': [1, 2, 4],
            'iou_threshold': 0.5,
            'conf_threshold': 0.1,
            'nms_threshold': 0.4
        })
    
    # Save template if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.yaml' or save_path.suffix == '.yml':
            with open(save_path, 'w') as f:
                yaml.dump(config_template, f, default_flow_style=False, indent=2)
        elif save_path.suffix == '.json':
            with open(save_path, 'w') as f:
                json.dump(config_template, f, indent=2)
        
        logger.info(f"Configuration template saved to: {save_path}")
    
    return config_template

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations with override taking precedence"""
    merged = base_config.copy()
    
    def update_nested_dict(base_dict, override_dict):
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    update_nested_dict(merged, override_config)
    
    logger.info("Configuration merge completed")
    return merged

def compare_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two configurations and return differences"""
    differences = {}
    
    all_keys = set(config1.keys()) | set(config2.keys())
    
    for key in all_keys:
        if key not in config1:
            differences[key] = {'status': 'missing_in_config1', 'config2_value': config2[key]}
        elif key not in config2:
            differences[key] = {'status': 'missing_in_config2', 'config1_value': config1[key]}
        elif config1[key] != config2[key]:
            differences[key] = {
                'status': 'different',
                'config1_value': config1[key],
                'config2_value': config2[key]
            }
    
    return differences

if __name__ == "__main__":
    # Test configuration validation
    print("Testing configuration validation...")
    
    # Create test config
    test_config = create_config_template('CBAM-STN-TPS-YOLO')
    
    try:
        # Test validation
        validator = ConfigValidator()
        validator.validate_training_config(test_config)
        validator.validate_model_config(test_config, 'CBAM-STN-TPS-YOLO')
        print("✅ Configuration validation test passed")
        
        # Test template creation
        template = create_config_template('STN-TPS-YOLO')
        print("✅ Template creation test passed")
        
        # Test config comparison
        differences = compare_configs(test_config, template)
        print(f"✅ Config comparison test passed: {len(differences)} differences found")
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
    
    print("Configuration validation tests completed!")