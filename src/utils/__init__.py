"""
Utility functions for visualization, evaluation, and configuration validation
Enhanced utilities for CBAM-STN-TPS-YOLO research framework
"""

import warnings
from pathlib import Path

# Core utility imports
from .visualization import (
    Visualizer,
    plot_training_curves,
    visualize_predictions,
    plot_attention_maps,
    visualize_transformations,
    create_comparison_plots,
    save_experiment_plots
)

from .evaluation import (
    ModelEvaluator,
    evaluate_model,
    benchmark_models,
    calculate_model_complexity,
    analyze_failure_cases,
    generate_evaluation_report
)

from .config_validator import (
    ConfigValidator,
    load_and_validate_config,
    validate_experiment_config,
    create_config_template,
    merge_configs
)

# Additional utility functions
def setup_experiment_directory(experiment_name: str, base_dir: str = "experiments") -> Path:
    """Setup experiment directory structure"""
    exp_dir = Path(base_dir) / experiment_name
    
    # Create subdirectories
    subdirs = ['logs', 'models', 'plots', 'results', 'configs']
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return exp_dir

def get_device_info():
    """Get detailed device information for experiments"""
    import torch
    
    device_info = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        device_info.update({
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(),
            'memory_allocated': torch.cuda.memory_allocated(),
            'memory_reserved': torch.cuda.memory_reserved(),
            'max_memory_allocated': torch.cuda.max_memory_allocated()
        })
    
    return device_info

def setup_logging(log_file: str = None, level: str = "INFO"):
    """Setup comprehensive logging for experiments"""
    import logging
    import sys
    from datetime import datetime
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

def save_experiment_metadata(experiment_dir: Path, config: dict, model_info: dict = None):
    """Save comprehensive experiment metadata"""
    import json
    import torch
    from datetime import datetime
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'device_info': get_device_info(),
        'pytorch_version': torch.__version__,
        'experiment_directory': str(experiment_dir)
    }
    
    if model_info:
        metadata['model_info'] = model_info
    
    # Save metadata
    with open(experiment_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return metadata

def calculate_flops(model, input_size=(1, 3, 640, 640)):
    """Calculate FLOPs for model complexity analysis"""
    try:
        from ptflops import get_model_complexity_info
        
        macs, params = get_model_complexity_info(
            model, input_size[1:], print_per_layer_stat=False, verbose=False
        )
        
        return {
            'macs': macs,
            'params': params,
            'input_size': input_size
        }
    except ImportError:
        warnings.warn("ptflops not available. Install with: pip install ptflops")
        return None

def create_model_summary(model, input_size=(1, 3, 640, 640)):
    """Create comprehensive model summary"""
    import torch
    
    # Basic model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'input_size': input_size
    }
    
    # Add FLOPs if available
    flops_info = calculate_flops(model, input_size)
    if flops_info:
        summary.update(flops_info)
    
    return summary

def compare_model_summaries(summaries: dict):
    """Compare multiple model summaries"""
    comparison = {}
    
    for model_name, summary in summaries.items():
        comparison[model_name] = {
            'params': summary['total_parameters'],
            'size_mb': summary['model_size_mb'],
            'trainable_params': summary['trainable_parameters']
        }
        
        if 'macs' in summary:
            comparison[model_name]['macs'] = summary['macs']
    
    return comparison

def export_model_for_deployment(model, export_path: str, input_size=(1, 3, 640, 640)):
    """Export model in various formats for deployment"""
    import torch
    from pathlib import Path
    
    export_path = Path(export_path)
    export_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    dummy_input = torch.randn(input_size)
    
    exports = {}
    
    # PyTorch Script
    try:
        scripted_model = torch.jit.trace(model, dummy_input)
        script_path = export_path / 'model_scripted.pt'
        scripted_model.save(script_path)
        exports['torchscript'] = str(script_path)
    except Exception as e:
        warnings.warn(f"TorchScript export failed: {e}")
    
    # ONNX
    try:
        import torch.onnx
        onnx_path = export_path / 'model.onnx'
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        exports['onnx'] = str(onnx_path)
    except Exception as e:
        warnings.warn(f"ONNX export failed: {e}")
    
    # TensorRT (if available)
    try:
        import tensorrt as trt
        # TensorRT export would go here
        exports['tensorrt'] = "Not implemented"
    except ImportError:
        pass
    
    return exports

def create_research_report(experiment_results: dict, output_path: str = "research_report.md"):
    """Generate comprehensive research report"""
    from datetime import datetime
    
    report_lines = [
        "# CBAM-STN-TPS-YOLO Research Report\n",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "## Experiment Overview\n",
        f"Total experiments conducted: {len(experiment_results)}\n\n"
    ]
    
    # Add results for each experiment
    for exp_name, results in experiment_results.items():
        report_lines.extend([
            f"### {exp_name.replace('_', ' ').title()}\n",
            f"- **Configuration**: {results.get('config', 'N/A')}\n",
            f"- **Best mAP**: {results.get('best_mAP', 'N/A'):.4f}\n",
            f"- **Final Metrics**: {results.get('final_metrics', {})}\n\n"
        ])
    
    # Save report
    with open(output_path, 'w') as f:
        f.writelines(report_lines)
    
    return output_path

# Convenience functions for common tasks
def quick_evaluate(model, data_loader, device, class_names, save_dir="results"):
    """Quick model evaluation with standard metrics"""
    evaluator = ModelEvaluator(model, data_loader, device, class_names)
    return evaluator.evaluate_model(save_dir=save_dir)

def quick_visualize(image, predictions, targets=None, class_names=None, save_path=None):
    """Quick prediction visualization"""
    visualizer = Visualizer(class_names or ['class_0', 'class_1', 'class_2'])
    return visualizer.plot_predictions(image, predictions, targets, save_path)

def setup_experiment(experiment_name: str, config_path: str = None, base_dir: str = "experiments"):
    """Complete experiment setup with all utilities"""
    # Create experiment directory
    exp_dir = setup_experiment_directory(experiment_name, base_dir)
    
    # Setup logging
    logger = setup_logging(log_file=exp_dir / 'logs' / 'experiment.log')
    
    # Load and validate config
    if config_path:
        config = load_and_validate_config(config_path)
    else:
        config = create_config_template()
    
    # Save experiment metadata
    metadata = save_experiment_metadata(exp_dir, config)
    
    logger.info(f"Experiment '{experiment_name}' setup completed")
    logger.info(f"Experiment directory: {exp_dir}")
    
    return {
        'experiment_dir': exp_dir,
        'config': config,
        'metadata': metadata,
        'logger': logger
    }

# Export all public components
__all__ = [
    # Core classes
    'Visualizer', 'ModelEvaluator', 'ConfigValidator',
    
    # Visualization functions
    'plot_training_curves', 'visualize_predictions', 'plot_attention_maps',
    'visualize_transformations', 'create_comparison_plots', 'save_experiment_plots',
    
    # Evaluation functions
    'evaluate_model', 'benchmark_models', 'calculate_model_complexity',
    'analyze_failure_cases', 'generate_evaluation_report',
    
    # Configuration functions
    'load_and_validate_config', 'validate_experiment_config',
    'create_config_template', 'merge_configs',
    
    # Utility functions
    'setup_experiment_directory', 'get_device_info', 'setup_logging',
    'save_experiment_metadata', 'calculate_flops', 'create_model_summary',
    'compare_model_summaries', 'export_model_for_deployment',
    'create_research_report',
    
    # Convenience functions
    'quick_evaluate', 'quick_visualize', 'setup_experiment'
]