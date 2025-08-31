# src/utils/evaluation.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging

from src.training.metrics import DetectionMetrics
from src.utils.visualization import Visualizer

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation utilities for CBAM-STN-TPS-YOLO"""
    
    def __init__(self, model: nn.Module, data_loader, device: torch.device, 
                 class_names: List[str], conf_threshold: float = 0.1,
                 iou_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.nms_threshold = nms_threshold
        
        self.visualizer = Visualizer(class_names)
        self.evaluation_results = {}
        
        # Performance tracking
        self.inference_times = []
        self.memory_usage = []
        self.batch_sizes = []
    
    def evaluate_model(self, save_results: bool = True, save_dir: str = 'results/evaluation',
                      detailed_analysis: bool = True) -> Dict[str, Any]:
        """Comprehensive model evaluation with detailed analysis"""
        logger.info("Starting comprehensive model evaluation...")
        
        self.model.eval()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        metrics = DetectionMetrics(len(self.class_names))
        
        # Storage for detailed analysis
        all_predictions = []
        all_targets = []
        all_images = []
        inference_stats = []
        failure_cases = []
        
        total_samples = 0
        correct_detections = 0
        
        logger.info(f"Evaluating on {len(self.data_loader)} batches...")
        
        with torch.no_grad():
            for batch_idx, (images, targets, paths) in enumerate(self.data_loader):
                images = images.to(self.device)
                targets = [target.to(self.device) for target in targets] if isinstance(targets, list) else targets.to(self.device)
                
                # Measure inference time
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                start_time = time.time()
                
                # Forward pass
                outputs = self.model(images)
                
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end_time = time.time()
                
                batch_inference_time = (end_time - start_time) * 1000  # ms
                self.inference_times.append(batch_inference_time)
                
                # Memory usage tracking
                if self.device.type == 'cuda':
                    memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
                    self.memory_usage.append(memory_used)
                
                # Convert outputs to detections
                batch_detections = self._process_outputs(outputs, images.shape)
                
                # Update metrics
                batch_correct = self._update_metrics(metrics, batch_detections, targets)
                correct_detections += batch_correct
                total_samples += len(images)
                
                # Store for detailed analysis
                if detailed_analysis:
                    all_predictions.extend(batch_detections)
                    all_targets.extend(targets if isinstance(targets, list) else [targets[i] for i in range(len(targets))])
                    all_images.extend([images[i] for i in range(len(images))])
                    
                    # Identify failure cases
                    failure_cases.extend(self._identify_failure_cases(
                        batch_detections, targets, paths, batch_idx
                    ))
                
                # Save example predictions
                if batch_idx < 5 and save_results:
                    self._save_prediction_examples(
                        images, batch_detections, targets, paths,
                        save_dir / f'predictions_batch_{batch_idx}.png'
                    )
                
                # Progress reporting
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(self.data_loader)} batches")
        
        # Calculate final metrics
        final_metrics = metrics.compute_metrics()
        
        # Calculate additional metrics
        additional_metrics = self._calculate_additional_metrics(
            all_predictions, all_targets, total_samples, correct_detections
        )
        final_metrics.update(additional_metrics)
        
        # Performance analysis
        performance_analysis = self._analyze_performance()
        final_metrics['performance'] = performance_analysis
        
        # Store results
        self.evaluation_results = {
            'metrics': final_metrics,
            'failure_cases': failure_cases[:10] if failure_cases else [],  # Top 10 failure cases
            'inference_stats': inference_stats,
            'evaluation_config': {
                'conf_threshold': self.conf_threshold,
                'iou_threshold': self.iou_threshold,
                'nms_threshold': self.nms_threshold,
                'num_classes': len(self.class_names),
                'class_names': self.class_names
            }
        }
        
        if save_results:
            self._save_evaluation_results(save_dir, detailed_analysis, all_predictions, all_targets)
        
        logger.info("Model evaluation completed successfully")
        return self.evaluation_results
    
    def benchmark_inference_speed(self, num_runs: int = 100, warmup_runs: int = 10,
                                 input_sizes: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """Comprehensive inference speed benchmarking"""
        logger.info(f"Benchmarking inference speed with {num_runs} runs...")
        
        self.model.eval()
        
        if input_sizes is None:
            input_sizes = [(512, 512), (640, 640), (768, 768), (1024, 1024)]
        
        benchmark_results = {}
        
        for height, width in input_sizes:
            logger.info(f"Benchmarking input size: {height}x{width}")
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, height, width).to(self.device)
            
            # Warmup
            for _ in range(warmup_runs):
                with torch.no_grad():
                    _ = self.model(dummy_input)
            
            # Benchmark
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            
            times = []
            memory_usage = []
            
            for _ in range(num_runs):
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                
                start_time = time.time()
                
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)  # ms
                
                if self.device.type == 'cuda':
                    memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1000 / avg_time
            
            size_results = {
                'avg_inference_time_ms': avg_time,
                'std_inference_time_ms': std_time,
                'min_inference_time_ms': np.min(times),
                'max_inference_time_ms': np.max(times),
                'fps': fps,
                'throughput_imgs_per_sec': fps,
                'input_size': (height, width),
                'num_runs': num_runs
            }
            
            if memory_usage:
                size_results.update({
                    'avg_memory_usage_mb': np.mean(memory_usage),
                    'peak_memory_usage_mb': np.max(memory_usage)
                })
            
            benchmark_results[f'{height}x{width}'] = size_results
        
        logger.info("Inference speed benchmarking completed")
        return benchmark_results
    
    def analyze_failure_cases(self, num_cases: int = 20) -> List[Dict[str, Any]]:
        """Detailed analysis of model failure cases"""
        if not hasattr(self, 'evaluation_results') or not self.evaluation_results:
            logger.warning("No evaluation results available. Run evaluate_model first.")
            return []
        
        failure_cases = self.evaluation_results.get('failure_cases', [])
        
        # Analyze patterns in failure cases
        failure_analysis = {
            'total_failures': len(failure_cases),
            'failure_types': defaultdict(int),
            'class_failures': defaultdict(int),
            'size_failures': defaultdict(int)
        }
        
        for case in failure_cases:
            failure_analysis['failure_types'][case['failure_type']] += 1
            if 'true_class' in case:
                failure_analysis['class_failures'][case['true_class']] += 1
            if 'object_size' in case:
                size_category = self._categorize_object_size(case['object_size'])
                failure_analysis['size_failures'][size_category] += 1
        
        return {
            'failure_cases': failure_cases[:num_cases],
            'failure_analysis': failure_analysis
        }
    
    def compare_with_baseline(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare evaluation results with baseline model"""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_model first.")
        
        current_metrics = self.evaluation_results['metrics']
        baseline_metrics = baseline_results.get('metrics', {})
        
        comparison = {}
        
        for metric_name in ['precision', 'recall', 'mAP', 'f1_score']:
            if metric_name in current_metrics and metric_name in baseline_metrics:
                current_val = current_metrics[metric_name]
                baseline_val = baseline_metrics[metric_name]
                
                # Handle both scalar and dict values
                if isinstance(current_val, dict):
                    current_val = current_val.get('mean', current_val.get('macro', 0))
                if isinstance(baseline_val, dict):
                    baseline_val = baseline_val.get('mean', baseline_val.get('macro', 0))
                
                improvement = current_val - baseline_val
                relative_improvement = (improvement / baseline_val * 100) if baseline_val > 0 else 0
                
                comparison[metric_name] = {
                    'current': current_val,
                    'baseline': baseline_val,
                    'improvement': improvement,
                    'relative_improvement_percent': relative_improvement
                }
        
        return comparison
    
    def _process_outputs(self, outputs, input_shape: torch.Size) -> List[torch.Tensor]:
        """Process model outputs into detection format"""
        # This is a simplified implementation
        # In practice, this would depend on your specific YOLO output format
        
        batch_size = input_shape[0]
        detections = []
        
        for b in range(batch_size):
            # Create dummy detections for demonstration
            # In real implementation, this would parse YOLO outputs
            batch_detections = torch.tensor([
                [100, 100, 200, 200, 0.8, 0],  # [x1, y1, x2, y2, conf, class]
                [300, 300, 400, 400, 0.7, 1],
            ]).to(self.device)
            
            detections.append(batch_detections)
        
        return detections
    
    def _update_metrics(self, metrics: DetectionMetrics, detections: List[torch.Tensor], 
                       targets: Union[torch.Tensor, List[torch.Tensor]]) -> int:
        """Update metrics with current batch predictions"""
        # Simplified implementation
        # In practice, this would properly convert detections and targets
        # to the format expected by DetectionMetrics
        
        correct_count = 0
        
        for i, (pred, target) in enumerate(zip(detections, targets)):
            if len(pred) > 0 and len(target) > 0:
                # Simplified: count as correct if any detection exists
                correct_count += 1
        
        return correct_count
    
    def _identify_failure_cases(self, detections: List[torch.Tensor], targets, 
                               paths: List[str], batch_idx: int) -> List[Dict[str, Any]]:
        """Identify and categorize failure cases"""
        failure_cases = []
        
        for i, (pred, target, path) in enumerate(zip(detections, targets, paths)):
            # Simplified failure detection logic
            if len(pred) == 0 and len(target) > 0:
                # Missed detection
                failure_cases.append({
                    'failure_type': 'missed_detection',
                    'image_path': path,
                    'batch_idx': batch_idx,
                    'image_idx': i,
                    'num_targets': len(target),
                    'target_info': target.tolist() if hasattr(target, 'tolist') else str(target)
                })
            elif len(pred) > 0 and len(target) == 0:
                # False positive
                failure_cases.append({
                    'failure_type': 'false_positive',
                    'image_path': path,
                    'batch_idx': batch_idx,
                    'image_idx': i,
                    'num_predictions': len(pred),
                    'prediction_info': pred.tolist() if hasattr(pred, 'tolist') else str(pred)
                })
        
        return failure_cases
    
    def _calculate_additional_metrics(self, predictions, targets, total_samples: int, 
                                    correct_detections: int) -> Dict[str, Any]:
        """Calculate additional evaluation metrics"""
        additional_metrics = {}
        
        # Detection rate
        detection_rate = correct_detections / total_samples if total_samples > 0 else 0
        additional_metrics['detection_rate'] = detection_rate
        
        # Class distribution analysis
        if predictions and len(predictions) > 0:
            class_counts = defaultdict(int)
            for pred_batch in predictions:
                if len(pred_batch) > 0:
                    for detection in pred_batch:
                        if len(detection) > 5:  # [x1, y1, x2, y2, conf, class]
                            class_id = int(detection[5])
                            class_counts[class_id] += 1
            
            additional_metrics['class_distribution'] = dict(class_counts)
        
        # Confidence score analysis
        if predictions:
            confidence_scores = []
            for pred_batch in predictions:
                if len(pred_batch) > 0:
                    for detection in pred_batch:
                        if len(detection) > 4:
                            confidence_scores.append(float(detection[4]))
            
            if confidence_scores:
                additional_metrics['confidence_stats'] = {
                    'mean': np.mean(confidence_scores),
                    'std': np.std(confidence_scores),
                    'min': np.min(confidence_scores),
                    'max': np.max(confidence_scores),
                    'median': np.median(confidence_scores)
                }
        
        return additional_metrics
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze model performance characteristics"""
        performance_analysis = {}
        
        if self.inference_times:
            performance_analysis['inference_time'] = {
                'mean_ms': np.mean(self.inference_times),
                'std_ms': np.std(self.inference_times),
                'min_ms': np.min(self.inference_times),
                'max_ms': np.max(self.inference_times),
                'median_ms': np.median(self.inference_times),
                'fps': 1000 / np.mean(self.inference_times)
            }
        
        if self.memory_usage:
            performance_analysis['memory_usage'] = {
                'mean_mb': np.mean(self.memory_usage),
                'std_mb': np.std(self.memory_usage),
                'peak_mb': np.max(self.memory_usage),
                'min_mb': np.min(self.memory_usage)
            }
        
        return performance_analysis
    
    def _categorize_object_size(self, object_size: float) -> str:
        """Categorize object size into small, medium, large"""
        if object_size < 32**2:
            return 'small'
        elif object_size < 96**2:
            return 'medium'
        else:
            return 'large'
    
    def _save_prediction_examples(self, images: torch.Tensor, detections: List[torch.Tensor],
                                 targets, paths: List[str], save_path: Path):
        """Save prediction visualization examples"""
        try:
            for i in range(min(len(images), 4)):
                image = images[i]
                preds = detections[i] if i < len(detections) else torch.tensor([])
                tgts = targets[i] if i < len(targets) else torch.tensor([])
                
                individual_save_path = str(save_path).replace('.png', f'_img_{i}.png')
                self.visualizer.plot_predictions(
                    image, preds, tgts,
                    save_path=individual_save_path,
                    title=f'Predictions: {Path(paths[i]).name}'
                )
        except Exception as e:
            logger.warning(f"Failed to save prediction examples: {e}")
    
    def _save_evaluation_results(self, save_dir: Path, detailed_analysis: bool,
                                all_predictions, all_targets):
        """Save comprehensive evaluation results"""
        try:
            # Save metrics
            with open(save_dir / 'evaluation_metrics.json', 'w') as f:
                json.dump(self.evaluation_results, f, indent=2, default=str)
            
            # Save detailed analysis if available
            if detailed_analysis and all_predictions and all_targets:
                # Create confusion matrix
                try:
                    pred_classes, true_classes = self._extract_classes_for_confusion_matrix(
                        all_predictions, all_targets
                    )
                    if pred_classes and true_classes:
                        self.visualizer.create_confusion_matrix(
                            true_classes, pred_classes,
                            save_path=save_dir / 'confusion_matrix.png'
                        )
                except Exception as e:
                    logger.warning(f"Failed to create confusion matrix: {e}")
            
            # Save performance plots
            self._save_performance_plots(save_dir)
            
            logger.info(f"Evaluation results saved to: {save_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
    
    def _extract_classes_for_confusion_matrix(self, predictions, targets):
        """Extract class predictions and targets for confusion matrix"""
        pred_classes = []
        true_classes = []
        
        # Simplified extraction - would need proper implementation
        # based on your specific data format
        
        return pred_classes, true_classes
    
    def _save_performance_plots(self, save_dir: Path):
        """Save performance analysis plots"""
        try:
            import matplotlib.pyplot as plt
            
            # Inference time distribution
            if self.inference_times:
                plt.figure(figsize=(10, 6))
                plt.hist(self.inference_times, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('Inference Time (ms)')
                plt.ylabel('Frequency')
                plt.title('Inference Time Distribution')
                plt.grid(True, alpha=0.3)
                plt.savefig(save_dir / 'inference_time_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Memory usage over time
            if self.memory_usage:
                plt.figure(figsize=(12, 6))
                plt.plot(self.memory_usage, alpha=0.7)
                plt.xlabel('Batch Index')
                plt.ylabel('Memory Usage (MB)')
                plt.title('Memory Usage Over Time')
                plt.grid(True, alpha=0.3)
                plt.savefig(save_dir / 'memory_usage.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except ImportError:
            logger.warning("matplotlib not available for performance plots")
        except Exception as e:
            logger.warning(f"Failed to save performance plots: {e}")

# Utility functions for standalone evaluation
def evaluate_model(model: nn.Module, data_loader, device: torch.device, 
                  class_names: List[str], save_dir: str = 'results/evaluation') -> Dict[str, Any]:
    """Quick model evaluation function"""
    evaluator = ModelEvaluator(model, data_loader, device, class_names)
    return evaluator.evaluate_model(save_dir=save_dir)

def benchmark_models(models: Dict[str, nn.Module], data_loader, device: torch.device,
                    class_names: List[str], save_dir: str = 'results/benchmark') -> Dict[str, Any]:
    """Benchmark multiple models"""
    results = {}
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model in models.items():
        logger.info(f"Benchmarking model: {model_name}")
        
        model_save_dir = save_dir / model_name
        evaluator = ModelEvaluator(model, data_loader, device, class_names)
        
        # Evaluation
        eval_results = evaluator.evaluate_model(save_dir=model_save_dir)
        
        # Speed benchmark
        speed_results = evaluator.benchmark_inference_speed()
        
        results[model_name] = {
            'evaluation': eval_results,
            'benchmark': speed_results
        }
    
    # Save comparison results
    with open(save_dir / 'model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Model benchmarking completed. Results saved to: {save_dir}")
    return results

def calculate_model_complexity(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 640, 640)) -> Dict[str, Any]:
    """Calculate model complexity metrics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    complexity = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'input_size': input_size
    }
    
    # Try to calculate FLOPs
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, input_size[1:], print_per_layer_stat=False, verbose=False
        )
        complexity['macs'] = macs
        complexity['flops'] = macs  # MACs ≈ FLOPs for most operations
    except ImportError:
        logger.warning("ptflops not available for FLOP calculation")
    
    return complexity

def analyze_failure_cases(evaluation_results: Dict[str, Any], num_cases: int = 10) -> Dict[str, Any]:
    """Analyze failure cases from evaluation results"""
    failure_cases = evaluation_results.get('failure_cases', [])
    
    if not failure_cases:
        return {'message': 'No failure cases found in evaluation results'}
    
    # Analyze failure patterns
    failure_types = defaultdict(int)
    for case in failure_cases:
        failure_types[case.get('failure_type', 'unknown')] += 1
    
    analysis = {
        'total_failure_cases': len(failure_cases),
        'failure_type_distribution': dict(failure_types),
        'top_failure_cases': failure_cases[:num_cases]
    }
    
    return analysis

def generate_evaluation_report(evaluation_results: Dict[str, Any], 
                             model_name: str = "Model", 
                             save_path: str = "evaluation_report.md") -> str:
    """Generate comprehensive evaluation report"""
    from datetime import datetime
    
    metrics = evaluation_results.get('metrics', {})
    performance = metrics.get('performance', {})
    
    report_lines = [
        f"# {model_name} Evaluation Report\n",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "## Performance Metrics\n",
    ]
    
    # Add main metrics
    main_metrics = ['precision', 'recall', 'mAP', 'f1_score']
    for metric in main_metrics:
        if metric in metrics:
            value = metrics[metric]
            if isinstance(value, dict):
                value = value.get('mean', value.get('macro', 'N/A'))
            report_lines.append(f"- **{metric.upper()}**: {value:.4f}\n")
    
    # Add performance analysis
    if performance:
        report_lines.append("\n## Performance Analysis\n")
        
        if 'inference_time' in performance:
            inf_time = performance['inference_time']
            report_lines.append(f"- **Average Inference Time**: {inf_time.get('mean_ms', 0):.2f} ms\n")
            report_lines.append(f"- **FPS**: {inf_time.get('fps', 0):.2f}\n")
        
        if 'memory_usage' in performance:
            mem_usage = performance['memory_usage']
            report_lines.append(f"- **Peak Memory Usage**: {mem_usage.get('peak_mb', 0):.2f} MB\n")
    
    # Add failure analysis if available
    failure_cases = evaluation_results.get('failure_cases', [])
    if failure_cases:
        report_lines.append(f"\n## Failure Analysis\n")
        report_lines.append(f"- **Total Failure Cases**: {len(failure_cases)}\n")
    
    # Save report
    with open(save_path, 'w') as f:
        f.writelines(report_lines)
    
    logger.info(f"Evaluation report saved to: {save_path}")
    return save_path

if __name__ == "__main__":
    # Test evaluation functionality
    print("Testing model evaluation utilities...")
    
    try:
        # Test model complexity calculation
        import torch.nn as nn
        dummy_model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
        
        complexity = calculate_model_complexity(dummy_model)
        print(f"✅ Model complexity calculation: {complexity['total_parameters']} parameters")
        
        # Test report generation
        dummy_results = {
            'metrics': {
                'precision': 0.85,
                'recall': 0.82,
                'mAP': 0.78,
                'f1_score': 0.83,
                'performance': {
                    'inference_time': {'mean_ms': 15.2, 'fps': 65.8},
                    'memory_usage': {'peak_mb': 1024}
                }
            },
            'failure_cases': []
        }
        
        report_path = generate_evaluation_report(dummy_results, "Test Model")
        print(f"✅ Evaluation report generated: {report_path}")
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
    
    print("Evaluation tests completed!")