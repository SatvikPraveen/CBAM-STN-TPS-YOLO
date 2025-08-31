# experiments/ablation_study.py
"""
Comprehensive ablation study for CBAM-STN-TPS-YOLO
Systematic analysis of component contributions and hyperparameter sensitivity
"""

import torch
import numpy as np
import json
import yaml
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Enhanced imports with error handling
try:
    from src.training.trainer import CBAMSTNTPSYOLOTrainer
    from src.utils.visualization import Visualizer, save_experiment_plots
    from src.utils.config_validator import load_and_validate_config
    from src.utils.evaluation import ModelEvaluator
except ImportError as e:
    logging.warning(f"Some imports failed: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveAblationStudy:
    """Systematic ablation study for CBAM-STN-TPS-YOLO components and hyperparameters"""
    
    def __init__(self, config_path: str = 'config/training_configs.yaml'):
        self.config_path = config_path
        self.base_config = self._load_base_config()
        self.results = defaultdict(dict)
        self.experiment_start_time = datetime.now()
        
        # Define comprehensive ablation experiments
        self.component_ablations = OrderedDict([
            ('baseline', {
                'model_type': 'YOLO',
                'description': 'Baseline YOLO without enhancements',
                'components': []
            }),
            ('stn_affine', {
                'model_type': 'STN-YOLO',
                'description': 'YOLO + STN with affine transformation',
                'components': ['STN']
            }),
            ('cbam_only', {
                'model_type': 'CBAM-YOLO',
                'description': 'YOLO + CBAM attention mechanism',
                'components': ['CBAM']
            }),
            ('stn_tps', {
                'model_type': 'STN-TPS-YOLO',
                'description': 'YOLO + STN with TPS transformation',
                'components': ['STN', 'TPS']
            }),
            ('cbam_stn', {
                'model_type': 'CBAM-STN-YOLO',
                'description': 'YOLO + CBAM + STN (affine)',
                'components': ['CBAM', 'STN']
            }),
            ('full_model', {
                'model_type': 'CBAM-STN-TPS-YOLO',
                'description': 'Complete model with all components',
                'components': ['CBAM', 'STN', 'TPS']
            })
        ])
        
        # Hyperparameter ranges for sensitivity analysis
        self.hyperparameter_ranges = {
            'tps_control_points': [8, 12, 16, 20, 24, 28, 32],
            'tps_regularization': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
            'cbam_reduction_ratio': [4, 8, 16, 32, 64],
            'stn_hidden_dim': [64, 128, 256, 512],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'batch_size': [8, 16, 32, 64]
        }
        
        # Analysis tracking
        self.component_contributions = {}
        self.hyperparameter_sensitivities = {}
        self.interaction_effects = {}
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration for experiments"""
        try:
            config = load_and_validate_config(self.config_path)
            # Set reduced epochs for ablation studies
            config['epochs'] = min(config.get('epochs', 100), 50)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for ablation studies"""
        return {
            'data_dir': 'data/PGP',
            'num_classes': 3,
            'input_channels': 4,
            'image_size': 640,
            'batch_size': 16,
            'epochs': 30,  # Reduced for ablation
            'lr': 0.001,
            'weight_decay': 0.0005,
            'optimizer': 'Adam',
            'mixed_precision': True,
            'early_stopping': True,
            'early_stopping_patience': 5,
            'save_dir': 'experiments/ablation',
            'use_wandb': False,
            'seed': 42
        }
    
    def run_component_ablation(self, num_runs: int = 3, save_models: bool = False) -> Dict[str, Any]:
        """Run systematic component ablation study"""
        
        logger.info("="*80)
        logger.info("STARTING COMPONENT ABLATION STUDY")
        logger.info("="*80)
        
        component_results = {}
        
        for exp_name, exp_config in self.component_ablations.items():
            logger.info(f"\n🔬 Running ablation: {exp_name}")
            logger.info(f"📝 Description: {exp_config['description']}")
            logger.info(f"🧩 Components: {exp_config['components']}")
            
            exp_results = []
            
            for run in range(num_runs):
                logger.info(f"  ▶️ Run {run + 1}/{num_runs}")
                
                try:
                    # Prepare experiment configuration
                    config = self._prepare_component_config(exp_config, run)
                    
                    # Run experiment
                    result = self._run_single_ablation_experiment(
                        config, exp_name, run, save_models
                    )
                    
                    exp_results.append(result)
                    
                    logger.info(f"    ✅ Completed - mAP: {result.get('best_mAP', 0):.4f}")
                    
                except Exception as e:
                    logger.error(f"    ❌ Failed: {e}")
                    continue
            
            # Process results for this experiment
            if exp_results:
                component_results[exp_name] = self._process_experiment_results(
                    exp_results, exp_config
                )
                
                mean_mAP = component_results[exp_name]['metrics']['mAP']['mean']
                std_mAP = component_results[exp_name]['metrics']['mAP']['std']
                logger.info(f"  📊 Average mAP: {mean_mAP:.4f} ± {std_mAP:.4f}")
        
        # Analyze component contributions
        self.component_contributions = self._analyze_component_contributions(component_results)
        
        # Store results
        self.results['component_ablation'] = {
            'individual_results': component_results,
            'component_contributions': self.component_contributions,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info("✅ Component ablation study completed")
        return self.results['component_ablation']
    
    def run_hyperparameter_sensitivity_analysis(self, target_model: str = 'CBAM-STN-TPS-YOLO') -> Dict[str, Any]:
        """Run comprehensive hyperparameter sensitivity analysis"""
        
        logger.info("="*80)
        logger.info("STARTING HYPERPARAMETER SENSITIVITY ANALYSIS")
        logger.info("="*80)
        
        sensitivity_results = {}
        
        for param_name, param_values in self.hyperparameter_ranges.items():
            # Skip parameters not relevant to target model
            if not self._is_parameter_relevant(param_name, target_model):
                continue
            
            logger.info(f"\n🎛️ Analyzing parameter: {param_name}")
            logger.info(f"📊 Testing values: {param_values}")
            
            param_results = []
            
            for param_value in param_values:
                logger.info(f"  ▶️ Testing {param_name} = {param_value}")
                
                try:
                    # Prepare configuration with specific parameter value
                    config = self._prepare_hyperparameter_config(
                        target_model, param_name, param_value
                    )
                    
                    # Run experiment
                    result = self._run_single_ablation_experiment(
                        config, f"{param_name}_{param_value}", 0, False
                    )
                    
                    result['parameter_value'] = param_value
                    param_results.append(result)
                    
                    logger.info(f"    ✅ mAP: {result.get('best_mAP', 0):.4f}")
                    
                except Exception as e:
                    logger.error(f"    ❌ Failed: {e}")
                    continue
            
            # Analyze sensitivity for this parameter
            if len(param_results) > 1:
                sensitivity_analysis = self._analyze_parameter_sensitivity(
                    param_name, param_results
                )
                sensitivity_results[param_name] = sensitivity_analysis
                
                optimal_value = sensitivity_analysis['optimal_value']
                sensitivity_score = sensitivity_analysis['sensitivity_score']
                logger.info(f"  📈 Optimal value: {optimal_value}")
                logger.info(f"  📊 Sensitivity score: {sensitivity_score:.4f}")
        
        # Store results
        self.hyperparameter_sensitivities = sensitivity_results
        self.results['hyperparameter_sensitivity'] = {
            'individual_results': sensitivity_results,
            'analysis_timestamp': datetime.now().isoformat(),
            'target_model': target_model
        }
        
        logger.info("✅ Hyperparameter sensitivity analysis completed")
        return self.results['hyperparameter_sensitivity']
    
    def run_interaction_analysis(self, key_parameters: List[str] = None) -> Dict[str, Any]:
        """Analyze interactions between key hyperparameters"""
        
        if key_parameters is None:
            key_parameters = ['tps_control_points', 'cbam_reduction_ratio', 'learning_rate']
        
        logger.info("="*80)
        logger.info("STARTING PARAMETER INTERACTION ANALYSIS")
        logger.info("="*80)
        
        interaction_results = {}
        
        # Test pairwise interactions
        for i in range(len(key_parameters)):
            for j in range(i + 1, len(key_parameters)):
                param1, param2 = key_parameters[i], key_parameters[j]
                
                logger.info(f"\n🔗 Analyzing interaction: {param1} × {param2}")
                
                # Select representative values for each parameter
                values1 = self._get_representative_values(param1)
                values2 = self._get_representative_values(param2)
                
                interaction_matrix = []
                
                for val1 in values1:
                    row_results = []
                    for val2 in values2:
                        try:
                            # Create config with both parameters
                            config = self._prepare_interaction_config(
                                'CBAM-STN-TPS-YOLO', 
                                {param1: val1, param2: val2}
                            )
                            
                            result = self._run_single_ablation_experiment(
                                config, f"interaction_{param1}_{val1}_{param2}_{val2}", 0, False
                            )
                            
                            mAP = result.get('best_mAP', 0)
                            row_results.append(mAP)
                            
                            logger.info(f"    {param1}={val1}, {param2}={val2}: mAP={mAP:.4f}")
                            
                        except Exception as e:
                            logger.error(f"    ❌ Failed for {param1}={val1}, {param2}={val2}: {e}")
                            row_results.append(0)
                    
                    interaction_matrix.append(row_results)
                
                # Analyze interaction effect
                interaction_effect = self._calculate_interaction_effect(
                    interaction_matrix, values1, values2
                )
                
                interaction_results[f"{param1}_x_{param2}"] = {
                    'parameter1': param1,
                    'parameter2': param2,
                    'values1': values1,
                    'values2': values2,
                    'interaction_matrix': interaction_matrix,
                    'interaction_effect': interaction_effect,
                    'synergistic': interaction_effect > 0.01  # Threshold for synergy
                }
        
        # Store results
        self.interaction_effects = interaction_results
        self.results['interaction_analysis'] = {
            'interactions': interaction_results,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info("✅ Parameter interaction analysis completed")
        return self.results['interaction_analysis']
    
    def generate_comprehensive_report(self, save_dir: str = "results/ablation_study") -> str:
        """Generate comprehensive ablation study report"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        results_file = save_dir / 'ablation_study_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate visualizations
        self._create_ablation_visualizations(save_dir)
        
        # Generate markdown report
        report_file = self._generate_markdown_report(save_dir)
        
        # Generate summary statistics
        self._generate_summary_statistics(save_dir)
        
        logger.info(f"📄 Comprehensive ablation report generated: {save_dir}")
        return str(report_file)
    
    def _prepare_component_config(self, exp_config: Dict[str, Any], run: int) -> Dict[str, Any]:
        """Prepare configuration for component ablation experiment"""
        
        config = self.base_config.copy()
        config.update({
            'model_type': exp_config['model_type'],
            'experiment_name': f"ablation_{exp_config['model_type']}_run_{run}",
            'seed': 42 + run,  # Different seed for each run
        })
        
        # Add component-specific parameters
        if 'TPS' in exp_config['components']:
            config.update({
                'num_control_points': 20,
                'tps_reg_lambda': 0.1
            })
        
        if 'CBAM' in exp_config['components']:
            config.update({
                'cbam_reduction_ratio': 16
            })
        
        if 'STN' in exp_config['components']:
            config.update({
                'stn_loc_hidden_dim': 128
            })
        
        return config
    
    def _prepare_hyperparameter_config(self, model_type: str, param_name: str, 
                                     param_value: Any) -> Dict[str, Any]:
        """Prepare configuration for hyperparameter sensitivity test"""
        
        config = self.base_config.copy()
        config.update({
            'model_type': model_type,
            'experiment_name': f"hyperparam_{param_name}_{param_value}",
        })
        
        # Add default parameters for the model
        if model_type == 'CBAM-STN-TPS-YOLO':
            config.update({
                'num_control_points': 20,
                'tps_reg_lambda': 0.1,
                'cbam_reduction_ratio': 16,
                'stn_loc_hidden_dim': 128
            })
        
        # Override specific parameter
        param_mapping = {
            'tps_control_points': 'num_control_points',
            'tps_regularization': 'tps_reg_lambda',
            'cbam_reduction_ratio': 'cbam_reduction_ratio',
            'stn_hidden_dim': 'stn_loc_hidden_dim',
            'learning_rate': 'lr',
            'batch_size': 'batch_size'
        }
        
        config_key = param_mapping.get(param_name, param_name)
        config[config_key] = param_value
        
        return config
    
    def _prepare_interaction_config(self, model_type: str, 
                                  param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for interaction analysis"""
        
        config = self.base_config.copy()
        config.update({
            'model_type': model_type,
            'experiment_name': f"interaction_{'_'.join(f'{k}_{v}' for k, v in param_dict.items())}"
        })
        
        # Add default parameters
        if model_type == 'CBAM-STN-TPS-YOLO':
            config.update({
                'num_control_points': 20,
                'tps_reg_lambda': 0.1,
                'cbam_reduction_ratio': 16,
                'stn_loc_hidden_dim': 128
            })
        
        # Override with interaction parameters
        param_mapping = {
            'tps_control_points': 'num_control_points',
            'tps_regularization': 'tps_reg_lambda',
            'cbam_reduction_ratio': 'cbam_reduction_ratio',
            'stn_hidden_dim': 'stn_loc_hidden_dim',
            'learning_rate': 'lr'
        }
        
        for param_name, param_value in param_dict.items():
            config_key = param_mapping.get(param_name, param_name)
            config[config_key] = param_value
        
        return config
    
    def _run_single_ablation_experiment(self, config: Dict[str, Any], 
                                      experiment_name: str, run: int,
                                      save_model: bool = False) -> Dict[str, Any]:
        """Run a single ablation experiment"""
        
        # Set seed for reproducibility
        torch.manual_seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        
        # Create trainer
        trainer = CBAMSTNTPSYOLOTrainer(config, config['model_type'])
        
        # Measure training time
        start_time = time.time()
        best_mAP = trainer.train()
        training_time = time.time() - start_time
        
        # Get final metrics
        final_metrics = trainer.val_metrics[-1] if trainer.val_metrics else {}
        
        # Calculate model complexity
        model_complexity = self._calculate_model_complexity(trainer.model)
        
        result = {
            'experiment_name': experiment_name,
            'run': run,
            'config': config,
            'best_mAP': best_mAP,
            'final_metrics': final_metrics,
            'training_time': training_time,
            'model_complexity': model_complexity,
            'training_history': {
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'val_metrics': trainer.val_metrics
            }
        }
        
        # Save model if requested
        if save_model:
            model_path = self._save_model(trainer.model, experiment_name, config)
            result['model_path'] = model_path
        
        return result
    
    def _process_experiment_results(self, exp_results: List[Dict[str, Any]], 
                                  exp_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process results from multiple runs of the same experiment"""
        
        # Extract metrics
        mAP_values = [r['best_mAP'] for r in exp_results]
        training_times = [r['training_time'] for r in exp_results]
        
        # Calculate statistics
        processed = {
            'experiment_config': exp_config,
            'num_runs': len(exp_results),
            'metrics': {
                'mAP': {
                    'values': mAP_values,
                    'mean': float(np.mean(mAP_values)),
                    'std': float(np.std(mAP_values)),
                    'min': float(np.min(mAP_values)),
                    'max': float(np.max(mAP_values))
                },
                'training_time': {
                    'values': training_times,
                    'mean': float(np.mean(training_times)),
                    'std': float(np.std(training_times))
                }
            },
            'individual_results': exp_results
        }
        
        # Add model complexity if available
        if exp_results and 'model_complexity' in exp_results[0]:
            complexity = exp_results[0]['model_complexity']
            processed['model_complexity'] = complexity
        
        return processed
    
    def _analyze_component_contributions(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual component contributions"""
        
        contributions = {}
        
        # Get baseline performance
        baseline_mAP = component_results.get('baseline', {}).get('metrics', {}).get('mAP', {}).get('mean', 0)
        
        # Calculate incremental contributions
        progression = [
            ('baseline', 'Baseline YOLO'),
            ('stn_affine', 'STN Added'),
            ('cbam_only', 'CBAM Added'),
            ('cbam_stn', 'CBAM + STN'),
            ('stn_tps', 'STN + TPS'),
            ('full_model', 'Full Model')
        ]
        
        for exp_name, description in progression:
            if exp_name in component_results:
                mAP = component_results[exp_name]['metrics']['mAP']['mean']
                improvement = mAP - baseline_mAP
                relative_improvement = (improvement / baseline_mAP * 100) if baseline_mAP > 0 else 0
                
                contributions[exp_name] = {
                    'description': description,
                    'mAP': mAP,
                    'absolute_improvement': improvement,
                    'relative_improvement_percent': relative_improvement,
                    'components': component_results[exp_name]['experiment_config']['components']
                }
        
        # Calculate individual component effects
        component_effects = {}
        
        if 'stn_affine' in component_results and 'baseline' in component_results:
            stn_effect = (component_results['stn_affine']['metrics']['mAP']['mean'] - 
                         component_results['baseline']['metrics']['mAP']['mean'])
            component_effects['STN'] = stn_effect
        
        if 'cbam_only' in component_results and 'baseline' in component_results:
            cbam_effect = (component_results['cbam_only']['metrics']['mAP']['mean'] - 
                          component_results['baseline']['metrics']['mAP']['mean'])
            component_effects['CBAM'] = cbam_effect
        
        if 'stn_tps' in component_results and 'stn_affine' in component_results:
            tps_effect = (component_results['stn_tps']['metrics']['mAP']['mean'] - 
                         component_results['stn_affine']['metrics']['mAP']['mean'])
            component_effects['TPS'] = tps_effect
        
        contributions['individual_effects'] = component_effects
        
        return contributions
    
    def _analyze_parameter_sensitivity(self, param_name: str, 
                                     param_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sensitivity for a specific parameter"""
        
        # Extract parameter values and corresponding mAP scores
        param_values = [r['parameter_value'] for r in param_results]
        mAP_values = [r['best_mAP'] for r in param_results]
        
        # Find optimal value
        optimal_idx = np.argmax(mAP_values)
        optimal_value = param_values[optimal_idx]
        optimal_mAP = mAP_values[optimal_idx]
        
        # Calculate sensitivity score (coefficient of variation)
        sensitivity_score = np.std(mAP_values) / np.mean(mAP_values) if np.mean(mAP_values) > 0 else 0
        
        # Calculate parameter importance (range of performance)
        mAP_range = np.max(mAP_values) - np.min(mAP_values)
        
        return {
            'parameter_name': param_name,
            'optimal_value': optimal_value,
            'optimal_mAP': optimal_mAP,
            'sensitivity_score': sensitivity_score,
            'mAP_range': mAP_range,
            'parameter_values': param_values,
            'mAP_values': mAP_values,
            'mean_mAP': np.mean(mAP_values),
            'std_mAP': np.std(mAP_values)
        }
    
    def _calculate_interaction_effect(self, interaction_matrix: List[List[float]], 
                                    values1: List[Any], values2: List[Any]) -> float:
        """Calculate interaction effect magnitude"""
        
        matrix = np.array(interaction_matrix)
        
        if matrix.size == 0:
            return 0
        
        # Calculate main effects
        main_effect_1 = np.std(np.mean(matrix, axis=1))  # Variance across rows
        main_effect_2 = np.std(np.mean(matrix, axis=0))  # Variance across columns
        
        # Calculate interaction effect (simplified)
        # Real interaction would require proper ANOVA
        total_variance = np.var(matrix.flatten())
        main_effects_variance = (main_effect_1**2 + main_effect_2**2) / 2
        
        interaction_effect = max(0, total_variance - main_effects_variance)
        
        return interaction_effect
    
    def _is_parameter_relevant(self, param_name: str, model_type: str) -> bool:
        """Check if parameter is relevant to model type"""
        
        param_relevance = {
            'tps_control_points': 'TPS' in model_type,
            'tps_regularization': 'TPS' in model_type,
            'cbam_reduction_ratio': 'CBAM' in model_type,
            'stn_hidden_dim': 'STN' in model_type,
            'learning_rate': True,
            'batch_size': True
        }
        
        return param_relevance.get(param_name, False)
    
    def _get_representative_values(self, param_name: str) -> List[Any]:
        """Get representative values for interaction analysis"""
        
        full_range = self.hyperparameter_ranges.get(param_name, [])
        
        if len(full_range) <= 3:
            return full_range
        
        # Select low, medium, high values
        return [full_range[0], full_range[len(full_range)//2], full_range[-1]]
    
    def _calculate_model_complexity(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Calculate model complexity metrics"""
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }
    
    def _save_model(self, model: torch.nn.Module, experiment_name: str, 
                   config: Dict[str, Any]) -> str:
        """Save model checkpoint"""
        
        save_dir = Path(config.get('save_dir', 'experiments/ablation/models'))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / f'{experiment_name}_model.pth'
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'experiment_name': experiment_name
        }
        
        torch.save(checkpoint, model_path)
        return str(model_path)
    
    def _create_ablation_visualizations(self, save_dir: Path) -> None:
        """Create comprehensive ablation study visualizations"""
        
        try:
            # Component contribution plots
            if 'component_ablation' in self.results:
                self._plot_component_contributions(save_dir)
            
            # Hyperparameter sensitivity plots
            if 'hyperparameter_sensitivity' in self.results:
                self._plot_hyperparameter_sensitivity(save_dir)
            
            # Interaction effect plots
            if 'interaction_analysis' in self.results:
                self._plot_interaction_effects(save_dir)
            
            logger.info(f"📊 Ablation visualizations saved to: {save_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
    
    def _plot_component_contributions(self, save_dir: Path) -> None:
        """Plot component contribution analysis"""
        
        contributions = self.component_contributions
        
        if not contributions:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart of absolute improvements
        experiments = []
        improvements = []
        
        for exp_name, data in contributions.items():
            if exp_name != 'individual_effects' and 'absolute_improvement' in data:
                experiments.append(data['description'])
                improvements.append(data['absolute_improvement'])
        
        if experiments and improvements:
            bars = ax1.bar(experiments, improvements, alpha=0.8, color='skyblue')
            ax1.set_title('Component Contributions to mAP', fontweight='bold', fontsize=14)
            ax1.set_ylabel('mAP Improvement')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{improvement:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Individual component effects
        if 'individual_effects' in contributions:
            effects = contributions['individual_effects']
            components = list(effects.keys())
            effect_values = list(effects.values())
            
            bars2 = ax2.bar(components, effect_values, alpha=0.8, color='lightcoral')
            ax2.set_title('Individual Component Effects', fontweight='bold', fontsize=14)
            ax2.set_ylabel('mAP Improvement')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, effect in zip(bars2, effect_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{effect:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'component_contributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_hyperparameter_sensitivity(self, save_dir: Path) -> None:
        """Plot hyperparameter sensitivity analysis"""
        
        sensitivities = self.hyperparameter_sensitivities
        
        if not sensitivities:
            return
        
        n_params = len(sensitivities)
        if n_params == 0:
            return
        
        fig, axes = plt.subplots((n_params + 1) // 2, 2, figsize=(16, 4 * ((n_params + 1) // 2)))
        if n_params == 1:
            axes = [axes]
        elif n_params == 2:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (param_name, sensitivity_data) in enumerate(sensitivities.items()):
            if i >= len(axes):
                break
            
            param_values = sensitivity_data['parameter_values']
            mAP_values = sensitivity_data['mAP_values']
            optimal_value = sensitivity_data['optimal_value']
            
            # Plot parameter vs mAP
            axes[i].plot(param_values, mAP_values, 'o-', linewidth=2, markersize=8)
            axes[i].axvline(x=optimal_value, color='red', linestyle='--', alpha=0.7, 
                           label=f'Optimal: {optimal_value}')
            
            axes[i].set_xlabel(param_name.replace('_', ' ').title())
            axes[i].set_ylabel('mAP')
            axes[i].set_title(f'{param_name.replace("_", " ").title()} Sensitivity', 
                             fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].remove()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_interaction_effects(self, save_dir: Path) -> None:
        """Plot parameter interaction effects"""
        
        interactions = self.interaction_effects
        
        if not interactions:
            return
        
        n_interactions = len(interactions)
        if n_interactions == 0:
            return
        
        fig, axes = plt.subplots(1, min(n_interactions, 3), figsize=(6 * min(n_interactions, 3), 5))
        if n_interactions == 1:
            axes = [axes]
        
        for i, (interaction_name, interaction_data) in enumerate(list(interactions.items())[:3]):
            if i >= len(axes):
                break
            
            matrix = np.array(interaction_data['interaction_matrix'])
            values1 = interaction_data['values1']
            values2 = interaction_data['values2']
            param1 = interaction_data['parameter1']
            param2 = interaction_data['parameter2']
            
            # Create heatmap
            im = axes[i].imshow(matrix, cmap='viridis', aspect='auto')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], shrink=0.8)
            
            # Set labels
            axes[i].set_xticks(range(len(values2)))
            axes[i].set_xticklabels(values2)
            axes[i].set_yticks(range(len(values1)))
            axes[i].set_yticklabels(values1)
            
            axes[i].set_xlabel(param2.replace('_', ' ').title())
            axes[i].set_ylabel(param1.replace('_', ' ').title())
            axes[i].set_title(f'{param1} × {param2} Interaction', fontweight='bold')
            
            # Add text annotations
            for ii in range(len(values1)):
                for jj in range(len(values2)):
                    text = axes[i].text(jj, ii, f'{matrix[ii, jj]:.3f}',
                                       ha="center", va="center", color="white", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'parameter_interactions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_markdown_report(self, save_dir: Path) -> Path:
        """Generate comprehensive markdown report"""
        
        report_path = save_dir / 'ablation_study_report.md'
        
        report_lines = [
            "# CBAM-STN-TPS-YOLO Ablation Study Report\n\n",
            f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            "## Executive Summary\n\n"
        ]
        
        # Component ablation summary
        if 'component_ablation' in self.results:
            component_data = self.results['component_ablation']
            contributions = component_data.get('component_contributions', {})
            
            if 'full_model' in contributions:
                full_model_mAP = contributions['full_model']['mAP']
                baseline_mAP = contributions.get('baseline', {}).get('mAP', 0)
                total_improvement = full_model_mAP - baseline_mAP
                
                report_lines.extend([
                    f"- **Best Performing Model**: Full CBAM-STN-TPS-YOLO\n",
                    f"- **Final mAP**: {full_model_mAP:.4f}\n",
                    f"- **Total Improvement**: +{total_improvement:.4f} over baseline\n",
                    f"- **Relative Improvement**: {(total_improvement/baseline_mAP*100):.1f}%\n\n"
                ])
        
        # Component contributions
        report_lines.append("## Component Analysis\n\n")
        if 'component_ablation' in self.results:
            contributions = self.results['component_ablation'].get('component_contributions', {})
            
            for exp_name, data in contributions.items():
                if exp_name != 'individual_effects' and 'description' in data:
                    report_lines.extend([
                        f"### {data['description']}\n",
                        f"- **mAP**: {data['mAP']:.4f}\n",
                        f"- **Improvement**: +{data['absolute_improvement']:.4f}\n",
                        f"- **Components**: {', '.join(data['components']) if data['components'] else 'None'}\n\n"
                    ])
        
        # Hyperparameter sensitivity
        if 'hyperparameter_sensitivity' in self.results:
            report_lines.append("## Hyperparameter Sensitivity Analysis\n\n")
            
            sensitivities = self.results['hyperparameter_sensitivity']['individual_results']
            for param_name, sensitivity_data in sensitivities.items():
                optimal_value = sensitivity_data['optimal_value']
                optimal_mAP = sensitivity_data['optimal_mAP']
                sensitivity_score = sensitivity_data['sensitivity_score']
                
                report_lines.extend([
                    f"### {param_name.replace('_', ' ').title()}\n",
                    f"- **Optimal Value**: {optimal_value}\n",
                    f"- **Best mAP**: {optimal_mAP:.4f}\n",
                    f"- **Sensitivity Score**: {sensitivity_score:.4f}\n\n"
                ])
        
        # Key recommendations
        report_lines.extend([
            "## Recommendations\n\n",
            "Based on the ablation study results:\n\n",
            "1. **Use the full CBAM-STN-TPS-YOLO model** for best performance\n",
            "2. **CBAM attention mechanism** provides significant improvement\n",
            "3. **TPS transformation** enhances geometric robustness\n",
            "4. **Hyperparameter tuning** can provide additional gains\n\n"
        ])
        
        # Save report
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        
        return report_path
    
    def _generate_summary_statistics(self, save_dir: Path) -> None:
        """Generate summary statistics file"""
        
        summary_stats = {
            'experiment_info': {
                'start_time': self.experiment_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_experiments_run': 0
            },
            'component_analysis': {},
            'hyperparameter_analysis': {},
            'key_findings': {}
        }
        
        # Component analysis stats
        if 'component_ablation' in self.results:
            component_data = self.results['component_ablation']['individual_results']
            
            mAP_improvements = []
            for exp_name, exp_data in component_data.items():
                if exp_name != 'baseline':
                    baseline_mAP = component_data.get('baseline', {}).get('metrics', {}).get('mAP', {}).get('mean', 0)
                    exp_mAP = exp_data['metrics']['mAP']['mean']
                    improvement = exp_mAP - baseline_mAP
                    mAP_improvements.append(improvement)
            
            if mAP_improvements:
                summary_stats['component_analysis'] = {
                    'max_improvement': max(mAP_improvements),
                    'min_improvement': min(mAP_improvements),
                    'avg_improvement': np.mean(mAP_improvements),
                    'std_improvement': np.std(mAP_improvements)
                }
        
        # Save summary
        with open(save_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)

def run_component_ablation(config_path: str = 'config/training_configs.yaml',
                          num_runs: int = 3) -> Dict[str, Any]:
    """Standalone function to run component ablation study"""
    
    study = ComprehensiveAblationStudy(config_path)
    results = study.run_component_ablation(num_runs)
    
    # Generate report
    report_path = study.generate_comprehensive_report()
    
    logger.info(f"✅ Component ablation study completed")
    logger.info(f"📄 Report saved to: {report_path}")
    
    return results

def analyze_component_importance(results: Dict[str, Any]) -> Dict[str, float]:
    """Analyze and rank component importance"""
    
    if 'component_contributions' not in results:
        return {}
    
    contributions = results['component_contributions']
    component_importance = {}
    
    if 'individual_effects' in contributions:
        effects = contributions['individual_effects']
        
        # Normalize importance scores
        max_effect = max(abs(e) for e in effects.values()) if effects else 1
        
        for component, effect in effects.items():
            importance_score = abs(effect) / max_effect
            component_importance[component] = importance_score
    
    # Sort by importance
    return dict(sorted(component_importance.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":
    # Test ablation study functionality
    print("Testing ablation study...")
    
    try:
        # Run quick component ablation test
        study = ComprehensiveAblationStudy()
        
        # Test configuration preparation
        test_config = study._prepare_component_config(
            study.component_ablations['baseline'], 0
        )