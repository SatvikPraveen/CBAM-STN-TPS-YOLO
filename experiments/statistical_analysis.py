# experiments/statistical_analysis.py
"""
Advanced statistical analysis for CBAM-STN-TPS-YOLO experimental results
Implements comprehensive statistical testing and analysis as described in the research paper
"""

import time
import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

# Scientific computing imports
try:
    from scipy import stats
    from scipy.stats import shapiro, levene, mannwhitneyu, wilcoxon
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
except ImportError as e:
    warnings.warn(f"Some scientific packages not available: {e}")

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """Comprehensive statistical analysis for model comparison experiments"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.analysis_results = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def perform_comprehensive_analysis(self, results_file: str) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis on experimental results"""
        
        logger.info("Starting comprehensive statistical analysis...")
        
        # Load experimental results
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except FileNotFoundError:
            logger.error(f"Results file not found: {results_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in results file: {e}")
            return {}
        
        # Extract processed results
        if 'processed_results' in results:
            processed_results = results['processed_results']
        else:
            processed_results = results
        
        # Perform various statistical tests
        analysis = {
            'metadata': {
                'analysis_timestamp': time.time(),
                'significance_level': self.significance_level,
                'results_file': results_file
            },
            'descriptive_statistics': self._compute_descriptive_statistics(processed_results),
            'normality_tests': self._test_normality(processed_results),
            'variance_tests': self._test_variance_homogeneity(processed_results),
            'pairwise_comparisons': self._perform_pairwise_comparisons(processed_results),
            'effect_size_analysis': self._analyze_effect_sizes(processed_results),
            'robustness_analysis': self._analyze_robustness(processed_results),
            'confidence_intervals': self._compute_confidence_intervals(processed_results),
            'power_analysis': self._perform_power_analysis(processed_results)
        }
        
        # Store results
        self.analysis_results = analysis
        
        # Generate visualizations
        self._create_statistical_visualizations(processed_results, Path(results_file).parent)
        
        # Generate summary report
        summary_report = self._generate_statistical_summary(analysis)
        analysis['summary_report'] = summary_report
        
        logger.info("Statistical analysis completed successfully")
        return analysis
    
    def _compute_descriptive_statistics(self, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comprehensive descriptive statistics"""
        
        logger.info("Computing descriptive statistics...")
        
        descriptive_stats = {}
        metrics_to_analyze = ['best_mAP', 'precision', 'recall', 'f1_score']
        
        for model_type, aug_results in processed_results.items():
            descriptive_stats[model_type] = {}
            
            for aug_config, results in aug_results.items():
                metrics = results.get('metrics', {})
                aug_stats = {}
                
                for metric in metrics_to_analyze:
                    if metric in metrics:
                        values = metrics[metric].get('values', [])
                        if len(values) > 0:
                            aug_stats[metric] = {
                                'count': len(values),
                                'mean': float(np.mean(values)),
                                'std': float(np.std(values, ddof=1)),
                                'min': float(np.min(values)),
                                'max': float(np.max(values)),
                                'median': float(np.median(values)),
                                'q1': float(np.percentile(values, 25)),
                                'q3': float(np.percentile(values, 75)),
                                'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
                                'skewness': float(stats.skew(values)),
                                'kurtosis': float(stats.kurtosis(values)),
                                'cv': float(np.std(values, ddof=1) / np.mean(values)) if np.mean(values) != 0 else 0
                            }
                
                descriptive_stats[model_type][aug_config] = aug_stats
        
        return descriptive_stats
    
    def _test_normality(self, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test normality of distributions using Shapiro-Wilk test"""
        
        logger.info("Testing normality of distributions...")
        
        normality_tests = {}
        metrics_to_test = ['best_mAP', 'precision', 'recall', 'f1_score']
        
        for model_type, aug_results in processed_results.items():
            normality_tests[model_type] = {}
            
            for aug_config, results in aug_results.items():
                metrics = results.get('metrics', {})
                aug_normality = {}
                
                for metric in metrics_to_test:
                    if metric in metrics:
                        values = metrics[metric].get('values', [])
                        if len(values) >= 3:  # Minimum sample size for Shapiro-Wilk
                            try:
                                statistic, p_value = shapiro(values)
                                aug_normality[metric] = {
                                    'statistic': float(statistic),
                                    'p_value': float(p_value),
                                    'is_normal': p_value > self.significance_level,
                                    'test': 'Shapiro-Wilk'
                                }
                            except Exception as e:
                                logger.warning(f"Normality test failed for {model_type}-{aug_config}-{metric}: {e}")
                
                normality_tests[model_type][aug_config] = aug_normality
        
        return normality_tests
    
    def _test_variance_homogeneity(self, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test homogeneity of variances using Levene's test"""
        
        logger.info("Testing variance homogeneity...")
        
        variance_tests = {}
        metrics_to_test = ['best_mAP', 'precision', 'recall', 'f1_score']
        
        for metric in metrics_to_test:
            # Collect all values for this metric across models
            all_values = []
            model_labels = []
            
            for model_type, aug_results in processed_results.items():
                no_aug_results = aug_results.get('no_aug', {})
                if metric in no_aug_results.get('metrics', {}):
                    values = no_aug_results['metrics'][metric].get('values', [])
                    if len(values) > 0:
                        all_values.append(values)
                        model_labels.append(model_type)
            
            if len(all_values) >= 2:
                try:
                    statistic, p_value = levene(*all_values)
                    variance_tests[metric] = {
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'homogeneous_variances': p_value > self.significance_level,
                        'test': 'Levene',
                        'models_tested': model_labels
                    }
                except Exception as e:
                    logger.warning(f"Variance test failed for {metric}: {e}")
        
        return variance_tests
    
    def _perform_pairwise_comparisons(self, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive pairwise statistical comparisons"""
        
        logger.info("Performing pairwise statistical comparisons...")
        
        pairwise_results = {}
        metrics_to_compare = ['best_mAP', 'precision', 'recall', 'f1_score']
        
        # Get list of models
        models = list(processed_results.keys())
        
        for metric in metrics_to_compare:
            metric_comparisons = {}
            
            # Extract data for all models
            model_data = {}
            for model_type in models:
                no_aug_results = processed_results[model_type].get('no_aug', {})
                if metric in no_aug_results.get('metrics', {}):
                    values = no_aug_results['metrics'][metric].get('values', [])
                    if len(values) > 0:
                        model_data[model_type] = values
            
            # Perform pairwise comparisons
            model_list = list(model_data.keys())
            for i in range(len(model_list)):
                for j in range(i + 1, len(model_list)):
                    model1, model2 = model_list[i], model_list[j]
                    
                    comparison_key = f"{model1}_vs_{model2}"
                    comparison_result = self._compare_two_models(
                        model_data[model1], model_data[model2], model1, model2
                    )
                    metric_comparisons[comparison_key] = comparison_result
            
            pairwise_results[metric] = metric_comparisons
        
        return pairwise_results
    
    def _compare_two_models(self, values1: List[float], values2: List[float], 
                           model1: str, model2: str) -> Dict[str, Any]:
        """Compare two models using appropriate statistical tests"""
        
        comparison = {
            'model1': model1,
            'model2': model2,
            'n1': len(values1),
            'n2': len(values2),
            'mean1': float(np.mean(values1)),
            'mean2': float(np.mean(values2)),
            'std1': float(np.std(values1, ddof=1)),
            'std2': float(np.std(values2, ddof=1)),
            'mean_difference': float(np.mean(values1) - np.mean(values2))
        }
        
        # Test for normality (simplified)
        normal1 = len(values1) < 3 or shapiro(values1)[1] > self.significance_level
        normal2 = len(values2) < 3 or shapiro(values2)[1] > self.significance_level
        
        if normal1 and normal2:
            # Use parametric tests
            if len(values1) == len(values2):
                # Paired t-test (if samples are paired)
                try:
                    t_stat, p_val = stats.ttest_rel(values1, values2)
                    test_used = "Paired t-test"
                except:
                    # Fall back to independent t-test
                    t_stat, p_val = stats.ttest_ind(values1, values2)
                    test_used = "Independent t-test"
            else:
                # Independent t-test
                t_stat, p_val = stats.ttest_ind(values1, values2)
                test_used = "Independent t-test"
            
            comparison.update({
                'test_statistic': float(t_stat),
                'p_value': float(p_val),
                'test_used': test_used
            })
        else:
            # Use non-parametric tests
            if len(values1) == len(values2):
                # Wilcoxon signed-rank test
                try:
                    w_stat, p_val = wilcoxon(values1, values2)
                    test_used = "Wilcoxon signed-rank test"
                except:
                    # Fall back to Mann-Whitney U
                    u_stat, p_val = mannwhitneyu(values1, values2, alternative='two-sided')
                    test_used = "Mann-Whitney U test"
                    w_stat = u_stat
            else:
                # Mann-Whitney U test
                u_stat, p_val = mannwhitneyu(values1, values2, alternative='two-sided')
                test_used = "Mann-Whitney U test"
                w_stat = u_stat
            
            comparison.update({
                'test_statistic': float(w_stat),
                'p_value': float(p_val),
                'test_used': test_used
            })
        
        # Determine significance
        comparison['significant'] = p_val < self.significance_level
        comparison['significance_level'] = self.significance_level
        
        # Calculate effect size
        effect_size = self._calculate_effect_size(values1, values2)
        comparison.update(effect_size)
        
        return comparison
    
    def _calculate_effect_size(self, values1: List[float], values2: List[float]) -> Dict[str, Any]:
        """Calculate various effect size measures"""
        
        mean1, mean2 = np.mean(values1), np.mean(values2)
        std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
        n1, n2 = len(values1), len(values2)
        
        # Cohen's d
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Glass's delta (using std of control group - assume model2 is control)
        glass_delta = (mean1 - mean2) / std2 if std2 > 0 else 0
        
        # Hedges' g (bias-corrected Cohen's d)
        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
        hedges_g = cohens_d * correction_factor
        
        return {
            'cohens_d': float(cohens_d),
            'glass_delta': float(glass_delta),
            'hedges_g': float(hedges_g),
            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d)),
            'practical_significance': abs(cohens_d) > 0.5  # Threshold for practical significance
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude according to Cohen's conventions"""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def _analyze_effect_sizes(self, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive effect size analysis"""
        
        logger.info("Analyzing effect sizes...")
        
        effect_analysis = {
            'effect_size_summary': {},
            'practical_significance': {},
            'effect_size_distribution': {}
        }
        
        # Reference model (typically the baseline or best performing)
        reference_models = ['YOLO', 'CBAM-STN-TPS-YOLO']
        
        for ref_model in reference_models:
            if ref_model not in processed_results:
                continue
                
            effect_analysis['effect_size_summary'][f'{ref_model}_as_reference'] = {}
            
            ref_data = processed_results[ref_model].get('no_aug', {}).get('metrics', {})
            ref_mAP = ref_data.get('best_mAP', {}).get('values', [])
            
            if not ref_mAP:
                continue
            
            for model_type, aug_results in processed_results.items():
                if model_type == ref_model:
                    continue
                
                model_data = aug_results.get('no_aug', {}).get('metrics', {})
                model_mAP = model_data.get('best_mAP', {}).get('values', [])
                
                if len(model_mAP) > 0:
                    effect_size = self._calculate_effect_size(model_mAP, ref_mAP)
                    effect_analysis['effect_size_summary'][f'{ref_model}_as_reference'][model_type] = effect_size
        
        return effect_analysis
    
    def _analyze_robustness(self, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze robustness across different augmentation conditions"""
        
        logger.info("Analyzing robustness across augmentation conditions...")
        
        robustness_analysis = {
            'consistency_scores': {},
            'robustness_ranking': {},
            'augmentation_sensitivity': {}
        }
        
        for model_type, aug_results in processed_results.items():
            model_mAPs = []
            augmentation_effects = {}
            
            # Collect mAP values across all augmentations
            for aug_config, results in aug_results.items():
                mAP_values = results.get('metrics', {}).get('best_mAP', {}).get('values', [])
                if len(mAP_values) > 0:
                    mean_mAP = np.mean(mAP_values)
                    model_mAPs.append(mean_mAP)
                    augmentation_effects[aug_config] = {
                        'mean_mAP': mean_mAP,
                        'std_mAP': np.std(mAP_values, ddof=1),
                        'values': mAP_values
                    }
            
            # Calculate robustness metrics
            if len(model_mAPs) > 1:
                robustness_score = 1 / (1 + np.std(model_mAPs))  # Higher is more robust
                consistency_score = np.min(model_mAPs) / np.max(model_mAPs)  # Ratio of worst to best
                
                robustness_analysis['consistency_scores'][model_type] = {
                    'robustness_score': float(robustness_score),
                    'consistency_score': float(consistency_score),
                    'mean_performance': float(np.mean(model_mAPs)),
                    'std_performance': float(np.std(model_mAPs)),
                    'performance_range': float(np.max(model_mAPs) - np.min(model_mAPs)),
                    'coefficient_of_variation': float(np.std(model_mAPs) / np.mean(model_mAPs))
                }
                
                robustness_analysis['augmentation_sensitivity'][model_type] = augmentation_effects
        
        # Rank models by robustness
        if robustness_analysis['consistency_scores']:
            sorted_models = sorted(
                robustness_analysis['consistency_scores'].items(),
                key=lambda x: x[1]['robustness_score'],
                reverse=True
            )
            robustness_analysis['robustness_ranking'] = [
                {'model': model, 'score': data['robustness_score']}
                for model, data in sorted_models
            ]
        
        return robustness_analysis
    
    def _compute_confidence_intervals(self, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute confidence intervals for key metrics"""
        
        logger.info("Computing confidence intervals...")
        
        confidence_intervals = {}
        confidence_level = 0.95
        alpha = 1 - confidence_level
        
        for model_type, aug_results in processed_results.items():
            confidence_intervals[model_type] = {}
            
            for aug_config, results in aug_results.items():
                metrics = results.get('metrics', {})
                aug_ci = {}
                
                for metric_name, metric_data in metrics.items():
                    values = metric_data.get('values', [])
                    if len(values) > 1:
                        mean = np.mean(values)
                        std_error = stats.sem(values)  # Standard error of the mean
                        
                        # t-distribution critical value
                        df = len(values) - 1
                        t_critical = stats.t.ppf(1 - alpha/2, df)
                        
                        margin_of_error = t_critical * std_error
                        
                        aug_ci[metric_name] = {
                            'mean': float(mean),
                            'confidence_level': confidence_level,
                            'lower_bound': float(mean - margin_of_error),
                            'upper_bound': float(mean + margin_of_error),
                            'margin_of_error': float(margin_of_error),
                            'std_error': float(std_error)
                        }
                
                confidence_intervals[model_type][aug_config] = aug_ci
        
        return confidence_intervals
    
    def _perform_power_analysis(self, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical power analysis"""
        
        logger.info("Performing power analysis...")
        
        power_analysis = {
            'sample_size_recommendations': {},
            'achieved_power': {},
            'minimum_detectable_effect': {}
        }
        
        # This is a simplified power analysis
        # In practice, you might want to use specialized libraries like statsmodels
        
        alpha = self.significance_level
        desired_power = 0.8
        
        for model_type, aug_results in processed_results.items():
            no_aug_results = aug_results.get('no_aug', {})
            mAP_data = no_aug_results.get('metrics', {}).get('best_mAP', {})
            
            if 'values' in mAP_data:
                values = mAP_data['values']
                n = len(values)
                std = np.std(values, ddof=1)
                
                # Estimate minimum detectable effect for current sample size
                # This is a simplified calculation
                if n > 1:
                    t_critical = stats.t.ppf(1 - alpha/2, n-1)
                    t_power = stats.t.ppf(desired_power, n-1)
                    
                    # Minimum detectable effect (Cohen's d)
                    min_effect = (t_critical + t_power) * np.sqrt(2/n)
                    
                    power_analysis['minimum_detectable_effect'][model_type] = {
                        'current_sample_size': n,
                        'min_detectable_effect_cohens_d': float(min_effect),
                        'min_detectable_difference': float(min_effect * std),
                        'interpretation': self._interpret_effect_size(min_effect)
                    }
        
        return power_analysis
    
    def _create_statistical_visualizations(self, processed_results: Dict[str, Any], 
                                         output_dir: Path) -> None:
        """Create comprehensive statistical visualizations"""
        
        logger.info("Creating statistical visualizations...")
        
        try:
            # Create plots directory
            plots_dir = output_dir / 'statistical_plots'
            plots_dir.mkdir(exist_ok=True)
            
            # 1. Box plots for model comparison
            self._create_model_comparison_boxplots(processed_results, plots_dir)
            
            # 2. Effect size visualization
            self._create_effect_size_plot(plots_dir)
            
            # 3. Robustness analysis plots
            self._create_robustness_plots(processed_results, plots_dir)
            
            # 4. Statistical significance matrix
            self._create_significance_matrix(plots_dir)
            
            # 5. Distribution plots
            self._create_distribution_plots(processed_results, plots_dir)
            
            logger.info(f"Statistical visualizations saved to: {plots_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create statistical visualizations: {e}")
    
    def _create_model_comparison_boxplots(self, processed_results: Dict[str, Any], 
                                        output_dir: Path) -> None:
        """Create box plots for model comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = ['best_mAP', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
            
            # Prepare data for plotting
            plot_data = []
            model_names = []
            
            for model_type, aug_results in processed_results.items():
                no_aug_results = aug_results.get('no_aug', {})
                if metric in no_aug_results.get('metrics', {}):
                    values = no_aug_results['metrics'][metric].get('values', [])
                    if len(values) > 0:
                        for value in values:
                            plot_data.append({'Model': model_type, metric: value})
            
            if plot_data:
                df = pd.DataFrame(plot_data)
                sns.boxplot(data=df, x='Model', y=metric, ax=axes[i])
                axes[i].set_title(f'{metric.upper()} Distribution by Model', fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_effect_size_plot(self, output_dir: Path) -> None:
        """Create effect size visualization"""
        
        if not hasattr(self, 'analysis_results') or 'pairwise_comparisons' not in self.analysis_results:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        comparisons = self.analysis_results['pairwise_comparisons'].get('best_mAP', {})
        
        comparison_names = []
        effect_sizes = []
        p_values = []
        
        for comp_name, comp_data in comparisons.items():
            comparison_names.append(comp_name.replace('_vs_', ' vs '))
            effect_sizes.append(comp_data.get('cohens_d', 0))
            p_values.append(comp_data.get('p_value', 1))
        
        if effect_sizes:
            # Create scatter plot
            colors = ['red' if p < 0.05 else 'blue' for p in p_values]
            scatter = ax.scatter(effect_sizes, range(len(effect_sizes)), 
                               c=colors, s=100, alpha=0.7)
            
            # Add vertical lines for effect size thresholds
            ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
            ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.8)
            
            ax.set_yticks(range(len(comparison_names)))
            ax.set_yticklabels(comparison_names)
            ax.set_xlabel("Cohen's d (Effect Size)")
            ax.set_title('Effect Size Analysis (mAP Comparisons)', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add significance annotation
            ax.text(0.02, 0.98, 'Red: p < 0.05\nBlue: p ≥ 0.05', 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'effect_size_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_robustness_plots(self, processed_results: Dict[str, Any], 
                               output_dir: Path) -> None:
        """Create robustness analysis plots"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Robustness across augmentations
        models = []
        aug_configs = []
        mAP_values = []
        
        for model_type, aug_results in processed_results.items():
            for aug_config, results in aug_results.items():
                mAP_data = results.get('metrics', {}).get('best_mAP', {})
                if 'mean' in mAP_data:
                    models.append(model_type)
                    aug_configs.append(aug_config)
                    mAP_values.append(mAP_data['mean'])
        
        if mAP_values:
            # Create pivot table for heatmap
            df = pd.DataFrame({
                'Model': models,
                'Augmentation': aug_configs,
                'mAP': mAP_values
            })
            
            pivot_table = df.pivot(index='Model', columns='Augmentation', values='mAP')
            
            # Heatmap
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                       ax=ax1, cbar_kws={'label': 'mAP'})
            ax1.set_title('Model Performance Across Augmentations', fontweight='bold')
            ax1.set_xlabel('Augmentation Type')
            ax1.set_ylabel('Model Type')
        
        # Variance analysis
        if hasattr(self, 'analysis_results') and 'robustness_analysis' in self.analysis_results:
            robustness_data = self.analysis_results['robustness_analysis'].get('consistency_scores', {})
            
            if robustness_data:
                models = list(robustness_data.keys())
                robustness_scores = [data['robustness_score'] for data in robustness_data.values()]
                
                bars = ax2.bar(models, robustness_scores, alpha=0.7, color='skyblue')
                ax2.set_title('Model Robustness Scores', fontweight='bold')
                ax2.set_ylabel('Robustness Score')
                ax2.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, score in zip(bars, robustness_scores):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'robustness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_significance_matrix(self, output_dir: Path) -> None:
        """Create statistical significance matrix"""
        
        if not hasattr(self, 'analysis_results') or 'pairwise_comparisons' not in self.analysis_results:
            return
        
        comparisons = self.analysis_results['pairwise_comparisons'].get('best_mAP', {})
        
        if not comparisons:
            return
        
        # Extract model names
        all_models = set()
        for comp_name in comparisons.keys():
            models = comp_name.split('_vs_')
            all_models.update(models)
        
        all_models = sorted(list(all_models))
        n_models = len(all_models)
        
        # Create significance matrix
        sig_matrix = np.ones((n_models, n_models))  # Default to 1 (not significant)
        p_matrix = np.ones((n_models, n_models))
        
        for comp_name, comp_data in comparisons.items():
            models = comp_name.split('_vs_')
            if len(models) == 2:
                i = all_models.index(models[0])
                j = all_models.index(models[1])
                
                is_sig = comp_data.get('significant', False)
                p_val = comp_data.get('p_value', 1.0)
                
                sig_matrix[i, j] = 0 if is_sig else 1
                sig_matrix[j, i] = 0 if is_sig else 1
                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Significance matrix
        sns.heatmap(sig_matrix, annot=True, fmt='.0f', cmap='RdYlGn',
                   xticklabels=all_models, yticklabels=all_models,
                   ax=ax1, cbar_kws={'label': 'Significant (0) / Not Significant (1)'})
        ax1.set_title('Statistical Significance Matrix', fontweight='bold')
        
        # P-value matrix
        sns.heatmap(p_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=all_models, yticklabels=all_models,
                   ax=ax2, cbar_kws={'label': 'p-value'})
        ax2.set_title('P-value Matrix', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'significance_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_distribution_plots(self, processed_results: Dict[str, Any], 
                                 output_dir: Path) -> None:
        """Create distribution plots for normality assessment"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for model_type, aug_results in processed_results.items():
            if plot_idx >= len(axes):
                break
            
            no_aug_results = aug_results.get('no_aug', {})
            mAP_data = no_aug_results.get('metrics', {}).get('best_mAP', {})
            
            if 'values' in mAP_data:
                values = mAP_data['values']
                
                # Histogram with normal distribution overlay
                axes[plot_idx].hist(values, bins=10, density=True, alpha=0.7, 
                                  color='skyblue', edgecolor='black')
                
                # Overlay normal distribution
                mean, std = np.mean(values), np.std(values)
                x = np.linspace(min(values), max(values), 100)
                axes[plot_idx].plot(x, stats.norm.pdf(x, mean, std), 'r-', 
                                  linewidth=2, label='Normal fit')
                
                axes[plot_idx].set_title(f'{model_type} mAP Distribution', fontweight='bold')
                axes[plot_idx].set_xlabel('mAP')
                axes[plot_idx].set_ylabel('Density')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                
                plot_idx += 1
        
        # Remove unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].remove()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_statistical_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive statistical summary report"""
        
        summary_lines = [
            "# Statistical Analysis Summary\n\n",
            f"**Analysis Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**Significance Level**: {self.significance_level}\n\n",
            
            "## Key Findings\n\n"
        ]
        
        # Pairwise comparison summary
        if 'pairwise_comparisons' in analysis:
            mAP_comparisons = analysis['pairwise_comparisons'].get('best_mAP', {})
            significant_comparisons = []
            
            for comp_name, comp_data in mAP_comparisons.items():
                if comp_data.get('significant', False):
                    effect_size = comp_data.get('cohens_d', 0)
                    effect_interp = self._interpret_effect_size(abs(effect_size))
                    significant_comparisons.append(
                        f"- **{comp_name.replace('_vs_', ' vs ')}**: "
                        f"p = {comp_data.get('p_value', 1):.4f}, "
                        f"Cohen's d = {effect_size:.3f} ({effect_interp} effect)"
                    )
            
            if significant_comparisons:
                summary_lines.append("### Statistically Significant Differences (mAP)\n\n")
                summary_lines.extend([comp + "\n" for comp in significant_comparisons])
            else:
                summary_lines.append("### No statistically significant differences found in mAP comparisons\n\n")
        
        # Robustness summary
        if 'robustness_analysis' in analysis:
            robustness_ranking = analysis['robustness_analysis'].get('robustness_ranking', [])
            if robustness_ranking:
                summary_lines.append("### Model Robustness Ranking\n\n")
                for i, model_data in enumerate(robustness_ranking[:3], 1):
                    summary_lines.append(
                        f"{i}. **{model_data['model']}**: "
                        f"Robustness Score = {model_data['score']:.3f}\n"
                    )
        
        # Normality test summary
        if 'normality_tests' in analysis:
            summary_lines.append("\n### Normality Test Results\n\n")
            summary_lines.append("Models with normally distributed mAP values:\n")
            
            for model_type, aug_results in analysis['normality_tests'].items():
                no_aug_data = aug_results.get('no_aug', {})
                mAP_normality = no_aug_data.get('best_mAP', {})
                if mAP_normality and mAP_normality.get('is_normal', False):
                    p_val = mAP_normality.get('p_value', 0)
                    summary_lines.append(f"- **{model_type}**: Normal (p = {p_val:.4f})\n")
        
        return ''.join(summary_lines)

# Standalone functions for backward compatibility
def perform_statistical_analysis(results_file: str = 'results/experimental_results.json') -> Dict[str, Any]:
    """Perform comprehensive statistical analysis on experimental results"""
    
    analyzer = StatisticalAnalyzer()
    
    try:
        analysis_results = analyzer.perform_comprehensive_analysis(results_file)
        
        # Save analysis results
        results_path = Path(results_file)
        output_file = results_path.parent / 'statistical_analysis_results.json'
        
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Generate and save summary report
        summary_file = results_path.parent / 'statistical_summary.md'
        with open(summary_file, 'w') as f:
            f.write(analysis_results.get('summary_report', ''))
        
        logger.info(f"Statistical analysis completed. Results saved to: {output_file}")
        logger.info(f"Summary report saved to: {summary_file}")
        
        # Print key findings to console
        _print_key_findings(analysis_results)
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Statistical analysis failed: {e}")
        return {}

def compute_statistical_significance(model1_values: List[float], 
                                   model2_values: List[float],
                                   model1_name: str = "Model 1",
                                   model2_name: str = "Model 2") -> Dict[str, Any]:
    """Compute statistical significance between two models"""
    
    analyzer = StatisticalAnalyzer()
    return analyzer._compare_two_models(model1_values, model2_values, model1_name, model2_name)

def generate_comparison_report(results: Dict[str, Any], 
                             output_path: str = "comparison_report.md") -> str:
    """Generate a formatted comparison report"""
    
    analyzer = StatisticalAnalyzer()
    analyzer.analysis_results = results
    
    summary = analyzer._generate_statistical_summary(results)
    
    with open(output_path, 'w') as f:
        f.write(summary)
    
    logger.info(f"Comparison report saved to: {output_path}")
    return output_path

def _print_key_findings(analysis_results: Dict[str, Any]) -> None:
    """Print key statistical findings to console"""
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - KEY FINDINGS")
    print("="*80)
    
    # Significant comparisons
    if 'pairwise_comparisons' in analysis_results:
        mAP_comparisons = analysis_results['pairwise_comparisons'].get('best_mAP', {})
        significant_found = False
        
        print("\n📊 SIGNIFICANT DIFFERENCES (mAP):")
        print("-" * 50)
        
        for comp_name, comp_data in mAP_comparisons.items():
            if comp_data.get('significant', False):
                significant_found = True
                models = comp_name.replace('_vs_', ' vs ')
                p_val = comp_data.get('p_value', 1)
                effect_size = comp_data.get('cohens_d', 0)
                effect_interp = StatisticalAnalyzer()._interpret_effect_size(abs(effect_size))
                
                significance_symbol = "🔥" if abs(effect_size) > 0.8 else "✅"
                print(f"{significance_symbol} {models}")
                print(f"   p-value: {p_val:.6f}")
                print(f"   Effect size: {effect_size:.3f} ({effect_interp})")
                print(f"   Mean difference: {comp_data.get('mean_difference', 0):.4f}")
                print()
        
        if not significant_found:
            print("❌ No statistically significant differences found")
    
    # Robustness ranking
    if 'robustness_analysis' in analysis_results:
        robustness_data = analysis_results['robustness_analysis']
        
        if 'robustness_ranking' in robustness_data:
            print("\n🛡️  ROBUSTNESS RANKING:")
            print("-" * 30)
            
            for i, model_data in enumerate(robustness_data['robustness_ranking'][:5], 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📍"
                print(f"{medal} {i}. {model_data['model']}: {model_data['score']:.3f}")
    
    # Best overall model
    if 'summary' in analysis_results:
        summary = analysis_results['summary']
        best_model = summary.get('best_overall_model', 'Unknown')
        best_mAP = summary.get('best_overall_mAP', 0)
        
        print(f"\n🏆 BEST PERFORMING MODEL:")
        print("-" * 35)
        print(f"🌟 {best_model}")
        print(f"📈 mAP: {best_mAP:.4f}")
    
    # Effect size summary
    if 'effect_size_analysis' in analysis_results:
        print(f"\n📏 EFFECT SIZE INTERPRETATION:")
        print("-" * 40)
        print("• Negligible: < 0.2")
        print("• Small: 0.2 - 0.5")
        print("• Medium: 0.5 - 0.8") 
        print("• Large: > 0.8")
    
    print("\n" + "="*80)

class AdvancedStatisticalAnalyzer(StatisticalAnalyzer):
    """Extended statistical analyzer with advanced techniques"""
    
    def __init__(self, significance_level: float = 0.05):
        super().__init__(significance_level)
        self.bootstrap_iterations = 1000
        self.permutation_iterations = 1000
    
    def perform_bootstrap_analysis(self, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform bootstrap analysis for confidence intervals"""
        
        logger.info("Performing bootstrap analysis...")
        
        bootstrap_results = {}
        
        for model_type, aug_results in processed_results.items():
            no_aug_results = aug_results.get('no_aug', {})
            mAP_values = no_aug_results.get('metrics', {}).get('best_mAP', {}).get('values', [])
            
            if len(mAP_values) >= 3:
                bootstrap_samples = []
                
                # Generate bootstrap samples
                for _ in range(self.bootstrap_iterations):
                    bootstrap_sample = np.random.choice(mAP_values, size=len(mAP_values), replace=True)
                    bootstrap_samples.append(np.mean(bootstrap_sample))
                
                # Calculate bootstrap statistics
                bootstrap_results[model_type] = {
                    'bootstrap_mean': float(np.mean(bootstrap_samples)),
                    'bootstrap_std': float(np.std(bootstrap_samples)),
                    'bootstrap_ci_95': [
                        float(np.percentile(bootstrap_samples, 2.5)),
                        float(np.percentile(bootstrap_samples, 97.5))
                    ],
                    'bootstrap_ci_99': [
                        float(np.percentile(bootstrap_samples, 0.5)),
                        float(np.percentile(bootstrap_samples, 99.5))
                    ],
                    'original_mean': float(np.mean(mAP_values)),
                    'bias_estimate': float(np.mean(bootstrap_samples) - np.mean(mAP_values))
                }
        
        return bootstrap_results
    
    def perform_permutation_test(self, values1: List[float], values2: List[float]) -> Dict[str, Any]:
        """Perform permutation test for difference in means"""
        
        observed_diff = np.mean(values1) - np.mean(values2)
        combined_values = values1 + values2
        n1 = len(values1)
        
        permuted_diffs = []
        
        for _ in range(self.permutation_iterations):
            # Randomly permute the combined data
            permuted = np.random.permutation(combined_values)
            
            # Split into two groups
            group1 = permuted[:n1]
            group2 = permuted[n1:]
            
            # Calculate difference in means
            permuted_diff = np.mean(group1) - np.mean(group2)
            permuted_diffs.append(permuted_diff)
        
        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        
        return {
            'observed_difference': float(observed_diff),
            'permutation_p_value': float(p_value),
            'permutation_distribution_mean': float(np.mean(permuted_diffs)),
            'permutation_distribution_std': float(np.std(permuted_diffs)),
            'significant': p_value < self.significance_level
        }
    
    def perform_bayesian_analysis(self, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Bayesian analysis (simplified implementation)"""
        
        logger.info("Performing Bayesian analysis...")
        
        # This is a simplified Bayesian approach
        # In practice, you might want to use libraries like PyMC3 or Stan
        
        bayesian_results = {}
        
        # Use informative priors based on domain knowledge
        # Prior: mAP ~ Beta(alpha=2, beta=2) scaled to [0,1] range
        prior_alpha = 2
        prior_beta = 2
        
        for model_type, aug_results in processed_results.items():
            no_aug_results = aug_results.get('no_aug', {})
            mAP_values = no_aug_results.get('metrics', {}).get('best_mAP', {}).get('values', [])
            
            if len(mAP_values) > 0:
                # Convert to success/failure counts (simplified)
                # Assume mAP > 0.5 is "success"
                successes = sum(1 for x in mAP_values if x > 0.5)
                failures = len(mAP_values) - successes
                
                # Update prior with data (Beta-Binomial conjugate)
                posterior_alpha = prior_alpha + successes
                posterior_beta = prior_beta + failures
                
                # Calculate posterior statistics
                posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
                posterior_var = (posterior_alpha * posterior_beta) / \
                               ((posterior_alpha + posterior_beta)**2 * (posterior_alpha + posterior_beta + 1))
                
                # Credible intervals
                credible_interval_95 = stats.beta.interval(0.95, posterior_alpha, posterior_beta)
                credible_interval_99 = stats.beta.interval(0.99, posterior_alpha, posterior_beta)
                
                bayesian_results[model_type] = {
                    'posterior_mean': float(posterior_mean),
                    'posterior_variance': float(posterior_var),
                    'posterior_alpha': float(posterior_alpha),
                    'posterior_beta': float(posterior_beta),
                    'credible_interval_95': [float(credible_interval_95[0]), float(credible_interval_95[1])],
                    'credible_interval_99': [float(credible_interval_99[0]), float(credible_interval_99[1])],
                    'probability_better_than_baseline': float(1 - stats.beta.cdf(0.5, posterior_alpha, posterior_beta))
                }
        
        return bayesian_results
    
    def perform_meta_analysis(self, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform meta-analysis across different experimental conditions"""
        
        logger.info("Performing meta-analysis...")
        
        meta_results = {}
        
        for model_type, aug_results in processed_results.items():
            studies = []  # Each augmentation condition is treated as a separate "study"
            
            for aug_config, results in aug_results.items():
                mAP_data = results.get('metrics', {}).get('best_mAP', {})
                values = mAP_data.get('values', [])
                
                if len(values) > 1:
                    study_mean = np.mean(values)
                    study_se = stats.sem(values)
                    study_n = len(values)
                    
                    studies.append({
                        'condition': aug_config,
                        'mean': study_mean,
                        'se': study_se,
                        'n': study_n,
                        'weight': 1 / (study_se**2) if study_se > 0 else 1
                    })
            
            if len(studies) > 1:
                # Calculate weighted average (fixed-effects model)
                total_weight = sum(study['weight'] for study in studies)
                weighted_mean = sum(study['mean'] * study['weight'] for study in studies) / total_weight
                
                # Calculate heterogeneity (I²)
                weighted_var = 1 / total_weight
                q_statistic = sum(study['weight'] * (study['mean'] - weighted_mean)**2 for study in studies)
                df = len(studies) - 1
                
                # I² statistic (proportion of total variation due to heterogeneity)
                i_squared = max(0, (q_statistic - df) / q_statistic) if q_statistic > 0 else 0
                
                meta_results[model_type] = {
                    'pooled_effect': float(weighted_mean),
                    'pooled_se': float(np.sqrt(weighted_var)),
                    'pooled_ci_95': [
                        float(weighted_mean - 1.96 * np.sqrt(weighted_var)),
                        float(weighted_mean + 1.96 * np.sqrt(weighted_var))
                    ],
                    'q_statistic': float(q_statistic),
                    'i_squared': float(i_squared),
                    'heterogeneity': 'high' if i_squared > 0.75 else 'moderate' if i_squared > 0.5 else 'low',
                    'num_studies': len(studies),
                    'individual_studies': studies
                }
        
        return meta_results

def run_advanced_statistical_analysis(results_file: str) -> Dict[str, Any]:
    """Run advanced statistical analysis with bootstrap, permutation, and Bayesian methods"""
    
    analyzer = AdvancedStatisticalAnalyzer()
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    processed_results = results.get('processed_results', results)
    
    # Perform advanced analyses
    advanced_analysis = {
        'basic_analysis': analyzer.perform_comprehensive_analysis(results_file),
        'bootstrap_analysis': analyzer.perform_bootstrap_analysis(processed_results),
        'bayesian_analysis': analyzer.perform_bayesian_analysis(processed_results),
        'meta_analysis': analyzer.perform_meta_analysis(processed_results)
    }
    
    # Save advanced results
    results_path = Path(results_file)
    output_file = results_path.parent / 'advanced_statistical_analysis.json'
    
    with open(output_file, 'w') as f:
        json.dump(advanced_analysis, f, indent=2, default=str)
    
    logger.info(f"Advanced statistical analysis completed. Results saved to: {output_file}")
    
    return advanced_analysis

if __name__ == "__main__":
    # Test statistical analysis functionality
    print("Testing statistical analysis...")
    
    try:
        # Create dummy experimental results for testing
        dummy_results = {
            'processed_results': {
                'YOLO': {
                    'no_aug': {
                        'metrics': {
                            'best_mAP': {
                                'values': [0.65, 0.67, 0.64, 0.66, 0.68],
                                'mean': 0.66,
                                'std': 0.015
                            }
                        }
                    }
                },
                'CBAM-STN-TPS-YOLO': {
                    'no_aug': {
                        'metrics': {
                            'best_mAP': {
                                'values': [0.78, 0.80, 0.77, 0.79, 0.81],
                                'mean': 0.79,
                                'std': 0.016
                            }
                        }
                    }
                }
            }
        }
        
        # Save dummy results
        test_file = 'test_results.json'
        with open(test_file, 'w') as f:
            json.dump(dummy_results, f, indent=2)
        
        # Test statistical analysis
        analysis_results = perform_statistical_analysis(test_file)
        
        if analysis_results:
            print("✅ Statistical analysis test passed")
            print(f"✅ Found {len(analysis_results.get('pairwise_comparisons', {}))} comparison metrics")
        else:
            print("❌ Statistical analysis test failed")
        
        # Test advanced analysis
        advanced_results = run_advanced_statistical_analysis(test_file)
        
        if advanced_results:
            print("✅ Advanced statistical analysis test passed")
        
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Statistical analysis test failed: {e}")
    
    print("Statistical analysis tests completed!")