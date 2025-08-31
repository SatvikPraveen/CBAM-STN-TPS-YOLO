# src/utils/visualization.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
import warnings
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """Advanced visualization utilities for CBAM-STN-TPS-YOLO research"""
    
    def __init__(self, class_names: List[str] = None, colors: List[str] = None, 
                 style: str = 'modern'):
        self.class_names = class_names or ['Cotton', 'Rice', 'Corn', 'Soybean', 'Wheat']
        
        # Color schemes
        if colors is None:
            if style == 'modern':
                self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                              '#DDA0DD', '#F4A460', '#87CEEB', '#98FB98', '#F0E68C']
            elif style == 'academic':
                self.colors = ['red', 'blue', 'green', 'orange', 'purple', 
                              'brown', 'pink', 'gray', 'olive', 'cyan']
            else:
                self.colors = colors
        else:
            self.colors = colors
        
        # Set plotting style
        plt.style.use('seaborn-v0_8' if style == 'modern' else 'default')
        sns.set_palette("husl")
        
        # Figure settings
        self.figure_dpi = 300
        self.figure_size = (12, 8)
        
    def plot_predictions(self, image: Union[torch.Tensor, np.ndarray], 
                        predictions: Union[torch.Tensor, List], 
                        targets: Union[torch.Tensor, List] = None,
                        save_path: Optional[str] = None, 
                        title: str = "Model Predictions",
                        confidence_threshold: float = 0.5,
                        show_confidence: bool = True) -> None:
        """Enhanced prediction visualization with confidence scores and styling"""
        
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)
        
        # Process image
        display_image = self._prepare_image_for_display(image)
        ax.imshow(display_image)
        
        # Plot predictions
        if predictions is not None and len(predictions) > 0:
            self._plot_bounding_boxes(
                ax, predictions, display_image.shape[:2], 
                box_type='prediction', confidence_threshold=confidence_threshold,
                show_confidence=show_confidence
            )
        
        # Plot ground truth targets
        if targets is not None and len(targets) > 0:
            self._plot_bounding_boxes(
                ax, targets, display_image.shape[:2], 
                box_type='target'
            )
        
        # Styling
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        self._add_prediction_legend(ax, predictions, targets)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            logger.info(f"Prediction visualization saved to: {save_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def visualize_attention_maps(self, image: Union[torch.Tensor, np.ndarray],
                               channel_attention: torch.Tensor,
                               spatial_attention: torch.Tensor,
                               save_path: Optional[str] = None,
                               title: str = "CBAM Attention Visualization") -> None:
        """Advanced CBAM attention visualization with heatmaps"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Prepare image
        display_image = self._prepare_image_for_display(image)
        
        # Original image
        axes[0, 0].imshow(display_image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Process attention maps
        channel_att = self._process_attention_map(channel_attention, target_size=display_image.shape[:2])
        spatial_att = self._process_attention_map(spatial_attention, target_size=display_image.shape[:2])
        
        # Channel attention visualization
        im1 = axes[0, 1].imshow(channel_att, cmap='jet', alpha=0.8)
        axes[0, 1].imshow(display_image, alpha=0.4)
        axes[0, 1].set_title('Channel Attention Map', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
        cbar1.set_label('Attention Weight', fontsize=12)
        
        # Spatial attention visualization
        im2 = axes[0, 2].imshow(spatial_att, cmap='hot', alpha=0.8)
        axes[0, 2].imshow(display_image, alpha=0.4)
        axes[0, 2].set_title('Spatial Attention Map', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        cbar2 = plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
        cbar2.set_label('Attention Weight', fontsize=12)
        
        # Combined attention
        combined_att = channel_att * spatial_att
        im3 = axes[1, 0].imshow(combined_att, cmap='viridis', alpha=0.8)
        axes[1, 0].imshow(display_image, alpha=0.4)
        axes[1, 0].set_title('Combined Attention', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
        cbar3.set_label('Attention Weight', fontsize=12)
        
        # Attention-weighted image
        weighted_image = self._apply_attention_to_image(display_image, combined_att)
        axes[1, 1].imshow(weighted_image)
        axes[1, 1].set_title('Attention-Weighted Image', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Attention statistics
        self._plot_attention_statistics(axes[1, 2], channel_att, spatial_att, combined_att)
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"Attention visualization saved to: {save_path}")
        else:
            plt.show()
    
    def visualize_tps_transformation(self, original_image: Union[torch.Tensor, np.ndarray],
                                   transformed_image: Union[torch.Tensor, np.ndarray],
                                   control_points: Optional[torch.Tensor] = None,
                                   grid_size: int = 20, save_path: Optional[str] = None,
                                   title: str = "TPS Transformation Visualization") -> None:
        """Advanced TPS transformation visualization with control points and deformation grid"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Process images
        orig_img = self._prepare_image_for_display(original_image)
        trans_img = self._prepare_image_for_display(transformed_image)
        
        # Original image with grid
        axes[0, 0].imshow(orig_img)
        self._add_deformation_grid(axes[0, 0], orig_img.shape[:2], grid_size, color='cyan', alpha=0.6)
        if control_points is not None:
            self._plot_control_points(axes[0, 0], control_points, orig_img.shape[:2])
        axes[0, 0].set_title('Original Image with Control Grid', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Transformed image
        axes[0, 1].imshow(trans_img)
        axes[0, 1].set_title('TPS Transformed Image', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Difference map
        if orig_img.shape == trans_img.shape:
            diff_map = np.abs(orig_img.astype(float) - trans_img.astype(float)).mean(axis=2)
            im_diff = axes[1, 0].imshow(diff_map, cmap='hot')
            axes[1, 0].set_title('Transformation Difference Map', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im_diff, ax=axes[1, 0], shrink=0.8)
        
        # Transformation analysis
        self._plot_transformation_analysis(axes[1, 1], original_image, transformed_image, control_points)
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"TPS visualization saved to: {save_path}")
        else:
            plt.show()
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float],
                           val_metrics: List[Dict], save_path: Optional[str] = None,
                           title: str = "Training Progress") -> None:
        """Enhanced training curves with multiple metrics and smooth styling"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves with smoothing
        axes[0, 0].plot(epochs, self._smooth_curve(train_losses), 'b-', linewidth=3, 
                       label='Train Loss', alpha=0.8)
        axes[0, 0].fill_between(epochs, self._smooth_curve(train_losses), alpha=0.3, color='blue')
        axes[0, 0].plot(epochs, self._smooth_curve(val_losses), 'r-', linewidth=3, 
                       label='Val Loss', alpha=0.8)
        axes[0, 0].fill_between(epochs, self._smooth_curve(val_losses), alpha=0.3, color='red')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Extract metrics
        metrics_data = self._extract_metrics_from_history(val_metrics)
        
        # mAP curve
        if 'mAP' in metrics_data:
            mAPs = metrics_data['mAP']
            axes[0, 1].plot(epochs, self._smooth_curve(mAPs), 'g-', linewidth=3, alpha=0.8)
            axes[0, 1].fill_between(epochs, self._smooth_curve(mAPs), alpha=0.3, color='green')
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('mAP', fontsize=12)
            axes[0, 1].set_title('Mean Average Precision', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Precision and Recall
        if 'precision' in metrics_data and 'recall' in metrics_data:
            precisions = metrics_data['precision']
            recalls = metrics_data['recall']
            axes[0, 2].plot(epochs, self._smooth_curve(precisions), 'purple', 
                           linewidth=3, label='Precision', alpha=0.8)
            axes[0, 2].plot(epochs, self._smooth_curve(recalls), 'orange', 
                           linewidth=3, label='Recall', alpha=0.8)
            axes[0, 2].set_xlabel('Epoch', fontsize=12)
            axes[0, 2].set_ylabel('Score', fontsize=12)
            axes[0, 2].set_title('Precision and Recall', fontsize=14, fontweight='bold')
            axes[0, 2].legend(fontsize=12)
            axes[0, 2].grid(True, alpha=0.3)
        
        # F1-Score
        if 'f1_score' in metrics_data:
            f1_scores = metrics_data['f1_score']
            axes[1, 0].plot(epochs, self._smooth_curve(f1_scores), 'brown', linewidth=3, alpha=0.8)
            axes[1, 0].fill_between(epochs, self._smooth_curve(f1_scores), alpha=0.3, color='brown')
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('F1-Score', fontsize=12)
            axes[1, 0].set_title('F1-Score', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate schedule (if available)
        if 'learning_rate' in metrics_data:
            lrs = metrics_data['learning_rate']
            axes[1, 1].semilogy(epochs, lrs, 'darkorange', linewidth=3, alpha=0.8)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Training summary statistics
        self._plot_training_summary(axes[1, 2], train_losses, val_losses, metrics_data)
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"Training curves saved to: {save_path}")
        else:
            plt.show()
    
    def create_confusion_matrix(self, y_true: List[int], y_pred: List[int],
                              save_path: Optional[str] = None,
                              title: str = "Confusion Matrix",
                              normalize: bool = True) -> None:
        """Enhanced confusion matrix with statistical annotations"""
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = cm_norm
            fmt = '.2f'
            cmap = 'Blues'
        else:
            cm_display = cm
            fmt = 'd'
            cmap = 'Blues'
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Confusion matrix heatmap
        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap=cmap, 
                   xticklabels=self.class_names[:len(np.unique(y_true))], 
                   yticklabels=self.class_names[:len(np.unique(y_true))],
                   ax=ax1, cbar_kws={'shrink': 0.8})
        
        ax1.set_title(f'{title} {"(Normalized)" if normalize else "(Counts)"}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        
        # Classification report visualization
        report = classification_report(y_true, y_pred, output_dict=True, 
                                     target_names=self.class_names[:len(np.unique(y_true))])
        self._plot_classification_report(ax2, report)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"Confusion matrix saved to: {save_path}")
        else:
            plt.show()
    
    def create_model_comparison_plot(self, results: Dict[str, Dict], 
                                   metrics: List[str] = None,
                                   save_path: Optional[str] = None,
                                   title: str = "Model Comparison") -> None:
        """Create comprehensive model comparison visualization"""
        
        if metrics is None:
            metrics = ['precision', 'recall', 'mAP', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        models = list(results.keys())
        
        # Bar chart comparison for each metric
        for i, metric in enumerate(metrics[:4]):
            if i >= len(axes):
                break
                
            values = []
            errors = []
            
            for model in models:
                model_results = results[model]
                if metric in model_results:
                    metric_data = model_results[metric]
                    if isinstance(metric_data, dict):
                        values.append(metric_data.get('mean', 0))
                        errors.append(metric_data.get('std', 0))
                    else:
                        values.append(metric_data)
                        errors.append(0)
                else:
                    values.append(0)
                    errors.append(0)
            
            bars = axes[i].bar(models, values, yerr=errors, capsize=5, 
                              alpha=0.8, color=self.colors[:len(models)])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            axes[i].set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric.upper(), fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"Model comparison plot saved to: {save_path}")
        else:
            plt.show()
    
    def create_ablation_study_plot(self, ablation_results: Dict[str, Any],
                                 save_path: Optional[str] = None,
                                 title: str = "Ablation Study Results") -> None:
        """Create ablation study visualization"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Component contribution analysis
        components = list(ablation_results.keys())
        contributions = [ablation_results[comp].get('mean_mAP', 0) for comp in components]
        
        bars = ax1.bar(components, contributions, alpha=0.8, 
                      color=self.colors[:len(components)])
        
        # Add value labels
        for bar, value in zip(bars, contributions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Component Contribution Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylabel('mAP', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Cumulative improvement
        cumulative_improvements = np.cumsum([contributions[0]] + 
                                           [contributions[i] - contributions[0] 
                                            for i in range(1, len(contributions))])
        
        ax2.plot(components, cumulative_improvements, 'o-', linewidth=3, 
                markersize=8, alpha=0.8)
        ax2.fill_between(components, cumulative_improvements, alpha=0.3)
        ax2.set_title('Cumulative Performance Improvement', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Cumulative mAP', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"Ablation study plot saved to: {save_path}")
        else:
            plt.show()
    
    # Helper methods
    def _prepare_image_for_display(self, image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Prepare image tensor/array for matplotlib display"""
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # Batch dimension
                image = image[0]
            if image.shape[0] <= 4:  # CHW format
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        
        # Handle different channel counts
        if image.shape[2] > 3:
            # Multi-spectral - take first 3 channels or create pseudo-RGB
            if image.shape[2] == 4:  # RGBN
                image = image[:, :, :3]  # Take RGB
            else:
                image = image[:, :, :3]
        
        # Normalize to 0-255 range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def _process_attention_map(self, attention: torch.Tensor, 
                             target_size: Tuple[int, int]) -> np.ndarray:
        """Process attention tensor for visualization"""
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()
        
        # Handle different attention formats
        if attention.ndim == 4:  # Batch, Channel, Height, Width
            attention = attention[0]  # Take first batch
        if attention.ndim == 3:  # Channel, Height, Width
            attention = np.mean(attention, axis=0)  # Average across channels
        
        # Resize to target size
        if attention.shape != target_size:
            attention = cv2.resize(attention, (target_size[1], target_size[0]))
        
        # Normalize to 0-1
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        return attention
    
    def _plot_bounding_boxes(self, ax, boxes, image_shape, box_type='prediction',
                           confidence_threshold=0.5, show_confidence=True):
        """Plot bounding boxes with styling"""
        h, w = image_shape
        
        for box in boxes:
            if len(box) == 0:
                continue
                
            if box_type == 'prediction' and len(box) >= 6:
                # [x1, y1, x2, y2, conf, class] or [class, x_center, y_center, w, h, conf]
                if len(box) == 6 and box[4] < 1.0:  # Confidence score format
                    x1, y1, x2, y2, conf, cls = box[:6]
                    if conf < confidence_threshold:
                        continue
                else:  # YOLO format [class, x_center, y_center, w, h]
                    cls, x_center, y_center, width, height = box[:5]
                    conf = box[5] if len(box) > 5 else 1.0
                    
                    # Convert to corner format
                    x1 = (x_center - width/2) * w
                    y1 = (y_center - height/2) * h
                    x2 = (x_center + width/2) * w
                    y2 = (y_center + height/2) * h
                
                color = self.colors[int(cls) % len(self.colors)]
                linestyle = '-'
                linewidth = 2
                
                # Plot box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=linewidth, edgecolor=color, facecolor='none',
                    linestyle=linestyle
                )
                ax.add_patch(rect)
                
                # Add label
                label = f"{self.class_names[int(cls)]}"
                if show_confidence and len(box) > 5:
                    label += f": {conf:.2f}"
                
                ax.text(x1, y1-5, label, color=color, fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
            elif box_type == 'target' and len(box) >= 5:
                # Ground truth format [class, x_center, y_center, w, h]
                cls, x_center, y_center, width, height = box[:5]
                
                # Convert to corner format
                x1 = (x_center - width/2) * w
                y1 = (y_center - height/2) * h
                x2 = (x_center + width/2) * w
                y2 = (y_center + height/2) * h
                
                # Plot target box with dashed line
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=3, edgecolor='black', facecolor='none', linestyle='--'
                )
                ax.add_patch(rect)
                
                # Add target label
                label = f"GT: {self.class_names[int(cls)]}"
                ax.text(x2-50, y2+15, label, color='black', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    def _add_prediction_legend(self, ax, predictions, targets):
        """Add legend for predictions and targets"""
        legend_elements = []
        
        if predictions is not None and len(predictions) > 0:
            legend_elements.append(plt.Line2D([0], [0], color='blue', lw=2, label='Predictions'))
        
        if targets is not None and len(targets) > 0:
            legend_elements.append(plt.Line2D([0], [0], color='black', lw=2, 
                                            linestyle='--', label='Ground Truth'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    def _add_deformation_grid(self, ax, image_shape, grid_size, color='white', alpha=0.7):
        """Add deformation grid overlay"""
        h, w = image_shape
        
        # Vertical lines
        for x in range(0, w, w // grid_size):
            ax.axvline(x=x, color=color, alpha=alpha, linewidth=1)
        
        # Horizontal lines
        for y in range(0, h, h // grid_size):
            ax.axhline(y=y, color=color, alpha=alpha, linewidth=1)
    
    def _plot_control_points(self, ax, control_points, image_shape):
        """Plot TPS control points"""
        if control_points is not None:
            h, w = image_shape
            
            if isinstance(control_points, torch.Tensor):
                points = control_points.cpu().numpy()
            else:
                points = control_points
            
            # Assume control points are in normalized coordinates
            points_x = points[:, 0] * w
            points_y = points[:, 1] * h
            
            ax.scatter(points_x, points_y, c='red', s=50, marker='o', 
                      edgecolors='white', linewidth=2, alpha=0.8, label='Control Points')
    
    def _apply_attention_to_image(self, image, attention_map):
        """Apply attention weighting to image"""
        # Normalize attention map
        attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Apply attention as multiplicative mask
        weighted_image = image.copy().astype(float)
        for c in range(weighted_image.shape[2]):
            weighted_image[:, :, c] *= attention_norm
        
        # Normalize back to 0-255
        weighted_image = (weighted_image / weighted_image.max() * 255).astype(np.uint8)
        
        return weighted_image
    
    def _plot_attention_statistics(self, ax, channel_att, spatial_att, combined_att):
        """Plot attention statistics"""
        stats_data = {
            'Channel': [channel_att.mean(), channel_att.std(), channel_att.max()],
            'Spatial': [spatial_att.mean(), spatial_att.std(), spatial_att.max()],
            'Combined': [combined_att.mean(), combined_att.std(), combined_att.max()]
        }
        
        x = np.arange(3)
        width = 0.25
        
        ax.bar(x - width, stats_data['Channel'], width, label='Channel', alpha=0.8)
        ax.bar(x, stats_data['Spatial'], width, label='Spatial', alpha=0.8)
        ax.bar(x + width, stats_data['Combined'], width, label='Combined', alpha=0.8)
        
        ax.set_xlabel('Statistic')
        ax.set_ylabel('Value')
        ax.set_title('Attention Map Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(['Mean', 'Std', 'Max'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_transformation_analysis(self, ax, original, transformed, control_points):
        """Plot transformation analysis statistics"""
        # Calculate transformation metrics
        if isinstance(original, torch.Tensor):
            orig_tensor = original
        else:
            orig_tensor = torch.from_numpy(original)
        
        if isinstance(transformed, torch.Tensor):
            trans_tensor = transformed
        else:
            trans_tensor = torch.from_numpy(transformed)
        
        # Calculate various transformation metrics
        mse = F.mse_loss(trans_tensor, orig_tensor).item()
        
        # Structural similarity (simplified)
        ssim_approx = 1 - mse / (orig_tensor.var().item() + 1e-8)
        
        metrics = ['MSE', 'SSIM (approx)', 'Control Points']
        values = [mse, ssim_approx, len(control_points) if control_points is not None else 0]
        
        bars = ax.bar(metrics, values, alpha=0.8, color=['red', 'green', 'blue'])
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Transformation Analysis')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    def _smooth_curve(self, data, window_size=5):
        """Apply smoothing to curves for better visualization"""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            smoothed.append(np.mean(data[start_idx:end_idx]))
        
        return smoothed
    
    def _extract_metrics_from_history(self, val_metrics):
        """Extract metrics from validation history"""
        metrics_data = {}
        
        if not val_metrics:
            return metrics_data
        
        # Get all metric names from first entry
        if len(val_metrics) > 0:
            sample_metrics = val_metrics[0]
            for key in sample_metrics.keys():
                metrics_data[key] = []
        
        # Extract values for each epoch
        for epoch_metrics in val_metrics:
            for key, value in epoch_metrics.items():
                if isinstance(value, dict):
                    metrics_data[key].append(value.get('mean', value.get('macro', 0)))
                else:
                    metrics_data[key].append(value)
        
        return metrics_data
    
    def _plot_training_summary(self, ax, train_losses, val_losses, metrics_data):
        """Plot training summary statistics"""
        summary_stats = {
            'Final Train Loss': train_losses[-1] if train_losses else 0,
            'Final Val Loss': val_losses[-1] if val_losses else 0,
            'Best mAP': max(metrics_data.get('mAP', [0])),
            'Best F1': max(metrics_data.get('f1_score', [0]))
        }
        
        metrics = list(summary_stats.keys())
        values = list(summary_stats.values())
        
        bars = ax.bar(metrics, values, alpha=0.8, color=self.colors[:len(metrics)])
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Training Summary', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_classification_report(self, ax, report):
        """Plot classification report as heatmap"""
        # Extract metrics for each class
        classes = [key for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]
        
        metrics_matrix = []
        for cls in classes:
            metrics_matrix.append([
                report[cls]['precision'],
                report[cls]['recall'],
                report[cls]['f1-score']
            ])
        
        metrics_matrix = np.array(metrics_matrix)
        
        # Create heatmap
        sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=['Precision', 'Recall', 'F1-Score'],
                   yticklabels=classes, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title('Classification Report', fontsize=14, fontweight='bold')

# Utility functions for standalone use
def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                        val_metrics: List[Dict], save_path: Optional[str] = None) -> None:
    """Standalone function for plotting training curves"""
    visualizer = Visualizer()
    visualizer.plot_training_curves(train_losses, val_losses, val_metrics, save_path)

def visualize_predictions(image, predictions, targets=None, class_names=None, 
                         save_path=None) -> None:
    """Standalone function for prediction visualization"""
    visualizer = Visualizer(class_names)
    visualizer.plot_predictions(image, predictions, targets, save_path)

def plot_attention_maps(image, channel_attention, spatial_attention, 
                       save_path=None) -> None:
    """Standalone function for attention visualization"""
    visualizer = Visualizer()
    visualizer.visualize_attention_maps(image, channel_attention, spatial_attention, save_path)

def visualize_transformations(original_image, transformed_image, control_points=None,
                            save_path=None) -> None:
    """Standalone function for transformation visualization"""
    visualizer = Visualizer()
    visualizer.visualize_tps_transformation(original_image, transformed_image, 
                                          control_points, save_path=save_path)

def create_comparison_plots(results: Dict[str, Dict], save_path=None) -> None:
    """Standalone function for model comparison plots"""
    visualizer = Visualizer()
    visualizer.create_model_comparison_plot(results, save_path=save_path)

def save_experiment_plots(experiment_data: Dict[str, Any], save_dir: str) -> None:
    """Save all experiment visualization plots"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = Visualizer()
    
    # Training curves
    if 'training_history' in experiment_data:
        history = experiment_data['training_history']
        visualizer.plot_training_curves(
            history.get('train_losses', []),
            history.get('val_losses', []),
            history.get('val_metrics', []),
            save_path=save_dir / 'training_curves.png'
        )
    
    # Model comparison
    if 'model_results' in experiment_data:
        visualizer.create_model_comparison_plot(
            experiment_data['model_results'],
            save_path=save_dir / 'model_comparison.png'
        )
    
    # Ablation study
    if 'ablation_results' in experiment_data:
        visualizer.create_ablation_study_plot(
            experiment_data['ablation_results'],
            save_path=save_dir / 'ablation_study.png'
        )
    
    logger.info(f"All experiment plots saved to: {save_dir}")

if __name__ == "__main__":
    # Test visualization functionality
    print("Testing visualization utilities...")
    
    try:
        # Test with dummy data
        visualizer = Visualizer(['Cotton', 'Rice', 'Corn'])
        
        # Create dummy image and predictions
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        dummy_predictions = torch.tensor([
            [100, 100, 200, 200, 0.9, 0],  # [x1, y1, x2, y2, conf, class]
            [300, 300, 400, 400, 0.8, 1]
        ])
        dummy_targets = torch.tensor([
            [0, 0.3, 0.3, 0.2, 0.2],  # [class, x_center, y_center, w, h]
            [1, 0.7, 0.7, 0.2, 0.2]
        ])
        
        # Test prediction visualization
        print("✅ Creating prediction visualization...")
        visualizer.plot_predictions(dummy_image, dummy_predictions, dummy_targets)
        
        # Test training curves
        print("✅ Creating training curves...")
        dummy_train_losses = [1.0, 0.8, 0.6, 0.4, 0.3]
        dummy_val_losses = [1.1, 0.9, 0.7, 0.5, 0.4]
        dummy_metrics = [
            {'mAP': 0.5, 'precision': 0.6, 'recall': 0.5, 'f1_score': 0.55},
            {'mAP': 0.6, 'precision': 0.7, 'recall': 0.6, 'f1_score': 0.65},
            {'mAP': 0.7, 'precision': 0.8, 'recall': 0.7, 'f1_score': 0.75},
            {'mAP': 0.75, 'precision': 0.82, 'recall': 0.73, 'f1_score': 0.77},
            {'mAP': 0.78, 'precision': 0.85, 'recall': 0.75, 'f1_score': 0.80}
        ]
        
        visualizer.plot_training_curves(dummy_train_losses, dummy_val_losses, dummy_metrics)
        
        print("✅ Visualization tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
    
    print("Visualization tests completed!")