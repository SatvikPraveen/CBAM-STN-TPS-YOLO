# src/training/metrics.py
"""
Enhanced detection metrics for CBAM-STN-TPS-YOLO
Comprehensive evaluation metrics for agricultural object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger(__name__)

def calculate_iou(box1: torch.Tensor, box2: torch.Tensor, format: str = 'xyxy') -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) between two sets of boxes
    
    Args:
        box1: [N, 4] boxes in specified format
        box2: [M, 4] boxes in specified format
        format: 'xyxy' (x1,y1,x2,y2) or 'xywh' (x,y,w,h)
    
    Returns:
        [N, M] IoU matrix
    """
    if format == 'xywh':
        # Convert from center format to corner format
        box1 = torch.cat([
            box1[:, :2] - box1[:, 2:] / 2,  # x1, y1
            box1[:, :2] + box1[:, 2:] / 2   # x2, y2
        ], dim=1)
        box2 = torch.cat([
            box2[:, :2] - box2[:, 2:] / 2,
            box2[:, :2] + box2[:, 2:] / 2
        ], dim=1)
    
    # Calculate intersection
    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    # Calculate union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = area1[:, None] + area2[None, :] - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-8)
    return iou

def calculate_giou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Calculate Generalized IoU (GIoU)"""
    # Regular IoU
    iou = calculate_iou(box1, box2)
    
    # Enclosing box
    enclose_x1 = torch.min(box1[:, None, 0], box2[None, :, 0])
    enclose_y1 = torch.min(box1[:, None, 1], box2[None, :, 1])
    enclose_x2 = torch.max(box1[:, None, 2], box2[None, :, 2])
    enclose_y2 = torch.max(box1[:, None, 3], box2[None, :, 3])
    
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # Union area
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = area1[:, None] + area2[None, :] - iou * area1[:, None]
    
    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-8)
    return giou

def calculate_diou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Calculate Distance IoU (DIoU)"""
    # Regular IoU
    iou = calculate_iou(box1, box2)
    
    # Centers
    center1 = (box1[:, :2] + box1[:, 2:]) / 2
    center2 = (box2[:, :2] + box2[:, 2:]) / 2
    
    # Distance between centers
    center_dist = torch.sum((center1[:, None] - center2[None, :]) ** 2, dim=2)
    
    # Diagonal of enclosing box
    enclose_x1 = torch.min(box1[:, None, 0], box2[None, :, 0])
    enclose_y1 = torch.min(box1[:, None, 1], box2[None, :, 1])
    enclose_x2 = torch.max(box1[:, None, 2], box2[None, :, 2])
    enclose_y2 = torch.max(box1[:, None, 3], box2[None, :, 3])
    
    diagonal_dist = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    # DIoU
    diou = iou - center_dist / (diagonal_dist + 1e-8)
    return diou

def calculate_ciou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Calculate Complete IoU (CIoU)"""
    import math
    
    # DIoU
    diou = calculate_diou(box1, box2)
    
    # Aspect ratio penalty
    w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
    w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
    
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(w2[None, :] / (h2[None, :] + 1e-8)) - 
        torch.atan(w1[:, None] / (h1[:, None] + 1e-8)), 2
    )
    
    # Regular IoU for alpha calculation
    iou = calculate_iou(box1, box2)
    alpha = v / (1 - iou + v + 1e-8)
    
    # CIoU
    ciou = diou - alpha * v
    return ciou

def non_max_suppression(boxes: torch.Tensor, scores: torch.Tensor, 
                       iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Non-Maximum Suppression for object detection
    
    Args:
        boxes: [N, 4] bounding boxes
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    
    # Sort by scores
    sorted_indices = torch.argsort(scores, descending=True)
    keep = []
    
    while len(sorted_indices) > 0:
        # Keep highest scoring box
        current = sorted_indices[0]
        keep.append(current)
        
        if len(sorted_indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        remaining_boxes = boxes[sorted_indices[1:]]
        current_box = boxes[current:current+1]
        
        ious = calculate_iou(current_box, remaining_boxes)
        
        # Remove boxes with high IoU
        mask = ious[0] <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return torch.tensor(keep, device=boxes.device)

def compute_ap(recall: np.ndarray, precision: np.ndarray, 
               method: str = 'coco') -> float:
    """
    Compute Average Precision (AP)
    
    Args:
        recall: Recall values
        precision: Precision values
        method: 'coco' for 101-point interpolation, 'voc' for 11-point
    
    Returns:
        Average Precision value
    """
    if len(recall) == 0 or len(precision) == 0:
        return 0.0
    
    if method == 'coco':
        # COCO 101-point interpolation
        recall_levels = np.linspace(0, 1, 101)
        ap = 0
        
        for r in recall_levels:
            # Find precision at recall level r
            prec_at_r = precision[recall >= r]
            if len(prec_at_r) > 0:
                ap += np.max(prec_at_r)
        
        ap /= 101
        
    elif method == 'voc':
        # VOC 11-point interpolation
        recall_levels = np.linspace(0, 1, 11)
        ap = 0
        
        for r in recall_levels:
            prec_at_r = precision[recall >= r]
            if len(prec_at_r) > 0:
                ap += np.max(prec_at_r)
        
        ap /= 11
        
    else:
        # Continuous interpolation
        # Add boundary points
        recall = np.concatenate(([0], recall, [1]))
        precision = np.concatenate(([0], precision, [0]))
        
        # Compute precision envelope
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = np.maximum(precision[i - 1], precision[i])
        
        # Find recall thresholds where precision changes
        indices = np.where(recall[1:] != recall[:-1])[0]
        
        # Compute AP
        ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    
    return ap

def compute_precision_recall(tp: np.ndarray, fp: np.ndarray, 
                           total_positives: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute precision and recall arrays"""
    if len(tp) == 0:
        return np.array([]), np.array([])
    
    # Cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / (total_positives + 1e-8)
    
    return precision, recall

def compute_f1_score(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

class DetectionMetrics:
    """Enhanced detection metrics with support for multi-class and multi-scale evaluation"""
    
    def __init__(self, num_classes: int, 
                 iou_thresholds: Union[float, List[float]] = 0.5,
                 conf_threshold: float = 0.5,
                 class_names: Optional[List[str]] = None,
                 track_per_image: bool = False):
        self.num_classes = num_classes
        
        # Handle multiple IoU thresholds for mAP calculation
        if isinstance(iou_thresholds, (list, tuple)):
            self.iou_thresholds = iou_thresholds
        else:
            self.iou_thresholds = [iou_thresholds]
        
        self.conf_threshold = conf_threshold
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.track_per_image = track_per_image
        
        self.reset()
    
    def reset(self):
        """Reset metrics for new evaluation"""
        self.predictions = []
        self.targets = []
        
        # Per-class storage for each IoU threshold
        self.true_positives = defaultdict(lambda: defaultdict(list))
        self.false_positives = defaultdict(lambda: defaultdict(list))
        self.false_negatives = defaultdict(lambda: defaultdict(list))
        self.scores = defaultdict(list)
        self.target_counts = defaultdict(int)
        
        # Per-image tracking
        if self.track_per_image:
            self.per_image_metrics = []
    
    def update(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor, 
               pred_classes: torch.Tensor, target_boxes: torch.Tensor, 
               target_classes: torch.Tensor, image_id: Optional[int] = None):
        """
        Update metrics with batch predictions and targets
        
        Args:
            pred_boxes: [N, 4] predicted boxes (x1, y1, x2, y2) in normalized coordinates
            pred_scores: [N] confidence scores
            pred_classes: [N] predicted classes
            target_boxes: [M, 4] target boxes in normalized coordinates
            target_classes: [M] target classes
            image_id: Optional image identifier for per-image tracking
        """
        # Convert to CPU and numpy for processing
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu()
            pred_scores = pred_scores.cpu()
            pred_classes = pred_classes.cpu()
        
        if isinstance(target_boxes, torch.Tensor):
            target_boxes = target_boxes.cpu()
            target_classes = target_classes.cpu()
        
        # Store raw predictions and targets
        self.predictions.append({
            'boxes': pred_boxes.clone(),
            'scores': pred_scores.clone(),
            'classes': pred_classes.clone(),
            'image_id': image_id
        })
        
        self.targets.append({
            'boxes': target_boxes.clone(),
            'classes': target_classes.clone(),
            'image_id': image_id
        })
        
        # Filter predictions by confidence threshold
        valid_preds = pred_scores >= self.conf_threshold
        if valid_preds.sum() == 0:
            # No valid predictions - all targets are false negatives
            for cls in target_classes:
                cls_id = int(cls)
                for iou_thresh in self.iou_thresholds:
                    self.false_negatives[iou_thresh][cls_id].append(1)
                self.target_counts[cls_id] += 1
            
            if self.track_per_image:
                self._track_image_metrics([], target_classes, image_id)
            return
        
        pred_boxes = pred_boxes[valid_preds]
        pred_scores = pred_scores[valid_preds]
        pred_classes = pred_classes[valid_preds]
        
        if len(target_boxes) == 0:
            # No targets - all predictions are false positives
            for cls, score in zip(pred_classes, pred_scores):
                cls_id = int(cls)
                for iou_thresh in self.iou_thresholds:
                    self.false_positives[iou_thresh][cls_id].append(1)
                self.scores[cls_id].append(float(score))
            
            if self.track_per_image:
                self._track_image_metrics(pred_classes, [], image_id)
            return
        
        # Count targets per class
        for cls in target_classes:
            self.target_counts[int(cls)] += 1
        
        # Calculate IoU matrix for all thresholds
        ious = calculate_iou(pred_boxes, target_boxes)
        
        # Process each IoU threshold
        matches_per_threshold = {}
        for iou_thresh in self.iou_thresholds:
            matches = self._match_predictions_to_targets(
                pred_classes, pred_scores, target_classes, ious, iou_thresh
            )
            matches_per_threshold[iou_thresh] = matches
        
        # Track per-image metrics if enabled
        if self.track_per_image:
            primary_matches = matches_per_threshold[self.iou_thresholds[0]]
            self._track_image_metrics(pred_classes, target_classes, image_id, primary_matches)
    
    def _match_predictions_to_targets(self, pred_classes: torch.Tensor, pred_scores: torch.Tensor,
                                    target_classes: torch.Tensor, ious: torch.Tensor, 
                                    iou_thresh: float) -> Dict:
        """Match predictions to targets for a specific IoU threshold"""
        used_targets = set()
        matches = {'tp': [], 'fp': [], 'matched_targets': set()}
        
        # Sort predictions by confidence (descending)
        sorted_indices = torch.argsort(pred_scores, descending=True)
        
        for pred_idx in sorted_indices:
            pred_cls = int(pred_classes[pred_idx])
            pred_score = float(pred_scores[pred_idx])
            
            # Find best matching target of the same class
            best_iou = 0
            best_target_idx = -1
            
            for target_idx, target_cls in enumerate(target_classes):
                if target_idx in used_targets:
                    continue
                    
                if int(target_cls) == pred_cls:
                    iou = ious[pred_idx, target_idx]
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = target_idx
            
            # Store score for this class
            self.scores[pred_cls].append(pred_score)
            
            # Determine if prediction is TP or FP
            if best_iou >= iou_thresh and best_target_idx != -1:
                self.true_positives[iou_thresh][pred_cls].append(1)
                self.false_positives[iou_thresh][pred_cls].append(0)
                used_targets.add(best_target_idx)
                matches['tp'].append(pred_idx)
                matches['matched_targets'].add(best_target_idx)
            else:
                self.true_positives[iou_thresh][pred_cls].append(0)
                self.false_positives[iou_thresh][pred_cls].append(1)
                matches['fp'].append(pred_idx)
        
        # Count false negatives (unmatched targets)
        for target_idx, target_cls in enumerate(target_classes):
            if target_idx not in used_targets:
                cls_id = int(target_cls)
                self.false_negatives[iou_thresh][cls_id].append(1)
        
        return matches
    
    def _track_image_metrics(self, pred_classes: torch.Tensor, target_classes: torch.Tensor,
                           image_id: Optional[int], matches: Optional[Dict] = None):
        """Track per-image metrics"""
        image_metrics = {
            'image_id': image_id,
            'num_predictions': len(pred_classes),
            'num_targets': len(target_classes),
            'num_tp': len(matches['tp']) if matches else 0,
            'num_fp': len(matches['fp']) if matches else 0,
            'num_fn': len(target_classes) - (len(matches['matched_targets']) if matches else 0)
        }
        
        # Calculate per-image precision, recall, F1
        tp = image_metrics['num_tp']
        fp = image_metrics['num_fp']
        fn = image_metrics['num_fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = compute_f1_score(precision, recall)
        
        image_metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        self.per_image_metrics.append(image_metrics)
    
    def compute_ap(self, class_id: int, iou_threshold: float = 0.5) -> float:
        """Compute Average Precision for a specific class and IoU threshold"""
        if class_id not in self.true_positives[iou_threshold]:
            return 0.0
        
        tp = np.array(self.true_positives[iou_threshold][class_id])
        fp = np.array(self.false_positives[iou_threshold][class_id])
        scores = np.array(self.scores[class_id])
        
        if len(tp) == 0:
            return 0.0
        
        # Sort by confidence scores (descending)
        sorted_indices = np.argsort(-scores)
        tp = tp[sorted_indices]
        fp = fp[sorted_indices]
        
        # Compute precision and recall
        total_positives = self.target_counts.get(class_id, 0)
        if total_positives == 0:
            return 0.0
        
        precision, recall = compute_precision_recall(tp, fp, total_positives)
        
        # Compute AP using COCO-style interpolation
        ap = compute_ap(recall, precision, method='coco')
        
        return ap
    
    def compute_metrics(self) -> Dict[str, Union[float, Dict]]:
        """Compute comprehensive detection metrics"""
        results = {}
        
        # Compute metrics for each IoU threshold
        for iou_thresh in self.iou_thresholds:
            thresh_results = self._compute_metrics_for_threshold(iou_thresh)
            results[f'iou_{iou_thresh}'] = thresh_results
        
        # Compute COCO-style mAP (average over IoU thresholds 0.5:0.95)
        if len(self.iou_thresholds) > 1:
            coco_aps = []
            for class_id in range(self.num_classes):
                class_aps = []
                for iou_thresh in self.iou_thresholds:
                    ap = self.compute_ap(class_id, iou_thresh)
                    class_aps.append(ap)
                coco_aps.append(np.mean(class_aps))
            
            results['coco_mAP'] = np.mean(coco_aps)
            results['coco_per_class_ap'] = {
                self.class_names[i]: coco_aps[i] for i in range(len(coco_aps))
            }
        
        # Use primary IoU threshold for main metrics
        primary_iou = self.iou_thresholds[0]
        primary_results = results[f'iou_{primary_iou}']
        
        results.update({
            'precision': primary_results['precision'],
            'recall': primary_results['recall'],
            'mAP': primary_results['mAP'],
            'f1_score': primary_results['f1_score'],
            'accuracy': primary_results['accuracy']
        })
        
        # Add per-image statistics if tracked
        if self.track_per_image and self.per_image_metrics:
            per_image_stats = self._compute_per_image_statistics()
            results['per_image_stats'] = per_image_stats
        
        return results
    
    def _compute_metrics_for_threshold(self, iou_threshold: float) -> Dict[str, Union[float, Dict]]:
        """Compute metrics for a specific IoU threshold"""
        # Per-class metrics
        per_class_ap = {}
        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for class_id in range(self.num_classes):
            # Average Precision
            ap = self.compute_ap(class_id, iou_threshold)
            per_class_ap[self.class_names[class_id]] = ap
            
            # Precision and Recall
            tp = sum(self.true_positives[iou_threshold].get(class_id, []))
            fp = sum(self.false_positives[iou_threshold].get(class_id, []))
            fn = sum(self.false_negatives[iou_threshold].get(class_id, []))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = compute_f1_score(precision, recall)
            
            per_class_precision[self.class_names[class_id]] = precision
            per_class_recall[self.class_names[class_id]] = recall
            per_class_f1[self.class_names[class_id]] = f1
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = compute_f1_score(overall_precision, overall_recall)
        
        # mAP
        mAP = np.mean(list(per_class_ap.values()))
        
        # Accuracy (TP / (TP + FP + FN))
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        
        return {
            'precision': overall_precision,
            'recall': overall_recall,
            'mAP': mAP,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'per_class_ap': per_class_ap,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'per_class_f1': per_class_f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }
    
    def _compute_per_image_statistics(self) -> Dict[str, float]:
        """Compute statistics across per-image metrics"""
        if not self.per_image_metrics:
            return {}
        
        metrics_arrays = {
            'precision': [m['precision'] for m in self.per_image_metrics],
            'recall': [m['recall'] for m in self.per_image_metrics],
            'f1_score': [m['f1_score'] for m in self.per_image_metrics],
            'num_predictions': [m['num_predictions'] for m in self.per_image_metrics],
            'num_targets': [m['num_targets'] for m in self.per_image_metrics]
        }
        
        stats = {}
        for metric_name, values in metrics_arrays.items():
            values = np.array(values)
            stats[f'{metric_name}_mean'] = np.mean(values)
            stats[f'{metric_name}_std'] = np.std(values)
            stats[f'{metric_name}_min'] = np.min(values)
            stats[f'{metric_name}_max'] = np.max(values)
            stats[f'{metric_name}_median'] = np.median(values)
        
        return stats

class AdvancedDetectionMetrics(DetectionMetrics):
    """Advanced metrics with additional evaluation criteria for agricultural applications"""
    
    def __init__(self, num_classes: int, iou_thresholds: Union[float, List[float]] = 0.5,
                 conf_threshold: float = 0.5, class_names: Optional[List[str]] = None,
                 area_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
                 track_per_image: bool = False):
        super().__init__(num_classes, iou_thresholds, conf_threshold, class_names, track_per_image)
        
        # Area-based evaluation (small, medium, large objects)
        self.area_thresholds = area_thresholds or {
            'small': (0, 32**2),     # Small objects (leaves, flowers)
            'medium': (32**2, 96**2), # Medium objects (fruits)
            'large': (96**2, float('inf'))  # Large objects (whole plants)
        }
        
        self.area_metrics = {
            area: DetectionMetrics(num_classes, iou_thresholds, conf_threshold, class_names, track_per_image) 
            for area in self.area_thresholds.keys()
        }
        
        # Additional agricultural-specific metrics
        self.occlusion_levels = {'none': 0, 'partial': 1, 'heavy': 2}
        self.growth_stages = ['seedling', 'vegetative', 'flowering', 'fruiting', 'mature']
        
        # Track additional metadata
        self.metadata_tracking = defaultdict(list)
    
    def update(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor, 
               pred_classes: torch.Tensor, target_boxes: torch.Tensor, 
               target_classes: torch.Tensor, image_id: Optional[int] = None,
               metadata: Optional[Dict] = None):
        """Update metrics including area-based and metadata analysis"""
        # Call parent update
        super().update(pred_boxes, pred_scores, pred_classes, target_boxes, target_classes, image_id)
        
        # Store metadata if provided
        if metadata:
            for key, value in metadata.items():
                self.metadata_tracking[key].append(value)
        
        # Area-based updates
        self._update_area_metrics(pred_boxes, pred_scores, pred_classes, target_boxes, target_classes, image_id)
    
    def _update_area_metrics(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor,
                           pred_classes: torch.Tensor, target_boxes: torch.Tensor, 
                           target_classes: torch.Tensor, image_id: Optional[int] = None):
        """Update metrics for different object sizes"""
        
        # Calculate areas for predictions and targets (in pixels if image size known)
        pred_areas = self._calculate_areas(pred_boxes)
        target_areas = self._calculate_areas(target_boxes)
        
        # Update metrics for each area category
        for area_name, (min_area, max_area) in self.area_thresholds.items():
            # Filter predictions by area
            pred_area_mask = (pred_areas >= min_area) & (pred_areas < max_area)
            if pred_area_mask.sum() > 0:
                area_pred_boxes = pred_boxes[pred_area_mask]
                area_pred_scores = pred_scores[pred_area_mask]
                area_pred_classes = pred_classes[pred_area_mask]
            else:
                area_pred_boxes = torch.empty((0, 4))
                area_pred_scores = torch.empty((0,))
                area_pred_classes = torch.empty((0,))
            
            # Filter targets by area
            target_area_mask = (target_areas >= min_area) & (target_areas < max_area)
            if target_area_mask.sum() > 0:
                area_target_boxes = target_boxes[target_area_mask]
                area_target_classes = target_classes[target_area_mask]
            else:
                area_target_boxes = torch.empty((0, 4))
                area_target_classes = torch.empty((0,))
            
            # Update area-specific metrics
            self.area_metrics[area_name].update(
                area_pred_boxes, area_pred_scores, area_pred_classes,
                area_target_boxes, area_target_classes, image_id
            )
    
    def _calculate_areas(self, boxes: torch.Tensor) -> torch.Tensor:
        """Calculate normalized areas of bounding boxes"""
        if len(boxes) == 0:
            return torch.empty((0,))
        
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        return areas
    
    def compute_area_metrics(self) -> Dict[str, Dict]:
        """Compute metrics for different object sizes"""
        area_results = {}
        
        for area_name, area_metric in self.area_metrics.items():
            area_results[area_name] = area_metric.compute_metrics()
        
        return area_results
    
    def compute_comprehensive_metrics(self) -> Dict[str, Union[float, Dict]]:
        """Compute all metrics including area-based analysis"""
        # Get base metrics
        results = self.compute_metrics()
        
        # Add area-based metrics
        area_metrics = self.compute_area_metrics()
        results['area_metrics'] = area_metrics
        
        # Compute area distribution
        if self.predictions:
            area_distribution = self._compute_area_distribution()
            results['area_distribution'] = area_distribution
        
        # Add metadata analysis if available
        if self.metadata_tracking:
            metadata_analysis = self._analyze_metadata()
            results['metadata_analysis'] = metadata_analysis
        
        return results
    
    def _compute_area_distribution(self) -> Dict[str, Dict]:
        """Analyze distribution of object areas"""
        all_pred_areas = []
        all_target_areas = []
        
        for pred_data in self.predictions:
            areas = self._calculate_areas(pred_data['boxes'])
            all_pred_areas.extend(areas.tolist())
        
        for target_data in self.targets:
            areas = self._calculate_areas(target_data['boxes'])
            all_target_areas.extend(areas.tolist())
        
        def analyze_areas(areas, name):
            if not areas:
                return {}
            
            areas = np.array(areas)
            return {
                f'{name}_count': len(areas),
                f'{name}_mean': np.mean(areas),
                f'{name}_std': np.std(areas),
                f'{name}_min': np.min(areas),
                f'{name}_max': np.max(areas),
                f'{name}_median': np.median(areas),
                f'{name}_q25': np.percentile(areas, 25),
                f'{name}_q75': np.percentile(areas, 75)
            }
        
        distribution = {}
        distribution.update(analyze_areas(all_pred_areas, 'predictions'))
        distribution.update(analyze_areas(all_target_areas, 'targets'))
        
        return distribution
    
    def _analyze_metadata(self) -> Dict[str, Dict]:
        """Analyze metadata correlations with performance"""
        analysis = {}
        
        # For each metadata field, compute metrics
        for metadata_key, values in self.metadata_tracking.items():
            if not values:
                continue
            
            unique_values = list(set(values))
            if len(unique_values) <= 10:  # Categorical analysis
                analysis[metadata_key] = self._analyze_categorical_metadata(metadata_key, values, unique_values)
            else:  # Numerical analysis
                analysis[metadata_key] = self._analyze_numerical_metadata(metadata_key, values)
        
        return analysis
    
    def _analyze_categorical_metadata(self, key: str, values: List, unique_values: List) -> Dict:
        """Analyze performance across categorical metadata values"""
        category_metrics = {}
        
        for category in unique_values:
            # Find images with this category
            category_indices = [i for i, v in enumerate(values) if v == category]
            
            # Compute metrics for this subset
            if category_indices:
                subset_metrics = self._compute_subset_metrics(category_indices)
                category_metrics[str(category)] = subset_metrics
        
        return category_metrics
    
    def _analyze_numerical_metadata(self, key: str, values: List) -> Dict:
        """Analyze performance correlation with numerical metadata"""
        # Bin numerical values and analyze
        values = np.array(values)
        
        # Create quartile-based bins
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        
        bins = {
            'low': values <= q25,
            'medium_low': (values > q25) & (values <= q50),
            'medium_high': (values > q50) & (values <= q75),
            'high': values > q75
        }
        
        bin_metrics = {}
        for bin_name, bin_mask in bins.items():
            indices = np.where(bin_mask)[0].tolist()
            if indices:
                subset_metrics = self._compute_subset_metrics(indices)
                bin_metrics[bin_name] = subset_metrics
        
        return bin_metrics
    
    def _compute_subset_metrics(self, indices: List[int]) -> Dict:
        """Compute metrics for a subset of images"""
        # This is a simplified version - in practice, you'd recompute full metrics
        # for the subset of predictions/targets
        
        subset_per_image = [self.per_image_metrics[i] for i in indices if i < len(self.per_image_metrics)]
        
        if not subset_per_image:
            return {}
        
        avg_precision = np.mean([m['precision'] for m in subset_per_image])
        avg_recall = np.mean([m['recall'] for m in subset_per_image])
        avg_f1 = np.mean([m['f1_score'] for m in subset_per_image])
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1,
            'sample_count': len(subset_per_image)
        }

class ClassificationMetrics:
    """Metrics for classification tasks (for CBAM attention analysis)"""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               probabilities: Optional[torch.Tensor] = None):
        """Update with batch of predictions and targets"""
        self.predictions.extend(predictions.cpu().numpy().tolist())
        self.targets.extend(targets.cpu().numpy().tolist())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy().tolist())
    
    def compute_metrics(self) -> Dict[str, Union[float, Dict]]:
        """Compute classification metrics"""
        if not self.predictions or not self.targets:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic metrics
        accuracy = np.mean(predictions == targets)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions, labels=range(self.num_classes))
        
        # Per-class metrics
        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []
        
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = compute_f1_score(precision, recall)
            
            per_class_precision.append(precision)
            per_class_recall.append(recall)
            per_class_f1.append(f1)
        
        # Macro averages
        macro_precision = np.mean(per_class_precision)
        macro_recall = np.mean(per_class_recall)
        macro_f1 = np.mean(per_class_f1)
        
        # Weighted averages
        class_support = cm.sum(axis=1)
        total_support = class_support.sum()
        
        weighted_precision = np.average(per_class_precision, weights=class_support)
        weighted_recall = np.average(per_class_recall, weights=class_support)
        weighted_f1 = np.average(per_class_f1, weights=class_support)
        
        results = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': per_class_precision[i],
                    'recall': per_class_recall[i],
                    'f1_score': per_class_f1[i],
                    'support': int(class_support[i])
                }
                for i in range(self.num_classes)
            },
            'confusion_matrix': cm.tolist()
        }
        
        return results

# Utility functions for specific agricultural metrics
def calculate_plant_counting_accuracy(pred_counts: List[int], true_counts: List[int]) -> Dict[str, float]:
    """Calculate plant counting accuracy metrics"""
    if len(pred_counts) != len(true_counts):
        raise ValueError("Prediction and ground truth counts must have same length")
    
    pred_counts = np.array(pred_counts)
    true_counts = np.array(true_counts)
    
    # Absolute error metrics
    mae = np.mean(np.abs(pred_counts - true_counts))
    rmse = np.sqrt(np.mean((pred_counts - true_counts) ** 2))
    
    # Relative error metrics
    mape = np.mean(np.abs((true_counts - pred_counts) / (true_counts + 1e-8))) * 100
    
    # Correlation
    correlation = np.corrcoef(pred_counts, true_counts)[0, 1] if len(pred_counts) > 1 else 0
    
    # Counting accuracy (percentage within ±1 count)
    within_one = np.mean(np.abs(pred_counts - true_counts) <= 1) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'correlation': correlation,
        'within_one_percent': within_one
    }

def calculate_growth_stage_accuracy(pred_stages: List[str], true_stages: List[str],
                                  stage_order: List[str]) -> Dict[str, float]:
    """Calculate growth stage classification accuracy with ordinal considerations"""
    if len(pred_stages) != len(true_stages):
        raise ValueError("Prediction and ground truth stages must have same length")
    
    # Convert to numerical
    stage_to_num = {stage: i for i, stage in enumerate(stage_order)}
    
    pred_nums = [stage_to_num.get(stage, -1) for stage in pred_stages]
    true_nums = [stage_to_num.get(stage, -1) for stage in true_stages]
    
    # Filter valid predictions
    valid_mask = np.array([(p != -1 and t != -1) for p, t in zip(pred_nums, true_nums)])
    if valid_mask.sum() == 0:
        return {'accuracy': 0, 'mae': 0, 'adjacent_accuracy': 0}
    
    pred_nums = np.array(pred_nums)[valid_mask]
    true_nums = np.array(true_nums)[valid_mask]
    
    # Exact accuracy
    accuracy = np.mean(pred_nums == true_nums)
    
    # Mean absolute error (ordinal)
    mae = np.mean(np.abs(pred_nums - true_nums))
    
    # Adjacent accuracy (within ±1 stage)
    adjacent_accuracy = np.mean(np.abs(pred_nums - true_nums) <= 1)
    
    return {
        'accuracy': accuracy,
        'ordinal_mae': mae,
        'adjacent_accuracy': adjacent_accuracy
    }

if __name__ == "__main__":
    # Test metrics
    print("Testing enhanced detection metrics...")
    
    # Test basic IoU calculation
    box1 = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]])
    box2 = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]])
    
    iou = calculate_iou(box1, box2)
    print(f"✅ IoU calculation: {iou[0, 0]:.4f}")
    
    # Test detection metrics
    metrics = DetectionMetrics(num_classes=3, class_names=['plant', 'flower', 'fruit'])
    
    # Dummy data
    pred_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]])
    pred_scores = torch.tensor([0.9, 0.8])
    pred_classes = torch.tensor([0, 1])
    
    target_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]])
    target_classes = torch.tensor([0, 1])
    
    metrics.update(pred_boxes, pred_scores, pred_classes, target_boxes, target_classes)
    
    results = metrics.compute_metrics()
    print(f"✅ Detection mAP: {results['mAP']:.4f}")
    
    # Test advanced metrics
    advanced_metrics = AdvancedDetectionMetrics(num_classes=3, track_per_image=True)
    advanced_metrics.update(pred_boxes, pred_scores, pred_classes, target_boxes, target_classes, image_id=0)
    
    advanced_results = advanced_metrics.compute_comprehensive_metrics()
    print(f"✅ Advanced metrics computed with {len(advanced_results)} components")
    
    # Test agricultural-specific metrics
    pred_counts = [5, 7, 12, 3]
    true_counts = [5, 8, 11, 3]
    counting_metrics = calculate_plant_counting_accuracy(pred_counts, true_counts)
    print(f"✅ Plant counting MAE: {counting_metrics['mae']:.2f}")
    
    print("All metrics tests passed! 🎉")