# src/training/losses.py
"""
Enhanced loss functions for CBAM-STN-TPS-YOLO
Comprehensive loss implementations for agricultural object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

class CIoULoss(nn.Module):
    """Complete Intersection over Union Loss with aspect ratio penalty"""
    
    def __init__(self, eps: float = 1e-8, reduction: str = 'mean'):
        super(CIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: [N, 4] (x1, y1, x2, y2)
            target_boxes: [N, 4] (x1, y1, x2, y2)
        
        Returns:
            CIoU loss value
        """
        # Ensure valid box format
        pred_boxes = torch.clamp(pred_boxes, min=0, max=1)
        target_boxes = torch.clamp(target_boxes, min=0, max=1)
        
        # Calculate intersection
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + self.eps)
        
        # Enclosing box (smallest box containing both boxes)
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        enclose_w = enclose_x2 - enclose_x1 + self.eps
        enclose_h = enclose_y2 - enclose_y1 + self.eps
        c_squared = enclose_w ** 2 + enclose_h ** 2
        
        # Distance between centers
        pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        
        d_squared = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
        
        # Aspect ratio penalty
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0] + self.eps
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1] + self.eps
        target_w = target_boxes[:, 2] - target_boxes[:, 0] + self.eps
        target_h = target_boxes[:, 3] - target_boxes[:, 1] + self.eps
        
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2
        )
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)
        
        # CIoU
        ciou = iou - d_squared / c_squared - alpha * v
        ciou_loss = 1 - ciou
        
        if self.reduction == 'mean':
            return ciou_loss.mean()
        elif self.reduction == 'sum':
            return ciou_loss.sum()
        else:
            return ciou_loss

class DIoULoss(nn.Module):
    """Distance Intersection over Union Loss"""
    
    def __init__(self, eps: float = 1e-8, reduction: str = 'mean'):
        super(DIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Calculate DIoU loss"""
        # Calculate IoU
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area
        
        iou = inter_area / (union_area + self.eps)
        
        # Enclosing box diagonal
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        c_squared = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + self.eps
        
        # Center distance
        pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        
        d_squared = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
        
        # DIoU
        diou = iou - d_squared / c_squared
        diou_loss = 1 - diou
        
        if self.reduction == 'mean':
            return diou_loss.mean()
        elif self.reduction == 'sum':
            return diou_loss.sum()
        else:
            return diou_loss

class DistributedFocalLoss(nn.Module):
    """Enhanced Focal Loss for addressing severe class imbalance in agricultural datasets"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 reduction: str = 'mean', label_smoothing: float = 0.0):
        super(DistributedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, C] predictions (logits)
            targets: [N] ground truth labels
        
        Returns:
            Focal loss value
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = self._smooth_labels(targets, inputs.size(1))
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha weight
        if isinstance(self.alpha, (list, tuple)):
            alpha_t = torch.tensor(self.alpha)[targets]
        else:
            alpha_t = self.alpha
        
        # Focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _smooth_labels(self, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Apply label smoothing"""
        smoothed = torch.full_like(targets, self.label_smoothing / (num_classes - 1), dtype=torch.float)
        smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        return smoothed

class TverskyLoss(nn.Module):
    """Tversky Loss for handling class imbalance with adjustable precision/recall trade-off"""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, C, H, W] predictions
            targets: [N, C, H, W] targets
        
        Returns:
            Tversky loss value
        """
        # Apply softmax to predictions
        inputs = F.softmax(inputs, dim=1)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Tversky components
        true_pos = (inputs * targets).sum()
        false_neg = (targets * (1 - inputs)).sum()
        false_pos = ((1 - targets) * inputs).sum()
        
        # Tversky Index
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        
        return 1 - tversky

class ComboLoss(nn.Module):
    """Combination of multiple loss functions for comprehensive training"""
    
    def __init__(self, losses: Dict[str, nn.Module], weights: Dict[str, float]):
        super(ComboLoss, self).__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights
        
        # Validate weights
        assert set(losses.keys()) == set(weights.keys()), "Loss and weight keys must match"
        assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate weighted combination of losses"""
        total_loss = 0
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(predictions, targets)
            total_loss += self.weights[name] * loss_value
        
        return total_loss

class AdaptiveLoss(nn.Module):
    """Adaptive loss that adjusts weights based on training progress"""
    
    def __init__(self, base_losses: Dict[str, nn.Module], 
                 initial_weights: Dict[str, float],
                 adaptation_rate: float = 0.01):
        super(AdaptiveLoss, self).__init__()
        self.base_losses = nn.ModuleDict(base_losses)
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(weight))
            for name, weight in initial_weights.items()
        })
        self.adaptation_rate = adaptation_rate
        self.step_count = 0
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate adaptive weighted loss"""
        total_loss = 0
        individual_losses = {}
        
        # Calculate individual losses
        for name, loss_fn in self.base_losses.items():
            loss_value = loss_fn(predictions, targets)
            individual_losses[name] = loss_value
        
        # Update weights based on loss magnitudes (optional adaptation)
        if self.training and self.step_count % 100 == 0:
            self._adapt_weights(individual_losses)
        
        # Normalize weights
        total_weight = sum(torch.sigmoid(w) for w in self.weights.values())
        
        # Calculate weighted loss
        for name, loss_value in individual_losses.items():
            weight = torch.sigmoid(self.weights[name]) / total_weight
            total_loss += weight * loss_value
        
        self.step_count += 1
        return total_loss
    
    def _adapt_weights(self, losses: Dict[str, torch.Tensor]):
        """Adapt weights based on loss performance"""
        with torch.no_grad():
            # Increase weight for losses that are decreasing slowly
            for name, loss_value in losses.items():
                if hasattr(self, f'prev_{name}_loss'):
                    prev_loss = getattr(self, f'prev_{name}_loss')
                    if loss_value > prev_loss * 0.95:  # Not decreasing much
                        self.weights[name] += self.adaptation_rate
                
                setattr(self, f'prev_{name}_loss', loss_value.item())

class YOLOLoss(nn.Module):
    """Enhanced YOLO loss with multiple IoU variants and adaptive weighting"""
    
    def __init__(self, num_classes: int, 
                 lambda_coord: float = 5.0, 
                 lambda_noobj: float = 0.5, 
                 lambda_class: float = 1.0,
                 lambda_obj: float = 1.0,
                 iou_loss_type: str = 'ciou',
                 use_focal_loss: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 anchors: Optional[List[List[List[int]]]] = None,
                 grid_sizes: List[int] = [13, 26, 52]):
        super(YOLOLoss, self).__init__()
        
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.lambda_obj = lambda_obj
        self.use_focal_loss = use_focal_loss
        
        # IoU loss selection
        if iou_loss_type == 'ciou':
            self.iou_loss = CIoULoss()
        elif iou_loss_type == 'diou':
            self.iou_loss = DIoULoss()
        else:
            raise ValueError(f"Unknown IoU loss type: {iou_loss_type}")
        
        # Classification loss
        if use_focal_loss:
            self.cls_loss = DistributedFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.cls_loss = nn.CrossEntropyLoss()
        
        # Objectness loss
        self.obj_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Default anchors for 3 scales (optimized for agricultural objects)
        if anchors is None:
            self.anchors = [
                [[10, 13], [16, 30], [33, 23]],      # Small objects (leaves, flowers)
                [[30, 61], [62, 45], [59, 119]],     # Medium objects (fruits, stems)  
                [[116, 90], [156, 198], [373, 326]]  # Large objects (whole plants)
            ]
        else:
            self.anchors = anchors
            
        self.grid_sizes = grid_sizes
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, predictions: List[torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: List of tensors [batch_size, num_anchors*(5+num_classes), H, W]
            targets: [batch_size, max_targets, 6] (batch_idx, class, x, y, w, h)
        
        Returns:
            Dictionary containing total loss and component losses
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        
        total_loss = torch.tensor(0.0, device=device)
        loss_components = {
            'coord_loss': torch.tensor(0.0, device=device),
            'obj_loss': torch.tensor(0.0, device=device),
            'noobj_loss': torch.tensor(0.0, device=device),
            'cls_loss': torch.tensor(0.0, device=device)
        }
        
        for scale_idx, pred in enumerate(predictions):
            batch_size, _, H, W = pred.shape
            num_anchors = len(self.anchors[scale_idx])
            
            # Reshape predictions: [batch_size, num_anchors, H, W, 5+num_classes]
            pred = pred.view(batch_size, num_anchors, 5 + self.num_classes, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # Extract prediction components
            pred_xy = torch.sigmoid(pred[..., 0:2])    # Center coordinates
            pred_wh = pred[..., 2:4]                   # Width and height (log space)
            pred_conf = pred[..., 4]                   # Objectness confidence
            pred_cls = pred[..., 5:]                   # Class predictions
            
            # Build targets for this scale
            obj_mask, noobj_mask, target_boxes, target_cls, target_conf = self._build_targets(
                targets, self.anchors[scale_idx], H, W, device, scale_idx
            )
            
            # Coordinate loss (only for positive samples)
            if obj_mask.sum() > 0:
                # Convert predictions to absolute coordinates
                pred_boxes_abs = self._decode_predictions(pred_xy, pred_wh, self.anchors[scale_idx], H, W)
                
                # Apply IoU loss
                coord_loss = self.iou_loss(
                    pred_boxes_abs[obj_mask],
                    target_boxes[obj_mask]
                )
                loss_components['coord_loss'] += coord_loss
            
            # Objectness loss
            obj_loss = self.obj_loss(pred_conf, target_conf.float())
            obj_loss_pos = (obj_mask.float() * obj_loss).sum() / max(obj_mask.sum(), 1)
            obj_loss_neg = (noobj_mask.float() * obj_loss).sum() / max(noobj_mask.sum(), 1)
            
            loss_components['obj_loss'] += obj_loss_pos
            loss_components['noobj_loss'] += obj_loss_neg
            
            # Classification loss (only for positive samples)
            if obj_mask.sum() > 0:
                pred_cls_pos = pred_cls[obj_mask]
                target_cls_pos = target_cls[obj_mask].long()
                
                if self.use_focal_loss:
                    cls_loss = self.cls_loss(pred_cls_pos, target_cls_pos)
                else:
                    cls_loss = F.cross_entropy(pred_cls_pos, target_cls_pos)
                
                loss_components['cls_loss'] += cls_loss
        
        # Combine losses with weights
        total_loss = (
            self.lambda_coord * loss_components['coord_loss'] +
            self.lambda_obj * loss_components['obj_loss'] +
            self.lambda_noobj * loss_components['noobj_loss'] +
            self.lambda_class * loss_components['cls_loss']
        )
        
        loss_components['total_loss'] = total_loss
        return loss_components
    
    def _build_targets(self, targets: torch.Tensor, anchors: List[List[int]], 
                      H: int, W: int, device: torch.device, scale_idx: int) -> Tuple:
        """Build target tensors for loss calculation"""
        batch_size = targets.shape[0]
        num_anchors = len(anchors)
        
        # Initialize target tensors
        obj_mask = torch.zeros(batch_size, num_anchors, H, W, dtype=torch.bool, device=device)
        noobj_mask = torch.ones(batch_size, num_anchors, H, W, dtype=torch.bool, device=device)
        target_boxes = torch.zeros(batch_size, num_anchors, H, W, 4, device=device)
        target_cls = torch.zeros(batch_size, num_anchors, H, W, device=device, dtype=torch.long)
        target_conf = torch.zeros(batch_size, num_anchors, H, W, device=device)
        
        for batch_idx in range(batch_size):
            batch_targets = targets[batch_idx]
            batch_targets = batch_targets[batch_targets[:, 1] >= 0]  # Filter valid targets
            
            if len(batch_targets) == 0:
                continue
            
            for target in batch_targets:
                cls_id = int(target[1])
                gx, gy, gw, gh = target[2:6]
                
                # Convert to grid coordinates
                gi = int(gx * W)
                gj = int(gy * H)
                
                # Ensure within bounds
                gi = max(0, min(gi, W - 1))
                gj = max(0, min(gj, H - 1))
                
                # Find best matching anchor
                target_wh = torch.tensor([gw, gh], device=device)
                anchor_ious = []
                
                for anchor_idx, anchor in enumerate(anchors):
                    anchor_wh = torch.tensor([anchor[0] / W, anchor[1] / H], device=device)
                    iou = self._bbox_iou_wh(target_wh, anchor_wh)
                    anchor_ious.append(iou)
                
                best_anchor = torch.argmax(torch.tensor(anchor_ious))
                
                # Set positive target
                obj_mask[batch_idx, best_anchor, gj, gi] = True
                noobj_mask[batch_idx, best_anchor, gj, gi] = False
                
                # Target coordinates (relative to grid cell)
                target_boxes[batch_idx, best_anchor, gj, gi] = torch.tensor([
                    gx * W - gi,  # x offset in grid cell
                    gy * H - gj,  # y offset in grid cell
                    gw, gh        # normalized width and height
                ])
                
                target_cls[batch_idx, best_anchor, gj, gi] = cls_id
                target_conf[batch_idx, best_anchor, gj, gi] = 1.0
        
        return obj_mask, noobj_mask, target_boxes, target_cls, target_conf
    
    def _decode_predictions(self, pred_xy: torch.Tensor, pred_wh: torch.Tensor,
                          anchors: List[List[int]], H: int, W: int) -> torch.Tensor:
        """Decode predictions to absolute bounding boxes"""
        batch_size, num_anchors, grid_h, grid_w, _ = pred_xy.shape
        
        # Create grid coordinates
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).float().to(pred_xy.device)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(batch_size, num_anchors, -1, -1, -1)
        
        # Anchor tensors
        anchor_tensor = torch.tensor(anchors, device=pred_xy.device, dtype=torch.float32)
        anchor_tensor = anchor_tensor.view(1, num_anchors, 1, 1, 2)
        anchor_tensor = anchor_tensor.expand(batch_size, -1, H, W, -1)
        
        # Decode coordinates
        pred_xy = pred_xy + grid
        pred_wh = torch.exp(pred_wh) * anchor_tensor / torch.tensor([W, H], device=pred_xy.device)
        
        # Convert to corner format (x1, y1, x2, y2)
        pred_x1 = (pred_xy[..., 0] - pred_wh[..., 0] / 2) / W
        pred_y1 = (pred_xy[..., 1] - pred_wh[..., 1] / 2) / H
        pred_x2 = (pred_xy[..., 0] + pred_wh[..., 0] / 2) / W
        pred_y2 = (pred_xy[..., 1] + pred_wh[..., 1] / 2) / H
        
        pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
        return pred_boxes
    
    def _bbox_iou_wh(self, wh1: torch.Tensor, wh2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two boxes given only width and height"""
        w1, h1 = wh1
        w2, h2 = wh2
        
        inter_area = torch.min(w1, w2) * torch.min(h1, h2)
        union_area = w1 * h1 + w2 * h2 - inter_area
        
        return inter_area / (union_area + 1e-8)

# Utility functions
def calculate_loss(loss_type: str, **kwargs) -> nn.Module:
    """Factory function to create loss functions"""
    loss_registry = {
        'yolo': YOLOLoss,
        'ciou': CIoULoss,
        'diou': DIoULoss,
        'focal': DistributedFocalLoss,
        'tversky': TverskyLoss,
        'combo': ComboLoss,
        'adaptive': AdaptiveLoss
    }
    
    loss_type = loss_type.lower()
    if loss_type not in loss_registry:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_registry.keys())}")
    
    return loss_registry[loss_type](**kwargs)

def get_loss_weights_for_dataset(dataset_type: str) -> Dict[str, float]:
    """Get recommended loss weights for different agricultural datasets"""
    
    weight_presets = {
        'PGP': {
            'lambda_coord': 5.0,
            'lambda_obj': 1.0,
            'lambda_noobj': 0.5,
            'lambda_class': 2.0  # Higher due to class imbalance in agricultural data
        },
        'MelonFlower': {
            'lambda_coord': 10.0,  # Higher for small object detection
            'lambda_obj': 1.0,
            'lambda_noobj': 0.3,
            'lambda_class': 1.5
        },
        'GlobalWheat': {
            'lambda_coord': 5.0,
            'lambda_obj': 1.0,
            'lambda_noobj': 0.5,
            'lambda_class': 1.0  # Single class dataset
        },
        'default': {
            'lambda_coord': 5.0,
            'lambda_obj': 1.0,
            'lambda_noobj': 0.5,
            'lambda_class': 1.0
        }
    }
    
    return weight_presets.get(dataset_type, weight_presets['default'])

if __name__ == "__main__":
    # Test loss functions
    print("Testing enhanced loss functions...")
    
    # Test CIoU Loss
    ciou_loss = CIoULoss()
    pred_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]])
    target_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]])
    
    loss_value = ciou_loss(pred_boxes, target_boxes)
    print(f"✅ CIoU Loss: {loss_value:.4f}")
    
    # Test Focal Loss
    focal_loss = DistributedFocalLoss(num_classes=3)
    pred_logits = torch.randn(10, 3)
    target_labels = torch.randint(0, 3, (10,))
    
    focal_value = focal_loss(pred_logits, target_labels)
    print(f"✅ Focal Loss: {focal_value:.4f}")
    
    # Test YOLO Loss
    yolo_loss = YOLOLoss(num_classes=3)
    dummy_predictions = [torch.randn(2, 24, 13, 13)]  # 3 anchors * (5 + 3 classes) = 24
    dummy_targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.2, 0.2],  # batch_idx, class, x, y, w, h
        [1, 1, 0.3, 0.3, 0.1, 0.1]
    ])
    
    yolo_losses = yolo_loss(dummy_predictions, dummy_targets)
    print(f"✅ YOLO Loss: {yolo_losses['total_loss']:.4f}")
    
    print("All loss function tests passed! 🎉")