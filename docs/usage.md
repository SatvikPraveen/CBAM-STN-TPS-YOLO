# Usage Guide

## Quick Start

### 1. Train a Model

```bash
# Train the complete model
python experiments/run_experiments.py --model CBAM-STN-TPS-YOLO --single

# Train with custom config
python -m src.training.trainer --config config/training_configs.yaml
```

### 2. Evaluate Model

```bash
python -m src.utils.evaluation --checkpoint results/best_cbam_stn_tps_yolo.pth
```

### 3. Run Inference

```bash
# Single image
python -m src.inference.predict \
    --checkpoint results/best_cbam_stn_tps_yolo.pth \
    --input image.jpg \
    --output results/

# Batch processing
python -m src.inference.predict \
    --checkpoint results/best_cbam_stn_tps_yolo.pth \
    --input images/ \
    --output results/predictions/
```

## Advanced Usage

### Reproduce Paper Results

```bash
# Run complete experimental suite
python experiments/run_experiments.py

# Statistical analysis
python experiments/statistical_analysis.py

# Ablation study
python experiments/ablation_study.py
```

### Custom Training

#### Modify Configuration

Edit `config/training_configs.yaml`:

```yaml
# Training parameters
epochs: 200
batch_size: 16
lr: 0.001

# Model parameters
num_control_points: 25
tps_reg_lambda: 0.05

# Data augmentation
augmentation:
  rotation_deg: 15
  shear_deg: 15
  crop_factor: 0.2
```

#### Custom Model Variants

```python
from src.models import create_model

# Create model variant
model = create_model(
    model_type='CBAM-STN-TPS-YOLO',
    num_classes=5,  # Custom classes
    num_control_points=30,  # More control points
    backbone_channels=[64, 128, 256, 512, 1024]  # Larger backbone
)
```

### Custom Dataset

#### 1. Create Dataset Class

```python
from src.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        # Load your annotations
        self.annotations = self._load_annotations()

    def __getitem__(self, idx):
        # Load image and targets
        image = self._load_image(idx)
        targets = self._load_targets(idx)
        return image, targets, path
```

#### 2. Update Data Loader

```python
from torch.utils.data import DataLoader

dataset = CustomDataset('data/custom')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

### Visualization

#### Training Curves

```python
from src.utils.visualization import Visualizer

viz = Visualizer()
viz.plot_training_curves(train_losses, val_losses, val_metrics)
```

#### Attention Maps

```python
# Visualize CBAM attention
viz.plot_attention_maps(image, channel_attention, spatial_attention)
```

#### TPS Transformation

```python
# Visualize TPS warping
viz.visualize_tps_transformation(original_image, transformed_image)
```

### Model Deployment

#### Export to ONNX

```python
import torch

model.eval()
dummy_input = torch.randn(1, 3, 512, 512)

torch.onnx.export(
    model, dummy_input, "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

#### TensorRT Optimization

```python
# For Jetson deployment
import tensorrt as trt

# Convert ONNX to TensorRT
# (Implementation depends on specific deployment needs)
```

### Experiment Tracking

#### Weights & Biases

```yaml
# In config
use_wandb: true
```

```python
import wandb

wandb.init(project="cbam-stn-tps-yolo")
# Training metrics automatically logged
```

#### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/Train', loss, epoch)
```

## API Reference

### Models

- `CBAM_STN_TPS_YOLO`: Complete model
- `create_model()`: Model factory function
- `CBAM`: Attention module
- `STN_TPS`: Spatial transformer with TPS

### Training

- `CBAMSTNTPSYOLOTrainer`: Training class
- `YOLOLoss`: Complete loss function
- `DetectionMetrics`: Evaluation metrics

### Data

- `PGPDataset`: Plant Growth & Phenotyping dataset
- `TestAugmentations`: Robustness testing
- `DatasetPreprocessor`: Data preparation

### Utils

- `Visualizer`: Plotting and visualization
- `ModelEvaluator`: Model evaluation
- `ModelPredictor`: Inference pipeline

## Examples

See [notebooks/](../notebooks/) for detailed examples:

- `data_exploration.ipynb`: Dataset analysis
- `model_analysis.ipynb`: Model behavior
- `results_visualization.ipynb`: Results plotting

## Performance Tips

### Memory Optimization

```python
# Use gradient checkpointing
torch.utils.checkpoint.checkpoint()

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
```

### Speed Optimization

```python
# Use DataLoader optimizations
dataloader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

### Multi-GPU Training

```python
# Data parallel
model = nn.DataParallel(model)

# Distributed training
import torch.distributed as dist
```

## Troubleshooting

### Common Issues

#### Low mAP Performance

1. Check learning rate
2. Verify data augmentation
3. Increase training epochs
4. Check label quality

#### Slow Training

1. Reduce batch size
2. Use mixed precision
3. Optimize data loading
4. Check GPU utilization

#### Memory Issues

1. Reduce batch size
2. Use gradient accumulation
3. Clear cache: `torch.cuda.empty_cache()`

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Verbose training
config['verbose'] = True
```
