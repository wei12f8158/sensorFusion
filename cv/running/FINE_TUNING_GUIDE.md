# Fine-tuning Guide for YOLO Model

## Overview
This guide helps you fine-tune your YOLO model to improve detection accuracy and fix localization issues.

## Why Fine-tune?
- **Better localization**: More accurate bounding boxes
- **Higher confidence**: More reliable detections
- **Environment adaptation**: Works better in your specific setup
- **Class-specific improvements**: Better detection for specific objects

## Step 1: Collect Training Data

### Using the Data Collection Script
```bash
cd /home/wei/sensorFusion/cv/running
python3 collect_training_data.py
```

**Controls:**
- Press `s` to save frame with detections
- Press `c` to capture frame without detections
- Press `q` to quit

### Manual Data Collection
1. **Capture diverse images** of your objects
2. **Vary lighting conditions** (bright, dim, shadows)
3. **Different angles** and positions
4. **Multiple objects** in same frame
5. **Background variations** (cluttered, clean)

## Step 2: Prepare Dataset

### Create Dataset Structure
```bash
python3 fine_tune_model.py --prepare
```

This creates:
```
datasets/fine_tune_dataset/
├── images/
│   ├── train/     # 80% of images
│   ├── val/       # 15% of images
│   └── test/      # 5% of images
└── labels/
    ├── train/     # YOLO format labels
    ├── val/
    └── test/
```

### YOLO Label Format
Each image needs a corresponding `.txt` file with:
```
class_id center_x center_y width height
```

Example:
```
6 0.5 0.3 0.2 0.15  # Plate at center
4 0.2 0.7 0.1 0.08  # Glove on left
```

## Step 3: Fine-tune Model

### Quick Fine-tuning
```bash
python3 fine_tune_model.py --fine-tune
```

### Manual Fine-tuning
```python
from ultralytics import YOLO

# Load current model
model = YOLO('current_model.pt')

# Fine-tune
results = model.train(
    data='dataset.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    lr0=0.001,
    patience=10
)
```

## Step 4: Evaluate Results

### Check Training Metrics
- **mAP (mean Average Precision)**: Should increase
- **Precision**: Higher is better
- **Recall**: Higher is better
- **Loss**: Should decrease

### Test Fine-tuned Model
```bash
# Update config.yaml with new model path
# Test with your application
python3 runImage.py
```

## Fine-tuning Parameters

### Learning Rate
- **Start with**: 0.001
- **If loss doesn't decrease**: Try 0.0001
- **If loss oscillates**: Try 0.00001

### Epochs
- **Start with**: 50
- **Monitor validation loss**: Stop when it plateaus
- **Use early stopping**: Prevents overfitting

### Batch Size
- **GPU memory dependent**: 16, 32, 64
- **CPU training**: 8, 16
- **Larger batch**: More stable training

## Advanced Techniques

### 1. Transfer Learning
```python
# Freeze early layers
model = YOLO('pretrained.pt')
for param in model.model.backbone.parameters():
    param.requires_grad = False
```

### 2. Data Augmentation
```python
# Add to training config
augmentation = {
    'hsv_h': 0.015,  # HSV-Hue augmentation
    'hsv_s': 0.7,    # HSV-Saturation augmentation
    'hsv_v': 0.4,    # HSV-Value augmentation
    'degrees': 0.0,  # Image rotation
    'translate': 0.1, # Image translation
    'scale': 0.5,    # Image scaling
    'shear': 0.0,    # Image shear
    'perspective': 0.0, # Image perspective
    'flipud': 0.0,   # Flip up-down
    'fliplr': 0.5,   # Flip left-right
    'mosaic': 1.0,   # Mosaic augmentation
    'mixup': 0.0,    # Mixup augmentation
}
```

### 3. Class Balancing
If some classes are underrepresented:
```python
# Use class weights
class_weights = [1.0, 2.0, 1.5, 1.0, 1.0, 1.5, 2.0, 1.0, 1.0]
```

## Troubleshooting

### Common Issues

1. **Overfitting**
   - Reduce epochs
   - Increase data augmentation
   - Add dropout layers

2. **Underfitting**
   - Increase epochs
   - Increase model capacity
   - Reduce regularization

3. **Poor Localization**
   - Collect more diverse data
   - Increase bounding box accuracy in labels
   - Use smaller learning rate

4. **Low Confidence**
   - Improve image quality
   - Better lighting conditions
   - More training data

### Performance Tips

1. **Use GPU** if available
2. **Monitor memory usage**
3. **Save checkpoints** regularly
4. **Use validation set** to prevent overfitting

## Expected Improvements

After fine-tuning, you should see:
- ✅ **Better bounding box accuracy**
- ✅ **Higher confidence scores**
- ✅ **Fewer false positives**
- ✅ **Better performance in your environment**
- ✅ **More consistent detections**

## Next Steps

1. **Test the fine-tuned model**
2. **Collect more data** if needed
3. **Iterate** on the process
4. **Deploy** the improved model

Remember: Fine-tuning is an iterative process. Start small and gradually improve! 