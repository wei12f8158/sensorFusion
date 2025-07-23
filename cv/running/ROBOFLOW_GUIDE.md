# Roboflow Guide for Model Improvement

## Overview
Roboflow is a powerful platform for data annotation, model training, and deployment. This guide shows you how to use Roboflow to improve your YOLO model.

## Why Roboflow?

### **Advantages**
- ✅ **Web-based annotation** (no installation needed)
- ✅ **AI-assisted annotation** (saves 50-70% time)
- ✅ **Built-in training** with GPU acceleration
- ✅ **Team collaboration** features
- ✅ **Version control** for datasets
- ✅ **Easy export** to multiple formats

### **Perfect for Your Case**
- **Bottle detection improvement** (32% → 70-85%)
- **Quick annotation** with AI assistance
- **Professional training** platform
- **Easy model deployment**

## Step-by-Step Workflow

### **Step 1: Prepare Your Data**

```bash
cd /home/wei/sensorFusion/cv/running

# Prepare data for Roboflow
python3 roboflow_workflow.py --prepare

# Collect additional data
python3 roboflow_workflow.py --collect
```

**What this does:**
- Creates `roboflow_data/` directory
- Copies existing images
- Creates upload scripts
- Prepares configuration files

### **Step 2: Create Roboflow Account**

1. **Go to**: https://app.roboflow.com
2. **Sign up** for free account
3. **Verify email** address
4. **Get API key** from account settings

### **Step 3: Create Project**

1. **Click "Create Project"**
2. **Project name**: `sensor_fusion_improvement`
3. **Task**: Object Detection
4. **Format**: YOLO
5. **Classes**: 9 (apple, ball, bottle, clip, glove, lid, plate, spoon, tape_spool)

### **Step 4: Upload Images**

#### **Manual Upload**
1. Go to your project
2. Click "Upload Images"
3. Select all images from `roboflow_data/images/`
4. Wait for upload to complete

#### **API Upload** (Automated)
```bash
# Install Roboflow
pip install roboflow

# Edit upload script
nano roboflow_data/upload_to_roboflow.py
# Replace YOUR_API_KEY and YOUR_WORKSPACE

# Run upload
cd roboflow_data
python3 upload_to_roboflow.py
```

### **Step 5: Annotate Data**

#### **Annotation Guidelines**

**Priority 1: Bottle Class**
- Draw **tight bounding boxes** around bottles
- Include **different angles** and lighting
- Focus on **interaction scenarios** (glove + bottle)
- Ensure **consistent annotation style**

**Other Classes**
- Maintain **consistent bounding boxes**
- Include **all visible objects**
- Quality over quantity

#### **Annotation Tips**

1. **Use AI Assistance**:
   - Roboflow has pre-trained models
   - Can auto-detect common objects
   - Saves significant time

2. **Quality Control**:
   - Review all annotations
   - Ensure bounding boxes are tight
   - Check class labels are correct

3. **Focus Areas**:
   - **Bottle detection** (currently 32% confidence)
   - **Interaction scenarios** (glove reaching for bottle)
   - **Different lighting conditions**

### **Step 6: Train Model**

#### **Training Configuration**

1. **Go to "Train" tab**
2. **Select model**: YOLOv8
3. **Training parameters**:
   - **Epochs**: 100
   - **Batch size**: 16
   - **Image size**: 640
   - **Learning rate**: 0.001

#### **Advanced Settings**

```yaml
# Training configuration
model: yolov8n.pt  # or yolov8s.pt for better accuracy
epochs: 100
batch_size: 16
imgsz: 640
lr0: 0.001
patience: 15
save_period: 10

# Data augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10.0
translate: 0.1
scale: 0.5
fliplr: 0.5
mosaic: 1.0
```

### **Step 7: Evaluate Results**

#### **Training Metrics**
- **mAP@0.5**: Should be > 0.85
- **Precision**: Should be > 0.80
- **Recall**: Should be > 0.80
- **Bottle class**: Should show significant improvement

#### **Expected Improvements**
```
Metric          | Before | After
----------------|--------|-------
Bottle mAP@0.5  | 0.32   | 0.75-0.85
Bottle Precision| 0.32   | 0.70-0.80
Bottle Recall   | 0.30   | 0.75-0.85
Overall mAP@0.5 | 0.87   | 0.90-0.95
```

### **Step 8: Download Model**

#### **Manual Download**
1. Go to "Export" tab
2. Select "YOLOv8" format
3. Click "Download"
4. Extract the ZIP file

#### **API Download**
```bash
# Create download script
python3 roboflow_workflow.py --download

# Edit script with your API key
nano download_roboflow_model.py

# Run download
python3 download_roboflow_model.py
```

### **Step 9: Deploy Improved Model**

#### **Replace Current Model**
```bash
# Copy new model to your directory
cp downloaded_model/best.pt yolo8_19_UL-New_day2_fullLabTopCam_60epochs_improved.pt

# Update config.yaml
nano ../../config.yaml
# Change weightsFile_rpi to new model path
```

#### **Test Improved Model**
```bash
# Test the improved model
python3 runImage.py

# Verify bottle detection improved
python3 test_universal_object_fix.py
```

## Roboflow Features

### **1. AI-Assisted Annotation**
- **Pre-trained models** for common objects
- **Auto-detect** objects in images
- **Smart suggestions** for bounding boxes
- **Batch processing** capabilities

### **2. Data Augmentation**
- **Built-in augmentation** tools
- **Custom augmentation** pipelines
- **Quality control** features
- **Version management**

### **3. Model Training**
- **GPU acceleration** (much faster than local)
- **Hyperparameter optimization**
- **Model comparison** tools
- **Performance analytics**

### **4. Team Collaboration**
- **Multiple annotators** can work simultaneously
- **Quality control** workflows
- **Progress tracking**
- **Comment system**

## Cost Considerations

### **Free Tier**
- **1,000 images** per project
- **Basic training** features
- **Standard support**

### **Paid Plans**
- **Pro**: $50/month (10,000 images)
- **Enterprise**: Custom pricing
- **GPU training** included

### **For Your Project**
- **Free tier** should be sufficient
- **~500-1000 images** for good improvement
- **Focus on quality** over quantity

## Best Practices

### **1. Data Quality**
- **High-quality images** (good lighting, focus)
- **Diverse scenarios** (angles, lighting, backgrounds)
- **Consistent annotation** style
- **Quality control** review

### **2. Annotation Strategy**
- **Start with AI assistance**
- **Review and correct** AI suggestions
- **Focus on bottle class** (your priority)
- **Include edge cases** (partial occlusion, unusual angles)

### **3. Training Strategy**
- **Start with YOLOv8n** (faster training)
- **Increase epochs** if needed
- **Monitor validation** metrics
- **Use early stopping** to prevent overfitting

### **4. Evaluation**
- **Test on unseen data**
- **Compare with baseline** model
- **Focus on bottle** performance
- **Real-world testing** in your lab

## Troubleshooting

### **Common Issues**

1. **Low bottle confidence**:
   - Collect more bottle images
   - Improve annotation quality
   - Increase training epochs

2. **Overfitting**:
   - Reduce epochs
   - Increase data augmentation
   - Add more validation data

3. **Poor localization**:
   - Improve bounding box accuracy
   - Collect more diverse angles
   - Use smaller learning rate

### **Performance Tips**

1. **Use GPU training** (much faster)
2. **Batch upload** images
3. **Use AI assistance** for annotation
4. **Regular backups** of your work

## Expected Timeline

### **Week 1: Setup & Data Collection**
- Create Roboflow account
- Prepare and upload data
- Start annotation

### **Week 2: Annotation & Training**
- Complete annotation
- Train model
- Evaluate results

### **Week 3: Testing & Deployment**
- Download improved model
- Test in your environment
- Deploy to production

## Success Metrics

### **Quantitative**
- **Bottle confidence**: 32% → 70-85%
- **Detection rate**: 90% → 95%+
- **Localization accuracy**: Improved IoU

### **Qualitative**
- **Better bounding boxes** around bottles
- **More consistent** detections
- **Improved performance** in real-world scenarios

## Next Steps

1. **Start with data preparation**:
   ```bash
   python3 roboflow_workflow.py --prepare
   ```

2. **Collect additional data**:
   ```bash
   python3 roboflow_workflow.py --collect
   ```

3. **Create Roboflow account** and project

4. **Upload and annotate** your data

5. **Train and download** improved model

6. **Deploy and test** in your environment

**Roboflow will significantly accelerate your model improvement process!** 🚀 