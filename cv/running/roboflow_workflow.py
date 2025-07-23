#!/usr/bin/env python3
"""
Prepare data for Roboflow annotation and training
"""

import os
import sys
import cv2
import json
import logging
from datetime import datetime
import shutil

# Add the parent directory to the path
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_roboflow_data():
    """Prepare data for Roboflow upload"""
    logger.info("=== PREPARE DATA FOR ROBOFLOW ===")
    
    try:
        # Create Roboflow directory
        roboflow_dir = "roboflow_data"
        if os.path.exists(roboflow_dir):
            shutil.rmtree(roboflow_dir)
        os.makedirs(roboflow_dir)
        
        # Create subdirectories
        images_dir = os.path.join(roboflow_dir, "images")
        os.makedirs(images_dir)
        
        # Copy existing images if available
        existing_dirs = [
            "training_data_collection",
            "../datasets/fine_tune_dataset/images/train",
            "../datasets/fine_tune_dataset/images/val"
        ]
        
        copied_count = 0
        for source_dir in existing_dirs:
            if os.path.exists(source_dir):
                logger.info(f"Copying images from: {source_dir}")
                for file in os.listdir(source_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_path = os.path.join(source_dir, file)
                        dst_path = os.path.join(images_dir, f"img_{copied_count:04d}.jpg")
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
        
        logger.info(f"Copied {copied_count} images to {images_dir}")
        
        # Create Roboflow configuration
        roboflow_config = {
            "project_name": "sensor_fusion_improvement",
            "classes": [
                "apple",
                "ball", 
                "bottle",
                "clip",
                "glove",
                "lid",
                "plate",
                "spoon",
                "tape_spool"
            ],
            "description": "Improve bottle detection for robotic arm application",
            "upload_instructions": [
                "1. Go to https://app.roboflow.com",
                "2. Create new project: 'sensor_fusion_improvement'",
                "3. Upload images from: roboflow_data/images/",
                "4. Annotate with focus on bottle class",
                "5. Export as YOLO format"
            ]
        }
        
        # Save configuration
        config_file = os.path.join(roboflow_dir, "roboflow_config.json")
        with open(config_file, 'w') as f:
            json.dump(roboflow_config, f, indent=2)
        
        logger.info(f"Roboflow config saved: {config_file}")
        
        # Create upload script
        upload_script = os.path.join(roboflow_dir, "upload_to_roboflow.py")
        with open(upload_script, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
Upload data to Roboflow using their API
"""

import os
from roboflow import Roboflow

def upload_to_roboflow():
    """Upload data to Roboflow"""
    # Initialize Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    
    # Create or get project
    project = rf.workspace("YOUR_WORKSPACE").project("sensor_fusion_improvement")
    
    # Upload images
    images_dir = "images"
    for image_file in os.listdir(images_dir):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_dir, image_file)
            project.upload(image_path)
            print(f"Uploaded: {image_file}")

if __name__ == "__main__":
    upload_to_roboflow()
''')
        
        logger.info(f"Upload script created: {upload_script}")
        
        # Create README
        readme_file = os.path.join(roboflow_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write('''# Roboflow Data Preparation

## Overview
This directory contains data prepared for Roboflow annotation and training.

## Files
- `images/`: Raw images for annotation
- `roboflow_config.json`: Configuration for Roboflow project
- `upload_to_roboflow.py`: Script to upload data to Roboflow

## Steps to Use Roboflow

### 1. Create Roboflow Account
- Go to https://app.roboflow.com
- Sign up for free account

### 2. Create Project
- Project name: `sensor_fusion_improvement`
- Object Detection task
- YOLO format

### 3. Upload Images
- Upload all images from `images/` directory
- Or use the upload script (requires API key)

### 4. Annotate Data
- Focus on bottle class (currently 32% confidence)
- Ensure precise bounding boxes
- Annotate all objects in each image

### 5. Train Model
- Use Roboflow's training platform
- Export trained model
- Download YOLO weights

## Annotation Guidelines

### Bottle Class (Priority)
- Draw tight bounding boxes around bottles
- Include different angles and lighting
- Focus on interaction scenarios (glove + bottle)

### Other Classes
- Maintain consistent annotation style
- Include all visible objects
- Quality over quantity

## Expected Improvements
- Bottle confidence: 32% → 70-85%
- Better localization accuracy
- Higher detection rates
''')
        
        logger.info(f"README created: {readme_file}")
        
        logger.info("=== ROBOFLOW DATA PREPARATION COMPLETE ===")
        logger.info(f"Data ready in: {roboflow_dir}")
        logger.info("Next steps:")
        logger.info("1. Go to https://app.roboflow.com")
        logger.info("2. Create project: 'sensor_fusion_improvement'")
        logger.info("3. Upload images from roboflow_data/images/")
        logger.info("4. Start annotating with focus on bottle class")
        
        return True
        
    except Exception as e:
        logger.error(f"Roboflow data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def collect_data_for_roboflow():
    """Collect additional data specifically for Roboflow"""
    logger.info("=== COLLECT DATA FOR ROBOFLOW ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Create data collection directory
        data_dir = "roboflow_data_collection"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Determine device
        import platform
        machine = platform.machine()
        
        if machine == "aarch64":
            device = "rpi"
        else:
            import torch
            device = "cpu" 
            if torch.cuda.is_available(): device = "cuda" 
            if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"
        
        # Import and initialize components
        from modelRunTime import modelRunTime
        from distance import distanceCalculator
        from display import displayHandObject
        
        model_runtime = modelRunTime(configs, device)
        distance_calc = distanceCalculator(configs['training']['imageSize'], configs['runTime']['distSettings'])
        display_obj = displayHandObject(configs)
        
        # Capture frame
        logger.info("Capturing frame...")
        img_src = configs['runTime']['imgSrc']
        use_imx500 = configs['runTime']['use_imx500']
        
        if img_src == "camera" and not use_imx500:
            # USB camera
            camera_indices = [0, 2, 3, 8]
            cap = None
            
            for cam_id in camera_indices:
                logger.info(f"Trying USB camera index {cam_id}...")
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"USB camera index {cam_id} works!")
                        break
                    else:
                        cap.release()
                        cap = None
                else:
                    cap.release()
                    cap = None
            
            if cap is None:
                logger.error("Failed to open any USB camera")
                return False
        else:
            # IMX500 camera
            from picamera2 import Picamera2
            picam2 = Picamera2()
            picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
            picam2.start()
            cap = picam2
        
        logger.info("=== ROBOFLOW DATA COLLECTION ===")
        logger.info("Press 's' to save frame with detections")
        logger.info("Press 'c' to capture frame without detections")
        logger.info("Press 'b' to capture bottle-focused frame")
        logger.info("Press 'q' to quit")
        
        frame_count = 0
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error("Failed to capture frame")
                break
            
            # Create display frame
            display_frame = frame.copy()
            
            # Run detection
            predictions = model_runtime.runInference(frame)
            
            if predictions is not None and len(predictions) > 0:
                # Process through distance calculator
                distance_calc.zeroData()
                valid = distance_calc.loadData(predictions)
                
                if valid:
                    # Draw detections
                    display_obj.draw(display_frame, distance_calc, valid)
                    
                    # Show detection info
                    cv2.putText(display_frame, f"Detections: {len(predictions)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    for i, pred in enumerate(predictions):
                        class_id = int(pred[5])
                        confidence = float(pred[4])
                        class_names = ['apple', 'ball', 'bottle', 'clip', 'glove', 'lid', 'plate', 'spoon', 'tape_spool']
                        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                        cv2.putText(display_frame, f"{class_name}: {confidence:.3f}", (10, 60 + i*30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Roboflow Data Collection", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Quitting data collection")
                break
            elif key == ord('s'):
                # Save frame with detections
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                frame_filename = f"frame_{timestamp}.jpg"
                frame_path = os.path.join(data_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                logger.info(f"Saved frame with detections: {frame_filename}")
                frame_count += 1
            elif key == ord('c'):
                # Capture frame without detections
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                frame_filename = f"frame_{timestamp}.jpg"
                frame_path = os.path.join(data_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                logger.info(f"Captured frame: {frame_filename}")
                frame_count += 1
            elif key == ord('b'):
                # Capture bottle-focused frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                frame_filename = f"bottle_focus_{timestamp}.jpg"
                frame_path = os.path.join(data_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                logger.info(f"Captured bottle-focused frame: {frame_filename}")
                frame_count += 1
        
        # Clean up
        cv2.destroyAllWindows()
        if img_src == "camera" and not use_imx500:
            cap.release()
        else:
            picam2.close()
        
        logger.info(f"=== ROBOFLOW DATA COLLECTION COMPLETE ===")
        logger.info(f"Collected {frame_count} frames in: {data_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Roboflow data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_roboflow_model():
    """Download trained model from Roboflow"""
    logger.info("=== DOWNLOAD ROBOFLOW MODEL ===")
    
    try:
        # Create download script
        download_script = "download_roboflow_model.py"
        with open(download_script, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
Download trained model from Roboflow
"""

import os
from roboflow import Roboflow

def download_model():
    """Download trained model from Roboflow"""
    # Initialize Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    
    # Get project
    project = rf.workspace("YOUR_WORKSPACE").project("sensor_fusion_improvement")
    
    # Get latest version
    version = project.version
    
    # Download YOLO weights
    version.download("yolov8")
    
    print("Model downloaded successfully!")
    print("Check the downloaded directory for weights")

if __name__ == "__main__":
    download_model()
''')
        
        logger.info(f"Download script created: {download_script}")
        logger.info("To use:")
        logger.info("1. Replace 'YOUR_API_KEY' with your Roboflow API key")
        logger.info("2. Replace 'YOUR_WORKSPACE' with your workspace name")
        logger.info("3. Run: python3 download_roboflow_model.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Download script creation failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Roboflow workflow for model improvement")
    parser.add_argument("--prepare", action="store_true", help="Prepare data for Roboflow")
    parser.add_argument("--collect", action="store_true", help="Collect data for Roboflow")
    parser.add_argument("--download", action="store_true", help="Create download script")
    parser.add_argument("--all", action="store_true", help="Do all steps")
    
    args = parser.parse_args()
    
    if args.prepare or args.all:
        prepare_roboflow_data()
    
    if args.collect or args.all:
        collect_data_for_roboflow()
    
    if args.download or args.all:
        download_roboflow_model()
    
    if not any([args.prepare, args.collect, args.download, args.all]):
        print("Usage:")
        print("  python3 roboflow_workflow.py --prepare   # Prepare data for Roboflow")
        print("  python3 roboflow_workflow.py --collect   # Collect data for Roboflow")
        print("  python3 roboflow_workflow.py --download  # Create download script")
        print("  python3 roboflow_workflow.py --all       # Do all steps") 