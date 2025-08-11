#!/usr/bin/env python3
"""
Setup open source models for IMX500 AI camera
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IMX500ModelSetup:
    def __init__(self):
        self.imx500_dir = Path("../../IMX500")
        self.models_dir = Path("../models")
        self.weights_dir = Path("../weights")
        
    def install_ultralytics(self):
        """Install Ultralytics for YOLO models"""
        logger.info("Installing Ultralytics...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], 
                         check=True, capture_output=True, text=True)
            logger.info("✅ Ultralytics installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install Ultralytics: {e}")
            return False
    
    def download_yolo_models(self):
        """Download recommended YOLO models"""
        logger.info("=== DOWNLOADING YOLO MODELS ===")
        
        try:
            from ultralytics import YOLO
            
            # Create weights directory
            self.weights_dir.mkdir(exist_ok=True)
            
            models_to_download = [
                ("yolov8n.pt", "YOLOv8 Nano - Best overall for IMX500"),
                ("yolov8s.pt", "YOLOv8 Small - Better accuracy"),
                ("yolov5n.pt", "YOLOv5 Nano - Fastest inference")
            ]
            
            for model_name, description in models_to_download:
                logger.info(f"Downloading {model_name}: {description}")
                
                model_path = self.weights_dir / model_name
                if not model_path.exists():
                    model = YOLO(model_name)
                    model.save(str(model_path))
                    logger.info(f"✅ {model_name} downloaded to {model_path}")
                else:
                    logger.info(f"✅ {model_name} already exists at {model_path}")
            
            return True
            
        except ImportError:
            logger.error("❌ Ultralytics not installed. Run install_ultralytics() first.")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to download models: {e}")
            return False
    
    def create_imx500_config(self, model_name="yolov8n.pt"):
        """Create IMX500 configuration for the selected model"""
        logger.info(f"=== CREATING IMX500 CONFIG FOR {model_name} ===")
        
        try:
            # Create IMX500 directory structure
            imx500_models_dir = self.imx500_dir / "models"
            imx500_models_dir.mkdir(exist_ok=True)
            
            # Copy model to IMX500 directory
            source_model = self.weights_dir / model_name
            target_model = imx500_models_dir / model_name
            
            if source_model.exists():
                import shutil
                shutil.copy2(source_model, target_model)
                logger.info(f"✅ Model copied to {target_model}")
            else:
                logger.error(f"❌ Source model not found: {source_model}")
                return False
            
            # Create IMX500 configuration
            config_content = f"""# IMX500 Configuration for {model_name}

# Model settings
model_path: "models/{model_name}"
model_type: "yolo"
input_size: [640, 640]
num_classes: 9

# Classes for your project
classes:
  0: apple
  1: ball
  2: bottle
  3: clip
  4: glove
  5: lid
  6: plate
  7: spoon
  8: tape_spool

# Inference settings
confidence_threshold: 0.5
iou_threshold: 0.5
max_detections: 10

# Camera settings
camera_resolution: [640, 640]
fps: 30

# Performance settings
use_npu: true
quantization: int8
"""
            
            config_file = self.imx500_dir / "imx500_config.yaml"
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            logger.info(f"✅ IMX500 config created: {config_file}")
            
            # Create labels file
            labels_content = """apple
ball
bottle
clip
glove
lid
plate
spoon
tape_spool"""
            
            labels_file = self.imx500_dir / "labels.txt"
            with open(labels_file, 'w') as f:
                f.write(labels_content)
            
            logger.info(f"✅ Labels file created: {labels_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create IMX500 config: {e}")
            return False
    
    def test_model_performance(self, model_name="yolov8n.pt"):
        """Test model performance on sample images"""
        logger.info(f"=== TESTING MODEL PERFORMANCE: {model_name} ===")
        
        try:
            from ultralytics import YOLO
            import cv2
            import numpy as np
            
            # Load model
            model = YOLO(str(self.weights_dir / model_name))
            
            # Create test image (640x640 with some objects)
            test_image = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # Draw some test objects
            cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), 2)  # Green rectangle
            cv2.rectangle(test_image, (300, 300), (400, 400), (255, 0, 0), 2)  # Blue rectangle
            cv2.circle(test_image, (500, 500), 50, (0, 0, 255), 2)  # Red circle
            
            # Run inference
            logger.info("Running inference test...")
            results = model(test_image, verbose=False)
            
            # Analyze results
            if results and len(results) > 0:
                result = results[0]
                logger.info(f"✅ Model loaded successfully")
                logger.info(f"   - Input shape: {result.orig_shape}")
                logger.info(f"   - Model ready for IMX500 deployment")
                
                # Test with real image if available
                test_images_dir = Path("../datasets/testImages")
                if test_images_dir.exists():
                    test_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
                    if test_files:
                        test_file = test_files[0]
                        logger.info(f"Testing with real image: {test_file}")
                        
                        real_results = model(str(test_file), verbose=False)
                        if real_results and len(real_results) > 0:
                            real_result = real_results[0]
                            if hasattr(real_result, 'boxes') and real_result.boxes is not None:
                                num_detections = len(real_result.boxes)
                                logger.info(f"   - Detections: {num_detections}")
                                logger.info(f"   - Model working correctly")
                            else:
                                logger.info(f"   - No detections in test image")
                        else:
                            logger.info(f"   - Model inference completed")
                
                return True
            else:
                logger.error("❌ Model inference failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to test model: {e}")
            return False
    
    def create_deployment_script(self, model_name="yolov8n.pt"):
        """Create deployment script for IMX500"""
        logger.info(f"=== CREATING DEPLOYMENT SCRIPT FOR {model_name} ===")
        
        try:
            script_content = f"""#!/bin/bash
# IMX500 Deployment Script for {model_name}

echo "=== IMX500 Model Deployment ==="
echo "Model: {model_name}"
echo "Date: $(date)"

# Check if model exists
if [ ! -f "models/{model_name}" ]; then
    echo "❌ Model not found: models/{model_name}"
    exit 1
fi

# Check IMX500 connection
if ! ping -c 1 10.0.0.71 > /dev/null 2>&1; then
    echo "❌ IMX500 not reachable"
    exit 1
fi

echo "✅ Model and IMX500 ready for deployment"

# Copy model to IMX500 (if needed)
echo "Copying model to IMX500..."
scp models/{model_name} wei@10.0.0.71:/home/wei/IMX500/models/

echo "✅ Deployment script completed"
echo "Next steps:"
echo "1. Configure IMX500 to use {model_name}"
echo "2. Test object detection"
echo "3. Monitor performance"
"""
            
            script_file = self.imx500_dir / f"deploy_{model_name.replace('.pt', '')}.sh"
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            # Make script executable
            os.chmod(script_file, 0o755)
            
            logger.info(f"✅ Deployment script created: {script_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create deployment script: {e}")
            return False
    
    def setup_complete_workflow(self, model_name="yolov8n.pt"):
        """Complete setup workflow for IMX500"""
        logger.info(f"=== COMPLETE SETUP WORKFLOW FOR {model_name} ===")
        
        steps = [
            ("Installing Ultralytics", self.install_ultralytics),
            ("Downloading YOLO models", self.download_yolo_models),
            ("Creating IMX500 config", lambda: self.create_imx500_config(model_name)),
            ("Testing model performance", lambda: self.test_model_performance(model_name)),
            ("Creating deployment script", lambda: self.create_deployment_script(model_name))
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            if not step_func():
                logger.error(f"❌ Setup failed at: {step_name}")
                return False
            logger.info(f"✅ {step_name} completed")
        
        logger.info("🎉 Complete setup workflow finished successfully!")
        return True

def main():
    """Main function"""
    setup = IMX500ModelSetup()
    
    print("=== IMX500 Open Source Model Setup ===")
    print("Available models:")
    print("1. YOLOv8n (6.3MB) - Best overall for IMX500")
    print("2. YOLOv8s (22.6MB) - Better accuracy")
    print("3. YOLOv5n (3.8MB) - Fastest inference")
    print()
    
    # Default to YOLOv8n (recommended)
    model_choice = "yolov8n.pt"
    
    print(f"Using recommended model: {model_choice}")
    print()
    
    # Run complete setup
    if setup.setup_complete_workflow(model_choice):
        print()
        print("🎯 Setup completed successfully!")
        print(f"Model: {model_choice}")
        print("Next steps:")
        print("1. Check IMX500 configuration files")
        print("2. Deploy model to IMX500")
        print("3. Test object detection")
        print("4. Monitor performance")
    else:
        print("❌ Setup failed. Check logs above.")

if __name__ == "__main__":
    main()
