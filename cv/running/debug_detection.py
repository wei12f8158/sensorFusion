#!/usr/bin/env python3
"""
Debug script to identify why YOLO model isn't detecting objects
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_config():
    """Check the configuration settings"""
    logger.info("=== CONFIGURATION CHECK ===")
    
    config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
    configs = config.get_config()
    
    # Check model path
    model_path = Path(configs['training']['weightsFile_rpi'])
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model exists: {model_path.exists()}")
    
    # Check dataset path
    dataset_path = Path(configs['training']['dataSetDir']) / configs['training']['dataSet']
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Dataset exists: {dataset_path.exists()}")
    
    # Check thresholds
    hand_thresh = configs['runTime']['distSettings']['handThreshold']
    obj_thresh = configs['runTime']['distSettings']['objectThreshold']
    logger.info(f"Hand threshold: {hand_thresh}")
    logger.info(f"Object threshold: {obj_thresh}")
    
    # Check class mapping
    class_map = configs['runTime']['distSettings']['classMap']
    hand_class = configs['runTime']['distSettings']['handClass']
    logger.info(f"Class mapping: {class_map}")
    logger.info(f"Hand class: {hand_class}")
    
    return configs

def test_model_loading(configs):
    """Test if the model can be loaded"""
    logger.info("=== MODEL LOADING TEST ===")
    
    try:
        from rpiModel import RaspberryPiModel
        
        model_path = configs['training']['weightsFile_rpi']
        dataset_path = configs['training']['dataSetDir'] + '/' + configs['training']['dataSet']
        
        # Get thresholds
        thresh = min(configs['runTime']['distSettings']['handThreshold'],
                    configs['runTime']['distSettings']['objectThreshold'])
        
        logger.info(f"Loading model: {model_path}")
        logger.info(f"Dataset file: {dataset_path}")
        logger.info(f"Confidence threshold: {thresh}")
        
        model = RaspberryPiModel(
            model_file=model_path,
            names_file=dataset_path,
            conf_thresh=thresh,
            iou_thresh=configs['runTime']['distSettings']['nmsIouThreshold'],
            v8=True,
            use_gpu=False,
            num_threads=4
        )
        
        logger.info("✓ Model loaded successfully")
        logger.info(f"Input size: {model.input_size}")
        
        return model
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return None

def test_inference_on_test_image(model, configs):
    """Test inference on a simple test image"""
    logger.info("=== INFERENCE TEST ===")
    
    try:
        # Create a simple test image (640x640 with some colored rectangles)
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add some colored rectangles to simulate objects
        cv2.rectangle(test_img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green rectangle
        cv2.rectangle(test_img, (400, 300), (500, 400), (255, 0, 0), -1)  # Blue rectangle
        
        logger.info(f"Test image shape: {test_img.shape}")
        
        # Run inference
        from utils import get_image_tensor
        full_image, net_image, pad = get_image_tensor(test_img, model.input_size[0])
        logger.info(f"Preprocessed image shape: {net_image.shape}")
        
        # Run model
        pred = model.forward(net_image, with_nms=True)
        logger.info(f"Raw predictions shape: {pred.shape if hasattr(pred, 'shape') else 'No shape'}")
        logger.info(f"Raw predictions type: {type(pred)}")
        
        if len(pred) > 0:
            logger.info(f"✓ Found {len(pred)} detections")
            for i, det in enumerate(pred):
                logger.info(f"  Detection {i}: {det}")
        else:
            logger.info("✗ No detections found")
            
        return pred
        
    except Exception as e:
        logger.error(f"✗ Inference test failed: {e}")
        return None

def test_camera_capture():
    """Test if camera can capture images"""
    logger.info("=== CAMERA CAPTURE TEST ===")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("✗ Cannot open camera")
            return None
            
        ret, frame = cap.read()
        if not ret:
            logger.error("✗ Cannot read from camera")
            cap.release()
            return None
            
        logger.info(f"✓ Camera working - captured image shape: {frame.shape}")
        cap.release()
        return frame
        
    except Exception as e:
        logger.error(f"✗ Camera test failed: {e}")
        return None

def test_imx500_camera():
    """Test IMX500 camera if available"""
    logger.info("=== IMX500 CAMERA TEST ===")
    
    try:
        from picamera2 import Picamera2
        
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (1920, 1080)},
            controls={"FrameRate": 30},
            buffer_count=12
        )
        picam2.start(config, show_preview=False)
        
        # Capture a frame
        frame = picam2.capture_array()
        logger.info(f"✓ IMX500 camera working - captured image shape: {frame.shape}")
        
        picam2.stop()
        return frame
        
    except Exception as e:
        logger.error(f"✗ IMX500 camera test failed: {e}")
        return None

def main():
    """Main debugging function"""
    logger.info("Starting YOLO detection debugging...")
    
    # Check configuration
    configs = check_config()
    
    # Test model loading
    model = test_model_loading(configs)
    if model is None:
        logger.error("Cannot proceed without a working model")
        return
    
    # Test inference on test image
    test_inference_on_test_image(model, configs)
    
    # Test camera capture
    test_camera_capture()
    
    # Test IMX500 camera
    test_imx500_camera()
    
    logger.info("=== DEBUGGING COMPLETE ===")
    logger.info("Check the logs above for any issues.")

if __name__ == "__main__":
    main() 