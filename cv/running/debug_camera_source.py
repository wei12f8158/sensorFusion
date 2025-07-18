#!/usr/bin/env python3
"""
Comprehensive debugging script for Camera Source Mode issues
"""

import os
import sys
import cv2
import numpy as np
import logging
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_configuration():
    """Check if we're in the right mode"""
    logger.info("=== CONFIGURATION CHECK ===")
    
    config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
    configs = config.get_config()
    
    use_imx500 = configs['runTime'].get('use_imx500', False)
    imx500_ai_camera = configs['runTime'].get('imx500_ai_camera', False)
    
    logger.info(f"use_imx500: {use_imx500}")
    logger.info(f"imx500_ai_camera: {imx500_ai_camera}")
    
    if use_imx500 and not imx500_ai_camera:
        logger.info("✓ Configuration is correct for Camera Source Mode")
        return configs
    else:
        logger.error("✗ Configuration is NOT correct for Camera Source Mode")
        logger.error("Please switch to Camera Source Mode first")
        return None

def test_camera_capture():
    """Test basic camera capture"""
    logger.info("=== CAMERA CAPTURE TEST ===")
    
    try:
        from picamera2 import Picamera2
        
        logger.info("Initializing IMX500 camera at 640x640...")
        picam2 = Picamera2()
        
        config = picam2.create_preview_configuration(
            main={"size": (640, 640)},
            controls={"FrameRate": 30},
            buffer_count=12
        )
        
        picam2.start(config, show_preview=False)
        logger.info("✓ Camera started successfully")
        
        # Capture a few frames
        for i in range(5):
            image = picam2.capture_array()
            logger.info(f"Frame {i+1}: Shape={image.shape}, dtype={image.dtype}")
            
            # Convert to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"capture_test_{i+1}.jpg", image_bgr)
                logger.info(f"  Saved as capture_test_{i+1}.jpg")
        
        picam2.stop()
        logger.info("✓ Camera capture test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Camera capture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading"""
    logger.info("=== MODEL LOADING TEST ===")
    
    try:
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        model_path = configs['training']['weightsFile_rpi']
        dataset_path = configs['training']['dataSetDir'] + '/' + configs['training']['dataSet']
        
        logger.info(f"Model path: {model_path}")
        logger.info(f"Dataset path: {dataset_path}")
        
        # Check if files exist
        if not os.path.exists(model_path):
            logger.error(f"✗ Model file not found: {model_path}")
            return None
            
        if not os.path.exists(dataset_path):
            logger.error(f"✗ Dataset file not found: {dataset_path}")
            return None
        
        from rpiModel import RaspberryPiModel
        
        # Get thresholds
        thresh = min(configs['runTime']['distSettings']['handThreshold'],
                    configs['runTime']['distSettings']['objectThreshold'])
        
        logger.info(f"Loading model with confidence threshold: {thresh}")
        
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
        logger.info(f"Model input size: {model.input_size}")
        
        return model
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_inference_pipeline():
    """Test the complete inference pipeline"""
    logger.info("=== INFERENCE PIPELINE TEST ===")
    
    try:
        # Load model
        model = test_model_loading()
        if model is None:
            return False
        
        # Create test image
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add some objects
        cv2.rectangle(test_img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green
        cv2.rectangle(test_img, (400, 300), (500, 400), (255, 0, 0), -1)  # Blue
        cv2.rectangle(test_img, (250, 250), (350, 350), (0, 0, 255), -1)  # Red
        
        cv2.imwrite("test_inference.jpg", test_img)
        logger.info("Created test image: test_inference.jpg")
        
        # Test preprocessing
        from utils import get_image_tensor
        full_image, net_image, pad = get_image_tensor(test_img, model.input_size[0])
        
        logger.info(f"Preprocessing results:")
        logger.info(f"  Full image shape: {full_image.shape}")
        logger.info(f"  Network image shape: {net_image.shape}")
        logger.info(f"  Padding: {pad}")
        
        # Test model forward pass
        logger.info("Running model forward pass...")
        pred = model.forward(net_image, with_nms=True)
        
        logger.info(f"Model output:")
        logger.info(f"  Type: {type(pred)}")
        logger.info(f"  Shape: {pred.shape if hasattr(pred, 'shape') else 'No shape'}")
        logger.info(f"  Length: {len(pred)}")
        
        if len(pred) > 0:
            logger.info("✓ Found detections!")
            for i, det in enumerate(pred):
                logger.info(f"  Detection {i}: {det}")
        else:
            logger.warning("✗ No detections found")
            
            # Try without NMS
            logger.info("Trying without NMS...")
            pred_raw = model.forward(net_image, with_nms=False)
            logger.info(f"Raw predictions: {pred_raw.shape if hasattr(pred_raw, 'shape') else 'No shape'}")
            logger.info(f"Raw predictions length: {len(pred_raw)}")
            
            if len(pred_raw) > 0:
                logger.info("Found raw predictions before NMS:")
                for i, det in enumerate(pred_raw[:10]):  # Show first 10
                    logger.info(f"  Raw {i}: {det}")
            else:
                logger.error("No raw predictions either - model might not be working")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Inference pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_camera_to_model_pipeline():
    """Test the complete pipeline from camera to model"""
    logger.info("=== CAMERA TO MODEL PIPELINE TEST ===")
    
    try:
        # Load model
        model = test_model_loading()
        if model is None:
            return False
        
        # Initialize camera
        from picamera2 import Picamera2
        
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 640)},
            controls={"FrameRate": 30},
            buffer_count=12
        )
        
        picam2.start(config, show_preview=False)
        logger.info("✓ Camera initialized")
        
        # Capture and process a few frames
        for frame_num in range(3):
            logger.info(f"\n--- Processing Frame {frame_num + 1} ---")
            
            # Capture image
            image = picam2.capture_array()
            logger.info(f"Captured image shape: {image.shape}")
            
            # Convert to BGR
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"camera_frame_{frame_num+1}.jpg", image_bgr)
                logger.info(f"Saved camera frame: camera_frame_{frame_num+1}.jpg")
            
            # Run inference
            from utils import get_image_tensor
            full_image, net_image, pad = get_image_tensor(image, model.input_size[0])
            
            logger.info(f"Preprocessed shape: {net_image.shape}")
            
            # Model inference
            pred = model.forward(net_image, with_nms=True)
            
            logger.info(f"Detections: {len(pred)}")
            if len(pred) > 0:
                for i, det in enumerate(pred):
                    logger.info(f"  Detection {i}: {det}")
            else:
                logger.info("  No detections")
        
        picam2.stop()
        logger.info("✓ Camera to model pipeline test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Camera to model pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_thresholds():
    """Check if thresholds might be too high"""
    logger.info("=== THRESHOLD CHECK ===")
    
    try:
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        hand_thresh = configs['runTime']['distSettings']['handThreshold']
        obj_thresh = configs['runTime']['distSettings']['objectThreshold']
        
        logger.info(f"Hand threshold: {hand_thresh}")
        logger.info(f"Object threshold: {obj_thresh}")
        
        if hand_thresh > 0.5 or obj_thresh > 0.5:
            logger.warning("⚠️  Thresholds might be too high!")
            logger.warning("Try lowering them to 0.1 or 0.01 for testing")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Threshold check failed: {e}")
        return False

def main():
    """Main debugging function"""
    logger.info("Starting comprehensive Camera Source Mode debugging...")
    
    # Check configuration
    configs = check_configuration()
    if configs is None:
        return
    
    # Run all tests
    tests = [
        ("Camera Capture", test_camera_capture),
        ("Model Loading", test_model_loading),
        ("Inference Pipeline", test_inference_pipeline),
        ("Camera to Model Pipeline", test_camera_to_model_pipeline),
        ("Threshold Check", check_thresholds),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            if test_name == "Model Loading":
                result = test_func()
                results[test_name] = result is not None
            else:
                results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DEBUGGING SUMMARY")
    logger.info(f"{'='*60}")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✓ All tests passed! Camera Source Mode should work.")
    else:
        logger.info("✗ Some tests failed. Check the logs above for details.")
        logger.info("\nCommon issues and solutions:")
        logger.info("1. Thresholds too high - lower them in config.yaml")
        logger.info("2. Model file not found - check the path")
        logger.info("3. Camera not accessible - check permissions")
        logger.info("4. Memory issues - try reducing buffer_count")

if __name__ == "__main__":
    main() 