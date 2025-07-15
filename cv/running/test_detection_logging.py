#!/usr/bin/env python3
"""
Test script to run detection with detailed logging
This will help you see exactly what's happening during the detection process
"""

import os
import sys
import logging
import cv2
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Set up logging to show everything in the terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def test_detection():
    """Test detection with detailed logging"""
    
    # Load configuration
    config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
    configs = config.get_config()
    
    logger = logging.getLogger("test_detection")
    logger.info("=== Starting Detection Test ===")
    
    # Check if inference is enabled
    if not configs['debugs']['runInfer']:
        logger.warning("Inference is disabled in config. Set runInfer: True in config.yaml")
        return
    
    # Check model path
    model_path = os.path.join(configs['training']['weightsDir'], configs['training']['weightsFile_tpu'])
    logger.info(f"Model path: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Check if we have a test image
    test_image_path = configs['debugs'].get('imgSrc', '')
    if test_image_path and os.path.exists(test_image_path):
        logger.info(f"Using test image: {test_image_path}")
        image = cv2.imread(test_image_path)
    else:
        logger.info("No test image specified, using camera")
        # Try to open camera
        cam_id = configs['runTime']['camId']
        logger.info(f"Opening camera {cam_id}")
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            logger.error(f"Could not open camera {cam_id}")
            return
        
        ret, image = cap.read()
        cap.release()
        if not ret:
            logger.error("Could not read from camera")
            return
    
    logger.info(f"Image loaded: shape={image.shape}, dtype={image.dtype}")
    
    # Initialize the inference system
    from modelRunTime import modelRunTime
    
    # Determine device
    import platform
    machine = platform.machine()
    if machine == "aarch64":
        device = "rpi"  # Assume Raspberry Pi
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    try:
        infer = modelRunTime(configs, device)
        logger.info("Model loaded successfully")
        
        # Run inference
        logger.info("Running inference...")
        results, processed_image = infer.runInference(image)
        
        if isinstance(results, int):
            logger.error(f"Inference failed with code: {results}")
            return
        
        logger.info(f"Inference completed. Results type: {type(results)}")
        if hasattr(results, 'shape'):
            logger.info(f"Results shape: {results.shape}")
        
        if len(results) > 0:
            logger.info(f"Found {len(results)} detections:")
            for i, det in enumerate(results):
                logger.info(f"  Detection {i}: {det}")
        else:
            logger.info("No detections found")
        
        # Test distance calculation
        from distance import distanceCalculator
        dist_calc = distanceCalculator(
            configs['training']['imageSize'],
            configs['runTime']['distSettings']
        )
        
        logger.info("Testing distance calculation...")
        valid = dist_calc.loadData(results)
        logger.info(f"Distance calculation valid: {valid}")
        
        if valid:
            logger.info(f"Final results:")
            logger.info(f"  Hands detected: {dist_calc.nHands}")
            logger.info(f"  Objects detected: {dist_calc.nNonHand}")
            logger.info(f"  Best object class: {int(dist_calc.grabObject[5])}")
            logger.info(f"  Best object confidence: {dist_calc.grabObject[4]:.3f}")
            logger.info(f"  Hand confidence: {dist_calc.handConf:.3f}")
            logger.info(f"  Distance: {dist_calc.bestDist:.1f}mm")
        
        # Clean up
        if device == "tpu":
            infer.exit()
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("=== Detection Test Complete ===")

if __name__ == "__main__":
    test_detection() 