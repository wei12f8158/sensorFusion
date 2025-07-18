#!/usr/bin/env python3
"""
Simple test to verify YOLO model can detect objects
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

def main():
    """Simple test with a basic image"""
    logger.info("=== SIMPLE YOLO TEST ===")
    
    # Load config
    config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
    configs = config.get_config()
    
    # Model path
    model_path = configs['training']['weightsFile_rpi']
    dataset_path = configs['training']['dataSetDir'] + '/' + configs['training']['dataSet']
    
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {dataset_path}")
    
    # Check if files exist
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return
    
    try:
        from rpiModel import RaspberryPiModel
        
        # Load model with very low threshold
        model = RaspberryPiModel(
            model_file=model_path,
            names_file=dataset_path,
            conf_thresh=0.01,  # Very low threshold
            iou_thresh=0.45,
            v8=True,
            use_gpu=False,
            num_threads=4
        )
        
        logger.info("✓ Model loaded successfully")
        
        # Create a simple test image with a colored rectangle (simulating an object)
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add a red rectangle in the center (simulating an object)
        cv2.rectangle(test_img, (200, 200), (400, 400), (0, 0, 255), -1)
        
        # Add some text
        cv2.putText(test_img, "TEST OBJECT", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        logger.info(f"Test image shape: {test_img.shape}")
        
        # Save test image
        cv2.imwrite("test_image.jpg", test_img)
        logger.info("Saved test image as test_image.jpg")
        
        # Run inference
        from utils import get_image_tensor
        full_image, net_image, pad = get_image_tensor(test_img, model.input_size[0])
        
        logger.info(f"Preprocessed image shape: {net_image.shape}")
        logger.info(f"Padding: {pad}")
        
        # Run model
        pred = model.forward(net_image, with_nms=True)
        
        logger.info(f"Raw predictions shape: {pred.shape if hasattr(pred, 'shape') else 'No shape'}")
        logger.info(f"Raw predictions type: {type(pred)}")
        
        if len(pred) > 0:
            logger.info(f"✓ Found {len(pred)} detections!")
            for i, det in enumerate(pred):
                logger.info(f"  Detection {i}: {det}")
                logger.info(f"    Bbox: [{det[0]:.1f}, {det[1]:.1f}, {det[2]:.1f}, {det[3]:.1f}]")
                logger.info(f"    Confidence: {det[4]:.3f}")
                logger.info(f"    Class: {int(det[5])}")
        else:
            logger.warning("✗ No detections found - this might indicate an issue")
            
            # Try without NMS
            logger.info("Trying without NMS...")
            pred_raw = model.forward(net_image, with_nms=False)
            logger.info(f"Raw predictions without NMS: {pred_raw.shape if hasattr(pred_raw, 'shape') else 'No shape'}")
            if len(pred_raw) > 0:
                logger.info(f"Found {len(pred_raw)} raw predictions before NMS")
                for i, det in enumerate(pred_raw[:5]):  # Show first 5
                    logger.info(f"  Raw detection {i}: {det}")
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 