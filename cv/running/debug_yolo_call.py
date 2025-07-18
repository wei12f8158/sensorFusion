#!/usr/bin/env python3
"""
Debug YOLO model call directly
"""

import os
import sys
import cv2
import numpy as np
import logging

# Add the parent directory to the path
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_yolo_call():
    """Debug YOLO model call directly"""
    logger.info("=== DEBUG YOLO CALL ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        model_path = configs['training']['weightsFile_rpi']
        
        # Load test image
        test_img = cv2.imread('../../cv/datasets/testImages/appleHand_hand_4.jpg')
        if test_img is None:
            logger.error("Test image not found")
            return False
        
        logger.info(f"Test image shape: {test_img.shape}")
        
        # Test 1: Direct YOLO call with original image
        logger.info("--- Test 1: Direct YOLO call with original image ---")
        from ultralytics import YOLO
        yolo_model = YOLO(model_path)
        
        # Call with different confidence thresholds
        for conf in [0.001, 0.01, 0.1, 0.5]:
            logger.info(f"Testing with conf={conf}")
            results = yolo_model(test_img, conf=conf, iou=0.9, verbose=False)
            
            logger.info(f"  Results: {len(results)}")
            if len(results) > 0:
                result = results[0]
                logger.info(f"  Boxes: {len(result.boxes)}")
                if len(result.boxes) > 0:
                    for i, box in enumerate(result.boxes):
                        logger.info(f"    Box {i}: class={int(box.cls)}, conf={float(box.conf):.3f}")
                else:
                    logger.info("    No boxes found")
            else:
                logger.info("  No results")
        
        # Test 2: Direct YOLO call with preprocessed image
        logger.info("--- Test 2: Direct YOLO call with preprocessed image ---")
        from utils import get_image_tensor
        full_image, net_image, pad = get_image_tensor(test_img, 640)
        
        # Convert to uint8
        net_image_uint8 = (net_image * 255).astype(np.uint8)
        
        for conf in [0.001, 0.01, 0.1, 0.5]:
            logger.info(f"Testing with conf={conf}")
            results = yolo_model(net_image_uint8, conf=conf, iou=0.9, verbose=False)
            
            logger.info(f"  Results: {len(results)}")
            if len(results) > 0:
                result = results[0]
                logger.info(f"  Boxes: {len(result.boxes)}")
                if len(result.boxes) > 0:
                    for i, box in enumerate(result.boxes):
                        logger.info(f"    Box {i}: class={int(box.cls)}, conf={float(box.conf):.3f}")
                else:
                    logger.info("    No boxes found")
            else:
                logger.info("  No results")
        
        return True
        
    except Exception as e:
        logger.error(f"Debug YOLO call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_yolo_call() 