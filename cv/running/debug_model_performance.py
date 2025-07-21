#!/usr/bin/env python3
"""
Debug model performance to identify localization issues
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

def debug_model_performance():
    """Debug model performance with different confidence thresholds"""
    logger.info("=== MODEL PERFORMANCE DEBUG ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        model_path = configs['training']['weightsFile_rpi']
        
        # Load test image (use a saved frame from your camera)
        test_img = cv2.imread('test_frame_1.jpg')  # Use a recent frame
        if test_img is None:
            logger.error("Test image not found. Please save a frame first.")
            return False
        
        logger.info(f"Test image shape: {test_img.shape}")
        
        # Test with different confidence thresholds
        from ultralytics import YOLO
        yolo_model = YOLO(model_path)
        
        confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for conf in confidence_levels:
            logger.info(f"\n--- Testing with confidence threshold: {conf} ---")
            
            results = yolo_model(test_img, conf=conf, iou=0.6, verbose=False)
            
            if len(results) > 0:
                result = results[0]
                logger.info(f"  Detections: {len(result.boxes)}")
                
                for i, box in enumerate(result.boxes):
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    logger.info(f"    Detection {i}: class={class_id}, conf={confidence:.3f}")
                    logger.info(f"      Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                    
                    # Calculate center of bounding box
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    logger.info(f"      Center: ({center_x:.1f}, {center_y:.1f})")
            else:
                logger.info("  No detections")
        
        # Test with different IOU thresholds
        logger.info(f"\n--- Testing with different IOU thresholds ---")
        iou_levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for iou in iou_levels:
            results = yolo_model(test_img, conf=0.3, iou=iou, verbose=False)
            logger.info(f"  IOU {iou}: {len(results[0].boxes) if len(results) > 0 else 0} detections")
        
        return True
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_model_performance() 