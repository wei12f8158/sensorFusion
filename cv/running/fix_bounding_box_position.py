#!/usr/bin/env python3
"""
Fix bounding box position issues
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

def fix_bounding_box_position():
    """Fix bounding box position issues"""
    logger.info("=== FIX BOUNDING BOX POSITION ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        model_path = configs['training']['weightsFile_rpi']
        
        # Load test image
        test_img = cv2.imread('debug_frame_20250721_153537.jpg')
        if test_img is None:
            logger.error("Test image not found")
            return False
        
        logger.info(f"Test image shape: {test_img.shape}")
        
        # Run detection
        from ultralytics import YOLO
        yolo_model = YOLO(model_path)
        
        results = yolo_model(test_img, conf=0.3, iou=0.6, verbose=False)
        
        if len(results) > 0:
            result = results[0]
            logger.info(f"Detections: {len(result.boxes)}")
            
            # Create a copy for drawing
            img_with_boxes = test_img.copy()
            
            for i, box in enumerate(result.boxes):
                confidence = float(box.conf)
                class_id = int(box.cls)
                bbox = box.xyxy[0].cpu().numpy()
                
                logger.info(f"Detection {i}: class={class_id}, conf={confidence:.3f}")
                logger.info(f"  Raw bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                
                # Calculate center
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                logger.info(f"  Center: ({center_x:.1f}, {center_y:.1f})")
                
                # Draw bounding box
                color = (0, 255, 0) if class_id == 4 else (255, 255, 0)  # Green for hand, Cyan for plate
                cv2.rectangle(img_with_boxes, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            color, 2)
                cv2.circle(img_with_boxes, (int(center_x), int(center_y)), 5, color, -1)
                
                # Add label
                label = f"Class {class_id}: {confidence:.3f}"
                cv2.putText(img_with_boxes, label, 
                           (int(bbox[0]), int(bbox[1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save the result
            cv2.imwrite('fixed_bounding_boxes.jpg', img_with_boxes)
            logger.info("Saved fixed_bounding_boxes.jpg")
            
            # Compare with original debug image
            original_debug = cv2.imread('debug_annotated_20250721_153537.jpg')
            if original_debug is not None:
                logger.info("Comparing with original debug image...")
                logger.info(f"Original shape: {original_debug.shape}")
                logger.info(f"Fixed shape: {img_with_boxes.shape}")
                
                # Save comparison
                comparison = np.hstack([original_debug, img_with_boxes])
                cv2.imwrite('bounding_box_comparison.jpg', comparison)
                logger.info("Saved bounding_box_comparison.jpg")
        
        return True
        
    except Exception as e:
        logger.error(f"Fix bounding box position failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    fix_bounding_box_position() 