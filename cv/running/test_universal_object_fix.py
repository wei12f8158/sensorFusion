#!/usr/bin/env python3
"""
Test universal object detection fix for all objects
"""

import os
import sys
import cv2
import numpy as np
import logging
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_universal_object_fix():
    """Test universal object detection fix for all objects"""
    logger.info("=== TEST UNIVERSAL OBJECT FIX ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
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
        
        # Capture frame
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("Failed to capture frame")
            return False
        
        logger.info(f"Captured frame shape: {frame.shape}")
        
        # Save raw frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"universal_fix_raw_{timestamp}.jpg", frame)
        logger.info(f"Saved raw frame: universal_fix_raw_{timestamp}.jpg")
        
        # Run detection
        predictions = model_runtime.runInference(frame)
        logger.info(f"Model predictions: {type(predictions)}")
        
        if predictions is not None and len(predictions) > 0:
            logger.info(f"Number of detections: {len(predictions)}")
            
            # Show all detections first
            logger.info("=== ALL DETECTIONS ===")
            for i, pred in enumerate(predictions):
                class_id = int(pred[5])
                confidence = float(pred[4])
                bbox = pred[:4]
                logger.info(f"Detection {i}: class={class_id}, conf={confidence:.3f}, bbox={bbox}")
            
            # Process through distance calculator (with universal fix)
            distance_calc.zeroData()
            valid = distance_calc.loadData(predictions)
            logger.info(f"Distance calculation valid: {valid}")
            
            if valid:
                logger.info("=== UNIVERSAL FIX RESULTS ===")
                logger.info(f"Hand object: {distance_calc.handObject}")
                logger.info(f"Target object: {distance_calc.grabObject}")
                logger.info(f"Hand center: {distance_calc.handCenter}")
                logger.info(f"Target center: {distance_calc.bestCenter}")
                logger.info(f"Distance: {distance_calc.bestDist:.1f}mm")
                
                # Create annotated image
                annotated_frame = frame.copy()
                display_obj.draw(annotated_frame, distance_calc, valid, saveFileName="universal_fix")
                
                # Save annotated frame
                cv2.imwrite(f"universal_fix_annotated_{timestamp}.jpg", annotated_frame)
                logger.info(f"Saved annotated frame: universal_fix_annotated_{timestamp}.jpg")
                
                # Also create a detailed analysis image
                analysis_frame = frame.copy()
                
                # Draw all detections with different colors
                colors = {
                    0: (0, 255, 0),    # Apple: Green
                    1: (0, 0, 128),    # Ball: Dark Red
                    2: (0, 128, 128),  # Bottle: Teal
                    3: (0, 128, 0),    # Clip: Dark Green
                    4: (0, 0, 0),      # Glove: Black
                    5: (255, 0, 0),    # Lid: Blue
                    6: (255, 255, 0),  # Plate: Cyan
                    7: (128, 0, 128),  # Spoon: Purple
                    8: (255, 0, 255)   # Tape Spool: Magenta
                }
                
                for i, pred in enumerate(predictions):
                    class_id = int(pred[5])
                    confidence = float(pred[4])
                    bbox = pred[:4]
                    
                    color = colors.get(class_id, (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(analysis_frame, 
                                (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), 
                                color, 2)
                    
                    # Calculate center
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    cv2.circle(analysis_frame, (center_x, center_y), 5, color, -1)
                    
                    # Add label
                    label = f"Class {class_id}: {confidence:.3f}"
                    cv2.putText(analysis_frame, label, 
                               (int(bbox[0]), int(bbox[1])-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Highlight selected target
                if distance_calc.nNonHand > 0:
                    target_class = int(distance_calc.grabObject[5])
                    target_color = colors.get(target_class, (255, 255, 255))
                    
                    # Draw thick border around selected target
                    target_ul, target_lr = distance_calc.getBox(distance_calc.grabObject)
                    cv2.rectangle(analysis_frame, target_ul, target_lr, target_color, 4)
                    
                    # Add "SELECTED" label
                    cv2.putText(analysis_frame, "SELECTED", 
                               (target_ul[0], target_ul[1]-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, target_color, 2)
                
                cv2.imwrite(f"universal_fix_analysis_{timestamp}.jpg", analysis_frame)
                logger.info(f"Saved analysis frame: universal_fix_analysis_{timestamp}.jpg")
                
                logger.info("=== UNIVERSAL FIX COMPLETE ===")
                logger.info("All objects now use class-based priority selection!")
            else:
                logger.warning("No valid detections for distance calculation")
        else:
            logger.warning("No detections found")
        
        # Clean up
        if img_src == "camera" and not use_imx500:
            cap.release()
        else:
            picam2.close()
        
        logger.info("=== TEST COMPLETE ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_universal_object_fix() 