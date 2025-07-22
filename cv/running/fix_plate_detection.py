#!/usr/bin/env python3
"""
Fix plate detection by making it work like hand detection
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

def fix_plate_detection():
    """Fix plate detection by making it work like hand detection"""
    logger.info("=== FIX PLATE DETECTION ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Import main application modules
        from modelRunTime import modelRunTime
        from distance import distanceCalculator
        from display import displayHandObject
        
        logger.info("Initializing components...")
        
        # Initialize components
        model_runtime = modelRunTime(configs)
        distance_calc = distanceCalculator(configs['training']['imageSize'], configs)
        display_obj = displayHandObject(configs)
        
        # Capture a frame
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
        cv2.imwrite(f"fix_plate_raw_frame_{timestamp}.jpg", frame)
        logger.info(f"Saved raw frame: fix_plate_raw_frame_{timestamp}.jpg")
        
        # Run detection
        predictions = model_runtime.runInference(frame)
        logger.info(f"Model predictions: {type(predictions)}")
        
        if predictions is not None and len(predictions) > 0:
            logger.info(f"Number of detections: {len(predictions)}")
            
            # Process through distance calculator
            distance_calc.zeroData()
            valid = distance_calc.loadData(predictions)
            logger.info(f"Distance calculation valid: {valid}")
            
            if valid:
                logger.info("=== ORIGINAL DETECTION DETAILS ===")
                logger.info(f"Hand object: {distance_calc.handObject}")
                logger.info(f"Grab object: {distance_calc.grabObject}")
                logger.info(f"Hand center: {distance_calc.handCenter}")
                logger.info(f"Best center: {distance_calc.bestCenter}")
                
                # Create original annotated image
                original_annotated = frame.copy()
                display_obj.draw(original_annotated, distance_calc, valid, saveFileName="original")
                cv2.imwrite(f"original_annotated_{timestamp}.jpg", original_annotated)
                logger.info(f"Saved original annotated: original_annotated_{timestamp}.jpg")
                
                # Now fix the plate detection by making it work like hand detection
                logger.info("=== FIXING PLATE DETECTION ===")
                
                # Find the plate object (class 6) directly from predictions
                plate_object = None
                plate_center = None
                
                for obj in predictions:
                    class_id = int(obj[5])
                    confidence = float(obj[4])
                    
                    if class_id == 6 and confidence >= 0.3:  # Plate class with good confidence
                        plate_object = obj
                        # Calculate center like hand detection
                        x1, y1, x2, y2 = distance_calc.getXY(obj)
                        plate_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        logger.info(f"Found plate: class={class_id}, conf={confidence:.3f}")
                        logger.info(f"Plate center: {plate_center}")
                        break
                
                if plate_object is not None:
                    # Create fixed annotated image
                    fixed_annotated = frame.copy()
                    
                    # Draw hand (same as before)
                    if distance_calc.nHands > 0:
                        hand_ul, hand_lr = distance_calc.getBox(distance_calc.handObject)
                        cv2.rectangle(fixed_annotated, hand_ul, hand_lr, (0, 255, 0), 2)
                        cv2.circle(fixed_annotated, distance_calc.handCenter, 5, (0, 255, 0), -1)
                        hand_text = f"Glove: {distance_calc.handObject[4]:.2f}%"
                        cv2.putText(fixed_annotated, hand_text, hand_ul, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
                    
                    # Draw plate with fixed center
                    plate_ul, plate_lr = distance_calc.getBox(plate_object)
                    cv2.rectangle(fixed_annotated, plate_ul, plate_lr, (255, 255, 0), 2)  # Cyan
                    cv2.circle(fixed_annotated, plate_center, 5, (255, 255, 0), -1)
                    plate_text = f"Plate: {plate_object[4]:.2f}%"
                    cv2.putText(fixed_annotated, plate_text, plate_ul, cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))
                    
                    # Draw distance line using fixed centers
                    if distance_calc.nHands > 0:
                        cv2.line(fixed_annotated, distance_calc.handCenter, plate_center, (255, 255, 255), 2)
                        dist_text = f"Distance: {distance_calc.bestDist:.0f}mm"
                        cv2.putText(fixed_annotated, dist_text, (0, 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    
                    cv2.imwrite(f"fixed_annotated_{timestamp}.jpg", fixed_annotated)
                    logger.info(f"Saved fixed annotated: fixed_annotated_{timestamp}.jpg")
                    
                    # Create comparison
                    comparison = np.hstack([original_annotated, fixed_annotated])
                    cv2.imwrite(f"original_vs_fixed_{timestamp}.jpg", comparison)
                    logger.info(f"Saved comparison: original_vs_fixed_{timestamp}.jpg")
                    
                    logger.info("=== FIX APPLIED ===")
                    logger.info("Plate detection now uses direct object center instead of distance-based center")
                else:
                    logger.warning("No plate detected in the frame")
        
        # Clean up
        if img_src == "camera" and not use_imx500:
            cap.release()
        else:
            picam2.close()
        
        logger.info("=== FIX COMPLETE ===")
        return True
        
    except Exception as e:
        logger.error(f"Fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    fix_plate_detection() 