#!/usr/bin/env python3
"""
Debug the main application's coordinate processing
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

def debug_main_application():
    """Debug the main application's coordinate processing"""
    logger.info("=== DEBUG MAIN APPLICATION ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Import main application modules
        from modelRunTime import modelRunTime
        from distance import distanceCalculator
        from display import displayHandObject
        
        logger.info("Initializing main application components...")
        
        # Initialize components like the main app does
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
        cv2.imwrite(f"main_app_raw_frame_{timestamp}.jpg", frame)
        logger.info(f"Saved raw frame: main_app_raw_frame_{timestamp}.jpg")
        
        # Run detection through main application pipeline
        logger.info("Running detection through main application pipeline...")
        
        # Get model predictions
        predictions = model_runtime.runInference(frame)
        logger.info(f"Model predictions: {type(predictions)}")
        
        if predictions is not None and len(predictions) > 0:
            logger.info(f"Number of detections: {len(predictions)}")
            
            # Process through distance calculator
            distance_calc.zeroData()
            valid = distance_calc.loadData(predictions)
            logger.info(f"Distance calculation valid: {valid}")
            
            if valid:
                logger.info("=== DETECTION DETAILS ===")
                logger.info(f"Hand object: {distance_calc.handObject}")
                logger.info(f"Grab object: {distance_calc.grabObject}")
                logger.info(f"Hand center: {distance_calc.handCenter}")
                logger.info(f"Best center: {distance_calc.bestCenter}")
                logger.info(f"Best distance: {distance_calc.bestDist}")
                
                # Get bounding boxes
                if distance_calc.nHands > 0:
                    hand_ul, hand_lr = distance_calc.getBox(distance_calc.handObject)
                    logger.info(f"Hand bounding box: UL={hand_ul}, LR={hand_lr}")
                
                if distance_calc.nNonHand > 0:
                    obj_ul, obj_lr = distance_calc.getBox(distance_calc.grabObject)
                    logger.info(f"Object bounding box: UL={obj_ul}, LR={obj_lr}")
                
                # Create annotated image using main app display
                annotated_frame = frame.copy()
                display_obj.draw(annotated_frame, distance_calc, valid, saveFileName="main_app_debug")
                
                # Save annotated frame
                cv2.imwrite(f"main_app_annotated_{timestamp}.jpg", annotated_frame)
                logger.info(f"Saved annotated frame: main_app_annotated_{timestamp}.jpg")
                
                # Also run direct YOLO for comparison
                logger.info("Running direct YOLO for comparison...")
                from ultralytics import YOLO
                yolo_model = YOLO(configs['training']['weightsFile_rpi'])
                results = yolo_model(frame, conf=0.3, iou=0.6, verbose=False)
                
                if len(results) > 0:
                    result = results[0]
                    direct_annotated = frame.copy()
                    
                    for i, box in enumerate(result.boxes):
                        confidence = float(box.conf)
                        class_id = int(box.cls)
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        logger.info(f"Direct YOLO Detection {i}: class={class_id}, conf={confidence:.3f}")
                        logger.info(f"  Direct bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                        
                        color = (0, 255, 0) if class_id == 4 else (255, 255, 0)
                        cv2.rectangle(direct_annotated, 
                                    (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])), 
                                    color, 2)
                        
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        cv2.circle(direct_annotated, (int(center_x), int(center_y)), 5, color, -1)
                        
                        label = f"Direct Class {class_id}: {confidence:.3f}"
                        cv2.putText(direct_annotated, label, 
                                   (int(bbox[0]), int(bbox[1])-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    cv2.imwrite(f"direct_yolo_annotated_{timestamp}.jpg", direct_annotated)
                    logger.info(f"Saved direct YOLO annotated: direct_yolo_annotated_{timestamp}.jpg")
                    
                    # Create comparison
                    comparison = np.hstack([annotated_frame, direct_annotated])
                    cv2.imwrite(f"main_app_vs_direct_{timestamp}.jpg", comparison)
                    logger.info(f"Saved comparison: main_app_vs_direct_{timestamp}.jpg")
        
        # Clean up
        if img_src == "camera" and not use_imx500:
            cap.release()
        else:
            picam2.close()
        
        logger.info("=== DEBUG COMPLETE ===")
        return True
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_main_application() 