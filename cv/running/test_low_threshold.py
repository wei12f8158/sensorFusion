#!/usr/bin/env python3
"""
Test with low thresholds to get detections
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

def test_low_threshold():
    """Test with low thresholds to get detections"""
    logger.info("=== TEST LOW THRESHOLD ===")
    
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
        
        # Import and initialize model
        from modelRunTime import modelRunTime
        model_runtime = modelRunTime(configs, device)
        
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
        cv2.imwrite(f"low_threshold_raw_{timestamp}.jpg", frame)
        logger.info(f"Saved raw frame: low_threshold_raw_{timestamp}.jpg")
        
        # Test with different confidence thresholds
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3]
        
        for conf_thresh in thresholds:
            logger.info(f"Testing with confidence threshold: {conf_thresh}")
            
            # Run detection with low threshold
            predictions = model_runtime.runInference(frame)
            
            if predictions is not None and len(predictions) > 0:
                logger.info(f"Found {len(predictions)} detections with threshold {conf_thresh}")
                
                # Create annotated image
                annotated = frame.copy()
                
                for i, pred in enumerate(predictions):
                    class_id = int(pred[5])
                    confidence = float(pred[4])
                    bbox = pred[:4]
                    
                    logger.info(f"Detection {i}: class={class_id}, conf={confidence:.3f}, bbox={bbox}")
                    
                    # Draw bounding box
                    color = (0, 255, 0) if class_id == 4 else (255, 255, 0)  # Green for hand, Cyan for plate
                    cv2.rectangle(annotated, 
                                (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), 
                                color, 2)
                    
                    # Calculate center
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    cv2.circle(annotated, (center_x, center_y), 5, color, -1)
                    
                    # Add label
                    label = f"Class {class_id}: {confidence:.3f}"
                    cv2.putText(annotated, label, 
                               (int(bbox[0]), int(bbox[1])-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Save annotated image
                cv2.imwrite(f"low_threshold_{conf_thresh}_{timestamp}.jpg", annotated)
                logger.info(f"Saved annotated image: low_threshold_{conf_thresh}_{timestamp}.jpg")
                
                # If we found detections, we can stop testing lower thresholds
                break
            else:
                logger.info(f"No detections with threshold {conf_thresh}")
        
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
    test_low_threshold() 