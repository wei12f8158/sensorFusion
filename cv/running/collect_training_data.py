#!/usr/bin/env python3
"""
Collect training data for fine-tuning the model
"""

import os
import sys
import cv2
import numpy as np
import logging
from datetime import datetime
import json

# Add the parent directory to the path
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_training_data():
    """Collect training data for fine-tuning"""
    logger.info("=== COLLECT TRAINING DATA ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Create data collection directory
        data_dir = "training_data_collection"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Create subdirectories for each class
        classes = {
            0: "apple",
            1: "ball", 
            2: "bottle",
            3: "clip",
            4: "glove",
            5: "lid",
            6: "plate",
            7: "spoon",
            8: "tape_spool"
        }
        
        for class_id, class_name in classes.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
        
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
        
        logger.info("=== DATA COLLECTION MODE ===")
        logger.info("Press 's' to save current frame with detections")
        logger.info("Press 'q' to quit")
        logger.info("Press 'c' to capture frame without detections")
        
        frame_count = 0
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error("Failed to capture frame")
                break
            
            # Create display frame
            display_frame = frame.copy()
            
            # Run detection
            predictions = model_runtime.runInference(frame)
            
            if predictions is not None and len(predictions) > 0:
                # Process through distance calculator
                distance_calc.zeroData()
                valid = distance_calc.loadData(predictions)
                
                if valid:
                    # Draw detections
                    display_obj.draw(display_frame, distance_calc, valid)
                    
                    # Show detection info
                    cv2.putText(display_frame, f"Detections: {len(predictions)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    for i, pred in enumerate(predictions):
                        class_id = int(pred[5])
                        confidence = float(pred[4])
                        class_name = classes.get(class_id, f"class_{class_id}")
                        cv2.putText(display_frame, f"{class_name}: {confidence:.3f}", (10, 60 + i*30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Data Collection", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Quitting data collection")
                break
            elif key == ord('s'):
                # Save frame with detections
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                frame_filename = f"frame_{timestamp}.jpg"
                frame_path = os.path.join(data_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                # Save detection annotations
                if predictions is not None and len(predictions) > 0:
                    annotations = []
                    for pred in predictions:
                        class_id = int(pred[5])
                        confidence = float(pred[4])
                        bbox = pred[:4]
                        
                        annotation = {
                            "class_id": class_id,
                            "class_name": classes.get(class_id, f"class_{class_id}"),
                            "confidence": float(confidence),
                            "bbox": [float(x) for x in bbox],
                            "center": [float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)]
                        }
                        annotations.append(annotation)
                    
                    annotation_filename = f"frame_{timestamp}.json"
                    annotation_path = os.path.join(data_dir, annotation_filename)
                    
                    with open(annotation_path, 'w') as f:
                        json.dump(annotations, f, indent=2)
                    
                    logger.info(f"Saved frame and annotations: {frame_filename}")
                else:
                    logger.info(f"Saved frame (no detections): {frame_filename}")
                
                frame_count += 1
            elif key == ord('c'):
                # Capture frame without detections
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                frame_filename = f"frame_{timestamp}.jpg"
                frame_path = os.path.join(data_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                logger.info(f"Captured frame: {frame_filename}")
                frame_count += 1
        
        # Clean up
        cv2.destroyAllWindows()
        if img_src == "camera" and not use_imx500:
            cap.release()
        else:
            picam2.close()
        
        logger.info(f"=== DATA COLLECTION COMPLETE ===")
        logger.info(f"Collected {frame_count} frames")
        logger.info(f"Data saved in: {data_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    collect_training_data() 