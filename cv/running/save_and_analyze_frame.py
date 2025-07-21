#!/usr/bin/env python3
"""
Save camera frame and analyze for debugging localization issues
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

def save_and_analyze_frame():
    """Save a frame from the camera and analyze it"""
    logger.info("=== SAVE AND ANALYZE FRAME ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Initialize camera
        if configs['runTime'].get('use_imx500', False):
            logger.info("Using IMX500 camera...")
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (640, 640)},
                controls={"FrameRate": configs['runTime'].get('camRateHz', 30)},
                buffer_count=12
            )
            picam2.start(config, show_preview=False)
            
            # Capture frame
            frame = picam2.capture_array()
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            picam2.stop()
        else:
            logger.info("Using USB camera...")
            cam_id = configs['runTime'].get('camId', 0)
            cap = cv2.VideoCapture(cam_id)
            
            if not cap.isOpened():
                logger.error(f"Could not open camera {cam_id}")
                return False
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error("Could not capture frame")
                return False
        
        # Save frame with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Saved frame: {filename}")
        
        # Analyze frame
        logger.info(f"Frame shape: {frame.shape}")
        logger.info(f"Frame dtype: {frame.dtype}")
        logger.info(f"Frame value range: {frame.min()} to {frame.max()}")
        
        # Check for common issues
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        logger.info(f"Brightness: mean={gray.mean():.1f}, std={gray.std():.1f}")
        
        # Detect edges to see object boundaries
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        logger.info(f"Edge density: {edge_density:.3f}")
        
        # Save edge image for analysis
        cv2.imwrite(f"debug_edges_{timestamp}.jpg", edges)
        logger.info(f"Saved edge image: debug_edges_{timestamp}.jpg")
        
        # Run detection on the saved frame
        logger.info("\n--- Running detection on saved frame ---")
        model_path = configs['training']['weightsFile_rpi']
        from ultralytics import YOLO
        yolo_model = YOLO(model_path)
        
        results = yolo_model(frame, conf=0.1, iou=0.6, verbose=False)
        
        if len(results) > 0:
            result = results[0]
            logger.info(f"Detections: {len(result.boxes)}")
            
            for i, box in enumerate(result.boxes):
                confidence = float(box.conf)
                class_id = int(box.cls)
                bbox = box.xyxy[0].cpu().numpy()
                
                logger.info(f"  Detection {i}: class={class_id}, conf={confidence:.3f}")
                logger.info(f"    Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                
                # Calculate center
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                logger.info(f"    Center: ({center_x:.1f}, {center_y:.1f})")
                
                # Draw bounding box on frame for visualization
                cv2.rectangle(frame, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            (0, 255, 0), 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Class {class_id}: {confidence:.3f}", 
                           (int(bbox[0]), int(bbox[1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            logger.info("No detections found")
        
        # Save annotated frame
        annotated_filename = f"debug_annotated_{timestamp}.jpg"
        cv2.imwrite(annotated_filename, frame)
        logger.info(f"Saved annotated frame: {annotated_filename}")
        
        logger.info(f"\n📁 Files saved:")
        logger.info(f"  - {filename} (original frame)")
        logger.info(f"  - debug_edges_{timestamp}.jpg (edge detection)")
        logger.info(f"  - {annotated_filename} (with bounding boxes)")
        
        return True
        
    except Exception as e:
        logger.error(f"Save and analyze failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    save_and_analyze_frame() 