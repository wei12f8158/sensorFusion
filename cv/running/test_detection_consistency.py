#!/usr/bin/env python3
"""
Test detection consistency across multiple frames
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

def test_detection_consistency():
    """Test detection consistency across multiple frames"""
    logger.info("=== DETECTION CONSISTENCY TEST ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        model_path = configs['training']['weightsFile_rpi']
        
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
        else:
            logger.info("Using USB camera...")
            cam_id = configs['runTime'].get('camId', 0)
            cap = cv2.VideoCapture(cam_id)
            
            if not cap.isOpened():
                logger.error(f"Could not open camera {cam_id}")
                return False
        
        # Load model
        from ultralytics import YOLO
        yolo_model = YOLO(model_path)
        
        # Test multiple frames
        num_frames = 10
        detections = []
        
        logger.info(f"Testing consistency across {num_frames} frames...")
        
        for i in range(num_frames):
            # Capture frame
            if configs['runTime'].get('use_imx500', False):
                frame = picam2.capture_array()
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"Could not capture frame {i}")
                    continue
            
            # Run detection
            results = yolo_model(frame, conf=0.3, iou=0.6, verbose=False)
            
            frame_detections = []
            if len(results) > 0:
                result = results[0]
                for box in result.boxes:
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    # Calculate center
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    frame_detections.append({
                        'class': class_id,
                        'confidence': confidence,
                        'center': (center_x, center_y),
                        'bbox': bbox
                    })
            
            detections.append(frame_detections)
            logger.info(f"Frame {i+1}: {len(frame_detections)} detections")
            
            # Save first frame for analysis
            if i == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"consistency_test_frame_{timestamp}.jpg", frame)
        
        # Clean up camera
        if configs['runTime'].get('use_imx500', False):
            picam2.stop()
        else:
            cap.release()
        
        # Analyze consistency
        logger.info("\n=== CONSISTENCY ANALYSIS ===")
        
        # Count detections per class
        class_counts = {}
        for frame_detections in detections:
            for det in frame_detections:
                class_id = det['class']
                if class_id not in class_counts:
                    class_counts[class_id] = []
                class_counts[class_id].append(det['confidence'])
        
        for class_id, confidences in class_counts.items():
            logger.info(f"Class {class_id}:")
            logger.info(f"  Detected in {len(confidences)}/{num_frames} frames")
            logger.info(f"  Average confidence: {np.mean(confidences):.3f}")
            logger.info(f"  Confidence std: {np.std(confidences):.3f}")
            logger.info(f"  Min confidence: {np.min(confidences):.3f}")
            logger.info(f"  Max confidence: {np.max(confidences):.3f}")
        
        # Check for localization consistency
        logger.info("\n=== LOCALIZATION CONSISTENCY ===")
        for class_id in class_counts:
            centers = []
            for frame_detections in detections:
                for det in frame_detections:
                    if det['class'] == class_id:
                        centers.append(det['center'])
                        break
            
            if len(centers) > 1:
                centers = np.array(centers)
                center_std = np.std(centers, axis=0)
                logger.info(f"Class {class_id} center variation:")
                logger.info(f"  X std: {center_std[0]:.1f} pixels")
                logger.info(f"  Y std: {center_std[1]:.1f} pixels")
                
                if center_std[0] > 50 or center_std[1] > 50:
                    logger.warning(f"  ⚠️  High localization variation for class {class_id}")
                else:
                    logger.info(f"  ✅ Good localization consistency for class {class_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_detection_consistency() 