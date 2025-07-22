#!/usr/bin/env python3
"""
Diagnose differences between static image and real-time camera processing
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

def capture_and_analyze_frame():
    """Capture a frame from camera and analyze it"""
    logger.info("=== DIAGNOSE REAL-TIME VS STATIC ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        model_path = configs['training']['weightsFile_rpi']
        image_size = configs['training']['imageSize']
        
        logger.info(f"Model path: {model_path}")
        logger.info(f"Expected image size: {image_size}")
        
        # Initialize camera
        img_src = configs['runTime']['imgSrc']
        use_imx500 = configs['runTime']['use_imx500']
        logger.info(f"Image source: {img_src}")
        logger.info(f"Use IMX500: {use_imx500}")
        
        if img_src == "camera" and not use_imx500:
            cap = cv2.VideoCapture(0)  # USB camera
        else:
            # IMX500 camera
            from picamera2 import Picamera2
            picam2 = Picamera2()
            picam2.configure(picam2.create_preview_configuration(main={"size": (image_size[0], image_size[1])}))
            picam2.start()
            cap = picam2
        
        # Capture frame
        logger.info("Capturing frame from camera...")
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            return False
        
        logger.info(f"Captured frame shape: {frame.shape}")
        
        # Save the captured frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_filename = f"realtime_frame_{timestamp}.jpg"
        cv2.imwrite(frame_filename, frame)
        logger.info(f"Saved captured frame: {frame_filename}")
        
        # Load the static test image for comparison
        static_img = cv2.imread('debug_frame_20250721_153537.jpg')
        if static_img is not None:
            logger.info(f"Static image shape: {static_img.shape}")
            
            # Save comparison
            comparison = np.hstack([static_img, frame])
            cv2.imwrite(f"static_vs_realtime_{timestamp}.jpg", comparison)
            logger.info(f"Saved comparison: static_vs_realtime_{timestamp}.jpg")
        
        # Run detection on real-time frame
        from ultralytics import YOLO
        yolo_model = YOLO(model_path)
        
        logger.info("Running detection on real-time frame...")
        results = yolo_model(frame, conf=0.3, iou=0.6, verbose=False)
        
        if len(results) > 0:
            result = results[0]
            logger.info(f"Real-time detections: {len(result.boxes)}")
            
            # Create annotated image
            annotated_frame = frame.copy()
            
            for i, box in enumerate(result.boxes):
                confidence = float(box.conf)
                class_id = int(box.cls)
                bbox = box.xyxy[0].cpu().numpy()
                
                logger.info(f"Real-time Detection {i}: class={class_id}, conf={confidence:.3f}")
                logger.info(f"  Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                
                # Calculate center
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                logger.info(f"  Center: ({center_x:.1f}, {center_y:.1f})")
                
                # Draw bounding box
                color = (0, 255, 0) if class_id == 4 else (255, 255, 0)  # Green for hand, Cyan for plate
                cv2.rectangle(annotated_frame, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            color, 2)
                cv2.circle(annotated_frame, (int(center_x), int(center_y)), 5, color, -1)
                
                # Add label
                label = f"Class {class_id}: {confidence:.3f}"
                cv2.putText(annotated_frame, label, 
                           (int(bbox[0]), int(bbox[1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save annotated real-time frame
            cv2.imwrite(f"realtime_annotated_{timestamp}.jpg", annotated_frame)
            logger.info(f"Saved real-time annotated: realtime_annotated_{timestamp}.jpg")
        
        # Run detection on static image for comparison
        if static_img is not None:
            logger.info("Running detection on static image for comparison...")
            static_results = yolo_model(static_img, conf=0.3, iou=0.6, verbose=False)
            
            if len(static_results) > 0:
                static_result = static_results[0]
                logger.info(f"Static detections: {len(static_result.boxes)}")
                
                # Create annotated static image
                annotated_static = static_img.copy()
                
                for i, box in enumerate(static_result.boxes):
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    logger.info(f"Static Detection {i}: class={class_id}, conf={confidence:.3f}")
                    logger.info(f"  Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                    
                    # Calculate center
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    logger.info(f"  Center: ({center_x:.1f}, {center_y:.1f})")
                    
                    # Draw bounding box
                    color = (0, 255, 0) if class_id == 4 else (255, 255, 0)
                    cv2.rectangle(annotated_static, 
                                (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), 
                                color, 2)
                    cv2.circle(annotated_static, (int(center_x), int(center_y)), 5, color, -1)
                    
                    # Add label
                    label = f"Class {class_id}: {confidence:.3f}"
                    cv2.putText(annotated_static, label, 
                               (int(bbox[0]), int(bbox[1])-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Save annotated static image
                cv2.imwrite(f"static_annotated_{timestamp}.jpg", annotated_static)
                logger.info(f"Saved static annotated: static_annotated_{timestamp}.jpg")
                
                # Create side-by-side comparison
                if len(results) > 0:
                    comparison_annotated = np.hstack([annotated_static, annotated_frame])
                    cv2.imwrite(f"annotated_comparison_{timestamp}.jpg", comparison_annotated)
                    logger.info(f"Saved annotated comparison: annotated_comparison_{timestamp}.jpg")
        
        # Clean up
        if img_src == "camera" and not use_imx500:
            cap.release()
        else:
            picam2.close()
        
        logger.info("=== ANALYSIS COMPLETE ===")
        logger.info("Check the generated images to compare:")
        logger.info("1. static_vs_realtime_*.jpg - Raw image comparison")
        logger.info("2. static_annotated_*.jpg - Static image with boxes")
        logger.info("3. realtime_annotated_*.jpg - Real-time image with boxes")
        logger.info("4. annotated_comparison_*.jpg - Side-by-side comparison")
        
        return True
        
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    capture_and_analyze_frame() 