#!/usr/bin/env python3
"""
Test script for IMX500 AI camera mode
"""

import os
import sys
import cv2
import numpy as np
import logging
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imx500_ai_camera():
    """Test IMX500 AI camera mode"""
    logger.info("=== IMX500 AI CAMERA MODE TEST ===")
    
    try:
        from picamera2 import Picamera2
        from picamera2.devices import IMX500
        from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)
        
        # Load config
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Get IMX500 settings
        imx500_model = configs['runTime']['imx500_model']
        imx500_labels = configs['runTime'].get('imx500_labels', None)
        imx500_threshold = configs['runTime'].get('imx500_threshold', 0.5)
        imx500_iou = configs['runTime'].get('imx500_iou', 0.5)
        imx500_max_detections = configs['runTime'].get('imx500_max_detections', 10)
        
        logger.info(f"IMX500 model: {imx500_model}")
        logger.info(f"IMX500 labels: {imx500_labels}")
        logger.info(f"Threshold: {imx500_threshold}")
        logger.info(f"IOU: {imx500_iou}")
        logger.info(f"Max detections: {imx500_max_detections}")
        
        # Check if files exist
        if not os.path.exists(imx500_model):
            logger.error(f"IMX500 model file not found: {imx500_model}")
            return False
            
        if imx500_labels and not os.path.exists(imx500_labels):
            logger.error(f"IMX500 labels file not found: {imx500_labels}")
            return False
        
        # Initialize IMX500 as AI camera
        logger.info("Initializing IMX500 as AI camera...")
        imx500 = IMX500(imx500_model)
        intrinsics = imx500.network_intrinsics
        
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
            
        if imx500_labels:
            with open(imx500_labels, 'r') as f:
                intrinsics.labels = f.read().splitlines()
                logger.info(f"Loaded {len(intrinsics.labels)} labels: {intrinsics.labels}")
        
        intrinsics.update_with_defaults()
        
        # Initialize camera
        picam2 = Picamera2(imx500.camera_num)
        config = picam2.create_preview_configuration(
            controls={"FrameRate": intrinsics.inference_rate}, 
            buffer_count=12
        )
        
        logger.info("Starting IMX500 camera...")
        picam2.start(config, show_preview=False)
        logger.info("✓ IMX500 AI camera initialized successfully")
        
        # Test for a few seconds
        logger.info("Testing AI camera mode for 10 seconds...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 10:
            try:
                # Get detections from IMX500
                metadata = picam2.capture_metadata()
                detections = parse_detections(metadata, intrinsics, imx500, picam2, 
                                           imx500_threshold, imx500_iou, imx500_max_detections)
                
                # Capture image for display
                image = picam2.capture_array()
                
                # Convert from RGB to BGR for OpenCV
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                frame_count += 1
                
                if len(detections) > 0:
                    logger.info(f"Frame {frame_count}: Found {len(detections)} detections")
                    for i, det in enumerate(detections):
                        logger.info(f"  Detection {i}: class={det.category}, conf={det.conf:.3f}")
                        
                        # Draw bounding box
                        box = det.box
                        if hasattr(box, '__iter__') and len(box) == 4:
                            x1, y1, w, h = box
                            x2, y2 = x1 + w, y1 + h
                            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(image, f"{det.category}: {det.conf:.2f}", 
                                      (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    logger.info(f"Frame {frame_count}: No detections")
                
                # Display image
                cv2.imshow("IMX500 AI Camera Test", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                break
        
        # Cleanup
        picam2.stop()
        cv2.destroyAllWindows()
        
        logger.info(f"✓ AI camera test completed. Processed {frame_count} frames.")
        return True
        
    except Exception as e:
        logger.error(f"✗ IMX500 AI camera test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def parse_detections(metadata, intrinsics, imx500, picam2, threshold, iou, max_detections):
    """Parse detections from IMX500 metadata"""
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    
    if np_outputs is None:
        return []
        
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0], conf=threshold, iou_thres=iou, max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if intrinsics.bbox_normalization:
            boxes = boxes / input_h
        if intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)
        
    class Detection:
        def __init__(self, box, category, conf):
            self.box = box
            self.category = category
            self.conf = conf
            
    detections = [Detection(box, category, score) for box, score, category in zip(boxes, scores, classes) if score > threshold]
    return detections

def main():
    """Main function"""
    logger.info("Starting IMX500 AI camera mode test...")
    
    success = test_imx500_ai_camera()
    
    if success:
        logger.info("✓ IMX500 AI camera mode test completed successfully!")
        logger.info("You can now run the main application with AI camera mode enabled.")
    else:
        logger.error("✗ IMX500 AI camera mode test failed!")
        logger.error("Check the logs above for error details.")

if __name__ == "__main__":
    main() 