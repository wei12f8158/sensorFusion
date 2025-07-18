#!/usr/bin/env python3
"""
Simple test for Camera Source Mode with correct RGB/BGR conversion
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Simple test for Camera Source Mode"""
    logger.info("=== SIMPLE CAMERA SOURCE MODE TEST ===")
    
    # Check configuration
    config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
    configs = config.get_config()
    
    use_imx500 = configs['runTime'].get('use_imx500', False)
    imx500_ai_camera = configs['runTime'].get('imx500_ai_camera', False)
    
    logger.info(f"use_imx500: {use_imx500}")
    logger.info(f"imx500_ai_camera: {imx500_ai_camera}")
    
    if not (use_imx500 and not imx500_ai_camera):
        logger.error("Please switch to Camera Source Mode first!")
        return
    
    try:
        # Initialize camera
        from picamera2 import Picamera2
        
        logger.info("Initializing IMX500 camera at 640x640...")
        picam2 = Picamera2()
        
        config = picam2.create_preview_configuration(
            main={"size": (640, 640)},
            controls={"FrameRate": 30},
            buffer_count=12
        )
        
        picam2.start(config, show_preview=False)
        logger.info("✓ Camera started successfully")
        
        # Load model
        model_path = configs['training']['weightsFile_rpi']
        dataset_path = configs['training']['dataSetDir'] + '/' + configs['training']['dataSet']
        
        logger.info(f"Loading model: {model_path}")
        from rpiModel import RaspberryPiModel
        
        thresh = min(configs['runTime']['distSettings']['handThreshold'],
                    configs['runTime']['distSettings']['objectThreshold'])
        
        model = RaspberryPiModel(
            model_file=model_path,
            names_file=dataset_path,
            conf_thresh=thresh,
            iou_thresh=configs['runTime']['distSettings']['nmsIouThreshold'],
            v8=True,
            use_gpu=False,
            num_threads=4
        )
        
        logger.info("✓ Model loaded successfully")
        
        # Test a few frames
        for frame_num in range(5):
            logger.info(f"\n--- Frame {frame_num + 1} ---")
            
            # Capture image (RGB format from camera)
            image_rgb = picam2.capture_array()
            logger.info(f"Captured RGB image: {image_rgb.shape}")
            
            # Convert to BGR for OpenCV processing
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            logger.info(f"Converted to BGR: {image_bgr.shape}")
            
            # Save both versions for comparison
            cv2.imwrite(f"frame_{frame_num+1}_rgb.jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"frame_{frame_num+1}_bgr.jpg", image_bgr)
            
            # Run inference with BGR image
            from utils import get_image_tensor
            full_image, net_image, pad = get_image_tensor(image_bgr, model.input_size[0])
            
            logger.info(f"Preprocessed for model: {net_image.shape}")
            
            # Model inference
            pred = model.forward(net_image, with_nms=True)
            
            logger.info(f"Detections: {len(pred)}")
            if len(pred) > 0:
                logger.info("✓ Found detections!")
                for i, det in enumerate(pred):
                    logger.info(f"  Detection {i}: class={int(det[5])}, conf={det[4]:.3f}")
            else:
                logger.info("  No detections")
                
                # Try with lower threshold
                logger.info("Trying with lower threshold...")
                model.conf_thresh = 0.001  # Very low threshold
                pred_low = model.forward(net_image, with_nms=True)
                logger.info(f"With low threshold: {len(pred_low)} detections")
                if len(pred_low) > 0:
                    logger.info("✓ Found detections with low threshold!")
                    for i, det in enumerate(pred_low):
                        logger.info(f"  Detection {i}: class={int(det[5])}, conf={det[4]:.3f}")
        
        picam2.stop()
        logger.info("✓ Test completed successfully!")
        
        logger.info("\n=== SUMMARY ===")
        logger.info("Camera Source Mode should work correctly now.")
        logger.info("The key fix was using BGR format for model inference.")
        logger.info("Check the saved images to verify camera capture is working.")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 