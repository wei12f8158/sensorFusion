#!/usr/bin/env python3
"""
Test script for IMX500 Camera Source Mode (640x640 resolution)
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

def test_camera_source_mode():
    """Test IMX500 Camera Source Mode with 640x640 resolution"""
    logger.info("=== IMX500 CAMERA SOURCE MODE TEST (640x640) ===")
    
    try:
        from picamera2 import Picamera2
        
        # Load config
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Initialize IMX500 as camera source with 640x640 resolution
        logger.info("Initializing IMX500 as camera source at 640x640...")
        picam2 = Picamera2()
        
        config = picam2.create_preview_configuration(
            main={"size": (640, 640)},  # Match YOLO model input size
            controls={"FrameRate": configs['runTime'].get('camRateHz', 30)},
            buffer_count=12
        )
        
        picam2.start(config, show_preview=False)
        logger.info("✓ IMX500 camera source initialized at 640x640")
        
        # Test camera capture
        logger.info("Testing camera capture for 5 seconds...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5:
            try:
                # Capture image
                image = picam2.capture_array()
                
                # Convert from RGB to BGR for OpenCV
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                frame_count += 1
                logger.info(f"Frame {frame_count}: Captured image shape: {image.shape}")
                
                # Display image
                cv2.imshow("IMX500 Camera Source Test (640x640)", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                logger.error(f"Error capturing frame: {e}")
                break
        
        # Cleanup
        picam2.stop()
        cv2.destroyAllWindows()
        
        logger.info(f"✓ Camera source test completed. Captured {frame_count} frames at 640x640.")
        return True
        
    except Exception as e:
        logger.error(f"✗ IMX500 camera source test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_with_640x640():
    """Test YOLO model with 640x640 images"""
    logger.info("=== YOLO MODEL TEST WITH 640x640 IMAGES ===")
    
    try:
        # Load config
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Check if we're in camera source mode
        if not configs['runTime'].get('use_imx500', False) or configs['runTime'].get('imx500_ai_camera', False):
            logger.warning("Not in camera source mode. Please switch to camera source mode first.")
            return False
        
        from rpiModel import RaspberryPiModel
        
        # Load model
        model_path = configs['training']['weightsFile_rpi']
        dataset_path = configs['training']['dataSetDir'] + '/' + configs['training']['dataSet']
        
        logger.info(f"Loading model: {model_path}")
        logger.info(f"Dataset: {dataset_path}")
        
        # Get thresholds
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
        logger.info(f"Model input size: {model.input_size}")
        
        # Create a test image at 640x640
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add some colored rectangles to simulate objects
        cv2.rectangle(test_img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green rectangle
        cv2.rectangle(test_img, (400, 300), (500, 400), (255, 0, 0), -1)  # Blue rectangle
        
        # Save test image
        cv2.imwrite("test_640x640.jpg", test_img)
        logger.info("Saved test image as test_640x640.jpg")
        
        # Run inference
        from utils import get_image_tensor
        full_image, net_image, pad = get_image_tensor(test_img, model.input_size[0])
        
        logger.info(f"Preprocessed image shape: {net_image.shape}")
        logger.info(f"Padding: {pad}")
        
        # Run model
        pred = model.forward(net_image, with_nms=True)
        
        logger.info(f"Raw predictions shape: {pred.shape if hasattr(pred, 'shape') else 'No shape'}")
        
        if len(pred) > 0:
            logger.info(f"✓ Found {len(pred)} detections!")
            for i, det in enumerate(pred):
                logger.info(f"  Detection {i}: {det}")
        else:
            logger.warning("✗ No detections found - this might indicate an issue")
            
            # Try without NMS
            logger.info("Trying without NMS...")
            pred_raw = model.forward(net_image, with_nms=False)
            logger.info(f"Raw predictions without NMS: {pred_raw.shape if hasattr(pred_raw, 'shape') else 'No shape'}")
            if len(pred_raw) > 0:
                logger.info(f"Found {len(pred_raw)} raw predictions before NMS")
                for i, det in enumerate(pred_raw[:5]):  # Show first 5
                    logger.info(f"  Raw detection {i}: {det}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    logger.info("Starting IMX500 Camera Source Mode test...")
    
    # Test 1: Camera capture at 640x640
    logger.info("\n" + "="*50)
    success1 = test_camera_source_mode()
    
    # Test 2: Model inference with 640x640
    logger.info("\n" + "="*50)
    success2 = test_model_with_640x640()
    
    if success1 and success2:
        logger.info("\n✓ All tests passed! Camera Source Mode should work with 640x640 resolution.")
        logger.info("You can now run the main application in Camera Source Mode.")
    else:
        logger.error("\n✗ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main() 