#!/usr/bin/env python3
"""
Simple camera access test
"""

import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_camera_access():
    """Test different camera access methods"""
    logger.info("=== CAMERA ACCESS TEST ===")
    
    # Test USB camera
    logger.info("Testing USB camera...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            logger.info("USB camera (index 0) is available")
            ret, frame = cap.read()
            if ret and frame is not None:
                logger.info(f"USB camera frame shape: {frame.shape}")
                cv2.imwrite("usb_camera_test.jpg", frame)
                logger.info("Saved USB camera test image: usb_camera_test.jpg")
            else:
                logger.error("USB camera opened but failed to capture frame")
            cap.release()
        else:
            logger.error("USB camera (index 0) is not available")
    except Exception as e:
        logger.error(f"USB camera test failed: {e}")
    
    # Test IMX500 camera
    logger.info("Testing IMX500 camera...")
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (480, 640)}))
        picam2.start()
        logger.info("IMX500 camera is available")
        
        # Capture frame
        frame = picam2.capture_array()
        if frame is not None:
            logger.info(f"IMX500 camera frame shape: {frame.shape}")
            cv2.imwrite("imx500_camera_test.jpg", frame)
            logger.info("Saved IMX500 camera test image: imx500_camera_test.jpg")
        else:
            logger.error("IMX500 camera opened but failed to capture frame")
        
        picam2.close()
    except Exception as e:
        logger.error(f"IMX500 camera test failed: {e}")
    
    # Test other USB camera indices
    for i in range(1, 4):
        logger.info(f"Testing USB camera index {i}...")
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                logger.info(f"USB camera (index {i}) is available")
                ret, frame = cap.read()
                if ret and frame is not None:
                    logger.info(f"USB camera {i} frame shape: {frame.shape}")
                    cv2.imwrite(f"usb_camera_{i}_test.jpg", frame)
                    logger.info(f"Saved USB camera {i} test image: usb_camera_{i}_test.jpg")
                cap.release()
            else:
                logger.info(f"USB camera (index {i}) is not available")
        except Exception as e:
            logger.error(f"USB camera {i} test failed: {e}")
    
    logger.info("=== CAMERA TEST COMPLETE ===")

if __name__ == "__main__":
    test_camera_access() 