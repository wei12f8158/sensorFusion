#!/usr/bin/env python3
"""
Test camera ID 8 specifically
"""

import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_camera_8():
    """Test camera ID 8 specifically"""
    logger.info("=== TESTING CAMERA ID 8 ===")
    
    try:
        cap = cv2.VideoCapture(8)
        if cap.isOpened():
            logger.info("Camera ID 8 opened successfully")
            
            # Try to capture a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                logger.info(f"Camera ID 8 frame shape: {frame.shape}")
                cv2.imwrite("camera_8_test.jpg", frame)
                logger.info("Saved camera 8 test image: camera_8_test.jpg")
                
                # Show frame info
                logger.info(f"Frame type: {type(frame)}")
                logger.info(f"Frame dtype: {frame.dtype}")
                logger.info(f"Frame min/max values: {frame.min()}/{frame.max()}")
                
                return True
            else:
                logger.error("Camera ID 8 opened but failed to capture frame")
                return False
        else:
            logger.error("Failed to open camera ID 8")
            return False
    except Exception as e:
        logger.error(f"Camera ID 8 test failed: {e}")
        return False
    finally:
        if 'cap' in locals():
            cap.release()

if __name__ == "__main__":
    test_camera_8() 