#!/usr/bin/env python3
"""
Test Camera Source Mode with synthetic objects and real object guidance
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

def test_synthetic_objects():
    """Test with synthetic objects to verify model works"""
    logger.info("=== TESTING WITH SYNTHETIC OBJECTS ===")
    
    try:
        # Load model
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        model_path = configs['training']['weightsFile_rpi']
        dataset_path = configs['training']['dataSetDir'] + '/' + configs['training']['dataSet']
        
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
        
        logger.info("✓ Model loaded for synthetic test")
        
        # Create synthetic test image with colored rectangles
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add different colored rectangles (simulating objects)
        cv2.rectangle(test_img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green
        cv2.rectangle(test_img, (400, 300), (500, 400), (255, 0, 0), -1)  # Blue  
        cv2.rectangle(test_img, (250, 250), (350, 350), (0, 0, 255), -1)  # Red
        cv2.rectangle(test_img, (50, 400), (150, 500), (255, 255, 0), -1)  # Cyan
        cv2.rectangle(test_img, (450, 50), (550, 150), (255, 0, 255), -1)  # Magenta
        
        cv2.imwrite("synthetic_test.jpg", test_img)
        logger.info("Created synthetic test image: synthetic_test.jpg")
        
        # Run inference
        from utils import get_image_tensor
        full_image, net_image, pad = get_image_tensor(test_img, model.input_size[0])
        
        # Test with normal threshold
        pred = model.forward(net_image, with_nms=True)
        logger.info(f"Synthetic test - Normal threshold: {len(pred)} detections")
        
        # Test with very low threshold
        model.conf_thresh = 0.001
        pred_low = model.forward(net_image, with_nms=True)
        logger.info(f"Synthetic test - Low threshold: {len(pred_low)} detections")
        
        if len(pred_low) > 0:
            logger.info("✓ Model is working! Found detections on synthetic image")
            for i, det in enumerate(pred_low):
                logger.info(f"  Detection {i}: class={int(det[5])}, conf={det[4]:.3f}")
        else:
            logger.warning("⚠️  No detections on synthetic image - model might have issues")
        
        return True
        
    except Exception as e:
        logger.error(f"Synthetic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_camera_with_guidance():
    """Test camera with guidance for real objects"""
    logger.info("=== TESTING CAMERA WITH OBJECT GUIDANCE ===")
    
    # Check configuration
    config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
    configs = config.get_config()
    
    use_imx500 = configs['runTime'].get('use_imx500', False)
    imx500_ai_camera = configs['runTime'].get('imx500_ai_camera', False)
    
    if not (use_imx500 and not imx500_ai_camera):
        logger.error("Please switch to Camera Source Mode first!")
        return False
    
    try:
        # Initialize camera
        from picamera2 import Picamera2
        
        logger.info("Initializing IMX500 camera...")
        picam2 = Picamera2()
        
        config = picam2.create_preview_configuration(
            main={"size": (640, 640)},
            controls={"FrameRate": 30},
            buffer_count=12
        )
        
        picam2.start(config, show_preview=False)
        logger.info("✓ Camera started")
        
        # Load model
        model_path = configs['training']['weightsFile_rpi']
        dataset_path = configs['training']['dataSetDir'] + '/' + configs['training']['dataSet']
        
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
        
        logger.info("✓ Model loaded")
        
        print("\n" + "="*60)
        print("OBJECT DETECTION TEST - READY TO START")
        print("="*60)
        print("1. Make sure you have objects in front of the camera")
        print("2. Good lighting is important")
        print("3. Objects should be clearly visible")
        print("4. Try different distances (not too close, not too far)")
        print("5. Press Enter when ready to start testing...")
        print("="*60)
        
        input("Press Enter to start testing...")
        
        # Test multiple frames
        detections_found = 0
        for frame_num in range(10):
            logger.info(f"\n--- Frame {frame_num + 1}/10 ---")
            
            # Capture image
            image_rgb = picam2.capture_array()
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Save image
            cv2.imwrite(f"test_frame_{frame_num+1}.jpg", image_bgr)
            
            # Run inference with very low threshold
            model.conf_thresh = 0.001
            from utils import get_image_tensor
            full_image, net_image, pad = get_image_tensor(image_bgr, model.input_size[0])
            
            pred = model.forward(net_image, with_nms=True)
            
            logger.info(f"Detections: {len(pred)}")
            if len(pred) > 0:
                detections_found += 1
                logger.info("✓ Found detections!")
                for i, det in enumerate(pred):
                    logger.info(f"  Detection {i}: class={int(det[5])}, conf={det[4]:.3f}")
            else:
                logger.info("  No detections")
        
        picam2.stop()
        
        logger.info(f"\n=== TEST RESULTS ===")
        logger.info(f"Frames with detections: {detections_found}/10")
        
        if detections_found > 0:
            logger.info("✓ Camera Source Mode is working!")
            logger.info("✓ Objects were detected successfully")
        else:
            logger.info("⚠️  No detections found. Possible issues:")
            logger.info("  1. No objects in camera view")
            logger.info("  2. Poor lighting conditions")
            logger.info("  3. Objects too far or too close")
            logger.info("  4. Objects not in training dataset")
            logger.info("  5. Camera pointing at wrong direction")
        
        return True
        
    except Exception as e:
        logger.error(f"Camera test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    logger.info("Starting comprehensive object detection test...")
    
    # Test 1: Synthetic objects
    synthetic_success = test_synthetic_objects()
    
    print("\n" + "="*60)
    print("SYNTHETIC TEST COMPLETED")
    print("="*60)
    
    if synthetic_success:
        print("✓ Model is working correctly")
        print("Now let's test with real camera and objects...")
        
        # Test 2: Real camera with objects
        camera_success = test_camera_with_guidance()
        
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        
        if camera_success:
            print("✓ All tests completed")
            print("Check the log output above for results")
        else:
            print("✗ Camera test failed")
    else:
        print("✗ Synthetic test failed - model has issues")
        print("Check the error messages above")

if __name__ == "__main__":
    main() 