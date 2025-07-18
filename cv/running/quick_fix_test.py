#!/usr/bin/env python3
"""
Quick test to identify if the issue is in preprocessing or model wrapper
"""

import os
import sys
import cv2
import numpy as np
import logging

# Add the parent directory to the path
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_preprocessing_vs_wrapper():
    """Test if issue is in preprocessing or model wrapper"""
    logger.info("=== QUICK FIX TEST ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        model_path = configs['training']['weightsFile_rpi']
        
        # Load real test image
        test_img = cv2.imread('../../cv/datasets/testImages/appleHand_hand_4.jpg')
        if test_img is None:
            logger.error("Test image not found")
            return False
        
        logger.info(f"Test image shape: {test_img.shape}")
        cv2.imwrite("original_test_image.jpg", test_img)
        
        # Test 1: Ultralytics with original image
        logger.info("--- Test 1: Ultralytics with original image ---")
        from ultralytics import YOLO
        ultralytics_model = YOLO(model_path)
        results = ultralytics_model(test_img, conf=0.001, iou=0.9, verbose=False)
        
        logger.info(f"Ultralytics results: {len(results)}")
        if len(results) > 0:
            result = results[0]
            logger.info(f"Detections: {len(result.boxes)}")
            for i, box in enumerate(result.boxes):
                logger.info(f"  Detection {i}: class={int(box.cls)}, conf={float(box.conf):.3f}")
        else:
            logger.info("  No detections with ultralytics")
        
        # Test 2: Our preprocessing + Ultralytics
        logger.info("--- Test 2: Our preprocessing + Ultralytics ---")
        from utils import get_image_tensor
        full_image, net_image, pad = get_image_tensor(test_img, 640)
        
        logger.info(f"Our preprocessing output shape: {net_image.shape}")
        logger.info(f"Our preprocessing value range: {net_image.min():.3f} to {net_image.max():.3f}")
        
        # Convert back to uint8 for ultralytics
        net_image_uint8 = (net_image * 255).astype(np.uint8)
        cv2.imwrite("our_preprocessed_image.jpg", net_image_uint8)
        
        # Test ultralytics with our preprocessed image
        results2 = ultralytics_model(net_image_uint8, conf=0.001, iou=0.9, verbose=False)
        
        logger.info(f"Ultralytics with our preprocessing: {len(results2)}")
        if len(results2) > 0:
            result2 = results2[0]
            logger.info(f"Detections: {len(result2.boxes)}")
            for i, box in enumerate(result2.boxes):
                logger.info(f"  Detection {i}: class={int(box.cls)}, conf={float(box.conf):.3f}")
        else:
            logger.info("  No detections with our preprocessing + ultralytics")
        
        # Test 3: Our model wrapper
        logger.info("--- Test 3: Our model wrapper ---")
        from rpiModel import RaspberryPiModel
        
        thresh = min(configs['runTime']['distSettings']['handThreshold'],
                    configs['runTime']['distSettings']['objectThreshold'])
        
        our_model = RaspberryPiModel(
            model_file=model_path,
            names_file="../../cv/datasets/day2_partII_2138Images/data.yaml",
            conf_thresh=thresh,
            iou_thresh=configs['runTime']['distSettings']['nmsIouThreshold'],
            v8=True,
            use_gpu=False,
            num_threads=4
        )
        
        our_model.conf_thresh = 0.001
        pred = our_model.forward(net_image, with_nms=True)
        
        logger.info(f"Our model wrapper results: {len(pred)} detections")
        if len(pred) > 0:
            for i, det in enumerate(pred):
                logger.info(f"  Detection {i}: class={int(det[5])}, conf={det[4]:.3f}")
        else:
            logger.info("  No detections with our model wrapper")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if len(results) > 0 and len(results2) == 0:
            print("❌ ISSUE FOUND: Our preprocessing breaks the image!")
            print("   Ultralytics works with original but fails with our preprocessing")
        elif len(results) > 0 and len(results2) > 0 and len(pred) == 0:
            print("❌ ISSUE FOUND: Our model wrapper has a problem!")
            print("   Both ultralytics tests work, but our wrapper fails")
        elif len(results) > 0 and len(results2) > 0 and len(pred) > 0:
            print("✅ ALL TESTS PASS: Everything works!")
        else:
            print("⚠️  UNEXPECTED RESULTS: Need more investigation")
        
        return True
        
    except Exception as e:
        logger.error(f"Quick fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_preprocessing_vs_wrapper() 