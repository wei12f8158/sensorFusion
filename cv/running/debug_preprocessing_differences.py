#!/usr/bin/env python3
"""
Debug script to identify differences between AI Camera Mode and Camera Source Mode
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

def test_ultralytics_preprocessing():
    """Test the exact preprocessing that ultralytics.YOLO uses"""
    logger.info("=== TESTING ULTRALYTICS PREPROCESSING ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        model_path = configs['training']['weightsFile_rpi']
        
        # Load model with ultralytics
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        logger.info(f"Model loaded: {model_path}")
        logger.info(f"Model input size: {model.input_size}")
        
        # Create a test image (same as synthetic test)
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green
        cv2.rectangle(test_img, (400, 300), (500, 400), (255, 0, 0), -1)  # Blue  
        cv2.rectangle(test_img, (250, 250), (350, 350), (0, 0, 255), -1)  # Red
        
        cv2.imwrite("ultralytics_test_input.jpg", test_img)
        logger.info("Created test image: ultralytics_test_input.jpg")
        
        # Test with ultralytics directly
        results = model(test_img, conf=0.001, iou=0.9, verbose=False)
        
        logger.info(f"Ultralytics results: {len(results)}")
        if len(results) > 0:
            result = results[0]
            logger.info(f"Detections: {len(result.boxes)}")
            for i, box in enumerate(result.boxes):
                logger.info(f"  Detection {i}: class={int(box.cls)}, conf={float(box.conf):.3f}")
        else:
            logger.info("  No detections with ultralytics")
        
        return True
        
    except Exception as e:
        logger.error(f"Ultralytics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_preprocessing():
    """Test our custom preprocessing vs ultralytics"""
    logger.info("=== TESTING CUSTOM PREPROCESSING ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        model_path = configs['training']['weightsFile_rpi']
        
        # Load model with our custom wrapper
        from rpiModel import RaspberryPiModel
        
        thresh = min(configs['runTime']['distSettings']['handThreshold'],
                    configs['runTime']['distSettings']['objectThreshold'])
        
        model = RaspberryPiModel(
            model_file=model_path,
            names_file="../../cv/datasets/day2_partII_2138Images/data.yaml",
            conf_thresh=thresh,
            iou_thresh=configs['runTime']['distSettings']['nmsIouThreshold'],
            v8=True,
            use_gpu=False,
            num_threads=4
        )
        
        logger.info("✓ Custom model loaded")
        
        # Create same test image
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green
        cv2.rectangle(test_img, (400, 300), (500, 400), (255, 0, 0), -1)  # Blue  
        cv2.rectangle(test_img, (250, 250), (350, 350), (0, 0, 255), -1)  # Red
        
        cv2.imwrite("custom_test_input.jpg", test_img)
        logger.info("Created test image: custom_test_input.jpg")
        
        # Test with our custom preprocessing
        from utils import get_image_tensor
        full_image, net_image, pad = get_image_tensor(test_img, model.input_size[0])
        
        logger.info(f"Custom preprocessing - Input shape: {test_img.shape}")
        logger.info(f"Custom preprocessing - Output shape: {net_image.shape}")
        logger.info(f"Custom preprocessing - Pad: {pad}")
        
        # Save the preprocessed image
        cv2.imwrite("custom_preprocessed.jpg", (net_image * 255).astype(np.uint8))
        logger.info("Saved preprocessed image: custom_preprocessed.jpg")
        
        # Run inference
        model.conf_thresh = 0.001
        pred = model.forward(net_image, with_nms=True)
        
        logger.info(f"Custom preprocessing results: {len(pred)} detections")
        if len(pred) > 0:
            for i, det in enumerate(pred):
                logger.info(f"  Detection {i}: class={int(det[5])}, conf={det[4]:.3f}")
        else:
            logger.info("  No detections with custom preprocessing")
        
        return True
        
    except Exception as e:
        logger.error(f"Custom preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imx500_preprocessing():
    """Test IMX500 preprocessing (simulated)"""
    logger.info("=== TESTING IMX500 PREPROCESSING (SIMULATED) ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Create same test image
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green
        cv2.rectangle(test_img, (400, 300), (500, 400), (255, 0, 0), -1)  # Blue  
        cv2.rectangle(test_img, (250, 250), (350, 350), (0, 0, 255), -1)  # Red
        
        cv2.imwrite("imx500_test_input.jpg", test_img)
        logger.info("Created test image: imx500_test_input.jpg")
        
        # Simulate IMX500 preprocessing based on the code we found
        # From IMX500/picamera2/picamera2/devices/imx500/postprocess_yolov5.py
        
        # IMX500 uses yolov5n_preprocess function
        def yolov5n_preprocess(img):
            # AspectPreservingResizeWithPad
            new_height = 640
            new_width = 640
            pad_value = 114  # This is different from our 100!
            resize_method = 3  # area
            resize_ratio = max(img.shape[0] / new_height, img.shape[1] / new_width)
            height_tag = int(np.round(img.shape[0] / resize_ratio))
            width_tag = int(np.round(img.shape[1] / resize_ratio))
            pad_values = ((int((new_height - height_tag) / 2), int((new_height - height_tag) / 2 + 0.5)),
                          (int((new_width - width_tag) / 2), int((new_width - width_tag) / 2 + 0.5)),
                          (0, 0))

            resized_img = cv2.resize(img, (width_tag, height_tag), interpolation=resize_method)
            padded_img = np.pad(resized_img, pad_values, constant_values=pad_value)

            # Normalize
            mean = 0
            std = 255
            normalized_img = (padded_img - mean) / std

            return normalized_img
        
        # Apply IMX500 preprocessing
        imx500_processed = yolov5n_preprocess(test_img)
        
        logger.info(f"IMX500 preprocessing - Input shape: {test_img.shape}")
        logger.info(f"IMX500 preprocessing - Output shape: {imx500_processed.shape}")
        logger.info(f"IMX500 preprocessing - Value range: {imx500_processed.min():.3f} to {imx500_processed.max():.3f}")
        
        # Save the IMX500 preprocessed image
        cv2.imwrite("imx500_preprocessed.jpg", (imx500_processed * 255).astype(np.uint8))
        logger.info("Saved IMX500 preprocessed image: imx500_preprocessed.jpg")
        
        # Compare with our preprocessing
        from utils import get_image_tensor
        full_image, net_image, pad = get_image_tensor(test_img, 640)
        
        logger.info(f"Our preprocessing - Output shape: {net_image.shape}")
        logger.info(f"Our preprocessing - Value range: {net_image.min():.3f} to {net_image.max():.3f}")
        
        # Check if they're the same
        if np.allclose(imx500_processed, net_image, atol=1e-6):
            logger.info("✓ Preprocessing is identical!")
        else:
            logger.warning("⚠️  Preprocessing is different!")
            logger.info("This could be the cause of the detection issue")
            
            # Show differences
            diff = np.abs(imx500_processed - net_image)
            logger.info(f"Max difference: {diff.max():.6f}")
            logger.info(f"Mean difference: {diff.mean():.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"IMX500 preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_imx500_preprocessing():
    """Test our model with IMX500-style preprocessing"""
    logger.info("=== TESTING WITH IMX500 PREPROCESSING ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        model_path = configs['training']['weightsFile_rpi']
        
        # Load model
        from rpiModel import RaspberryPiModel
        
        thresh = min(configs['runTime']['distSettings']['handThreshold'],
                    configs['runTime']['distSettings']['objectThreshold'])
        
        model = RaspberryPiModel(
            model_file=model_path,
            names_file="../../cv/datasets/day2_partII_2138Images/data.yaml",
            conf_thresh=thresh,
            iou_thresh=configs['runTime']['distSettings']['nmsIouThreshold'],
            v8=True,
            use_gpu=False,
            num_threads=4
        )
        
        logger.info("✓ Model loaded")
        
        # Create test image
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green
        cv2.rectangle(test_img, (400, 300), (500, 400), (255, 0, 0), -1)  # Blue  
        cv2.rectangle(test_img, (250, 250), (350, 350), (0, 0, 255), -1)  # Red
        
        # Apply IMX500 preprocessing
        def yolov5n_preprocess(img):
            new_height = 640
            new_width = 640
            pad_value = 114
            resize_method = 3
            resize_ratio = max(img.shape[0] / new_height, img.shape[1] / new_width)
            height_tag = int(np.round(img.shape[0] / resize_ratio))
            width_tag = int(np.round(img.shape[1] / resize_ratio))
            pad_values = ((int((new_height - height_tag) / 2), int((new_height - height_tag) / 2 + 0.5)),
                          (int((new_width - width_tag) / 2), int((new_width - width_tag) / 2 + 0.5)),
                          (0, 0))

            resized_img = cv2.resize(img, (width_tag, height_tag), interpolation=resize_method)
            padded_img = np.pad(resized_img, pad_values, constant_values=pad_value)

            mean = 0
            std = 255
            normalized_img = (padded_img - mean) / std

            return normalized_img
        
        imx500_processed = yolov5n_preprocess(test_img)
        
        # Run inference with IMX500 preprocessing
        model.conf_thresh = 0.001
        pred = model.forward(imx500_processed, with_nms=True)
        
        logger.info(f"IMX500 preprocessing results: {len(pred)} detections")
        if len(pred) > 0:
            logger.info("🎉 SUCCESS! IMX500 preprocessing fixed the issue!")
            for i, det in enumerate(pred):
                logger.info(f"  Detection {i}: class={int(det[5])}, conf={det[4]:.3f}")
        else:
            logger.info("  Still no detections with IMX500 preprocessing")
        
        return True
        
    except Exception as e:
        logger.error(f"IMX500 preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    logger.info("Starting comprehensive preprocessing debug...")
    
    print("\n" + "="*60)
    print("PREPROCESSING DIFFERENCES DEBUG")
    print("="*60)
    print("This will test different preprocessing methods to find the issue")
    print("="*60)
    
    # Test 1: Ultralytics preprocessing
    ultralytics_success = test_ultralytics_preprocessing()
    
    print("\n" + "="*60)
    print("ULTRALYTICS TEST COMPLETED")
    print("="*60)
    
    # Test 2: Our custom preprocessing
    custom_success = test_custom_preprocessing()
    
    print("\n" + "="*60)
    print("CUSTOM PREPROCESSING TEST COMPLETED")
    print("="*60)
    
    # Test 3: IMX500 preprocessing comparison
    imx500_compare_success = test_imx500_preprocessing()
    
    print("\n" + "="*60)
    print("IMX500 COMPARISON COMPLETED")
    print("="*60)
    
    # Test 4: Test with IMX500 preprocessing
    imx500_test_success = test_with_imx500_preprocessing()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if imx500_test_success:
        print("✓ All tests completed")
        print("Check the results above to see if IMX500 preprocessing fixed the issue")
    else:
        print("✗ Some tests failed")

if __name__ == "__main__":
    main() 