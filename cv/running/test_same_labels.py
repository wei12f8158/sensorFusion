#!/usr/bin/env python3
"""
Test Camera Source Mode with the exact same labels file as AI Camera Mode
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

def test_with_ai_camera_labels():
    """Test using the exact same labels file as AI Camera Mode"""
    logger.info("=== TESTING WITH AI CAMERA LABELS ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Use the exact same labels file as AI Camera Mode
        ai_camera_labels = "../../IMX500/yolo8_19_UL-New_day2_fullLabTopCam_60epochs_imx_model/labels.txt"
        
        if not os.path.exists(ai_camera_labels):
            logger.error(f"AI Camera labels file not found: {ai_camera_labels}")
            return False
        
        # Read the labels
        with open(ai_camera_labels, 'r') as f:
            labels = f.read().splitlines()
        
        logger.info(f"Using AI Camera labels: {labels}")
        logger.info(f"Number of classes: {len(labels)}")
        
        # Load model with AI Camera labels
        model_path = configs['training']['weightsFile_rpi']
        
        from rpiModel import RaspberryPiModel
        
        thresh = min(configs['runTime']['distSettings']['handThreshold'],
                    configs['runTime']['distSettings']['objectThreshold'])
        
        model = RaspberryPiModel(
            model_file=model_path,
            names_file=ai_camera_labels,  # Use AI Camera labels instead of dataset labels
            conf_thresh=thresh,
            iou_thresh=configs['runTime']['distSettings']['nmsIouThreshold'],
            v8=True,
            use_gpu=False,
            num_threads=4
        )
        
        logger.info("✓ Model loaded with AI Camera labels")
        
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
        
        print("\n" + "="*60)
        print("TESTING WITH AI CAMERA LABELS")
        print("="*60)
        print("This test uses the EXACT same labels file as AI Camera Mode")
        print("If this works, the issue was the labels file!")
        print("Place objects in front of the camera and press Enter...")
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
            cv2.imwrite(f"ai_labels_frame_{frame_num+1}.jpg", image_bgr)
            
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
                    class_id = int(det[5])
                    class_name = labels[class_id] if class_id < len(labels) else f"Unknown({class_id})"
                    logger.info(f"  Detection {i}: {class_name}, conf={det[4]:.3f}")
            else:
                logger.info("  No detections")
        
        picam2.stop()
        
        logger.info(f"\n=== TEST RESULTS ===")
        logger.info(f"Frames with detections: {detections_found}/10")
        
        if detections_found > 0:
            logger.info("🎉 SUCCESS! AI Camera labels fixed the issue!")
            logger.info("The problem was the labels file difference!")
        else:
            logger.info("⚠️  Still no detections with AI Camera labels")
            logger.info("The issue is not the labels file")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_label_files():
    """Compare the two label files to see the differences"""
    logger.info("=== COMPARING LABEL FILES ===")
    
    try:
        # AI Camera labels
        ai_camera_labels = "../../IMX500/yolo8_19_UL-New_day2_fullLabTopCam_60epochs_imx_model/labels.txt"
        
        # Camera Source labels (from dataset)
        dataset_labels = "../../cv/datasets/day2_partII_2138Images/data.yaml"
        
        # Read AI Camera labels
        with open(ai_camera_labels, 'r') as f:
            ai_labels = f.read().splitlines()
        
        # Read dataset labels
        import yaml
        with open(dataset_labels, 'r') as f:
            dataset_data = yaml.safe_load(f)
        dataset_names = dataset_data.get('names', [])
        
        logger.info(f"AI Camera labels ({len(ai_labels)}): {ai_labels}")
        logger.info(f"Dataset labels ({len(dataset_names)}): {dataset_names}")
        
        # Compare
        if ai_labels == dataset_names:
            logger.info("✓ Labels are identical!")
            return True
        else:
            logger.warning("⚠️  Labels are different!")
            logger.info("This could be the cause of the detection issue")
            return False
            
    except Exception as e:
        logger.error(f"Label comparison failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting label comparison and test...")
    
    # First, compare the label files
    labels_match = compare_label_files()
    
    if labels_match:
        logger.info("Labels are the same, testing with AI Camera labels anyway...")
    
    # Test with AI Camera labels
    success = test_with_ai_camera_labels()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if success:
        print("✓ Test completed")
        print("Check the results above to see if AI Camera labels fixed the issue")
    else:
        print("✗ Test failed")

if __name__ == "__main__":
    main() 