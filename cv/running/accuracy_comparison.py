#!/usr/bin/env python3
"""
Compare accuracy between IMX500 and Pi 5 models
"""

import os
import sys
import cv2
import numpy as np
import logging
from datetime import datetime
import json

# Add the parent directory to the path
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pi5_accuracy():
    """Test accuracy on Pi 5 with USB camera"""
    logger.info("=== TESTING PI 5 ACCURACY ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Determine device
        import platform
        machine = platform.machine()
        
        if machine == "aarch64":
            device = "rpi"
        else:
            import torch
            device = "cpu" 
            if torch.cuda.is_available(): device = "cuda" 
            if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"
        
        # Import and initialize components
        from modelRunTime import modelRunTime
        from distance import distanceCalculator
        from display import displayHandObject
        
        model_runtime = modelRunTime(configs, device)
        distance_calc = distanceCalculator(configs['training']['imageSize'], configs['runTime']['distSettings'])
        display_obj = displayHandObject(configs)
        
        # Capture frame
        logger.info("Capturing frame with USB camera...")
        img_src = configs['runTime']['imgSrc']
        use_imx500 = configs['runTime']['use_imx500']
        
        if img_src == "camera" and not use_imx500:
            # USB camera
            camera_indices = [0, 2, 3, 8]
            cap = None
            
            for cam_id in camera_indices:
                logger.info(f"Trying USB camera index {cam_id}...")
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"USB camera index {cam_id} works!")
                        break
                    else:
                        cap.release()
                        cap = None
                else:
                    cap.release()
                    cap = None
            
            if cap is None:
                logger.error("Failed to open any USB camera")
                return None
        else:
            logger.error("IMX500 mode enabled, cannot test Pi 5 accuracy")
            return None
        
        # Capture multiple frames for accuracy testing
        results = []
        frame_count = 10
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            # Run detection
            predictions = model_runtime.runInference(frame)
            
            if predictions is not None and len(predictions) > 0:
                # Process through distance calculator
                distance_calc.zeroData()
                valid = distance_calc.loadData(predictions)
                
                if valid:
                    frame_result = {
                        'frame': i,
                        'detections': len(predictions),
                        'hand_detected': distance_calc.nHands > 0,
                        'target_detected': distance_calc.nNonHand > 0,
                        'hand_confidence': float(distance_calc.handObject[4]) if distance_calc.handObject is not None else 0,
                        'target_confidence': float(distance_calc.grabObject[4]) if distance_calc.grabObject is not None else 0,
                        'target_class': int(distance_calc.grabObject[5]) if distance_calc.grabObject is not None else -1,
                        'distance': distance_calc.bestDist,
                        'predictions': []
                    }
                    
                    for pred in predictions:
                        frame_result['predictions'].append({
                            'class_id': int(pred[5]),
                            'confidence': float(pred[4]),
                            'bbox': [float(x) for x in pred[:4]]
                        })
                    
                    results.append(frame_result)
                    
                    logger.info(f"Frame {i}: {len(predictions)} detections, hand_conf={frame_result['hand_confidence']:.3f}, target_conf={frame_result['target_confidence']:.3f}")
        
        # Clean up
        cap.release()
        
        return results
        
    except Exception as e:
        logger.error(f"Pi 5 accuracy test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_imx500_accuracy():
    """Test accuracy on IMX500"""
    logger.info("=== TESTING IMX500 ACCURACY ===")
    
    try:
        # Check if IMX500 model exists
        imx500_model = "../../IMX500/final_output/network.rpk"
        imx500_labels = "../../IMX500/yolo8_19_UL-New_day2_fullLabTopCam_60epochs_imx_model/labels.txt"
        
        if not os.path.exists(imx500_model):
            logger.error(f"IMX500 model not found: {imx500_model}")
            return None
        
        if not os.path.exists(imx500_labels):
            logger.error(f"IMX500 labels not found: {imx500_labels}")
            return None
        
        # Import IMX500 demo
        sys.path.insert(0, '../../IMX500/picamera2/examples/imx500')
        
        try:
            from imx500_object_detection_demo import run_imx500_detection
        except ImportError:
            logger.error("IMX500 demo not available")
            return None
        
        # Run IMX500 detection
        results = run_imx500_detection(imx500_model, imx500_labels, frame_count=10)
        
        return results
        
    except Exception as e:
        logger.error(f"IMX500 accuracy test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_accuracy_results(pi5_results, imx500_results):
    """Analyze and compare accuracy results"""
    logger.info("=== ACCURACY ANALYSIS ===")
    
    if pi5_results is None and imx500_results is None:
        logger.error("No results to analyze")
        return
    
    analysis = {
        'pi5': {},
        'imx500': {},
        'comparison': {}
    }
    
    # Analyze Pi 5 results
    if pi5_results:
        pi5_detections = [r['detections'] for r in pi5_results]
        pi5_hand_conf = [r['hand_confidence'] for r in pi5_results if r['hand_confidence'] > 0]
        pi5_target_conf = [r['target_confidence'] for r in pi5_results if r['target_confidence'] > 0]
        
        analysis['pi5'] = {
            'total_frames': len(pi5_results),
            'avg_detections': np.mean(pi5_detections) if pi5_detections else 0,
            'avg_hand_confidence': np.mean(pi5_hand_conf) if pi5_hand_conf else 0,
            'avg_target_confidence': np.mean(pi5_target_conf) if pi5_target_conf else 0,
            'hand_detection_rate': sum(1 for r in pi5_results if r['hand_detected']) / len(pi5_results),
            'target_detection_rate': sum(1 for r in pi5_results if r['target_detected']) / len(pi5_results)
        }
        
        logger.info("Pi 5 Results:")
        logger.info(f"  Total frames: {analysis['pi5']['total_frames']}")
        logger.info(f"  Avg detections: {analysis['pi5']['avg_detections']:.2f}")
        logger.info(f"  Avg hand confidence: {analysis['pi5']['avg_hand_confidence']:.3f}")
        logger.info(f"  Avg target confidence: {analysis['pi5']['avg_target_confidence']:.3f}")
        logger.info(f"  Hand detection rate: {analysis['pi5']['hand_detection_rate']:.2%}")
        logger.info(f"  Target detection rate: {analysis['pi5']['target_detection_rate']:.2%}")
    
    # Analyze IMX500 results
    if imx500_results:
        imx500_detections = [r['detections'] for r in imx500_results]
        imx500_hand_conf = [r['hand_confidence'] for r in imx500_results if r['hand_confidence'] > 0]
        imx500_target_conf = [r['target_confidence'] for r in imx500_results if r['target_confidence'] > 0]
        
        analysis['imx500'] = {
            'total_frames': len(imx500_results),
            'avg_detections': np.mean(imx500_detections) if imx500_detections else 0,
            'avg_hand_confidence': np.mean(imx500_hand_conf) if imx500_hand_conf else 0,
            'avg_target_confidence': np.mean(imx500_target_conf) if imx500_target_conf else 0,
            'hand_detection_rate': sum(1 for r in imx500_results if r['hand_detected']) / len(imx500_results),
            'target_detection_rate': sum(1 for r in imx500_results if r['target_detected']) / len(imx500_results)
        }
        
        logger.info("IMX500 Results:")
        logger.info(f"  Total frames: {analysis['imx500']['total_frames']}")
        logger.info(f"  Avg detections: {analysis['imx500']['avg_detections']:.2f}")
        logger.info(f"  Avg hand confidence: {analysis['imx500']['avg_hand_confidence']:.3f}")
        logger.info(f"  Avg target confidence: {analysis['imx500']['avg_target_confidence']:.3f}")
        logger.info(f"  Hand detection rate: {analysis['imx500']['hand_detection_rate']:.2%}")
        logger.info(f"  Target detection rate: {analysis['imx500']['target_detection_rate']:.2%}")
    
    # Compare results
    if pi5_results and imx500_results:
        analysis['comparison'] = {
            'confidence_diff': analysis['imx500']['avg_target_confidence'] - analysis['pi5']['avg_target_confidence'],
            'detection_rate_diff': analysis['imx500']['target_detection_rate'] - analysis['pi5']['target_detection_rate'],
            'detection_count_diff': analysis['imx500']['avg_detections'] - analysis['pi5']['avg_detections']
        }
        
        logger.info("Comparison:")
        logger.info(f"  Confidence difference (IMX500 - Pi5): {analysis['comparison']['confidence_diff']:.3f}")
        logger.info(f"  Detection rate difference (IMX500 - Pi5): {analysis['comparison']['detection_rate_diff']:.2%}")
        logger.info(f"  Detection count difference (IMX500 - Pi5): {analysis['comparison']['detection_count_diff']:.2f}")
        
        if analysis['comparison']['confidence_diff'] > 0:
            logger.info("✅ IMX500 has higher confidence")
        else:
            logger.info("⚠️ Pi 5 has higher confidence")
        
        if analysis['comparison']['detection_rate_diff'] > 0:
            logger.info("✅ IMX500 has higher detection rate")
        else:
            logger.info("⚠️ Pi 5 has higher detection rate")
    
    # Save analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = f"accuracy_analysis_{timestamp}.json"
    
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Analysis saved to: {analysis_file}")
    
    return analysis

def main():
    """Main accuracy comparison function"""
    logger.info("=== ACCURACY COMPARISON ===")
    
    # Test Pi 5 accuracy
    pi5_results = test_pi5_accuracy()
    
    # Test IMX500 accuracy
    imx500_results = test_imx500_accuracy()
    
    # Analyze results
    analysis = analyze_accuracy_results(pi5_results, imx500_results)
    
    logger.info("=== ACCURACY COMPARISON COMPLETE ===")

if __name__ == "__main__":
    main() 