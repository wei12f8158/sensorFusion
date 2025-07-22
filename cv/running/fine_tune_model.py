#!/usr/bin/env python3
"""
Fine-tune the YOLO model for better detection
"""

import os
import sys
import yaml
import logging
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_fine_tune_config():
    """Create fine-tuning configuration"""
    logger.info("=== CREATE FINE-TUNE CONFIG ===")
    
    try:
        # Load current configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Create fine-tuning config
        fine_tune_config = {
            'path': '../datasets/fine_tune_dataset',  # Dataset path
            'train': 'images/train',  # Train images
            'val': 'images/val',      # Validation images
            'test': 'images/test',    # Test images
            
            # Classes
            'nc': 9,  # Number of classes
            'names': [
                'apple',
                'ball', 
                'bottle',
                'clip',
                'glove',
                'lid',
                'plate',
                'spoon',
                'tape_spool'
            ]
        }
        
        # Save fine-tuning config
        config_path = '../datasets/fine_tune_dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(fine_tune_config, f, default_flow_style=False)
        
        logger.info(f"Created fine-tuning config: {config_path}")
        return config_path
        
    except Exception as e:
        logger.error(f"Failed to create fine-tuning config: {e}")
        return None

def fine_tune_model():
    """Fine-tune the YOLO model"""
    logger.info("=== FINE-TUNE MODEL ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Get current model path
        current_model = configs['training']['weightsFile_rpi']
        logger.info(f"Current model: {current_model}")
        
        # Create fine-tuning config
        fine_tune_config = create_fine_tune_config()
        if not fine_tune_config:
            return False
        
        # Fine-tuning parameters
        epochs = 50
        batch_size = 16
        img_size = 640
        learning_rate = 0.001
        
        # Import ultralytics
        from ultralytics import YOLO
        
        # Load the current model
        model = YOLO(current_model)
        logger.info("Loaded current model for fine-tuning")
        
        # Fine-tune the model
        logger.info(f"Starting fine-tuning for {epochs} epochs...")
        logger.info(f"Batch size: {batch_size}, Image size: {img_size}, Learning rate: {learning_rate}")
        
        # Fine-tune
        results = model.train(
            data=fine_tune_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=learning_rate,
            patience=10,
            save=True,
            save_period=10,
            project='fine_tuned_model',
            name=f'fine_tune_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        logger.info("Fine-tuning completed!")
        logger.info(f"Results saved in: fine_tuned_model/")
        
        # Get the best model path
        best_model = results.best
        logger.info(f"Best model: {best_model}")
        
        return True
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def prepare_dataset():
    """Prepare dataset for fine-tuning"""
    logger.info("=== PREPARE DATASET ===")
    
    try:
        # Create dataset structure
        dataset_dir = "../datasets/fine_tune_dataset"
        train_dir = os.path.join(dataset_dir, "images/train")
        val_dir = os.path.join(dataset_dir, "images/val")
        test_dir = os.path.join(dataset_dir, "images/test")
        train_labels = os.path.join(dataset_dir, "labels/train")
        val_labels = os.path.join(dataset_dir, "labels/val")
        test_labels = os.path.join(dataset_dir, "labels/test")
        
        # Create directories
        for dir_path in [train_dir, val_dir, test_dir, train_labels, val_labels, test_labels]:
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info(f"Created dataset structure in: {dataset_dir}")
        logger.info("Next steps:")
        logger.info("1. Copy training images to: images/train/")
        logger.info("2. Copy validation images to: images/val/")
        logger.info("3. Copy test images to: images/test/")
        logger.info("4. Create YOLO format labels in: labels/train/, labels/val/, labels/test/")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune YOLO model")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset structure")
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune the model")
    parser.add_argument("--all", action="store_true", help="Prepare dataset and fine-tune")
    
    args = parser.parse_args()
    
    if args.prepare or args.all:
        prepare_dataset()
    
    if args.fine_tune or args.all:
        fine_tune_model()
    
    if not any([args.prepare, args.fine_tune, args.all]):
        print("Usage:")
        print("  python3 fine_tune_model.py --prepare    # Prepare dataset structure")
        print("  python3 fine_tune_model.py --fine-tune  # Fine-tune the model")
        print("  python3 fine_tune_model.py --all        # Do both") 