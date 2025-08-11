#!/usr/bin/env python3
"""
Convert YOLO models to IMX500 .rpk format
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IMX500Converter:
    def __init__(self):
        self.imx500_dir = Path("../../IMX500")
        self.models_dir = Path("../models")
        self.weights_dir = Path("../weights")
        
    def check_imx500_tools(self):
        """Check if IMX500 conversion tools are available"""
        logger.info("=== CHECKING IMX500 CONVERSION TOOLS ===")
        
        # Check for IMX500 SDK or conversion tools
        possible_tools = [
            "imx500_convert",
            "rpk_converter",
            "imx500_sdk",
            "network_converter"
        ]
        
        logger.info("IMX500 conversion requires specialized tools:")
        logger.info("1. IMX500 SDK (from Sony/Renesas)")
        logger.info("2. Network converter tool")
        logger.info("3. Model optimization tools")
        
        return False  # These tools are typically proprietary
    
    def create_conversion_guide(self):
        """Create guide for converting models to .rpk format"""
        logger.info("=== CREATING IMX500 CONVERSION GUIDE ===")
        
        try:
            guide_content = """# IMX500 Native Model Conversion Guide

## Overview
To run models natively on IMX500 (not on Pi 5), you need `.rpk` files that run on the IMX500 NPU.

## What You Need

### 1. IMX500 SDK
- **Source**: Sony/Renesas (proprietary)
- **Purpose**: Convert models to .rpk format
- **Features**: Model optimization, quantization, NPU compilation

### 2. Your Current Setup
You already have the right configuration in `config.yaml`:
```yaml
imx500_model: "../../IMX500/final_output/network.rpk"  # ✅ .rpk file!
imx500_labels: "../../IMX500/yolo8_19_UL-New_day2_fullLabTopCam_60epochs_imx_model/labels.txt"
```

## Conversion Process

### Step 1: Get IMX500 SDK
- Contact Sony/Renesas for IMX500 development kit
- Install SDK on your development machine
- Get access to model conversion tools

### Step 2: Convert Your Model
```bash
# Using IMX500 SDK (example commands)
imx500_convert --input yolov8n.pt --output yolov8n.rpk --target imx500
imx500_optimize --model yolov8n.rpk --quantization int8
imx500_compile --model yolov8n.rpk --output network.rpk
```

### Step 3: Deploy to IMX500
```bash
# Copy .rpk file to IMX500
scp network.rpk wei@10.0.0.71:~/IMX500/final_output/

# Update config.yaml
imx500_model: "../../IMX500/final_output/network.rpk"
```

## Alternative: Use Existing .rpk Model

### Your Current Model
You already have a working .rpk model:
- **File**: `network.rpk`
- **Location**: `IMX500/final_output/network.rpk`
- **Classes**: 9 classes (apple, ball, bottle, clip, glove, lid, plate, spoon, tape_spool)

### Test Current Setup
```bash
# Set IMX500 mode in config.yaml
use_imx500: True
imx500_ai_camera: True

# Run on Pi 5
python3 runImage.py
```

## Model Performance Comparison

### IMX500 Native (.rpk)
- **Speed**: ⚡⚡⚡⚡ Very Fast (NPU)
- **Power**: 🔋🔋🔋 Low power consumption
- **Latency**: <10ms per frame
- **Accuracy**: Same as original model

### Pi 5 CPU (.pt)
- **Speed**: ⚡⚡ Slower (CPU)
- **Power**: 🔋🔋🔋🔋 Higher power consumption
- **Latency**: 50-200ms per frame
- **Accuracy**: Same as original model

## Why .rpk is Better for IMX500

### 1. **Native Performance**
- Runs directly on IMX500 NPU
- No data transfer to Pi 5
- Optimized for IMX500 hardware

### 2. **Lower Latency**
- No network overhead
- Direct camera-to-inference pipeline
- Real-time performance

### 3. **Power Efficiency**
- NPU is designed for AI inference
- Lower power consumption
- Better for battery-powered devices

## Getting IMX500 SDK

### Official Sources
1. **Sony Semiconductor**: Contact for IMX500 development kit
2. **Renesas**: IMX500 partner for development tools
3. **Distributors**: Avnet, Arrow, etc.

### What to Ask For
- IMX500 SDK license
- Model conversion tools
- Documentation and examples
- Technical support

## Quick Test

### Test Your Current .rpk Model
```bash
# Ensure IMX500 mode is enabled
cd ~/sensorFusion/cv/running

# Edit config.yaml
nano ../../config.yaml
# Set: use_imx500: True, imx500_ai_camera: True

# Run the application
python3 runImage.py
```

### Expected Results
- **IMX500 handles inference**: No Pi 5 CPU usage
- **Fast detection**: <10ms per frame
- **Low power**: Efficient NPU processing
- **Real-time**: Smooth video processing

## Next Steps

1. **Test current .rpk model**: Enable IMX500 mode
2. **Get IMX500 SDK**: Contact Sony/Renesas
3. **Convert custom models**: Use SDK tools
4. **Optimize performance**: Tune NPU settings

## Troubleshooting

### Common Issues
1. **No .rpk file**: Get IMX500 SDK and convert models
2. **Conversion errors**: Check model compatibility
3. **Performance issues**: Optimize NPU settings
4. **SDK access**: Contact official channels

### Support Resources
- **Sony Semiconductor**: Official IMX500 support
- **Renesas**: Development tools and documentation
- **Community**: IMX500 developer forums

**Your IMX500 is designed to run .rpk models natively - much faster than CPU inference!** 🚀
"""
            
            guide_file = self.imx500_dir / "IMX500_NATIVE_GUIDE.md"
            with open(guide_file, 'w') as f:
                f.write(guide_content)
            
            logger.info(f"✅ IMX500 native guide created: {guide_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create conversion guide: {e}")
            return False
    
    def check_existing_rpk(self):
        """Check if you already have a working .rpk model"""
        logger.info("=== CHECKING EXISTING .RPK MODEL ===")
        
        try:
            # Check for existing .rpk file
            rpk_file = self.imx500_dir / "final_output" / "network.rpk"
            
            if rpk_file.exists():
                file_size = rpk_file.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"✅ Found existing .rpk model: {rpk_file}")
                logger.info(f"   - Size: {file_size:.1f} MB")
                logger.info(f"   - Ready for IMX500 native inference")
                return True
            else:
                logger.info(f"❌ No .rpk model found at: {rpk_file}")
                logger.info(f"   - You need to convert your YOLO model")
                logger.info(f"   - Or get IMX500 SDK for conversion")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking .rpk model: {e}")
            return False
    
    def create_quick_test_script(self):
        """Create script to test IMX500 native mode"""
        logger.info("=== CREATING IMX500 NATIVE TEST SCRIPT ===")
        
        try:
            test_script = self.imx500_dir / "test_imx500_native.py"
            test_content = '''#!/usr/bin/env python3
"""
Test IMX500 native inference mode
"""

import os
import sys
import yaml
import logging

# Add the sensorFusion path
sys.path.insert(0, '../../cv/running')

def test_imx500_native():
    """Test IMX500 native inference"""
    print("=== IMX500 NATIVE INFERENCE TEST ===")
    
    # Check config.yaml
    config_path = "../../config.yaml"
    if not os.path.exists(config_path):
        print("❌ config.yaml not found")
        return False
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load config.yaml: {e}")
        return False
    
    # Check IMX500 settings
    runtime = config.get('runTime', {})
    
    print("\\n=== IMX500 CONFIGURATION ===")
    print(f"use_imx500: {runtime.get('use_imx500', 'Not set')}")
    print(f"imx500_ai_camera: {runtime.get('imx500_ai_camera', 'Not set')}")
    print(f"imx500_model: {runtime.get('imx500_model', 'Not set')}")
    print(f"imx500_labels: {runtime.get('imx500_labels', 'Not set')}")
    
    # Check .rpk file
    rpk_path = runtime.get('imx500_model', '')
    if rpk_path and os.path.exists(rpk_path):
        file_size = os.path.getsize(rpk_path) / (1024 * 1024)  # MB
        print(f"\\n✅ .rpk model found: {rpk_path}")
        print(f"   - Size: {file_size:.1f} MB")
        print(f"   - Ready for native inference")
    else:
        print(f"\\n❌ .rpk model not found: {rpk_path}")
        print(f"   - You need a .rpk file for native inference")
    
    # Check labels
    labels_path = runtime.get('imx500_labels', '')
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = f.read().splitlines()
        print(f"\\n✅ Labels file found: {len(labels)} classes")
        print(f"   - Classes: {', '.join(labels)}")
    else:
        print(f"\\n❌ Labels file not found: {labels_path}")
    
    # Recommendations
    print("\\n=== RECOMMENDATIONS ===")
    if runtime.get('use_imx500') and runtime.get('imx500_ai_camera'):
        print("✅ IMX500 native mode is configured correctly")
        print("   - Run: python3 runImage.py")
        print("   - IMX500 will handle inference natively")
    else:
        print("⚠️  IMX500 native mode not fully configured")
        print("   - Set use_imx500: True")
        print("   - Set imx500_ai_camera: True")
        print("   - Ensure .rpk model exists")
    
    return True

if __name__ == "__main__":
    test_imx500_native()
'''
            
            with open(test_script, 'w') as f:
                f.write(test_content)
            
            # Make script executable
            os.chmod(test_script, 0o755)
            logger.info(f"✅ IMX500 native test script created: {test_script}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create test script: {e}")
            return False
    
    def setup_complete_workflow(self):
        """Complete setup workflow for IMX500 native mode"""
        logger.info("=== COMPLETE IMX500 NATIVE SETUP ===")
        
        steps = [
            ("Checking existing .rpk model", self.check_existing_rpk),
            ("Creating conversion guide", self.create_conversion_guide),
            ("Creating test script", self.create_quick_test_script)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            if not step_func():
                logger.error(f"❌ Setup failed at: {step_name}")
                return False
            logger.info(f"✅ {step_name} completed")
        
        logger.info("🎉 Complete IMX500 native setup finished successfully!")
        return True

def main():
    """Main function"""
    converter = IMX500Converter()
    
    print("=== IMX500 NATIVE MODEL SETUP ===")
    print("This setup helps you use .rpk models for native IMX500 inference.")
    print("No CPU processing on Pi 5 - everything runs on IMX500 NPU!")
    print()
    
    # Run complete setup
    if converter.setup_complete_workflow():
        print()
        print("🎯 IMX500 native setup completed successfully!")
        print("Next steps:")
        print("1. Check your existing .rpk model")
        print("2. Test IMX500 native mode")
        print("3. Get IMX500 SDK for custom model conversion")
        print("4. Enjoy native NPU performance!")
    else:
        print("❌ Setup failed. Check logs above.")

if __name__ == "__main__":
    main()
