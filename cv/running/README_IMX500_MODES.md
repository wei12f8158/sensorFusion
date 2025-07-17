# IMX500 Dual-Mode Configuration

This document explains how to use the IMX500 sensor in two different modes for your sensor fusion project.

## Overview

The system now supports two IMX500 modes:

1. **Camera Source Mode**: IMX500 acts as a high-quality camera, inference runs on Raspberry Pi 5
2. **AI Camera Mode**: IMX500 handles both camera capture and inference (production mode)

## Configuration

Edit `config.yaml` to switch between modes:

```yaml
runTime:
    # IMX500 settings   
    use_imx500: False  # Set to True to use IMX500
    imx500_ai_camera: False  # Set to True for AI camera mode, False for camera source mode
```

## Mode 1: Camera Source Mode (Testing)

**Use case**: Test your model performance on Raspberry Pi 5 before deploying to IMX500

**Configuration**:
```yaml
runTime:
    use_imx500: True
    imx500_ai_camera: False
```

**What happens**:
- IMX500 captures high-quality images
- Images are sent to Raspberry Pi 5 for inference
- Standard YOLO model runs on Pi 5 CPU/GPU
- Good for testing and debugging

**Benefits**:
- Easy to test different models
- Familiar inference pipeline
- Better debugging capabilities
- Can use any YOLO format supported by Pi 5

## Mode 2: AI Camera Mode (Production)

**Use case**: Optimized production deployment with inference on IMX500

**Configuration**:
```yaml
runTime:
    use_imx500: True
    imx500_ai_camera: True
```

**What happens**:
- IMX500 captures images and runs inference
- Optimized ONNX model runs on IMX500's AI processor
- Results are processed directly on the sensor
- Maximum performance and efficiency

**Benefits**:
- Highest performance
- Lower power consumption
- Real-time inference
- Optimized for production

## Mode 3: Standard Camera Mode

**Use case**: Use regular camera (not IMX500)

**Configuration**:
```yaml
runTime:
    use_imx500: False
    imx500_ai_camera: False  # Ignored when use_imx500 is False
```

## Quick Start Guide

### Step 1: Test on Pi 5 (Camera Source Mode)

1. Set configuration for testing:
   ```yaml
   runTime:
       use_imx500: True
       imx500_ai_camera: False
   ```

2. Run the system:
   ```bash
   cd cv/running
   python3 runImage.py
   ```

3. Test your model performance and tune parameters

### Step 2: Deploy to IMX500 (AI Camera Mode)

1. Once satisfied with results, switch to AI camera mode:
   ```yaml
   runTime:
       use_imx500: True
       imx500_ai_camera: True
   ```

2. Ensure your model is converted to ONNX format and placed in the correct path:
   ```yaml
   runTime:
       imx500_model: "../../IMX500/final_output/network.rpk"
       imx500_labels: "../../IMX500/yolo8_19_UL-New_day2_fullLabTopCam_60epochs_imx_model/labels.txt"
   ```

3. Run the system:
   ```bash
   cd cv/running
   python3 runImage.py
   ```

## Model Conversion

For AI Camera Mode, you need to convert your YOLO model to ONNX format:

1. Export your trained model to ONNX:
   ```python
   # In your training script
   model.export(format='onnx', dynamic=True, simplify=True)
   ```

2. Convert ONNX to IMX500 format using the IMX500 tools

3. Update the model path in `config.yaml`

## Troubleshooting

### Camera Source Mode Issues

- **No image capture**: Check IMX500 connection and drivers
- **Slow inference**: Consider reducing image resolution or model complexity
- **Memory issues**: Reduce buffer count in camera configuration

### AI Camera Mode Issues

- **Model not found**: Verify ONNX model path and format
- **No detections**: Check model compatibility and labels file
- **Performance issues**: Ensure model is optimized for IMX500

## Performance Comparison

| Mode | Inference Location | Performance | Power Usage | Debugging |
|------|-------------------|-------------|-------------|-----------|
| Camera Source | Pi 5 CPU/GPU | Medium | Medium | Easy |
| AI Camera | IMX500 | High | Low | Harder |
| Standard | Pi 5 CPU/GPU | Low | High | Easy |

## Testing Your Configuration

Run the test script to verify your configuration:

```bash
cd cv/running
python3 test_imx500_modes.py
```

This will show you the current mode and provide configuration instructions.

## File Structure

```
cv/running/
├── runImage.py              # Main execution script
├── test_imx500_modes.py     # Configuration test script
├── README_IMX500_MODES.md   # This documentation
└── config.yaml              # Configuration file
```

## Notes

- Always test in Camera Source Mode first
- Switch to AI Camera Mode only after thorough testing
- Keep backup configurations for different modes
- Monitor system resources when switching modes 