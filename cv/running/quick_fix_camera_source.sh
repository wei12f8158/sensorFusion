#!/bin/bash

# Quick fix script for Camera Source Mode issues

echo "Quick Fix for Camera Source Mode"
echo "================================"

cd /home/wei/sensorFusion/cv/running

# Backup current config
cp ../../config.yaml ../../config.yaml.backup

echo "1. Lowering confidence thresholds..."
# Lower thresholds to very low values for testing
sed -i 's/handThreshold: 0.1/handThreshold: 0.01/' ../../config.yaml
sed -i 's/objectThreshold: 0.1/objectThreshold: 0.01/' ../../config.yaml

echo "2. Running comprehensive debug..."
python3 debug_camera_source.py

echo ""
echo "3. If still not working, trying alternative approaches..."

# Try with different model settings
echo "   - Testing with different model parameters..."
python3 -c "
import sys
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser
config = ConfigParser('../../config.yaml')
configs = config.get_config()

# Print current settings
print('Current settings:')
print(f'Model path: {configs[\"training\"][\"weightsFile_rpi\"]}')
print(f'Hand threshold: {configs[\"runTime\"][\"distSettings\"][\"handThreshold\"]}')
print(f'Object threshold: {configs[\"runTime\"][\"distSettings\"][\"objectThreshold\"]}')
print(f'Use IMX500: {configs[\"runTime\"].get(\"use_imx500\", False)}')
print(f'AI Camera: {configs[\"runTime\"].get(\"imx500_ai_camera\", False)}')
"

echo ""
echo "4. Testing with ultra-low thresholds..."
# Create ultra-low threshold config
cat > ../../config_ultra_low.yaml << 'EOF'
debugs:
    debug: True 
    showInfResults: True
    dispResults: True
    tpuThreadTimeout: 0.5
    runInfer: True
    saveImages: False
    logFile: 'log_ultra_low.txt'
    videoFile: ""

training:
    dataSetDir: "../datasets"
    dataSet: "day2_partII_2138Images/data.yaml"
    modelsDir: "../models"
    weightsDir: ""
    modelFile: "yolo11n.yaml"
    weightsForTransfer: "yolo11n.pt"
    weightsFile: "IMX500/yolo8_19_UL-New_day2_fullLabTopCam_60epochs.pt"
    weightsFile_rpi: "/home/wei/YOLO/yolo8_19_UL-New_day2_fullLabTopCam_60epochs.pt"
    weightsFile_tpu: "IMX500/yolo8_19_UL-New_day2_fullLabTopCam_60epochs.pt"
    transLearn: True
    freezeLayer: 11
    imageSize: [480, 640]
    epochs: 30

runTime:
    use_imx500: True
    imx500_ai_camera: False  # Camera Source Mode
    imx500_model: "../../IMX500/final_output/network.rpk"
    imx500_labels: "../../IMX500/yolo8_19_UL-New_day2_fullLabTopCam_60epochs_imx_model/labels.txt"
    imx500_threshold: 0.001  # Ultra low
    imx500_iou: 0.5
    imx500_max_detections: 10

    imageDir: "../images_capture/day1"
    imgSrc: "camera"
    nCameras: 1
    focus: 15
    camId: 0
    camId_2: "rtsp://192.168.1.254:554/"
    camRateHz: 5

    rpi_use_gpu: False
    rpi_num_threads: 4

    distSettings:
        classMap: [0, 1, 0, 3, 4, 5, 6, 7, 8]
        imagePxlPer_mm: .51
        handThreshold: 0.001  # Ultra low
        objectThreshold: 0.001  # Ultra low
        nmsIouThreshold: 0.90
        handClass: 4

    displaySettings:
        fullScreen: False
        handLineTh: 2
        objLineTh: 2
        distLineTh: 2
        runCamOnce: False

timeSync:
    gpio_chip: 4
    gpio_pin : 13

servos:
    i2c:
        port: "/dev/i2c-1"
        device: 0x40
        clock_MHz: 26.4

    servos: 
        pwm_Hz: 50
        leavRunning: False

comms:
    port: "None"
    speed: 115200
    dataBits: 8
    stobBits: 1
    parity: 'N'
    id: "CV"
EOF

echo "5. Testing with ultra-low thresholds..."
# Test with ultra-low config
cp ../../config_ultra_low.yaml ../../config.yaml
python3 debug_camera_source.py

echo ""
echo "6. Restoring original config..."
cp ../../config.yaml.backup ../../config.yaml

echo ""
echo "Quick fix completed!"
echo "Check the debug output above for issues."
echo ""
echo "If still not working, try:"
echo "1. Check if model file exists: /home/wei/YOLO/yolo8_19_UL-New_day2_fullLabTopCam_60epochs.pt"
echo "2. Check camera permissions"
echo "3. Try running with sudo (if needed)"
echo "4. Check system memory usage" 