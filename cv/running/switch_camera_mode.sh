#!/bin/bash

# Script to switch between IMX500 camera modes

echo "IMX500 Camera Mode Switcher"
echo "=========================="
echo "1. AI Camera Mode (inference on IMX500)"
echo "2. Camera Source Mode (inference on Pi 5)"
echo "3. Test AI Camera Mode"
echo "4. Test Camera Source Mode"
echo "5. Show current mode"
echo "6. Exit"
echo ""

read -p "Choose an option (1-6): " choice

case $choice in
    1)
        echo "Switching to AI Camera Mode..."
        # Backup current config
        cp ../../config.yaml ../../config.yaml.backup
        
        # Update config for AI camera mode
        sed -i 's/imx500_ai_camera: False/imx500_ai_camera: True/' ../../config.yaml
        sed -i 's/imx500_ai_camera: false/imx500_ai_camera: true/' ../../config.yaml
        
        echo "✓ Switched to AI Camera Mode"
        echo "The IMX500 will now handle inference internally."
        ;;
    2)
        echo "Switching to Camera Source Mode..."
        # Backup current config
        cp ../../config.yaml ../../config.yaml.backup
        
        # Update config for camera source mode
        sed -i 's/imx500_ai_camera: True/imx500_ai_camera: False/' ../../config.yaml
        sed -i 's/imx500_ai_camera: true/imx500_ai_camera: false/' ../../config.yaml
        
        echo "✓ Switched to Camera Source Mode"
        echo "The Pi 5 will handle inference using the YOLO model."
        ;;
    3)
        echo "Testing AI Camera Mode..."
        python3 test_imx500_ai_camera.py
        ;;
    4)
        echo "Testing Camera Source Mode..."
        python3 debug_detection.py
        ;;
    5)
        echo "Current camera mode:"
        if grep -q "imx500_ai_camera: True" ../../config.yaml; then
            echo "✓ AI Camera Mode (inference on IMX500)"
        elif grep -q "imx500_ai_camera: False" ../../config.yaml; then
            echo "✓ Camera Source Mode (inference on Pi 5)"
        else
            echo "? Unknown mode"
        fi
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option. Please choose 1-6."
        exit 1
        ;;
esac

echo ""
echo "To run the main application:"
echo "  python3 runImage.py"
echo ""
echo "To restore previous config:"
echo "  cp ../../config.yaml.backup ../../config.yaml" 