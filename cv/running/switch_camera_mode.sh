#!/bin/bash

# Script to switch between camera modes

echo "Camera Mode Switcher"
echo "==================="
echo "1. AI Camera Mode (IMX500 inference)"
echo "2. Camera Source Mode (IMX500 + Pi 5 inference)"
echo "3. USB Camera Mode (USB camera + Pi 5 inference)"
echo "4. Test AI Camera Mode"
echo "5. Test Camera Source Mode"
echo "6. Test USB Camera Mode"
echo "7. Show current mode"
echo "8. Exit"
echo ""

read -p "Choose an option (1-8): " choice

case $choice in
    1)
        echo "Switching to AI Camera Mode..."
        # Backup current config
        cp ../../config.yaml ../../config.yaml.backup
        
        # Update config for AI camera mode
        sed -i 's/use_imx500: False/use_imx500: True/' ../../config.yaml
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
        sed -i 's/use_imx500: False/use_imx500: True/' ../../config.yaml
        sed -i 's/imx500_ai_camera: True/imx500_ai_camera: False/' ../../config.yaml
        sed -i 's/imx500_ai_camera: true/imx500_ai_camera: false/' ../../config.yaml
        
        echo "✓ Switched to Camera Source Mode"
        echo "The Pi 5 will handle inference using the YOLO model."
        ;;
    3)
        echo "Switching to USB Camera Mode..."
        # Backup current config
        cp ../../config.yaml ../../config.yaml.backup
        
        # Update config for USB camera mode
        sed -i 's/use_imx500: True/use_imx500: False/' ../../config.yaml
        
        echo "✓ Switched to USB Camera Mode"
        echo "The Pi 5 will handle inference using USB camera input."
        ;;
    4)
        echo "Testing AI Camera Mode..."
        python3 test_imx500_ai_camera.py
        ;;
    5)
        echo "Testing Camera Source Mode..."
        python3 debug_detection.py
        ;;
    6)
        echo "Testing USB Camera Mode..."
        echo "Checking USB camera availability..."
        ls /dev/video* 2>/dev/null || echo "No USB cameras found"
        echo "Available video devices:"
        v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-ctl not available"
        ;;
    7)
        echo "Current camera mode:"
        if grep -q "use_imx500: False" ../../config.yaml; then
            echo "✓ USB Camera Mode (USB camera + Pi 5 inference)"
        elif grep -q "imx500_ai_camera: True" ../../config.yaml; then
            echo "✓ AI Camera Mode (IMX500 inference)"
        elif grep -q "imx500_ai_camera: False" ../../config.yaml; then
            echo "✓ Camera Source Mode (IMX500 + Pi 5 inference)"
        else
            echo "? Unknown mode"
        fi
        ;;
    8)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option. Please choose 1-8."
        exit 1
        ;;
esac

echo ""
echo "To run the main application:"
echo "  python3 runImage.py"
echo ""
echo "To restore previous config:"
echo "  cp ../../config.yaml.backup ../../config.yaml" 