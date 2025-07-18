#!/bin/bash
# Switch to USB Camera Mode

echo "🔄 Switching to USB Camera Mode..."
echo "   - Uses USB camera (not IMX500)"
echo "   - Inference runs on Raspberry Pi 5 CPU"
echo "   - Same pipeline as Camera Source Mode"

# Backup current config
cp ../../config.yaml ../../config.yaml.backup

# Update config for USB camera mode
sed -i 's/use_imx500: True/use_imx500: False/' ../../config.yaml

echo "✅ Switched to USB Camera Mode!"
echo "   Run: python3 runImage.py"
echo ""
echo "📝 To switch back to IMX500:"
echo "   ./switch_to_ai_camera.sh      # AI Camera Mode"
echo "   ./switch_to_camera_source.sh  # Camera Source Mode"
echo ""
echo "🔧 USB Camera ID is set to 0"
echo "   To change camera ID, edit config.yaml: camId: 1" 