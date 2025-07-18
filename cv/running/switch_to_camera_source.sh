#!/bin/bash
# Switch to Camera Source Mode (inference on Pi 5)

echo "🔄 Switching to Camera Source Mode..."
echo "   - Inference runs on Raspberry Pi 5 CPU"
echo "   - Uses .pt model file"
echo "   - More flexible, can modify inference pipeline"

# Backup current config
cp ../../config.yaml ../../config.yaml.backup

# Update config for camera source mode
sed -i 's/imx500_ai_camera: True/imx500_ai_camera: False/' ../../config.yaml

echo "✅ Switched to Camera Source Mode!"
echo "   Run: python3 runImage.py"
echo ""
echo "📝 To switch back: ./switch_to_ai_camera.sh" 