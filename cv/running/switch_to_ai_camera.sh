#!/bin/bash
# Switch to AI Camera Mode (inference on IMX500)

echo "🔄 Switching to AI Camera Mode..."
echo "   - Inference runs on IMX500 processor"
echo "   - Uses .rpk model file"
echo "   - Faster inference, lower power usage"

# Backup current config
cp ../../config.yaml ../../config.yaml.backup

# Update config for AI camera mode
sed -i 's/imx500_ai_camera: False/imx500_ai_camera: True/' ../../config.yaml

echo "✅ Switched to AI Camera Mode!"
echo "   Run: python3 runImage.py"
echo ""
echo "📝 To switch back: ./switch_to_camera_source.sh" 