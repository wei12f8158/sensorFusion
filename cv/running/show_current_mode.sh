#!/bin/bash
# Show current camera mode

echo "📷 Current Camera Mode:"
echo ""

# Check current mode
if grep -q "imx500_ai_camera: True" ../../config.yaml; then
    echo "🟢 AI Camera Mode (Inference on IMX500)"
    echo "   - Uses .rpk model file"
    echo "   - Faster inference"
    echo "   - Lower power usage"
    echo ""
    echo "📝 To switch: ./switch_to_camera_source.sh"
elif grep -q "imx500_ai_camera: False" ../../config.yaml; then
    echo "🔵 Camera Source Mode (Inference on Pi 5)"
    echo "   - Uses .pt model file"
    echo "   - More flexible"
    echo "   - Can modify inference pipeline"
    echo ""
    echo "📝 To switch: ./switch_to_ai_camera.sh"
else
    echo "❓ Unknown mode - check config.yaml"
fi

echo ""
echo "🔧 Current settings:"
grep -A 5 "imx500_ai_camera" ../../config.yaml 