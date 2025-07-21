#!/bin/bash
# Quick fix for object localization issues

echo "🔧 Quick Fix for Object Localization"
echo "===================================="

# Backup current config
cp ../../config.yaml ../../config.yaml.backup

echo "Applying recommended settings for better localization..."

# Increase object threshold to reduce false positives
sed -i 's/objectThreshold: 0.1/objectThreshold: 0.3/' ../../config.yaml

# Increase hand threshold slightly
sed -i 's/handThreshold: 0.1/handThreshold: 0.2/' ../../config.yaml

# Adjust NMS IOU threshold for better box selection
sed -i 's/nmsIouThreshold: 0.90/nmsIouThreshold: 0.6/' ../../config.yaml

echo "✅ Applied quick fix:"
echo "  - Object threshold: 0.1 → 0.3"
echo "  - Hand threshold: 0.1 → 0.2" 
echo "  - NMS IOU threshold: 0.90 → 0.6"
echo ""
echo "🎯 This should:"
echo "  - Reduce false positive detections"
echo "  - Improve bounding box accuracy"
echo "  - Better distinguish between objects"
echo ""
echo "🚀 Test the fix:"
echo "  python3 runImage.py"
echo ""
echo "📝 To revert:"
echo "  cp ../../config.yaml.backup ../../config.yaml" 