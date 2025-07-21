#!/bin/bash
# Aggressive fix for object localization issues

echo "🔧 Aggressive Fix for Object Localization"
echo "========================================="

# Backup current config
cp ../../config.yaml ../../config.yaml.backup

echo "Applying aggressive settings for better localization..."

# Much higher object threshold to eliminate false positives
sed -i 's/objectThreshold: [0-9.]*/objectThreshold: 0.6/' ../../config.yaml

# Higher hand threshold
sed -i 's/handThreshold: [0-9.]*/handThreshold: 0.4/' ../../config.yaml

# Lower NMS IOU threshold for stricter box selection
sed -i 's/nmsIouThreshold: [0-9.]*/nmsIouThreshold: 0.4/' ../../config.yaml

echo "✅ Applied aggressive fix:"
echo "  - Object threshold: → 0.6 (very high)"
echo "  - Hand threshold: → 0.4"
echo "  - NMS IOU threshold: → 0.4 (stricter)"
echo ""
echo "🎯 This should:"
echo "  - Eliminate most false positive detections"
echo "  - Only show high-confidence detections"
echo "  - Force better object distinction"
echo ""
echo "⚠️  Warning: This may reduce total detections"
echo "   If too aggressive, use adjust_detection_parameters.sh"
echo ""
echo "🚀 Test the fix:"
echo "  python3 runImage.py" 