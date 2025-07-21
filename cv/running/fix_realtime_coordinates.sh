#!/bin/bash
# Fix real-time coordinate processing issues

echo "🔧 Fix Real-time Coordinate Processing"
echo "======================================"

# Backup current config
cp ../../config.yaml ../../config.yaml.backup

echo "The issue is in the real-time processing pipeline."
echo "Static images work correctly, but live camera has mislocalized boxes."
echo ""

echo "🚨 Most likely causes:"
echo "1. Image preprocessing differences between static and real-time"
echo "2. Coordinate system scaling issues"
echo "3. Buffer/frame processing timing"
echo "4. Camera resolution mismatch"
echo ""

echo "🔍 Diagnostic steps:"
echo "1. Run: python3 diagnose_realtime_vs_static.py"
echo "   This will capture a live frame and compare with static processing"
echo ""

echo "2. Check the generated comparison images:"
echo "   - static_vs_realtime_*.jpg (raw image comparison)"
echo "   - annotated_comparison_*.jpg (detection comparison)"
echo ""

echo "3. Look for differences in:"
echo "   - Image resolution/size"
echo "   - Bounding box coordinates"
echo "   - Detection confidence"
echo ""

echo "🎯 Quick fixes to try:"
echo ""

echo "A) If image sizes differ:"
echo "   - Check camera configuration vs expected size"
echo "   - Ensure consistent image preprocessing"
echo ""

echo "B) If coordinates are offset:"
echo "   - Check for coordinate system conversion errors"
echo "   - Verify scaling factors in real-time pipeline"
echo ""

echo "C) If preprocessing differs:"
echo "   - Ensure same normalization for static and real-time"
echo "   - Check for different image formats (RGB vs BGR)"
echo ""

echo "🚀 Next steps:"
echo "   1. Run the diagnostic: python3 diagnose_realtime_vs_static.py"
echo "   2. Compare the generated images"
echo "   3. Identify the specific difference"
echo "   4. Apply targeted fix based on findings"
echo ""

echo "📝 To revert:"
echo "   cp ../../config.yaml.backup ../../config.yaml" 