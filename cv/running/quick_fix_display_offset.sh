#!/bin/bash
# Quick fix for display offset issues

echo "🔧 Quick Fix for Display Offset"
echo "==============================="

# Backup current config
cp ../../config.yaml ../../config.yaml.backup

echo "The issue might be a coordinate system offset."
echo "Let's check and fix potential display issues:"
echo ""

# Check if there are any coordinate scaling issues
echo "1. Checking coordinate system..."
echo "   - Model outputs coordinates in image space"
echo "   - Display should use same coordinate system"
echo ""

echo "2. Potential fixes:"
echo "   a) Check if model output needs scaling"
echo "   b) Verify coordinate conversion in distance.py"
echo "   c) Ensure display uses correct coordinates"
echo ""

echo "3. Quick test:"
echo "   Run this command to test direct model output:"
echo "   python3 fix_bounding_box_position.py"
echo ""

echo "4. If the direct model output is correct but display is wrong:"
echo "   The issue is in the coordinate conversion pipeline"
echo ""

echo "5. If both are wrong:"
echo "   The issue is with the model itself"
echo ""

echo "🚀 Next steps:"
echo "   1. Run: python3 fix_bounding_box_position.py"
echo "   2. Compare fixed_bounding_boxes.jpg with debug_annotated_20250721_153537.jpg"
echo "   3. If they match: issue is in display pipeline"
echo "   4. If they don't match: issue is with model"
echo ""
echo "📝 To revert:"
echo "   cp ../../config.yaml.backup ../../config.yaml" 