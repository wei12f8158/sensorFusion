#!/bin/bash
# Switch to pre-trained YOLO model for better detection

echo "🔄 Switching to Pre-trained YOLO Model"
echo "====================================="

# Backup current config
cp ../../config.yaml ../../config.yaml.backup

echo "Current model: $(grep 'weightsFile_rpi:' ../../config.yaml | awk '{print $2}')"
echo ""

echo "Available pre-trained models:"
echo "1. YOLOv8n (nano) - Fast, good for real-time"
echo "2. YOLOv8s (small) - Balanced speed/accuracy"
echo "3. YOLOv8m (medium) - Higher accuracy"
echo "4. YOLOv8l (large) - Best accuracy, slower"
echo "5. Keep current model"
echo ""

read -p "Choose a model (1-5): " choice

case $choice in
    1)
        echo "Switching to YOLOv8n..."
        sed -i 's|weightsFile_rpi: ".*"|weightsFile_rpi: "yolov8n.pt"|' ../../config.yaml
        echo "✓ Switched to YOLOv8n (nano)"
        ;;
    2)
        echo "Switching to YOLOv8s..."
        sed -i 's|weightsFile_rpi: ".*"|weightsFile_rpi: "yolov8s.pt"|' ../../config.yaml
        echo "✓ Switched to YOLOv8s (small)"
        ;;
    3)
        echo "Switching to YOLOv8m..."
        sed -i 's|weightsFile_rpi: ".*"|weightsFile_rpi: "yolov8m.pt"|' ../../config.yaml
        echo "✓ Switched to YOLOv8m (medium)"
        ;;
    4)
        echo "Switching to YOLOv8l..."
        sed -i 's|weightsFile_rpi: ".*"|weightsFile_rpi: "yolov8l.pt"|' ../../config.yaml
        echo "✓ Switched to YOLOv8l (large)"
        ;;
    5)
        echo "Keeping current model"
        ;;
    *)
        echo "Invalid choice. Keeping current model."
        ;;
esac

echo ""
echo "🎯 Pre-trained models:"
echo "  - Trained on COCO dataset (80 classes)"
echo "  - Better generalization"
echo "  - More robust to different conditions"
echo ""
echo "⚠️  Note: Pre-trained models detect different classes"
echo "   You may need to adjust class mapping"
echo ""
echo "🚀 Test the new model:"
echo "  python3 runImage.py" 