#!/bin/bash
# Adjust detection parameters for better object localization

echo "🔧 Detection Parameter Adjuster"
echo "=============================="
echo "Current settings:"
echo "  Hand threshold: $(grep 'handThreshold:' ../../config.yaml | awk '{print $2}')"
echo "  Object threshold: $(grep 'objectThreshold:' ../../config.yaml | awk '{print $2}')"
echo "  NMS IOU threshold: $(grep 'nmsIouThreshold:' ../../config.yaml | awk '{print $2}')"
echo ""

echo "Options:"
echo "1. Increase object threshold (reduce false positives)"
echo "2. Decrease object threshold (detect more objects)"
echo "3. Adjust NMS IOU threshold (reduce overlapping boxes)"
echo "4. Reset to default values"
echo "5. Show current settings"
echo "6. Exit"
echo ""

read -p "Choose an option (1-6): " choice

case $choice in
    1)
        echo "Increasing object threshold..."
        current_thresh=$(grep 'objectThreshold:' ../../config.yaml | awk '{print $2}')
        new_thresh=$(echo "$current_thresh + 0.1" | bc -l)
        sed -i "s/objectThreshold: $current_thresh/objectThreshold: $new_thresh/" ../../config.yaml
        echo "✓ Object threshold increased to $new_thresh"
        ;;
    2)
        echo "Decreasing object threshold..."
        current_thresh=$(grep 'objectThreshold:' ../../config.yaml | awk '{print $2}')
        new_thresh=$(echo "$current_thresh - 0.05" | bc -l)
        if (( $(echo "$new_thresh > 0" | bc -l) )); then
            sed -i "s/objectThreshold: $current_thresh/objectThreshold: $new_thresh/" ../../config.yaml
            echo "✓ Object threshold decreased to $new_thresh"
        else
            echo "❌ Cannot decrease below 0"
        fi
        ;;
    3)
        echo "Adjusting NMS IOU threshold..."
        current_iou=$(grep 'nmsIouThreshold:' ../../config.yaml | awk '{print $2}')
        echo "Current IOU threshold: $current_iou"
        read -p "Enter new IOU threshold (0.1-1.0): " new_iou
        if (( $(echo "$new_iou >= 0.1 && $new_iou <= 1.0" | bc -l) )); then
            sed -i "s/nmsIouThreshold: $current_iou/nmsIouThreshold: $new_iou/" ../../config.yaml
            echo "✓ NMS IOU threshold set to $new_iou"
        else
            echo "❌ Invalid IOU threshold (must be 0.1-1.0)"
        fi
        ;;
    4)
        echo "Resetting to default values..."
        sed -i 's/objectThreshold: [0-9.]*/objectThreshold: 0.25/' ../../config.yaml
        sed -i 's/handThreshold: [0-9.]*/handThreshold: 0.25/' ../../config.yaml
        sed -i 's/nmsIouThreshold: [0-9.]*/nmsIouThreshold: 0.45/' ../../config.yaml
        echo "✓ Reset to default values"
        ;;
    5)
        echo "Current settings:"
        echo "  Hand threshold: $(grep 'handThreshold:' ../../config.yaml | awk '{print $2}')"
        echo "  Object threshold: $(grep 'objectThreshold:' ../../config.yaml | awk '{print $2}')"
        echo "  NMS IOU threshold: $(grep 'nmsIouThreshold:' ../../config.yaml | awk '{print $2}')"
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option. Please choose 1-6."
        exit 1
        ;;
esac

echo ""
echo "💡 Tips for better localization:"
echo "  - Higher object threshold = fewer false positives"
echo "  - Lower IOU threshold = less overlapping boxes"
echo "  - Try increasing object threshold to 0.3-0.5"
echo ""
echo "To test changes:"
echo "  python3 runImage.py" 