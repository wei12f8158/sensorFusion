#!/bin/bash
# Fix Ultralytics installation on Raspberry Pi 5

echo "=== FIXING ULTRALYTICS INSTALLATION ON PI 5 ==="

# Check Python version
echo "Python version:"
python3 --version

# Check pip version
echo "Pip version:"
pip3 --version

# Try different installation methods
echo "Trying installation method 1: pip3 install ultralytics"
if pip3 install ultralytics; then
    echo "✅ Method 1 successful!"
    exit 0
fi

echo "Method 1 failed. Trying method 2: pip3 install --user ultralytics"
if pip3 install --user ultralytics; then
    echo "✅ Method 2 successful!"
    exit 0
fi

echo "Method 2 failed. Trying method 3: upgrade pip first"
if pip3 install --upgrade pip; then
    echo "Pip upgraded. Trying ultralytics installation again..."
    if pip3 install ultralytics; then
        echo "✅ Method 3 successful!"
        exit 0
    fi
fi

echo "Method 3 failed. Trying method 4: create virtual environment"
if python3 -m venv yolo_env; then
    echo "Virtual environment created. Activating..."
    source yolo_env/bin/activate
    
    if pip install ultralytics; then
        echo "✅ Method 4 successful in virtual environment!"
        echo "To use: source yolo_env/bin/activate"
        exit 0
    fi
fi

echo "Method 4 failed. Trying method 5: install from source"
if git clone https://github.com/ultralytics/ultralytics.git; then
    cd ultralytics
    if pip3 install -e .; then
        echo "✅ Method 5 successful!"
        exit 0
    fi
fi

echo "❌ All installation methods failed."
echo "Please check your Pi 5 system and try manually."
