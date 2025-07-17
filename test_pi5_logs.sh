#!/bin/bash

# Test Pi 5 Log Pushing Setup

echo "=== Testing Pi 5 Log Pushing Setup ==="
echo ""

echo "1. Testing SSH connection to Pi 5..."
if ping -c 1 10.0.0.71 > /dev/null 2>&1; then
    echo "✅ Pi 5 is reachable"
else
    echo "❌ Pi 5 not reachable"
    exit 1
fi

echo ""
echo "2. Testing SSH key on Pi 5..."
ssh wei@10.0.0.71 "ls -la ~/.ssh/id_ed25519*" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ SSH key exists on Pi 5"
else
    echo "❌ SSH key not found on Pi 5"
fi

echo ""
echo "3. Testing GitHub SSH connection from Pi 5..."
ssh wei@10.0.0.71 "ssh -T git@github.com" 2>&1 | head -1
if [ $? -eq 1 ]; then
    echo "⚠️  GitHub connection needs SSH key setup"
    echo "   Follow the guide in PI5_GITHUB_SETUP.md"
else
    echo "✅ GitHub connection working"
fi

echo ""
echo "4. Testing log push script on Pi 5..."
ssh wei@10.0.0.71 "ls -la ~/push_logs.sh" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Log push script exists"
else
    echo "❌ Log push script not found"
fi

echo ""
echo "5. Testing sensorFusion repo on Pi 5..."
ssh wei@10.0.0.71 "cd ~/sensorFusion && git remote -v" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Git repo configured"
else
    echo "❌ Git repo not configured"
fi

echo ""
echo "=== Next Steps ==="
echo ""
echo "If any tests failed, follow the guide in PI5_GITHUB_SETUP.md"
echo ""
echo "To test log pushing manually:"
echo "ssh wei@10.0.0.71"
echo "~/push_logs.sh"
echo ""
echo "To view logs on GitHub:"
echo "https://github.com/wei12f8158/sensorFusion/tree/pi5-logs" 