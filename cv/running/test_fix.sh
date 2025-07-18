#!/bin/bash

# Test script to verify the IndexError fix
echo "Testing the IndexError fix..."

# Change to the running directory
cd /home/wei/sensorFusion/cv/running

# Run the application for a short time to test the fix
echo "Running runImage.py for 10 seconds to test the fix..."
timeout 10s python3 runImage.py

echo "Test completed. Check the log file for any remaining errors." 