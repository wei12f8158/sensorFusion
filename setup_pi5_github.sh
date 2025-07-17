#!/bin/bash

# Setup Pi 5 for GitHub SSH access
# This script helps configure Pi 5 to push logs to GitHub

echo "=== Pi 5 GitHub SSH Setup ==="
echo ""

# Pi 5 SSH public key
PI5_PUBLIC_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFK3G1nWt7zTwdsTzpiEix59ZnNJ/4YMgev1o8mphXB7 pi5-logs@raspberrypi"

echo "Pi 5 SSH Public Key:"
echo "$PI5_PUBLIC_KEY"
echo ""

echo "=== Steps to Add SSH Key to GitHub ==="
echo ""
echo "1. Copy the SSH key above"
echo "2. Go to GitHub.com → Settings → SSH and GPG keys"
echo "3. Click 'New SSH key'"
echo "4. Title: 'Pi 5 Logs'"
echo "5. Key type: Authentication Key"
echo "6. Paste the key and click 'Add SSH key'"
echo ""

read -p "Have you added the SSH key to GitHub? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please add the SSH key to GitHub first, then run this script again."
    exit 1
fi

echo ""
echo "=== Setting up Pi 5 for GitHub access ==="
echo ""

# Test SSH connection from Pi 5 to GitHub
echo "Testing SSH connection from Pi 5 to GitHub..."
ssh wei@10.0.0.71 "ssh -T git@github.com"

echo ""
echo "=== Setting up Git configuration on Pi 5 ==="
echo ""

# Configure Git on Pi 5
ssh wei@10.0.0.71 "git config --global user.name 'Pi 5 Logs'"
ssh wei@10.0.0.71 "git config --global user.email 'pi5-logs@raspberrypi.local'"

echo "✅ Git configured on Pi 5"

echo ""
echo "=== Creating log push script on Pi 5 ==="
echo ""

# Create a script on Pi 5 to push logs
LOG_SCRIPT='#!/bin/bash

# Pi 5 Log Push Script
# This script pushes logs to GitHub

LOG_DIR="$HOME/sensorFusion/cv/running/logs"
REPO_DIR="$HOME/sensorFusion"
BRANCH="pi5-logs"

# Create logs directory if it doesn'\''t exist
mkdir -p "$LOG_DIR"

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/log_$TIMESTAMP.txt"

# Copy current log to timestamped file
if [ -f "$REPO_DIR/cv/running/log_2.txt" ]; then
    cp "$REPO_DIR/cv/running/log_2.txt" "$LOG_FILE"
    echo "Log copied to $LOG_FILE"
else
    echo "No log file found at $REPO_DIR/cv/running/log_2.txt"
    exit 1
fi

# Navigate to repo directory
cd "$REPO_DIR"

# Create logs branch if it doesn'\''t exist
if ! git show-ref --verify --quiet refs/heads/$BRANCH; then
    git checkout -b $BRANCH
else
    git checkout $BRANCH
fi

# Add log file
git add "$LOG_FILE"

# Commit with timestamp
git commit -m "Add log from Pi 5 - $TIMESTAMP"

# Push to GitHub
git push origin $BRANCH

echo "✅ Log pushed to GitHub branch: $BRANCH"
echo "Log file: $LOG_FILE"
'

# Write the script to Pi 5
ssh wei@10.0.0.71 "cat > ~/push_logs.sh << 'EOF'
$LOG_SCRIPT
EOF"

# Make the script executable
ssh wei@10.0.0.71 "chmod +x ~/push_logs.sh"

echo "✅ Log push script created on Pi 5"

echo ""
echo "=== Setting up automatic log pushing ==="
echo ""

# Create a cron job to push logs every hour
CRON_JOB="0 * * * * $HOME/push_logs.sh >> $HOME/push_logs.log 2>&1"

echo "To set up automatic log pushing every hour, run this on Pi 5:"
echo "crontab -e"
echo "Add this line:"
echo "$CRON_JOB"
echo ""

echo "=== Manual Usage ==="
echo ""
echo "To manually push logs from Pi 5:"
echo "ssh wei@10.0.0.71"
echo "~/push_logs.sh"
echo ""

echo "=== View Logs on GitHub ==="
echo ""
echo "Logs will be pushed to: https://github.com/wei12f8158/sensorFusion/tree/pi5-logs"
echo ""

echo "✅ Pi 5 GitHub setup complete!" 