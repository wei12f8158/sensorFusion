# Pi 5 GitHub SSH Setup Guide

## Step 1: Add SSH Key to GitHub

1. **Copy this SSH public key**:
   ```
   ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFK3G1nWt7zTwdsTzpiEix59ZnNJ/4YMgev1o8mphXB7 pi5-logs@raspberrypi
   ```

2. **Go to GitHub**:
   - Visit: https://github.com/settings/keys
   - Click "New SSH key"
   - Title: `Pi 5 Logs`
   - Key type: `Authentication Key`
   - Paste the key above
   - Click "Add SSH key"

## Step 2: Test Connection

SSH into your Pi 5 and test the GitHub connection:

```bash
ssh wei@10.0.0.71
ssh -T git@github.com
```

You should see: `Hi username! You've successfully authenticated...`

## Step 3: Configure Git on Pi 5

```bash
ssh wei@10.0.0.71
git config --global user.name "Pi 5 Logs"
git config --global user.email "pi5-logs@raspberrypi.local"
```

## Step 4: Create Log Push Script

The script `~/push_logs.sh` has been created on your Pi 5. It will:

- Copy current log file to timestamped version
- Push to GitHub branch `pi5-logs`
- Create organized log history

## Step 5: Test Manual Log Push

```bash
ssh wei@10.0.0.71
~/push_logs.sh
```

## Step 6: Set Up Automatic Logging (Optional)

To push logs every hour:

```bash
ssh wei@10.0.0.71
crontab -e
```

Add this line:
```
0 * * * * ~/push_logs.sh >> ~/push_logs.log 2>&1
```

## Step 7: View Logs on GitHub

Logs will be available at:
https://github.com/wei12f8158/sensorFusion/tree/pi5-logs

## Troubleshooting

### If SSH connection fails:
```bash
ssh wei@10.0.0.71
ssh-keyscan -H github.com >> ~/.ssh/known_hosts
ssh -T git@github.com
```

### If push fails:
```bash
ssh wei@10.0.0.71
cd ~/sensorFusion
git remote -v
# Should show your GitHub repo
```

### Manual log push:
```bash
ssh wei@10.0.0.71
cd ~/sensorFusion
git checkout -b pi5-logs
cp cv/running/log_2.txt logs/log_$(date +%Y%m%d_%H%M%S).txt
git add logs/
git commit -m "Add log from Pi 5"
git push origin pi5-logs
```

## Benefits

✅ **Automatic backup** of logs  
✅ **Version control** for log history  
✅ **Remote access** to logs from anywhere  
✅ **Organized** by timestamp  
✅ **Separate branch** to keep logs organized 