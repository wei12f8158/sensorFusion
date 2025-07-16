#!/bin/bash

echo "=== Setting up SSH Authentication for GitHub ==="
echo ""

# Check if SSH key already exists
if [ -f ~/.ssh/id_ed25519 ]; then
    echo "SSH key already exists at ~/.ssh/id_ed25519"
    read -p "Do you want to generate a new one? (y/n): " generate_new
    if [ "$generate_new" != "y" ]; then
        echo "Using existing SSH key"
    else
        echo "Backing up existing key..."
        mv ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.backup
        mv ~/.ssh/id_ed25519.pub ~/.ssh/id_ed25519.pub.backup
    fi
fi

# Generate SSH key if needed
if [ ! -f ~/.ssh/id_ed25519 ] || [ "$generate_new" = "y" ]; then
    echo ""
    echo "Step 1: Generating SSH key..."
    read -p "Enter your email address: " email_address
    
    ssh-keygen -t ed25519 -C "$email_address" -f ~/.ssh/id_ed25519 -N ""
    echo "SSH key generated successfully!"
fi

# Start SSH agent and add key
echo ""
echo "Step 2: Adding SSH key to agent..."
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Display public key
echo ""
echo "Step 3: Your public SSH key (copy this to GitHub):"
echo "=================================================="
cat ~/.ssh/id_ed25519.pub
echo "=================================================="
echo ""

echo "Step 4: Add this key to GitHub:"
echo "1. Go to GitHub.com → Settings → SSH and GPG keys"
echo "2. Click 'New SSH key'"
echo "3. Title: 'Raspberry Pi 5'"
echo "4. Paste the key above"
echo "5. Click 'Add SSH key'"
echo ""

read -p "Press Enter after adding the key to GitHub..."

# Test SSH connection
echo ""
echo "Step 5: Testing SSH connection..."
ssh -T git@github.com

# Change git remote to SSH
echo ""
echo "Step 6: Changing git remote to SSH..."
current_remote=$(git remote get-url origin)
if [[ $current_remote == https://* ]]; then
    new_remote=$(echo $current_remote | sed 's|https://github.com/|git@github.com:|')
    git remote set-url origin "$new_remote"
    echo "Changed remote from HTTPS to SSH"
else
    echo "Remote is already using SSH"
fi

echo ""
echo "Step 7: Testing git operations..."
echo "Testing git pull..."
git pull

echo ""
echo "=== SSH Setup Complete! ==="
echo "You can now use git push/pull without entering credentials."
echo ""
echo "To test:"
echo "  git push"
echo "  git pull" 