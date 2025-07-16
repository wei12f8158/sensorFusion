#!/bin/bash

echo "=== GitHub Personal Access Token Setup ==="
echo ""
echo "Step 1: Create a Personal Access Token on GitHub.com"
echo "1. Go to GitHub.com and log in"
echo "2. Click your profile picture → Settings"
echo "3. Scroll down to 'Developer settings' (bottom left)"
echo "4. Click 'Personal access tokens' → 'Tokens (classic)'"
echo "5. Click 'Generate new token' → 'Generate new token (classic)'"
echo "6. Give it a name like 'Raspberry Pi 5'"
echo "7. Select scopes: check 'repo' (full control of private repositories)"
echo "8. Click 'Generate token'"
echo "9. COPY THE TOKEN (you won't see it again!)"
echo ""
echo "Step 2: Configure Git on this Raspberry Pi"
echo ""

# Configure git user
read -p "Enter your GitHub username: " github_username
read -p "Enter your GitHub email: " github_email

git config --global user.name "$github_username"
git config --global user.email "$github_email"

echo ""
echo "Git configured with:"
echo "Username: $github_username"
echo "Email: $github_email"
echo ""
echo "Step 3: Test the token"
echo "Now try to push something:"
echo "git push"
echo ""
echo "When prompted:"
echo "Username: $github_username"
echo "Password: [paste your personal access token here]"
echo ""
echo "Note: The token should be 40+ characters long"
echo "If you get 'Support for password authentication was removed', you need to use the token" 