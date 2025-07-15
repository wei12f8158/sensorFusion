# Setting up GitHub Personal Access Token on Raspberry Pi 5

## Step 1: Create Personal Access Token on GitHub
1. Go to GitHub.com and log in
2. Click your profile picture → Settings
3. Scroll down to "Developer settings" (bottom left)
4. Click "Personal access tokens" → "Tokens (classic)"
5. Click "Generate new token" → "Generate new token (classic)"
6. Give it a name like "Raspberry Pi 5"
7. Select scopes: check "repo" (full control of private repositories)
8. Click "Generate token"
9. **COPY THE TOKEN** (you won't see it again!)

## Step 2: Configure Git on Raspberry Pi 5
```bash
# On your Raspberry Pi 5
git config --global user.name "wei12f8158"
git config --global user.email "your-email@example.com"

# Store credentials (optional, but convenient)
git config --global credential.helper store
```

## Step 3: Test the Token
```bash
# Try to push something
echo "test" > test_file.txt
git add test_file.txt
git commit -m "test commit"
git push

# When prompted:
# Username: wei12f8158
# Password: [paste your personal access token here]
```

## Step 4: Alternative - Use Git Credential Manager
```bash
# Install git credential manager
sudo apt-get install git-credential-manager

# Configure it
git config --global credential.helper manager
```

## Troubleshooting
- If you get "Support for password authentication was removed", you need to use the token
- Make sure the token has "repo" permissions
- The token should be 40+ characters long
- Don't share your token with anyone 