#!/bin/bash

# Simple Hugging Face Spaces Deployment Script

set -e

echo "🚀 Deploying Research Paper Classification API to Hugging Face Spaces"
echo ""

# Get Hugging Face username
read -p "Enter your Hugging Face username: " HF_USERNAME
export HF_USERNAME

# Get Hugging Face token
read -s -p "Enter your Hugging Face token: " HF_TOKEN
export HF_TOKEN
echo ""

# Space configuration
SPACE_NAME="research-paper-classifier"

echo ""
echo "📋 Configuration:"
echo "Username: ${HF_USERNAME}"
echo "Space Name: ${SPACE_NAME}"
echo ""

# Check if huggingface_hub is available
if ! python3 -c "import huggingface_hub" &> /dev/null; then
    echo "❌ huggingface_hub is not installed. Installing..."
    pip3 install huggingface_hub
fi

# Test authentication
echo "🔐 Testing authentication..."
python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
user = api.whoami()
print(f'✅ Authenticated as: {user}')
"

# Create the space
echo "🚀 Creating Hugging Face Space..."
python3 -c "
import os
from huggingface_hub import create_repo
create_repo(
    repo_id=f'{os.environ[\"HF_USERNAME\"]}/{os.environ[\"SPACE_NAME\"]}',
    repo_type='space',
    space_sdk='docker',
    token=os.environ['HF_TOKEN']
)
print('✅ Space created successfully!')
"

# Initialize git repository for the space
echo "🔧 Setting up git repository..."
git init
git add .
git commit -m "Initial commit for Hugging Face Space"

# Add the remote and push
echo "📤 Pushing to Hugging Face Spaces..."
git remote add origin https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}
git push -u origin main

echo ""
echo "✅ Deployment successful!"
echo "🌐 Space URL: https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
echo "📊 Health check: https://${SPACE_NAME}-${HF_USERNAME}.hf.space/health"
echo "📚 API docs: https://${SPACE_NAME}-${HF_USERNAME}.hf.space/docs"
echo ""
echo "🎉 Your Research Paper Classification API is now live on Hugging Face Spaces!"
echo "💡 The space will take a few minutes to build and deploy." 