#!/bin/bash

# Hugging Face Spaces Deployment Script
# This script deploys the Research Paper Classification API to Hugging Face Spaces

set -e

# Configuration
SPACE_NAME="research-paper-classifier"
USERNAME=${HF_USERNAME:-"your-username"}
SPACE_TYPE="docker"

echo "🚀 Deploying Research Paper Classification API to Hugging Face Spaces"
echo "Space Name: ${SPACE_NAME}"
echo "Username: ${USERNAME}"
echo "Space Type: ${SPACE_TYPE}"

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" &> /dev/null; then
    echo "❌ huggingface_hub is not installed. Installing..."
    pip install huggingface_hub
fi

# Check if user is authenticated
if ! huggingface-cli whoami &> /dev/null; then
    echo "🔐 Please authenticate with Hugging Face:"
    huggingface-cli login
fi

# Create app.py for Hugging Face Spaces
echo "📝 Creating app.py for Hugging Face Spaces..."
cat > app.py << 'EOF'
import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import your FastAPI app
from api.main import app

# Add CORS middleware for Hugging Face Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
EOF

# Create README.md for the space
echo "📝 Creating README.md for the space..."
cat > README.md << 'EOF'
# Research Paper Classification API

This space hosts a FastAPI application for classifying research paper abstracts into cancer and non-cancer categories.

## API Endpoints

- `GET /` - Health check and model information
- `GET /health` - Health check endpoint
- `POST /classify` - Classify a single abstract
- `POST /batch-classify` - Classify multiple abstracts
- `POST /extract-diseases` - Extract diseases from abstract
- `GET /model-info` - Get model information
- `GET /docs` - Interactive API documentation

## Usage

Send a POST request to `/classify` with JSON body:

```json
{
  "pubmed_id": "12345",
  "abstract": "Your research paper abstract here..."
}
```

## Model Information

- **Model Type**: Baseline classifier (can be configured for fine-tuned models)
- **Supported Categories**: Cancer, Non-Cancer
- **Disease Extraction**: Rule-based extraction of disease mentions

## Environment Variables

- `USE_BASELINE`: Set to "true" for baseline classifier
- `USE_LANGCHAIN`: Set to "true" to enable LangChain analysis
- `MODEL_PATH`: Path to the model directory
EOF

# Create .gitattributes for the space
echo "📝 Creating .gitattributes for the space..."
cat > .gitattributes << 'EOF'
*.py filter=lfs diff=lfs merge=lfs -text
*.json filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
EOF

# Create requirements.txt for the space
echo "📝 Creating requirements.txt for the space..."
cp requirements.txt requirements-spaces.txt

# Initialize git repository for the space
echo "🔧 Initializing git repository..."
git init
git add .
git commit -m "Initial commit for Hugging Face Space"

# Create the space on Hugging Face
echo "🚀 Creating space on Hugging Face..."
huggingface-cli repo create ${SPACE_NAME} --type space --space-sdk docker

# Add the remote and push
echo "📤 Pushing to Hugging Face Spaces..."
git remote add origin https://huggingface.co/spaces/${USERNAME}/${SPACE_NAME}
git push -u origin main

echo "✅ Deployment successful!"
echo "🌐 Space URL: https://huggingface.co/spaces/${USERNAME}/${SPACE_NAME}"
echo "📊 Health check: https://${SPACE_NAME}-${USERNAME}.hf.space/health"
echo "📚 API docs: https://${SPACE_NAME}-${USERNAME}.hf.space/docs"

echo ""
echo "🎉 Your Research Paper Classification API is now live on Hugging Face Spaces!"
echo "💡 The space will take a few minutes to build and deploy."
echo "🗑️ To delete the space: huggingface-cli repo delete ${USERNAME}/${SPACE_NAME}" 