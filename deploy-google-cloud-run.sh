#!/bin/bash

# Google Cloud Run Deployment Script
# This script deploys the Research Paper Classification API to Google Cloud Run

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"your-gcp-project-id"}
SERVICE_NAME="research-paper-classifier"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying Research Paper Classification API to Google Cloud Run"
echo "Project ID: ${PROJECT_ID}"
echo "Service Name: ${SERVICE_NAME}"
echo "Region: ${REGION}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "🔐 Please authenticate with Google Cloud:"
    gcloud auth login
fi

# Set the project
echo "📋 Setting project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and push the Docker image
echo "🏗️ Building and pushing Docker image..."
gcloud builds submit --tag ${IMAGE_NAME}

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 80 \
    --max-instances 10 \
    --set-env-vars="USE_BASELINE=true,USE_LANGCHAIN=false,MODEL_PATH=models/phi2-lora-cancer" \
    --port 8000

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo "✅ Deployment successful!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo "📊 Health check: ${SERVICE_URL}/health"
echo "📚 API docs: ${SERVICE_URL}/docs"

# Test the deployment
echo "🧪 Testing the deployment..."
sleep 10
curl -f "${SERVICE_URL}/health" || echo "⚠️ Health check failed, but deployment might still be starting up"

echo ""
echo "🎉 Your Research Paper Classification API is now live!"
echo "💡 To update the deployment, run this script again."
echo "🗑️ To delete the service: gcloud run services delete ${SERVICE_NAME} --region=${REGION}" 