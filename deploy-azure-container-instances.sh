#!/bin/bash

# Azure Container Instances Deployment Script
# This script deploys the Research Paper Classification API to Azure Container Instances

set -e

# Configuration
RESOURCE_GROUP="research-paper-classifier-rg"
LOCATION="eastus"
CONTAINER_NAME="research-paper-classifier"
REGISTRY_NAME="researchpaperclassifier"
IMAGE_NAME="${REGISTRY_NAME}.azurecr.io/${CONTAINER_NAME}:latest"

echo "🚀 Deploying Research Paper Classification API to Azure Container Instances"
echo "Resource Group: ${RESOURCE_GROUP}"
echo "Location: ${LOCATION}"
echo "Container Name: ${CONTAINER_NAME}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI is not installed. Please install it first:"
    echo "https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if user is authenticated
if ! az account show &> /dev/null; then
    echo "🔐 Please authenticate with Azure:"
    az login
fi

# Create resource group
echo "📋 Creating resource group..."
az group create --name ${RESOURCE_GROUP} --location ${LOCATION}

# Create Azure Container Registry
echo "🏗️ Creating Azure Container Registry..."
az acr create --resource-group ${RESOURCE_GROUP} --name ${REGISTRY_NAME} --sku Basic

# Enable admin user for ACR
echo "🔧 Enabling admin user for ACR..."
az acr update -n ${REGISTRY_NAME} --admin-enabled true

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name ${REGISTRY_NAME} --resource-group ${RESOURCE_GROUP} --query "loginServer" --output tsv)

# Login to ACR
echo "🔐 Logging into ACR..."
az acr login --name ${REGISTRY_RNAME}

# Build and push the Docker image
echo "🏗️ Building and pushing Docker image..."
docker build -t ${IMAGE_NAME} .
docker push ${IMAGE_NAME}

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name ${REGISTRY_NAME} --query "username" --output tsv)
ACR_PASSWORD=$(az acr credential show --name ${REGISTRY_NAME} --query "passwords[0].value" --output tsv)

# Deploy to Container Instances
echo "🚀 Deploying to Azure Container Instances..."
az container create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${CONTAINER_NAME} \
    --image ${IMAGE_NAME} \
    --registry-login-server ${ACR_LOGIN_SERVER} \
    --registry-username ${ACR_USERNAME} \
    --registry-password ${ACR_PASSWORD} \
    --dns-name-label ${CONTAINER_NAME} \
    --ports 8000 \
    --environment-variables \
        USE_BASELINE=true \
        USE_LANGCHAIN=false \
        MODEL_PATH=models/phi2-lora-cancer \
    --memory 2 \
    --cpu 2

# Get the public IP
PUBLIC_IP=$(az container show \
    --resource-group ${RESOURCE_GROUP} \
    --name ${CONTAINER_NAME} \
    --query "ipAddress.ip" \
    --output tsv)

echo "✅ Deployment successful!"
echo "🌐 Service URL: http://${PUBLIC_IP}:8000"
echo "📊 Health check: http://${PUBLIC_IP}:8000/health"
echo "📚 API docs: http://${PUBLIC_IP}:8000/docs"

# Test the deployment
echo "🧪 Testing the deployment..."
sleep 30
curl -f "http://${PUBLIC_IP}:8000/health" || echo "⚠️ Health check failed, but deployment might still be starting up"

echo ""
echo "🎉 Your Research Paper Classification API is now live on Azure Container Instances!"
echo "💡 To update the deployment, run this script again."
echo "🗑️ To delete the resources: az group delete --name ${RESOURCE_GROUP} --yes" 