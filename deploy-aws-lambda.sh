#!/bin/bash

# AWS Lambda Deployment Script using SAM
# This script deploys the Research Paper Classification API to AWS Lambda

set -e

# Configuration
STACK_NAME="research-paper-classifier"
REGION="us-east-1"
BUCKET_NAME="research-paper-classifier-deployment"

echo "🚀 Deploying Research Paper Classification API to AWS Lambda"
echo "Stack Name: ${STACK_NAME}"
echo "Region: ${REGION}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first:"
    echo "https://aws.amazon.com/cli/"
    exit 1
fi

# Check if SAM CLI is installed
if ! command -v sam &> /dev/null; then
    echo "❌ AWS SAM CLI is not installed. Please install it first:"
    echo "https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html"
    exit 1
fi

# Check if user is authenticated
if ! aws sts get-caller-identity &> /dev/null; then
    echo "🔐 Please configure AWS credentials:"
    aws configure
fi

# Create S3 bucket for deployment artifacts
echo "🪣 Creating S3 bucket for deployment artifacts..."
aws s3 mb s3://${BUCKET_NAME} --region ${REGION} || echo "Bucket already exists"

# Build the SAM application
echo "🏗️ Building SAM application..."
sam build

# Deploy the application
echo "🚀 Deploying to AWS Lambda..."
sam deploy \
    --stack-name ${STACK_NAME} \
    --s3-bucket ${BUCKET_NAME} \
    --region ${REGION} \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides \
        UseBaseline=true \
        UseLangchain=false \
        ModelPath=models/phi2-lora-cancer

# Get the API Gateway URL
API_URL=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --region ${REGION} \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
    --output text)

echo "✅ Deployment successful!"
echo "🌐 API Gateway URL: ${API_URL}"
echo "📊 Health check: ${API_URL}/health"
echo "📚 API docs: ${API_URL}/docs"

# Test the deployment
echo "🧪 Testing the deployment..."
sleep 10
curl -f "${API_URL}/health" || echo "⚠️ Health check failed, but deployment might still be starting up"

echo ""
echo "🎉 Your Research Paper Classification API is now live on AWS Lambda!"
echo "💡 To update the deployment, run this script again."
echo "🗑️ To delete the stack: sam delete --stack-name ${STACK_NAME} --region ${REGION}" 