#!/bin/bash

# Master Deployment Script
# This script provides options to deploy the Research Paper Classification API to different cloud platforms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Research Paper Classifier API${NC}"
    echo -e "${BLUE}  Deployment Options${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    
    print_status "Prerequisites check passed!"
}

# Function to build Docker image locally
build_docker_image() {
    print_status "Building Docker image..."
    docker build -t research-paper-classifier:latest .
    print_status "Docker image built successfully!"
}

# Function to run locally with Docker
deploy_local() {
    print_status "Deploying locally with Docker..."
    docker run -d \
        --name research-paper-classifier \
        -p 8000:8000 \
        -e USE_BASELINE=true \
        -e USE_LANGCHAIN=false \
        -e MODEL_PATH=models/phi2-lora-cancer \
        research-paper-classifier:latest
    
    print_status "Local deployment successful!"
    echo "🌐 Service URL: http://localhost:8000"
    echo "📊 Health check: http://localhost:8000/health"
    echo "📚 API docs: http://localhost:8000/docs"
}

# Function to deploy to Google Cloud Run
deploy_google_cloud_run() {
    print_status "Deploying to Google Cloud Run..."
    
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI is not installed. Please install it first:"
        echo "https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Set project ID
    read -p "Enter your Google Cloud Project ID: " PROJECT_ID
    export PROJECT_ID
    
    # Run the deployment script
    chmod +x deploy-google-cloud-run.sh
    ./deploy-google-cloud-run.sh
}

# Function to deploy to AWS Lambda
deploy_aws_lambda() {
    print_status "Deploying to AWS Lambda..."
    
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first:"
        echo "https://aws.amazon.com/cli/"
        exit 1
    fi
    
    if ! command -v sam &> /dev/null; then
        print_error "AWS SAM CLI is not installed. Please install it first:"
        echo "https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html"
        exit 1
    fi
    
    # Run the deployment script
    chmod +x deploy-aws-lambda.sh
    ./deploy-aws-lambda.sh
}

# Function to deploy to Hugging Face Spaces
deploy_huggingface_spaces() {
    print_status "Deploying to Hugging Face Spaces..."
    
    if ! python -c "import huggingface_hub" &> /dev/null; then
        print_warning "huggingface_hub is not installed. Installing..."
        pip install huggingface_hub
    fi
    
    # Set username
    read -p "Enter your Hugging Face username: " HF_USERNAME
    export HF_USERNAME
    
    # Run the deployment script
    chmod +x deploy-huggingface-spaces.sh
    ./deploy-huggingface-spaces.sh
}

# Function to deploy to Azure Container Instances
deploy_azure() {
    print_status "Deploying to Azure Container Instances..."
    
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI is not installed. Please install it first:"
        echo "https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi
    
    # Run the deployment script
    chmod +x deploy-azure-container-instances.sh
    ./deploy-azure-container-instances.sh
}

# Main menu
main_menu() {
    print_header
    echo ""
    echo "Choose your deployment option:"
    echo "1) Deploy locally with Docker"
    echo "2) Deploy to Google Cloud Run"
    echo "3) Deploy to AWS Lambda"
    echo "4) Deploy to Hugging Face Spaces"
    echo "5) Deploy to Azure Container Instances"
    echo "6) Build Docker image only"
    echo "7) Exit"
    echo ""
    read -p "Enter your choice (1-7): " choice
    
    case $choice in
        1)
            check_prerequisites
            build_docker_image
            deploy_local
            ;;
        2)
            check_prerequisites
            build_docker_image
            deploy_google_cloud_run
            ;;
        3)
            check_prerequisites
            build_docker_image
            deploy_aws_lambda
            ;;
        4)
            check_prerequisites
            build_docker_image
            deploy_huggingface_spaces
            ;;
        5)
            check_prerequisites
            build_docker_image
            deploy_azure
            ;;
        6)
            check_prerequisites
            build_docker_image
            print_status "Docker image built successfully!"
            ;;
        7)
            print_status "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please enter a number between 1-7."
            main_menu
            ;;
    esac
}

# Check if script is run with arguments
if [ $# -eq 0 ]; then
    main_menu
else
    case $1 in
        "local")
            check_prerequisites
            build_docker_image
            deploy_local
            ;;
        "gcp"|"google")
            check_prerequisites
            build_docker_image
            deploy_google_cloud_run
            ;;
        "aws"|"lambda")
            check_prerequisites
            build_docker_image
            deploy_aws_lambda
            ;;
        "hf"|"huggingface")
            check_prerequisites
            build_docker_image
            deploy_huggingface_spaces
            ;;
        "azure")
            check_prerequisites
            build_docker_image
            deploy_azure
            ;;
        "build")
            check_prerequisites
            build_docker_image
            ;;
        *)
            print_error "Invalid argument. Usage:"
            echo "  ./deploy.sh                    # Interactive menu"
            echo "  ./deploy.sh local              # Deploy locally"
            echo "  ./deploy.sh gcp                # Deploy to Google Cloud Run"
            echo "  ./deploy.sh aws                # Deploy to AWS Lambda"
            echo "  ./deploy.sh hf                 # Deploy to Hugging Face Spaces"
            echo "  ./deploy.sh azure              # Deploy to Azure"
            echo "  ./deploy.sh build              # Build Docker image only"
            exit 1
            ;;
    esac
fi 