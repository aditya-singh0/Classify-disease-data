# Deployment Guide

This guide provides step-by-step instructions for deploying the Research Paper Classification API to various cloud platforms.

## Prerequisites

Before deploying, ensure you have the following installed:

- **Docker** - [Install Docker](https://docs.docker.com/get-docker/)
- **Git** - [Install Git](https://git-scm.com/downloads)
- **Python 3.9+** - [Install Python](https://www.python.org/downloads/)

## Quick Start

Use the master deployment script for easy deployment:

```bash
# Make the script executable
chmod +x deploy.sh

# Run interactive menu
./deploy.sh

# Or deploy directly to a specific platform
./deploy.sh local      # Deploy locally
./deploy.sh gcp        # Deploy to Google Cloud Run
./deploy.sh aws        # Deploy to AWS Lambda
./deploy.sh hf         # Deploy to Hugging Face Spaces
./deploy.sh azure      # Deploy to Azure
```

## Platform-Specific Deployment

### 1. Local Deployment with Docker

**Easiest option for testing and development.**

```bash
# Build and run locally
./deploy.sh local

# Or manually:
docker build -t research-paper-classifier .
docker run -d --name research-paper-classifier -p 8000:8000 research-paper-classifier
```

**Access your API:**
- Service URL: http://localhost:8000
- Health Check: http://localhost:8000/health
- API Docs: http://localhost:8000/docs

### 2. Google Cloud Run

**Recommended for production deployments.**

#### Prerequisites:
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- Google Cloud account with billing enabled

#### Deployment:
```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"

# Deploy
./deploy-google-cloud-run.sh
```

#### Features:
- Auto-scaling
- Pay-per-use pricing
- HTTPS by default
- Global CDN

### 3. AWS Lambda

**Serverless option for cost-effective deployments.**

#### Prerequisites:
- [AWS CLI](https://aws.amazon.com/cli/)
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- AWS account

#### Deployment:
```bash
# Configure AWS credentials
aws configure

# Deploy
./deploy-aws-lambda.sh
```

#### Features:
- Serverless (no server management)
- Pay-per-request
- Auto-scaling
- Integration with AWS services

### 4. Hugging Face Spaces

**Perfect for ML-focused applications and demos.**

#### Prerequisites:
- [Hugging Face account](https://huggingface.co/join)
- `huggingface_hub` Python package

#### Deployment:
```bash
# Set your username
export HF_USERNAME="your-username"

# Deploy
./deploy-huggingface-spaces.sh
```

#### Features:
- Free hosting for ML applications
- Built-in model hosting
- Easy sharing and collaboration
- Automatic HTTPS

### 5. Azure Container Instances

**Microsoft's container hosting solution.**

#### Prerequisites:
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- Azure account

#### Deployment:
```bash
# Login to Azure
az login

# Deploy
./deploy-azure-container-instances.sh
```

#### Features:
- Container-native
- Pay-per-second billing
- Integration with Azure services
- Global deployment

## Configuration Options

### Environment Variables

You can customize the deployment by setting these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_BASELINE` | `true` | Use baseline classifier instead of fine-tuned model |
| `USE_LANGCHAIN` | `false` | Enable LangChain analysis |
| `MODEL_PATH` | `models/phi2-lora-cancer` | Path to the model directory |

### Example Customization

```bash
# Deploy with fine-tuned model and LangChain
docker run -d \
  --name research-paper-classifier \
  -p 8000:8000 \
  -e USE_BASELINE=false \
  -e USE_LANGCHAIN=true \
  -e MODEL_PATH=models/phi2-lora-cancer \
  research-paper-classifier
```

## API Endpoints

Once deployed, your API will have the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and model information |
| `/health` | GET | Health check endpoint |
| `/classify` | POST | Classify a single abstract |
| `/batch-classify` | POST | Classify multiple abstracts |
| `/extract-diseases` | POST | Extract diseases from abstract |
| `/model-info` | GET | Get detailed model information |
| `/docs` | GET | Interactive API documentation |

### Example API Usage

```bash
# Classify a single abstract
curl -X POST "http://your-api-url/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "pubmed_id": "12345",
    "abstract": "This study investigates the role of p53 mutations in breast cancer development..."
  }'

# Health check
curl "http://your-api-url/health"
```

## Monitoring and Logs

### Local Docker
```bash
# View logs
docker logs research-paper-classifier

# Monitor resource usage
docker stats research-paper-classifier
```

### Cloud Platforms

Each platform provides its own monitoring:

- **Google Cloud Run**: Cloud Console > Cloud Run > Your Service
- **AWS Lambda**: CloudWatch Logs and Metrics
- **Hugging Face Spaces**: Space dashboard
- **Azure**: Azure Monitor and Log Analytics

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Stop existing container
   docker stop research-paper-classifier
   docker rm research-paper-classifier
   ```

2. **Memory issues**
   - Increase memory allocation in cloud platform settings
   - Use baseline model instead of fine-tuned model

3. **Authentication errors**
   - Ensure you're logged in to the respective cloud platform
   - Check API keys and permissions

4. **Model loading failures**
   - Verify model files are present in the `models/` directory
   - Check `MODEL_PATH` environment variable

### Getting Help

- Check the logs: `docker logs research-paper-classifier`
- Verify health endpoint: `curl http://your-api-url/health`
- Review API documentation: `http://your-api-url/docs`

## Cost Optimization

### Google Cloud Run
- Set minimum instances to 0 for cost savings
- Use appropriate memory/CPU allocation

### AWS Lambda
- Optimize function timeout and memory
- Use provisioned concurrency for consistent performance

### Hugging Face Spaces
- Free tier available
- Consider paid plans for production use

### Azure Container Instances
- Use appropriate VM size
- Consider Azure Container Apps for auto-scaling

## Security Considerations

1. **API Keys**: Implement authentication for production deployments
2. **HTTPS**: All cloud platforms provide HTTPS by default
3. **CORS**: Configure CORS settings for your domain
4. **Rate Limiting**: Implement rate limiting for public APIs
5. **Input Validation**: The API includes input validation, but review for your use case

## Updates and Maintenance

### Updating the Deployment

```bash
# Pull latest changes
git pull origin main

# Rebuild and redeploy
./deploy.sh [platform]
```

### Backup and Recovery

- **Models**: Keep backups of your trained models
- **Configuration**: Version control your deployment scripts
- **Data**: Implement data backup strategies for production

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the API documentation at `/docs`
3. Check the logs for error messages
4. Open an issue in the project repository 