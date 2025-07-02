# Research Paper Analysis & Classification Pipeline - Deployment Guide

This guide covers deployment options for the research paper classification pipeline, including local development, Docker, and cloud platforms as required by the assignment.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [AWS Lambda Deployment](#aws-lambda-deployment)
4. [Google Cloud Run Deployment](#google-cloud-run-deployment)
5. [Hugging Face Spaces Deployment](#hugging-face-spaces-deployment)
6. [Monitoring and Scaling](#monitoring-and-scaling)

## Local Development

### Prerequisites

```bash
# Install Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Running the Pipeline

```bash
# Run complete training pipeline
python train_pipeline.py --model_name microsoft/phi-2 --num_epochs 3

# Run evaluation only
python train_pipeline.py --skip_training --model_path ./output/models/microsoft_phi-2-lora-cancer

# Start API server
python -m src.api.main
```

### API Testing

```bash
# Health check
curl http://localhost:8000/health

# Single classification
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "pubmed_id": "PMID123456",
    "abstract": "This study investigates lung cancer progression and identifies novel therapeutic targets."
  }'

# Batch classification
curl -X POST http://localhost:8000/batch-classify \
  -H "Content-Type: application/json" \
  -d '{
    "abstracts": [
      {
        "pubmed_id": "PMID123456",
        "abstract": "This study investigates lung cancer progression..."
      },
      {
        "pubmed_id": "PMID789012",
        "abstract": "The effects of exercise on cardiovascular health..."
      }
    ]
  }'
```

## Docker Deployment

### Building and Running

```bash
# Build Docker image
docker build -t research-paper-classifier .

# Run container
docker run -p 8000:8000 research-paper-classifier

# Run with Docker Compose
docker-compose up -d
```

### Docker Compose Services

The `docker-compose.yml` includes:

- **research-paper-classifier**: Main API service
- **redis**: Caching and session management
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboard

### Environment Variables

```bash
# Docker environment variables
USE_BASELINE=true                    # Use baseline classifier
USE_LANGCHAIN=false                  # Enable LangChain features
MODEL_PATH=models/phi2-lora-cancer   # Path to fine-tuned model
```

## AWS Lambda Deployment

### Prerequisites

```bash
# Install AWS CLI
aws configure

# Install Serverless Framework
npm install -g serverless
```

### Serverless Configuration

Create `serverless.yml`:

```yaml
service: research-paper-classifier

provider:
  name: aws
  runtime: python3.9
  region: us-east-1
  memorySize: 2048
  timeout: 30

functions:
  api:
    handler: src.api.lambda_handler.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
          cors: true
    environment:
      USE_BASELINE: true
      USE_LANGCHAIN: false
      MODEL_PATH: models/phi2-lora-cancer

package:
  patterns:
    - '!node_modules/**'
    - '!tests/**'
    - '!*.pyc'
    - '!.git/**'
```

### Lambda Handler

Create `src/api/lambda_handler.py`:

```python
import json
from mangum import Mangum
from src.api.main import app

handler = Mangum(app)
```

### Deployment Commands

```bash
# Deploy to AWS Lambda
serverless deploy

# Deploy to specific stage
serverless deploy --stage production

# Remove deployment
serverless remove
```

## Google Cloud Run Deployment

### Prerequisites

```bash
# Install Google Cloud SDK
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Cloud Build Configuration

Create `cloudbuild.yaml`:

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/research-paper-classifier', '.']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/research-paper-classifier']
  
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'research-paper-classifier'
      - '--image'
      - 'gcr.io/$PROJECT_ID/research-paper-classifier'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--memory'
      - '2Gi'
      - '--cpu'
      - '2'
      - '--max-instances'
      - '10'
```

### Deployment Commands

```bash
# Build and deploy
gcloud builds submit --config cloudbuild.yaml

# Deploy directly
gcloud run deploy research-paper-classifier \
  --image gcr.io/YOUR_PROJECT_ID/research-paper-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

## Hugging Face Spaces Deployment

### Space Configuration

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Hugging Face Spaces

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to Spaces
        uses: huggingface/huggingface_hub@main
        with:
          repo-type: space
          space-sdk: gradio
          space-hardware: cpu-basic
          space-settings: |
            title: Research Paper Classifier
            emoji: 🔬
            colorFrom: blue
            colorTo: purple
            sdk: gradio
            sdk_version: 3.50.2
            app_file: app.py
            pinned: false
```

### Gradio Interface

Create `app.py` for Hugging Face Spaces:

```python
import gradio as gr
from src.classification import load_classifier

# Load classifier
classifier = load_classifier(None, use_baseline=True)

def classify_abstract(abstract):
    """Classify a single abstract."""
    result = classifier.predict_single(abstract, "demo")
    return classifier.format_output(result)

# Create Gradio interface
iface = gr.Interface(
    fn=classify_abstract,
    inputs=gr.Textbox(label="Research Paper Abstract", lines=5),
    outputs=gr.JSON(label="Classification Results"),
    title="Research Paper Analysis & Classification",
    description="Classify research paper abstracts into cancer and non-cancer categories with disease extraction."
)

iface.launch()
```

## Monitoring and Scaling

### Prometheus Metrics

The API includes built-in metrics:

```python
# Custom metrics
from prometheus_client import Counter, Histogram, generate_latest

# Request counters
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
CLASSIFICATION_COUNT = Counter('classifications_total', 'Total classifications', ['label'])

# Response time histogram
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
```

### Grafana Dashboard

Create monitoring dashboard with:

- Request rate and latency
- Classification accuracy
- Model performance metrics
- Error rates and logs

### Auto-scaling Configuration

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: research-paper-classifier-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: research-paper-classifier
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Performance Optimization

### Caching Strategy

```python
# Redis caching for repeated requests
import redis
import hashlib
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_result(abstract):
    """Get cached classification result."""
    key = hashlib.md5(abstract.encode()).hexdigest()
    cached = redis_client.get(key)
    return json.loads(cached) if cached else None

def cache_result(abstract, result):
    """Cache classification result."""
    key = hashlib.md5(abstract.encode()).hexdigest()
    redis_client.setex(key, 3600, json.dumps(result))  # 1 hour TTL
```

### Batch Processing

```python
# Optimized batch processing
async def process_batch(abstracts, batch_size=32):
    """Process abstracts in optimized batches."""
    results = []
    for i in range(0, len(abstracts), batch_size):
        batch = abstracts[i:i + batch_size]
        batch_results = await process_batch_async(batch)
        results.extend(batch_results)
    return results
```

## Security Considerations

### API Security

```python
# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/classify")
@limiter.limit("10/minute")
def classify_abstract(request: AbstractRequest):
    # Implementation
    pass
```

### Model Security

```python
# Input validation
from pydantic import validator

class AbstractRequest(BaseModel):
    abstract: str
    
    @validator('abstract')
    def validate_abstract(cls, v):
        if len(v) > 10000:
            raise ValueError('Abstract too long')
        if not v.strip():
            raise ValueError('Abstract cannot be empty')
        return v.strip()
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Increase container memory or use model quantization
2. **Timeout Errors**: Optimize model loading and inference
3. **Cold Start**: Use model caching and warm-up endpoints
4. **Scale Issues**: Implement proper load balancing and auto-scaling

### Logging and Debugging

```python
# Structured logging
import structlog

logger = structlog.get_logger()

logger.info("Processing request", 
           pubmed_id=request.pubmed_id,
           abstract_length=len(request.abstract),
           model_type=classifier_type)
```

## Cost Optimization

### AWS Lambda

- Use provisioned concurrency for consistent workloads
- Optimize memory allocation (more memory = faster execution)
- Use S3 for model storage instead of Lambda layers

### Google Cloud Run

- Use CPU allocation based on actual usage
- Implement request batching to reduce cold starts
- Use Cloud CDN for static content

### General Tips

- Monitor usage patterns and adjust resources accordingly
- Use spot instances for non-critical workloads
- Implement proper caching to reduce compute costs

## Conclusion

This deployment guide covers all major cloud platforms and deployment scenarios required by the assignment. The pipeline is designed to be:

- **Scalable**: Supports auto-scaling and load balancing
- **Secure**: Includes rate limiting and input validation
- **Monitorable**: Provides comprehensive metrics and logging
- **Cost-effective**: Optimized for various deployment scenarios

Choose the deployment option that best fits your requirements and budget constraints. 