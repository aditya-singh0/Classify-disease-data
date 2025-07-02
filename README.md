# Research Paper Analysis & Classification Pipeline

A comprehensive pipeline for fine-tuning small language models to classify research paper abstracts into cancer and non-cancer categories, with advanced disease extraction capabilities.

## 🎯 Project Overview

This project implements a complete research paper analysis and classification system that meets all requirements from the assignment:

- **Fine-tunes small language models** (Gemma, Phi, etc.) using LoRA for efficient training
- **Multi-label classification** of research paper abstracts (Cancer vs Non-Cancer)
- **Disease extraction** from abstracts with enhanced LangChain integration
- **Comprehensive evaluation** comparing baseline vs fine-tuned model performance
- **Structured outputs** with confidence scores and detailed analysis
- **Cloud deployment** ready for AWS Lambda, Google Cloud Run, and Hugging Face Spaces

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Running the Pipeline

```bash
# Complete training and evaluation pipeline
python train_pipeline.py --model_name microsoft/phi-2 --num_epochs 3

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
```

## 📊 Expected Outputs

### Classification Results

```json
{
  "abstract_id": "PMID123456",
  "classification": {
    "predicted_labels": ["Cancer"],
    "confidence_scores": {
      "Cancer": 0.92,
      "Non-Cancer": 0.08
    }
  },
  "disease_extraction": {
    "extracted_diseases": ["lung cancer", "non-small cell lung cancer"]
  }
}
```

### Performance Comparison

**Baseline Model Performance:**
- Accuracy: 85%
- F1-score: 0.78
- Confusion Matrix: [320, 80; 50, 550]

**Fine-tuned Model Performance:**
- Accuracy: 92%
- F1-score: 0.86
- Confusion Matrix: [350, 50; 30, 570]

**Improvement Analysis:**
- Accuracy increased by 7% after fine-tuning
- Reduction in false negatives, improving model reliability
- Fine-tuned model provides better classification confidence

## 🏗️ Architecture

### Core Components

1. **Data Processing** (`src/data_processing.py`)
   - Handles PubMed data loading and cleaning
   - Text preprocessing and normalization
   - Disease extraction using rule-based and LangChain methods

2. **Model Training** (`src/model_training.py`)
   - LoRA fine-tuning for multiple model types (Phi, Gemma, etc.)
   - Configurable training parameters
   - Baseline vs fine-tuned comparison

3. **Classification** (`src/classification.py`)
   - Inference engine with confidence scoring
   - Enhanced disease extraction
   - LangChain integration for structured analysis

4. **Evaluation** (`src/evaluation.py`)
   - Comprehensive performance metrics
   - Confusion matrix visualization
   - ROC and Precision-Recall curves

5. **API** (`src/api/main.py`)
   - RESTful API with FastAPI
   - Batch processing support
   - Health monitoring and model management

### Model Support

| Model | LoRA Config | Max Length | Use Case |
|-------|-------------|------------|----------|
| microsoft/phi-2 | r=16, α=32 | 2048 | High performance |
| microsoft/phi-1_5 | r=16, α=32 | 2048 | Balanced |
| google/gemma-2b | r=16, α=32 | 2048 | Multilingual |
| microsoft/DialoGPT-medium | r=8, α=16 | 1024 | Lightweight |

## 🔧 Advanced Features

### LangChain Integration

The pipeline includes enhanced LangChain integration for:

- **Structured disease extraction** using LLM prompting
- **Citation analysis** and reference extraction
- **Abstract summarization** and key concept identification
- **Fallback mechanisms** to rule-based extraction

### Agentic Workflow

- **Modular pipeline design** for easy orchestration
- **Background task processing** for model reloading
- **Comprehensive error handling** and logging
- **Scalable architecture** for production deployment

### Cloud Deployment

Ready for deployment on multiple platforms:

- **AWS Lambda** with serverless configuration
- **Google Cloud Run** with auto-scaling
- **Hugging Face Spaces** with Gradio interface
- **Docker containers** with monitoring stack

## 📈 Performance Metrics

### Evaluation Framework

The pipeline provides comprehensive evaluation including:

- **Accuracy, Precision, Recall, F1-score**
- **Per-class metrics** (Cancer/Non-Cancer specific)
- **ROC curves** and AUC scores
- **Precision-Recall curves**
- **Confusion matrices** with visualizations
- **Statistical significance testing**

### Expected Improvements

Based on the assignment requirements:

- **7% accuracy improvement** after fine-tuning
- **Reduced false negatives** for better cancer detection
- **Enhanced confidence scoring** for reliable predictions
- **Improved disease extraction** accuracy

## 🛠️ Development

### Project Structure

```
Velsera/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   ├── data_processing.py       # Data loading and preprocessing
│   ├── model_training.py        # LoRA fine-tuning
│   ├── classification.py        # Inference and disease extraction
│   └── evaluation.py            # Performance evaluation
├── Dataset/
│   ├── Cancer/                  # Cancer research papers
│   └── Non-Cancer/              # Non-cancer research papers
├── data/
│   └── sample_abstracts.csv     # Sample data
├── models/                      # Trained models
├── output/                      # Pipeline outputs
├── tests/                       # Unit tests
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container configuration
├── docker-compose.yml           # Multi-service deployment
├── DEPLOYMENT.md                # Deployment guide
└── README.md                    # This file
```

### Running Tests

```bash
# Run unit tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Development Setup

```bash
# Install in development mode
pip install -e .

# Run linting
black src/
flake8 src/

# Type checking
mypy src/
```

## 🌐 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check and model info |
| GET | `/health` | Health status |
| POST | `/classify` | Single abstract classification |
| POST | `/batch-classify` | Batch classification |
| POST | `/extract-diseases` | Disease extraction only |
| GET | `/model-info` | Detailed model information |
| POST | `/reload-model` | Reload model (background) |
| GET | `/api-docs` | API documentation |

### Request/Response Examples

**Single Classification:**
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "pubmed_id": "PMID123456",
    "abstract": "This study investigates lung cancer progression..."
  }'
```

**Batch Classification:**
```bash
curl -X POST http://localhost:8000/batch-classify \
  -H "Content-Type: application/json" \
  -d '{
    "abstracts": [
      {"pubmed_id": "PMID123456", "abstract": "..."},
      {"pubmed_id": "PMID789012", "abstract": "..."}
    ]
  }'
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run
docker build -t research-paper-classifier .
docker run -p 8000:8000 research-paper-classifier

# Or with Docker Compose
docker-compose up -d
```

### Cloud Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on:

- AWS Lambda deployment
- Google Cloud Run deployment
- Hugging Face Spaces deployment
- Kubernetes orchestration
- Monitoring and scaling

## 📊 Monitoring and Observability

### Built-in Metrics

- **Request rate and latency**
- **Classification accuracy tracking**
- **Model performance metrics**
- **Error rates and logs**
- **Resource utilization**

### Visualization

- **Real-time dashboards** with Grafana
- **Performance comparison charts**
- **Confusion matrix heatmaps**
- **ROC and PR curves**

## 🔒 Security

### Security Features

- **Input validation** and sanitization
- **Rate limiting** to prevent abuse
- **CORS configuration** for web access
- **Error handling** without information leakage
- **Model security** best practices

### Best Practices

- Use environment variables for sensitive configuration
- Implement proper authentication for production
- Regular security updates and dependency scanning
- Monitor for anomalous usage patterns

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for all classes and methods
- Write comprehensive unit tests
- Update documentation as needed

## 📚 References

### Research Papers

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2103.08547)
- [Medical Text Classification](https://arxiv.org/abs/2004.12393)

### Technologies

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft)
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://python.langchain.com/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the transformers library and model hub
- Microsoft for the Phi models
- Google for the Gemma models
- The research community for the medical datasets

## 📞 Support

For questions, issues, or contributions:

1. Check the [documentation](DEPLOYMENT.md)
2. Search existing [issues](../../issues)
3. Create a new issue with detailed information
4. Contact the development team

---

**Note**: This pipeline is designed for research and educational purposes. For clinical applications, additional validation and regulatory compliance may be required. 