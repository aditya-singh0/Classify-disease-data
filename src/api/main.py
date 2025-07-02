from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import os
import json
from datetime import datetime
from src.classification import load_classifier, ClassificationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Paper Analysis & Classification API",
    description="API for classifying research paper abstracts into cancer and non-cancer categories with disease extraction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load classifier at startup (can be swapped for baseline or fine-tuned)
MODEL_PATH = os.getenv("MODEL_PATH", "models/phi2-lora-cancer")
USE_BASELINE = os.getenv("USE_BASELINE", "true").lower() == "true"  # Default to baseline
USE_LANGCHAIN = os.getenv("USE_LANGCHAIN", "false").lower() == "true"

# Initialize classifier safely
try:
    if USE_BASELINE or not os.path.exists(MODEL_PATH):
        logger.info("Using baseline classifier")
        classifier = load_classifier(None, use_baseline=True, use_langchain=USE_LANGCHAIN)
    else:
        logger.info(f"Loading fine-tuned classifier from {MODEL_PATH}")
        classifier = load_classifier(MODEL_PATH, use_baseline=False, use_langchain=USE_LANGCHAIN)
except Exception as e:
    logger.warning(f"Failed to load fine-tuned model, falling back to baseline: {e}")
    classifier = load_classifier(None, use_baseline=True, use_langchain=USE_LANGCHAIN)

# Pydantic models for request/response
class AbstractRequest(BaseModel):
    pubmed_id: Optional[str] = "unknown"
    abstract: str

class BatchRequest(BaseModel):
    abstracts: List[AbstractRequest]

class ClassificationResponse(BaseModel):
    abstract_id: str
    classification: Dict[str, Any]
    disease_extraction: Dict[str, Any]
    langchain_analysis: Optional[Dict[str, Any]] = None

class BatchResponse(BaseModel):
    results: List[ClassificationResponse]
    total_processed: int
    successful_predictions: int
    summary: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    classifier_type: str
    langchain_enabled: bool
    model_info: Dict[str, Any]

@app.get("/", response_model=HealthResponse)
def root():
    """Root endpoint with health check and model information."""
    return {
        "status": "ok",
        "classifier_type": "baseline" if USE_BASELINE else "fine-tuned",
        "langchain_enabled": USE_LANGCHAIN,
        "model_info": {
            "model_path": MODEL_PATH if not USE_BASELINE else "baseline",
            "use_baseline": USE_BASELINE,
            "use_langchain": USE_LANGCHAIN,
            "timestamp": datetime.now().isoformat()
        }
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "classifier_type": "baseline" if USE_BASELINE else "fine-tuned",
        "langchain_enabled": USE_LANGCHAIN,
        "model_info": {
            "model_path": MODEL_PATH if not USE_BASELINE else "baseline",
            "use_baseline": USE_BASELINE,
            "use_langchain": USE_LANGCHAIN,
            "timestamp": datetime.now().isoformat()
        }
    }

@app.post("/classify", response_model=ClassificationResponse)
def classify_abstract(request: AbstractRequest):
    """
    Classify a single research paper abstract.
    
    Returns:
    - Predicted labels (Cancer/Non-Cancer)
    - Confidence scores for each category
    - Extracted diseases from the abstract
    - Optional LangChain analysis
    """
    try:
        result: ClassificationResult = classifier.predict_single(request.abstract, request.pubmed_id)
        return classifier.format_output(result)
    except Exception as e:
        logger.error(f"Error in /classify: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-classify", response_model=BatchResponse)
def batch_classify(request: BatchRequest):
    """
    Classify multiple research paper abstracts in batch.
    
    Returns:
    - Results for each abstract
    - Summary statistics
    - Processing information
    """
    try:
        abstracts = [a.dict() for a in request.abstracts]
        results = classifier.predict_batch(abstracts)
        return classifier.format_batch_output(results)
    except Exception as e:
        logger.error(f"Error in /batch-classify: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-diseases")
def extract_diseases(request: AbstractRequest):
    """
    Extract diseases from a research paper abstract without classification.
    
    Returns:
    - List of extracted diseases
    - Disease categories (cancer types, medical conditions)
    - Confidence scores
    """
    try:
        # Use the disease extractor directly
        disease_extraction = classifier.extract_diseases(request.abstract)
        
        return {
            "abstract_id": request.pubmed_id,
            "disease_extraction": disease_extraction,
            "extraction_method": disease_extraction.get("extraction_method", "rule_based")
        }
    except Exception as e:
        logger.error(f"Error in /extract-diseases: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def get_model_info():
    """Get detailed information about the loaded model."""
    try:
        model_info = {
            "model_type": "baseline" if USE_BASELINE else "fine-tuned",
            "model_path": MODEL_PATH if not USE_BASELINE else "baseline",
            "langchain_enabled": USE_LANGCHAIN,
            "device": getattr(classifier, 'device', 'unknown'),
            "max_length": getattr(classifier, 'max_length', 'unknown'),
            "label_map": getattr(classifier, 'label_map', {}),
            "loaded_at": datetime.now().isoformat()
        }
        
        return model_info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-model")
def reload_model(background_tasks: BackgroundTasks):
    """Reload the model (useful for switching between baseline and fine-tuned)."""
    try:
        global classifier
        
        # Reload classifier in background
        def reload():
            global classifier
            try:
                if USE_BASELINE or not os.path.exists(MODEL_PATH):
                    classifier = load_classifier(None, use_baseline=True, use_langchain=USE_LANGCHAIN)
                else:
                    classifier = load_classifier(MODEL_PATH, use_baseline=False, use_langchain=USE_LANGCHAIN)
                logger.info("Model reloaded successfully")
            except Exception as e:
                logger.error(f"Failed to reload model: {e}")
        
        background_tasks.add_task(reload)
        
        return {
            "status": "reloading",
            "message": "Model reload started in background",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in /reload-model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api-docs")
def get_api_documentation():
    """Get API documentation and usage examples."""
    return {
        "title": "Research Paper Analysis & Classification API",
        "version": "1.0.0",
        "description": "API for classifying research paper abstracts and extracting diseases",
        "endpoints": {
            "GET /": "Health check and model information",
            "GET /health": "Health check endpoint",
            "POST /classify": "Classify a single abstract",
            "POST /batch-classify": "Classify multiple abstracts",
            "POST /extract-diseases": "Extract diseases from abstract",
            "GET /model-info": "Get detailed model information",
            "POST /reload-model": "Reload the model"
        },
        "example_requests": {
            "single_classification": {
                "url": "/classify",
                "method": "POST",
                "body": {
                    "pubmed_id": "PMID123456",
                    "abstract": "This study investigates lung cancer progression..."
                }
            },
            "batch_classification": {
                "url": "/batch-classify",
                "method": "POST",
                "body": {
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
                }
            }
        },
        "expected_responses": {
            "classification": {
                "abstract_id": "PMID123456",
                "classification": {
                    "predicted_labels": ["Cancer"],
                    "confidence_scores": {
                        "Cancer": 0.92,
                        "Non-Cancer": 0.08
                    }
                },
                "disease_extraction": {
                    "extracted_diseases": ["lung cancer"]
                }
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 