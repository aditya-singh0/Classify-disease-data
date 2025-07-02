"""
Classification module for research paper abstracts.
Handles model inference, disease extraction, and structured output generation.
Integrates with LangChain for enhanced prompt-based querying.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import re
from dataclasses import dataclass
from src.data_processing import AbstractProcessor

# LangChain imports for enhanced functionality
try:
    from langchain.llms import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import BaseOutputParser
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Enhanced features will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Structured result for classification output."""
    pubmed_id: str
    predicted_labels: List[str]
    confidence_scores: Dict[str, float]
    extracted_diseases: List[str]
    raw_probabilities: np.ndarray
    langchain_analysis: Optional[Dict[str, Any]] = None


class DiseaseExtractor:
    """Enhanced disease extraction using LangChain and rule-based methods."""
    
    def __init__(self, use_langchain: bool = False):
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        self.abstract_processor = AbstractProcessor()
        
        if self.use_langchain:
            self._setup_langchain()
    
    def _setup_langchain(self):
        """Setup LangChain for disease extraction."""
        try:
            # Disease extraction prompt
            disease_prompt = PromptTemplate(
                input_variables=["abstract"],
                template="""
                Analyze the following medical abstract and extract all mentioned diseases, conditions, and medical terms.
                
                Abstract: {abstract}
                
                Please provide a JSON response with the following structure:
                {{
                    "diseases": ["disease1", "disease2"],
                    "cancer_types": ["cancer_type1", "cancer_type2"],
                    "medical_conditions": ["condition1", "condition2"],
                    "confidence": 0.95
                }}
                
                Focus on:
                - Cancer types (lung cancer, breast cancer, etc.)
                - Other diseases (diabetes, hypertension, etc.)
                - Medical conditions and syndromes
                - Be specific and accurate
                
                Response:
                """
            )
            
            # For now, we'll use a simple approach since we don't have a local LLM
            # In production, you would load a local model here
            self.disease_chain = None
            logger.info("LangChain setup completed (using rule-based fallback)")
            
        except Exception as e:
            logger.warning(f"LangChain setup failed: {e}")
            self.use_langchain = False
    
    def extract_diseases(self, text: str) -> Dict[str, Any]:
        """Extract diseases using both rule-based and LangChain methods."""
        # Rule-based extraction
        rule_based_diseases = self.abstract_processor.extract_diseases(text)
        
        result = {
            "diseases": rule_based_diseases,
            "cancer_types": [d for d in rule_based_diseases if "cancer" in d.lower()],
            "medical_conditions": [d for d in rule_based_diseases if "cancer" not in d.lower()],
            "confidence": 0.85,
            "extraction_method": "rule_based"
        }
        
        # LangChain extraction if available
        if self.use_langchain and self.disease_chain is not None:
            try:
                langchain_result = self.disease_chain.run(abstract=text)
                # Parse LangChain result and merge with rule-based
                # This would be implemented based on the actual LLM response
                result["langchain_analysis"] = langchain_result
                result["extraction_method"] = "hybrid"
            except Exception as e:
                logger.warning(f"LangChain extraction failed: {e}")
        
        return result


class CancerClassifier:
    """Cancer/non-cancer classifier with enhanced disease extraction."""
    
    def __init__(
        self,
        model_path: str,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        device: Optional[str] = None,
        use_langchain: bool = False
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_langchain = use_langchain
        
        self.tokenizer = None
        self.model = None
        self.abstract_processor = AbstractProcessor()
        self.disease_extractor = DiseaseExtractor(use_langchain=use_langchain)
        
        self.label_map = {0: "Non-Cancer", 1: "Cancer"}
        self.id2label = {0: "Non-Cancer", 1: "Cancer"}
        self.label2id = {"Non-Cancer": 0, "Cancer": 1}
        
        logger.info(f"Initializing classifier with device: {self.device}")
        logger.info(f"LangChain enabled: {self.use_langchain}")
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        try:
            logger.info(f"Loading tokenizer from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Loading base model from {self.model_name}")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            logger.info(f"Loading LoRA adapter from {self.model_path}")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text."""
        return self.abstract_processor.clean_abstract(text)
    
    def extract_diseases(self, text: str) -> Dict[str, Any]:
        """Extract diseases from text using enhanced extractor."""
        return self.disease_extractor.extract_diseases(text)
    
    def predict_single(self, abstract: str, pubmed_id: str = "unknown") -> ClassificationResult:
        """Predict classification for a single abstract."""
        # Preprocess text
        cleaned_text = self.preprocess_text(abstract)
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
        
        # Get predictions
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence_scores = {
            self.label_map[0]: probabilities[0][0].item(),
            self.label_map[1]: probabilities[0][1].item()
        }
        
        # Get predicted labels (multi-label approach)
        predicted_labels = []
        threshold = 0.5
        for label, score in confidence_scores.items():
            if score > threshold:
                predicted_labels.append(label)
        
        # If no labels above threshold, use the highest scoring one
        if not predicted_labels:
            predicted_labels = [self.label_map[predicted_class]]
        
        # Extract diseases with enhanced method
        disease_extraction = self.extract_diseases(cleaned_text)
        extracted_diseases = disease_extraction.get("diseases", [])
        
        # LangChain analysis if available
        langchain_analysis = None
        if self.use_langchain:
            langchain_analysis = disease_extraction.get("langchain_analysis")
        
        return ClassificationResult(
            pubmed_id=pubmed_id,
            predicted_labels=predicted_labels,
            confidence_scores=confidence_scores,
            extracted_diseases=extracted_diseases,
            raw_probabilities=probabilities.cpu().numpy(),
            langchain_analysis=langchain_analysis
        )
    
    def predict_batch(self, abstracts: List[Dict[str, str]]) -> List[ClassificationResult]:
        """Predict classification for multiple abstracts."""
        results = []
        
        for abstract_data in abstracts:
            pubmed_id = abstract_data.get('pubmed_id', 'unknown')
            abstract = abstract_data.get('abstract', '')
            
            try:
                result = self.predict_single(abstract, pubmed_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing abstract {pubmed_id}: {e}")
                # Create error result
                error_result = ClassificationResult(
                    pubmed_id=pubmed_id,
                    predicted_labels=["Error"],
                    confidence_scores={"Error": 0.0},
                    extracted_diseases=[],
                    raw_probabilities=np.array([[0.0, 0.0]])
                )
                results.append(error_result)
        
        return results
    
    def format_output(self, result: ClassificationResult) -> Dict[str, Any]:
        """Format result for API output according to assignment requirements."""
        return {
            "abstract_id": result.pubmed_id,
            "classification": {
                "predicted_labels": result.predicted_labels,
                "confidence_scores": result.confidence_scores
            },
            "disease_extraction": {
                "extracted_diseases": result.extracted_diseases
            },
            "langchain_analysis": result.langchain_analysis if result.langchain_analysis else None
        }
    
    def format_batch_output(self, results: List[ClassificationResult]) -> Dict[str, Any]:
        """Format batch results for API output."""
        return {
            "results": [self.format_output(result) for result in results],
            "total_processed": len(results),
            "successful_predictions": len([r for r in results if "Error" not in r.predicted_labels]),
            "summary": {
                "cancer_count": len([r for r in results if "Cancer" in r.predicted_labels]),
                "non_cancer_count": len([r for r in results if "Non-Cancer" in r.predicted_labels]),
                "average_confidence": np.mean([
                    max(r.confidence_scores.values()) for r in results if "Error" not in r.predicted_labels
                ])
            }
        }


class BaselineClassifier:
    """Baseline classifier using rule-based approach."""
    
    def __init__(self, use_langchain: bool = False):
        self.abstract_processor = AbstractProcessor()
        self.disease_extractor = DiseaseExtractor(use_langchain=use_langchain)
        self.use_langchain = use_langchain
    
    def predict_single(self, abstract: str, pubmed_id: str = "unknown") -> ClassificationResult:
        """Predict using rule-based approach."""
        # Clean text
        cleaned_text = self.abstract_processor.clean_abstract(abstract)
        
        # Rule-based classification
        is_cancer = self.abstract_processor.is_cancer_related(cleaned_text)
        
        # Set confidence scores based on keyword presence
        cancer_keywords = len([k for k in self.abstract_processor.cancer_keywords if k in cleaned_text.lower()])
        confidence = min(0.9, 0.5 + (cancer_keywords * 0.1))
        
        predicted_labels = ["Cancer"] if is_cancer else ["Non-Cancer"]
        confidence_scores = {
            "Cancer": confidence if is_cancer else 1 - confidence,
            "Non-Cancer": 1 - confidence if is_cancer else confidence
        }
        
        # Extract diseases
        disease_extraction = self.disease_extractor.extract_diseases(cleaned_text)
        extracted_diseases = disease_extraction.get("diseases", [])
        
        # LangChain analysis if available
        langchain_analysis = None
        if self.use_langchain:
            langchain_analysis = disease_extraction.get("langchain_analysis")
        
        return ClassificationResult(
            pubmed_id=pubmed_id,
            predicted_labels=predicted_labels,
            confidence_scores=confidence_scores,
            extracted_diseases=extracted_diseases,
            raw_probabilities=np.array([[confidence_scores["Non-Cancer"], confidence_scores["Cancer"]]]),
            langchain_analysis=langchain_analysis
        )
    
    def predict_batch(self, abstracts: List[Dict[str, str]]) -> List[ClassificationResult]:
        """Predict for multiple abstracts using baseline approach."""
        results = []
        
        for abstract_data in abstracts:
            pubmed_id = abstract_data.get('pubmed_id', 'unknown')
            abstract = abstract_data.get('abstract', '')
            
            try:
                result = self.predict_single(abstract, pubmed_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing abstract {pubmed_id}: {e}")
                # Create error result
                error_result = ClassificationResult(
                    pubmed_id=pubmed_id,
                    predicted_labels=["Error"],
                    confidence_scores={"Error": 0.0},
                    extracted_diseases=[],
                    raw_probabilities=np.array([[0.0, 0.0]])
                )
                results.append(error_result)
        
        return results
    
    def format_output(self, result: ClassificationResult) -> Dict[str, Any]:
        """Format result for API output."""
        return {
            "abstract_id": result.pubmed_id,
            "classification": {
                "predicted_labels": result.predicted_labels,
                "confidence_scores": result.confidence_scores
            },
            "disease_extraction": {
                "extracted_diseases": result.extracted_diseases
            },
            "langchain_analysis": result.langchain_analysis if result.langchain_analysis else None
        }
    
    def format_batch_output(self, results: List[ClassificationResult]) -> Dict[str, Any]:
        """Format batch results for API output."""
        return {
            "results": [self.format_output(result) for result in results],
            "total_processed": len(results),
            "successful_predictions": len([r for r in results if "Error" not in r.predicted_labels]),
            "summary": {
                "cancer_count": len([r for r in results if "Cancer" in r.predicted_labels]),
                "non_cancer_count": len([r for r in results if "Non-Cancer" in r.predicted_labels]),
                "average_confidence": np.mean([
                    max(r.confidence_scores.values()) for r in results if "Error" not in r.predicted_labels
                ])
            }
        }


def load_classifier(model_path: Optional[str], use_baseline: bool = False, use_langchain: bool = False, model_name: str = "distilbert-base-uncased") -> Any:
    """Load classifier (fine-tuned or baseline)."""
    if use_baseline or model_path is None:
        logger.info("Loading baseline classifier")
        return BaselineClassifier(use_langchain=use_langchain)
    else:
        logger.info(f"Loading fine-tuned classifier from {model_path}")
        return CancerClassifier(model_path, model_name=model_name, use_langchain=use_langchain)


def create_sample_predictions():
    """Create sample predictions for demonstration."""
    # Sample abstracts
    sample_abstracts = [
        {
            "pubmed_id": "PMID123456",
            "abstract": "This study investigates the molecular mechanisms of lung cancer progression and identifies novel therapeutic targets for non-small cell lung cancer treatment."
        },
        {
            "pubmed_id": "PMID789012",
            "abstract": "The effects of exercise on cardiovascular health and blood pressure regulation in elderly patients with hypertension."
        },
        {
            "pubmed_id": "PMID345678",
            "abstract": "Breast cancer screening methods and early detection strategies in high-risk populations with BRCA mutations."
        }
    ]
    
    # Load baseline classifier
    classifier = load_classifier(None, use_baseline=True, use_langchain=False)
    
    # Make predictions
    results = classifier.predict_batch(sample_abstracts)
    
    # Format output
    output = classifier.format_batch_output(results)
    
    print("Sample Predictions:")
    print(json.dumps(output, indent=2))
    
    return output


if __name__ == "__main__":
    create_sample_predictions() 