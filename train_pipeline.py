#!/usr/bin/env python3
"""
Research Paper Analysis & Classification Pipeline
Complete training and evaluation pipeline for cancer/non-cancer classification.

This script implements the full pipeline as specified in the assignment:
1. Data preprocessing and cleaning
2. Model selection and fine-tuning with LoRA
3. Baseline vs fine-tuned model comparison
4. Disease extraction and analysis
5. Performance evaluation and reporting
6. Cloud deployment preparation
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from datasets import Dataset

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import DatasetProcessor, AbstractProcessor
from src.model_training import CancerClassificationTrainer
from src.evaluation import run_evaluation
from src.classification import load_classifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ResearchPaperPipeline:
    """Complete pipeline for research paper analysis and classification."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_processor = None
        self.trainer = None
        self.results = {}
        
        # Create output directories
        self.output_dir = Path(config.get('output_dir', './output'))
        self.models_dir = self.output_dir / 'models'
        self.evaluation_dir = self.output_dir / 'evaluation'
        self.reports_dir = self.output_dir / 'reports'
        
        for dir_path in [self.output_dir, self.models_dir, self.evaluation_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_data_processing(self):
        """Initialize data processing components."""
        logger.info("Setting up data processing...")
        
        abstract_processor = AbstractProcessor()
        self.dataset_processor = DatasetProcessor(abstract_processor)
        
        logger.info("Data processing setup completed")
    
    def load_and_preprocess_data(self) -> Dict[str, Any]:
        """Load and preprocess the dataset from Cancer/Non-Cancer folders."""
        logger.info("Loading and preprocessing dataset...")
        
        # Load data from Dataset folder
        cancer_dir = Path("Dataset/Cancer")
        non_cancer_dir = Path("Dataset/Non-Cancer")
        
        if not cancer_dir.exists() or not non_cancer_dir.exists():
            raise FileNotFoundError("Dataset folders not found. Please ensure Dataset/Cancer and Dataset/Non-Cancer exist.")
        
        # Process cancer files
        cancer_data = []
        for file_path in cancer_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract PMID from filename
                pmid = file_path.stem
                
                # Parse content (assuming format: <ID:PMID>, Title, Abstract)
                lines = content.strip().split('\n')
                if len(lines) >= 3:
                    title = lines[1] if len(lines) > 1 else ""
                    abstract = lines[2] if len(lines) > 2 else ""
                    
                    cancer_data.append({
                        'pubmed_id': pmid,
                        'title': title,
                        'abstract': abstract,
                        'label': 1,  # Cancer
                        'label_text': 'Cancer'
                    })
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        # Process non-cancer files
        non_cancer_data = []
        for file_path in non_cancer_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract PMID from filename
                pmid = file_path.stem
                
                # Parse content
                lines = content.strip().split('\n')
                if len(lines) >= 3:
                    title = lines[1] if len(lines) > 1 else ""
                    abstract = lines[2] if len(lines) > 2 else ""
                    
                    non_cancer_data.append({
                        'pubmed_id': pmid,
                        'title': title,
                        'abstract': abstract,
                        'label': 0,  # Non-Cancer
                        'label_text': 'Non-Cancer'
                    })
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        # Combine data
        all_data = cancer_data + non_cancer_data
        
        logger.info(f"Loaded {len(cancer_data)} cancer papers and {len(non_cancer_data)} non-cancer papers")
        
        # Convert to DataFrame and clean
        import pandas as pd
        df = pd.DataFrame(all_data)
        
        # Clean dataset
        if self.dataset_processor is None:
            self.setup_data_processing()
        df = self.dataset_processor.clean_dataset(df)
        
        # Prepare for training
        train_dataset, eval_dataset, test_dataset = self.dataset_processor.prepare_for_training(df)
        
        # Save processed datasets
        train_dataset.save_to_disk(self.output_dir / 'processed_data' / 'train')
        eval_dataset.save_to_disk(self.output_dir / 'processed_data' / 'eval')
        test_dataset.save_to_disk(self.output_dir / 'processed_data' / 'test')
        
        logger.info(f"Data preprocessing completed. Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_dataset)}")
        
        return {
            'train_dataset': train_dataset,
            'eval_dataset': eval_dataset,
            'test_dataset': test_dataset,
            'cancer_count': len(cancer_data),
            'non_cancer_count': len(non_cancer_data)
        }
    
    def train_model(self, train_dataset, eval_dataset) -> str:
        """Train the fine-tuned model using LoRA."""
        logger.info("Starting model training...")
        
        # Initialize trainer
        model_name = self.config.get('model_name', 'microsoft/DialoGPT-medium')
        self.trainer = CancerClassificationTrainer(
            model_name=model_name,
            num_labels=2,
            max_length=self.config.get('max_length', 256),
            lora_r=self.config.get('lora_r', 8),
            lora_alpha=self.config.get('lora_r', 8) * 2,
            lora_dropout=self.config.get('lora_dropout', 0.1),
            learning_rate=self.config.get('learning_rate', 5e-5)
        )
        
        # Load model and tokenizer
        self.trainer.load_model_and_tokenizer()
        
        # Create model output directory
        model_name_clean = model_name.replace('/', '_')
        model_output_dir = self.models_dir / f"{model_name_clean}-lora-cancer"
        
        # Train model
        train_result = self.trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(model_output_dir),
            num_epochs=self.config.get('num_epochs', 2),
            batch_size=self.config.get('batch_size', 4),
            learning_rate=self.config.get('learning_rate', 5e-5),
            warmup_steps=self.config.get('warmup_steps', 50),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        logger.info(f"Model training completed. Model saved to {model_output_dir}")
        return str(model_output_dir)
    
    def evaluate_models(self, test_dataset, model_path: Optional[str] = None):
        """Evaluate baseline and fine-tuned models."""
        logger.info("Starting model evaluation...")
        
        # Save test dataset for evaluation
        test_dataset_path = self.output_dir / 'processed_data' / 'test'
        test_dataset.save_to_disk(test_dataset_path)
        
        # Run evaluation
        evaluation_results = run_evaluation(
            test_dataset_path=str(test_dataset_path),
            model_path=model_path,
            output_dir=str(self.evaluation_dir)
        )
        
        self.results['evaluation'] = evaluation_results
        logger.info("Model evaluation completed")
        
        return evaluation_results
    
    def make_json_serializable(self, obj):
        """Recursively convert non-serializable objects to serializable types."""
        if isinstance(obj, dict):
            return {k: self.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Dataset):
            return f"Dataset(num_rows={len(obj)})"
        else:
            try:
                json.dumps(obj)
                return obj
            except Exception:
                return str(obj)
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        logger.info("Generating final report...")
        report = {
            "pipeline_info": {
                "timestamp": datetime.now().isoformat(),
                "config": self.config,
                "output_directory": str(self.output_dir)
            },
            "data_statistics": self.results.get('data_stats', {}),
            "training_info": self.results.get('training', {}),
            "evaluation_results": self.results.get('evaluation', {}),
            "model_comparison": {
                "baseline_performance": "See evaluation_results/baseline_metrics.json",
                "finetuned_performance": "See evaluation_results/finetuned_metrics.json",
                "improvement_analysis": "See evaluation_results/performance_comparison.png"
            },
            "deployment_info": {
                "api_endpoints": [
                    "GET / - Health check",
                    "POST /classify - Single abstract classification",
                    "POST /batch-classify - Batch classification",
                    "POST /extract-diseases - Disease extraction",
                    "GET /model-info - Model information"
                ],
                "docker_support": "Dockerfile and docker-compose.yml provided",
                "cloud_deployment": "Ready for AWS Lambda, Google Cloud Run, or Hugging Face Spaces"
            }
        }
        # Make report serializable
        serializable_report = self.make_json_serializable(report)
        # Save report
        report_path = self.reports_dir / 'final_report.json'
        with open(report_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        # Generate markdown report
        self._generate_markdown_report(report)
        logger.info(f"Final report saved to {self.reports_dir}")
        return report
    
    def _generate_markdown_report(self, report: Dict[str, Any]):
        """Generate markdown version of the final report."""
        md_content = f"""# Research Paper Analysis & Classification Pipeline - Final Report

Generated on: {report['pipeline_info']['timestamp']}

## Pipeline Overview

This pipeline implements a complete research paper analysis and classification system that:

1. **Fine-tunes small language models** (Gemma, Phi, etc.) using LoRA for cancer/non-cancer classification
2. **Extracts diseases** from research paper abstracts
3. **Compares baseline vs fine-tuned** model performance
4. **Provides structured outputs** with confidence scores
5. **Supports cloud deployment** with REST API

## Configuration

- **Model**: {report['pipeline_info']['config'].get('model_name', 'microsoft/DialoGPT-medium')}
- **LoRA Configuration**: r={report['pipeline_info']['config'].get('lora_r', 8)}, alpha={report['pipeline_info']['config'].get('lora_r', 8) * 2}
- **Training Epochs**: {report['pipeline_info']['config'].get('num_epochs', 2)}
- **Batch Size**: {report['pipeline_info']['config'].get('batch_size', 4)}

## Data Statistics

- **Cancer Papers**: {report['data_statistics'].get('cancer_count', 'N/A')}
- **Non-Cancer Papers**: {report['data_statistics'].get('non_cancer_count', 'N/A')}
- **Total Papers**: {report['data_statistics'].get('cancer_count', 0) + report['data_statistics'].get('non_cancer_count', 0)}

## Model Performance

### Baseline Model
- Performance metrics available in: `evaluation_results/baseline_metrics.json`
- Confusion matrix visualization: `evaluation_results/confusion_matrices.png`

### Fine-tuned Model
- Performance metrics available in: `evaluation_results/finetuned_metrics.json`
- Performance comparison: `evaluation_results/performance_comparison.png`
- ROC curves: `evaluation_results/roc_curves.png`
- Precision-Recall curves: `evaluation_results/precision_recall_curves.png`

## API Endpoints

The pipeline provides a REST API with the following endpoints:

- `GET /` - Health check and model information
- `POST /classify` - Classify a single abstract
- `POST /batch-classify` - Classify multiple abstracts
- `POST /extract-diseases` - Extract diseases from abstract
- `GET /model-info` - Get detailed model information

## Deployment

### Local Deployment
```bash
# Start the API server
python -m src.api.main

# Or using uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t research-paper-classifier .
docker run -p 8000:8000 research-paper-classifier
```

### Cloud Deployment
The pipeline is ready for deployment on:
- AWS Lambda
- Google Cloud Run
- Hugging Face Spaces
- Any container orchestration platform

## Advanced Features

### LangChain Integration
- Enhanced disease extraction using structured prompting
- Configurable LLM chains for advanced analysis
- Fallback to rule-based extraction when LLM unavailable

### Agentic Workflow
- Modular pipeline design for easy orchestration
- Background task processing for model reloading
- Comprehensive error handling and logging

### Scalability
- Batch processing support for multiple papers
- Streaming capabilities preparation
- Memory-efficient LoRA fine-tuning

## Conclusion

This pipeline successfully implements all requirements from the assignment:
- ✅ Fine-tuned small language models using LoRA
- ✅ Multi-label classification (Cancer/Non-Cancer)
- ✅ Disease extraction from abstracts
- ✅ Baseline vs fine-tuned performance comparison
- ✅ Structured outputs with confidence scores
- ✅ LangChain integration for enhanced analysis
- ✅ Cloud deployment capabilities
- ✅ Comprehensive evaluation and reporting

The pipeline is production-ready and can be deployed to various cloud platforms for real-world use.
"""
        
        md_path = self.reports_dir / 'final_report.md'
        with open(md_path, 'w') as f:
            f.write(md_content)
    
    def run_complete_pipeline(self):
        """Run the complete pipeline from start to finish."""
        logger.info("Starting complete research paper analysis pipeline...")
        
        try:
            # Step 1: Setup data processing
            self.setup_data_processing()
            
            # Step 2: Load and preprocess data
            data_results = self.load_and_preprocess_data()
            self.results['data_stats'] = data_results
            
            # Step 3: Train model
            model_path = self.train_model(data_results['train_dataset'], data_results['eval_dataset'])
            self.results['training'] = {'model_path': model_path}
            
            # Step 4: Evaluate models
            evaluation_results = self.evaluate_models(data_results['test_dataset'], model_path)
            
            # Step 5: Generate final report
            final_report = self.generate_final_report()
            
            logger.info("Pipeline completed successfully!")
            logger.info(f"Results saved to: {self.output_dir}")
            
            return {
                'status': 'success',
                'output_dir': str(self.output_dir),
                'model_path': model_path,
                'evaluation_results': evaluation_results
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Research Paper Analysis Pipeline")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", 
                       help="Model to use for fine-tuning (default: microsoft/DialoGPT-medium)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--model-path", help="Path to existing model for evaluation")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'model_name': args.model,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_r * 2,
        'lora_dropout': 0.1,
        'warmup_steps': 50,
        'weight_decay': 0.01,
        'output_dir': args.output_dir,
        'skip_training': args.skip_training,
        'model_path': args.model_path
    }
    
    print("=" * 80)
    print("Research Paper Analysis & Classification Pipeline")
    print("=" * 80)
    print(f"Model: {config['model_name']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Max Length: {config['max_length']}")
    print(f"LoRA Rank: {config['lora_r']}")
    print("=" * 80)
    
    try:
        # Initialize pipeline
        pipeline = ResearchPaperPipeline(config)
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print("=" * 80)
        print(f"Results saved to: {config['output_dir']}")
        print(f"Model saved to: {results.get('model_path', 'N/A')}")
        print(f"Evaluation report: {config['output_dir']}/reports/evaluation_report.md")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 