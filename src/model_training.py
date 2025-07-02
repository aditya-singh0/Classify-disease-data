"""
Model training module for fine-tuning language models on research paper abstracts.
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning of various models.
Supports Gemma, Phi, and other small language models as required by the assignment.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
    Trainer, DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import logging
import argparse
import os
import json
from typing import Dict, Any, Tuple, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for different model types."""
    
    MODELS = {
        "distilbert-base-uncased": {
            "max_length": 512,
            "target_modules": ["q_lin", "v_lin", "k_lin", "out_lin", "lin1", "lin2"],
            "lora_r": 4,
            "lora_alpha": 8,
            "learning_rate": 2e-5
        },
        "microsoft/phi-2": {
            "max_length": 2048,
            "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-5
        },
        "microsoft/phi-1_5": {
            "max_length": 2048,
            "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-5
        },
        "google/gemma-2b": {
            "max_length": 2048,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-5
        },
        "microsoft/DialoGPT-medium": {
            "max_length": 1024,
            "target_modules": ["c_attn", "c_proj", "c_fc", "c_proj"],
            "lora_r": 8,
            "lora_alpha": 16,
            "learning_rate": 3e-5
        }
    }


class CancerClassificationTrainer:
    """Trainer for cancer/non-cancer classification using LoRA fine-tuning."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        max_length: Optional[int] = None,
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: float = 0.1,
        learning_rate: Optional[float] = None
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.lora_dropout = lora_dropout
        
        # Get model-specific configuration
        model_config = ModelConfig.MODELS.get(model_name, ModelConfig.MODELS["microsoft/phi-2"])
        self.max_length = max_length or model_config["max_length"]
        self.lora_r = lora_r or model_config["lora_r"]
        self.lora_alpha = lora_alpha or model_config["lora_alpha"]
        self.learning_rate = learning_rate or model_config["learning_rate"]
        self.target_modules = model_config["target_modules"]
        
        # Force CPU usage to avoid memory issues
        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model: {model_name}")
        logger.info(f"LoRA config: r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.baseline_metrics = None
        self.finetuned_metrics = None
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to a simpler model if the primary one fails
            logger.info("Falling back to distilbert-base-uncased")
            self.model_name = "distilbert-base-uncased"
            self.load_model_and_tokenizer()
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize the input examples."""
        # Use cleaned_abstract if available, otherwise use abstract
        texts = examples.get('cleaned_abstract', examples.get('abstract', []))
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Add labels
        tokenized['labels'] = examples['label']
        
        return tokenized
    
    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Calculate per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'cancer_precision': class_precision[1] if len(class_precision) > 1 else 0,
            'cancer_recall': class_recall[1] if len(class_recall) > 1 else 0,
            'cancer_f1': class_f1[1] if len(class_f1) > 1 else 0,
            'non_cancer_precision': class_precision[0] if len(class_precision) > 0 else 0,
            'non_cancer_recall': class_recall[0] if len(class_recall) > 0 else 0,
            'non_cancer_f1': class_f1[0] if len(class_f1) > 0 else 0
        }
    
    def prepare_datasets(self, train_dataset: Dataset, eval_dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Prepare datasets for training."""
        logger.info("Tokenizing datasets...")
        
        # Tokenize train dataset
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # Tokenize eval dataset
        eval_dataset = eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        logger.info("Datasets tokenized successfully")
        return train_dataset, eval_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: Optional[float] = None,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100
    ):
        """Train the model using LoRA fine-tuning."""
        logger.info("Starting model training...")
        
        # Use instance learning rate if not provided
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets(train_dataset, eval_dataset)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Training arguments - compatible with older transformers versions
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            report_to=None,  # Disable wandb/tensorboard
            remove_unused_columns=False,
            push_to_hub=False,
            dataloader_pin_memory=False
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save the model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Log training results
        logger.info(f"Training completed. Loss: {train_result.training_loss}")
        
        return train_result
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        logger.info("Evaluating model...")
        
        # Prepare dataset
        eval_dataset, _ = self.prepare_datasets(eval_dataset, eval_dataset)
        
        # Evaluate
        results = self.trainer.evaluate(eval_dataset)
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def get_baseline_performance(self, test_dataset: Dataset) -> Dict[str, float]:
        """Get baseline performance without fine-tuning."""
        logger.info("Computing baseline performance...")
        
        # Use the base model without LoRA for baseline
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Prepare dataset
        test_dataset, _ = self.prepare_datasets(test_dataset, test_dataset)
        
        # Create temporary trainer for baseline
        temp_trainer = Trainer(
            model=base_model,
            args=TrainingArguments(output_dir="./temp", remove_unused_columns=False),
            compute_metrics=self.compute_metrics
        )
        
        # Evaluate baseline
        baseline_results = temp_trainer.evaluate(test_dataset)
        
        # Clean up
        del base_model
        del temp_trainer
        
        self.baseline_metrics = baseline_results
        logger.info(f"Baseline performance: {baseline_results}")
        
        return baseline_results
    
    def compare_performance(self, finetuned_results: Dict[str, float]) -> Dict[str, Any]:
        """Compare baseline vs fine-tuned performance."""
        if self.baseline_metrics is None:
            logger.warning("Baseline metrics not available. Run get_baseline_performance first.")
            return {}
        
        comparison = {
            "baseline": self.baseline_metrics,
            "finetuned": finetuned_results,
            "improvements": {}
        }
        
        # Calculate improvements
        for metric in ["accuracy", "precision", "recall", "f1"]:
            if metric in self.baseline_metrics and metric in finetuned_results:
                baseline_val = self.baseline_metrics[metric]
                finetuned_val = finetuned_results[metric]
                improvement = finetuned_val - baseline_val
                improvement_pct = (improvement / baseline_val) * 100 if baseline_val > 0 else 0
                
                comparison["improvements"][metric] = {
                    "absolute": improvement,
                    "percentage": improvement_pct
                }
        
        logger.info("Performance comparison completed")
        return comparison
    
    def save_performance_report(self, comparison: Dict[str, Any], output_dir: str):
        """Save performance report and visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison as JSON
        report_path = os.path.join(output_dir, "performance_report.json")
        with open(report_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Create confusion matrix visualization
        if "finetuned" in comparison and "confusion_matrix" in comparison["finetuned"]:
            self._plot_confusion_matrix(
                comparison["finetuned"]["confusion_matrix"],
                os.path.join(output_dir, "confusion_matrix.png")
            )
        
        # Create performance comparison chart
        self._plot_performance_comparison(comparison, os.path.join(output_dir, "performance_comparison.png"))
        
        logger.info(f"Performance report saved to {output_dir}")
    
    def _plot_confusion_matrix(self, cm: List[List[int]], save_path: str):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Predicted Non-Cancer', 'Predicted Cancer'],
            yticklabels=['Actual Non-Cancer', 'Actual Cancer']
        )
        plt.title('Confusion Matrix - Fine-tuned Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, comparison: Dict[str, Any], save_path: str):
        """Plot performance comparison between baseline and fine-tuned models."""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        baseline_values = [comparison['baseline'].get(m, 0) for m in metrics]
        finetuned_values = [comparison['finetuned'].get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
        plt.bar(x + width/2, finetuned_values, width, label='Fine-tuned', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Performance Comparison: Baseline vs Fine-tuned Model')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (baseline, finetuned) in enumerate(zip(baseline_values, finetuned_values)):
            plt.text(i - width/2, baseline + 0.01, f'{baseline:.3f}', ha='center', va='bottom')
            plt.text(i + width/2, finetuned + 0.01, f'{finetuned:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train cancer classification model")
    parser.add_argument("--model", default="microsoft/phi-2", help="Model to use")
    parser.add_argument("--train_data", required=True, help="Path to training data")
    parser.add_argument("--eval_data", required=True, help="Path to evaluation data")
    parser.add_argument("--output_dir", default="./models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CancerClassificationTrainer(model_name=args.model)
    trainer.load_model_and_tokenizer()
    
    # Load datasets
    from datasets import load_from_disk
    train_dataset = load_from_disk(args.train_data)
    eval_dataset = load_from_disk(args.eval_data)
    
    # Create output directory
    model_name = args.model.replace("/", "_")
    output_dir = os.path.join(args.output_dir, f"{model_name}-lora-cancer")
    
    # Train model
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Evaluate
    results = trainer.evaluate(eval_dataset)
    print(f"Final evaluation results: {results}")


if __name__ == "__main__":
    main() 