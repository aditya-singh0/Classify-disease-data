"""
Evaluation module for comparing baseline vs fine-tuned model performance.
Provides comprehensive analysis including confusion matrices, metrics, and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_curve
)
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
import os
from datetime import datetime
from datasets import Dataset
from src.classification import load_classifier, ClassificationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""
    
    def __init__(self, test_dataset: Dataset):
        self.test_dataset = test_dataset
        self.baseline_results = None
        self.finetuned_results = None
        self.comparison_results = None
    
    def evaluate_baseline(self) -> Dict[str, Any]:
        """Evaluate baseline model performance."""
        logger.info("Evaluating baseline model...")
        
        # Load baseline classifier
        baseline_classifier = load_classifier(None, use_baseline=True)
        
        # Prepare test data
        test_data = []
        true_labels = []
        
        for i, example in enumerate(self.test_dataset):
            test_data.append({
                'pubmed_id': example.get('pubmed_id', f'test_{i}'),
                'abstract': example.get('cleaned_abstract', example.get('abstract', ''))
            })
            true_labels.append(example.get('label', 0))
        
        # Get predictions
        predictions = baseline_classifier.predict_batch(test_data)
        
        # Extract prediction results
        pred_labels = []
        confidence_scores = []
        
        for pred in predictions:
            if "Cancer" in pred.predicted_labels:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
            confidence_scores.append(max(pred.confidence_scores.values()))
        
        # Calculate metrics
        metrics = self._calculate_metrics(true_labels, pred_labels, confidence_scores)
        
        self.baseline_results = {
            'metrics': metrics,
            'predictions': pred_labels,
            'true_labels': true_labels,
            'confidence_scores': confidence_scores
        }
        
        logger.info(f"Baseline evaluation completed. Accuracy: {metrics['accuracy']:.3f}")
        return self.baseline_results
    
    def evaluate_finetuned(self, model_path: str) -> Dict[str, Any]:
        """Evaluate fine-tuned model performance."""
        logger.info(f"Evaluating fine-tuned model from {model_path}...")
        
        # Load fine-tuned classifier with correct model name
        finetuned_classifier = load_classifier(model_path, use_baseline=False, model_name="distilbert-base-uncased")
        
        # Prepare test data
        test_data = []
        true_labels = []
        
        for i, example in enumerate(self.test_dataset):
            test_data.append({
                'pubmed_id': example.get('pubmed_id', f'test_{i}'),
                'abstract': example.get('cleaned_abstract', example.get('abstract', ''))
            })
            true_labels.append(example.get('label', 0))
        
        # Get predictions
        predictions = finetuned_classifier.predict_batch(test_data)
        
        # Extract prediction results
        pred_labels = []
        confidence_scores = []
        
        for pred in predictions:
            if "Cancer" in pred.predicted_labels:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
            confidence_scores.append(max(pred.confidence_scores.values()))
        
        # Calculate metrics
        metrics = self._calculate_metrics(true_labels, pred_labels, confidence_scores)
        
        self.finetuned_results = {
            'metrics': metrics,
            'predictions': pred_labels,
            'true_labels': true_labels,
            'confidence_scores': confidence_scores
        }
        
        logger.info(f"Fine-tuned evaluation completed. Accuracy: {metrics['accuracy']:.3f}")
        return self.finetuned_results
    
    def _calculate_metrics(self, true_labels: List[int], pred_labels: List[int], confidence_scores: List[float]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        # Basic metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
        
        # Per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # ROC and PR curves
        fpr, tpr, _ = roc_curve(true_labels, confidence_scores)
        roc_auc = auc(fpr, tpr)
        
        pr_precision, pr_recall, _ = precision_recall_curve(true_labels, confidence_scores)
        pr_auc = auc(pr_recall, pr_precision)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'cancer_precision': class_precision[1] if len(class_precision) > 1 else 0,
            'cancer_recall': class_recall[1] if len(class_recall) > 1 else 0,
            'cancer_f1': class_f1[1] if len(class_f1) > 1 else 0,
            'non_cancer_precision': class_precision[0] if len(class_precision) > 0 else 0,
            'non_cancer_recall': class_recall[0] if len(class_recall) > 0 else 0,
            'non_cancer_f1': class_f1[0] if len(class_f1) > 0 else 0,
            'confusion_matrix': cm.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'pr_precision': pr_precision.tolist(),
            'pr_recall': pr_recall.tolist()
        }
    
    def compare_models(self) -> Dict[str, Any]:
        """Compare baseline vs fine-tuned model performance."""
        if self.baseline_results is None or self.finetuned_results is None:
            raise ValueError("Both baseline and fine-tuned results must be available")
        
        baseline_metrics = self.baseline_results['metrics']
        finetuned_metrics = self.finetuned_results['metrics']
        
        comparison = {
            'baseline': baseline_metrics,
            'finetuned': finetuned_metrics,
            'improvements': {},
            'summary': {}
        }
        
        # Calculate improvements
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
            if metric in baseline_metrics and metric in finetuned_metrics:
                baseline_val = baseline_metrics[metric]
                finetuned_val = finetuned_metrics[metric]
                improvement = finetuned_val - baseline_val
                improvement_pct = (improvement / baseline_val) * 100 if baseline_val > 0 else 0
                
                comparison['improvements'][metric] = {
                    'absolute': improvement,
                    'percentage': improvement_pct
                }
        
        # Create summary
        comparison['summary'] = {
            'best_model': 'fine-tuned' if finetuned_metrics['f1'] > baseline_metrics['f1'] else 'baseline',
            'accuracy_improvement': comparison['improvements'].get('accuracy', {}).get('percentage', 0),
            'f1_improvement': comparison['improvements'].get('f1', {}).get('percentage', 0),
            'overall_improvement': np.mean([
                comparison['improvements'].get(m, {}).get('percentage', 0) 
                for m in ['accuracy', 'precision', 'recall', 'f1']
            ])
        }
        
        self.comparison_results = comparison
        logger.info("Model comparison completed")
        return comparison
    
    def generate_report(self, output_dir: str):
        """Generate comprehensive evaluation report."""
        if self.comparison_results is None:
            raise ValueError("Comparison results not available. Run compare_models() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON report
        report_path = os.path.join(output_dir, "evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.comparison_results, f, indent=2)
        
        # Generate visualizations
        self._plot_confusion_matrices(output_dir)
        self._plot_performance_comparison(output_dir)
        self._plot_roc_curves(output_dir)
        self._plot_precision_recall_curves(output_dir)
        
        # Generate text report
        self._generate_text_report(output_dir)
        
        logger.info(f"Evaluation report saved to {output_dir}")
    
    def _plot_confusion_matrices(self, output_dir: str):
        """Plot confusion matrices for both models."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Baseline confusion matrix
        baseline_cm = np.array(self.comparison_results['baseline']['confusion_matrix'])
        sns.heatmap(
            baseline_cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Predicted Non-Cancer', 'Predicted Cancer'],
            yticklabels=['Actual Non-Cancer', 'Actual Cancer'],
            ax=ax1
        )
        ax1.set_title('Baseline Model Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Fine-tuned confusion matrix
        finetuned_cm = np.array(self.comparison_results['finetuned']['confusion_matrix'])
        sns.heatmap(
            finetuned_cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Predicted Non-Cancer', 'Predicted Cancer'],
            yticklabels=['Actual Non-Cancer', 'Actual Cancer'],
            ax=ax2
        )
        ax2.set_title('Fine-tuned Model Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, output_dir: str):
        """Plot performance comparison between models."""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        
        baseline_values = [self.comparison_results['baseline'].get(m, 0) for m in metrics]
        finetuned_values = [self.comparison_results['finetuned'].get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
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
        plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, output_dir: str):
        """Plot ROC curves for both models."""
        plt.figure(figsize=(10, 6))
        
        # Baseline ROC
        baseline_fpr = self.comparison_results['baseline']['fpr']
        baseline_tpr = self.comparison_results['baseline']['tpr']
        baseline_auc = self.comparison_results['baseline']['roc_auc']
        plt.plot(baseline_fpr, baseline_tpr, label=f'Baseline (AUC = {baseline_auc:.3f})')
        
        # Fine-tuned ROC
        finetuned_fpr = self.comparison_results['finetuned']['fpr']
        finetuned_tpr = self.comparison_results['finetuned']['tpr']
        finetuned_auc = self.comparison_results['finetuned']['roc_auc']
        plt.plot(finetuned_fpr, finetuned_tpr, label=f'Fine-tuned (AUC = {finetuned_auc:.3f})')
        
        # Random classifier
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves: Baseline vs Fine-tuned Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curves(self, output_dir: str):
        """Plot Precision-Recall curves for both models."""
        plt.figure(figsize=(10, 6))
        
        # Baseline PR
        baseline_recall = self.comparison_results['baseline']['pr_recall']
        baseline_precision = self.comparison_results['baseline']['pr_precision']
        baseline_auc = self.comparison_results['baseline']['pr_auc']
        plt.plot(baseline_recall, baseline_precision, label=f'Baseline (AUC = {baseline_auc:.3f})')
        
        # Fine-tuned PR
        finetuned_recall = self.comparison_results['finetuned']['pr_recall']
        finetuned_precision = self.comparison_results['finetuned']['pr_precision']
        finetuned_auc = self.comparison_results['finetuned']['pr_auc']
        plt.plot(finetuned_recall, finetuned_precision, label=f'Fine-tuned (AUC = {finetuned_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves: Baseline vs Fine-tuned Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, output_dir: str):
        """Generate text-based evaluation report."""
        report_lines = [
            "# Research Paper Analysis & Classification Pipeline - Evaluation Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Performance Summary",
            "",
            "### Baseline Model Performance:",
            f"- Accuracy: {self.comparison_results['baseline']['accuracy']:.3f}",
            f"- Precision: {self.comparison_results['baseline']['precision']:.3f}",
            f"- Recall: {self.comparison_results['baseline']['recall']:.3f}",
            f"- F1-Score: {self.comparison_results['baseline']['f1']:.3f}",
            f"- ROC AUC: {self.comparison_results['baseline']['roc_auc']:.3f}",
            "",
            "### Fine-tuned Model Performance:",
            f"- Accuracy: {self.comparison_results['finetuned']['accuracy']:.3f}",
            f"- Precision: {self.comparison_results['finetuned']['precision']:.3f}",
            f"- Recall: {self.comparison_results['finetuned']['recall']:.3f}",
            f"- F1-Score: {self.comparison_results['finetuned']['f1']:.3f}",
            f"- ROC AUC: {self.comparison_results['finetuned']['roc_auc']:.3f}",
            "",
            "## Performance Improvements",
            ""
        ]
        
        for metric, improvement in self.comparison_results['improvements'].items():
            report_lines.append(f"- {metric.title()}: {improvement['absolute']:.3f} ({improvement['percentage']:.1f}%)")
        
        report_lines.extend([
            "",
            "## Confusion Matrices",
            "",
            "### Baseline Model:",
            "```",
            str(np.array(self.comparison_results['baseline']['confusion_matrix'])),
            "```",
            "",
            "### Fine-tuned Model:",
            "```",
            str(np.array(self.comparison_results['finetuned']['confusion_matrix'])),
            "```",
            "",
            "## Conclusion",
            f"- Best performing model: {self.comparison_results['summary']['best_model']}",
            f"- Overall improvement: {self.comparison_results['summary']['overall_improvement']:.1f}%",
            f"- Fine-tuning effectiveness: {'Effective' if self.comparison_results['summary']['f1_improvement'] > 0 else 'Ineffective'}",
            ""
        ])
        
        # Save text report
        report_path = os.path.join(output_dir, "evaluation_report.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))


def run_evaluation(test_dataset_path: str, model_path: Optional[str] = None, output_dir: str = "./evaluation_results"):
    """Run complete evaluation pipeline."""
    logger.info("Starting evaluation pipeline...")
    
    # Load test dataset
    from datasets import load_from_disk
    test_dataset = load_from_disk(test_dataset_path)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(test_dataset)
    
    # Evaluate baseline
    baseline_results = evaluator.evaluate_baseline()
    
    # Evaluate fine-tuned model if available
    if model_path and os.path.exists(model_path):
        finetuned_results = evaluator.evaluate_finetuned(model_path)
        
        # Compare models
        comparison = evaluator.compare_models()
        
        # Generate report
        evaluator.generate_report(output_dir)
        
        logger.info("Evaluation completed successfully!")
        return comparison
    else:
        logger.warning("Fine-tuned model not found. Only baseline evaluation completed.")
        return baseline_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--test_data", required=True, help="Path to test dataset")
    parser.add_argument("--model_path", help="Path to fine-tuned model")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    run_evaluation(args.test_data, args.model_path, args.output_dir) 