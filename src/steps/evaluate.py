"""
Model Evaluation Step for CIFAR-10 CNN Classifier

This module handles model evaluation with comprehensive metrics,
confusion matrix generation, and MLflow logging.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import mlflow


# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class ModelEvaluator:
    """
    Comprehensive model evaluator for classification tasks.
    
    Provides:
    - Classification metrics (accuracy, precision, recall, F1)
    - Confusion matrix visualization
    - Per-class performance analysis
    - MLflow logging
    """
    
    def __init__(
        self,
        class_names: List[str] = None,
        output_dir: str = "reports"
    ):
        """
        Initialize the evaluator.
        
        Args:
            class_names: List of class names
            output_dir: Directory to save evaluation reports
        """
        self.class_names = class_names or CLASS_NAMES
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(
        self,
        model: keras.Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            model: Trained Keras model
            X_test: Test images
            y_test: Test labels
            save_plots: Whether to save visualization plots
            
        Returns:
            Evaluation results dictionary
        """
        print("\n" + "="*60)
        print("Model Evaluation")
        print("="*60)
        
        # Get predictions
        print("\nGenerating predictions...")
        y_pred_probs = model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_probs)
        
        # Generate classification report
        class_report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print summary
        print(f"\nEvaluation Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
        
        # Print per-class results
        print(f"\nPer-class Performance:")
        for i, class_name in enumerate(self.class_names):
            class_metrics = class_report[class_name]
            print(f"  {class_name:12s}: P={class_metrics['precision']:.3f}, "
                  f"R={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}")
        
        results = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_probs.tolist()
        }
        
        # Save plots
        if save_plots:
            self._plot_confusion_matrix(cm, y_test, y_pred)
            self._plot_per_class_metrics(class_report)
            self._plot_prediction_samples(X_test, y_test, y_pred)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_probs: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Calculate per-class accuracy
        per_class_acc = []
        for i in range(len(self.class_names)):
            mask = y_true == i
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == i).mean()
                per_class_acc.append(float(class_acc))
        metrics['per_class_accuracy'] = per_class_acc
        
        return metrics
    
    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """Generate and save confusion matrix plot."""
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion'}
        )
        
        plt.title('Confusion Matrix (Normalized)', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Confusion matrix saved: {save_path}")
    
    def _plot_per_class_metrics(self, class_report: Dict) -> None:
        """Plot per-class precision, recall, and F1 scores."""
        metrics_data = {
            'Precision': [],
            'Recall': [],
            'F1-Score': []
        }
        
        for class_name in self.class_names:
            metrics_data['Precision'].append(class_report[class_name]['precision'])
            metrics_data['Recall'].append(class_report[class_name]['recall'])
            metrics_data['F1-Score'].append(class_report[class_name]['f1-score'])
        
        # Create bar plot
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.bar(x - width, metrics_data['Precision'], width, label='Precision', color='steelblue')
        ax.bar(x, metrics_data['Recall'], width, label='Recall', color='darkorange')
        ax.bar(x + width, metrics_data['F1-Score'], width, label='F1-Score', color='forestgreen')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'per_class_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Per-class metrics plot saved: {save_path}")
    
    def _plot_prediction_samples(
        self,
        X_test: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_samples: int = 25
    ) -> None:
        """Plot sample predictions with correct/incorrect labels."""
        # Select random samples
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        
        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Denormalize if needed
            img = X_test[idx]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            axes[i].imshow(img)
            axes[i].axis('off')
            
            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]
            
            color = 'green' if y_true[idx] == y_pred[idx] else 'red'
            axes[i].set_title(f'T: {true_label}\nP: {pred_label}', 
                             fontsize=8, color=color)
        
        plt.suptitle('Sample Predictions (T=True, P=Predicted)', fontsize=14)
        plt.tight_layout()
        
        save_path = self.output_dir / 'prediction_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Prediction samples saved: {save_path}")
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        save_results = {
            'metrics': results['metrics'],
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix']
        }
        
        save_path = self.output_dir / 'evaluation_results.json'
        with open(save_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        print(f"  Evaluation results saved: {save_path}")


def evaluate_step(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config_path: str = "params.yaml"
) -> Dict[str, Any]:
    """
    ZenML step for model evaluation.
    
    Evaluates the trained model on test data with comprehensive metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing evaluation metrics and artifacts
    """
    print("\n" + "="*60)
    print("STEP: Model Evaluation")
    print("="*60)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        class_names=CLASS_NAMES,
        output_dir="reports"
    )
    
    # Evaluate
    results = evaluator.evaluate(model, X_test, y_test, save_plots=True)
    
    # Log to MLflow
    try:
        mlflow.log_metrics({
            'test_accuracy': results['metrics']['accuracy'],
            'test_precision_macro': results['metrics']['precision_macro'],
            'test_recall_macro': results['metrics']['recall_macro'],
            'test_f1_macro': results['metrics']['f1_macro']
        })
        
        # Log artifacts
        mlflow.log_artifact(str(evaluator.output_dir / 'confusion_matrix.png'))
        mlflow.log_artifact(str(evaluator.output_dir / 'per_class_metrics.png'))
        mlflow.log_artifact(str(evaluator.output_dir / 'prediction_samples.png'))
        mlflow.log_artifact(str(evaluator.output_dir / 'evaluation_results.json'))
        
        print("\nEvaluation artifacts logged to MLflow")
    except Exception as e:
        print(f"Warning: Could not log to MLflow: {e}")
    
    return {
        'metrics': results['metrics'],
        'evaluation_results_path': str(evaluator.output_dir / 'evaluation_results.json'),
        'confusion_matrix_path': str(evaluator.output_dir / 'confusion_matrix.png')
    }


if __name__ == "__main__":
    print("Evaluation step - requires trained model and test data")
