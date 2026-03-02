"""
Model Training Step for CIFAR-10 CNN Classifier

This module handles the training of the CNN model with MLflow tracking
and checkpointing.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import mlflow
import mlflow.tensorflow


class ModelTrainer:
    """
    Trainer for the CIFAR-10 CNN model.
    
    Handles model training with:
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - MLflow experiment tracking
    """
    
    def __init__(
        self,
        model: keras.Model,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "checkpoints",
        use_mlflow: bool = True
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Keras model to train
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Initial learning rate
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory for model checkpoints
            use_mlflow: Whether to log to MLflow
        """
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.use_mlflow = use_mlflow
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = None
        self.best_val_accuracy = 0.0
        
    def _get_callbacks(self) -> list:
        """
        Get training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callback_list = []
        
        # Learning rate reduction on plateau
        lr_reducer = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(lr_reducer)
        
        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)
        
        # Model checkpoint
        checkpoint_path = self.checkpoint_dir / "best_model.keras"
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        callback_list.append(model_checkpoint)
        
        # CSV logger
        csv_logger = callbacks.CSVLogger(
            'training_log.csv',
            separator=',',
            append=False
        )
        callback_list.append(csv_logger)
        
        # MLflow callback
        if self.use_mlflow:
            mlflow_callback = MLflowCallback()
            callback_list.append(mlflow_callback)
        
        return callback_list
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            
        Returns:
            Training results dictionary
        """
        print("\n" + "="*60)
        print("Training CNN Model")
        print("="*60)
        
        print(f"\nTraining configuration:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        # Enable MLflow autolog
        if self.use_mlflow:
            mlflow.tensorflow.autolog(
                log_models=True,
                log_datasets=False,
                registered_model_name="cifar10_cnn"
            )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self._get_callbacks(),
            verbose=1
        )
        
        # Get best metrics
        self.best_val_accuracy = max(self.history.history['val_accuracy'])
        best_epoch = self.history.history['val_accuracy'].index(self.best_val_accuracy) + 1
        
        print(f"\nTraining complete!")
        print(f"  Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"  Best epoch: {best_epoch}")
        
        # Compile results
        results = {
            'best_val_accuracy': float(self.best_val_accuracy),
            'best_epoch': best_epoch,
            'final_train_loss': float(self.history.history['loss'][-1]),
            'final_train_accuracy': float(self.history.history['accuracy'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]),
            'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
            'total_epochs': len(self.history.history['loss']),
            'history': {
                'loss': [float(x) for x in self.history.history['loss']],
                'accuracy': [float(x) for x in self.history.history['accuracy']],
                'val_loss': [float(x) for x in self.history.history['val_loss']],
                'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
            }
        }
        
        return results
    
    def get_trained_model(self) -> keras.Model:
        """Return the trained model."""
        return self.model
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        print(f"Model saved to: {path}")


class MLflowCallback(callbacks.Callback):
    """
    Custom MLflow callback for additional logging.
    """
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of each epoch."""
        if logs:
            mlflow.log_metrics({
                'train_loss': logs.get('loss', 0),
                'train_accuracy': logs.get('accuracy', 0),
                'val_loss': logs.get('val_loss', 0),
                'val_accuracy': logs.get('val_accuracy', 0),
                'learning_rate': float(self.model.optimizer.learning_rate.numpy())
            }, step=epoch)


def train_step(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config_path: str = "params.yaml"
) -> Dict[str, Any]:
    """
    ZenML step for model training.
    
    Trains the CNN model with the provided data.
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing trained model and training metrics
    """
    print("\n" + "="*60)
    print("STEP: Model Training")
    print("="*60)
    
    # Load configuration
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    train_config = config.get('train', {})
    mlflow_config = config.get('mlflow', {})
    
    # Start MLflow run
    experiment_name = mlflow_config.get('experiment_name', 'cifar10_cnn_classifier')
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(
        run_name=f"{mlflow_config.get('run_name_prefix', 'training_run')}_{os.urandom(4).hex()}"
    ):
        # Log parameters
        mlflow.log_params({
            'epochs': train_config.get('epochs', 50),
            'batch_size': train_config.get('batch_size', 64),
            'learning_rate': train_config.get('learning_rate', 0.001),
            'optimizer': train_config.get('optimizer', 'adam'),
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        })
        
        # Create model
        from src.models.cnn_model import create_model
        model = create_model(config_path)
        
        # Log model summary
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        mlflow.log_text('\n'.join(summary_list), 'model_summary.txt')
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            epochs=train_config.get('epochs', 50),
            batch_size=train_config.get('batch_size', 64),
            learning_rate=train_config.get('learning_rate', 0.001),
            early_stopping_patience=train_config.get('early_stopping', {}).get('patience', 10),
            use_mlflow=True
        )
        
        # Train
        results = trainer.train(X_train, y_train, X_val, y_val)
        
        # Log final metrics
        mlflow.log_metrics({
            'best_val_accuracy': results['best_val_accuracy'],
            'best_epoch': results['best_epoch'],
            'final_train_loss': results['final_train_loss'],
            'final_val_loss': results['final_val_loss']
        })
        
        # Save model
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "trained_model.keras"
        trainer.save_model(str(model_path))
        
        # Save training history
        with open(models_dir / "training_history.json", 'w') as f:
            json.dump(results['history'], f, indent=2)
        
        # Save metrics for DVC
        metrics = {
            'train_accuracy': results['final_train_accuracy'],
            'val_accuracy': results['final_val_accuracy'],
            'train_loss': results['final_train_loss'],
            'val_loss': results['final_val_loss'],
            'best_val_accuracy': results['best_val_accuracy'],
            'best_epoch': results['best_epoch']
        }
        
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return {
            'model': trainer.get_trained_model(),
            'model_path': str(model_path),
            'metrics': metrics,
            'training_results': results
        }


if __name__ == "__main__":
    # Test training (requires preprocessed data)
    print("Training step - requires preprocessed data")
