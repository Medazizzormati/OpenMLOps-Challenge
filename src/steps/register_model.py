"""
Model Registration Step for MLflow Model Registry

This module handles registering trained models with MLflow
for version control and deployment management.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import tensorflow as tf
from tensorflow import keras


class ModelRegistry:
    """
    Handles model registration with MLflow Model Registry.
    
    Provides functionality to:
    - Register new model versions
    - Transition model stages (Staging, Production, Archived)
    - Compare model versions
    - Manage model metadata
    """
    
    def __init__(
        self,
        model_name: str,
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize the model registry.
        
        Args:
            model_name: Name for the registered model
            tracking_uri: MLflow tracking URI (default: from environment)
        """
        self.model_name = model_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient()
        
        # Create registered model if not exists
        self._ensure_registered_model()
    
    def _ensure_registered_model(self) -> None:
        """Create registered model if it doesn't exist."""
        try:
            self.client.get_registered_model(self.model_name)
            print(f"Registered model '{self.model_name}' exists")
        except MlflowException:
            self.client.create_registered_model(self.model_name)
            print(f"Created registered model '{self.model_name}'")
    
    def register_model(
        self,
        model_path: str,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Register a model version.
        
        Args:
            model_path: Path to the saved model
            run_id: MLflow run ID (default: current active run)
            tags: Optional tags for the model version
            description: Description for the model version
            
        Returns:
            Model version string
        """
        print(f"\nRegistering model version...")
        
        # Get run ID if not provided
        if run_id is None:
            active_run = mlflow.active_run()
            if active_run:
                run_id = active_run.info.run_id
            else:
                raise ValueError("No active MLflow run. Please provide run_id.")
        
        # Register model
        model_uri = f"runs:/{run_id}/model"
        
        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=self.model_name,
                tags=tags
            )
            
            version = model_version.version
            print(f"  Model registered: {self.model_name} v{version}")
            
            # Update description
            if description:
                self.client.update_model_version(
                    name=self.model_name,
                    version=version,
                    description=description
                )
            
            return version
            
        except Exception as e:
            print(f"  Error registering model: {e}")
            raise
    
    def transition_stage(
        self,
        version: str,
        stage: str = "Staging",
        archive_existing: bool = True
    ) -> None:
        """
        Transition a model version to a new stage.
        
        Args:
            version: Model version to transition
            stage: Target stage (None, Staging, Production, Archived)
            archive_existing: Whether to archive existing versions in target stage
        """
        print(f"\nTransitioning model v{version} to '{stage}'...")
        
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
        
        print(f"  Model v{version} transitioned to '{stage}'")
    
    def get_latest_versions(
        self,
        stages: Optional[list] = None
    ) -> list:
        """
        Get latest model versions.
        
        Args:
            stages: List of stages to filter by
            
        Returns:
            List of model versions
        """
        if stages is None:
            stages = ["None", "Staging", "Production"]
        
        versions = self.client.get_latest_versions(
            self.model_name,
            stages=stages
        )
        
        return versions
    
    def get_production_model(self) -> Optional[Dict[str, Any]]:
        """
        Get the current production model.
        
        Returns:
            Dictionary with production model info or None
        """
        versions = self.get_latest_versions(stages=["Production"])
        
        if versions:
            version = versions[0]
            return {
                'version': version.version,
                'run_id': version.run_id,
                'status': version.status,
                'creation_timestamp': version.creation_timestamp,
                'description': version.description
            }
        
        return None
    
    def compare_versions(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        v1 = self.client.get_model_version(self.model_name, version1)
        v2 = self.client.get_model_version(self.model_name, version2)
        
        # Get metrics from runs
        run1 = self.client.get_run(v1.run_id)
        run2 = self.client.get_run(v2.run_id)
        
        comparison = {
            'version1': {
                'version': version1,
                'metrics': run1.data.metrics,
                'stage': v1.current_stage
            },
            'version2': {
                'version': version2,
                'metrics': run2.data.metrics,
                'stage': v2.current_stage
            }
        }
        
        return comparison


def register_model_step(
    model_path: str,
    metrics: Dict[str, float],
    config_path: str = "params.yaml"
) -> Dict[str, Any]:
    """
    ZenML step for model registration with MLflow.
    
    Registers the trained model in MLflow Model Registry and
    transitions it to the appropriate stage.
    
    Args:
        model_path: Path to the saved model
        metrics: Dictionary of model metrics
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing registration details
    """
    print("\n" + "="*60)
    print("STEP: Model Registration")
    print("="*60)
    
    # Load configuration
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    model_name = config.get('serving', {}).get('model_name', 'cifar10_classifier')
    
    # Get current run
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else None
    
    # Create registry
    registry = ModelRegistry(model_name=model_name)
    
    # Prepare tags
    tags = {
        'framework': 'tensorflow',
        'task': 'image_classification',
        'dataset': 'cifar10'
    }
    
    # Add metric tags
    for metric_name, metric_value in metrics.items():
        tags[f'metric_{metric_name}'] = str(round(metric_value, 4))
    
    # Prepare description
    description = f"""CNN model for CIFAR-10 classification.
    
Metrics:
- Accuracy: {metrics.get('accuracy', metrics.get('val_accuracy', 'N/A')):.4f}
- F1 Score (macro): {metrics.get('f1_macro', 'N/A')}

Architecture:
- Input: 32x32x3 images
- Output: 10 classes (softmax)
"""
    
    # Register model
    version = registry.register_model(
        model_path=model_path,
        run_id=run_id,
        tags=tags,
        description=description
    )
    
    # Transition to Staging for testing
    registry.transition_stage(version, stage="Staging")
    
    # Get model info
    model_info = {
        'model_name': model_name,
        'version': version,
        'stage': 'Staging',
        'run_id': run_id,
        'model_uri': f"models:/{model_name}/{version}"
    }
    
    print(f"\nModel Registration Summary:")
    print(f"  Model Name: {model_name}")
    print(f"  Version: {version}")
    print(f"  Stage: Staging")
    print(f"  Model URI: {model_info['model_uri']}")
    
    # Save registration info
    registration_path = Path("models/registration_info.json")
    with open(registration_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    return model_info


if __name__ == "__main__":
    print("Model registration step - requires trained model")
