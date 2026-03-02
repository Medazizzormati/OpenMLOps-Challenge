"""
ZenML Stack Setup and Configuration

This module handles ZenML stack initialization with MLflow integration.
"""

import os
from typing import Optional
from zenml.client import Client
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLflowExperimentTrackerSettings,
)
from zenml.integrations.mlflow.flavors.mlflow_model_deployer_flavor import (
    MLflowModelDeployerSettings,
)


def setup_zenml_stack(
    stack_name: str = "mlops_stack",
    mlflow_tracking_uri: Optional[str] = None,
) -> None:
    """
    Setup ZenML stack with MLflow integration.
    
    Args:
        stack_name: Name for the custom stack
        mlflow_tracking_uri: MLflow tracking URI
    """
    client = Client()
    
    # Get or create MLflow tracking URI
    if mlflow_tracking_uri is None:
        mlflow_tracking_uri = os.environ.get(
            'MLFLOW_TRACKING_URI',
            'http://localhost:5000'
        )
    
    print(f"Setting up ZenML stack: {stack_name}")
    print(f"MLflow tracking URI: {mlflow_tracking_uri}")
    
    try:
        # Check if stack already exists
        stack = client.get_stack(stack_name)
        print(f"Stack '{stack_name}' already exists")
    except KeyError:
        # Create new stack
        print(f"Creating new stack: {stack_name}")
        
        # Get default orchestrator
        orchestrator = client.active_stack.orchestrator
        
        # Get default artifact store
        artifact_store = client.active_stack.artifact_store
        
        # Register MLflow experiment tracker
        experiment_tracker = client.create_or_update_component(
            name="mlflow_tracker",
            type="experiment_tracker",
            configuration={
                "flavor": "mlflow",
                "tracking_uri": mlflow_tracking_uri,
                "experiment_name": "cifar10_cnn_classifier"
            }
        )
        
        # Register MLflow model deployer
        model_deployer = client.create_or_update_component(
            name="mlflow_deployer",
            type="model_deployer",
            configuration={
                "flavor": "mlflow",
                "tracking_uri": mlflow_tracking_uri
            }
        )
        
        # Create stack
        client.create_stack(
            name=stack_name,
            components={
                "orchestrator": orchestrator.id,
                "artifact_store": artifact_store.id,
                "experiment_tracker": experiment_tracker.id,
                "model_deployer": model_deployer.id
            }
        )
        
        print(f"Stack '{stack_name}' created successfully")
    
    # Set as active stack
    client.activate_stack(stack_name)
    print(f"Stack '{stack_name}' activated")


def get_mlflow_tracking_uri() -> str:
    """Get the MLflow tracking URI from environment or default."""
    return os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')


def get_minio_config() -> dict:
    """Get MinIO configuration from environment."""
    return {
        'endpoint': os.environ.get('DVC_S3_ENDPOINT_URL', 'http://localhost:9000'),
        'access_key': os.environ.get('AWS_ACCESS_KEY_ID', 'minioadmin'),
        'secret_key': os.environ.get('AWS_SECRET_ACCESS_KEY', 'minioadmin123'),
        'bucket': os.environ.get('DVC_BUCKET', 'mlops-data')
    }


if __name__ == "__main__":
    setup_zenml_stack()
