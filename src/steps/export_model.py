"""
Model Export Step for Serving-Ready Format

This module handles exporting the trained model in formats
ready for serving and deployment.
"""

import os
import json
import shutil
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml


class ModelExporter:
    """
    Exports trained models in various serving-ready formats.
    
    Supports:
    - SavedModel format (TensorFlow Serving)
    - TensorFlow Lite (Edge deployment)
    - ONNX format (Cross-platform)
    - HDF5 format (Keras native)
    """
    
    def __init__(
        self,
        model: keras.Model,
        output_dir: str = "models/serving"
    ):
        """
        Initialize the exporter.
        
        Args:
            model: Trained Keras model
            output_dir: Directory for exported models
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_savedmodel(
        self,
        version: int = 1
    ) -> str:
        """
        Export model in TensorFlow SavedModel format.
        
        This format is compatible with TensorFlow Serving.
        
        Args:
            version: Model version number
            
        Returns:
            Path to exported model
        """
        export_path = self.output_dir / "savedmodel" / str(version)
        
        # Remove existing export
        if export_path.exists():
            shutil.rmtree(export_path)
        
        # Export
        tf.saved_model.save(
            self.model,
            str(export_path),
            signatures=self._get_serving_signature()
        )
        
        print(f"  SavedModel exported: {export_path}")
        return str(export_path)
    
    def _get_serving_signature(self):
        """Create serving signature for the model."""
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 32, 32, 3], 
                                                     dtype=tf.float32, 
                                                     name='input')])
        def serve(input_tensor):
            return {'predictions': self.model(input_tensor)}
        
        return serve
    
    def export_tflite(self) -> str:
        """
        Export model in TensorFlow Lite format.
        
        Suitable for mobile and edge deployment.
        
        Returns:
            Path to TFLite model
        """
        export_path = self.output_dir / "tflite"
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimize for size and latency
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for quantization (optional)
        def representative_dataset():
            # Generate dummy data for quantization calibration
            for _ in range(100):
                yield [np.random.randn(1, 32, 32, 3).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        tflite_path = export_path / "model.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Save labels
        labels_path = export_path / "labels.txt"
        with open(labels_path, 'w') as f:
            labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck']
            f.write('\n'.join(labels))
        
        print(f"  TFLite model exported: {tflite_path}")
        return str(tflite_path)
    
    def export_h5(self) -> str:
        """
        Export model in HDF5 format.
        
        Standard Keras format for portability.
        
        Returns:
            Path to HDF5 model
        """
        export_path = self.output_dir / "h5"
        export_path.mkdir(parents=True, exist_ok=True)
        
        h5_path = export_path / "model.h5"
        self.model.save(h5_path)
        
        print(f"  HDF5 model exported: {h5_path}")
        return str(h5_path)
    
    def export_keras(self) -> str:
        """
        Export model in Keras native format (.keras).
        
        Returns:
            Path to Keras model
        """
        export_path = self.output_dir / "keras"
        export_path.mkdir(parents=True, exist_ok=True)
        
        keras_path = export_path / "model.keras"
        self.model.save(keras_path)
        
        print(f"  Keras model exported: {keras_path}")
        return str(keras_path)
    
    def export_weights(self) -> str:
        """
        Export model weights only.
        
        Useful for transfer learning scenarios.
        
        Returns:
            Path to weights file
        """
        export_path = self.output_dir / "weights"
        export_path.mkdir(parents=True, exist_ok=True)
        
        weights_path = export_path / "model_weights.h5"
        self.model.save_weights(weights_path)
        
        print(f"  Model weights exported: {weights_path}")
        return str(weights_path)
    
    def create_model_card(self, metrics: Dict[str, float]) -> str:
        """
        Create a model card with metadata.
        
        Args:
            metrics: Dictionary of model metrics
            
        Returns:
            Path to model card file
        """
        model_card = {
            'name': 'CIFAR-10 CNN Classifier',
            'version': '1.0.0',
            'description': 'Convolutional Neural Network for image classification on CIFAR-10',
            'architecture': {
                'type': 'CNN',
                'input_shape': [32, 32, 3],
                'output_classes': 10,
                'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
            },
            'metrics': metrics,
            'framework': 'TensorFlow/Keras',
            'dataset': {
                'name': 'CIFAR-10',
                'train_samples': 50000,
                'test_samples': 10000
            },
            'usage': {
                'input': 'Normalized RGB images (32x32)',
                'output': 'Class probabilities (softmax)'
            }
        }
        
        card_path = self.output_dir / "model_card.json"
        with open(card_path, 'w') as f:
            json.dump(model_card, f, indent=2)
        
        print(f"  Model card created: {card_path}")
        return str(card_path)
    
    def export_all(
        self,
        metrics: Dict[str, float],
        version: int = 1
    ) -> Dict[str, str]:
        """
        Export model in all supported formats.
        
        Args:
            metrics: Dictionary of model metrics
            version: Model version number
            
        Returns:
            Dictionary of exported paths
        """
        print("\nExporting model in all formats...")
        
        paths = {
            'savedmodel': self.export_savedmodel(version),
            'tflite': self.export_tflite(),
            'h5': self.export_h5(),
            'keras': self.export_keras(),
            'weights': self.export_weights(),
            'model_card': self.create_model_card(metrics)
        }
        
        return paths


def export_model_step(
    model: keras.Model,
    metrics: Dict[str, float],
    version: int = 1,
    config_path: str = "params.yaml"
) -> Dict[str, Any]:
    """
    ZenML step for model export.
    
    Exports the trained model in serving-ready formats.
    
    Args:
        model: Trained Keras model
        metrics: Dictionary of model metrics
        version: Model version number
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing export paths
    """
    print("\n" + "="*60)
    print("STEP: Model Export")
    print("="*60)
    
    # Create exporter
    exporter = ModelExporter(model=model, output_dir="models/serving")
    
    # Export in all formats
    export_paths = exporter.export_all(metrics=metrics, version=version)
    
    print(f"\nExport Summary:")
    for format_name, path in export_paths.items():
        print(f"  {format_name}: {path}")
    
    # Create export manifest
    manifest = {
        'model_name': 'cifar10_classifier',
        'version': version,
        'formats': {
            'savedmodel': {
                'path': export_paths['savedmodel'],
                'description': 'TensorFlow SavedModel for TensorFlow Serving'
            },
            'tflite': {
                'path': export_paths['tflite'],
                'description': 'TensorFlow Lite for mobile/edge deployment'
            },
            'keras': {
                'path': export_paths['keras'],
                'description': 'Keras native format'
            }
        },
        'metrics': metrics
    }
    
    manifest_path = Path("models/export_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nExport manifest saved: {manifest_path}")
    
    return {
        'export_paths': export_paths,
        'manifest_path': str(manifest_path)
    }


if __name__ == "__main__":
    print("Model export step - requires trained model")
