"""
CNN Model Architecture for CIFAR-10 Classification

This module implements a Convolutional Neural Network for image classification
on the CIFAR-10 dataset. The architecture is configurable via the params.yaml file.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Dict, Any, Optional
import yaml
import os


class CIFAR10CNN:
    """
    CNN Model for CIFAR-10 Image Classification.
    
    This class provides a configurable CNN architecture with:
    - Multiple convolutional layers with batch normalization
    - Max pooling for spatial dimension reduction
    - Dropout for regularization
    - Dense layers for classification
    
    Attributes:
        config: Configuration dictionary from params.yaml
        model: The compiled Keras model
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize the CNN model.
        
        Args:
            config_path: Path to the parameters configuration file
        """
        self.config = self._load_config(config_path)
        self.model: Optional[keras.Model] = None
        self._build_model()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'train': {
                    'architecture': {
                        'conv_layers': [
                            {'filters': 32, 'kernel_size': 3, 'activation': 'relu', 'pooling': 'max'},
                            {'filters': 64, 'kernel_size': 3, 'activation': 'relu', 'pooling': 'max'},
                            {'filters': 128, 'kernel_size': 3, 'activation': 'relu', 'pooling': 'max'},
                        ],
                        'dense_layers': [
                            {'units': 256, 'activation': 'relu', 'dropout': 0.5},
                            {'units': 128, 'activation': 'relu', 'dropout': 0.3},
                        ],
                        'output': {'units': 10, 'activation': 'softmax'}
                    }
                }
            }
    
    def _build_model(self) -> None:
        """
        Build the CNN model architecture based on configuration.
        
        The architecture consists of:
        1. Input layer for 32x32x3 images
        2. Multiple convolutional blocks (Conv2D + BatchNorm + Activation + Pooling)
        3. Flatten layer
        4. Dense layers with dropout
        5. Output layer with softmax activation
        """
        arch_config = self.config.get('train', {}).get('architecture', {})
        conv_layers = arch_config.get('conv_layers', [])
        dense_layers = arch_config.get('dense_layers', [])
        output_config = arch_config.get('output', {'units': 10, 'activation': 'softmax'})
        
        # Input layer
        inputs = layers.Input(shape=(32, 32, 3), name='input_layer')
        
        # Normalize pixel values
        x = layers.Rescaling(1./255, name='rescaling')(inputs)
        
        # Convolutional blocks
        for i, conv_config in enumerate(conv_layers):
            filters = conv_config.get('filters', 32)
            kernel_size = conv_config.get('kernel_size', 3)
            activation = conv_config.get('activation', 'relu')
            pooling = conv_config.get('pooling', 'max')
            use_batch_norm = conv_config.get('batch_norm', True)
            
            # Convolutional layer
            x = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                kernel_regularizer=regularizers.l2(0.001),
                name=f'conv_{i+1}'
            )(x)
            
            # Batch normalization
            if use_batch_norm:
                x = layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
            
            # Activation
            x = layers.Activation(activation, name=f'activation_{i+1}')(x)
            
            # Pooling
            if pooling == 'max':
                x = layers.MaxPooling2D(pool_size=(2, 2), name=f'max_pool_{i+1}')(x)
            elif pooling == 'avg':
                x = layers.AveragePooling2D(pool_size=(2, 2), name=f'avg_pool_{i+1}')(x)
            
            # Add dropout for deeper layers
            if i >= 1:
                x = layers.Dropout(0.25, name=f'conv_dropout_{i+1}')(x)
        
        # Flatten
        x = layers.Flatten(name='flatten')(x)
        
        # Dense layers
        for i, dense_config in enumerate(dense_layers):
            units = dense_config.get('units', 128)
            activation = dense_config.get('activation', 'relu')
            dropout_rate = dense_config.get('dropout', 0.5)
            
            x = layers.Dense(
                units=units,
                kernel_regularizer=regularizers.l2(0.001),
                name=f'dense_{i+1}'
            )(x)
            
            x = layers.BatchNormalization(name=f'dense_batch_norm_{i+1}')(x)
            x = layers.Activation(activation, name=f'dense_activation_{i+1}')(x)
            x = layers.Dropout(dropout_rate, name=f'dense_dropout_{i+1}')(x)
        
        # Output layer
        outputs = layers.Dense(
            units=output_config.get('units', 10),
            activation=output_config.get('activation', 'softmax'),
            name='output_layer'
        )(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='cifar10_cnn')
        
    def compile_model(
        self,
        learning_rate: float = 0.001,
        optimizer: str = 'adam',
        loss: str = 'sparse_categorical_crossentropy',
        metrics: Optional[list] = None
    ) -> None:
        """
        Compile the model with specified optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for the optimizer
            optimizer: Name of the optimizer ('adam', 'sgd', 'rmsprop')
            loss: Loss function name
            metrics: List of metrics to track
        """
        if metrics is None:
            metrics = ['accuracy']
        
        # Select optimizer
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def get_model(self) -> keras.Model:
        """Return the compiled Keras model."""
        return self.model


def create_model(config_path: str = "params.yaml") -> keras.Model:
    """
    Factory function to create and return a compiled CNN model.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Compiled Keras model ready for training
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_config = config.get('train', {})
    
    cnn = CIFAR10CNN(config_path)
    cnn.compile_model(
        learning_rate=train_config.get('learning_rate', 0.001),
        optimizer=train_config.get('optimizer', 'adam'),
        loss=train_config.get('loss', 'sparse_categorical_crossentropy'),
        metrics=['accuracy']
    )
    
    return cnn.get_model()


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print("\n" + "="*50)
    print("CNN Model Architecture for CIFAR-10")
    print("="*50 + "\n")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
