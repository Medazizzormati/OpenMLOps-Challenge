"""
Data Preprocessing Step for CIFAR-10 Dataset

This module handles preprocessing of images including normalization,
data augmentation, and feature engineering.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import yaml
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ImagePreprocessor:
    """
    Preprocessor for CIFAR-10 images.
    
    Handles normalization, augmentation, and other preprocessing operations.
    """
    
    def __init__(
        self,
        normalize: bool = True,
        augmentation: bool = True,
        target_size: Tuple[int, int] = (32, 32)
    ):
        """
        Initialize the preprocessor.
        
        Args:
            normalize: Whether to normalize pixel values
            augmentation: Whether to apply data augmentation
            target_size: Target image size (height, width)
        """
        self.normalize = normalize
        self.augmentation = augmentation
        self.target_size = target_size
        self.mean = None
        self.std = None
        
    def fit(self, images: np.ndarray) -> 'ImagePreprocessor':
        """
        Fit the preprocessor on training data.
        
        Computes normalization statistics from the training set.
        
        Args:
            images: Training images array
            
        Returns:
            Self for chaining
        """
        if self.normalize:
            self.mean = images.mean(axis=(0, 1, 2), keepdims=True)
            self.std = images.std(axis=(0, 1, 2), keepdims=True)
            # Avoid division by zero
            self.std = np.where(self.std == 0, 1, self.std)
            
        return self
    
    def transform(
        self, 
        images: np.ndarray,
        training: bool = False
    ) -> np.ndarray:
        """
        Transform images using fitted preprocessor.
        
        Args:
            images: Images array to transform
            training: Whether this is training data (for augmentation)
            
        Returns:
            Transformed images array
        """
        processed = images.copy()
        
        # Normalize
        if self.normalize:
            if self.mean is not None and self.std is not None:
                processed = (processed - self.mean) / self.std
            else:
                processed = processed.astype(np.float32) / 255.0
        
        # Apply augmentation only during training
        if training and self.augmentation:
            processed = self._augment(processed)
            
        return processed
    
    def _augment(self, images: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to images.
        
        Includes random flips, rotations, and shifts.
        
        Args:
            images: Images array to augment
            
        Returns:
            Augmented images array
        """
        augmented = []
        
        for img in images:
            # Random horizontal flip
            if np.random.random() > 0.5:
                img = np.fliplr(img)
            
            # Random rotation (small angles for CIFAR-10)
            angle = np.random.uniform(-15, 15)
            img = self._rotate_image(img, angle)
            
            # Random width/height shift
            shift_x = np.random.randint(-4, 5)
            shift_y = np.random.randint(-4, 5)
            img = np.roll(img, shift_x, axis=0)
            img = np.roll(img, shift_y, axis=1)
            
            augmented.append(img)
            
        return np.array(augmented)
    
    def _rotate_image(
        self, 
        image: np.ndarray, 
        angle: float
    ) -> np.ndarray:
        """Rotate image by given angle."""
        from scipy.ndimage import rotate
        return rotate(image, angle, reshape=False, mode='reflect')
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Get normalization parameters."""
        return {
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'normalize': self.normalize
        }


def create_augmentation_pipeline() -> keras.Sequential:
    """
    Create a Keras data augmentation pipeline.
    
    Returns:
        Sequential model with augmentation layers
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")


def preprocess_cifar10(
    config_path: str = "params.yaml"
) -> Dict[str, Any]:
    """
    Preprocess CIFAR-10 dataset for training.
    
    This function can be called directly for DVC pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with preprocessing metadata
    """
    print("\n" + "="*60)
    print("Preprocessing CIFAR-10 Dataset")
    print("="*60)
    
    # Load configuration
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    preprocess_config = config.get('preprocess', {})
    
    # Load data
    processed_dir = Path("data/processed")
    
    X_train = np.load(processed_dir / "train_images.npy")
    y_train = np.load(processed_dir / "train_labels.npy")
    X_test = np.load(processed_dir / "test_images.npy")
    y_test = np.load(processed_dir / "test_labels.npy")
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        normalize=preprocess_config.get('normalize', True),
        augmentation=preprocess_config.get('augmentation', True)
    )
    
    # Fit and transform
    preprocessor.fit(X_train)
    
    X_train_processed = preprocessor.transform(X_train, training=True)
    X_test_processed = preprocessor.transform(X_test, training=False)
    
    # Save processed data
    np.save(processed_dir / "train_images_processed.npy", X_train_processed)
    np.save(processed_dir / "test_images_processed.npy", X_test_processed)
    
    print(f"\nProcessed data shapes:")
    print(f"  Train: {X_train_processed.shape}")
    print(f"  Test: {X_test_processed.shape}")
    print(f"  Value range: [{X_train_processed.min():.3f}, {X_train_processed.max():.3f}]")
    
    return {
        'train_shape': list(X_train_processed.shape),
        'test_shape': list(X_test_processed.shape),
        'normalization': preprocessor.get_normalization_params()
    }


def preprocess_step(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config_path: str = "params.yaml"
) -> Dict[str, Any]:
    """
    ZenML step for data preprocessing.
    
    Normalizes and optionally augments the image data.
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        X_test: Test images
        y_test: Test labels
        config_path: Path to configuration file
        
    Returns:
        Dictionary with preprocessed data and metadata
    """
    print("\n" + "="*60)
    print("STEP: Data Preprocessing")
    print("="*60)
    
    # Load configuration
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    preprocess_config = config.get('preprocess', {})
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        normalize=preprocess_config.get('normalize', True),
        augmentation=preprocess_config.get('augmentation', True)
    )
    
    # Fit on training data
    print("\nFitting preprocessor on training data...")
    preprocessor.fit(X_train)
    
    # Transform all datasets
    print("Transforming datasets...")
    X_train_processed = preprocessor.transform(X_train, training=True)
    X_val_processed = preprocessor.transform(X_val, training=False)
    X_test_processed = preprocessor.transform(X_test, training=False)
    
    # Print statistics
    print(f"\nPreprocessing summary:")
    print(f"  Normalization: {preprocess_config.get('normalize', True)}")
    print(f"  Augmentation: {preprocess_config.get('augmentation', True)}")
    print(f"  Train shape: {X_train_processed.shape}")
    print(f"  Val shape: {X_val_processed.shape}")
    print(f"  Test shape: {X_test_processed.shape}")
    
    # Compute dataset statistics
    stats = {
        'train_mean': float(X_train_processed.mean()),
        'train_std': float(X_train_processed.std()),
        'train_min': float(X_train_processed.min()),
        'train_max': float(X_train_processed.max())
    }
    
    print(f"\nDataset statistics:")
    print(f"  Mean: {stats['train_mean']:.4f}")
    print(f"  Std: {stats['train_std']:.4f}")
    print(f"  Min: {stats['train_min']:.4f}")
    print(f"  Max: {stats['train_max']:.4f}")
    
    # Save normalization params
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    norm_params = preprocessor.get_normalization_params()
    np.save(processed_dir / "norm_mean.npy", preprocessor.mean)
    np.save(processed_dir / "norm_std.npy", preprocessor.std)
    
    return {
        'X_train': X_train_processed,
        'y_train': y_train,
        'X_val': X_val_processed,
        'y_val': y_val,
        'X_test': X_test_processed,
        'y_test': y_test,
        'normalization_params': norm_params,
        'statistics': stats,
        'preprocessor': preprocessor
    }


if __name__ == "__main__":
    # Test preprocessing
    from ingest_data import load_raw_data
    from split_data import split_data_step
    
    train_images, train_labels, test_images, test_labels = load_raw_data()
    split_result = split_data_step(train_images, train_labels, test_images, test_labels)
    
    result = preprocess_step(
        split_result['X_train'],
        split_result['y_train'],
        split_result['X_val'],
        split_result['y_val'],
        split_result['X_test'],
        split_result['y_test']
    )
    
    print(f"\nPreprocessing complete!")
