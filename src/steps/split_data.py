"""
Data Splitting Step for CIFAR-10 Dataset

This module handles splitting the training data into training and validation sets,
with stratified sampling to maintain class distribution.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import yaml
import os


class DataSplitter:
    """
    Handles data splitting with stratification support.
    
    Provides functionality to split data while maintaining class distribution
    across all splits.
    """
    
    def __init__(
        self,
        validation_split: float = 0.1,
        random_seed: int = 42,
        stratify: bool = True
    ):
        """
        Initialize the data splitter.
        
        Args:
            validation_split: Fraction of training data to use for validation
            random_seed: Random seed for reproducibility
            stratify: Whether to use stratified splitting
        """
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.stratify = stratify
        np.random.seed(random_seed)
        
    def stratified_split(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        test_size: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform stratified split of data.
        
        Ensures that each class is represented proportionally in both splits.
        
        Args:
            images: Input images array
            labels: Input labels array
            test_size: Fraction for the validation/test split
            
        Returns:
            Tuple of (train_images, val_images, train_labels, val_labels)
        """
        unique_labels = np.unique(labels)
        
        train_indices = []
        val_indices = []
        
        for label in unique_labels:
            # Get indices for this class
            class_indices = np.where(labels == label)[0]
            np.random.shuffle(class_indices)
            
            # Calculate split point
            n_val = int(len(class_indices) * test_size)
            
            # Split
            val_indices.extend(class_indices[:n_val])
            train_indices.extend(class_indices[n_val:])
        
        # Shuffle final indices
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        
        return (
            images[train_indices],
            images[val_indices],
            labels[train_indices],
            labels[val_indices]
        )
    
    def simple_split(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        test_size: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform simple random split without stratification.
        
        Args:
            images: Input images array
            labels: Input labels array
            test_size: Fraction for the validation/test split
            
        Returns:
            Tuple of (train_images, val_images, train_labels, val_labels)
        """
        n_samples = len(images)
        indices = np.random.permutation(n_samples)
        
        n_val = int(n_samples * test_size)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        return (
            images[train_indices],
            images[val_indices],
            labels[train_indices],
            labels[val_indices]
        )
    
    def split(
        self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.
        
        Args:
            images: Training images array
            labels: Training labels array
            
        Returns:
            Tuple of (train_images, val_images, train_labels, val_labels)
        """
        if self.stratify:
            return self.stratified_split(images, labels, self.validation_split)
        else:
            return self.simple_split(images, labels, self.validation_split)


def verify_split_distribution(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Verify class distribution across splits.
    
    Args:
        train_labels: Training labels
        val_labels: Validation labels
        test_labels: Optional test labels
        
    Returns:
        Distribution statistics dictionary
    """
    unique_classes = np.unique(np.concatenate([train_labels, val_labels]))
    
    stats = {
        'train': {},
        'val': {},
        'train_percent': {},
        'val_percent': {}
    }
    
    train_total = len(train_labels)
    val_total = len(val_labels)
    
    for cls in unique_classes:
        train_count = np.sum(train_labels == cls)
        val_count = np.sum(val_labels == cls)
        
        stats['train'][int(cls)] = int(train_count)
        stats['val'][int(cls)] = int(val_count)
        stats['train_percent'][int(cls)] = float(train_count / train_total * 100)
        stats['val_percent'][int(cls)] = float(val_count / val_total * 100)
    
    if test_labels is not None:
        stats['test'] = {}
        stats['test_percent'] = {}
        test_total = len(test_labels)
        
        for cls in unique_classes:
            test_count = np.sum(test_labels == cls)
            stats['test'][int(cls)] = int(test_count)
            stats['test_percent'][int(cls)] = float(test_count / test_total * 100)
    
    return stats


def split_data_step(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    config_path: str = "params.yaml"
) -> Dict[str, Any]:
    """
    ZenML step for data splitting.
    
    Splits training data into training and validation sets,
    with stratified sampling to maintain class distribution.
    
    Args:
        train_images: Training images array
        train_labels: Training labels array
        test_images: Test images array
        test_labels: Test labels array
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing split data arrays and metadata
    """
    print("\n" + "="*60)
    print("STEP: Data Splitting")
    print("="*60)
    
    # Load configuration
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    data_config = config.get('data', {})
    validation_split = data_config.get('validation_split', 0.1)
    random_seed = data_config.get('random_seed', 42)
    
    # Create splitter
    splitter = DataSplitter(
        validation_split=validation_split,
        random_seed=random_seed,
        stratify=True
    )
    
    # Split training data
    print(f"\nSplitting training data (validation_split={validation_split})...")
    X_train, X_val, y_train, y_val = splitter.split(train_images, train_labels)
    
    # Verify distribution
    print("\nVerifying class distribution...")
    distribution_stats = verify_split_distribution(y_train, y_val, test_labels)
    
    # Print summary
    print(f"\nData split summary:")
    print(f"  Original training samples: {len(train_images)}")
    print(f"  New training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(test_images)}")
    
    print(f"\nClass distribution (train/val):")
    for cls in sorted(distribution_stats['train'].keys()):
        train_pct = distribution_stats['train_percent'][cls]
        val_pct = distribution_stats['val_percent'][cls]
        print(f"  Class {cls}: {train_pct:.1f}% / {val_pct:.1f}%")
    
    # Save split data
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "X_val.npy", X_val)
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "X_test.npy", test_images)
    np.save(processed_dir / "y_test.npy", test_labels)
    
    print(f"\nSplit data saved to: {processed_dir}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': test_images,
        'y_test': test_labels,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(test_images),
        'distribution_stats': distribution_stats
    }


if __name__ == "__main__":
    # Test splitting
    from ingest_data import load_raw_data
    
    train_images, train_labels, test_images, test_labels = load_raw_data()
    result = split_data_step(train_images, train_labels, test_images, test_labels)
    
    print(f"\nShapes:")
    print(f"  X_train: {result['X_train'].shape}")
    print(f"  X_val: {result['X_val'].shape}")
    print(f"  X_test: {result['X_test'].shape}")
