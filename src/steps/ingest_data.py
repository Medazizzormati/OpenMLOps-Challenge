"""
Data Ingestion Step for CIFAR-10 Dataset

This module handles downloading and storing the CIFAR-10 dataset
using DVC for version control.
"""

import os
import pickle
import tarfile
import urllib.request
from typing import Tuple, Dict, Any
import numpy as np
from pathlib import Path
import hashlib


# CIFAR-10 dataset URLs and metadata
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"

# Dataset paths
RAW_DATA_DIR = Path("data/raw/cifar-10")
PROCESSED_DATA_DIR = Path("data/processed")


def verify_md5(filepath: str, expected_md5: str) -> bool:
    """Verify the MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest() == expected_md5


def download_cifar10() -> Path:
    """
    Download CIFAR-10 dataset from the official source.
    
    This function downloads the CIFAR-10 dataset, verifies its integrity,
    extracts it to the raw data directory, and returns the path.
    
    Returns:
        Path to the extracted dataset directory
        
    Raises:
        RuntimeError: If download or extraction fails
    """
    print("="*60)
    print("Downloading CIFAR-10 Dataset")
    print("="*60)
    
    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    tar_path = RAW_DATA_DIR / "cifar-10-python.tar.gz"
    
    # Download if not exists
    if not tar_path.exists():
        print(f"\nDownloading from: {CIFAR10_URL}")
        print("This may take a few minutes...")
        
        try:
            urllib.request.urlretrieve(CIFAR10_URL, tar_path)
            print(f"Downloaded to: {tar_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
    else:
        print(f"Dataset already downloaded: {tar_path}")
    
    # Verify MD5
    print("\nVerifying download integrity...")
    if not verify_md5(str(tar_path), CIFAR10_MD5):
        raise RuntimeError("MD5 verification failed. Download may be corrupted.")
    print("MD5 verification passed!")
    
    # Extract
    print("\nExtracting dataset...")
    extract_dir = RAW_DATA_DIR
    
    if not (extract_dir / "cifar-10-batches-py").exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        print(f"Extracted to: {extract_dir}")
    else:
        print("Dataset already extracted.")
    
    # Create metadata file for DVC
    metadata = {
        'dataset': 'CIFAR-10',
        'source': CIFAR10_URL,
        'classes': [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ],
        'num_classes': 10,
        'image_shape': [32, 32, 3],
        'train_samples': 50000,
        'test_samples': 10000
    }
    
    import json
    with open(RAW_DATA_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nDataset ready!")
    print(f"Location: {extract_dir / 'cifar-10-batches-py'}")
    
    return extract_dir / "cifar-10-batches-py"


def _load_cifar_batch(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single CIFAR-10 batch file.
    
    Args:
        filepath: Path to the batch file
        
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    with open(filepath, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        images = datadict['data']
        labels = datadict['labels']
        
        # Reshape images: (N, 3072) -> (N, 32, 32, 3)
        images = images.reshape((-1, 3, 32, 32))
        images = np.transpose(images, (0, 2, 3, 1))
        
        return images, np.array(labels)


def load_raw_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the raw CIFAR-10 dataset from downloaded files.
    
    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels)
    """
    data_dir = RAW_DATA_DIR / "cifar-10-batches-py"
    
    if not data_dir.exists():
        print("Dataset not found. Downloading...")
        data_dir = download_cifar10()
    
    # Load training data (5 batches)
    train_images = []
    train_labels = []
    
    for i in range(1, 6):
        batch_file = data_dir / f"data_batch_{i}"
        images, labels = _load_cifar_batch(str(batch_file))
        train_images.append(images)
        train_labels.append(labels)
    
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)
    
    # Load test data
    test_file = data_dir / "test_batch"
    test_images, test_labels = _load_cifar_batch(str(test_file))
    
    print(f"\nDataset loaded:")
    print(f"  Training samples: {len(train_images)}")
    print(f"  Test samples: {len(test_images)}")
    print(f"  Image shape: {train_images.shape[1:]}")
    
    return train_images, train_labels, test_images, test_labels


class DataIngestionConfig:
    """Configuration for data ingestion."""
    
    def __init__(self, config_path: str = "params.yaml"):
        import yaml
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        self.dataset_name = config.get('data', {}).get('dataset_name', 'cifar-10')
        self.validation_split = config.get('data', {}).get('validation_split', 0.1)
        self.random_seed = config.get('data', {}).get('random_seed', 42)


def ingest_data_step() -> Dict[str, Any]:
    """
    ZenML step for data ingestion.
    
    Downloads the CIFAR-10 dataset and prepares it for the pipeline.
    
    Returns:
        Dictionary containing dataset paths and metadata
    """
    from zenml import step
    
    print("\n" + "="*60)
    print("STEP: Data Ingestion")
    print("="*60)
    
    # Download dataset
    data_dir = download_cifar10()
    
    # Load raw data
    train_images, train_labels, test_images, test_labels = load_raw_data()
    
    # Prepare output
    output = {
        'data_dir': str(data_dir),
        'train_images_path': str(PROCESSED_DATA_DIR / "train_images.npy"),
        'train_labels_path': str(PROCESSED_DATA_DIR / "train_labels.npy"),
        'test_images_path': str(PROCESSED_DATA_DIR / "test_images.npy"),
        'test_labels_path': str(PROCESSED_DATA_DIR / "test_labels.npy"),
        'metadata': {
            'train_samples': len(train_images),
            'test_samples': len(test_images),
            'image_shape': list(train_images.shape[1:]),
            'num_classes': 10
        }
    }
    
    # Save processed data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DATA_DIR / "train_images.npy", train_images)
    np.save(PROCESSED_DATA_DIR / "train_labels.npy", train_labels)
    np.save(PROCESSED_DATA_DIR / "test_images.npy", test_images)
    np.save(PROCESSED_DATA_DIR / "test_labels.npy", test_labels)
    
    print(f"\nData saved to: {PROCESSED_DATA_DIR}")
    
    return output


if __name__ == "__main__":
    # Test data ingestion
    result = ingest_data_step()
    print("\nIngestion Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
