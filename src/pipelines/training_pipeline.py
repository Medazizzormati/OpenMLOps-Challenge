"""
ZenML Training Pipeline for CIFAR-10 CNN Classifier

This module defines the complete training pipeline with all required steps:
- ingest_data: Download and load CIFAR-10 via DVC
- validate_data: Ensure data quality
- split_data: Create train/val/test splits
- preprocess: Normalize and augment images
- train: Train CNN model
- evaluate: Compute metrics and generate reports
- register_model: Register with MLflow
- export_model: Export for serving-ready formats
"""

import os
import sys
import json
from typing import Dict, Any, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import yaml
import mlflow
import mlflow.tensorflow
from zenml import pipeline, step
from zenml.config import DockerSettings


# Docker settings for containerized execution
docker_settings = DockerSettings(
    required_integrations=["mlflow"],
    requirements=[
        "tensorflow==2.15.0",
        "mlflow==2.9.2",
        "numpy==1.26.2",
        "pandas==2.1.3",
        "scikit-learn==1.3.2",
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
        "boto3==1.34.0",
        "evidently==0.4.8",
    ]
)


# =============================================================================
# PIPELINE STEPS
# =============================================================================

@step
def setup_mlflow(config_path: str = "params.yaml") -> str:
    """
    Setup MLflow experiment and tracking.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        MLflow experiment name
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mlflow_config = config.get('mlflow', {})
    experiment_name = mlflow_config.get('experiment_name', 'cifar10_cnn_classifier')
    
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")
    
    return experiment_name


@step
def ingest_data() -> Dict[str, Any]:
    """
    Download and load CIFAR-10 dataset via DVC.
    
    Returns:
        Dictionary containing dataset paths and metadata
    """
    import pickle
    import tarfile
    import urllib.request
    
    print("\n" + "="*60)
    print("STEP: Data Ingestion")
    print("="*60)
    
    # Constants
    CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    RAW_DATA_DIR = Path("data/raw/cifar-10")
    PROCESSED_DATA_DIR = Path("data/processed")
    
    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    tar_path = RAW_DATA_DIR / "cifar-10-python.tar.gz"
    
    # Download if not exists
    if not tar_path.exists():
        print(f"Downloading CIFAR-10 from: {CIFAR10_URL}")
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)
        print(f"Downloaded to: {tar_path}")
    
    # Extract
    if not (RAW_DATA_DIR / "cifar-10-batches-py").exists():
        print("Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(RAW_DATA_DIR)
    
    # Load data
    def load_batch(filepath):
        with open(filepath, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            images = datadict['data'].reshape((-1, 3, 32, 32))
            images = np.transpose(images, (0, 2, 3, 1))
            return images, np.array(datadict['labels'])
    
    data_dir = RAW_DATA_DIR / "cifar-10-batches-py"
    
    # Load training data
    train_images = []
    train_labels = []
    for i in range(1, 6):
        images, labels = load_batch(str(data_dir / f"data_batch_{i}"))
        train_images.append(images)
        train_labels.append(labels)
    
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)
    
    # Load test data
    test_images, test_labels = load_batch(str(data_dir / "test_batch"))
    
    # Save processed data
    np.save(PROCESSED_DATA_DIR / "train_images.npy", train_images)
    np.save(PROCESSED_DATA_DIR / "train_labels.npy", train_labels)
    np.save(PROCESSED_DATA_DIR / "test_images.npy", test_images)
    np.save(PROCESSED_DATA_DIR / "test_labels.npy", test_labels)
    
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Image shape: {train_images.shape[1:]}")
    
    return {
        'train_samples': len(train_images),
        'test_samples': len(test_images),
        'image_shape': list(train_images.shape[1:]),
        'processed_dir': str(PROCESSED_DATA_DIR)
    }


@step
def validate_data(processed_dir: str) -> Dict[str, Any]:
    """
    Validate data quality and integrity.
    
    Args:
        processed_dir: Directory with processed data
        
    Returns:
        Validation report dictionary
    """
    print("\n" + "="*60)
    print("STEP: Data Validation")
    print("="*60)
    
    processed_dir = Path(processed_dir)
    
    train_images = np.load(processed_dir / "train_images.npy")
    train_labels = np.load(processed_dir / "train_labels.npy")
    test_images = np.load(processed_dir / "test_images.npy")
    test_labels = np.load(processed_dir / "test_labels.npy")
    
    errors = []
    warnings = []
    
    # Validate shapes
    if train_images.shape[1:] != (32, 32, 3):
        errors.append(f"Invalid train image shape: {train_images.shape[1:]}")
    if test_images.shape[1:] != (32, 32, 3):
        errors.append(f"Invalid test image shape: {test_images.shape[1:]}")
    
    # Validate labels
    if train_labels.min() < 0 or train_labels.max() > 9:
        errors.append(f"Invalid label range: [{train_labels.min()}, {train_labels.max()}]")
    
    # Check for missing values
    if np.any(np.isnan(train_images.astype(float))):
        errors.append("NaN values found in training images")
    
    print(f"Validation passed: {len(errors)} errors, {len(warnings)} warnings")
    
    if errors:
        raise ValueError(f"Data validation failed: {errors}")
    
    return {
        'passed': True,
        'errors': errors,
        'warnings': warnings,
        'train_shape': list(train_images.shape),
        'test_shape': list(test_images.shape)
    }


@step
def split_data(processed_dir: str, config_path: str = "params.yaml") -> Dict[str, Any]:
    """
    Split training data into train and validation sets.
    
    Args:
        processed_dir: Directory with processed data
        config_path: Path to configuration file
        
    Returns:
        Dictionary with split information
    """
    print("\n" + "="*60)
    print("STEP: Data Splitting")
    print("="*60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    validation_split = config.get('data', {}).get('validation_split', 0.1)
    random_seed = config.get('data', {}).get('random_seed', 42)
    
    processed_dir = Path(processed_dir)
    
    train_images = np.load(processed_dir / "train_images.npy")
    train_labels = np.load(processed_dir / "train_labels.npy")
    test_images = np.load(processed_dir / "test_images.npy")
    test_labels = np.load(processed_dir / "test_labels.npy")
    
    # Stratified split
    np.random.seed(random_seed)
    
    train_indices = []
    val_indices = []
    
    for label in range(10):
        class_indices = np.where(train_labels == label)[0]
        np.random.shuffle(class_indices)
        n_val = int(len(class_indices) * validation_split)
        val_indices.extend(class_indices[:n_val])
        train_indices.extend(class_indices[n_val:])
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    X_train = train_images[train_indices]
    y_train = train_labels[train_indices]
    X_val = train_images[val_indices]
    y_val = train_labels[val_indices]
    
    # Save splits
    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "X_val.npy", X_val)
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "X_test.npy", test_images)
    np.save(processed_dir / "y_test.npy", test_labels)
    
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(test_images)}")
    
    return {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(test_images),
        'processed_dir': str(processed_dir)
    }


@step
def preprocess(processed_dir: str, config_path: str = "params.yaml") -> Dict[str, Any]:
    """
    Preprocess images with normalization.
    
    Args:
        processed_dir: Directory with split data
        config_path: Path to configuration file
        
    Returns:
        Dictionary with preprocessing metadata
    """
    print("\n" + "="*60)
    print("STEP: Data Preprocessing")
    print("="*60)
    
    processed_dir = Path(processed_dir)
    
    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    
    # Normalize to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # Save preprocessed data
    np.save(processed_dir / "X_train_processed.npy", X_train)
    np.save(processed_dir / "X_val_processed.npy", X_val)
    np.save(processed_dir / "X_test_processed.npy", X_test)
    
    print(f"Data normalized to [0, 1]")
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return {
        'processed_dir': str(processed_dir),
        'train_mean': float(X_train.mean()),
        'train_std': float(X_train.std())
    }


@step(enable_cache=False)
def train_model(processed_dir: str, config_path: str = "params.yaml") -> Dict[str, Any]:
    """
    Train the CNN model.
    
    Args:
        processed_dir: Directory with preprocessed data
        config_path: Path to configuration file
        
    Returns:
        Dictionary with model path and metrics
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks
    
    print("\n" + "="*60)
    print("STEP: Model Training")
    print("="*60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_config = config.get('train', {})
    epochs = train_config.get('epochs', 50)
    batch_size = train_config.get('batch_size', 64)
    learning_rate = train_config.get('learning_rate', 0.001)
    
    processed_dir = Path(processed_dir)
    
    # Load data
    X_train = np.load(processed_dir / "X_train_processed.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_val = np.load(processed_dir / "X_val_processed.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    
    # Build CNN model
    inputs = layers.Input(shape=(32, 32, 3), name='input_layer')
    
    x = layers.Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='cifar10_cnn')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    model_path = Path("models/trained_model.keras")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    cb_list = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(str(model_path), save_best_only=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1
    )
    
    best_val_acc = max(history.history['val_accuracy'])
    
    # Save model
    model.save(model_path)
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved: {model_path}")
    
    return {
        'model_path': str(model_path),
        'best_val_accuracy': float(best_val_acc),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1])
    }


@step
def evaluate_model(model_path: str, processed_dir: str) -> Dict[str, Any]:
    """
    Evaluate the trained model on test data.
    
    Args:
        model_path: Path to trained model
        processed_dir: Directory with test data
        
    Returns:
        Dictionary with evaluation metrics
    """
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print("\n" + "="*60)
    print("STEP: Model Evaluation")
    print("="*60)
    
    model = keras.models.load_model(model_path)
    
    processed_dir = Path(processed_dir)
    X_test = np.load(processed_dir / "X_test_processed.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    
    # Predict
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Log to MLflow
    mlflow.log_metrics({
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1
    })
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_macro': float(f1)
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1: {f1:.4f}")
    
    return {
        'metrics': metrics,
        'model_path': model_path
    }


@step
def register_model(model_path: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Register model with MLflow Model Registry.
    
    Args:
        model_path: Path to trained model
        metrics: Evaluation metrics
        
    Returns:
        Dictionary with registration info
    """
    print("\n" + "="*60)
    print("STEP: Model Registration")
    print("="*60)
    
    model_name = "cifar10_classifier"
    
    # Log model to MLflow
    mlflow.tensorflow.log_model(
        tf_saved_model_dir=model_path,
        tf_meta_graph_tags=["serve"],
        artifact_path="model",
        registered_model_name=model_name
    )
    
    print(f"Model registered: {model_name}")
    
    return {
        'model_name': model_name,
        'model_path': model_path,
        'metrics': metrics
    }


@step
def export_model(model_path: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Export model in serving-ready formats.
    
    Args:
        model_path: Path to trained model
        metrics: Model metrics
        
    Returns:
        Dictionary with export paths
    """
    import tensorflow as tf
    from tensorflow import keras
    
    print("\n" + "="*60)
    print("STEP: Model Export")
    print("="*60)
    
    model = keras.models.load_model(model_path)
    
    export_dir = Path("models/serving")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Export SavedModel
    savedmodel_path = export_dir / "savedmodel"
    tf.saved_model.save(model, str(savedmodel_path))
    print(f"SavedModel exported: {savedmodel_path}")
    
    # Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = export_dir / "model.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model exported: {tflite_path}")
    
    # Save model card
    model_card = {
        'name': 'CIFAR-10 CNN Classifier',
        'accuracy': metrics.get('accuracy', 0),
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck'],
        'input_shape': [32, 32, 3]
    }
    
    with open(export_dir / "model_card.json", 'w') as f:
        json.dump(model_card, f, indent=2)
    
    return {
        'savedmodel_path': str(savedmodel_path),
        'tflite_path': str(tflite_path),
        'model_card_path': str(export_dir / "model_card.json")
    }


# =============================================================================
# PIPELINE DEFINITION
# =============================================================================

@pipeline(
    name="training_pipeline",
    enable_cache=False,
    settings={"docker": docker_settings}
)
def training_pipeline(config_path: str = "params.yaml") -> None:
    """
    Complete training pipeline for CIFAR-10 CNN classifier.
    
    Steps:
    1. Setup MLflow
    2. Ingest data
    3. Validate data
    4. Split data
    5. Preprocess
    6. Train model
    7. Evaluate model
    8. Register model
    9. Export model
    """
    print("\n" + "="*70)
    print("TRAINING PIPELINE: CIFAR-10 CNN Classifier")
    print("="*70)
    
    # Step 1: Setup MLflow
    experiment_name = setup_mlflow(config_path)
    
    # Step 2: Ingest data
    ingest_result = ingest_data()
    
    # Step 3: Validate data
    validation_result = validate_data(ingest_result['processed_dir'])
    
    # Step 4: Split data
    split_result = split_data(ingest_result['processed_dir'], config_path)
    
    # Step 5: Preprocess
    preprocess_result = preprocess(split_result['processed_dir'], config_path)
    
    # Step 6: Train model
    train_result = train_model(preprocess_result['processed_dir'], config_path)
    
    # Step 7: Evaluate model
    eval_result = evaluate_model(
        train_result['model_path'],
        preprocess_result['processed_dir']
    )
    
    # Step 8: Register model
    register_result = register_model(
        train_result['model_path'],
        eval_result['metrics']
    )
    
    # Step 9: Export model
    export_result = export_model(
        train_result['model_path'],
        eval_result['metrics']
    )
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)


def run_training_pipeline():
    """Run the training pipeline."""
    training_pipeline()


if __name__ == "__main__":
    run_training_pipeline()
