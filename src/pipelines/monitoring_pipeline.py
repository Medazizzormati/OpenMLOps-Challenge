"""
ZenML Monitoring Pipeline for CIFAR-10 CNN Classifier

This module defines the monitoring pipeline with all required steps:
- collect_inference_data: Store inference logs
- run_evidently_report: Drift detection
- trigger_decision: Decide if retrain needed
- store_monitoring_artifacts: Save reports
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml
import mlflow
from zenml import pipeline, step
from zenml.config import DockerSettings


# Docker settings
docker_settings = DockerSettings(
    required_integrations=["mlflow"],
    requirements=[
        "tensorflow==2.15.0",
        "mlflow==2.9.2",
        "evidently==0.4.8",
        "numpy==1.26.2",
        "pandas==2.1.3",
        "scikit-learn==1.3.2",
    ]
)


# =============================================================================
# PIPELINE STEPS
# =============================================================================

@step
def setup_mlflow_monitoring() -> str:
    """
    Setup MLflow for monitoring.
    
    Returns:
        MLflow experiment name
    """
    experiment_name = "cifar10_monitoring"
    
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")
    
    return experiment_name


@step
def load_model_and_reference_data(
    model_path: str = "models/trained_model.keras",
    processed_dir: str = "data/processed"
) -> Dict[str, Any]:
    """
    Load trained model and reference data.
    
    Args:
        model_path: Path to trained model
        processed_dir: Path to processed data
        
    Returns:
        Dictionary with model and data
    """
    import tensorflow as tf
    from tensorflow import keras
    
    print("\n" + "="*60)
    print("STEP: Load Model and Data")
    print("="*60)
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load reference data (first 1000 training samples)
    X_ref = np.load(Path(processed_dir) / "X_train.npy")[:1000]
    X_ref_processed = np.load(Path(processed_dir) / "X_train_processed.npy")[:1000]
    
    # Load test data for inference simulation
    X_test = np.load(Path(processed_dir) / "X_test.npy")
    X_test_processed = np.load(Path(processed_dir) / "X_test_processed.npy")
    
    # Get reference predictions
    ref_predictions = model.predict(X_ref_processed, verbose=0)
    
    print(f"Model loaded: {model_path}")
    print(f"Reference data: {X_ref.shape}")
    print(f"Test data: {X_test.shape}")
    
    return {
        'model': model,
        'X_ref': X_ref,
        'X_ref_processed': X_ref_processed,
        'ref_predictions': ref_predictions,
        'X_test': X_test,
        'X_test_processed': X_test_processed
    }


@step
def collect_inference_data(
    data: Dict[str, Any],
    n_samples: int = 500,
    simulate_drift: bool = False
) -> Dict[str, Any]:
    """
    Collect inference data (simulated for demo).
    
    Args:
        data: Dictionary with model and test data
        n_samples: Number of samples to collect
        simulate_drift: Whether to simulate drift
        
    Returns:
        Dictionary with inference data
    """
    print("\n" + "="*60)
    print("STEP: Collect Inference Data")
    print("="*60)
    
    model = data['model']
    X_test = data['X_test']
    X_test_processed = data['X_test_processed']
    
    # Sample random test images
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    current_images = X_test[indices].copy()
    current_images_processed = X_test_processed[indices].copy()
    
    # Apply drift if simulating
    if simulate_drift:
        print("  Simulating data drift...")
        # Add brightness shift
        brightness_shift = np.random.uniform(0.2, 0.4)
        current_images_processed = current_images_processed + brightness_shift
        # Add noise
        noise = np.random.normal(0, 0.1, current_images_processed.shape)
        current_images_processed = current_images_processed + noise
        # Clip to valid range
        current_images_processed = np.clip(current_images_processed, 0, 1)
    
    # Get predictions
    current_predictions = model.predict(current_images_processed, verbose=0)
    
    # Store inference logs
    log_dir = Path("inference_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"inference_{timestamp}.npz"
    
    np.savez(
        log_file,
        images=current_images,
        predictions=current_predictions,
        metadata=json.dumps({
            'n_samples': n_samples,
            'simulate_drift': simulate_drift,
            'timestamp': timestamp
        })
    )
    
    print(f"  Collected {n_samples} inference samples")
    print(f"  Drift simulation: {'ON' if simulate_drift else 'OFF'}")
    print(f"  Log saved: {log_file}")
    
    return {
        'current_images': current_images,
        'current_images_processed': current_images_processed,
        'current_predictions': current_predictions,
        'log_file': str(log_file),
        'simulate_drift': simulate_drift
    }


@step
def run_evidently_report(
    data: Dict[str, Any],
    inference_data: Dict[str, Any],
    drift_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Run Evidently drift detection report.
    
    Args:
        data: Reference data dictionary
        inference_data: Current inference data
        drift_threshold: Threshold for drift detection
        
    Returns:
        Drift detection results
    """
    print("\n" + "="*60)
    print("STEP: Run Evidently Report")
    print("="*60)
    
    import pandas as pd
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import DatasetDriftMetric, DataDriftTable
    
    # Convert images to DataFrames with statistics
    def images_to_df(images):
        stats = []
        for i, img in enumerate(images[:500]):  # Limit to 500 for speed
            stats.append({
                'image_id': i,
                'mean_pixel': img.mean(),
                'std_pixel': img.std(),
                'min_pixel': img.min(),
                'max_pixel': img.max(),
            })
        return pd.DataFrame(stats)
    
    ref_df = images_to_df(data['X_ref_processed'])
    cur_df = images_to_df(inference_data['current_images_processed'])
    
    # Create drift report
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])
    
    report.run(reference_data=ref_df, current_data=cur_df)
    
    # Extract results
    results = report.as_dict()
    
    dataset_drift = results['metrics'][0]['result']['dataset_drift']
    drift_share = results['metrics'][0]['result']['drift_share']
    
    # Calculate prediction drift (KL divergence)
    ref_pred = data['ref_predictions']
    cur_pred = inference_data['current_predictions']
    
    ref_classes = np.argmax(ref_pred, axis=1)
    cur_classes = np.argmax(cur_pred, axis=1)
    
    ref_dist = np.bincount(ref_classes, minlength=10) / len(ref_classes)
    cur_dist = np.bincount(cur_classes, minlength=10) / len(cur_classes)
    
    # KL divergence
    epsilon = 1e-10
    kl_divergence = np.sum(ref_dist * np.log((ref_dist + epsilon) / (cur_dist + epsilon)))
    
    prediction_drift = kl_divergence > 0.1
    
    # Save report
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_path = report_dir / f"drift_report_{timestamp}.html"
    report.save_html(str(html_path))
    
    drift_results = {
        'dataset_drift': dataset_drift,
        'drift_share': float(drift_share),
        'prediction_drift': prediction_drift,
        'kl_divergence': float(kl_divergence),
        'html_report': str(html_path),
        'timestamp': timestamp
    }
    
    # Save JSON report
    json_path = report_dir / f"drift_report_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(drift_results, f, indent=2)
    
    print(f"  Data Drift Detected: {dataset_drift}")
    print(f"  Drift Share: {drift_share:.2%}")
    print(f"  Prediction Drift: {prediction_drift}")
    print(f"  KL Divergence: {kl_divergence:.4f}")
    print(f"  Report saved: {html_path}")
    
    return drift_results


@step
def trigger_decision(
    drift_results: Dict[str, Any],
    drift_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Make retrain trigger decision based on drift results.
    
    Args:
        drift_results: Results from drift detection
        drift_threshold: Threshold for triggering retrain
        
    Returns:
        Decision dictionary
    """
    print("\n" + "="*60)
    print("STEP: Trigger Decision")
    print("="*60)
    
    data_drift = drift_results.get('dataset_drift', False)
    prediction_drift = drift_results.get('prediction_drift', False)
    
    should_retrain = data_drift or prediction_drift
    trigger_reasons = []
    
    if data_drift:
        trigger_reasons.append('data_drift')
    if prediction_drift:
        trigger_reasons.append('prediction_drift')
    
    decision = {
        'should_retrain': should_retrain,
        'trigger_reasons': trigger_reasons,
        'data_drift_detected': data_drift,
        'prediction_drift_detected': prediction_drift,
        'drift_share': drift_results.get('drift_share', 0),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save decision
    decision_path = Path("reports/retrain_decision.json")
    with open(decision_path, 'w') as f:
        json.dump(decision, f, indent=2)
    
    # Print decision
    print(f"\n  Retrain Trigger: {'YES' if should_retrain else 'NO'}")
    if trigger_reasons:
        print(f"  Trigger Reasons: {', '.join(trigger_reasons)}")
    
    if should_retrain:
        # Create trigger file
        trigger_file = Path("retrain.trigger")
        with open(trigger_file, 'w') as f:
            f.write(f"Retrain triggered at {datetime.now().isoformat()}\n")
        print(f"  Trigger file created: {trigger_file}")
    
    return decision


@step
def store_monitoring_artifacts(
    drift_results: Dict[str, Any],
    decision: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Store monitoring artifacts to MLflow and local storage.
    
    Args:
        drift_results: Drift detection results
        decision: Retrain decision
        
    Returns:
        Storage summary
    """
    print("\n" + "="*60)
    print("STEP: Store Monitoring Artifacts")
    print("="*60)
    
    report_dir = Path("reports")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create comprehensive monitoring report
    full_report = {
        'drift_analysis': drift_results,
        'retrain_decision': decision,
        'timestamp': timestamp
    }
    
    report_path = report_dir / f"monitoring_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f"  Report saved: {report_path}")
    
    # Log to MLflow
    try:
        with mlflow.start_run(run_name=f"monitoring_{timestamp}"):
            mlflow.log_metric('drift_share', decision.get('drift_share', 0))
            mlflow.log_metric('data_drift_detected', int(decision.get('data_drift_detected', False)))
            mlflow.log_metric('should_retrain', int(decision.get('should_retrain', False)))
            
            if drift_results.get('html_report'):
                mlflow.log_artifact(drift_results['html_report'])
            
            mlflow.log_artifact(str(report_path))
            
            print("  Artifacts logged to MLflow")
    except Exception as e:
        print(f"  Warning: Could not log to MLflow: {e}")
    
    # Update monitoring history
    history_path = report_dir / "monitoring_history.json"
    history = []
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
    
    history.append({
        'timestamp': timestamp,
        'drift_detected': decision.get('data_drift_detected', False),
        'should_retrain': decision.get('should_retrain', False),
        'report_path': str(report_path)
    })
    
    # Keep last 100 entries
    history = history[-100:]
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"  Monitoring history updated: {len(history)} entries")
    
    return {
        'report_path': str(report_path),
        'artifacts_logged': True
    }


# =============================================================================
# PIPELINE DEFINITION
# =============================================================================

@pipeline(
    name="monitoring_pipeline",
    enable_cache=False,
    settings={"docker": docker_settings}
)
def monitoring_pipeline(
    model_path: str = "models/trained_model.keras",
    processed_dir: str = "data/processed",
    n_samples: int = 500,
    simulate_drift: bool = False,
    drift_threshold: float = 0.05
) -> None:
    """
    Complete monitoring pipeline for CIFAR-10 CNN classifier.
    
    Steps:
    1. Setup MLflow
    2. Load model and reference data
    3. Collect inference data
    4. Run Evidently report
    5. Make trigger decision
    6. Store artifacts
    """
    print("\n" + "="*70)
    print("MONITORING PIPELINE: CIFAR-10 CNN Classifier")
    print("="*70)
    
    # Step 1: Setup MLflow
    experiment_name = setup_mlflow_monitoring()
    
    # Step 2: Load model and data
    data = load_model_and_reference_data(model_path, processed_dir)
    
    # Step 3: Collect inference data
    inference_data = collect_inference_data(data, n_samples, simulate_drift)
    
    # Step 4: Run Evidently report
    drift_results = run_evidently_report(data, inference_data, drift_threshold)
    
    # Step 5: Make trigger decision
    decision = trigger_decision(drift_results, drift_threshold)
    
    # Step 6: Store artifacts
    storage_result = store_monitoring_artifacts(drift_results, decision)
    
    print("\n" + "="*70)
    print("MONITORING PIPELINE COMPLETE")
    print("="*70)
    print(f"Drift Detected: {decision['data_drift_detected']}")
    print(f"Retrain Triggered: {decision['should_retrain']}")


def run_monitoring_pipeline(
    model_path: str = "models/trained_model.keras",
    simulate_drift: bool = False
):
    """
    Run the monitoring pipeline.
    
    Args:
        model_path: Path to trained model
        simulate_drift: Whether to simulate drift for testing
    """
    monitoring_pipeline(
        model_path=model_path,
        simulate_drift=simulate_drift
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run monitoring pipeline")
    parser.add_argument('--model-path', default='models/trained_model.keras')
    parser.add_argument('--simulate-drift', action='store_true')
    parser.add_argument('--n-samples', type=int, default=500)
    
    args = parser.parse_args()
    
    run_monitoring_pipeline(
        model_path=args.model_path,
        simulate_drift=args.simulate_drift
    )
