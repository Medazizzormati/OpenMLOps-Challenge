"""
Monitoring Pipeline Steps

This module contains all steps for the ZenML monitoring pipeline:
- collect_inference_data
- run_evidently_report
- trigger_decision
- store_monitoring_artifacts
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import yaml
import tensorflow as tf
from tensorflow import keras
import mlflow


class InferenceDataCollector:
    """
    Collects and stores inference data for monitoring.
    """
    
    def __init__(
        self,
        log_dir: str = "inference_logs",
        window_size: int = 1000
    ):
        """
        Initialize the data collector.
        
        Args:
            log_dir: Directory to store inference logs
            window_size: Number of samples to collect before saving
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        
    def collect(
        self,
        images: np.ndarray,
        predictions: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Collect inference data.
        
        Args:
            images: Input images
            predictions: Model predictions
            metadata: Optional metadata
            
        Returns:
            Collection summary
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save data
        log_file = self.log_dir / f"inference_{timestamp}.npz"
        np.savez(
            log_file,
            images=images,
            predictions=predictions,
            metadata=json.dumps(metadata or {})
        )
        
        print(f"  Inference data logged: {log_file}")
        
        return {
            'log_file': str(log_file),
            'samples_collected': len(images),
            'timestamp': timestamp
        }
    
    def get_recent_inferences(
        self,
        n_samples: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Get recent inference data.
        
        Args:
            n_samples: Number of samples to retrieve
            
        Returns:
            Dictionary with images and predictions
        """
        log_files = sorted(self.log_dir.glob("inference_*.npz"), reverse=True)
        
        all_images = []
        all_predictions = []
        
        for log_file in log_files:
            if len(all_images) >= n_samples:
                break
                
            data = np.load(log_file, allow_pickle=True)
            all_images.append(data['images'])
            all_predictions.append(data['predictions'])
        
        if not all_images:
            return {'images': np.array([]), 'predictions': np.array([])}
        
        images = np.concatenate(all_images)[:n_samples]
        predictions = np.concatenate(all_predictions)[:n_samples]
        
        return {
            'images': images,
            'predictions': predictions
        }
    
    def generate_simulated_inference(
        self,
        model: keras.Model,
        test_images: np.ndarray,
        n_samples: int = 500,
        apply_drift: bool = False,
        drift_intensity: float = 0.3
    ) -> Dict[str, np.ndarray]:
        """
        Generate simulated inference data for testing.
        
        Optionally applies drift to simulate real-world scenarios.
        
        Args:
            model: Trained model
            test_images: Test images to sample from
            n_samples: Number of samples to generate
            apply_drift: Whether to apply simulated drift
            drift_intensity: Intensity of drift to apply
            
        Returns:
            Dictionary with images and predictions
        """
        # Sample random images
        indices = np.random.choice(len(test_images), n_samples, replace=False)
        images = test_images[indices].copy()
        
        # Apply drift if requested
        if apply_drift:
            # Simulate brightness shift
            brightness_shift = np.random.uniform(
                -drift_intensity, drift_intensity
            )
            images = images + brightness_shift
            
            # Simulate noise
            noise = np.random.normal(0, drift_intensity * 0.5, images.shape)
            images = images + noise
            
            # Clip to valid range
            images = np.clip(images, images.min(), images.max())
        
        # Get predictions
        predictions = model.predict(images, verbose=0)
        
        # Log to inference logs
        self.collect(images, predictions, {
            'simulated': True,
            'drift_applied': apply_drift,
            'drift_intensity': drift_intensity if apply_drift else 0
        })
        
        return {
            'images': images,
            'predictions': predictions
        }


def collect_inference_data_step(
    model_path: str,
    test_images: np.ndarray,
    n_samples: int = 500,
    simulate_drift: bool = False
) -> Dict[str, Any]:
    """
    ZenML step to collect inference data.
    
    Args:
        model_path: Path to trained model
        test_images: Test images to sample from
        n_samples: Number of samples to collect
        simulate_drift: Whether to simulate drift
        
    Returns:
        Collected inference data
    """
    print("\n" + "="*60)
    print("STEP: Collect Inference Data")
    print("="*60)
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Create collector
    collector = InferenceDataCollector()
    
    # Generate inference data
    inference_data = collector.generate_simulated_inference(
        model=model,
        test_images=test_images,
        n_samples=n_samples,
        apply_drift=simulate_drift
    )
    
    print(f"\n  Collected {len(inference_data['images'])} inference samples")
    print(f"  Drift simulation: {'ON' if simulate_drift else 'OFF'}")
    
    return inference_data


def run_evidently_report_step(
    reference_images: np.ndarray,
    current_images: np.ndarray,
    reference_predictions: Optional[np.ndarray] = None,
    current_predictions: Optional[np.ndarray] = None,
    reference_metrics: Optional[Dict] = None,
    drift_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    ZenML step to run Evidently drift report.
    
    Args:
        reference_images: Reference images (training data)
        current_images: Current inference images
        reference_predictions: Reference predictions
        current_predictions: Current predictions
        reference_metrics: Reference metrics
        drift_threshold: Drift detection threshold
        
    Returns:
        Drift detection results
    """
    print("\n" + "="*60)
    print("STEP: Run Evidently Report")
    print("="*60)
    
    from src.monitoring.evidently_monitor import run_evidently_monitoring
    
    # Sample reference data if too large
    max_samples = 2000
    if len(reference_images) > max_samples:
        indices = np.random.choice(len(reference_images), max_samples, replace=False)
        reference_images = reference_images[indices]
        if reference_predictions is not None:
            reference_predictions = reference_predictions[indices]
    
    results = run_evidently_monitoring(
        reference_images=reference_images,
        current_images=current_images,
        reference_predictions=reference_predictions,
        current_predictions=current_predictions,
        reference_metrics=reference_metrics,
        current_metrics=None,
        drift_threshold=drift_threshold
    )
    
    # Print summary
    print(f"\n  Data Drift Detected: {results['data_drift'].get('dataset_drift', 'N/A')}")
    print(f"  Drift Share: {results['data_drift'].get('drift_share', 'N/A'):.2%}")
    
    if results.get('prediction_drift'):
        print(f"  Prediction Drift: {results['prediction_drift'].get('prediction_drift_detected', 'N/A')}")
        print(f"  KL Divergence: {results['prediction_drift'].get('kl_divergence', 'N/A'):.4f}")
    
    return results


def trigger_decision_step(
    drift_results: Dict[str, Any],
    drift_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    ZenML step to make retrain trigger decision.
    
    Analyzes drift results and determines if retraining is needed.
    
    Args:
        drift_results: Results from drift detection
        drift_threshold: Threshold for triggering retrain
        
    Returns:
        Decision results
    """
    print("\n" + "="*60)
    print("STEP: Trigger Decision")
    print("="*60)
    
    # Check data drift
    data_drift_detected = drift_results.get('data_drift', {}).get('dataset_drift', False)
    drift_share = drift_results.get('data_drift', {}).get('drift_share', 0)
    
    # Check prediction drift
    pred_drift_detected = drift_results.get('prediction_drift', {}).get('prediction_drift_detected', False)
    
    # Check performance degradation
    perf_degradation = drift_results.get('performance_degradation', {}).get('performance_degradation_detected', False)
    
    # Make decision
    should_retrain = False
    trigger_reasons = []
    
    if data_drift_detected:
        should_retrain = True
        trigger_reasons.append('data_drift')
    
    if pred_drift_detected:
        should_retrain = True
        trigger_reasons.append('prediction_drift')
    
    if perf_degradation:
        should_retrain = True
        trigger_reasons.append('performance_degradation')
    
    decision = {
        'should_retrain': should_retrain,
        'trigger_reasons': trigger_reasons,
        'data_drift_detected': data_drift_detected,
        'prediction_drift_detected': pred_drift_detected,
        'performance_degradation_detected': perf_degradation,
        'drift_share': float(drift_share),
        'timestamp': datetime.now().isoformat()
    }
    
    # Print decision
    print(f"\n  Retrain Trigger: {'YES' if should_retrain else 'NO'}")
    
    if trigger_reasons:
        print(f"  Trigger Reasons: {', '.join(trigger_reasons)}")
    
    if data_drift_detected:
        print(f"  Drift Share: {drift_share:.2%}")
    
    # Save decision
    decision_path = Path("reports") / "retrain_decision.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    with open(decision_path, 'w') as f:
        json.dump(decision, f, indent=2)
    
    return decision


def store_monitoring_artifacts_step(
    drift_results: Dict[str, Any],
    decision: Dict[str, Any],
    report_dir: str = "reports"
) -> Dict[str, Any]:
    """
    ZenML step to store monitoring artifacts.
    
    Saves all reports and logs to MLflow and local storage.
    
    Args:
        drift_results: Drift detection results
        decision: Retrain decision
        report_dir: Directory to save reports
        
    Returns:
        Storage summary
    """
    print("\n" + "="*60)
    print("STEP: Store Monitoring Artifacts")
    print("="*60)
    
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save complete monitoring report
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
            # Log metrics
            mlflow.log_metric('drift_share', decision.get('drift_share', 0))
            mlflow.log_metric('data_drift_detected', int(decision.get('data_drift_detected', False)))
            mlflow.log_metric('should_retrain', int(decision.get('should_retrain', False)))
            
            # Log artifacts
            mlflow.log_artifact(str(report_path))
            
            # Log drift report if exists
            drift_report_html = list(report_dir.glob("drift_report_*.html"))
            if drift_report_html:
                mlflow.log_artifact(str(drift_report_html[-1]))
            
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
        'artifacts_logged': True,
        'history_entries': len(history)
    }


if __name__ == "__main__":
    print("Monitoring pipeline steps module")
