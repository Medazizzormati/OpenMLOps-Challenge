"""
Evidently Monitoring Module for Drift Detection

This module implements drift detection using Evidently AI
for monitoring model performance and data quality.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
    ColumnSummaryMetric,
    ClassificationQualityMetric
)
from evidently.metrics.base_metric import generate_column_metrics


class DriftMonitor:
    """
    Monitor for detecting data drift using Evidently.
    
    Monitors:
    - Data drift (feature distribution changes)
    - Prediction drift
    - Model performance degradation
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        drift_threshold: float = 0.05,
        output_dir: str = "reports"
    ):
        """
        Initialize the drift monitor.
        
        Args:
            reference_data: Reference dataset (training data)
            drift_threshold: Threshold for drift detection (p-value)
            output_dir: Directory to save reports
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        column_mapping: Optional[ColumnMapping] = None
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.
        
        Args:
            current_data: Current dataset to compare
            column_mapping: Column mapping for Evidently
            
        Returns:
            Drift detection results
        """
        print("\nRunning drift detection...")
        
        # Create data drift report
        data_drift_report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
        
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Extract results
        results = data_drift_report.as_dict()
        
        # Parse drift metrics
        drift_metrics = {
            'dataset_drift': results['metrics'][0]['result']['dataset_drift'],
            'drift_share': results['metrics'][0]['result']['drift_share'],
            'number_of_drifted_columns': results['metrics'][0]['result']['number_of_drifted_columns'],
            'number_of_columns': results['metrics'][0]['result']['number_of_columns'],
            'drifted_columns': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Get per-column drift details
        if 'metrics' in results and len(results['metrics']) > 1:
            drift_table = results['metrics'][1]['result']
            if 'drift_by_columns' in drift_table:
                for col_name, col_data in drift_table['drift_by_columns'].items():
                    if col_data.get('drift_detected', False):
                        drift_metrics['drifted_columns'].append({
                            'column': col_name,
                            'drift_score': col_data.get('drift_score', 0),
                            'p_value': col_data.get('p_value', 1)
                        })
        
        # Save HTML report
        report_path = self.output_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        data_drift_report.save_html(str(report_path))
        print(f"  Drift report saved: {report_path}")
        
        # Save JSON report
        json_path = report_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(drift_metrics, f, indent=2)
        print(f"  Drift metrics saved: {json_path}")
        
        return drift_metrics
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect drift in model predictions.
        
        Args:
            reference_predictions: Predictions on reference data
            current_predictions: Predictions on current data
            class_names: Optional class names
            
        Returns:
            Prediction drift results
        """
        print("\nAnalyzing prediction drift...")
        
        # Convert to DataFrames
        ref_pred_df = pd.DataFrame({
            'prediction': reference_predictions,
            'prediction_index': np.argmax(reference_predictions, axis=1) 
                               if len(reference_predictions.shape) > 1 
                               else reference_predictions
        })
        
        cur_pred_df = pd.DataFrame({
            'prediction': current_predictions,
            'prediction_index': np.argmax(current_predictions, axis=1)
                               if len(current_predictions.shape) > 1
                               else current_predictions
        })
        
        # Calculate class distribution
        ref_dist = np.bincount(ref_pred_df['prediction_index'], minlength=10)
        cur_dist = np.bincount(cur_pred_df['prediction_index'], minlength=10)
        
        # Normalize
        ref_dist = ref_dist / ref_dist.sum()
        cur_dist = cur_dist / cur_dist.sum()
        
        # Calculate KL divergence
        def kl_divergence(p, q):
            """Calculate KL divergence."""
            epsilon = 1e-10
            p = np.clip(p, epsilon, 1)
            q = np.clip(q, epsilon, 1)
            return np.sum(p * np.log(p / q))
        
        kl_div = kl_divergence(ref_dist, cur_dist)
        
        results = {
            'prediction_drift_detected': kl_div > 0.1,
            'kl_divergence': float(kl_div),
            'reference_distribution': ref_dist.tolist(),
            'current_distribution': cur_dist.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        if class_names:
            results['class_names'] = class_names
        
        return results
    
    def check_performance_degradation(
        self,
        reference_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Check for model performance degradation.
        
        Args:
            reference_metrics: Metrics on reference data
            current_metrics: Current metrics
            threshold: Degradation threshold (relative)
            
        Returns:
            Degradation analysis results
        """
        print("\nChecking for performance degradation...")
        
        degradation = {}
        significant_degradation = False
        
        for metric_name, ref_value in reference_metrics.items():
            if metric_name in current_metrics:
                cur_value = current_metrics[metric_name]
                
                # Calculate relative change
                if ref_value != 0:
                    relative_change = (cur_value - ref_value) / abs(ref_value)
                else:
                    relative_change = 0
                
                # For metrics like accuracy (higher is better)
                is_degraded = relative_change < -threshold
                significant_degradation = significant_degradation or is_degraded
                
                degradation[metric_name] = {
                    'reference_value': float(ref_value),
                    'current_value': float(cur_value),
                    'relative_change': float(relative_change),
                    'is_degraded': is_degraded
                }
        
        results = {
            'performance_degradation_detected': significant_degradation,
            'metrics': degradation,
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        return results


def images_to_dataframe(images: np.ndarray) -> pd.DataFrame:
    """
    Convert image arrays to DataFrame for Evidently.
    
    Args:
        images: Image array (N, H, W, C)
        
    Returns:
        DataFrame with image statistics
    """
    # Compute image statistics
    stats = []
    for i, img in enumerate(images):
        stats.append({
            'image_id': i,
            'mean_pixel': img.mean(),
            'std_pixel': img.std(),
            'min_pixel': img.min(),
            'max_pixel': img.max(),
            'mean_r': img[:,:,0].mean() if img.shape[-1] >= 1 else 0,
            'mean_g': img[:,:,1].mean() if img.shape[-1] >= 2 else 0,
            'mean_b': img[:,:,2].mean() if img.shape[-1] >= 3 else 0,
        })
    
    return pd.DataFrame(stats)


def run_evidently_monitoring(
    reference_images: np.ndarray,
    current_images: np.ndarray,
    reference_predictions: Optional[np.ndarray] = None,
    current_predictions: Optional[np.ndarray] = None,
    reference_metrics: Optional[Dict[str, float]] = None,
    current_metrics: Optional[Dict[str, float]] = None,
    drift_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Run complete Evidently monitoring suite.
    
    Args:
        reference_images: Reference (training) images
        current_images: Current (inference) images
        reference_predictions: Reference predictions
        current_predictions: Current predictions
        reference_metrics: Reference metrics
        current_metrics: Current metrics
        drift_threshold: Drift detection threshold
        
    Returns:
        Complete monitoring results
    """
    print("\n" + "="*60)
    print("Evidently Drift Monitoring")
    print("="*60)
    
    # Convert images to DataFrames
    ref_df = images_to_dataframe(reference_images)
    cur_df = images_to_dataframe(current_images)
    
    # Create monitor
    monitor = DriftMonitor(
        reference_data=ref_df,
        drift_threshold=drift_threshold
    )
    
    # Run data drift detection
    data_drift = monitor.detect_data_drift(cur_df)
    
    # Run prediction drift detection if predictions provided
    prediction_drift = None
    if reference_predictions is not None and current_predictions is not None:
        prediction_drift = monitor.detect_prediction_drift(
            reference_predictions, current_predictions
        )
    
    # Check performance degradation if metrics provided
    performance_check = None
    if reference_metrics and current_metrics:
        performance_check = monitor.check_performance_degradation(
            reference_metrics, current_metrics
        )
    
    # Compile results
    results = {
        'data_drift': data_drift,
        'prediction_drift': prediction_drift,
        'performance_degradation': performance_check,
        'overall_drift_detected': data_drift.get('dataset_drift', False),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save combined report
    report_path = Path("reports") / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nMonitoring report saved: {report_path}")
    
    return results


if __name__ == "__main__":
    # Test monitoring
    print("Evidently monitoring module")
