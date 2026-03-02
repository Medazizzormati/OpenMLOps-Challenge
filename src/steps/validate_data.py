"""
Data Validation Step for CIFAR-10 Dataset

This module implements data validation to ensure dataset quality
before training.
"""

import os
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json


class DataValidator:
    """
    Validator for CIFAR-10 dataset quality checks.
    
    Performs the following validations:
    - Shape validation
    - Data type validation
    - Value range validation
    - Label validation
    - Missing value check
    - Class distribution check
    """
    
    def __init__(self):
        self.validation_results = []
        self.errors = []
        self.warnings = []
        
    def validate_shape(
        self, 
        images: np.ndarray, 
        labels: np.ndarray,
        expected_image_shape: Tuple[int, int, int] = (32, 32, 3),
        expected_num_classes: int = 10
    ) -> bool:
        """Validate image and label shapes."""
        is_valid = True
        
        # Check image shape
        if len(images.shape) != 4:
            self.errors.append(
                f"Images should be 4D array, got {len(images.shape)}D"
            )
            is_valid = False
        elif images.shape[1:] != expected_image_shape:
            self.errors.append(
                f"Image shape mismatch. Expected {expected_image_shape}, "
                f"got {images.shape[1:]}"
            )
            is_valid = False
            
        # Check labels shape
        if len(labels.shape) != 1:
            self.errors.append(
                f"Labels should be 1D array, got {len(labels.shape)}D"
            )
            is_valid = False
            
        # Check sample count match
        if len(images) != len(labels):
            self.errors.append(
                f"Mismatch between images ({len(images)}) and labels ({len(labels)})"
            )
            is_valid = False
            
        self.validation_results.append({
            'check': 'shape_validation',
            'passed': is_valid,
            'image_shape': list(images.shape),
            'label_shape': list(labels.shape)
        })
        
        return is_valid
    
    def validate_dtypes(
        self, 
        images: np.ndarray, 
        labels: np.ndarray
    ) -> bool:
        """Validate data types."""
        is_valid = True
        
        # Check image dtype
        if images.dtype not in [np.uint8, np.float32, np.float64]:
            self.warnings.append(
                f"Unusual image dtype: {images.dtype}. Expected uint8 or float."
            )
            
        # Check label dtype
        if labels.dtype not in [np.int32, np.int64, np.uint8]:
            self.errors.append(
                f"Invalid label dtype: {labels.dtype}. Expected integer type."
            )
            is_valid = False
            
        self.validation_results.append({
            'check': 'dtype_validation',
            'passed': is_valid,
            'image_dtype': str(images.dtype),
            'label_dtype': str(labels.dtype)
        })
        
        return is_valid
    
    def validate_value_range(
        self, 
        images: np.ndarray,
        expected_min: int = 0,
        expected_max: int = 255
    ) -> bool:
        """Validate pixel value range."""
        actual_min = images.min()
        actual_max = images.max()
        
        is_valid = (
            actual_min >= expected_min and 
            actual_max <= expected_max
        )
        
        if not is_valid:
            self.warnings.append(
                f"Pixel values outside expected range [{expected_min}, {expected_max}]. "
                f"Actual range: [{actual_min}, {actual_max}]"
            )
            
        self.validation_results.append({
            'check': 'value_range',
            'passed': is_valid,
            'min_value': int(actual_min),
            'max_value': int(actual_max)
        })
        
        return is_valid
    
    def validate_labels(
        self, 
        labels: np.ndarray,
        num_classes: int = 10
    ) -> bool:
        """Validate label values."""
        is_valid = True
        
        unique_labels = np.unique(labels)
        
        # Check for invalid labels
        if unique_labels.min() < 0:
            self.errors.append(
                f"Negative label found: {unique_labels.min()}"
            )
            is_valid = False
            
        if unique_labels.max() >= num_classes:
            self.errors.append(
                f"Label exceeds num_classes. Max label: {unique_labels.max()}, "
                f"num_classes: {num_classes}"
            )
            is_valid = False
            
        # Check all classes present
        if len(unique_labels) != num_classes:
            self.warnings.append(
                f"Not all classes present. Found {len(unique_labels)} classes, "
                f"expected {num_classes}"
            )
            
        self.validation_results.append({
            'check': 'label_validation',
            'passed': is_valid,
            'unique_labels': unique_labels.tolist(),
            'num_classes_found': len(unique_labels)
        })
        
        return is_valid
    
    def check_missing_values(
        self, 
        images: np.ndarray, 
        labels: np.ndarray
    ) -> bool:
        """Check for missing or corrupted values."""
        is_valid = True
        
        # Check for NaN/Inf in images
        if np.any(np.isnan(images)):
            count = np.sum(np.isnan(images))
            self.errors.append(f"NaN values found in images: {count}")
            is_valid = False
            
        if np.any(np.isinf(images)):
            count = np.sum(np.isinf(images))
            self.errors.append(f"Inf values found in images: {count}")
            is_valid = False
            
        # Check for NaN in labels
        if np.any(np.isnan(labels.astype(float))):
            count = np.sum(np.isnan(labels.astype(float)))
            self.errors.append(f"NaN values found in labels: {count}")
            is_valid = False
            
        self.validation_results.append({
            'check': 'missing_values',
            'passed': is_valid,
            'nan_count': int(np.sum(np.isnan(images.astype(float))))
        })
        
        return is_valid
    
    def analyze_class_distribution(
        self, 
        labels: np.ndarray,
        num_classes: int = 10
    ) -> Dict[int, int]:
        """Analyze and report class distribution."""
        distribution = {}
        for i in range(num_classes):
            distribution[i] = int(np.sum(labels == i))
            
        # Check for imbalance
        counts = list(distribution.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        # Flag significant imbalance (>20% deviation)
        for class_id, count in distribution.items():
            deviation = abs(count - mean_count) / mean_count if mean_count > 0 else 0
            if deviation > 0.2:
                self.warnings.append(
                    f"Class {class_id} shows imbalance. Count: {count}, "
                    f"Mean: {mean_count:.0f}, Deviation: {deviation*100:.1f}%"
                )
                
        self.validation_results.append({
            'check': 'class_distribution',
            'passed': True,
            'distribution': distribution,
            'mean_count': float(mean_count),
            'std_count': float(std_count)
        })
        
        return distribution
    
    def get_report(self) -> Dict[str, Any]:
        """Generate validation report."""
        all_passed = all(r['passed'] for r in self.validation_results)
        
        return {
            'overall_passed': all_passed and len(self.errors) == 0,
            'total_checks': len(self.validation_results),
            'passed_checks': sum(1 for r in self.validation_results if r['passed']),
            'errors': self.errors,
            'warnings': self.warnings,
            'details': self.validation_results
        }


def validate_data_step(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray
) -> Dict[str, Any]:
    """
    ZenML step for data validation.
    
    Validates the integrity and quality of the dataset.
    
    Args:
        train_images: Training images array
        train_labels: Training labels array
        test_images: Test images array
        test_labels: Test labels array
        
    Returns:
        Validation report dictionary
    """
    print("\n" + "="*60)
    print("STEP: Data Validation")
    print("="*60)
    
    validator = DataValidator()
    
    # Validate training data
    print("\nValidating training data...")
    validator.validate_shape(train_images, train_labels)
    validator.validate_dtypes(train_images, train_labels)
    validator.validate_value_range(train_images)
    validator.validate_labels(train_labels)
    validator.check_missing_values(train_images, train_labels)
    train_distribution = validator.analyze_class_distribution(train_labels)
    
    # Validate test data
    print("Validating test data...")
    validator.validate_shape(test_images, test_labels)
    validator.validate_dtypes(test_images, test_labels)
    validator.validate_value_range(test_images)
    validator.validate_labels(test_labels)
    validator.check_missing_values(test_images, test_labels)
    test_distribution = validator.analyze_class_distribution(test_labels)
    
    # Generate report
    report = validator.get_report()
    report['train_distribution'] = train_distribution
    report['test_distribution'] = test_distribution
    
    # Print summary
    print(f"\nValidation Summary:")
    print(f"  Total checks: {report['total_checks']}")
    print(f"  Passed: {report['passed_checks']}")
    print(f"  Errors: {len(report['errors'])}")
    print(f"  Warnings: {len(report['warnings'])}")
    
    if report['errors']:
        print("\nErrors:")
        for error in report['errors']:
            print(f"  ❌ {error}")
            
    if report['warnings']:
        print("\nWarnings:")
        for warning in report['warnings']:
            print(f"  ⚠️  {warning}")
    
    if report['overall_passed']:
        print("\n✅ Data validation PASSED")
    else:
        print("\n❌ Data validation FAILED")
        raise ValueError("Data validation failed. See errors above.")
    
    # Save validation report
    report_path = Path("reports/data_validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    # Test validation
    from ingest_data import load_raw_data
    
    train_images, train_labels, test_images, test_labels = load_raw_data()
    report = validate_data_step(train_images, train_labels, test_images, test_labels)
