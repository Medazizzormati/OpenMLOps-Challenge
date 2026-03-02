#!/usr/bin/env python3
"""
Test Script for OpenMLOps Challenge

This script verifies that all components are correctly set up.
Run this after installing requirements.txt
"""

import sys
import os
from pathlib import Path

def check_file_structure():
    """Check that all required files exist."""
    print("\n" + "="*60)
    print("Checking File Structure...")
    print("="*60)
    
    required_files = [
        'docker-compose.yml',
        'dvc.yaml',
        'params.yaml',
        'requirements.txt',
        'README.md',
        '.gitignore',
        '.dvc/config',
        'configs/config.yaml',
        'docker/init-db.sql',
        'docker/mlflow/Dockerfile',
        'docker/zenml/Dockerfile',
        'docker/training/Dockerfile',
        'docker/monitoring/Dockerfile',
        'docker/jupyter/Dockerfile',
        'src/__init__.py',
        'src/models/cnn_model.py',
        'src/steps/ingest_data.py',
        'src/steps/validate_data.py',
        'src/steps/split_data.py',
        'src/steps/preprocess.py',
        'src/steps/train.py',
        'src/steps/evaluate.py',
        'src/steps/register_model.py',
        'src/steps/export_model.py',
        'src/steps/monitoring_steps.py',
        'src/monitoring/evidently_monitor.py',
        'src/pipelines/training_pipeline.py',
        'src/pipelines/monitoring_pipeline.py',
        'src/utils/zenml_setup.py',
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ MISSING: {file}")
            all_exist = False
    
    return all_exist


def check_python_syntax():
    """Check Python syntax for all .py files."""
    print("\n" + "="*60)
    print("Checking Python Syntax...")
    print("="*60)
    
    py_files = list(Path('.').rglob('*.py'))
    all_valid = True
    
    for py_file in py_files:
        try:
            with open(py_file, 'r') as f:
                compile(f.read(), str(py_file), 'exec')
            print(f"  ✓ {py_file}")
        except SyntaxError as e:
            print(f"  ✗ SYNTAX ERROR in {py_file}: {e}")
            all_valid = False
    
    return all_valid


def check_yaml_files():
    """Check YAML file syntax."""
    print("\n" + "="*60)
    print("Checking YAML Files...")
    print("="*60)
    
    try:
        import yaml
    except ImportError:
        print("  ! yaml module not installed, skipping")
        return True
    
    yaml_files = ['docker-compose.yml', 'dvc.yaml', 'params.yaml', 'configs/config.yaml']
    all_valid = True
    
    for yaml_file in yaml_files:
        if Path(yaml_file).exists():
            try:
                with open(yaml_file, 'r') as f:
                    yaml.safe_load(f)
                print(f"  ✓ {yaml_file}")
            except yaml.YAMLError as e:
                print(f"  ✗ YAML ERROR in {yaml_file}: {e}")
                all_valid = False
        else:
            print(f"  ✗ MISSING: {yaml_file}")
            all_valid = False
    
    return all_valid


def check_dockerfiles():
    """Check Dockerfile syntax."""
    print("\n" + "="*60)
    print("Checking Dockerfiles...")
    print("="*60)
    
    dockerfile_dirs = ['docker/mlflow', 'docker/zenml', 'docker/training', 
                       'docker/monitoring', 'docker/jupyter']
    
    all_valid = True
    for df_dir in dockerfile_dirs:
        df_path = Path(df_dir) / 'Dockerfile'
        if df_path.exists():
            with open(df_path, 'r') as f:
                content = f.read()
                if 'FROM' in content and 'WORKDIR' in content:
                    print(f"  ✓ {df_path}")
                else:
                    print(f"  ✗ INCOMPLETE: {df_path}")
                    all_valid = False
        else:
            print(f"  ✗ MISSING: {df_path}")
            all_valid = False
    
    return all_valid


def check_pipeline_steps():
    """Check that all required pipeline steps are defined."""
    print("\n" + "="*60)
    print("Checking Pipeline Steps...")
    print("="*60)
    
    # Training pipeline steps
    training_steps = [
        'setup_mlflow',
        'ingest_data',
        'validate_data',
        'split_data',
        'preprocess',
        'train_model',
        'evaluate_model',
        'register_model',
        'export_model'
    ]
    
    # Monitoring pipeline steps
    monitoring_steps = [
        'setup_mlflow_monitoring',
        'load_model_and_reference_data',
        'collect_inference_data',
        'run_evidently_report',
        'trigger_decision',
        'store_monitoring_artifacts'
    ]
    
    # Check training pipeline
    with open('src/pipelines/training_pipeline.py', 'r') as f:
        training_content = f.read()
    
    print("Training Pipeline Steps:")
    all_valid = True
    for step in training_steps:
        if f'def {step}' in training_content:
            print(f"  ✓ {step}")
        else:
            print(f"  ✗ MISSING: {step}")
            all_valid = False
    
    # Check monitoring pipeline
    with open('src/pipelines/monitoring_pipeline.py', 'r') as f:
        monitoring_content = f.read()
    
    print("\nMonitoring Pipeline Steps:")
    for step in monitoring_steps:
        if f'def {step}' in monitoring_content:
            print(f"  ✓ {step}")
        else:
            print(f"  ✗ MISSING: {step}")
            all_valid = False
    
    return all_valid


def check_docker_compose_services():
    """Check Docker Compose services."""
    print("\n" + "="*60)
    print("Checking Docker Compose Services...")
    print("="*60)
    
    try:
        import yaml
    except ImportError:
        print("  ! yaml module not installed, skipping")
        return True
    
    required_services = ['minio', 'mlflow', 'postgres', 'zenml']
    
    with open('docker-compose.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    services = config.get('services', {})
    
    all_present = True
    for service in required_services:
        if service in services:
            print(f"  ✓ {service}")
        else:
            print(f"  ✗ MISSING: {service}")
            all_present = False
    
    return all_present


def check_directories():
    """Check that required directories exist."""
    print("\n" + "="*60)
    print("Checking Directories...")
    print("="*60)
    
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'models',
        'models/serving',
        'reports',
        'inference_logs',
        'artifacts',
        'checkpoints'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ! {dir_path}/ (will be created)")
    
    # Create missing directories
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return True


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("OpenMLOps Challenge - Setup Verification")
    print("="*60)
    
    os.chdir(Path(__file__).parent)
    
    results = {
        'File Structure': check_file_structure(),
        'Python Syntax': check_python_syntax(),
        'YAML Files': check_yaml_files(),
        'Dockerfiles': check_dockerfiles(),
        'Pipeline Steps': check_pipeline_steps(),
        'Docker Services': check_docker_compose_services(),
        'Directories': check_directories()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {check}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL CHECKS PASSED! Project is ready to run.")
        print("\nNext steps:")
        print("  1. docker compose up -d")
        print("  2. python -m src.pipelines.training_pipeline")
        print("  3. python -m src.pipelines.monitoring_pipeline --simulate-drift")
    else:
        print("SOME CHECKS FAILED. Please fix the issues above.")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
