# OpenMLOps Challenge - CIFAR-10 CNN Classifier

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![ZenML](https://img.shields.io/badge/ZenML-0.55.5-purple.svg)](https://zenml.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9.2-blue.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.36.1-green.svg)](https://dvc.org/)
[![Evidently](https://img.shields.io/badge/Evidently-0.4.8-pink.svg)](https://evidentlyai.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docker.com/)

A complete MLOps workflow for image classification using open-source tools. This project implements a CNN classifier for the CIFAR-10 dataset with full MLOps lifecycle management.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Pipeline Details](#pipeline-details)
- [Monitoring & Drift Detection](#monitoring--drift-detection)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project demonstrates a production-ready MLOps workflow that includes:

| Component | Tool | Purpose |
|-----------|------|---------|
| **Source Control** | Git | Version control for code |
| **Data Versioning** | DVC | Track and version datasets with MinIO remote storage |
| **Experiment Tracking** | MLflow | Track experiments, metrics, parameters, and artifacts |
| **Pipeline Orchestration** | ZenML | Define and run ML pipelines |
| **Monitoring** | Evidently | Data drift detection and model monitoring |
| **Containerization** | Docker | All components run in containers |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenMLOps Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    Git       │    │    DVC       │    │   MinIO      │      │
│  │  (Source)    │◄───│  (Data)      │───►│  (Storage)   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    ZenML Pipeline                        │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │ Ingest  │→│Validate │→│ Split   │→│Preprocess│      │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │   │
│  │        │                                            │    │   │
│  │        ▼                                            ▼    │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │ Train   │→│Evaluate │→│Register │→│ Export  │       │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   MLflow     │    │  Evidently   │    │  Artifacts   │      │
│  │  (Tracking)  │    │ (Monitoring) │    │  (Storage)   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## ⚙️ Prerequisites

- **Docker** (>= 24.0)
- **Docker Compose** (>= 2.20)
- **Git** (>= 2.30)
- **Python** (>= 3.10) - for local development
- **8GB+ RAM** recommended
- **10GB+ disk space** for data and models

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd openmlops-challenge
```

### 2. Start Infrastructure Services

Start all required services (MinIO, MLflow, ZenML, PostgreSQL):

```bash
docker compose up -d
```

Wait for all services to be healthy (approximately 1-2 minutes):

```bash
docker compose ps
```

Expected output:
```
NAME                 STATUS    PORTS
openmlops-minio      running   9000, 9001
openmlops-mlflow     running   5000
openmlops-postgres   running   5432
openmlops-zenml      running   8080
```

### 3. Initialize MinIO Bucket for DVC

Create the required bucket for DVC:

```bash
# Using MinIO client (mc)
docker run --rm --network mlops-network minio/mc \
  bash -c "mc alias set myminio http://minio:9000 minioadmin minioadmin123 && \
           mc mb myminio/mlops-data --ignore-existing && \
           mc mb myminio/mlflow-artifacts --ignore-existing"
```

### 4. Initialize DVC

```bash
# Initialize DVC (if not already done)
dvc init

# Configure DVC remote
dvc remote add -d minio_remote s3://mlops-data
dvc remote modify minio_remote endpointurl http://localhost:9000
dvc remote modify minio_remote access_key_id minioadmin
dvc remote modify minio_remote secret_access_key minioadmin123
```

### 5. Download and Version Data

```bash
# Download CIFAR-10 dataset
dvc repro download_data

# Push data to remote storage
dvc push
```

### 6. Run Training Pipeline

```bash
# Run training pipeline with Docker
docker compose --profile training run training-pipeline

# OR run locally
python -m src.pipelines.training_pipeline
```

### 7. Monitor Training

Access MLflow UI at: http://localhost:5000

View:
- Experiment metrics
- Model parameters
- Training artifacts
- Model versions

### 8. Run Monitoring Pipeline

```bash
# Run monitoring pipeline
docker compose --profile monitoring run monitoring-pipeline

# OR run locally
python -m src.pipelines.monitoring_pipeline

# Run with simulated drift to test retrain trigger
python -m src.pipelines.monitoring_pipeline --simulate-drift
```

### 9. View Monitoring Reports

- **Evidently Reports**: `reports/drift_report_*.html`
- **MLflow Monitoring**: http://localhost:5000
- **ZenML Dashboard**: http://localhost:8080

## 📁 Project Structure

```
openmlops-challenge/
├── docker-compose.yml          # Main compose file for all services
├── dvc.yaml                    # DVC pipeline definition
├── params.yaml                 # Configuration parameters
├── requirements.txt            # Python dependencies
│
├── configs/
│   └── config.yaml             # Application configuration
│
├── docker/
│   ├── init-db.sql             # Database initialization
│   ├── mlflow/
│   │   └── Dockerfile          # MLflow server
│   ├── zenml/
│   │   └── Dockerfile          # ZenML server
│   ├── training/
│   │   └── Dockerfile          # Training container
│   ├── monitoring/
│   │   └── Dockerfile          # Monitoring container
│   └── jupyter/
│       └── Dockerfile          # Jupyter notebook server
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn_model.py        # CNN model architecture
│   │
│   ├── steps/
│   │   ├── __init__.py
│   │   ├── ingest_data.py      # Data ingestion step
│   │   ├── validate_data.py    # Data validation step
│   │   ├── split_data.py       # Data splitting step
│   │   ├── preprocess.py       # Preprocessing step
│   │   ├── train.py            # Training step
│   │   ├── evaluate.py         # Evaluation step
│   │   ├── register_model.py   # Model registration step
│   │   ├── export_model.py     # Model export step
│   │   └── monitoring_steps.py # Monitoring pipeline steps
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py    # Training pipeline
│   │   └── monitoring_pipeline.py  # Monitoring pipeline
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── evidently_monitor.py    # Evidently drift detection
│   │
│   └── utils/
│       ├── __init__.py
│       └── zenml_setup.py      # ZenML configuration
│
├── data/                       # Data directory (DVC tracked)
│   ├── raw/                    # Raw CIFAR-10 data
│   └── processed/              # Processed data
│
├── models/                     # Saved models
│   ├── trained_model.keras     # Trained model
│   └── serving/                # Serving-ready formats
│
├── reports/                    # Generated reports
│   ├── confusion_matrix.png
│   ├── evaluation_results.json
│   └── drift_report_*.html
│
├── inference_logs/             # Inference data logs
│
└── .dvc/                       # DVC configuration
    └── config
```

## 🔄 Pipeline Details

### Training Pipeline

The training pipeline consists of 8 sequential steps:

```python
@pipeline(name="training_pipeline")
def training_pipeline():
    # Step 1: Ingest data from DVC
    ingest_result = ingest_data_step()
    
    # Step 2: Validate data quality
    validation_report = validate_data_step(...)
    
    # Step 3: Split into train/val/test
    split_result = split_data_step(...)
    
    # Step 4: Preprocess images
    preprocess_result = preprocess_step(...)
    
    # Step 5: Train CNN model
    train_result = train_step(...)
    
    # Step 6: Evaluate on test set
    eval_result = evaluate_step(...)
    
    # Step 7: Register with MLflow
    register_result = register_model_step(...)
    
    # Step 8: Export for serving
    export_result = export_model_step(...)
```

### Monitoring Pipeline

The monitoring pipeline runs continuously to detect drift:

```python
@pipeline(name="monitoring_pipeline")
def monitoring_pipeline():
    # Step 1: Collect inference data
    inference_data = collect_inference_data_step(...)
    
    # Step 2: Run Evidently drift report
    drift_results = run_evidently_report_step(...)
    
    # Step 3: Make retrain decision
    decision = trigger_decision_step(drift_results)
    
    # Step 4: Store monitoring artifacts
    store_monitoring_artifacts_step(drift_results, decision)
```

## 📊 Monitoring & Drift Detection

### Drift Detection Features

1. **Data Drift Detection**
   - Feature distribution changes
   - Statistical tests (KS test, Chi-squared)
   - Per-column drift analysis

2. **Prediction Drift Detection**
   - Output distribution changes
   - KL divergence measurement
   - Class distribution shifts

3. **Performance Monitoring**
   - Accuracy tracking
   - Per-class performance
   - Degradation alerts

### Retrain Trigger Logic

Retraining is automatically triggered when:

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Data drift detected | > 5% features | Trigger retrain |
| Prediction drift | KL divergence > 0.1 | Trigger retrain |
| Performance drop | > 5% accuracy loss | Trigger retrain |

### Viewing Reports

1. **Evidently Reports**: Open `reports/drift_report_*.html` in browser
2. **MLflow Dashboard**: Navigate to http://localhost:5000
3. **ZenML Dashboard**: Navigate to http://localhost:8080

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio:9000` | MinIO endpoint |
| `AWS_ACCESS_KEY_ID` | `minioadmin` | MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | `minioadmin123` | MinIO secret key |
| `DVC_REMOTE_URL` | `s3://mlops-data` | DVC remote storage |

### Pipeline Parameters (`params.yaml`)

```yaml
# Training parameters
train:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  optimizer: adam

# Monitoring parameters
monitoring:
  drift_threshold: 0.05
  auto_retrain: true

# MLflow configuration
mlflow:
  experiment_name: cifar10_cnn_classifier
```

## 🛠️ Common Commands

### Infrastructure Management

```bash
# Start all services
docker compose up -d

# View service logs
docker compose logs -f mlflow
docker compose logs -f zenml

# Stop all services
docker compose down

# Remove all data (clean slate)
docker compose down -v
```

### DVC Commands

```bash
# Initialize DVC
dvc init

# Run specific pipeline stage
dvc repro download_data
dvc repro train

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# View pipeline status
dvc dag
```

### MLflow Commands

```bash
# List experiments
mlflow experiments list

# Download model
mlflow artifacts download --run-id <run_id>

# Serve model locally
mlflow models serve -m "models:/cifar10_classifier/Production" -p 5001
```

### ZenML Commands

```bash
# Initialize ZenML
zenml init

# List stacks
zenml stack list

# Create stack
zenml stack register mlops_stack ...

# Run pipeline
python -m src.pipelines.training_pipeline
```

## 🔧 Troubleshooting

### Common Issues

1. **MinIO Connection Failed**
   ```bash
   # Check MinIO is running
   docker compose ps minio
   
   # Check MinIO health
   curl http://localhost:9000/minio/health/live
   ```

2. **MLflow Database Error**
   ```bash
   # Restart PostgreSQL
   docker compose restart postgres
   
   # Check logs
   docker compose logs postgres
   ```

3. **DVC Push Failed**
   ```bash
   # Verify MinIO bucket exists
   mc ls myminio/
   
   # Check DVC remote config
   dvc remote list
   ```

4. **Out of Memory During Training**
   - Reduce batch size in `params.yaml`
   - Reduce number of epochs
   - Limit training data samples

### Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin123 |
| MLflow UI | http://localhost:5000 | - |
| ZenML Dashboard | http://localhost:8080 | - |



## 📄 License

This project is created for the OpenMLOps Challenge competition.

---

## 🎬 Demo Video Checklist

For the 2-5 minute demo video, showcase:

- [ ] `docker compose up` - Infrastructure startup
- [ ] Training run with MLflow tracking
- [ ] Evidently monitoring report generation
- [ ] Drift detection → Retrain trigger workflow

---

*Built with ❤️ for the OpenMLOps Challenge*
"# OpenMLOps-Challenge" 
