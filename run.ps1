# OpenMLOps Challenge - Complete Workflow Script (Windows PowerShell)
# This script runs the entire MLOps workflow from setup to monitoring

Write-Host "============================================================"
Write-Host "OpenMLOps Challenge - CIFAR-10 CNN Classifier"
Write-Host "Complete MLOps Workflow"
Write-Host "============================================================"

# Function to print section headers
function Print-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host $Title -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
}

# Step 1: Check Prerequisites
Print-Section "Step 1: Checking Prerequisites"

Write-Host "Checking Docker..."
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "  Docker: $(docker --version)"
} else {
    Write-Host "Docker not found. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

Write-Host "Checking Docker Compose..."
if (docker compose version) {
    Write-Host "  Docker Compose: installed"
} else {
    Write-Host "Docker Compose not found." -ForegroundColor Red
    exit 1
}

# Step 2: Start Infrastructure
Print-Section "Step 2: Starting Infrastructure Services"

Write-Host "Starting Docker services..."
docker compose up -d

Write-Host "Waiting for services to be healthy (30s)..."
Start-Sleep -Seconds 30

Write-Host "Checking service status..."
docker compose ps

# Step 3: Initialize MinIO
Print-Section "Step 3: Initializing MinIO Storage"

Write-Host "Creating MinIO buckets..."
docker run --rm --network openmlops_mlops-network minio/mc bash -c @"
    mc alias set myminio http://minio:9000 minioadmin minioadmin123 && \
    mc mb myminio/mlops-data --ignore-existing && \
    mc mb myminio/mlflow-artifacts --ignore-existing
"@

# Step 4: Initialize DVC
Print-Section "Step 4: Initializing DVC"

Write-Host "Configuring DVC remote..."
dvc remote add -d minio_remote s3://mlops-data --force 2>$null
dvc remote modify minio_remote endpointurl http://localhost:9000 2>$null
dvc remote modify minio_remote access_key_id minioadmin 2>$null
dvc remote modify minio_remote secret_access_key minioadmin123 2>$null

Write-Host "DVC remote configuration:"
dvc remote list

# Step 5: Download and Prepare Data
Print-Section "Step 5: Downloading CIFAR-10 Dataset"

Write-Host "Running data ingestion..."
python -c "from src.steps.ingest_data import download_cifar10; download_cifar10()"

Write-Host "Processing and splitting data..."
python -c @"
from src.steps.ingest_data import load_raw_data
from src.steps.split_data import split_data_step

train_images, train_labels, test_images, test_labels = load_raw_data()
split_data_step(train_images, train_labels, test_images, test_labels)
"@

Write-Host "Pushing data to DVC remote..."
dvc add data/
git add data/*.dvc .gitignore 2>$null
dvc push

# Step 6: Run Training Pipeline
Print-Section "Step 6: Running Training Pipeline"

Write-Host "Starting training pipeline..."
python -m src.pipelines.training_pipeline

# Step 7: View Training Results
Print-Section "Step 7: Training Results"

Write-Host "Opening MLflow UI..."
Write-Host "  URL: http://localhost:5000"
Write-Host ""
Write-Host "Model artifacts:"
Get-ChildItem models/

# Step 8: Run Monitoring Pipeline
Print-Section "Step 8: Running Monitoring Pipeline"

Write-Host "Running monitoring pipeline..."
python -m src.pipelines.monitoring_pipeline --n-samples 500

# Step 9: Simulate Drift
Print-Section "Step 9: Simulating Drift and Testing Retrain Trigger"

Write-Host "Running monitoring pipeline with simulated drift..."
python -m src.pipelines.monitoring_pipeline --simulate-drift --n-samples 500

# Final Summary
Print-Section "Workflow Complete!"

Write-Host ""
Write-Host "Summary:" -ForegroundColor Green
Write-Host "  • Infrastructure: Running"
Write-Host "  • Data: Downloaded and versioned"
Write-Host "  • Model: Trained and registered"
Write-Host "  • Monitoring: Active"
Write-Host "  • Drift detection: Tested"
Write-Host ""
Write-Host "Access Points:" -ForegroundColor Yellow
Write-Host "  • MLflow UI:      http://localhost:5000"
Write-Host "  • MinIO Console:  http://localhost:9001"
Write-Host "  • ZenML Dashboard: http://localhost:8080"
Write-Host ""
Write-Host "Reports:" -ForegroundColor Yellow
Get-ChildItem reports/

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "OpenMLOps Challenge workflow completed successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
