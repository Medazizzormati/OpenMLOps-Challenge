#!/bin/bash
# OpenMLOps Challenge - Complete Workflow Script
# This script runs the entire MLOps workflow from setup to monitoring

set -e

echo "============================================================"
echo "OpenMLOps Challenge - CIFAR-10 CNN Classifier"
echo "Complete MLOps Workflow"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}============================================================${NC}"
}

# Function to check command success
check_success() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 completed successfully${NC}"
    else
        echo -e "${RED}✗ $1 failed${NC}"
        exit 1
    fi
}

# Step 1: Check Prerequisites
print_section "Step 1: Checking Prerequisites"

echo "Checking Docker..."
if command -v docker &> /dev/null; then
    echo "  Docker: $(docker --version)"
else
    echo -e "${RED}Docker not found. Please install Docker.${NC}"
    exit 1
fi

echo "Checking Docker Compose..."
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    echo "  Docker Compose: installed"
else
    echo -e "${RED}Docker Compose not found. Please install Docker Compose.${NC}"
    exit 1
fi

# Step 2: Start Infrastructure
print_section "Step 2: Starting Infrastructure Services"

echo "Starting Docker services..."
docker compose up -d

echo "Waiting for services to be healthy (30s)..."
sleep 30

echo "Checking service status..."
docker compose ps

check_success "Infrastructure startup"

# Step 3: Initialize MinIO
print_section "Step 3: Initializing MinIO Storage"

echo "Creating MinIO buckets..."
docker run --rm --network openmlops_mlops-network minio/mc bash -c "\
    mc alias set myminio http://minio:9000 minioadmin minioadmin123 && \
    mc mb myminio/mlops-data --ignore-existing && \
    mc mb myminio/mlflow-artifacts --ignore-existing" || echo "MinIO buckets may already exist"

check_success "MinIO initialization"

# Step 4: Initialize DVC
print_section "Step 4: Initializing DVC"

echo "Configuring DVC remote..."
dvc remote add -d minio_remote s3://mlops-data --force 2>/dev/null || true
dvc remote modify minio_remote endpointurl http://localhost:9000 2>/dev/null || true
dvc remote modify minio_remote access_key_id minioadmin 2>/dev/null || true
dvc remote modify minio_remote secret_access_key minioadmin123 2>/dev/null || true

echo "DVC remote configuration:"
dvc remote list

check_success "DVC configuration"

# Step 5: Download and Prepare Data
print_section "Step 5: Downloading CIFAR-10 Dataset"

echo "Running data ingestion..."
python -c "from src.steps.ingest_data import download_cifar10; download_cifar10()"

echo "Processing and splitting data..."
python -c "
from src.steps.ingest_data import load_raw_data
from src.steps.split_data import split_data_step

train_images, train_labels, test_images, test_labels = load_raw_data()
split_data_step(train_images, train_labels, test_images, test_labels)
"

echo "Pushing data to DVC remote..."
dvc add data/
git add data/*.dvc .gitignore 2>/dev/null || true
dvc push

check_success "Data preparation"

# Step 6: Run Training Pipeline
print_section "Step 6: Running Training Pipeline"

echo "Starting training pipeline..."
python -m src.pipelines.training_pipeline

check_success "Training pipeline"

# Step 7: View Training Results
print_section "Step 7: Training Results"

echo "Opening MLflow UI..."
echo "  URL: http://localhost:5000"
echo ""
echo "Model artifacts:"
ls -la models/

# Step 8: Run Monitoring Pipeline (without drift)
print_section "Step 8: Running Monitoring Pipeline (Normal)"

echo "Running monitoring pipeline..."
python -m src.pipelines.monitoring_pipeline --n-samples 500

check_success "Monitoring pipeline (normal)"

# Step 9: Simulate Drift and Retrain
print_section "Step 9: Simulating Drift and Testing Retrain Trigger"

echo "Running monitoring pipeline with simulated drift..."
python -m src.pipelines.monitoring_pipeline --simulate-drift --n-samples 500

check_success "Monitoring pipeline (with drift)"

# Step 10: Final Summary
print_section "Workflow Complete!"

echo ""
echo -e "${GREEN}Summary:${NC}"
echo "  • Infrastructure: Running"
echo "  • Data: Downloaded and versioned"
echo "  • Model: Trained and registered"
echo "  • Monitoring: Active"
echo "  • Drift detection: Tested"
echo ""
echo -e "${YELLOW}Access Points:${NC}"
echo "  • MLflow UI:      http://localhost:5000"
echo "  • MinIO Console:  http://localhost:9001"
echo "  • ZenML Dashboard: http://localhost:8080"
echo ""
echo -e "${YELLOW}Reports:${NC}"
ls -la reports/

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}OpenMLOps Challenge workflow completed successfully!${NC}"
echo -e "${GREEN}============================================================${NC}"
