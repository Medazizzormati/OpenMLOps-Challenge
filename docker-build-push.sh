#!/bin/bash
# =============================================================================
# OpenMLOps Challenge - Docker Build & Push Script
# =============================================================================
# This script builds all Docker images and pushes them to Docker Hub
# 
# Usage:
#   1. Login to Docker Hub: docker login
#   2. Run this script: chmod +x docker-build-push.sh && ./docker-build-push.sh
# =============================================================================

DOCKER_USER="medaziz977"
VERSION="1.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}OpenMLOps Challenge - Docker Build & Push${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Docker Hub User: ${DOCKER_USER}"
echo "Version: ${VERSION}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed!${NC}"
    exit 1
fi

# Check if logged in to Docker Hub
echo -e "${YELLOW}Checking Docker Hub login...${NC}"
if ! docker info 2>&1 | grep -q "Username"; then
    echo -e "${YELLOW}Please login to Docker Hub first:${NC}"
    echo "  docker login"
    exit 1
fi
echo -e "${GREEN}✓ Logged in to Docker Hub${NC}"
echo ""

# =============================================================================
# Build Images
# =============================================================================
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Building Docker Images${NC}"
echo -e "${BLUE}============================================================${NC}"

# 1. MLflow Server
echo ""
echo -e "${YELLOW}[1/5] Building MLflow Server...${NC}"
docker build -t ${DOCKER_USER}/openmlops-mlflow:${VERSION} -t ${DOCKER_USER}/openmlops-mlflow:latest -f docker/mlflow/Dockerfile docker/mlflow/
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ MLflow image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build MLflow image${NC}"
fi

# 2. ZenML Server
echo ""
echo -e "${YELLOW}[2/5] Building ZenML Server...${NC}"
docker build -t ${DOCKER_USER}/openmlops-zenml:${VERSION} -t ${DOCKER_USER}/openmlops-zenml:latest -f docker/zenml/Dockerfile docker/zenml/
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ ZenML image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build ZenML image${NC}"
fi

# 3. Training Pipeline
echo ""
echo -e "${YELLOW}[3/5] Building Training Pipeline...${NC}"
docker build -t ${DOCKER_USER}/openmlops-training:${VERSION} -t ${DOCKER_USER}/openmlops-training:latest -f docker/training/Dockerfile .
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build Training image${NC}"
fi

# 4. Monitoring Pipeline
echo ""
echo -e "${YELLOW}[4/5] Building Monitoring Pipeline...${NC}"
docker build -t ${DOCKER_USER}/openmlops-monitoring:${VERSION} -t ${DOCKER_USER}/openmlops-monitoring:latest -f docker/monitoring/Dockerfile .
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Monitoring image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build Monitoring image${NC}"
fi

# 5. Jupyter Notebook
echo ""
echo -e "${YELLOW}[5/5] Building Jupyter Notebook...${NC}"
docker build -t ${DOCKER_USER}/openmlops-jupyter:${VERSION} -t ${DOCKER_USER}/openmlops-jupyter:latest -f docker/jupyter/Dockerfile .
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Jupyter image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build Jupyter image${NC}"
fi

# =============================================================================
# Push Images
# =============================================================================
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Pushing Images to Docker Hub${NC}"
echo -e "${BLUE}============================================================${NC}"

IMAGES=(
    "openmlops-mlflow"
    "openmlops-zenml"
    "openmlops-training"
    "openmlops-monitoring"
    "openmlops-jupyter"
)

for IMAGE in "${IMAGES[@]}"; do
    echo ""
    echo -e "${YELLOW}Pushing ${IMAGE}...${NC}"
    
    # Push with version tag
    docker push ${DOCKER_USER}/${IMAGE}:${VERSION}
    
    # Push with latest tag
    docker push ${DOCKER_USER}/${IMAGE}:latest
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${IMAGE} pushed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to push ${IMAGE}${NC}"
    fi
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Build & Push Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Images pushed to Docker Hub:"
echo ""
for IMAGE in "${IMAGES[@]}"; do
    echo "  • ${DOCKER_USER}/${IMAGE}:${VERSION}"
    echo "  • ${DOCKER_USER}/${IMAGE}:latest"
done
echo ""
echo -e "${YELLOW}View your images at:${NC}"
echo "  https://hub.docker.com/u/${DOCKER_USER}"
echo ""
echo -e "${YELLOW}To use these images, update docker-compose.yml:${NC}"
echo "  Replace 'build:' with 'image: ${DOCKER_USER}/openmlops-xxx:${VERSION}'"
