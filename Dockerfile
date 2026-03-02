# =============================================================================
# OpenMLOps Challenge - Complete Application Image
# =============================================================================

FROM python:3.10-slim

LABEL maintainer="salah.gontara@polytecsousse.tn"
LABEL description="OpenMLOps Challenge - CIFAR-10 CNN Classifier"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/reports /app/inference_logs /app/artifacts

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

# Default command
CMD ["python", "-m", "src.pipelines.training_pipeline"]
