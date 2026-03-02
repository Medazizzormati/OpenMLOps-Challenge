FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p /app/data /app/reports /app/inference_logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# Default command
CMD ["python", "-m", "src.pipelines.monitoring_pipeline"]
