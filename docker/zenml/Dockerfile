FROM python:3.10-slim

WORKDIR /zenml

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install ZenML with MLflow integration
RUN pip install --no-cache-dir \
    zenml==0.55.5 \
    zenml-mlflow==0.1.2 \
    psycopg2-binary==2.9.9

# Create ZenML directory
RUN mkdir -p /root/.zenml

EXPOSE 8080

# Start ZenML server
CMD ["zenml", "up", "--host", "0.0.0.0", "--port", "8080"]
