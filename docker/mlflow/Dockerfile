FROM python:3.10-slim

WORKDIR /mlflow

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    boto3==1.34.0 \
    psycopg2-binary==2.9.9

# Create artifacts directory
RUN mkdir -p /mlflow/artifacts

EXPOSE 5000

# Start MLflow server
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql://mlflow:mlflow123@postgres:5432/mlflow", \
     "--default-artifact-root", "s3://mlflow-artifacts/", \
     "--serve-artifacts"]
