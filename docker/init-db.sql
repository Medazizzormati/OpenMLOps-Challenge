-- Initialize databases for MLflow and ZenML

-- Create MLflow database
CREATE USER mlflow WITH PASSWORD 'mlflow123';
CREATE DATABASE mlflow OWNER mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

-- Create ZenML database
CREATE USER zenml WITH PASSWORD 'zenml123';
CREATE DATABASE zenml OWNER zenml;
GRANT ALL PRIVILEGES ON DATABASE zenml TO zenml;

-- Grant schema permissions
\c mlflow
GRANT ALL ON SCHEMA public TO mlflow;

\c zenml
GRANT ALL ON SCHEMA public TO zenml;
