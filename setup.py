"""
Setup script for OpenMLOps Challenge package.
"""

from setuptools import setup, find_packages

setup(
    name="openmlops-challenge",
    version="1.0.0",
    description="CIFAR-10 CNN Classifier with MLOps workflow",
    author="OpenMLOps Challenge",
    author_email="salah.gontara@polytecsousse.tn",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "tensorflow==2.15.0",
        "numpy==1.26.2",
        "pillow==10.1.0",
        "zenml==0.55.5",
        "zenml-mlflow==0.1.2",
        "mlflow==2.9.2",
        "dvc[s3]==3.36.1",
        "evidently==0.4.8",
        "pandas==2.1.3",
        "scikit-learn==1.3.2",
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
        "python-dotenv==1.0.0",
        "pyyaml==6.0.1",
        "boto3==1.34.0",
        "minio==7.2.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
