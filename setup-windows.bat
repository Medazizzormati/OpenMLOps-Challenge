@echo off
REM =============================================================================
REM OpenMLOps Challenge - Windows Setup Script
REM =============================================================================
REM Run this script in the project folder after extraction
REM =============================================================================

echo ============================================================
echo OpenMLOps Challenge - Windows Setup
echo ============================================================
echo.

REM Create directory structure
echo Creating directories...
mkdir src 2>nul
mkdir src\steps 2>nul
mkdir src\models 2>nul
mkdir src\pipelines 2>nul
mkdir src\monitoring 2>nul
mkdir src\utils 2>nul
mkdir src\data 2>nul
mkdir docker 2>nul
mkdir docker\mlflow 2>nul
mkdir docker\zenml 2>nul
mkdir docker\training 2>nul
mkdir docker\monitoring 2>nul
mkdir docker\jupyter 2>nul
mkdir data 2>nul
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir models 2>nul
mkdir reports 2>nul
mkdir inference_logs 2>nul
mkdir configs 2>nul
mkdir .dvc 2>nul

echo Done!
echo.
echo ============================================================
echo Now check if files exist in current folder:
echo ============================================================
dir /b
echo.
echo If you see files listed above, run:
echo   docker build -t medaziz977/openmlops-challenge:latest .
echo   docker push medaziz977/openmlops-challenge:latest
echo.
pause
