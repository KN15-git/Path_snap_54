@echo off
title PathSnap - Laptop 1 Backend (FastAPI :8000)
cd /d "%~dp0"
echo ============================================================================
echo Starting PathSnap Backend (Laptop 1) on http://0.0.0.0:8000 ...
echo ============================================================================
if exist "C:\Users\Kruthik\miniconda3\python.exe" (
    "C:\Users\Kruthik\miniconda3\python.exe" run_server.py
) else (
    python run_server.py
)
pause
