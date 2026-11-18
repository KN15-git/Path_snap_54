@echo off
title PathSnap - Laptop 1 Backend (FastAPI :8000)
cd /d "%~dp0"
echo ============================================================================
echo Starting PathSnap Backend (Laptop 1) on http://0.0.0.0:8000 ...
echo ============================================================================
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python run_server.py
) else (
    "C:\Users\Kruthik\miniconda3\python.exe" run_server.py
)
pause
