@echo off
title PathSnap - Laptop 2 Frontend (Streamlit :8501)
cd /d "%~dp0"
echo ============================================================================
echo Starting PathSnap Frontend UI (Laptop 2) on http://localhost:8501 ...
echo ============================================================================
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python -m streamlit run app.py --server.port 8501
) else (
    "C:\Users\Kruthik\miniconda3\python.exe" -m streamlit run app.py --server.port 8501
)
pause
