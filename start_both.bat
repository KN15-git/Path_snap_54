@echo off
title PathSnap - Distributed Map-Matching Launcher
cd /d "%~dp0"
echo ============================================================================
echo Launching PathSnap Backend (:8000) and Frontend (:8501) in separate windows...
echo ============================================================================
start "PathSnap Laptop 1 Backend" cmd /k ""%~dp0start_backend.bat""
timeout /t 3 /nobreak >nul
start "PathSnap Laptop 2 Frontend" cmd /k ""%~dp0start_frontend.bat""
echo Done! Both servers have been launched.

