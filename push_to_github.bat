@echo off
title Push PathSnap to GitHub
cd /d "%~dp0"
echo ============================================================================
echo Pushing PathSnap to https://github.com/KN15-git/Path_snap_54 ...
echo ============================================================================
git push origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Repository successfully pushed to GitHub!
) else (
    echo [ERROR] Push failed. Please check your GitHub credentials or permissions.
)
pause
