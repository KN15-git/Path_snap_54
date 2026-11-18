@echo off
title Allow PathSnap Port 8000 in Windows Firewall
echo Requesting Administrator privileges to allow Port 8000 through Windows Firewall...
powershell -Command "Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -Command New-NetFirewallRule -DisplayName \"\"PathSnap Backend Port 8000\"\" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow' -Verb RunAs"
echo.
echo If a UAC prompt appeared, click "Yes" to allow.
pause
