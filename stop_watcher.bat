@echo off
title Stop AI-ML Auto-Push Watcher
echo Stopping any running AI-ML watcher PowerShell processes...
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*watch_and_push.ps1*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host ('Stopped process ' + $_.ProcessId) }"
echo Done.
pause
