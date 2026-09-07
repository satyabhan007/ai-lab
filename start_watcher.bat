@echo off
title AI-ML Auto-Push Watcher
echo Starting AI-ML auto-sync watcher in background...
powershell -WindowStyle Hidden -ExecutionPolicy Bypass -File "D:\test\account rotate\account_rotator\watch_and_push.ps1"
echo Watcher running silently in background.
echo Check logs at: D:\test\account rotate\AI-ML\auto_sync.log
pause
