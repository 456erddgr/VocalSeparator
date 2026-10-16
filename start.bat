@echo off
chcp 65001 >nul
title VocalSeparator
cd /d "%~dp0"

set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "PATH=%~dp0bin;%~dp0venv\Lib\site-packages\torch\lib;%PATH%"

echo ============================================
echo  VocalSeparator - Local Vocal Separation
echo  UI will open at http://127.0.0.1:7861
echo  Keep this window open while working.
echo ============================================
echo.

"%~dp0venv\Scripts\python.exe" "%~dp0app.py"

echo.
echo [Process exited] Press any key to close.
pause >nul
