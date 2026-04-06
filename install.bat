@echo off
echo ============================================
echo  RS Nodes - Installation
echo ============================================
echo.

cd /d "%~dp0"

:: Initialize LTX-2 submodule (required for LoRA training)
echo [1/2] Initializing LTX-2 submodule...
git submodule update --init
echo.

:: Install Python dependencies (won't touch torch/ComfyUI packages)
echo [2/2] Installing Python dependencies...
pip install -r requirements.txt
echo.

echo ============================================
echo  Installation complete!
echo ============================================
pause
