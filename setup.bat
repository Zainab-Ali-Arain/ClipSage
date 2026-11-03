@echo off
echo ============================================
echo   ClipSage - one-time setup
echo ============================================
cd /d "%~dp0"

echo [1/3] Creating Python 3.11 virtual environment...
py -3.11 -m venv venv
if errorlevel 1 (
    echo ERROR: Python 3.11 not found. Install it from python.org and run setup.bat again.
    pause
    exit /b 1
)

echo [2/3] Installing packages (first time downloads ~2 GB, mostly PyTorch)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: package installation failed - see the messages above.
    pause
    exit /b 1
)

echo [3/3] Creating .env for your YouTube API key...
if not exist ".env" copy ".env.example" ".env" >nul

echo.
echo Setup complete.
echo Next: open .env in Notepad, paste your YouTube API key, then run run_pipeline.bat
pause
