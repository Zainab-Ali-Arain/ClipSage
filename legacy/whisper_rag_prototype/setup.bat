@echo off
echo ============================================
echo   ClipSage - One-time setup
echo ============================================
echo.

REM Go to the folder this script is in (so it works no matter where you double-click it from)
cd /d "%~dp0"

echo [1/3] Creating Python 3.11 virtual environment...
py -3.11 -m venv venv
if errorlevel 1 (
    echo.
    echo ERROR: Could not create venv with Python 3.11.
    echo Make sure Python 3.11 is installed, then try again.
    pause
    exit /b 1
)

echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/3] Installing required packages (this can take a few minutes)...
cd backend
pip install -r requirements.txt

echo.
echo ============================================
echo   Setup complete!
echo   Next: run run_backend.bat
echo ============================================
pause
