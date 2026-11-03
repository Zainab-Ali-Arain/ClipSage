@echo off
echo ============================================
echo   ClipSage - one-time setup
echo ============================================
cd /d "%~dp0"

echo [1/3] Creating Python virtual environment...
REM Prefer the "py" launcher with 3.11; otherwise use whatever "python" is on PATH.
set "PYEXE="
py -3.11 --version >nul 2>&1 && set "PYEXE=py -3.11"
if not defined PYEXE (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
    echo.
    echo ERROR: Python 3.10, 3.11 or 3.12 was not found.
    echo Install Python 3.11 from https://www.python.org/downloads/release/python-3119/
    echo During install tick "Add python.exe to PATH", then close and reopen VS Code
    echo and run setup.bat again.
    pause
    exit /b 1
)
echo Using: %PYEXE%
%PYEXE% -m venv venv
if errorlevel 1 (
    echo ERROR: could not create the virtual environment.
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
