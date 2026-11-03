@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
cd backend
echo Starting ClipSage backend at http://localhost:8000
echo (Leave this window open. Press CTRL+C to stop.)
echo.
uvicorn main:app --reload
