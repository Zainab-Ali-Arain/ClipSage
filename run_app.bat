@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Starting ClipSage at http://localhost:8501  (leave this window open, CTRL+C to stop)
streamlit run app.py
