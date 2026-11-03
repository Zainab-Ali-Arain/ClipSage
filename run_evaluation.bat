@echo off
REM Usage:  run_evaluation.bat pool   -> create the relevance-labelling sheet
REM         run_evaluation.bat all    -> run every automatic evaluation + write the report
cd /d "%~dp0"
call venv\Scripts\activate.bat
if "%~1"=="" (
    python -m clipsage.evaluate all
) else (
    python -m clipsage.evaluate %*
)
pause
