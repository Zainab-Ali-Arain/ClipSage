@echo off
REM Builds the dataset and the search index (Stages 1-3). Safe to re-run: it resumes.
cd /d "%~dp0"
call venv\Scripts\activate.bat

echo [Stage 1] Collecting video metadata and transcripts (YouTube Data API)...
python -m clipsage.collect %*
if errorlevel 1 goto :error

echo [Stage 2] Cleaning transcripts and building 30-60 s segments...
python -m clipsage.preprocess
if errorlevel 1 goto :error

echo [Stage 3] Building TF-IDF and Sentence-BERT indexes...
python -m clipsage.index
if errorlevel 1 goto :error

echo [Report] Dataset statistics...
python -m clipsage.explore
python -m clipsage.topics --lda-topics 10

echo.
echo Done. Start the app with run_app.bat
pause
exit /b 0

:error
echo.
echo Something went wrong - read the error message above.
pause
exit /b 1
