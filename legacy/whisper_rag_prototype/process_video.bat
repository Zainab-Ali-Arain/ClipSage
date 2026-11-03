@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat

if "%~1"=="" (
    echo Usage: drag and drop a video file onto this .bat file
    echo    or run: process_video.bat path\to\video.mp4
    pause
    exit /b 1
)

REM Stay in the project root so "data" always means the shared top-level data folder
if not exist "data" mkdir "data"

for %%F in ("%~1") do set "VIDEO_NAME=%%~nF" & set "VIDEO_EXT=%%~xF"

echo [Step 0/2] Copying video into data\ (so the browser can play it later)...
copy /Y "%~1" "data\%VIDEO_NAME%%VIDEO_EXT%" >nul

echo [Step 1/2] Transcribing video: %VIDEO_NAME%%VIDEO_EXT%
python backend\transcribe.py "data\%VIDEO_NAME%%VIDEO_EXT%"

echo.
echo [Step 2/2] Building searchable index...
python backend\build_index.py "data\%VIDEO_NAME%_transcript.json"

echo.
echo Done! You can now ask questions about this video.
pause
