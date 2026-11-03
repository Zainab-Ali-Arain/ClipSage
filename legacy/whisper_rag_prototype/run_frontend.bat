@echo off
cd /d "%~dp0frontend"

if not exist "node_modules" (
    echo Installing frontend packages for the first time...
    npm install
)

echo Starting ClipSage frontend at http://localhost:5173
npm run dev
