@echo off
title SearXNG Image Finder [DEV MODE]
echo ========================================================
echo   Starting Backend (port 8000) & Vite React (port 5173)
echo ========================================================

cd /d "%~dp0"

start "FastAPI Backend" cmd /k "cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
start "React Vite Dev" cmd /k "cd frontend && npm run dev"

timeout /t 3 /nobreak >nul
start http://localhost:5173

echo Servers started!

