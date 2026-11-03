@echo off
title SearXNG Product Image Finder
echo ========================================================
echo        SearXNG Product Image Finder (CSV/Excel)
echo ========================================================
echo.
echo Starting backend server on http://localhost:8000 ...
echo.

cd /d "%~dp0"

:: Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH!
    pause
    exit /b
)

:: Start browser after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8000"

:: Start FastAPI backend
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000

pause

