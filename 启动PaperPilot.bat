@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "env\Scripts\python.exe" set "PY=%~dp0env\Scripts\python.exe"
if not defined PY if exist "%USERPROFILE%\.conda\envs\paperpilot\python.exe" set "PY=%USERPROFILE%\.conda\envs\paperpilot\python.exe"
if not defined PY if exist "%USERPROFILE%\.conda\envs\paperpilot\Scripts\python.exe" set "PY=%USERPROFILE%\.conda\envs\paperpilot\Scripts\python.exe"
if not defined PY (
  for /f "delims=" %%I in ('where python 2^>nul') do if not defined PY set "PY=%%I"
)

if not defined PY (
  echo [X] Python not found. Install Python 3.12, then run:  pip install -e ".[dev]"
  echo.
  pause
  exit /b 1
)

set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

"%PY%" "%~dp0scripts\launch.py"
if errorlevel 1 pause

endlocal
