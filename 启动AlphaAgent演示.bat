@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" start_demo.py
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" start_demo.py
) else (
  python start_demo.py
)
pause
