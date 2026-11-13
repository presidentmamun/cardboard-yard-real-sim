@echo off
rem One-time setup: creates .venv next to this file and installs the packages.
cd /d "%~dp0"
if not exist cardboard_yard_real\env.py (echo Unzip the whole folder first, then run this from the unzipped folder. & pause & exit /b 1)
python -m venv .venv || (echo Python 3.10+ not found. Install it from python.org first, tick "Add python.exe to PATH". & pause & exit /b 1)
.venv\Scripts\python.exe -m pip install -r requirements.txt || (echo Package install failed, see the messages above. & pause & exit /b 1)
echo.
echo Setup done. Now run run_manual.bat or run_demo.bat
if not "%1"=="nopause" pause
