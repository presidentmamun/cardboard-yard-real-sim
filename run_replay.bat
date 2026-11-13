@echo off
rem Watch a saved drive again: drag its CSV onto this file, or: run_replay.bat drives\<file>.csv
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run: installing the packages, this takes a few minutes...
  call setup.bat nopause || exit /b 1
)
if "%~1"=="" (echo Drag a drive CSV onto run_replay.bat, or run: run_replay.bat drives\^<file^>.csv & pause & exit /b 1)
.venv\Scripts\python.exe replay.py %*
pause
