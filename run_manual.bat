@echo off
rem Drive the loader yourself with the keyboard (see README.md for keys).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run: installing the packages, this takes a few minutes...
  call setup.bat nopause || exit /b 1
)
.venv\Scripts\python.exe manual.py %*
pause
