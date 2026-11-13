@echo off
rem Watch the scripted expert loader work in a live 3D window.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run: installing the packages, this takes a few minutes...
  call setup.bat nopause || exit /b 1
)
.venv\Scripts\python.exe demo.py --viewer --steps 3000 %*
pause
