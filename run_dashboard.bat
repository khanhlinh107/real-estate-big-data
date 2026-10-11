@echo off
cd /d "%~dp0"
if not exist .\.venv\Scripts\python.exe (
  echo Creating Python virtual environment...
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Could not create the environment. Install Python 3.12 and try again.
    pause
    exit /b 1
  )
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  if errorlevel 1 (
    echo Could not update pip.
    pause
    exit /b 1
  )
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
  )
)
start "" http://127.0.0.1:5000
.\.venv\Scripts\python.exe .\web_app\app.py
