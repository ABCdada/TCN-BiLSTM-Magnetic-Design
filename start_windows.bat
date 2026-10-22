@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv 2>nul
  if not exist ".venv\Scripts\python.exe" python -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\verify_weight_manifest.py --manifest weights\manifest.json || exit /b 1
python scripts\verify_cold_start_manifest.py --manifest weights\cold_start_manifest.json || exit /b 1

if "%MAGNET_LISTEN%"=="" set MAGNET_LISTEN=127.0.0.1:8000
echo Starting at http://%MAGNET_LISTEN%/
python -m waitress --listen=%MAGNET_LISTEN% app:app
