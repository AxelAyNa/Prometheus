@echo off
cd /d "%~dp0"
python -m pip install -q flask pandas openpyxl pypdf google-genai python-dotenv
python server.py
pause