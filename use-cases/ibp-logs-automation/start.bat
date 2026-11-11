@echo off
REM start.bat — launch Flask backend + Angular dev server on Windows

REM ── Backend ──────────────────────────────────────────────────────────────────
echo Starting Flask backend on http://localhost:5001 ...
cd backend
call .venv\Scripts\activate.bat
start /B cmd /C "set FLASK_PORT=5001 && python app.py"
cd ..

REM ── Frontend ─────────────────────────────────────────────────────────────────
echo Starting Angular dev server on http://localhost:4200 ...
cd frontend
start /B cmd /C "set NG_CLI_ANALYTICS=false && npx ng serve --open"
cd ..

echo.
echo Both servers starting. Close this window to stop them.
pause
