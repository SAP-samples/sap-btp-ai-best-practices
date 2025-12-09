@echo off
echo ============================================================
echo   Video Incident Monitoring - Backend Service Launcher
echo ============================================================
echo.

REM Check if .env file exists
if not exist .env (
    echo WARNING: .env file not found!
    echo Please create .env file from .env.template and fill in your credentials.
    echo.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv (
    echo [1/4] Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        echo Please ensure Python 3.9+ is installed and in PATH.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
    echo.
) else (
    echo [1/4] Virtual environment already exists.
    echo.
)

REM Activate virtual environment
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo.

REM Install/upgrade dependencies
echo [3/4] Installing/updating dependencies...
echo This may take a few minutes...
pip install --upgrade pip
pip install -r backend_requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)
echo Dependencies installed successfully.
echo.

REM Start the server
echo [4/4] Starting OData Service...
echo.
echo ============================================================
echo   Backend Service Ready!
echo ============================================================
echo   Service URL:  http://localhost:5000
echo   Metadata:     http://localhost:5000/odata/v4/VideoIncidentService/$metadata
echo   Health Check: http://localhost:5000/health
echo ============================================================
echo.
echo Press Ctrl+C to stop the service.
echo.

python backend_odata_service.py

if errorlevel 1 (
    echo.
    echo ERROR: Backend service crashed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

pause
