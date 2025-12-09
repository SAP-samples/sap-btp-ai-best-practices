@echo off
echo ============================================================
echo   Video Incident Monitoring - Standalone Test Server
echo ============================================================
echo.

cd video-incident-monitor\webapp

echo Starting simple HTTP server for testing...
echo.
echo ============================================================
echo   Standalone Page: http://localhost:8080/analyze-standalone.html
echo   Video files: http://localhost:8080/Video/
echo   Audio files: http://localhost:8080/Audio/
echo ============================================================
echo.
echo Press Ctrl+C to stop the server.
echo.

python -m http.server 8080

pause
