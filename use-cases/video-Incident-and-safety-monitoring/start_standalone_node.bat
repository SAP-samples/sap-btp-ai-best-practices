@echo off
echo ============================================================
echo   Video Incident Monitoring - Standalone Test Server (Node)
echo ============================================================
echo.

cd video-incident-monitor

echo Starting Node.js HTTP server for testing...
echo.
echo ============================================================
echo   Standalone Page: http://localhost:8080/webapp/analyze-standalone.html
echo ============================================================
echo.
echo Press Ctrl+C to stop the server.
echo.

npx http-server . -p 8080

pause
