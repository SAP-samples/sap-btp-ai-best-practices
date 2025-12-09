@echo off
echo ============================================================
echo   Video Incident Monitoring - Frontend Launcher
echo ============================================================
echo.

cd video-incident-monitor

echo Starting SAP Fiori application...
echo.
echo ============================================================
echo   Frontend URLs:
echo ============================================================
echo   Analyze Page: http://localhost:8080/analyze-standalone.html
echo   List Report:  http://localhost:8080/test/flpSandbox.html
echo ============================================================
echo.

npm run start

pause
