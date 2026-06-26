@echo off
setlocal EnableDelayedExpansion

set "REPO_URL=https://github.tools.sap/sap-btp-ai-services-coe/Agent-skills-catalog.git"
set "CLONE_DIR=%TEMP%\agent-skills-catalog-%RANDOM%"
set "CATALOG_DIR=%CLONE_DIR%\skills"
set "SKILLS_DIR=%USERPROFILE%\.claude\skills"

echo Cloning skills catalog...
git clone --depth 1 "%REPO_URL%" "%CLONE_DIR%"
if errorlevel 1 (
    echo Error: git clone failed
    exit /b 1
)

if not exist "%SKILLS_DIR%" (
    mkdir "%SKILLS_DIR%"
)

set added=0
set skipped=0

for /d %%S in ("%CATALOG_DIR%\*") do (
    set "skill_name=%%~nxS"
    set "dest=%SKILLS_DIR%\!skill_name!"

    if exist "!dest!" (
        echo skip  !skill_name! (already exists^)
        set /a skipped+=1
    ) else (
        xcopy /e /i /q "%%S" "!dest!" >nul
        echo added !skill_name!
        set /a added+=1
    )
)

echo.
echo Done. Added: %added%  Skipped: %skipped%

echo Cleaning up cloned repo...
rd /s /q "%CLONE_DIR%"
endlocal
