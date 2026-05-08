@echo off
rem ============================================================================
rem  AlphaPulse — DEBUG launcher (keeps console open so errors are visible)
rem  Use this if the silent launcher exits without showing the window.
rem ============================================================================
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

"%PY%" -m crypto_trend
echo.
echo --- exit code: %errorlevel% ---
pause
endlocal
