@echo off
rem ============================================================================
rem  AlphaPulse — silent launcher (no console window, no rebuild)
rem  Double-click this file (or its desktop shortcut) to start the GUI.
rem  Source updates take effect immediately after `git pull` — no rebuild.
rem ============================================================================
setlocal
cd /d "%~dp0"

rem Prefer a project-local venv if one exists.
if exist ".venv\Scripts\pythonw.exe" (
    set "PYW=.venv\Scripts\pythonw.exe"
) else (
    set "PYW=pythonw"
)

rem `start ""` detaches so the cmd host can exit immediately.
start "" "%PYW%" -m crypto_trend
endlocal
