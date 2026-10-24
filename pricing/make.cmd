@echo off
REM Shim so `make <target>` works in cmd/PowerShell, where GNU make is absent.
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0tasks.py" %*
) else (
  python "%~dp0tasks.py" %*
)
