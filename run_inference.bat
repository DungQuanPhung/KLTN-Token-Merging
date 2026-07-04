@echo off
REM Double-click de chay toan bo he thong inference (Ollama + Backend + Frontend).
REM Tuong duong: powershell -ExecutionPolicy Bypass -File run_inference.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_inference.ps1" %*
pause
