@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -m venv .venv
    if errorlevel 1 goto :error
)
.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
if errorlevel 1 goto :error
.venv\Scripts\python.exe app.py --solver cuda-exact --galaxy-particles auto %*
if errorlevel 1 goto :error
exit /b 0
:error
echo CUDA launch failed. Check the message above. CPU launch: run_windows.bat
pause
exit /b 1
