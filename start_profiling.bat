@echo off
set "HIP_PLATFORM=amd"
set "ROCM_PATH=%ROCM_PATH%"
set "PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib\llvm\bin;%PATH%"
cd /d "G:\Strata"
".venv\Scripts\python.exe" "serve\server.py" "--engine" "strata" "--config" "strata-iq2_xs.json" "--port" "7860" "--open"
