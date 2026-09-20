@echo off
rem build the ternary kernel micro-benchmark with the same toolchain as the port
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 90
set VSLANG=1033
set CUDA_PATH=H:\cuda-13.3
set PATH=H:\cuda-13.3\bin;%PATH%
nvcc -O3 -std=c++17 -arch=sm_86 -I H:/ninfer-ternary/src/ninfer/src -o H:/ninfer-ternary/bench/kbench.exe H:/ninfer-ternary/bench/kbench.cu > H:/ninfer-ternary/bench/kbench_build.log 2>&1
set RC=%ERRORLEVEL%
echo NVCC_RC=%RC%
if not "%RC%"=="0" exit /b 1
echo KBENCH_BUILD_OK
exit /b 0
