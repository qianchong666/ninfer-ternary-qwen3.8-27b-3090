@echo off
rem sm_86 (RTX 3090) configure for the ternary NInfer port - CUDA 13.3 + CMake 4.4.3
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 90
set VSLANG=1033
set CUDA_PATH=H:\cuda-13.3
set PATH=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;H:\cuda-13.3\bin;%PATH%
"H:\cmake-4.4.3-windows-x86_64\bin\cmake.exe" -S H:/ninfer-ternary/src/ninfer -B H:/ninfer-ternary/build/b1 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER="H:/cuda-13.3/bin/nvcc.exe" -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=OFF -DNINFER_BUILD_MEDIA_ACQUIRE=OFF
if errorlevel 1 exit /b 1
echo CFG_OK
exit /b 0
