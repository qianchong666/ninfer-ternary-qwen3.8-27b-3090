@echo off
rem TRIAL: CUDA 13.1 + -allow-unsupported-compiler against VS 18 MSVC 19.51
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 90
set VSLANG=1033
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set PATH=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%
"C:\Program Files\CMake\bin\cmake.exe" -S H:/ninfer-ternary/src/ninfer -B H:/ninfer-ternary/build/try131 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe" -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=OFF -DNINFER_BUILD_MEDIA_ACQUIRE=OFF
if errorlevel 1 exit /b 1
echo CFG_OK
exit /b 0
