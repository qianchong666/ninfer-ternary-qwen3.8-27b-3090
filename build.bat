@echo off
rem sm_86 (RTX 3090) build for the ternary NInfer port - ninja directly so the exit code is real
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 90
set VSLANG=1033
set CUDA_PATH=H:\cuda-13.3
set PATH=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;H:\cuda-13.3\bin;%PATH%
ninja -C H:/ninfer-ternary/build/b1 -j 20 -k 0
set RC=%ERRORLEVEL%
echo NINJA_RC=%RC%
if not "%RC%"=="0" exit /b 1
echo BUILD_OK
exit /b 0
