@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d H:\ninfer-ternary\src\ninfer\src\ops\linear\ternary
"H:\cuda-13.3\bin\nvcc.exe" -O3 -arch=sm_86 -std=c++17 --ptxas-options=-v -IH:\ninfer-ternary\src\ninfer\src\ops\linear\ternary -o H:\ninfer-ternary\bench\mb.exe mb.cu
echo NVCC_RC=%ERRORLEVEL%
