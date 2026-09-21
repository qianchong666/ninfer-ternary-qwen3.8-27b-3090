#!/bin/bash
cd /h/ninfer-ternary || exit 1
T="$LOCALAPPDATA/Temp"
B="$T/peak_build.bat"
BLD='@echo off\r\ncall "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat" > nul\r\ncd /d H:/ninfer-ternary/src/ninfer/src/ops/linear/ternary\r\n"H:/cuda-13.3/bin/nvcc.exe" -O3 -arch=sm_86 -std=c++17 -o H:/ninfer-ternary/bench/mb_peak.exe mb_peak.cu\r\necho NVCC_RC=%%ERRORLEVEL%%\r\n'
printf "$BLD" > "$B"
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' cmd /c "$B" > "$T/peak_build.log" 2>&1
tr -d '\r' < "$T/peak_build.log" | grep -aE "NVCC_RC|error" | head -5
./bench/mb_peak.exe 2>/dev/null | tr -d '\r'
echo PEAK_DONE
