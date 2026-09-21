#!/bin/bash
# build + run the v11 surgical probe (fragment mapping) on GPU1
set -u
cd /h/ninfer-ternary || exit 1
T="$LOCALAPPDATA/Temp"
B="$T/v11_probe.bat"
BLD='@echo off\r\ncall "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat" > nul\r\ncd /d H:/ninfer-ternary/bench\r\n"H:/cuda-13.3/bin/nvcc.exe" -O3 -arch=sm_86 -std=c++17 --ptxas-options=-v'
printf "$BLD %s -o H:/ninfer-ternary/bench/v11_probe.exe v11_probe.cu\r\necho NVCC_RC=%%ERRORLEVEL%%\r\n" "${1:-}" > "$B"
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' cmd /c "$B" > "$T/v11_probe.log" 2>&1
tr -d '\r' < "$T/v11_probe.log" | grep -aE "NVCC_RC=|error|warning" | head -20
CUDA_VISIBLE_DEVICES=1 ./bench/v11_probe.exe 2>/dev/null | tr -d '\r'
