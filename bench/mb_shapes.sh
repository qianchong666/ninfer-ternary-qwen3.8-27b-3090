#!/bin/bash
cd /h/ninfer-ternary || exit 1
T="$LOCALAPPDATA/Temp"
B="$T/mb_build_sh.bat"
BLD='@echo off\r\ncall "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat" > nul\r\ncd /d H:/ninfer-ternary/src/ninfer/src/ops/linear/ternary\r\n"H:/cuda-13.3/bin/nvcc.exe" -O3 -arch=sm_86 -std=c++17 -DkV9_TILES=6 -IH:/ninfer-ternary/src/ninfer/src/ops/linear/ternary -o H:/ninfer-ternary/bench/mb.exe mb.cu\r\necho NVCC_RC=%%ERRORLEVEL%%\r\n'
printf "$BLD" > "$B"
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' cmd /c "$B" > "$T/sh_build.log" 2>&1
tr -d '\r' < "$T/sh_build.log" | grep -a NVCC_RC
run() { echo "--- $1 rows=$2 groups=$3 tokens=$4 ---"; ./bench/mb.exe $2 $3 $4 8 2>/dev/null | tr -d '\r' | grep -aE "^v8 |^v9 "; }
run q     4096  40  4636
run kv    6144  40  4636
run o     5120  48  4636
run gateup 34816 40 4636
run down  5120  136 4636
echo SHAPES_DONE
