#!/bin/bash
# v11 step: rebuild the isolated microbench with the given defines and run the REAL engine shape
# (rows=34816 k=5120) at the engine's real token block t=928 on GPU1 (GPU0 drives the display).
# usage: v11_mb.sh <tag> "<extra nvcc defines>" [mb_args...]
set -u
cd /h/ninfer-ternary || exit 1
T="$LOCALAPPDATA/Temp"
B="$T/mb_v11.bat"
TAG="$1"; shift
DEFS="${1:-}"; shift || true
ARGS="${*:-34816 40 928 8}"
BLD='@echo off\r\ncall "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat" > nul\r\ncd /d H:/ninfer-ternary/src/ninfer/src/ops/linear/ternary\r\n"H:/cuda-13.3/bin/nvcc.exe" -O3 -arch=sm_86 -std=c++17 --ptxas-options=-v'
printf "$BLD %s -IH:/ninfer-ternary/src/ninfer/src/ops/linear/ternary -o H:/ninfer-ternary/bench/mb.exe mb.cu\r\necho NVCC_RC=%%ERRORLEVEL%%\r\n" "$DEFS" > "$B"
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' cmd /c "$B" > "$T/mb_v11_$TAG.log" 2>&1
echo "===== $TAG  [$DEFS] ====="
tr -d '\r' < "$T/mb_v11_$TAG.log" | grep -aE "NVCC_RC=" | head -2
tr -d '\r' < "$T/mb_v11_$TAG.log" | grep -aE "error|Error" | head -20
tr -d '\r' < "$T/mb_v11_$TAG.log" | grep -aA4 "entry function 'ternary_pq2_mma_v1[12]" | grep -aE "registers|spill" | head -4
for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=1 ./bench/mb.exe $ARGS 2>/dev/null | tr -d '\r' | grep -aE "^v9 |^v11 |^v9 vs|^v11 vs"
done
