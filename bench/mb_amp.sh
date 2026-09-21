#!/bin/bash
cd /h/ninfer-ternary || exit 1
T="$LOCALAPPDATA/Temp"
B="$T/mb_build_amp.bat"
BLD='@echo off\r\ncall "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat" > nul\r\ncd /d H:/ninfer-ternary/src/ninfer/src/ops/linear/ternary\r\n"H:/cuda-13.3/bin/nvcc.exe" -O3 -arch=sm_86 -std=c++17 --ptxas-options=-v'
run() {
  printf "$BLD %s -IH:/ninfer-ternary/src/ninfer/src/ops/linear/ternary -o H:/ninfer-ternary/bench/mb.exe mb.cu\r\necho NVCC_RC=%%ERRORLEVEL%%\r\n" "$2" > "$B"
  echo "########## $1 ##########"
  MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' cmd /c "$B" > "$T/amp_$1.log" 2>&1
  tr -d '\r' < "$T/amp_$1.log" | grep -aE "NVCC_RC" | head -2
  ./bench/mb.exe 34816 40 4636 8 2>/dev/null | tr -d '\r' | grep -aE "^v9 "
}
run base "-DkV9_TILES=6"
run nob "-DkV9_TILES=6 -DkV9_NO_B=1"
run noa "-DkV9_TILES=6 -DkV9_NO_A=1"
run nodec "-DkV9_TILES=6 -DkV9_NO_DECODE=1"
run nbnA "-DkV9_TILES=6 -DkV9_NO_B=1 -DkV9_NO_A=1"
echo AMP_DONE
