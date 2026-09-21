#!/bin/bash
# v9 sweep: tiles-per-warp (occupancy vs A/B reuse) x ldmatrix x launch_bounds min-blocks.
# Each build reports the v9 kernel's real register count and runs the REAL shape.
cd /h/ninfer-ternary || exit 1
T="$LOCALAPPDATA/Temp"
B="$T/mb_build_v9.bat"

run() {
  printf '@echo off\r\ncall "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" > nul\r\ncd /d H:\\ninfer-ternary\\src\\ninfer\\src\\ops\\linear\\ternary\r\n"H:\\cuda-13.3\\bin\\nvcc.exe" -O3 -arch=sm_86 -std=c++17 --ptxas-options=-v %s -IH:\\ninfer-ternary\\src\\ninfer\\src\\ops\\linear\\ternary -o H:\\ninfer-ternary\\bench\\mb.exe mb.cu\r\necho NVCC_RC=%%ERRORLEVEL%%\r\n' "$2" > "$B"
  echo "########## $1   [$2] ##########"
  MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' cmd /c "$B" > "$T/mb_v9_$1.log" 2>&1
  tr -d '\r' < "$T/mb_v9_$1.log" | grep -aE "NVCC_RC=$|error:" | head -3
  tr -d '\r' < "$T/mb_v9_$1.log" | grep -aA3 "entry function 'ternary_pq2_mma_v9" | grep -aE "registers|spill" | head -2
  ./bench/mb.exe 34816 40 4636 8 2>/dev/null | tr -d '\r' | grep -aE "^v8 |^v9 |^v9 vs"
}

run t8_ld    "-DkV9_TILES=8 -DkV9_LDMATRIX=1"
run t8b4     "-DkV9_TILES=8 -DkV9_LDMATRIX=1 -DkV9_MINBLOCKS=4"
run t6_ld    "-DkV9_TILES=6 -DkV9_LDMATRIX=1"
run t4_ld    "-DkV9_TILES=4 -DkV9_LDMATRIX=1"
run t4_nold  "-DkV9_TILES=4 -DkV9_LDMATRIX=0"
run t4b4     "-DkV9_TILES=4 -DkV9_LDMATRIX=1 -DkV9_MINBLOCKS=4"
run t2_ld    "-DkV9_TILES=2 -DkV9_LDMATRIX=1"
echo V9_SWEEP_DONE
