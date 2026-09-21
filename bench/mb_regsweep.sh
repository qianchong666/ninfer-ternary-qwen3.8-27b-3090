#!/bin/bash
# Occupancy-vs-throughput curve for v8: force registers down with --maxrregcount,
# rebuild the isolated microbench, re-run at the REAL shape (rows=34816 k=5120 t=4636).
cd /h/ninfer-ternary || exit 1
T="$LOCALAPPDATA/Temp"
B="$T/mb_build_r.bat"
for R in 0 172 128 104; do
  if [ "$R" = "0" ]; then EXTRA=""; else EXTRA="--maxrregcount=$R"; fi
  printf '@echo off\r\ncall "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" > nul\r\ncd /d H:\\ninfer-ternary\\src\\ninfer\\src\\ops\\linear\\ternary\r\n"H:\\cuda-13.3\\bin\\nvcc.exe" -O3 -arch=sm_86 -std=c++17 --ptxas-options=-v %s -IH:\\ninfer-ternary\\src\\ninfer\\src\\ops\\linear\\ternary -o H:\\ninfer-ternary\\bench\\mb.exe mb.cu\r\necho NVCC_RC=%%ERRORLEVEL%%\r\n' "$EXTRA" > "$B"
  echo "########## maxrregcount=${R:-none} ##########"
  MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' cmd /c "$B" > "$T/mb_r$R.log" 2>&1
  tr -d '\r' < "$T/mb_r$R.log" | grep -aE "NVCC_RC|error" | head -3
  tr -d '\r' < "$T/mb_r$R.log" | grep -aE "Compiling entry|Used [0-9]+ registers|bytes spill" | sed 's/ for .sm_86.//' | sed 's/ptxas info    : //' | sed 's/_Z[0-9]*//' | grep -aE "v[0-9]|Used|spill" | tail -14

  echo "--- run (real shape) ---"
  ./bench/mb.exe 34816 40 4636 8 2>/dev/null | tr -d '\r' | grep -aE "v4|v7|v8"
done
echo SWEEP_DONE
