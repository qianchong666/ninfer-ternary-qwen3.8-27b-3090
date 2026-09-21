#!/bin/bash
# Dump and histogram the SASS of the v9 kernel, so the ~7.5 instructions per mma can be read off
# directly instead of guessed. Builds to a scratch exe (does not touch the sweep's mb.exe).
cd /h/ninfer-ternary || exit 1
T="$LOCALAPPDATA/Temp"
FLAGS="${1:--DkV9_TILES=8 -DkV9_LDMATRIX=1}"
printf '@echo off\r\ncall "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" > nul\r\ncd /d H:\\ninfer-ternary\\src\\ninfer\\src\\ops\\linear\\ternary\r\n"H:\\cuda-13.3\\bin\\nvcc.exe" -O3 -arch=sm_86 -std=c++17 %s -IH:\\ninfer-ternary\\src\\ninfer\\src\\ops\\linear\\ternary -o H:\\ninfer-ternary\\bench\\mb_sass.exe mb.cu\r\necho NVCC_RC=%%ERRORLEVEL%%\r\n' "$FLAGS" > "$T/mb_sass.bat"
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' cmd /c "$T/mb_sass.bat" > "$T/mb_sass_build.log" 2>&1
echo "SCASS build: $(tr -d '\r' < "$T/mb_sass_build.log" | grep -acE 'NVCC_RC=0')  flags=[$FLAGS]"
"$LOCALAPPDATA/../nvidia" 2>/dev/null || true
H:/cuda-13.3/bin/cuobjdump.exe -sass -arch sm_86 "H:\\ninfer-ternary\\bench\\mb_sass.exe" > "$T/sass_all.txt" 2>"$T/sass_err.txt"
echo "sass_all bytes: $(wc -c < "$T/sass_all.txt")"
# histogram opcodes of the v8 and v9 kernels only
awk '
/Function : /            { inv9 = ($0 ~ /mma_v9_kernel/); inv8 = ($0 ~ /mma_v8_kernel/); next }
(inv9 || inv8) && /\/\*[0-9a-f]{4,}\*\// {
  line = $0
  sub(/^.*\*\/[ \t]*/, "", line)          # drop the address part
  sub(/[ \t;].*$/, "", line)              # keep the opcode
  if (line ~ /^[A-Z][A-Z0-9.@_]*$/) { if (inv9) n9[line]++; else n8[line]++; tot9 += inv9; tot8 += inv8 }
}
END {
  printf "---- v8 kernel opcode histogram (top 30) ----\n"
  for (k in n8) printf "%8d  %s\n", n8[k], k | "sort -rn | head -30"
  close("sort -rn | head -30")
  printf "---- v9 kernel opcode histogram (top 30) ----\n"
  for (k in n9) printf "%8d  %s\n", n9[k], k | "sort -rn | head -30"
  close("sort -rn | head -30")
}
' "$T/sass_all.txt"
echo SASS_DUMP_DONE
