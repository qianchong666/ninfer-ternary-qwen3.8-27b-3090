#!/bin/bash
# Rebuild ONLY the ninfer CLI target. The resident ninfer-serve.exe (started by the user, :18099) is
# locked while running, so linking it would fail -- ninja builds the shared objects once and only
# relinks the CLI, which is all the prefill benchmark needs.
set -u
cd /h/ninfer-ternary || exit 1
T="$LOCALAPPDATA/Temp"
B="$T/build_ninfer.bat"
BLD='@echo off\r\ncall "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat" > nul\r\nset VSLANG=1033\r\nset CUDA_PATH=H:\\cuda-13.3\r\nset PATH=C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\Ninja;H:\\cuda-13.3\\bin;%%PATH%%\r\nninja -C H:/ninfer-ternary/build/b1 -j 20 -k 0 ninfer\r\nset RC=%%ERRORLEVEL%%\r\necho NINJA_RC=%%RC%%\r\n'
printf "$BLD" > "$B"
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' /c/Windows/System32/cmd.exe /c "$B" > "$T/v11_ninfer_build.log" 2>&1
tr -d '\r' < "$T/v11_ninfer_build.log" | grep -aE "NINJA_RC=|error|Error|FAILED" | head -20
