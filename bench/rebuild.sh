#!/bin/bash
P=$(/c/Windows/System32/WBEM/WMIC.exe process where "name='ninfer-serve.exe' and CommandLine like '%18099%'" get ProcessId /format:value 2>/dev/null | tr -d '\r' | grep -oE "[0-9]+" | head -1)
if [ -n "$P" ]; then /c/Windows/System32/taskkill.exe /F /PID "$P" >/dev/null 2>&1; echo "killed serve pid $P"; fi
cd /h/ninfer-ternary || exit 1
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' /c/Windows/System32/cmd.exe /c 'H:\ninfer-ternary\build.bat' 2>&1 | tr -d '\r' | tail -6
