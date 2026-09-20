#!/bin/bash
cd /h/ninfer-ternary || exit 1
L=$LOCALAPPDATA/Temp/build_v2c.log
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' /c/Windows/System32/cmd.exe /c 'H:\ninfer-ternary\build.bat' > "$L" 2>&1
echo "=== 错误行 ==="
grep -E "error|错误" "$L" | tr -d '\r' | head -14
echo "=== 尾 ==="
tail -3 "$L" | tr -d '\r'
