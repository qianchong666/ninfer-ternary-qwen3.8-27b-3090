#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
/c/Windows/System32/taskkill.exe /F /IM ncu.exe /T >/dev/null 2>&1
/c/Windows/System32/taskkill.exe /F /IM ninfer.exe /T >/dev/null 2>&1
sleep 5
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' /c/Windows/System32/cmd.exe /c 'H:\ninfer-ternary\build.bat' > "$T/build_v3b.log" 2>&1
grep -a -o BUILD_OK "$T/build_v3b.log" | head -1
if grep -aq BUILD_OK "$T/build_v3b.log"; then
  ./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer --prompt "$(cat "$T/prompt_tiny2.txt")" --max-new 8 --device 0 --max-context 8192 --kv-capacity 8192 --kv-dtype int8 --no-thinking > "$T/v3b_tiny.log" 2>&1
  grep -hE "prefill speed|decode speed" "$T/v3b_tiny.log" | tr -d '\r'
  ./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer --prompt "$(cat "$T/prompt_long.txt")" --max-new 16 --device 0 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking > "$T/v3b_long.log" 2>&1
  grep -hE "prefill speed|decode speed" "$T/v3b_long.log" | tr -d '\r'
else
  grep -a "error" "$T/build_v3b.log" | tr -d '\r' | head -4
fi
echo V3B_DONE
