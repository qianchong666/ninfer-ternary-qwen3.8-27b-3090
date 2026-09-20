#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
/c/Windows/System32/taskkill.exe /F /IM ninfer.exe /T >/dev/null 2>&1
sleep 4
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' /c/Windows/System32/cmd.exe /c 'H:\ninfer-ternary\build.bat' > "$T/build_v2cr.log" 2>&1
grep -a -o BUILD_OK "$T/build_v2cr.log" | head -1
if grep -aq BUILD_OK "$T/build_v2cr.log"; then
  ./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer --prompt "$(cat "$T/prompt_long.txt")" --max-new 4 --device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking > "$T/our_long_v2cr.log" 2>&1
  grep -hE "prefill speed|decode speed" "$T/our_long_v2cr.log" | tr -d '\r'
else
  grep -a "error" "$T/build_v2cr.log" | tr -d '\r' | head -4
fi
echo V2CR_DONE
