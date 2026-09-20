#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
P=$(/c/Windows/System32/WBEM/WMIC.exe process where "name='ninfer-serve.exe' and CommandLine like '%18099%'" get ProcessId /format:value 2>/dev/null | tr -d '\r' | grep -oE "[0-9]+" | head -1)
if [ -n "$P" ]; then /c/Windows/System32/taskkill.exe /F /PID "$P" >/dev/null 2>&1; echo "killed serve $P"; fi
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' /c/Windows/System32/cmd.exe /c 'H:\ninfer-ternary\build.bat' > "$T/build_v3.log" 2>&1
if grep -aq BUILD_OK "$T/build_v3.log"; then
  echo "BUILD_OK"
  ./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer --prompt "The capital of France is" --max-new 16 --device 1 --max-context 8192 --kv-capacity 8192 --kv-dtype int8 --no-thinking > "$T/v3_ab.log" 2>&1
  ./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer --prompt "$(cat "$T/prompt_long.txt")" --max-new 64 --device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking --spec mtp --draft-tokens 4 --lm-head-draft > "$T/v3_long.log" 2>&1
  echo -n "正确性: "; grep -h "Paris" "$T/v3_ab.log" 2>/dev/null | tr -d '\r' | head -1
  echo -n "长prompt: "; grep -hE "prefill speed" "$T/v3_long.log" 2>/dev/null | tr -d '\r' | head -1
  echo -n "MTP解码: "; grep -hE "decode speed" "$T/v3_long.log" 2>/dev/null | tr -d '\r' | head -1
  echo -n "接受长度: "; grep -hE "mtp acceptance length" "$T/v3_long.log" 2>/dev/null | tr -d '\r' | head -1
else
  grep -a "error" "$T/build_v3.log" | tr -d '\r' | head -6
fi
echo V3_CYCLE_DONE
