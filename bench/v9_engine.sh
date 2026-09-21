#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
# kill only OUR engine (H:/ninfer-ternary), never the H:/infer-3090 service
for p in $(/c/Windows/System32/tasklist.exe 2>/dev/null | grep -i ninfer | awk '{print $2}'); do
  ep=$(MSYS_NO_PATHCONV=1 /c/Windows/System32/wbem/WMIC.exe process where "ProcessId=$p" get ExecutablePath 2>/dev/null | grep -i ninfer)
  case "$ep" in *infer-ternary*) /c/Windows/System32/taskkill.exe /F /PID $p >/dev/null 2>&1;; esac
done
sleep 3
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' /c/Windows/System32/cmd.exe /c 'H:/ninfer-ternary/build.bat' > "$T/build_v9.log" 2>&1
grep -a -o BUILD_OK "$T/build_v9.log" | head -1
./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer \
  --prompt "$(cat "$T/prompt_long.txt")" --max-new 64 --device 1 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --no-thinking \
  --prefill-chunk 8192 --spec mtp --draft-tokens 4 --lm-head-draft > "$T/v9_engine.log" 2>&1
grep -hE "prefill speed|decode speed|reused prompt|accept" "$T/v9_engine.log" | tr -d '\r'
echo V9_ENGINE_DONE
