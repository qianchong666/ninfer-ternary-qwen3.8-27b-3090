#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer --prompt "$(cat "$T/prompt_long.txt")" --max-new 4 --device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking > "$T/our_long.log" 2>&1
grep -E "prefill speed|decode speed|prompt tokens" "$T/our_long.log" | tr -d '\r'
echo OUR_LONG_DONE
