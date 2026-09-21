#!/bin/bash
# Same long prompt and same flags as v3_engine.sh, but at the user's standard 256k context, to confirm
# the 1.24k tok/s prefill holds at the real deployment configuration (not just at 131072).
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer --prompt "$(cat "$T/prompt_long.txt")" --max-new 64 --device 1 --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --no-thinking --prefill-chunk 8192 --spec mtp --draft-tokens 4 --lm-head-draft > "$T/v8_256k.log" 2>&1
grep -hE "prefill speed|decode speed|reused prompt|out of memory|error" "$T/v8_256k.log" | tr -d '\r'
echo V8_256K_DONE
