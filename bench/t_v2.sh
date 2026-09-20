#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
P=$(cat "$T/prompt.txt")
B=./build/b1/apps/ninfer.exe
M=models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer
C="--device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking"
$B $M --prompt "The capital of France is" --max-new 16 --device 1 --max-context 8192 --kv-capacity 8192 --kv-dtype int8 --no-thinking > "$T/abi2.log" 2>&1
$B $M --prompt "$P" --max-new 512 $C > "$T/v2_dec.log" 2>&1
$B $M --prompt "$P" --max-new 512 $C --spec mtp --draft-tokens 4 --lm-head-draft > "$T/v2_k4.log" 2>&1
$B $M --messages "$T/long.json" --max-new 128 $C > "$T/v2_pre.log" 2>&1
echo "=== 正确性 ==="; grep -A2 -i "capital" "$T/abi2.log" | head -4
for f in v2_dec v2_k4 v2_pre; do echo "######## $f"; grep -E "prompt tokens|decode speed|prefill speed|mtp rounds|mtp acceptance length" "$T/$f.log" | sed "s/summary *//"; done
echo V2_DONE
