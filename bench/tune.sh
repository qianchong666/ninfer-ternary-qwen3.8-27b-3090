#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
B=./build/b1/apps/ninfer.exe
M=models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer
$B $M --prompt "The capital of France is" --max-new 16 --device 1 \
  --max-context 8192 --kv-capacity 8192 --kv-dtype int8 --no-thinking > "$T/tune_ab.log" 2>&1
$B $M --prompt "$(cat $T/prompt.txt)" --max-new 512 --device 1 \
  --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking \
  --spec mtp --draft-tokens 4 --lm-head-draft > "$T/tune_k4.log" 2>&1
sed -n '/capital/p' "$T/tune_ab.log" | tail -1
grep -E "decode speed|mtp rounds|mtp acceptance length" "$T/tune_k4.log"
