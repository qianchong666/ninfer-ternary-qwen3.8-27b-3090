#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
B=./build/b1/apps/ninfer.exe
M=models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer
C="--device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking"
for K in 4 5 6; do
  $B $M --prompt "$(cat "$T/prompt.txt")" --max-new 512 $C \
    --spec mtp --draft-tokens $K --lm-head-draft > "$T/ks$K.log" 2>&1
  echo "== K=$K"
  grep -E "decode speed|mtp rounds|mtp acceptance length" "$T/ks$K.log" | tr -d '\r'
done
echo KSWEEP_DONE
