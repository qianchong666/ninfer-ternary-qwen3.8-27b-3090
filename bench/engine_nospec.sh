#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
P="$T/prompt_long.txt"
M=models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer
for i in 1 2 3; do
  ./build/b1/apps/ninfer.exe $M --prompt "$(cat $P)" --max-new 8 --device 1 \
    --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --no-thinking \
    --prefill-chunk 8192 > $T/e3_$i.log 2>&1
  tr -d '\r' < $T/e3_$i.log | grep -aE "prefill speed|text prefill" | sed "s/^/run$i /"
done
