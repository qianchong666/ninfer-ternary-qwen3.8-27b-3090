#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
B=./build/b1/apps/ninfer.exe
M=models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer
for C in 512 1024 2048; do
  $B $M --prompt "$(cat "$T/prompt.txt")" --max-new 4 \
    --device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 \
    --no-thinking --prefill-chunk $C > "$T/ch$C.log" 2>&1
  echo -n "chunk=$C  "
  grep "prefill speed" "$T/ch$C.log" | tr -d '\r' | grep -oE "[0-9.]+ tok/s"
done
echo CHUNK_DONE
