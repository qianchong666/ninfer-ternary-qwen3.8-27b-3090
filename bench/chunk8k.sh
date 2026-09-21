#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
B=./build/b1/apps/ninfer.exe
M=models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer
P="$(cat "$T/prompt_long.txt")"
NINFER_MMA_DEBUG=1 $B $M --prompt "$P" --max-new 4 --device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking --prefill-chunk 8192 > "$T/chunk_8k.log" 2>&1
echo "== T (chunk=8192) =="
grep -aoE "\[mma\] rows=[0-9]+ k=[0-9]+ groups=[0-9]+ t=[0-9]+" "$T/chunk_8k.log" | sort -u | head -5
echo "== prefill =="
grep -hE "prefill speed" "$T/chunk_8k.log" | tr -d '\r'
echo CHUNK_DONE
