#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
B=./build/b1/apps/ninfer.exe
M=models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer
for KV in k8v4 int8 bf16; do
  $B $M --prompt "The capital of France is" --max-new 16 --device 1 \
    --max-context 262144 --kv-capacity 262144 --kv-dtype $KV --no-thinking \
    --spec mtp --draft-tokens 4 --lm-head-draft > "$T/t26_$KV.log" 2>&1
  echo "== KV=$KV exit=$?"
  grep -E "planned device|cudaError|error|out of memory|invalid|decode speed" "$T/t26_$KV.log" | head -4
done
echo T26_DONE
