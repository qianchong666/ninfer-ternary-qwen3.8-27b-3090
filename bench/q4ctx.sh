#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
B=./build/b1/apps/ninfer.exe
M=/h/ninfer-3090/models/qwen3_8_27b_huihui_abliterated.ninfer
for KV in k8v4 rk8v4; do
  $B $M --prompt "The capital of France is" --max-new 8 --device 1 \
    --max-context 262144 --kv-capacity 262144 --kv-dtype $KV --no-thinking \
    --spec mtp --draft-tokens 3 --lm-head-draft > "$T/q4_$KV.log" 2>&1
  echo "== KV=$KV exit=$?"
  grep -E "decode speed|prefill speed|planned device|cudaError|error|out of memory" "$T/q4_$KV.log" | head -5
done
echo Q4CTX_DONE
