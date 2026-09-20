#!/bin/bash
cd /h/ninfer-3090 || exit 1
T=$LOCALAPPDATA/Temp
E=./ninfer.exe
M=models/qwen3_8_27b_huihui_abliterated.ninfer
P="$(cat "$T/prompt.txt")"
SP="Explain in one paragraph how a GPU hides memory latency."
for C in 1024 4096; do
  for tag in a b; do
    $E "$M" --prompt "$P" --max-new 2 --device 1 --max-context 219136 --kv-capacity 219136 --kv-dtype rk8v4 --prefill-chunk $C --no-thinking > "$T/q_c${C}${tag}.log" 2>&1
    echo -n "长prompt chunk=$C 第${tag}次: "
    grep -h "prefill speed" "$T/q_c${C}${tag}.log" | tr -d '\r' | grep -oE "[0-9.]+ tok/s" | head -1
  done
done
$E "$M" --prompt "$SP" --max-new 2 --device 1 --max-context 219136 --kv-capacity 219136 --kv-dtype rk8v4 --prefill-chunk 1024 --no-thinking > "$T/q_short.log" 2>&1
echo -n "短prompt: "; grep -h "prefill speed" "$T/q_short.log" | tr -d '\r' | grep -oE "[0-9.]+ tok/s" | head -1
echo THEIR_VERIFY_DONE
