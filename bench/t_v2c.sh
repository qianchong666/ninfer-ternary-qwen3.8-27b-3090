#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
B=./build/b1/apps/ninfer.exe
M=models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer
$B $M --prompt "The capital of France is" --max-new 16 --device 1 --max-context 8192 --kv-capacity 8192 --kv-dtype int8 --no-thinking > "$T/v2c_ab.log" 2>&1
$B $M --prompt "$(cat "$T/prompt.txt")" --max-new 8 --device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking > "$T/v2c_pre.log" 2>&1
$B $M --prompt "$(cat "$T/prompt.txt")" --max-new 512 --device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking --spec mtp --draft-tokens 4 --lm-head-draft > "$T/v2c_k4.log" 2>&1
echo "== correctness"; grep -h "Paris" "$T/v2c_ab.log" | tr -d '\r' | head -1
echo "== prefill"; grep "prefill speed" "$T/v2c_pre.log" | tr -d '\r'
echo "== decode"; grep -E "decode speed|mtp acceptance" "$T/v2c_k4.log" | tr -d '\r'
echo V2C_TEST_DONE
