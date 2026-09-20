#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
NCU=$(cat "$T/ncu_path.txt")
export PATH="/h/cuda-13.3/bin:$PATH"
"$NCU" --metrics dram__bytes_read.sum,gpu__time_duration.sum --launch-count 2 -k regex:ternary_pq2_mma_v2 ./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer --prompt "$(cat "$T/prompt.txt")" --max-new 2 --device 0 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking > "$T/ncu_prefill.log" 2>&1
echo "RC=$?"
grep -E "dram__bytes_read|gpu__time_duration|void " "$T/ncu_prefill.log" | head -8
