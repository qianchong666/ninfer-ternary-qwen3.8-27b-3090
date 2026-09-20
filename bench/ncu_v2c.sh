#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
NCU=$(cat "$T/ncu_path.txt")
export PATH="/h/cuda-13.3/bin:$PATH"
"$NCU" --kernel-name regex:v2c --launch-count 2 --section SpeedOfLight --section Occupancy --metrics dram__bytes_read.sum,gpu__time_duration.sum ./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer --prompt "$(cat "$T/prompt.txt")" --max-new 2 --device 0 --max-context 32768 --kv-capacity 32768 --kv-dtype int8 --no-thinking > "$T/ncu_v2c.log" 2>&1
echo RC=$?
grep -E "dram__bytes_read|gpu__time_duration|Duration|Compute \(SM\)|Memory Throughput|Achieved Occupancy|Block Limit|Waves" "$T/ncu_v2c.log" | tr -d '\r' | head -22
