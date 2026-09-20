#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
NCU=$(cat "$T/ncu_path.txt")
export PATH="/h/cuda-13.3/bin:$PATH"
"$NCU" --kernel-name regex:v3 --launch-count 2 \
  --metrics launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,gpu__time_duration.sum \
  ./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer \
  --prompt "$(cat "$T/prompt_mid.txt")" --max-new 1 --device 0 \
  --max-context 32768 --kv-capacity 32768 --kv-dtype int8 --no-thinking \
  > "$T/ncu_v3b.log" 2>&1
echo RC=$?
grep -aE "registers_per_thread|warps_active|dram__bytes_read|wavefronts_mem_shared|gpu__time_duration" "$T/ncu_v3b.log" | tr -d '\r' | head -12
