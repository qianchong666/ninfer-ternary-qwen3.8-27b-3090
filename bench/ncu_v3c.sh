#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
NCU=$(cat "$T/ncu_path.txt")
export PATH="/h/cuda-13.3/bin:$PATH"
"$NCU" --kernel-name regex:v3 --launch-count 1 \
  --metrics launch__registers_per_thread,launch__occupancy_limit_registers,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,dram__bytes_read.sum,gpu__time_duration.sum \
  ./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer \
  --prompt "$(cat "$T/prompt_tiny2.txt")" --max-new 2 --device 0 \
  --max-context 8192 --kv-capacity 8192 --kv-dtype int8 --no-thinking > "$T/ncu_v3c.log" 2>&1
echo RC=$?
grep -aE "registers_per_thread|occupancy_limit_registers|warps_active|wavefronts_mem_shared|dram__bytes_read|gpu__time_duration|No kernels" "$T/ncu_v3c.log" | tr -d '\r' | head -10
