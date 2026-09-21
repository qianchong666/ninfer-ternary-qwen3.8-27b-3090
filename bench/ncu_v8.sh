#!/bin/bash
# ncu on the v8 mma kernel, isolated microbench, realistic shape (mlp gate/up of the 27B: rows=34816 k=5120)
cd /h/ninfer-ternary || exit 1
T="$LOCALAPPDATA/Temp"
NCU=$(cat "$T/ncu_path.txt")
export PATH="/h/cuda-13.3/bin:$PATH"
"$NCU" --kernel-name regex:mma_v8 --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section Occupancy \
  ./bench/mb.exe 34816 40 4636 8 > "$T/ncu_v8.log" 2>&1
echo "NCU_RC=$?"
tr -d '\r' < "$T/ncu_v8.log" | grep -aE "Duration|Elapsed|Compute \(SM\)|Memory Throughput|DRAM Throughput|Issued Warp|Active Warps|Eligible|No Eligible|Issued Ipc|SM Busy|Registers Per Thread|Block Limit|Achieved Occupancy|Achieved Active|Warp Cycles Per|Stall|stalled|Tensor|L1/TEX|L2 Cache" | head -45
echo NCU_V8_DONE
