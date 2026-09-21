#!/bin/bash
# v9 probe: exact limiter for the v8 mma kernel — tensor pipe %, L1 sub-units, stall reasons, regs
cd /h/ninfer-ternary || exit 1
T="$LOCALAPPDATA/Temp"
NCU=$(cat "$T/ncu_path.txt")
export PATH="/h/cuda-13.3/bin:$PATH"
"$NCU" --kernel-name regex:mma_v8 --launch-count 1 \
  --metrics launch__registers_per_thread,launch__shared_mem_per_block_static,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_tensor.sum,smsp__inst_executed.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,l1tex__throughput.avg.pct_of_peak_sustained_active,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,dram__bytes_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct \
  ./bench/mb.exe 34816 40 4636 8 > "$T/ncu_v9probe.log" 2>&1
echo "NCU_RC=$?"
tr -d '\r' < "$T/ncu_v9probe.log" | grep -aE "registers_per_thread|shared_mem_per_block|occupancy_limit|pipe_tensor|inst_executed|issue_active|throughput.avg|wavefronts|sectors_pipe|requests_pipe|bank_conflicts|dram__bytes_read|stalled" | head -40
echo NCU_V9PROBE_DONE
