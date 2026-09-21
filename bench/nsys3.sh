#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
N="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
M=models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer
"$N" profile --trace=cuda --force-overwrite=true -o "$T/n2" ./build/b1/apps/ninfer.exe "$M" \
  --prompt "$(cat "$T/prompt_long.txt")" --max-new 1 --device 1 --max-context 8192 --kv-capacity 8192 --no-thinking > "$T/n2.log" 2>&1
echo "PROF_RC=$?"
"$N" stats --report cuda_gpu_kern_sum --format table "$T/n2.nsys-rep" > "$T/n2_stats.txt" 2>&1
head -24 "$T/n2_stats.txt"
echo NSYS2_DONE
