#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
N="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
M=models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer
"$N" profile --trace=cuda --force-overwrite=true -o "$T/n1" ./build/b1/apps/ninfer.exe "$M" \
  --prompt "Hello world" --max-new 1 --device 1 --max-context 2048 --kv-capacity 2048 --no-thinking > "$T/n1.log" 2>&1
echo "PROF_RC=$?"
"$N" stats --report cuda_gpu_kern_sum --format table "$T/n1.nsys-rep" > "$T/n1_stats.txt" 2>&1
head -24 "$T/n1_stats.txt"
echo NSYS2_DONE
