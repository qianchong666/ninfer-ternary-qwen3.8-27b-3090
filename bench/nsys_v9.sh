#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' /c/Windows/System32/cmd.exe /c 'H:/ninfer-ternary/build.bat' > "$T/build_nsys.log" 2>&1
tr -d '\r' < "$T/build_nsys.log" | grep -a -o BUILD_OK | head -1
"$NSYS" profile --trace=cuda --stats=false --force-overwrite=true -o "$T/nsys_v9" \
  ./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer \
  --prompt "$(cat "$T/prompt_long.txt")" --max-new 8 --device 1 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --no-thinking \
  --prefill-chunk 8192 --spec mtp --draft-tokens 4 --lm-head-draft > "$T/nsys_run.log" 2>&1
tr -d '\r' < "$T/nsys_run.log" | grep -aE "prefill speed|text prefill" | head -3
"$NSYS" stats --report cuda_gpu_kern_sum --format table "$T/nsys_v9.nsys-rep" 2>/dev/null | head -30
echo NSYS_DONE
