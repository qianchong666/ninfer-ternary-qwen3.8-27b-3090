#!/bin/bash
cd /h/ninfer-3090 || exit 1
T=$LOCALAPPDATA/Temp
NCU=$(cat "$T/ncu_path.txt")
export PATH="/h/cuda-13.3/bin:$PATH"
"$NCU" --section SpeedOfLight --section Occupancy \
  --launch-skip 40 --launch-count 2 -k regex:small_t \
  ./ninfer.exe models/qwen3_8_27b_huihui_abliterated.ninfer \
  --prompt "The capital of France is" --max-new 32 \
  --device 1 --kv-dtype rk8v4 --max-context 16384 --kv-capacity 16384 \
  --prefill-chunk 1024 --no-thinking \
  --spec mtp --draft-tokens 3 --lm-head-draft > "$T/ncu_theirs.log" 2>&1
echo "RC=$?"
