#!/bin/bash
# Engine-side A/B harness. Long-prompt prefill speed (the number the v11 work is judged on, 3
# back-to-back runs in one session) plus one greedy fixed-prompt generation whose TEXT is compared
# between builds.
# usage: v11_engine.sh <tag>
set -u
cd /h/ninfer-ternary || exit 1
T="$LOCALAPPDATA/Temp"
TAG="${1:-base}"
BIN=./build/b1/apps/ninfer.exe
MODEL=models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer
FWD="--device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking"

for i in 1 2 3; do
  "$BIN" "$MODEL" --prompt "$(cat "$T/prompt_long.txt")" --max-new 4 $FWD > "$T/eng_${TAG}_long_$i.log" 2>&1
  echo -n "long[$i]: "
  tr -d '\r' < "$T/eng_${TAG}_long_$i.log" | grep -a "prefill speed" | grep -oE "[0-9.]+k? tok/s"
done

"$BIN" "$MODEL" --prompt "$(cat "$T/prompt_mid.txt")" --max-new 96 --temperature 0 --seed 1 $FWD > "$T/eng_${TAG}_acc.log" 2>&1
echo "acc done: $T/eng_${TAG}_acc.log"
tr -d '\r' < "$T/eng_${TAG}_acc.log" | grep -a "prefill speed" | head -1
echo ENGINE_DONE
