#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
F=src/ninfer/src/ops/linear/ternary/v2c_a.cuh
sed -i 's/constexpr int kV2cMinChunk = 16;/constexpr int kV2cMinChunk = 1000000;/' "$F"
bash bench/c_rebuild.sh > "$T/ab_build.log" 2>&1
./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer --prompt "$(cat "$T/prompt_long.txt")" --max-new 4 --device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking > "$T/our_long_v2only.log" 2>&1
echo -n "v2-only 长prompt预填: "; grep "prefill speed" "$T/our_long_v2only.log" | tr -d '\r' | grep -oE "[0-9.]+ tok/s"
sed -i 's/constexpr int kV2cMinChunk = 1000000;/constexpr int kV2cMinChunk = 16;/' "$F"
bash bench/c_rebuild.sh > "$T/ab_build2.log" 2>&1
echo -n "恢复构建: "; grep -c BUILD_OK "$T/ab_build2.log"
echo AB_DONE
