#!/bin/bash
# MTP draft-mode sweep with the new tensor-core verify path
cd /h/ninfer-ternary || exit 1
M="models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer"
T="$LOCALAPPDATA/Temp"
B="./build/b1/apps/ninfer.exe"
P="Write a long, detailed technical explanation of how ternary quantization reduces memory bandwidth requirements in LLM inference."
C="--max-new 512 --device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking"
$B "$M" --prompt "$P" $C --spec mtp --draft-tokens 2 --lm-head-draft > "$T/lmd_k2.log" 2>&1
$B "$M" --prompt "$P" $C --spec mtp --draft-tokens 3 --lm-head-draft > "$T/lmd_k3.log" 2>&1
$B "$M" --prompt "$P" $C --spec mtp --draft-tokens 4 --lm-head-draft > "$T/lmd_k4.log" 2>&1
$B "$M" --prompt "$P" $C --spec dflash --draft-tokens 3 > "$T/lmd_dflash.log" 2>&1
for f in lmd_k2 lmd_k3 lmd_k4 lmd_dflash; do
  echo "######## $f"
  grep -E "decode speed|prefill speed|mtp acceptance rate|mtp acceptance length|mtp rounds|dflash|error" "$T/$f.log" | sed 's/summary *//' | head -9
done
echo LMD_DONE
