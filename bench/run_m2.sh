#!/bin/bash
# round-2 engine bench: prefill via the new mma path + MTP K sweep
cd /h/ninfer-ternary || exit 1
M="models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer"
T="$LOCALAPPDATA/Temp"
B="./build/b1/apps/ninfer.exe"
C="--device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking"

"$B" "$M" --messages "$T/long.json" --max-new 128 $C > "$T/m2_long_nomtp.log" 2>&1; echo "LONG=$?"
"$B" "$M" --prompt "Write a long, detailed technical explanation of how ternary quantization reduces memory bandwidth requirements in LLM inference." --max-new 512 $C > "$T/m2_nomtp.log" 2>&1; echo "BASE=$?"
"$B" "$M" --prompt "Write a long, detailed technical explanation of how ternary quantization reduces memory bandwidth requirements in LLM inference." --max-new 512 $C --spec mtp --draft-tokens 3 > "$T/m2_k3.log" 2>&1; echo "K3=$?"
"$B" "$M" --prompt "Write a long, detailed technical explanation of how ternary quantization reduces memory bandwidth requirements in LLM inference." --max-new 512 $C --spec mtp --draft-tokens 5 > "$T/m2_k5.log" 2>&1; echo "K5=$?"
"$B" "$M" --prompt "Write a long, detailed technical explanation of how ternary quantization reduces memory bandwidth requirements in LLM inference." --max-new 512 $C --spec mtp --draft-tokens 7 > "$T/m2_k7.log" 2>&1; echo "K7=$?"

for f in m2_long_nomtp m2_nomtp m2_k3 m2_k5 m2_k7; do
  echo "######## $f"
  grep -E "prompt tokens|generated tokens|prefill speed|decode speed|throughput \(overall\)|mtp acceptance rate|mtp acceptance length|mtp rounds|planned device total" "$T/$f.log" | head -9
done
echo M2_DONE
