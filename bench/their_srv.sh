#!/bin/bash
cd /h/ninfer-3090 || exit 1
T=$LOCALAPPDATA/Temp
./ninfer-serve.exe models/qwen3_8_27b_huihui_abliterated.ninfer --host 127.0.0.1 --port 18110 --max-context 219136 --kv-capacity 219136 --kv-dtype rk8v4 --max-concurrency 1 --prefill-chunk 1024 --spec mtp --draft-tokens 3 --lm-head-draft --temperature 0.2 > "$T/their_srv.log" 2>&1 &
SRV=$!
for i in $(seq 1 60); do
  sleep 5
  if curl -s --noproxy '*' --max-time 3 http://127.0.0.1:18110/v1/models 2>/dev/null | grep -q max_model_len; then echo "READY after $((i*5))s"; break; fi
done
/c/Python314/python.exe H:/ninfer-ternary/bench/their_ttft.py
kill $SRV 2>/dev/null
sleep 2
echo THEIR_TTFT_DONE
