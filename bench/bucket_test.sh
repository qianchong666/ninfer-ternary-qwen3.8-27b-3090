#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
B=./build/b1/apps/ninfer.exe
M=models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer
P="$(cat "$T/prompt_long.txt")"
P4=$(/c/Windows/System32/WBEM/WMIC.exe process where "name='ninfer-serve.exe'" get ProcessId /format:value 2>/dev/null | tr -d '\r' | grep -oE "[0-9]+" | head -1)
[ -n "$P4" ] && /c/Windows/System32/taskkill.exe /F /PID "$P4" >/dev/null 2>&1 && echo "stopped serve $P4"
sleep 5
NINFER_MMA_DEBUG=1 $B $M --prompt "$P" --max-new 4 --device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking --prefill-batch-bucket 16384 > "$T/bucket.log" 2>&1
echo "== 探针看到的 T =="
grep -aoE "\[mma\] rows=[0-9]+ k=[0-9]+ groups=[0-9]+ t=[0-9]+" "$T/bucket.log" | sort -u | head -5
echo "== 预填 =="
grep -hE "prefill speed" "$T/bucket.log" | tr -d '\r'
echo BUCKET_DONE
