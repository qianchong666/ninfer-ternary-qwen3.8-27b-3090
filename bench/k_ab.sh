#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
B=./build/b1/apps/ninfer.exe
M=models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer
P="$(cat "$T/prompt_long.txt")"
P4=$(/c/Windows/System32/WBEM/WMIC.exe process where "name='ninfer-serve.exe'" get ProcessId /format:value 2>/dev/null | tr -d '\r' | grep -oE "[0-9]+" | head -1)
[ -n "$P4" ] && /c/Windows/System32/taskkill.exe /F /PID "$P4" >/dev/null 2>&1 && echo "stopped serve $P4"
sleep 6
$B $M --prompt "$P" --max-new 512 --device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking --spec mtp --draft-tokens 4 --lm-head-draft > "$T/k4_ab.log" 2>&1
$B $M --prompt "$P" --max-new 512 --device 1 --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --no-thinking --spec mtp --draft-tokens 2 > "$T/k2_ab.log" 2>&1
echo "== K=4 + lm-head-draft (current bat) =="
grep -hE "decode speed|acceptance" "$T/k4_ab.log" | tr -d '\r'
echo "== K=2 (upstream-measured best) =="
grep -hE "decode speed|acceptance" "$T/k2_ab.log" | tr -d '\r'
echo K_AB_DONE
