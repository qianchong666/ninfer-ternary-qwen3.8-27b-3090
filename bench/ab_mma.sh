#!/bin/bash
# A/B: mma path (default) vs reference path (NINFER_MMA=0), same prompt, greedy, no thinking
cd /h/ninfer-ternary || exit 1
M="models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer"
Q="The capital of France is"
COMMON="--max-new 16 --device 1 --max-context 4096 --kv-capacity 4096 --kv-dtype int8 --no-thinking"
echo "======== MMA ON (debug) ========"
NINFER_MMA_DEBUG=1 ./build/b1/apps/ninfer.exe "$M" --prompt "$Q" $COMMON 2>&1 | grep -vE "^\s*$" | head -14
echo
echo "======== MMA OFF (reference path) ========"
NINFER_MMA=0 ./build/b1/apps/ninfer.exe "$M" --prompt "$Q" $COMMON 2>&1 | grep -vE "^\s*$" | head -12
echo AB_DONE
