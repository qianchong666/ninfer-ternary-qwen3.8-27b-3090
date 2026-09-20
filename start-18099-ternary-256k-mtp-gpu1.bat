@echo off
rem Ternary-Bonsai-2-27B (ninfer, sm_86 build) - 256K context + MTP head, GPU1 single card
rem int8 KV is required at 256K: bf16 KV needs 18.67 GB runtime reservation, only 16.82 GB free on a 3090.
cd /d H:\ninfer-ternary
build\b1\apps\ninfer-serve.exe "models\Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer" ^
  --host 0.0.0.0 --port 18099 ^
  --device 1 ^
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 ^
  --max-concurrency 2 ^
  --spec mtp --draft-tokens 4 --lm-head-draft
