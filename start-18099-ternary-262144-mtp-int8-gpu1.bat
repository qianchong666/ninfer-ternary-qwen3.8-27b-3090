@echo off
rem Ternary-Bonsai-2-27B-Abliterated (ninfer sm_86 build) - 262144 context + MTP head, GPU1 single card.
rem KV dtype: rk8v4 is NOT usable here. The k8v4 / NVFP4 KV path needs the SM120 (Blackwell)
rem kind::f8f6f4 / e2m1 kernels and the engine refuses it on sm_86 with
rem "use --kv-dtype bf16|int8|fp8 instead". Measured at 262144 on one 3090:
rem   bf16 KV  -> does not fit (needs 18.68 GB runtime, 16.47 GB available)
rem   int8 KV  -> fits (plans 16.6 GiB device total) and is the fastest option: 80.1 tok/s decode
cd /d H:\ninfer-ternary
build\b1\apps\ninfer-serve.exe "models\Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer" ^
  --host 0.0.0.0 --port 18099 ^
  --device 1 ^
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 ^
  --max-concurrency 2 ^
  --spec mtp --draft-tokens 4 --lm-head-draft
