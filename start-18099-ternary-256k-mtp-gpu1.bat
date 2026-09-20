@echo off
rem Ternary-Bonsai-2-27B-Abliterated (ninfer sm_86 build) - 256k variant (same 262144 ctx) + MTP head, GPU1 single card.
rem KV dtype int8: bf16 KV does not fit at 256K on a 3090 (needs 18.68 GB, ~16.5 GB free).
rem
rem LOGGING - read this if the window ever looks frozen again. cmd has no tee, and BOTH
rem PowerShell approaches tried here silently stop showing new lines under Windows Terminal:
rem   [1] engine piped into powershell, echoing via dollar-underscore and Add-Content
rem       window froze right after "listening on", while the log FILE kept growing
rem   [2] Get-Content -Wait following the log file
rem       stopped tracking appends entirely (measured: file grew 1499 then 2029 bytes,
rem       the follower never printed the new lines)
rem So the engine is now wrapped in tee_engine.py, which flushes every line to BOTH this
rem window and serve-18099.log. Double-click this .bat and the log scrolls live.
rem NOTE: python is an ABSOLUTE path on purpose - a bare "python" resolves to Hermes' venv.
rem NOTE: an idle engine prints NOTHING (there is no periodic heartbeat), so a window that
rem       sits still between requests is normal. Send a request to see new lines.
chcp 65001 >nul
cd /d H:\ninfer-ternary

echo  ==========================================================
echo   Bonsai-27B   to   http://127.0.0.1:18099
echo   live log scrolls below; same lines also go to serve-18099.log
echo  ==========================================================
echo.

"C:\Users\Administrator\AppData\Local\Programs\Python\Python314\python.exe" "H:\ninfer-ternary\tee_engine.py" "H:\ninfer-ternary\serve-18099.log" -- "build\b1\apps\ninfer-serve.exe" "models\Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer" --host 0.0.0.0 --port 18099 --device 1 --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --max-concurrency 2 --spec mtp --draft-tokens 4 --lm-head-draft --cors --model-id Bonsai2-27B-Abliterated-PQ2

echo.
echo [engine exited]  full log: H:\ninfer-ternary\serve-18099.log
pause
