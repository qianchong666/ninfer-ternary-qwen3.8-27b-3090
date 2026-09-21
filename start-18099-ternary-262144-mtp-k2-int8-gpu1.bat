@echo off
rem Ternary-Bonsai-2-27B : MTP K=2 WITHOUT --lm-head-draft.
rem Upstream measured table: K=2 is best (accept 58.7%), K=3/4/5 collapse (40.9/33.2/22.9%),
rem and --lm-head-draft makes it worse (63.5% -> 55.0%).
rem NOTE: python is an ABSOLUTE path on purpose - a bare "python" resolves to Hermes' venv.
rem NOTE: an idle engine prints NOTHING (there is no periodic heartbeat), so a window that
rem       sits still between requests is normal. Send a request to see new lines.
chcp 65001 >nul
cd /d H:\ninfer-ternary

echo  ==========================================================
echo   Bonsai-27B   MTP K=2   to   http://127.0.0.1:18099
echo   live log scrolls below; same lines also go to serve-18099-k2.log
echo  ==========================================================
echo.

"C:\Users\Administrator\AppData\Local\Programs\Python\Python314\python.exe" "H:\ninfer-ternary\tee_engine.py" "H:\ninfer-ternary\serve-18099-k2.log" -- "build\b1\apps\ninfer-serve.exe" "models\Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer" --host 0.0.0.0 --port 18099 --device 1 --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --max-concurrency 2 --spec mtp --draft-tokens 2 --cors --model-id Bonsai2-27B-Abliterated-PQ2

echo.
echo [engine exited]  full log: H:\ninfer-ternary\serve-18099-k2.log
pause
