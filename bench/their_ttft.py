import json, time, urllib.request, os
T = os.environ['LOCALAPPDATA'] + '/Temp'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
prompt = open(T + '/prompt_long.txt', encoding='utf-8', errors='replace').read()

def ask(content, tag):
    body = json.dumps({"model": "qwen3.8-27b",
                       "messages": [{"role": "user", "content": content}],
                       "max_tokens": 1, "temperature": 0.2}).encode()
    req = urllib.request.Request("http://127.0.0.1:18110/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with opener.open(req, timeout=600) as r:
        data = json.loads(r.read().decode())
    dt = time.time() - t0
    pt = data.get("usage", {}).get("prompt_tokens", 0)
    print("%-14s prompt_tokens=%-6s wall=%6.2fs  -> %6.0f tok/s prefill" % (tag, pt, dt, (pt / dt) if dt else 0))

ask(prompt, "long#1")
ask(prompt, "long#2 same")
ask("Explain how a GPU hides memory latency in one paragraph.", "short")
