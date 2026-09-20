import json, time, urllib.request

P = ("Write a long, detailed technical explanation of how ternary quantization reduces "
     "memory bandwidth requirements in LLM inference.")
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))


MODEL_BY_PORT = {18090: "qwen3.8-27b", 18099: "Bonsai2-27B-Abliterated-PQ2"}


def call(port, prompt, mx):
    body = {"model": MODEL_BY_PORT.get(port, "qwen3.8-27b"), "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mx, "temperature": 0.2, "enable_thinking": False}
    data = json.dumps(body).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(op.open(req, timeout=900))
    except Exception as e:                                   # noqa: BLE001
        return None, None, None, str(e)[:80]
    dt = time.time() - t0
    u = r.get("usage", {})
    txt = (r["choices"][0]["message"].get("content") or "")
    return dt, u.get("completion_tokens"), u.get("prompt_tokens"), txt[:48].replace("\n", " ")


for port in (18090, 18099):
    dt, ct, pt, txt = call(port, P, 16)                       # warm up
    if ct is None:
        print("port %d: unreachable (%s)" % (port, txt))
        continue
    for i in range(2):
        dt, ct, pt, txt = call(port, P, 256)
        if ct is None:
            print("port %d: request failed (%s)" % (port, txt))
            break
        print("port %d: wall %5.2fs  prompt %d  gen %d  -> %6.1f tok/s wall   | %s"
              % (port, dt, pt, ct, ct / dt, txt))
print("API_AB_DONE")
