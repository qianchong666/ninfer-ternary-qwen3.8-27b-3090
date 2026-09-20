import re
data = open(r'H:/ninfer-3090/ninfer.exe', 'rb').read()
runs = [r.decode('ascii', 'ignore') for r in re.findall(rb'[ -~]{6,}', data)]
src = sorted({s for s in runs if re.search(r'\.(cpp|cu|cuh|h|hpp)$', s) and len(s) < 60})
print("=== 源码/目标文件名 ===")
for s in src[:60]:
    print(s)
print("=== 含 small_t / ternary / pq2 / ldmatrix 的短串 ===")
pat = re.compile(r'(?i)(small_t|ternary|pq2|ldmatrix|cp\.async|m16n8k|dp4a)')
seen = set()
for s in runs:
    if pat.search(s) and len(s) < 90 and s not in seen:
        seen.add(s)
print(len(seen))
for s in sorted(seen)[:40]:
    print(s)
