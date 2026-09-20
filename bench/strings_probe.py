import re, sys
path = r'H:/ninfer-3090/ninfer.exe'
data = open(path, 'rb').read()
print("size", len(data))
runs = re.findall(rb'[ -~]{8,}', data)
print("runs", len(runs))
key = re.compile(rb'(?i)(ldmatrix|mma\.sync|dp4a|wgmma|gemv|dequant|ternary|pq2|imma|m16n8k)')
hits = []
seen = set()
for r in runs:
    if key.search(r):
        t = r.decode('ascii', 'ignore')
        if t not in seen:
            seen.add(t)
            hits.append(t)
print("hits", len(hits))
for t in hits[:45]:
    print(t[:150])
