import re
data = open(r'H:/ninfer-3090/ninfer.exe', 'rb').read()
runs = [r.decode('ascii', 'ignore') for r in re.findall(rb'[ -~]{12,}', data)]
pat = re.compile(r'(SmallTSchedule|SmallTGeometry|RowsplitGemvSchedule|LaneMapping|ActivationAccess)')
seen = []
for s in runs:
    if pat.search(s) and s not in seen:
        seen.append(s)
print("unique:", len(seen))
for s in seen[:25]:
    print(s[:170])
