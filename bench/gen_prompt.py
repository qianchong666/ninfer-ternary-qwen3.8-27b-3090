import os
T = os.environ['LOCALAPPDATA'] + '/Temp'
base = ("The memory hierarchy of a modern GPU is organised around the observation that most of the "
        "latency a kernel experiences comes from waiting on DRAM rather than from arithmetic. A streaming "
        "multiprocessor therefore keeps a large register file, a software managed scratchpad, and a set of "
        "caches whose block sizes are tuned for the access patterns of tiled matrix multiplication. ")
paras = []
for i in range(28):
    paras.append("Section %d. %sVariant %d emphasises occupancy, vector width, and the number of independent "
                 "loads kept in flight by each warp." % (i + 1, base * 2, i))
txt = "\n\n".join(paras)
p = T + '/prompt_long.txt'
open(p, 'w', encoding='utf-8').write(txt)
print("bytes=%d  approx_tokens=%d" % (len(txt), len(txt) // 4))
