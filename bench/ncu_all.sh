#!/bin/bash
cd /h/ninfer-ternary || exit 1
T=$LOCALAPPDATA/Temp
NCU=$(cat "$T/ncu_path.txt")
export PATH="/h/cuda-13.3/bin:$PATH"
"$NCU" --csv --launch-count 12 --metrics gpu__time_duration.sum,dram__bytes_read.sum ./build/b1/apps/ninfer.exe models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer --prompt "$(cat "$T/prompt.txt")" --max-new 2 --device 0 --max-context 32768 --kv-capacity 32768 --kv-dtype int8 --no-thinking > "$T/ncu_all.log" 2>&1
echo RC=$?
/c/Python314/python.exe -c "
import csv,sys
rows=[r for r in csv.reader(open(r'$T/ncu_all.log',newline='')) if len(r)>3 and r[0].strip().isdigit()]
agg={}
for r in rows:
    name=r[2].strip()
    try: dur=float(r[3]); drb=float(r[4])
    except: continue
    a=agg.setdefault(name,[0,0.0,0.0]); a[0]+=1; a[1]+=dur; a[2]+=drb
for name,(n,dur,drb) in sorted(agg.items(), key=lambda kv:-kv[1][1]):
    print('%-52s n=%2d  %8.1f us  dram_read=%8.2f MB' % (name[:52], n, dur, drb/1048576))
"
