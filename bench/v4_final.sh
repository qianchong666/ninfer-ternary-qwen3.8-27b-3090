#!/bin/bash
cd /h/ninfer-ternary/src/ninfer/src/ops/linear/ternary || exit 1
python -c "
p='v4_a.cuh'
s=open(p).read().replace('constexpr int kV4TilesPerWarp = 16;','constexpr int kV4TilesPerWarp = 8;')
open(p,'w').write(s)
print('tiles_per_warp=8')"
grep -n "kV4TilesPerWarp = " v4_a.cuh
cd /h/ninfer-ternary || exit 1
bash bench/v3_engine.sh 2>&1 | tail -6
