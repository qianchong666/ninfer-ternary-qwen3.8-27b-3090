# v2：三元 27B 的张量核验证内核（sm_86 / 2×RTX 3090）

## 结论数字（RTX 3090，GPU1 单卡，256K ctx + int8 KV）

| 项目 | 原始参考内核 | v1 mma | **v2 mma** | 对照引擎 :18090 |
|---|---|---|---|---|
| 预填 3833 token | 39.7 | 154.8 | **220.4** | ~250 |
| 纯解码 T=1 | 46.8 | 45.0 | 44.8 | 34 |
| MTP K=4 解码（CLI） | 37.7（负收益） | 51.6 | **69.5** | 65.5 |
| :18099 链路墙钟 256 token | 33.2 | 46.9 | **66.3~67.3** | 53.0~63.2 |

启动口径：`--spec mtp --draft-tokens 4 --lm-head-draft`，256K ctx + int8 KV，GPU1。

## v2 的设计（每一条都有计数器证据）

1. **共享内存只放原始 2-bit 码字**：每行 32 B + 16 B padding（行距 48 B，让 8 个行地址落在 8 个
   不同 bank；同地址的 4 个 lane 走广播，零冲突）。每 warp 768 B，不再放解码后的 bf16（原 4.35 KB）。
   证据：v1 `Block Limit Shared Mem = 5`（占用率被锁死 48%），v2 变成 10，瓶颈转移到寄存器。
2. **解码搬进寄存器**：每个 k-tile 一次对齐的 4 字节 smem 读同时拿到需要的两个码字节
   （`(cpar>>1)*8` 选字节、`cpar&1` 选半字节），移位 + F2B 现场拼出 mma 的 A 片段。
3. **scale 折进片段**：`(code-1)*scale` 直接乘在拼片段那步，省掉独立累加器和整轮缩放
   （少 4 个寄存器 + 每 group 少 8 条 FMA）。
4. **k-split**：16 行/block，组范围在 4 个 warp 间切分，部分和在共享内存求和。
5. 没有抄对照引擎的"按 T 实例化内核"——sm_86 上 bf16 只有 `m16n8k16` 一种 mma 形状，
   T=4 也得跑满 n=8，对他们没收益的模式对我们同样没收益。

## 计数器对照（ncu）

| | 对照 `q4_small_t_mma_kernel` | 我们 v1 | 我们 v2 |
|---|---|---|---|
| Achieved Occupancy | 92.19% | 8.3% | 23.3% |
| DRAM Throughput | 39.1% | 21.6% | 19.6~24% |
| L1/TEX | 89.4% | 88.2% | 20~60% |
| 有效验证带宽 | — | 168 GB/s | **225 GB/s** |
| Block Limit | — | smem 5） | 寄存器 6（85 regs） |

## 开关与回退

- `NINFER_MMA=0` → 回退到引擎原生参考内核（数值严格一致，用于 A/B）
- `NINFER_MMA_DEBUG=1` → 打印真实传参、指针、对齐
- 正确性金标准：`--prompt "The capital of France is"` 必须输出 `The capital of France is **Paris**.`

## 文件

- `src/ninfer/src/ops/linear/ternary/ternary_mma_v2.cuh` — v2 内核（本文件对应的主体）
- `src/ninfer/src/ops/linear/ternary/ternary_rowsplit_gemm.cu` — v1 内核 + 派发 + 启动器（一行切内核名）
- `bench/t_v2.sh` — 四项基准；`bench/api_ab.py` — 同刻链路对打
- `start-18099-ternary-256k-mtp-gpu1.bat` — 服务启动（K=4 + lm-head-draft）

## 还能再压的地方

- **占用率 23% → 目标 45%**：现在卡在 85 寄存器（限制 6 block）。压到 ~42 寄存器即可翻倍。
- 预填 220 → 250：需要更细的 grid / 小 T 分块。
- 纯解码 44.8 t/s（336 GB/s，带宽利用 36%）：T=1 仍是 SIMT GEMV，换张量核是另一条路。
