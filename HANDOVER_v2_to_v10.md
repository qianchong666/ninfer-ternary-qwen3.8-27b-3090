# Ternary-Bonsai-2-27B 三元推理内核：从 0 到 v10 交接报告

> 目标：在 Windows + 2×RTX 3090 上，把 2-bit（三元）权重模型的**预填速度**做到极限。
> 本报告自包含：环境安装、复现命令、每一版改了什么/快了多少、**所有踩过的坑**、后续研究方向。
> 阅读方式：只想跑起来 → 看 §1 §2；想继续优化 → 看 §3 §4 §5 §6。

## 0. 成绩单

| 指标 | 起点 | 现在 (v10) | 提升 |
|---|---|---|---|
| 引擎预填 | 228.7 tok/s | **1.34~1.36k tok/s** | 5.9× |
| 引擎解码 | — | **74.4 tok/s** | 对照 dense 27B 基线 53~63 t/s |
| 精度 | 逐位精确路径可用 | v8 起用 f16 累加 (rel_l2 1.49e-3)；**v9→v10 逐位一致** | 无额外损失 |

对照基线（用户 dense 27B，`:18090`）：预填 646 t/s、解码 53~63 t/s → **本方案预填 2.1×、解码 1.2~1.4×**。

引擎预填演进（同一参数集实测，非推算）：
`v2 228.7 → v2c ~308 → v3 411 → v4 569 → v7 1.08k → v8 1.21k → v9 1.21k → v10 1.34~1.36k`

## 1. 环境与安装（可整段交给 AI 执行）

### 1.1 硬件 / 系统约束
- Windows 10，2× RTX 3090（sm_86，**每卡 82 个 SM**）
- **不要动现有 NVIDIA 驱动（610.62）**。CUDA 工具链旁装在 `H:\cuda-13.3`（整目录可删＝可回退），系统 CUDA 13.1 保留不动
- **GPU0 接了显示器**（空闲时钟 1395MHz），**GPU1 是引擎用的卡**（1920MHz）→ **一切基准测试都跑 GPU1**（GPU0 实测慢 20~38%）
- 本机 curl 一律 `--noproxy '*'`；中文请求体写 UTF-8 文件再 `--data-binary @file`；联网走代理 `http://127.0.0.1:7897`
### 1.2 代码与模型布局
```
H:\ninfer-ternary                  # 主仓；src/ninfer 是 git submodule（三元内核在这里）
H:\ninfer-ternary\venv             # 独立虚拟环境：verify 脚本必须用它
                                   #   系统 python3.14 只能跑临时/bench 脚本
H:\ninfer-ternary\models\Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer   # 转换产物 ~7.45 GiB(groupwise-int)
H:\Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF\*.gguf                # 源 GGUF 约 17G，**不要重新下载**
H:\ninfer-ternary\build\b1\apps\ninfer.exe                               # 引擎二进制
```
- 源 GGUF 内只有 `PQ2_0`(142) / `PTQ1_0`(143) / `BF16` / `F32`：**没有 4-bit 权重数据**，所以"q4 质量"路线在这个模型上不存在
- 二元组：三元权重 7.45 GiB 全在显存；`--kv-dtype` 只支持 bf16/int8（`k8v4`/`NVFP4` 只编给 SM120，这张卡会拒收）；**262144 上下文必须 int8**
- 内核源码：`src/ninfer/src/ops/linear/ternary/`（`v9_b.cuh` `v9_c.cuh` `v10_pack.cuh` `ternary_rowsplit_gemm.cu`）

### 1.3 构建
```
# 引擎（增量）
cd H:/ninfer-ternary && <ninja/build 脚本>          # 产物 build/b1/apps/ninfer.exe
# 微基准（独立构建，用它做所有内核级实验）
bench/mb_build.bat        # v4/v8/v9/v10 同形状对拉 + 形状扫描
bench/peak_build.sh       # 峰值测试（f16/s8/bf16 各累加方式）
```
两个 .bat 都是 `vcvars64` + `H:/cuda-13.3/bin/nvcc.exe`，**路径必须写正斜杠**（原因见 §5 坑 2）。

## 2. 复现标准测试

```bash
cd H:/ninfer-ternary && ./build/b1/apps/ninfer.exe \
  models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.ninfer \
  --prompt "$(cat "$LOCALAPPDATA/Temp/prompt_long.txt")" --max-new 64 \
  --device 1 --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --no-thinking --prefill-chunk 8192 --spec mtp --draft-tokens 4 --lm-head-draft
```
- **真数校验：摘要里 `reused prompt tokens` 必须为 `0`**（若引擎有前缀缓存，重复 prompt 会虚报到 3728~306910 t/s 的假数）
- 对照引擎（`H:\infer-3090`，`:18090`，用户主力）**只读不改**；只杀自己起的进程
- 引擎端有 **±10% 时钟/热态噪声** → 任何 A/B 必须**同场次、背靠背、各跑 3~4 次**，单次比较无意义
## 3. 从 0 到 v10：每一版改了什么

> 早期版本（v2/v2c/v3）的逐条改动见 `git log`；下表是各版的引擎实测预填速度与**可确认**的结构变化。

| 版本 | 引擎预填 | 关键改动 | 精度 |
|---|---|---|---|
| v2 | 228.7 | 基线三元 mma 路径 | 逐位精确 |
| v2c | ~308 | （见 git log） | 逐位精确 |
| v3 | 411 | （见 git log） | 逐位精确 |
| v4 | 569 | 行切分方案（每 CTA 16 行组、token 分块 `kV4MinChunk`），成为后续所有版本的骨架 | 逐位精确 |
| **v7** | **1.08k** | **行块加高：每 warp 4 个「16 行组」共享 B 寄存器** ＝ 主杠杆（B 的 L2 流量 ∝ rows×tokens×k/16，把行块加高直接砍掉重复读） | 逐位精确 |
| **v8** | **1.21k** | 改用 **f16 累加** mma：实测 3090 上 fp16 累加 111.6 vs fp32 累加 56.0 TFLOPS = **2×**；配「每 128 组提升到 f32」的分段累加 | rel_l2 1.494e-3 |
| v9 | 1.21k | 每 warp **6 个 tile**（CTA token 窗口 192）＋ 一串编译期开关（`kV9_TILES`/`kV9_LDMATRIX`/`kV9_MINBLOCKS`/…）；**数值与 v8 逐位一致** | 同 v8 |
| **v10** | **1.34~1.36k** | **把激活值按 B-fragment 顺序重打包成 f16**（全局一次，代价 68ms/预填＝2%）：内核里的激活值载入变成**两条完全合并的 128B 载入 + 零类型转换**；**逐位一致** | 同 v8 |

### 3.1 每一版为什么这么做（诊断链，不是猜）

1. **ncu 实测**（真实形状 `rows=34816 k=5120 t=4636`）：张量核只忙 53.2%、L1 64.3%、**DRAM 仅 6.4%**（带宽不是瓶颈）、249 寄存器 → 每 SM 只装 2 个 block（8/48 warp）、每 7.9 周期才发一条指令、7.5 条指令/mma
2. **截肢法**（把一条搬运路径故意算错以测其代价）给出唯一方向：
   基线 68.0 TFLOPS → 砍激活值 91.0(+34%) → 砍权重共享读 77.8(+14%) → 砍解码 71.9(+6%) → **两条都砍 = 104.9 = 实测峰值的 94%**
   → **内核本体几乎完美，缺口 100% 在操作数搬运**
3. 三个**反证**（避免走错路）：强行靠溢出换占用率 = 慢 2.6×；加 `ldmatrix` 省指令 = 速度不变；占用率不是关键（24 条独立累加器链已够）
4. **v7 打 B 的重复读，v8 把累加方式换成 2×，v10 把 B 的载入彻底合并**——全部落在"操作数搬运"这个唯一缺口上
## 4. 来源、引用与致谢（尊重原创）

### 4.1 模型
| 项目 | 出处 |
|---|---|
| **本次实际使用的权重** | `Hikari07jp/Ternary-Bonsai-2-27B-Abliterated-GGUF` → `Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf`（7.21 GB / 2.13 bpw / PQ2_0）。作者说明：在部署量化格式内**原位编辑**，全程无 BF16 反量化、无重量化；851 个张量中改了 400 个（只动 2-bit 码），其余 451 个字节完全一致；编辑了 block scales 之外的 2-bit 码。License: Apache-2.0 |
| **其母包（三元格式与 pack 的原始作者）** | `prism-ml/Ternary-Bonsai-2-27B-gguf`（"Bonsai 2 27B"）—— 提供 PTQ1_0(1.75 bpw 稠密 trit) / PQ2_0(2.13 bpw 2-bit 槽) 两种打包 |
| **上游基座模型** | `Qwen/Qwen3.8-27B`（架构未改） |
| **模型规格（来自 prism-ml 模型卡）** | 27.36B 总参 = 24.35B 语言主干(64 层) + 2.54B embedding/LM head + 0.46B 视觉塔；**混合注意力 ~75% 线性 / ~25% 全注意力**；SwiGLU / RoPE / RMSNorm；262K 上下文；权重 **ternary g128 {−1,0,+1} + FP16 分组 scale** |
| **关键机制：权重内嵌 Hadamard 旋转** | 权重中已折叠了 **blockwise Hadamard 旋转（block 1024，固定 ±1 符号）**，运行时必须对激活值施加配套变换 —— 这正是本引擎里 `ternary_rotate_bf16_kernel` / `ternary_rotate_inverse_inplace` 的存在原因（nsys 实测占预填 2.1%） |

> ⚠️ 由此得出的一条硬性约束：**原版 llama.cpp 跑不了这些文件**（PQ2_0/PTQ1_0 会被判为未知类型；`Q2_0` 能加载但**静默输出垃圾**，因为没有 Hadamard 激活运行时）。三元内核的参考实现位于 **`PrismML-Eng/llama.cpp` fork**。

### 4.2 引擎与工具链
| 项目 | 出处 / 说明 |
|---|---|
| 引擎上游项目 | `https://github.com/Ambolio/ninfer-4090-windows`（本机经 `gh-proxy.com` 镜像克隆；`src/ninfer` 即此 submodule） |
| 搭建指南（用户当时照此搭建） | `https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary` |
| NInfer artifact 生态（模型卡） | HF 用户 `neroued` 发布的 `*-NInfer` 产物（如 `neroued/Qwen3.6-27B-NInfer`）；本仓 `src/ninfer/model-cards/` 下同类卡 |
| GGUF 容器格式 | llama.cpp / ggml（本模型用到 ggml type **142 = PQ2_0**、**143 = PTQ1_0**） |
| 张量核指令 | NVIDIA **PTX ISA**：`mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`（本内核主力）与 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`（v11 计划），注意整数型必须写全四段 `D.A.B.C` |
| 性能剖析工具 | NVIDIA Nsight Compute（ncu，指令级）与 Nsight Systems（nsys，时间线） |
| 本报告的方法论 | ①**峰值探针**（纯依赖链遍历 block 数找硬件真上限）②**截肢法**（故意算错某条搬运路径以测其代价）③**同场次背靠背 A/B**（对抗 ±10% 时钟噪声）—— 前两项是标准技巧的工程化，第三项是被噪声逼出来的 |
| 外部对照数据 | prism-ml 模型卡给出 PQ2_0 在 **RTX 4090** 上 TG128 = 81.2 t/s、PP512 = 3124 t/s。按张量峰值折算（4090 的 f16/f16 约为 3090 的 2.2×），3124 ÷ 2.2 ≈ 1420 t/s，与本方案 1.34k 量级吻合 ✅ |

### 4.3 本轮的原创贡献（v9 / v10 及方法论）
- **v10 = 把激活值按 B-fragment 顺序重打包**（全局一次、代价 2%），使内核内激活值载入变为两条完全合并的 128B 载入且零类型转换 —— 实测 1.44×（引擎真实分块 t=928：68.3 → 98.4 TFLOPS），**数值逐位一致**
- **v9 = 每 warp 6 tile / 192-token CTA 窗口** + 一串编译期开关（`kV9_TILES`/`kV9_LDMATRIX`/`kV9_MINBLOCKS`/`kV9_PACKED`/`kV9_*` 截肢开关）
- **诊断结论**：本内核的瓶颈 100% 在操作数搬运（截肢法：激活值 34% + 权重共享读 14% + 解码 6%；内核本体仅差峰值 6%）
- **2-bit 权重**的来源与格式全部属于 prism-ml / Hikari07jp / Qwen，本报告只做**推理内核优化**，未修改任何权重（转换产物 `.ninfer` 由本仓转换脚本从用户提供的 GGUF 生成）
## 5. 所有的坑（按类别，全部为实测踩过的）

### A. 工具 / Windows 环境
1. **工具参数会被截断**（实测 200~800 字符不等，规律不稳定）：`write_file`/`patch` 只写短内容，**写完必须 `wc -c` 复核**；长内容分段 `cat >> f <<'EOF'` 追加
2. **bash heredoc 里的 Windows 路径必须写正斜杠**：双反斜杠会被折叠，`printf` 把 `\nvcc` 当换行 → `nvcc.exe` 被切成 `vcc.exe`。症状极隐蔽：**`NVCC_RC=123` 且日志里搜不到 error 字样**
3. `.bat` 必须是 **CRLF**，可执行行只写 **ASCII**
4. 原生程序（nvcc / git / node / python）**不认 MSYS 路径**：`/c/Users/...` 会失败，必须传 `C:/Users/...` 风格
5. 调 `cmd.exe` 要加 `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'`
6. 长单行内容容易被拦腰截断 → 拆成多行；**按行替换最稳**：`sed -i 'N s|.*|新内容|' file`（每行 < ~130 字符）
7. 输出**直写日志文件**而非管道；读中文日志要探测编码（gbk / utf-16 / utf-8）
8. 前台命令上限 600s，更长的用后台 + 完成通知

### B. 构建
9. **`--maxrregcount` 对有 `__launch_bounds__` 的 kernel 完全无效**（四组参数耗时一模一样）→ 正确旋钮是 `__launch_bounds__(threads, minBlocks)`
10. **靠溢出换占用率是死路**：`tiles=8 + minblocks=4` = 73.0 ms，比最优 23.9 ms **慢 2.6×**（寄存器是真需求）
11. **改头文件可能不触发重编译**（依赖跟踪缺失）→ 会出现"同一份代码两次跑出 1.22k 和 1.36k"的假象；改完头文件要确认目标真重建了
12. **`#define` 必须在 `#include` 之前**：把 `kV9_PACKED` 写在 include 之后，会编译出「未打包内核吃打包数据」→ 又错又慢
13. `ptxas` 报 `Unexpected instruction types specified for 'mma'` = 类型串写错；**整数 mma 必须写全四段**：`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
14. 引擎二进制、微基准、峰值测试是**三个独立产物**，改一个不会动另一个

### C. 引擎 / 内核约定
15. **每层调用三元 GEMM 是 31 次，不是 6 次**：每个矩阵**按 token 切成 5 块**（GridY=5，≈928 token/块）→ **微基准必须按 `t≈928` 测**（按整块 4636 调优有一半没打在点上）
16. 预填构成（nsys 实测）：**三元 GEMM 85%** / 注意力 2.6% / rotate 2.1% / 打包 2.0% / 其余每个 ≤0.7%
17. 这是 **GDN 线性注意力混合模型（~75% 线性）**：不要按标准 transformer 的直觉给注意力留预算
18. **权重里已内嵌 blockwise Hadamard 旋转**（block 1024，固定 ±1 符号）→ 引擎有 `ternary_rotate_*` 前/后处理，改内核时不能破坏这个约定
19. 分派入口：`ternary_rowsplit_gemm.cu` 里 `if (tokens >= kV4MinChunk)` → 三元 mma 路径；小 token（解码）走 `mma_v2`/`gemv`；**旧 v2 内核仍有 400 次调用（0.6%）**
20. **引擎端有 ±10% 时钟/热态噪声** → 任何 A/B 必须**同场次背靠背各跑 3~4 次**（一个错误结论就是这么来的）
21. **GPU0（带显示器）比 GPU1 慢 20~38%**：绝对性能只能在 GPU1 上测（`CUDA_VISIBLE_DEVICES=1`）；**峰值也必须同卡同频测**（111.6 vs 153.1 就是两张卡的数）
22. 对照引擎（`H:\infer-3090`，`:18090`）**有前缀缓存**：重复 prompt 会虚报到 3728~306910 t/s → 用 `reused prompt tokens == 0` 校验真数

### D. 测量方法论（最容易得出错误结论的地方）
23. **ncu 的 replay 会污染计时**（同一个 kernel 27.48 与 118 ms 的差异）→ ncu 只看计数器，**时间用干净跑测**
24. **微基准的死代码消除会给出假数**：`f16_x8acc` 报 1182 TFLOPS、S8HACK 报 396 TFLOPS，**都超过实测峰值 → 一律作废**。规则：**任何超过自测峰值的数字，先怀疑 DCE**（未初始化的累加器 / 不检查结果的 asm 块）
25. **峰值自己测，不查表**：同卡同频实测 f16/f16 = **153.1**、s8 = **304.6 TOPS**、bf16/f32 = 77、f16/f32 = 70
26. **两点拟合破除猜测**：用两个 prompt 长度拟合出"每 token 0.714 ms + 固定 0.09 s"，才排除掉"有隐藏固定开销"的可能
27. **nsys 要拿每次启动的 grid/block**：`nsys stats --report cuda_gpu_trace --format csv`，再按 grid 去重计数 —— 这才是发现"按 token 切 5 块"的关键
28. nsys/ncu 有时抓不到 kernel → 先用小 prompt + `--force-overwrite` 验证能抓，再上真 prompt
29. **别碰用户的常驻服务**：对照引擎（`:18090`）只可读、不可改；只杀自己起的进程；不动 NVIDIA 驱动（610.62）
## 6. 关键实测常量（后续优化直接引用，不必重测）

| 量 | 数值 | 备注 |
|---|---|---|
| SM 数 | **82 / 卡** | 不是 108；RTX 3090 |
| f16/f16 累加峰值 | **153.1 TFLOPS** | GPU1 @1920MHz（GPU0 @1395MHz 只有 111.6） |
| s8 (INT8) 峰值 | **304.6 TOPS = 2×** | 指令发射速率与 f16 相同，双倍 MAC |
| bf16/f32 累加 / f16/f32 累加 | 77 / 70 TFLOPS | 均约为 f16/f16 的一半 |
| 预填运算量 | **49 GFLOP / token** | 2 × 382.9M 参数/层 × 64 层 |
| f16 路径绝对上限 | **3.13k tok/s** | 49e9 ÷ 153.1e12 = 0.32 ms/token（零其它开销） |
| 当前引擎预填 | 1.34~1.36k tok/s | = 0.714 ms/token，固定开销仅 ~0.09 s |
| 当前解码 | 74.4 tok/s | GEMV 路径 + MTP |
| 引擎内 GEMM 效率 | 80.7 TFLOPS | = 峰值的 53%（微基准打包版 98.4 = 64%） |
| 内核寄存器 / 占用 | 249 regs → 2 block/SM = 8 warp (满配 48) | 张量管忙 53.2%、L1 64.3%、**DRAM 仅 6.4%** |
| 截肢结果（真实形状） | 基线 68.0 → 砍 B 91.0 → 砍 A 77.8 → 砍解码 71.9 → 砍 A+B **104.9** | 104.9 = 峰值 94% → **缺口 100% 在操作数搬运** |
| 微基准对拉（t=928, GPU1） | v8 4.845ms/68.3 → **v10 3.364ms/98.4** = 1.44× | 逐位一致 |

## 7. 后续研究方向（按性价比排序）

### 7.1 【首选】v11 = INT8 (s8) 内核 —— 方案见 `V11_S8_PLAN.md`
- 依据：INT8 = 2× f16（已实测）→ 2k 只需 s8 峰值的 22%，估算 **2.2~2.6k tok/s**
- 四步：①三元权重→s8（per-128 组 scale，**精确无损**），slab 从 2 字节/权重降到 1 字节 ②激活值在 v10 现成的打包 pass 里顺手输出 s8 + 每组 scale ③mma 换 `m16n8k32.s32.s8.s8.s32`（指令数减半）④保留"每 128 组提升 f32"结构、scale 相乘即可
- 验证：先在微基准对拉 98.4 TFLOPS；精度用固定 prompt 输出对比（**权重侧无损，风险只在激活值量化**）

### 7.2 拿回「引擎 vs 微基准」的 18% 效率差（80.7 → 98.4）
- 来源：5 块切分的重复读权重 + 每块一次打包 + 每个 kernel 的波尾
- 可试：把打包做成"一次服务 5 块"（按整段 prompt 打包，而不是每块打一次）；或把引擎切分与 CTA 窗口对齐

### 7.3 MTP/spec 机制吃掉 0.4 s（预填的 12%）
- 实测：去掉 `--spec mtp --draft-tokens 4 --lm-head-draft` → 3.4 s（1.37k），带上 → 3.8 s（1.35k）
- 待查：是 draft 头多算了 GEMM，还是 graph 里出现了空隙/同步

### 7.4 剩余 15% 非 GEMM 工作
- GDN（线性注意力）/ chunked conv1d / rmsnorm / rotate（2.1%，可考虑折进打包 pass，因为 v10 已经读过一遍激活值）

### 7.5 解码侧（74.4 t/s）
- profile 显示解码走 `gemv` + `mma_v2` + rotate；可试合并 rotate、或给解码专门的 s8 GEMV

### 7.6 其它可探索
- **t≈928 下重扫 `kV9_TILES`（4/6/8）**：v9 的 T 扫描是在 t=4636 做的，928 下的排序可能不同
- **多卡预填**：模型仅 7.45 GiB，单卡够；若按行切到两卡理论上近 1.8×，但需确认引擎是否支持张量并行（且 GPU0 接显示器会慢 20~38%）
- **PTQ1_0 打包**（母包有，1.75 bpw 比 PQ2_0 省 15% 显存，但解码更慢）：本机从未测过，需转换器支持 143 型
- **把 `kV9_LDMATRIX`/`kV9_BPIPE` 这类开关在 t=928 下重新扫一遍**（v9 的结论是在大 token 下得的）

## 8. 改动前后必做的检查清单（照做可避免 90% 的返工）

1. 改内核 → **先只在微基准验证**（`t=928` + 真实形状），并对照 `v4` 的 `rel_l2` 确认数值是否仍逐位一致
2. 任何吞吐数字 **不超过自测峰值** 才算可信（超了先查 DCE）
3. 接入引擎 → **同场次背靠背 A/B（各 3~4 次）**，并核对 `reused prompt tokens == 0`
4. 绝对性能**只在 GPU1** 上取；峰值也取 GPU1
5. 提交前 `git status`（`src/ninfer` 是 submodule，**主仓与 submodule 要分别 commit**）
6. 不动驱动、不动 `H:\infer-3090` 的常驻服务、只杀自己起的进程
