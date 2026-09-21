# ninfer-ternary（三元 Bonsai 推理加速工作区）

本仓是 [ninfer](https://github.com/Ambolio/ninfer-4090-windows)（Apache-2.0）在 **2-bit 三元权重**模型上的移植与内核优化工作区，
目标：把这个 27B 模型在 2×RTX 3090 上的**预填（prefill）**速度做到极限。

## 现状（可复现，实测）
| 指标 | 起点 | 现在 |
|---|---|---|
| 引擎预填 | 228.7 tok/s | **1.34~1.36k tok/s** |
| 引擎解码 | — | **74.4 tok/s** |
| 对照 dense 27B 基线 | 预填 646 / 解码 53~63 t/s | 预填 **2.1×**、解码 1.2~1.4× |

## 安装（交给 AI 照做）
```bash
git clone <本仓URL> && cd ninfer-ternary
git submodule update --init --recursive                 # 拉上游 ninfer 引擎
git -C src/ninfer apply ../../patches/ternary-v2-to-v10.patch   # 应用三元内核优化 v2→v10
# 构建见 HANDOVER_v2_to_v10.md §1.3；复现测试见 §2
```
补丁基线 = 上游 `6eb70a0`「v1.0.8」。上游若已前进，用 `git apply -3` 三方合并。

## 文档
- **[HANDOVER_v2_to_v10.md](HANDOVER_v2_to_v10.md)** —— 完整交接报告：环境安装、复现命令、v2→v10 每版改动、**29 条坑**、实测常量、后续研究方向。
- `src/ninfer/src/ops/linear/ternary/V11_S8_PLAN.md` —— 下一步（INT8/s8）实施方案（在补丁内）。

## 来源与致谢（尊重原创）
- **本仓内容** = 本工作区的移植与内核优化（v2→v10）+ 基准脚本 + 报告，Apache-2.0。
- **上游引擎**：[`Ambolio/ninfer-4090-windows`](https://github.com/Ambolio/ninfer-4090-windows)（Apache-2.0）。本仓以**子模块 + 补丁**方式引用，**未整包再分发**。
- **模型**：`Hikari07jp/Ternary-Bonsai-2-27B-Abliterated-GGUF` ← 母包 `prism-ml/Ternary-Bonsai-2-27B-gguf` ← 基座 `Qwen/Qwen3.8-27B`（均 Apache-2.0）。
- **三元内核参考实现**：`PrismML-Eng/llama.cpp` fork（原版 llama.cpp 跑不了 PQ2_0/PTQ1_0 文件）。
- **搭建指南**：`modelscope.cn/models/shensanshu/ninfer-ada-ternary`。
- 完整出处、引用、致谢见交接报告 §4。
