# NVFP4 Scale-Only Reload 设计与验证

日期：2026-08-25

## Reviewer Summary

这是一个 **default-off** 的 NVFP4 refit 优化：只把 MoE scale 合成 layerwise
tensor，packed NVFP4 weight 仍走现有逐 expert loader。vLLM patch 在目标 shape
一致时直接复制整层 scale；shape 不一致时回落到原有 per-expert loader，因此 TP
切分、EP expert mapping、padding 和量化 scale 语义仍由 vLLM 维护。

核心 review 判断是：它没有 Shuang 的 fully stacked reload 快，但拿到了约 9.6 秒
收益，同时没有替换 vLLM reload 生命周期，也没有给当前 per-token NVFP4 功能增加
新的 TP、EP 或 PP 限制。

## 结论

这个方案值得继续推进。它只合并 NVFP4 MoE 的 scale loader records，保留 packed
weight 的现有逐 expert 路径；在 4 节点 × 4 GB200 的同作业 A/B 中，NeMo 看到的完整
`transfer_and_update_weights` 从平均 **19.48 s** 降到 **9.92 s**，每次节省
**9.57 s（49.1%）**。

它没有 Shuang 的全量 stacked reload 快（后者稳定约 7.2--7.5 s），但已经拿到大部分
收益，并且耦合面明显更小：vLLM 只改一个生产文件，新增 96 行；不替换 vLLM 的
whole-model/layerwise reload 生命周期，也不自己重写 TP/EP/padding 规则。

## 正式 GPU A/B

- Slurm job：`2650701`，`COMPLETED (0:0)`，总耗时 15:43
- 资源：4 nodes × 4 GB200，节点 `ptyche0099--0102`
- 模型：Qwen3-30B-A3B
- 两个 variant 在同一个 allocation 中顺序执行，每个 3 个 GRPO steps
- Megatron train topology：EP=16、TP=1、PP=1
- vLLM rollout topology：EP=1、TP=1、PP=1
- baseline 和 scale-only 共用同一 NeMo checkout、同一 patched vLLM checkout、recipe、
  checkpoint 和软件环境；唯一功能差异是
  `experimental_scale_only_reload=False/True`

### NeMo 完整 refit wall time

`prepare_for_generation/transfer_and_update_weights` 覆盖 trainer export/transfer、vLLM
reload，以及并行 worker 汇合，是训练 step 真正等待的时间。

| variant | step 1 | step 2 | step 3 | mean |
|---|---:|---:|---:|---:|
| baseline | 19.26 s | 19.61 s | 19.58 s | 19.48 s |
| scale-only | 10.14 s | 9.71 s | 9.90 s | 9.92 s |
| saving | 9.12 s | 9.90 s | 9.68 s | 9.57 s (49.1%) |

`prepare_for_generation/total` 的平均值也从 21.77 s 降到 12.22 s，节省 9.56 s
（43.9%）。

### vLLM reload 本体

Ray 会把 16 个相似 worker 日志去重，因此下面列的是每 step 的显式 worker 值和
dedup 代表值，不应解释成严格的全 worker min/max：

| variant | step 1 | step 2 | step 3 |
|---|---:|---:|---:|
| baseline | 17.70 / 18.89 s | 17.95 / 19.24 s | 17.88 / 19.19 s |
| scale-only | 8.59 / 9.10 s | 8.27 / 8.84 s | 8.27 / 8.72 s |

step 1 的输入/生成统计完全相同（reward 0.5、mean generation length 3154.75、loss
0.0001、generation KL error 0.0113），总 step time 从 64.93 s 降到 54.08 s。
后续 rollout 本身是随机采样，因此不使用 step 2/3 的 reward 或总 step time 做数值
等价判断。

### 与 Shuang stacked reload 的位置

Shuang 的正式长跑 job `2643908` 是 8 nodes × 4 GPUs，稳定区间里 vLLM reload
约 6.4--6.8 s，NeMo 完整 transfer/update 约 7.2--7.5 s。它把 packed weights 和
scales 都堆成 layerwise tensors，所以比本 POC 的 9.92 s 还快约 2.5 s；但拓扑、
batch 和 recipe 不同，这里只用于确定量级，不作为严格同配方 A/B。

## 为什么 scale-only 有效

当前 42 个量化 MoE layers、128 experts、3 projections 的 baseline 每次向 vLLM loader
发送：

- packed weights：`42 * 128 * 3 = 16,128` records
- 两类 weight scales + input scale：`42 * 128 * 3 * 3 = 48,384` records
- 合计：64,512 records

scale-only 保留 16,128 个 packed-weight records；每层只发送 4 个 weight-scale tensor
和 2 个 per-token activation 常量，共 `42 * 6 = 252` 个 scale records。总数变为
16,380，减少 48,132（74.61%）。这正好切掉 phase-0 profiling 指出的 Python
loader/name bookkeeping 瓶颈，而不碰量化 kernel 和 packed-weight 布局。

per-token activation scale 不是 checkpoint state：kernel 在运行时按 token 动态求值。
因此 reload 时只需把 vLLM 对应的 input-scale slots 设成 layerwise ones，不需要像
weight block scales 一样逐 expert 搬运。

## 改动和维护面

### NeMo-RL

- 新的 default-off config：`experimental_scale_only_reload`
- 新模块 `nvfp4_scale_only_reload.py`：224 行
- 现有生产文件：+39/-10 行
- NeMo 生产改动合计：+263/-10 行

### vLLM

- 仅 `vllm/model_executor/layers/fused_moe/routed_experts.py`：+96 行
- 对应测试：+112 行
- patch 增加六类 full-expert scale 输入；目标 shape 完全一致时一次 `copy_`
- shape 不一致时递归调用原有 per-expert `weight_loader`，因此 TP slicing、EP global-to-local
  expert mapping、padding 和 quant scale 语义仍由 vLLM 原实现决定

NeMo + vLLM 生产代码合计 +359/-10；测试 +285/-2。

作为量级对比，Shuang 的 stacked commit `f1c750a66` 相对其 parent 的生产代码约
+754/-49 行，测试 +560/-32 行，另外还有 437 行 recipes/launch harness。两者基线不同，
不能把行数当成严格复杂度指标，但 vLLM 内部耦合面和自定义生命周期的差异是明确的。

## TP / EP / PP 支持边界

scale-only 没有新增任何 topology validator 或限制；它遵循当前 per-token NVFP4 功能的
既有支持范围：

- Megatron training EP：本次正式 3-step A/B 已实测 EP=16，成功。
- vLLM TP：没有被 scale-only 禁止。TP=2 的 scale slicing 走原始 vLLM loader，单测通过；
  但本次没有做 TP=2 端到端性能测试。
- vLLM PP：保留既有规则，PP>1 需要 async engine；scale-only 没有附加限制。本次端到端
  是 PP=1。
- vLLM EP：当前父功能本身就显式要求 EP=1，所以不能声称现在端到端支持 EP>1。
  新 full-scale loader 的 global-to-local expert mapping 已做单测，因此将来解除父功能限制时，
  scale-only 不需要另写一套 EP mapping。

所以“当前方案该支持的都支持”是成立的；需要避免把 **train EP=16** 和 **rollout vLLM
EP=1** 混成同一个限制。

## 验证记录

- NeMo NVFP4 targeted unit tests：24 passed
- vLLM padded MoE/full-scale tests：33 passed
- vLLM tests 覆盖 direct-copy、TP=2 slicing、synthetic EP expert mapping、错误 shape
- Ruff、compileall、`git diff --check`：通过
- 4-node GPU smoke job `2650682`：通过，确认 GB200、CUDA、patched vLLM source/marker
- 4-node formal A/B job `2650701`：通过，baseline 3 steps + scale-only 3 steps
- 完整 pyrefly 未完成；首次发现的两个 implicit attributes 已修复，但第二次全仓检查挂住后
  被中止，因此这里不宣称 full pyrefly pass

## 建议

可以先以 default-off experimental feature 合入 NeMo-RL，同时携带一个很小的 vLLM
patch。merge gate 建议至少包括现有 unit tests、无 patch 时的 early failure，以及一条
当前 4n4g recipe 的 smoke。后续如果 vLLM 接受通用 full-expert-scale loader，NeMo 侧
不再需要长期维护下游 patch。

## 使用方法

先在 vLLM `v0.26.0`（commit
`568afb3a13806beb53bb2e6bd518269357b237c0`）应用本分支携带的 patch：

```bash
git -C /path/to/vllm checkout v0.26.0
git -C /path/to/vllm apply --check \
  /path/to/RL/patches/vllm/v0.26.0-full-expert-scale-loader.patch
git -C /path/to/vllm apply \
  /path/to/RL/patches/vllm/v0.26.0-full-expert-scale-loader.patch
```

然后显式打开实验开关：

```yaml
policy:
  generation:
    nvfp4_pertoken_rollout:
      enabled: true
      experimental_scale_only_reload: true
```

默认值是 `false`。不开启时，现有 per-expert 路径完全不依赖 vLLM patch；开启时，
NeMo RL 会在 engine 创建前检查 `supports_full_expert_scale_loading` capability
marker，缺少 patch 会直接报错。关闭开关就是回滚路径。

patch 文件是
[`patches/vllm/v0.26.0-full-expert-scale-loader.patch`](../../patches/vllm/v0.26.0-full-expert-scale-loader.patch)，
包含 vLLM 生产代码和单测。它是 build-time source patch，不是 runtime monkey
patch。长期目标仍然是把通用 full-expert-scale loader 合入 vLLM upstream。
