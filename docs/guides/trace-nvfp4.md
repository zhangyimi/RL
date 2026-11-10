# TRACE: Rollout-Guided NVFP4 QAT

TRACE ([arXiv 2610.07767](https://arxiv.org/abs/2610.07767)) reduces the mismatch between the NVFP4 rollout and the training forward of MoE models.

## How It Works

- Training uses QAT: the routed experts run in BF16 on NVFP4 fake-quantized weights and activations, the same quantization the per-token NVFP4 rollout applies.
- During generation, vLLM records, for the inputs of the two routed-expert GEMMs (`fc1`, `fc2`), the lowest bit of every FP4 code plus the per-token and block scales. Neighboring FP4 values alternate this bit, so it tells which of the two nearest FP4 values the rollout rounded to.
- In the training forward, on the guided layers, each element is quantized with the rollout's scales and rounded to the neighbor with the rollout's bit instead of to nearest. Gradients pass straight through.
- Following the paper, guidance covers the latter half of the MoE layers by default; fake quantization covers all of them.

## Requirements

- Per-token NVFP4 rollout (`generation.nvfp4_pertoken_rollout.enabled: true`) with Megatron training.
- A TE precision recipe that keeps the routed experts in BF16 (`examples/te_precision/all_bf16_trace_qat.yaml`); the trainer checks this.
- Router replay (`policy.router_replay.enabled: true`).
- FlashInfer with the `fc2_input_out` argument of the TRT-LLM FP4 MoE (flashinfer-ai/flashinfer#9232). Until a release includes it, install a build of that PR into the vLLM worker venvs; TRACE refuses to start without it.

## Configuration

```yaml
policy:
  megatron_cfg:
    te_precision_config_file: examples/te_precision/all_bf16_trace_qat.yaml
  generation:
    nvfp4_pertoken_rollout:
      enabled: true
      trace:
        enabled: true
        layers: latter_half   # "all", or a list of decoder-layer indices
```

Ready-made recipes:

- `examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-megatron-te-nvfp4-pertoken-trace.yaml`
- `examples/configs/recipes/llm/grpo-nemotron35-lightning-30ba3b-8n4g-megatron-te-nvfp4-pertoken-trace.yaml`

The Qwen3 recipe has the settings of the runs below: every MoE layer in NVFP4, validation every 20 steps, synchronous vLLM scheduling, and `num_gpu_blocks_override: 16000` (only limits memory).

```sh
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-megatron-te-nvfp4-pertoken-trace.yaml
```

For the plain NVFP4 baseline, run the parent recipe with the same settings:

```sh
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-megatron-te-nvfp4-pertoken.yaml \
  grpo.max_num_steps=1500 grpo.val_period=20 \
  policy.megatron_cfg.first_last_layers_bf16=false \
  ++policy.generation.vllm_kwargs.async_scheduling=false \
  ++policy.generation.vllm_kwargs.num_gpu_blocks_override=16000
```

### Parallelism and async GRPO

Megatron TP/CP/EP and sequence packing need nothing extra. On the vLLM side, MoE layers must not be split by TP: use TP=1, or TP>1 with expert parallel of the same size. For example, for TP2/EP2 add:

```sh
  policy.generation.vllm_cfg.tensor_parallel_size=2 \
  policy.generation.vllm_cfg.expert_parallel_size=2
```

Async GRPO works with the async vLLM engine. KV-cache recompute after weight updates is required, and async GRPO itself does not support dynamic sampling or reward scaling/shaping:

```sh
  grpo.async_grpo.enabled=true \
  grpo.async_grpo.recompute_kv_cache_after_weight_updates=true \
  policy.generation.vllm_cfg.async_engine=true \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=4 \
  policy.generation.colocated.resources.gpus_per_node=4 \
  grpo.use_dynamic_sampling=false grpo.batch_multiplier=1.0 \
  grpo.reward_scaling.enabled=false grpo.reward_shaping.enabled=false
```

## Results

Qwen3-30B-A3B-Base, GRPO on DAPO-Math-17K, 8 nodes × 4 GB200, one run per row. The runs used the same method on an earlier NeMo-RL main (vLLM 0.29).

- **Mismatch KL**: KL between the rollout and training log-probs of the generated tokens (`train/gen_kl_error`), mean over the steps shown.
- **AIME24**: validation accuracy, mean of the evaluations around the step shown.

| Rollout / training | Mismatch KL, steps 980–1000 | Mismatch KL, steps 1300–1340 | AIME24, step 1000 | AIME24, step 1500 |
|---|---|---|---|---|
| BF16 / BF16 | 0.0006 | 0.0006 | 0.54 | 0.55 |
| NVFP4 / NVFP4 | 0.0123 | 0.0299 | 0.48 | collapsed at step ~1350 |
| NVFP4 / TRACE QAT | 0.0005 | 0.0005 | 0.51 | 0.58 |

Plain NVFP4 training drifts from the rollout as training goes on and collapses; with TRACE QAT the mismatch stays at the BF16 level for all 1500 steps.

## Limitations

- One bit cannot fix every element: when the training value is too far from the rollout value, neither neighbor matches. These cases grow with depth.
- The rollout-side hooks patch vLLM sources at fixed anchors. They are validated on the pinned vLLM nightly, and fail loudly on a version they do not match.
- SingleController and HTTP rollouts (`vllm_cfg.expose_http_server`, NeMo-Gym) are not supported yet.
