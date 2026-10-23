# TRACE: Rollout-Guided NVFP4 Rounding

TRACE ([arXiv 2610.07767](https://arxiv.org/abs/2610.07767)) reduces the mismatch between the NVFP4 rollout and the NVFP4 training forward of MoE models. It builds on the per-token NVFP4 path described in [End-to-End NVFP4 Training](../design-docs/te-nvfp4-per-token-rollout.md).

## How It Works

- During generation, vLLM records the lowest bit of every FP4 activation code at the selected quantization points. Neighboring FP4 values alternate this bit, so it tells which of the two nearest FP4 values the rollout rounded to.
- In the training forward, Megatron rounds the same element to the neighbor with that bit instead of rounding to nearest. Gradients pass straight through.
- The quantization points are the inputs of the two routed-expert GEMMs (`fc1`, `fc2`) and, optionally, the NVFP4 KV cache (`kv`).
- The record travels with the router-replay routes, so it follows the same padding, packing and parallel sharding.

**Difference from the paper:** the paper also stores the rollout's quantization scales, and uses them both to pick the code and to dequantize it. Here only the bits are sent. Training quantizes with Transformer Engine, which computes its own scales, so only the rounding direction is matched.

## Requirements

- Per-token NVFP4 rollout (`generation.nvfp4_pertoken_rollout.enabled: true`) with Megatron TE NVFP4 training.
- Router replay (`policy.router_replay.enabled: true`).
- vLLM tensor parallel size 1, synchronous vLLM engine (`vllm_cfg.async_engine: false`), GRPO.
- On the first run, FlashInfer JIT-compiles a patched TRT-LLM MoE module that exposes the `fc2` codes, which takes about 15 minutes. Point `FLASHINFER_WORKSPACE_BASE` (in `vllm_cfg.env_vars`) at a shared directory to reuse the build.

## Configuration

```yaml
policy:
  generation:
    nvfp4_pertoken_rollout:
      enabled: true
      trace:
        enabled: true
        sites: [fc1, fc2]   # add kv to also guide the NVFP4 KV cache
        bits: 1             # 2 ships two low code bits (experimental)
        layers: all         # or a list of decoder-layer indices
```

`layers: all` selects every NVFP4 routed-expert layer (and every attention layer for `kv`), read from the Hugging Face config. Layers kept in BF16 by `first_last_layers_bf16` are skipped.

Ready-made recipes:

- `examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-megatron-te-nvfp4-pertoken-trace.yaml`
- `examples/configs/recipes/llm/grpo-nemotron35-lightning-30ba3b-8n4g-megatron-te-nvfp4-pertoken-trace.yaml`

The Qwen3 recipe matches the 400-step runs below: every MoE layer in NVFP4, validation every 20 steps, and `num_gpu_blocks_override: 16000`. The block cap bounds the host memory that vLLM uses to keep routed-expert records for cached tokens, and it does not change numerics. For the plain NVFP4 baseline, run the parent recipe with the same overrides:

```sh
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-megatron-te-nvfp4-pertoken.yaml \
  grpo.max_num_steps=400 grpo.val_period=20 \
  policy.megatron_cfg.first_last_layers_bf16=false \
  ++policy.generation.vllm_kwargs.async_scheduling=false \
  ++policy.generation.vllm_kwargs.num_gpu_blocks_override=16000
```

## Results

Qwen3-30B-A3B-Base, GRPO on DAPO-Math-17K, 8 nodes × 4 GB200, 400 steps, one run per row. The mismatch KL is `train/gen_kl_error`, the median over steps 380–400.

| Rollout / training precision | Mismatch KL | AIME24 @ 200 | AIME24 @ 400 |
|---|---|---|---|
| BF16 | 0.0004 | 0.348 | 0.426 |
| NVFP4 W4A4 | 0.0058 | 0.336 | 0.398 |
| NVFP4 W4A4 + TRACE 1-bit | 0.0010 | 0.164 | 0.332 |

TRACE brings the mismatch close to BF16. In these runs, validation accuracy did not improve: it trailed plain NVFP4 for most of the run.

## NVFP4 KV Cache

Adding `kv` to `sites` switches the rollout to an NVFP4 KV cache (`kv_cache_dtype: nvfp4`). Training then applies the same NVFP4 quantization to K and V before attention.

- **Calibrated scales are required.** With vLLM's default scale of 1.0, small V values underflow and the rollout degrades badly. To calibrate, run a short BF16-KV job with `trace.kv.calib_dir: <dir>`, merge the per-rank maxima, and point `trace.kv.scales_file` at the result:

  ```sh
  python -m nemo_rl.quantization.trace.kv_calib <dir> kv_amax.json
  ```

  Scales for Qwen3-30B-A3B-Base are in `examples/trace/kv_amax_qwen3_30b_a3b_base.json`.
- The trtllm-gen attention kernel dequantizes the NVFP4 cache to FP8 and quantizes Q to FP8. Training mirrors the FP8 dequantization, but these FP8 steps leave an error floor that TRACE does not remove.
- `trace.kv.mode: rtn` applies plain round-to-nearest NVFP4 to K/V in training instead (a quantization-aware-training baseline).
- KV support is validated on Qwen3-30B-A3B only.

## Limitations

- Some elements still disagree after guidance, and the share grows with depth as the two forwards drift apart (Qwen3-30B-A3B, `fc1`: 0.02% at layer 4, about 4% at layer 40; much more with the NVFP4 KV cache). Almost all of them are two codes apart: the rollout value lies outside the two neighbors of the training value, so one bit cannot recover it.
- The rollout-side hooks patch vLLM and FlashInfer sources at fixed anchors. They are validated on the pinned vLLM nightly and FlashInfer 0.7, and fail loudly on a version they do not match.
- The async vLLM engine and SingleController are not supported yet.
