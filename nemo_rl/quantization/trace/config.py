# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""User configuration for TRACE (rollout-guided NVFP4 rounding).

``generation.nvfp4_pertoken_rollout.trace`` is resolved once on the driver
into ``NRL_TRACE_*`` environment variables that both the vLLM workers (rollout
capture) and the Megatron workers (training-side guidance) read, so the two
sides always derive the same payload layout (see :mod:`.layout`).
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

Site = Literal["fc1", "fc2", "kv"]


class TraceKVConfig(BaseModel, extra="forbid"):
    """NVFP4 KV-cache settings (only used when ``"kv"`` is in ``sites``)."""

    # ``trace``: guided rounding of K/V in the training forward; ``rtn``:
    # plain round-to-nearest NVFP4 KV fake quantization (QAT baseline).
    mode: Literal["trace", "rtn"] = "trace"
    # ``{layer: {"q", "k", "v": amax}}`` JSON from a calibration pass
    # (``calib_dir``). vLLM's default KV scale of 1.0 underflows V block scales.
    scales_file: Optional[str] = None
    scale_headroom: float = 1.25
    # Calibration pass: record per-layer q/k/v amax in the training logprob
    # forward (BF16 KV rollout) into ``<calib_dir>/rank<r>.json``; merge with
    # ``python -m nemo_rl.quantization.trace.kv_calib <calib_dir> <out.json>``.
    calib_dir: Optional[str] = None


class TraceConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    # Quantization sites: routed-expert fc1 input, fc2 input, NVFP4 KV cache.
    sites: list[Site] = Field(default_factory=lambda: ["fc1", "fc2"])
    # Rollout code bits shipped per element: 1 = mantissa bit (paper, b=1),
    # 2 = two low code bits (experimental).
    bits: Literal[1, 2] = 1
    # Global decoder-layer indices; "all" = every NVFP4 routed-expert layer
    # (fc1/fc2) and every attention layer (kv).
    layers: Union[Literal["all"], list[int]] = "all"
    kv: TraceKVConfig = Field(default_factory=TraceKVConfig)
    # Ship the rollout's per-token and block scales with fc1 records. The
    # training forward always quantizes with its own scales (TE computes them);
    # shipped scales are only read by the alignment probe.
    ship_scales: bool = False
    # Rollout/train FP4 alignment probe: JSON lines under ``probe_dir``;
    # ``probe_layers`` additionally ship full 4-bit codes for exact statistics.
    probe_dir: Optional[str] = None
    probe_layers: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "TraceConfig":
        if self.enabled and not self.sites:
            raise ValueError("trace.sites must not be empty when trace is enabled")
        if self.enabled and "kv" in self.sites and self.kv.scales_file is None:
            raise ValueError(
                "trace kv site needs trace.kv.scales_file (calibrated NVFP4 KV "
                "scales); see trace.kv.calib_dir"
            )
        return self


def _join(layers) -> str:
    return ":".join(str(i) for i in sorted(set(layers)))


def layer_types(hf_config: Any) -> list[str]:
    """Per decoder layer: ``"moe"``, ``"dense"``, ``"attention"`` or ``"mamba"``.

    Hybrid models (Nemotron-H) list mixer types; for a plain transformer every
    layer has attention, and it is a routed-expert layer unless the config
    marks it dense (Qwen3-MoE ``mlp_only_layers`` / ``decoder_sparse_step``).
    """
    cfg = getattr(hf_config, "text_config", None) or hf_config
    block_types = getattr(cfg, "layers_block_type", None)
    if block_types:
        return [str(t) for t in block_types]
    n = int(cfg.num_hidden_layers)
    dense = set(getattr(cfg, "mlp_only_layers", None) or [])
    step = int(getattr(cfg, "decoder_sparse_step", 1) or 1)
    return ["dense" if (i in dense or (i + 1) % step) else "moe" for i in range(n)]


def resolve_trace_env(
    trace: TraceConfig, hf_config: Any, bf16_layers: set[int]
) -> dict[str, str]:
    """Resolve ``trace`` into the ``NRL_TRACE_*`` env read by both workers."""
    if not trace.enabled:
        # A KV calibration pass runs without TRACE (BF16 KV rollout).
        return (
            {"NRL_TRACE_KV_CALIB_DIR": trace.kv.calib_dir} if trace.kv.calib_dir else {}
        )
    types = layer_types(hf_config)
    hybrid = any(t in ("attention", "mamba") for t in types)
    moe = [i for i, t in enumerate(types) if t == "moe" and i not in bf16_layers]
    attn = [i for i, t in enumerate(types) if t == "attention" or not hybrid]
    if trace.layers != "all":
        sel = set(trace.layers)
        moe = [i for i in moe if i in sel]
        attn = [i for i in attn if i in sel]
    mode = {1: "lsb", 2: "b2"}[trace.bits]
    env = {
        "NRL_TRACE_MODE": mode,
        "NRL_TRACE_SHIP_SCALES": "1" if trace.ship_scales else "0",
    }
    apply: list[str] = []
    if "fc1" in trace.sites:
        env["NRL_TRACE_FC1_LAYERS"] = _join(moe)
        apply.append("fc1")
    if "fc2" in trace.sites:
        env["NRL_TRACE_FC2_LAYERS"] = _join(moe)
        env["NRL_TRACE_FC2_BITS"] = str(trace.bits)
        apply.append("fc2")
    if "kv" in trace.sites:
        env["NRL_TRACE_KV_LAYERS"] = _join(attn)
        env["NRL_TRACE_KV_BITS"] = str(trace.bits)
        env["NRL_TRACE_KV_SCALES_FILE"] = str(trace.kv.scales_file)
        env["NRL_TRACE_KV_SCALE_HEADROOM"] = str(trace.kv.scale_headroom)
        # trtllm-gen writes the NVFP4-KV attention output through FP8; scale it.
        env["NRL_TRACE_KV_OSCALE"] = "1"
        # Training mirrors the kernel's FP8 dequant of NVFP4 K/V.
        apply += ["kv" if trace.kv.mode == "trace" else "kv_rtn", "kv8"]
    if trace.kv.calib_dir:
        env["NRL_TRACE_KV_CALIB_DIR"] = trace.kv.calib_dir
    env["NRL_TRACE_APPLY"] = ",".join(apply)
    if trace.probe_dir:
        env["NRL_TRACE_PROBE_DIR"] = trace.probe_dir
        if trace.probe_layers:
            env["NRL_TRACE_FULL_LAYERS"] = _join(trace.probe_layers)
    return env
