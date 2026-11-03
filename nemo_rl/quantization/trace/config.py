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
"""User configuration for TRACE (rollout-guided NVFP4 rounding, QAT training).

``generation.nvfp4_pertoken_rollout.trace`` is resolved once on the driver
into ``NRL_TRACE_*`` environment variables that both the vLLM workers (rollout
capture) and the Megatron workers (QAT) read, so the two sides always derive
the same payload layout (see :mod:`.layout`).
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel


class TraceConfig(BaseModel, extra="forbid"):
    # The routed experts train in BF16 on NVFP4 fake-quantized operands (every
    # NVFP4 MoE layer); the TE precision recipe must keep them BF16, e.g.
    # examples/te_precision/all_bf16_trace_qat.yaml.
    enabled: bool = False
    # Layers whose fc1/fc2 fake quantization follows the rollout's codes and
    # scales: "latter_half" (paper default) / "all" of the NVFP4 MoE layers,
    # or explicit global decoder-layer indices.
    layers: Union[Literal["latter_half", "all"], list[int]] = "latter_half"


def _join(layers) -> str:
    return ":".join(str(i) for i in sorted(set(layers)))


def layer_types(hf_config: Any) -> list[str]:
    """Per decoder layer: ``"moe"``, ``"dense"``, ``"attention"`` or ``"mamba"``.

    Hybrid models (Nemotron-H) list mixer types; for a plain transformer every
    layer is a routed-expert layer unless the config marks it dense (Qwen3-MoE
    ``mlp_only_layers`` / ``decoder_sparse_step``).
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
        return {}
    moe = [
        i
        for i, t in enumerate(layer_types(hf_config))
        if t == "moe" and i not in bf16_layers
    ]
    if trace.layers == "all":
        guided = moe
    elif trace.layers == "latter_half":
        guided = moe[len(moe) // 2 :]
    else:
        guided = [i for i in moe if i in set(trace.layers)]
    if not guided:
        raise ValueError(f"trace.layers={trace.layers} selects no NVFP4 MoE layer")
    return {
        "NRL_TRACE_LAYERS": _join(guided),
        "NRL_TRACE_QAT_LAYERS": _join(moe),
    }
