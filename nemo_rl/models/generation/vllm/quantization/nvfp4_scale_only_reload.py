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
"""Default-off layerwise scale reload for per-token NVFP4 rollouts."""

import re
from typing import Optional

import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken import (
    ModelOptNvFp4PerTokenFusedMoE,
    NvFp4PerTokenQuantizer,
    NvFp4PerTokenWorkerExtension,
)

_EXPERT_SCALE_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\."
    r"(?P<eid>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight_scale_2|weight_scale|input_scale)$"
)
_WEIGHT_SCALE_KINDS = ("weight_scale", "weight_scale_2")
_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class NvFp4ScaleOnlyCoalescer:
    """Coalesce expert scales while packed weights use the existing path."""

    def __init__(self, expected_experts: dict[str, int]) -> None:
        self._expected_experts = dict(expected_experts)
        self._pending: dict[str, dict[tuple[int, str, str], torch.Tensor]]
        self.coalesced_layers: int
        self.reset()

    def reset(self) -> None:
        self._pending = {}
        self.coalesced_layers = 0

    def process(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        out: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            match = _EXPERT_SCALE_RE.match(name)
            if match is None:
                out.append((name, tensor))
                continue

            layer_prefix = match.group("prefix")
            num_experts = self._expected_experts.get(layer_prefix)
            if num_experts is None:
                out.append((name, tensor))
                continue

            key = (
                int(match.group("eid")),
                match.group("proj"),
                match.group("kind"),
            )
            if key[2] == "input_scale":
                # Per-token activation scaling ignores checkpoint input scales.
                # Emit one pair of layerwise constants in _flush_layer instead.
                continue
            bucket = self._pending.setdefault(layer_prefix, {})
            if key in bucket:
                raise RuntimeError(
                    f"NVFP4 scale-only reload received duplicate tensor {name!r}"
                )
            bucket[key] = tensor

            expected_parts = (
                num_experts * len(_EXPERT_PROJECTIONS) * len(_WEIGHT_SCALE_KINDS)
            )
            if len(bucket) == expected_parts:
                out.extend(self._flush_layer(layer_prefix, num_experts, bucket))
                del self._pending[layer_prefix]
                self.coalesced_layers += 1
        return out

    @staticmethod
    def _flush_layer(
        layer_prefix: str,
        num_experts: int,
        bucket: dict[tuple[int, str, str], torch.Tensor],
    ) -> list[tuple[str, torch.Tensor]]:
        def part(expert_id: int, projection: str, kind: str) -> torch.Tensor:
            try:
                return bucket[(expert_id, projection, kind)]
            except KeyError as error:
                raise RuntimeError(
                    f"NVFP4 scale-only reload cannot flush {layer_prefix}; "
                    f"missing expert {expert_id} {projection}.{kind}"
                ) from error

        out: list[tuple[str, torch.Tensor]] = []
        for kind in _WEIGHT_SCALE_KINDS:

            def projection(projection_name: str) -> torch.Tensor:
                tensors = [
                    part(expert_id, projection_name, kind)
                    for expert_id in range(num_experts)
                ]
                if kind == "weight_scale":
                    return torch.stack(tensors)
                return torch.stack([tensor.reshape(()) for tensor in tensors])

            gate = projection("gate_proj")
            up = projection("up_proj")
            w2 = projection("down_proj")
            w13 = (
                torch.cat((gate, up), dim=1)
                if kind == "weight_scale"
                else torch.stack((gate, up), dim=1)
            )
            out.append((f"{layer_prefix}.gate_up_proj.{kind}", w13))
            out.append((f"{layer_prefix}.down_proj.{kind}", w2))

        device = part(0, "gate_proj", "weight_scale_2").device
        out.extend(
            (
                (
                    f"{layer_prefix}.gate_up_proj.input_scale",
                    torch.ones((num_experts, 2), device=device, dtype=torch.float32),
                ),
                (
                    f"{layer_prefix}.down_proj.input_scale",
                    torch.ones(num_experts, device=device, dtype=torch.float32),
                ),
            )
        )
        return out

    def finish(self) -> None:
        if self._pending:
            details = {
                prefix: len(parts) for prefix, parts in sorted(self._pending.items())
            }
            raise RuntimeError(
                f"NVFP4 scale-only reload ended with incomplete layers: {details}"
            )
        if self.coalesced_layers == 0:
            raise RuntimeError("NVFP4 scale-only reload emitted 0 complete layers")


class NvFp4ScaleOnlyReloadPreparer:
    """Compose existing quantization with scale-only coalescing."""

    def __init__(
        self,
        quantizer: NvFp4PerTokenQuantizer,
        expected_experts: dict[str, int],
    ) -> None:
        self.quantizer = quantizer
        self.coalescer = NvFp4ScaleOnlyCoalescer(expected_experts)

    def reset(self) -> None:
        self.quantizer.reset()
        self.coalescer.reset()

    def process(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        return self.coalescer.process(self.quantizer.process(weights))

    def finish(self) -> None:
        self.quantizer.finish()
        self.coalescer.finish()


def require_full_expert_scale_loader() -> None:
    """Fail before engine creation when the matching vLLM patch is absent."""
    if not getattr(RoutedExperts, "supports_full_expert_scale_loading", False):
        raise RuntimeError(
            "experimental_scale_only_reload requires the vLLM full-expert-scale "
            "loader patch"
        )


def _get_expected_experts(model: torch.nn.Module) -> dict[str, int]:
    require_full_expert_scale_loader()
    expected_experts = {
        module.layer_name: module.global_num_experts
        for module in model.modules()
        if isinstance(module, RoutedExperts)
        and isinstance(module.quant_method, ModelOptNvFp4PerTokenFusedMoE)
    }
    if not expected_experts:
        raise RuntimeError(
            "NVFP4 scale-only reload found no per-token NVFP4 RoutedExperts layers"
        )
    return expected_experts


class NvFp4PerTokenScaleOnlyWorkerExtension(NvFp4PerTokenWorkerExtension):
    """Default-off bulk scale reload using vLLM's native expert semantics."""

    _scale_only_preparer: Optional[NvFp4ScaleOnlyReloadPreparer] = None

    def _get_quantizer(self) -> NvFp4PerTokenQuantizer:
        if self._quantizer is None:
            self._quantizer = NvFp4PerTokenQuantizer(
                self.model_runner.model, emit_input_scales=False
            )
        return self._quantizer

    def _get_reload_weight_preparer(self) -> NvFp4ScaleOnlyReloadPreparer:
        if self._scale_only_preparer is None:
            self._scale_only_preparer = NvFp4ScaleOnlyReloadPreparer(
                self._get_quantizer(),
                _get_expected_experts(self.model_runner.model),
            )
        return self._scale_only_preparer
