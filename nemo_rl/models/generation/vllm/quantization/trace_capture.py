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
"""Rollout-side TRACE capture (vLLM worker process).

Writes each selected layer's rollout FP4 record (see
:mod:`nemo_rl.quantization.trace.layout`) into the extra rows of vLLM's
``RoutedExpertsCapturer.device_buffer``. Everything downstream of that buffer
is vLLM's / NeMo-RL's router-replay path, unchanged.

Hooks (installed by :func:`apply_trace_capture_patches`):

* ``_get_routed_experts_shape``: widened by the layout's extra rows, in every
  process that sizes a routed-experts buffer (worker and scheduler).
* ``RoutedExpertsCapturer.__init__``: registers the capturer here and wraps
  ``MoERunner._forward_impl`` to remember which MoE layer is running. The
  in-kernel routed-experts capture fires only after the kernel, so it cannot
  tell the fc1/fc2 hooks inside the kernel call which layer they belong to.
* fc1: vLLM's ``quantize_nvfp4_per_token_input`` as bound in the TRT-LLM
  NVFP4 experts module (no vLLM source patch).
* fc2: ``TraceFc2CaptureMixin.apply`` on NeMo-RL's per-token experts class
  (the FlashInfer overlay writes the record from inside the MoE kernel).

All writes are fixed-shape device ops into a persistent buffer, so they are
recorded correctly when vLLM captures CUDA graphs.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from nemo_rl.quantization.trace import layout as trace_layout

_CAPTURER: Any = None
_LAYOUT: Optional[trace_layout.TraceLayout] = None
_NUM_LAYERS: int = 0
_CURRENT_MOE_LAYER: Optional[int] = None


def register_capturer(capturer: Any, vllm_config: Any) -> None:
    """Called from the patched ``RoutedExpertsCapturer.__init__``."""
    global _CAPTURER, _LAYOUT, _NUM_LAYERS
    if not trace_layout.trace_requested():
        return
    layout = trace_layout.layout_for_hf_config(vllm_config.model_config.hf_text_config)
    num_layers = int(vllm_config.model_config.get_total_num_hidden_layers())
    buf_layers = int(capturer.device_buffer.shape[1])
    if buf_layers != num_layers + layout.extra_rows:
        raise RuntimeError(
            f"TRACE: routed-experts buffer has {buf_layers} layer rows, expected "
            f"{num_layers} + {layout.extra_rows}; the shape patch did not apply."
        )
    if vllm_config.parallel_config.data_parallel_size != 1:
        raise RuntimeError("TRACE capture supports vLLM data_parallel_size=1 only")
    _CAPTURER, _LAYOUT, _NUM_LAYERS = capturer, layout, num_layers
    from nemo_rl.models.generation.vllm.quantization import trace_fi_fc2

    if not trace_fi_fc2._INSTALLED:
        raise RuntimeError("TRACE fc2 capture requires the FlashInfer overlay")
    if (
        capturer.device_buffer.dtype != torch.uint8
        or not capturer.device_buffer.is_contiguous()
    ):
        raise RuntimeError(
            "TRACE fc2 capture writes records into a contiguous uint8 transit buffer"
        )

    _install_layer_tracking()
    _install_fc1_capture()
    print(
        f"[trace] rollout capture active: layout={layout.to_json()} "
        f"extra_rows={layout.extra_rows}",
        flush=True,
    )


def _install_layer_tracking() -> None:
    """Record the running MoE layer around each runner call.

    ``_forward_impl`` runs eagerly inside the opaque ``moe_forward`` custom op,
    so under CUDA-graph capture this host code runs once per captured layer and
    the per-layer record offsets are baked into the graph.
    """
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    if getattr(MoERunner, "_nrl_trace_layer", False):
        return
    orig = MoERunner._forward_impl

    def _forward_impl(self, *args, **kwargs):
        global _CURRENT_MOE_LAYER
        _CURRENT_MOE_LAYER = int(self.layer_id)
        try:
            return orig(self, *args, **kwargs)
        finally:
            _CURRENT_MOE_LAYER = None

    MoERunner._forward_impl = _forward_impl
    MoERunner._nrl_trace_layer = True


def _install_fc1_capture() -> None:
    """Capture fc1 codes from the per-token input quantize of the TRT-LLM experts."""
    from vllm.model_executor.layers.fused_moe.experts import trtllm_nvfp4_moe

    orig = trtllm_nvfp4_moe.quantize_nvfp4_per_token_input
    if getattr(orig, "_nrl_trace", False):
        return

    def quantize_nvfp4_per_token_input(hidden_states: torch.Tensor):
        out = orig(hidden_states)
        capture_fc1(*out)
        return out

    quantize_nvfp4_per_token_input._nrl_trace = True  # type: ignore[attr-defined]
    trtllm_nvfp4_moe.quantize_nvfp4_per_token_input = quantize_nvfp4_per_token_input


def _write(rec: trace_layout.SiteRecord, payload: torch.Tensor) -> None:
    n = payload.shape[0]
    row0 = _NUM_LAYERS + rec.row_offset
    buf = _CAPTURER.device_buffer
    buf[:n, row0 : row0 + rec.rows, :] = payload.to(buf.dtype)


def capture_fc1(
    packed_codes: torch.Tensor, block_scale: torch.Tensor, token_scale: torch.Tensor
) -> None:
    if _LAYOUT is None or _CURRENT_MOE_LAYER is None:
        return
    rec = _LAYOUT.fc1_records.get(_CURRENT_MOE_LAYER)
    if rec is None:
        return
    h = _LAYOUT.hidden_size
    n = packed_codes.shape[0]
    payload = trace_layout.encode_fc1(
        _LAYOUT,
        rec,
        packed_codes.view(torch.uint8).reshape(n, -1)[:, : h // 2],
        block_scale.view(torch.uint8).reshape(n, -1)[:, : h // 16],
        token_scale.reshape(n),
    )
    _write(rec, payload)


class TraceFc2CaptureMixin:
    """Mixed into NeMo-RL's per-token TRT-LLM experts class (fc2 capture)."""

    def apply(self, hidden_states: torch.Tensor, *args, **kwargs):
        rec = None
        if _LAYOUT is not None and _CURRENT_MOE_LAYER is not None:
            rec = _LAYOUT.fc2_records.get(_CURRENT_MOE_LAYER)
        if rec is None:
            return super().apply(hidden_states, *args, **kwargs)  # type: ignore[misc]
        from nemo_rl.models.generation.vllm.quantization import trace_fi_fc2

        buf = _CAPTURER.device_buffer
        # The kernel writes [marker | slots] straight into this layer's record
        # (layout.encode_fc2 format); the pointer is baked per layer under CUDA graphs.
        trace_fi_fc2.set_fc2_out(
            buf,
            byte_offset=(_NUM_LAYERS + rec.row_offset) * _LAYOUT.topk,
            row_stride=buf.shape[1] * buf.shape[2],
            record_width=_LAYOUT.moe_intermediate_size,
        )
        try:
            out = super().apply(hidden_states, *args, **kwargs)  # type: ignore[misc]
        finally:
            trace_fi_fc2.set_fc2_out(None)
        return out


# -- source patches ---------------------------------------------------------

_SHAPE_MARKER = "NeMo-RL patch (TRACE routed-experts extra rows)"
_INIT_MARKER = "NeMo-RL patch (TRACE capturer registration)"


def _patch_routed_experts_capturer(logger) -> None:
    from nemo_rl.models.generation.vllm.patches import (
        _get_vllm_file,
        _locked_file_patch,
    )

    path = _get_vllm_file("model_executor/layers/fused_moe/routed_experts_capturer.py")
    shape_old = "    num_layers = model_config.get_total_num_hidden_layers()\n"
    shape_new = (
        shape_old + f"    # {_SHAPE_MARKER}\n"
        "    from nemo_rl.quantization.trace.layout import extra_rows_for_vllm_config\n"
        "    num_layers += extra_rows_for_vllm_config(vllm_config)\n"
    )
    init_old = (
        "        self.tp_size = vllm_config.parallel_config.tensor_parallel_size\n"
    )
    init_new = (
        init_old + f"        # {_INIT_MARKER}\n"
        "        from nemo_rl.models.generation.vllm.quantization import trace_capture\n"
        "        trace_capture.register_capturer(self, vllm_config)\n"
    )
    # Narrow the transit buffer to uint8 when TRACE rows ride on it (4x less
    # memory; routes for <=256 experts and payload bytes both fit).
    dtype_old = "            dtype=torch.int32,\n            device=current_platform.device_type,\n"
    dtype_new = (
        "            dtype=(torch.uint8 if extra_rows_for_vllm_config(vllm_config) else torch.int32),\n"
        "            device=current_platform.device_type,\n"
    )
    with _locked_file_patch(path) as (content, write_back):
        if _SHAPE_MARKER in content and _INIT_MARKER in content:
            return
        for old in (shape_old, init_old, dtype_old):
            if content.count(old) != 1:
                raise RuntimeError(
                    f"TRACE patch anchor not found exactly once in {path}: {old!r}"
                )
        content = (
            content.replace(shape_old, shape_new, 1)
            .replace(init_old, init_new, 1)
            .replace(dtype_old, dtype_new, 1)
        )
        # extra_rows_for_vllm_config is referenced inside __init__ for the dtype.
        content = content.replace(
            "class RoutedExpertsCapturer:\n",
            "from nemo_rl.quantization.trace.layout import extra_rows_for_vllm_config\n\n\n"
            "class RoutedExpertsCapturer:\n",
            1,
        )
        write_back(content)
    logger.info("Patched routed_experts_capturer.py for TRACE capture.")


def apply_trace_capture_patches(logger) -> None:
    """Install the source patches when ``NRL_TRACE_*`` requests capture."""
    if not trace_layout.trace_requested():
        return
    _patch_routed_experts_capturer(logger)
