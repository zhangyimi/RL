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
* fc2: ``TraceFc1CaptureMixin.apply`` on NeMo-RL's per-token experts class.
* kv: ``FlashInferImpl.do_kv_cache_update`` reads the just-written NVFP4 cache
  slots back (the cache itself is the ground truth for what attention reads).

All writes are fixed-shape device ops into a persistent buffer, so they are
recorded correctly when vLLM captures CUDA graphs.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import torch

from nemo_rl.quantization.trace import layout as trace_layout

_CAPTURER: Any = None
_LAYOUT: Optional[trace_layout.TraceLayout] = None
_NUM_LAYERS: int = 0
_CURRENT_MOE_LAYER: Optional[int] = None
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


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
    if layout.dbg_records:
        _install_dbg_capture()
    if layout.fc2_records:
        from nemo_rl.models.generation.vllm.quantization import trace_fi_fc2

        if not trace_fi_fc2._INSTALLED:
            raise RuntimeError("TRACE fc2 capture requires the FlashInfer overlay")
        if capturer.device_buffer.dtype != torch.uint8 or not capturer.device_buffer.is_contiguous():
            raise RuntimeError("TRACE fc2 capture writes records into a contiguous uint8 transit buffer")

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


def _fast_capture() -> bool:
    """Fused Triton capture unless ``NRL_TRACE_TORCH_CAPTURE=1`` (reference path)."""
    return os.environ.get("NRL_TRACE_TORCH_CAPTURE") != "1"


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
    buf = _CAPTURER.device_buffer
    if _fast_capture() and not _LAYOUT.ship_scales and buf.is_cuda and buf.dtype == torch.uint8:
        # One kernel per layer instead of ~10 small encode ops per decode step.
        from nemo_rl.quantization.trace.encode_triton import encode_fc1_into

        encode_fc1_into(
            buf, (_NUM_LAYERS + rec.row_offset) * _LAYOUT.topk,
            packed_codes.view(torch.uint8).reshape(n, -1), rec.mode, h,
        )
        return
    payload = trace_layout.encode_fc1(
        _LAYOUT,
        rec,
        packed_codes.view(torch.uint8).reshape(n, -1)[:, : h // 2],
        block_scale.view(torch.uint8).reshape(n, -1)[:, : h // 16],
        token_scale.reshape(n),
    )
    _write(rec, payload)


def _install_dbg_capture() -> None:
    """Capture raw BF16 MoE input/output and router logits on debug layers.

    ``MoERunner._forward_impl`` runs eagerly inside the opaque ``moe_forward``
    custom op for both BF16 and NVFP4 experts, so the same hook covers both.
    """
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    if getattr(MoERunner, "_nrl_trace_dbg", False):
        return
    orig = MoERunner._forward_impl

    def _forward_impl(self, hidden_states, router_logits, *args, **kwargs):
        rec = _LAYOUT.dbg_records.get(int(getattr(self, "layer_id", -1))) if _LAYOUT else None
        x = hidden_states.clone() if rec is not None else None
        out = orig(self, hidden_states, router_logits, *args, **kwargs)
        if rec is not None:
            y = out[1] if isinstance(out, tuple) else out
            n = x.shape[0]
            _write(rec, trace_layout.encode_dbg(
                _LAYOUT, rec, x.reshape(n, -1), y.reshape(n, -1), router_logits.reshape(n, -1)
            ))
        return out

    MoERunner._forward_impl = _forward_impl
    MoERunner._nrl_trace_dbg = True


class TraceFc1CaptureMixin:
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
            bits=_LAYOUT.fc2_bits,
            record_width=_LAYOUT.moe_intermediate_size,
        )
        try:
            out = super().apply(hidden_states, *args, **kwargs)  # type: ignore[misc]
        finally:
            trace_fi_fc2.set_fc2_out(None)
        return out


def capture_kv(impl: Any, layer: Any, kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    """Read back the NVFP4 K/V records just written for this step's tokens."""
    if _LAYOUT is None or not getattr(impl, "is_kvcache_nvfp4", False):
        return
    m = _LAYER_RE.search(getattr(layer, "layer_name", "") or "")
    if m is None:
        return
    rec = _LAYOUT.kv_records.get(int(m.group(1)))
    if rec is None:
        return
    hk = impl.num_kv_heads
    if hk != _LAYOUT.num_kv_heads:
        raise RuntimeError(f"TRACE kv: layout has {_LAYOUT.num_kv_heads} KV heads, impl {hk}")
    buf = _CAPTURER.device_buffer
    if _fast_capture() and buf.is_cuda and buf.dtype == torch.uint8 and buf.is_contiguous():
        # One kernel per layer: gather codes/scales from the paged cache and
        # write the encode_kv record in place.
        from nemo_rl.quantization.trace.encode_triton import capture_kv_into

        capture_kv_into(
            buf, (_NUM_LAYERS + rec.row_offset) * _LAYOUT.topk, kv_cache, slot_mapping,
            hk, _LAYOUT.head_dim, rec.mode,
        )
        return
    from vllm.utils.torch_utils import nvfp4_split_data_scale

    n = slot_mapping.shape[0]
    valid = slot_mapping >= 0
    slots = slot_mapping.clamp(min=0)
    block_size = kv_cache.shape[2]
    blk, off = slots // block_size, slots % block_size
    sides = []
    # kv_cache: (num_blocks, 2 * Hkv, block_size, full_dim), K heads first.
    for is_v, side in ((False, kv_cache[:, :hk]), (True, kv_cache[:, hk:])):
        data, sf = nvfp4_split_data_scale(side)  # (B, Hkv, N, D/2), (B, Hkv, N, D/16)
        sf = sf.view(torch.uint8)
        if not is_v:
            sides += [data[blk, :, off], sf[blk, :, off]]
            continue
        # V scales use TRT-LLM's 4-token interleave inside each (page, head)
        # [N, S] tile: flat[(t//4)*4*S + s*4 + t%4] = scale[t, s]
        # (FlashInfer nvfp4_quantize_paged_kv_cache). Un-swizzle per token.
        s_dim = sf.shape[-1]
        s_idx = torch.arange(s_dim, device=sf.device)
        flat = (off // 4).unsqueeze(1) * (4 * s_dim) + s_idx * 4 + (off % 4).unsqueeze(1)
        sf_t = sf.permute(0, 2, 3, 1)  # (B, N, S, Hkv)
        v_sf = sf_t[blk.unsqueeze(1), flat // s_dim, flat % s_dim]  # (n, S, Hkv)
        sides += [data[blk, :, off], v_sf.permute(0, 2, 1)]
    payload = trace_layout.encode_kv(
        _LAYOUT, rec, sides[0], sides[1], sides[2], sides[3], valid
    )
    _write(rec, payload[:n])


# -- calibrated NVFP4 KV scales --------------------------------------------

KV_SCALES_FILE_ENV = "NRL_TRACE_KV_SCALES_FILE"


def kv_scales_from_file(path: str, headroom: float = 1.0) -> dict[int, dict[str, float]]:
    """Per-layer dequant scales from a ``{layer: {q,k,v: amax}}`` JSON.

    NVFP4 K/V: ``amax / 448``, i.e. block scales stay in ``[0, 448 / 6]``.
    The trtllm-gen attention kernel dequantizes NVFP4 K/V to FP8 E4M3
    internally, so ``code * block_scale`` (up to ``6 * block_scale``) must fit
    in 448 (FlashInfer ``nvfp4_quantize_paged_kv_cache`` uses the same
    ``global_sf = 448 / amax``). ``amax / (6 * 448)`` overflows that FP8
    intermediate and breaks the rollout.
    FP8 query (vLLM quantizes Q to e4m3 with NVFP4 KV): ``amax / 448``.
    """
    import json

    with open(path) as f:
        amax = json.load(f)
    out = {}
    for layer, d in amax.items():
        out[int(layer)] = {
            "k": d["k"] * headroom / 448.0,
            "v": d["v"] * headroom / 448.0,
            "q": d["q"] * headroom / 448.0,
        }
    return out


def on_bind(model: Any) -> None:
    """Called before CUDA-graph capture: install calibrated KV scales.

    vLLM leaves NVFP4 KV scales at 1.0 without a calibrated checkpoint, which
    pushes small V blocks into E4M3 subnormals/zero. FlashInfer caches
    ``bmm1/bmm2_scale`` on first forward, so reset them as well.
    """
    path = os.environ.get(KV_SCALES_FILE_ENV)
    if not path:
        return
    headroom = float(os.environ.get("NRL_TRACE_KV_SCALE_HEADROOM", "1.0"))
    scales = kv_scales_from_file(path, headroom)
    n = 0
    for module in model.modules():
        if not hasattr(module, "_k_scale") or not hasattr(module, "layer_name"):
            continue
        m = _LAYER_RE.search(module.layer_name or "")
        if m is None or int(m.group(1)) not in scales:
            continue
        sc = scales[int(m.group(1))]
        which = os.environ.get("NRL_TRACE_KV_SET", "kvq")
        sc = {k: (v if k in which else 1.0) for k, v in sc.items()}
        for name in ("k", "v", "q"):
            getattr(module, f"_{name}_scale").fill_(sc[name])
            setattr(module, f"_{name}_scale_float", sc[name])
        module._k_scale_cpu = torch.tensor(sc["k"], dtype=torch.float32)
        module._v_scale_cpu = torch.tensor(sc["v"], dtype=torch.float32)
        impl = getattr(module, "impl", None)
        if impl is not None:
            for attr in ("bmm1_scale", "bmm2_scale"):
                if hasattr(impl, attr):
                    setattr(impl, attr, None)
        n += 1
    print(f"[trace] calibrated NVFP4 KV scales installed on {n} attention layers", flush=True)


# -- source patches ---------------------------------------------------------

_SHAPE_MARKER = "NeMo-RL patch (TRACE routed-experts extra rows)"
_INIT_MARKER = "NeMo-RL patch (TRACE capturer registration)"
_BIND_MARKER = "NeMo-RL patch (TRACE bind hook)"
_KV_MARKER = "NeMo-RL patch (TRACE kv capture)"
_OSCALE_MARKER = "NeMo-RL patch (NVFP4 KV fp8 output scale)"
KV_OSCALE_ENV = "NRL_TRACE_KV_OSCALE"


def _patch_routed_experts_capturer(logger) -> None:
    from nemo_rl.models.generation.vllm.patches import _get_vllm_file, _locked_file_patch

    path = _get_vllm_file("model_executor/layers/fused_moe/routed_experts_capturer.py")
    shape_old = "    num_layers = model_config.get_total_num_hidden_layers()\n"
    shape_new = (
        shape_old
        + f"    # {_SHAPE_MARKER}\n"
        "    from nemo_rl.quantization.trace.layout import extra_rows_for_vllm_config\n"
        "    num_layers += extra_rows_for_vllm_config(vllm_config)\n"
    )
    init_old = "        self.tp_size = vllm_config.parallel_config.tensor_parallel_size\n"
    init_new = (
        init_old
        + f"        # {_INIT_MARKER}\n"
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
    bind_old = (
        '    """Attach capture callbacks to the target model\'s MoE routers."""\n'
    )
    bind_new = (
        bind_old
        + f"    # {_BIND_MARKER}\n"
        "    from nemo_rl.models.generation.vllm.quantization import trace_capture\n"
        "    trace_capture.on_bind(model)\n"
    )
    with _locked_file_patch(path) as (content, write_back):
        if _SHAPE_MARKER in content and _INIT_MARKER in content and _BIND_MARKER in content:
            return
        for old in (shape_old, init_old, bind_old, dtype_old):
            if content.count(old) != 1:
                raise RuntimeError(f"TRACE patch anchor not found exactly once in {path}: {old!r}")
        content = (
            content.replace(shape_old, shape_new, 1)
            .replace(init_old, init_new, 1)
            .replace(bind_old, bind_new, 1)
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


def _patch_flashinfer_kv_update(logger) -> None:
    from nemo_rl.models.generation.vllm.patches import _get_vllm_file, _locked_file_patch

    path = _get_vllm_file("v1/attention/backends/flashinfer.py")
    old = (
        "                self.cache_dtype,\n"
        "                layer._k_scale,\n"
        "                layer._v_scale,\n"
        "            )\n"
        "\n"
        "\n"
        "def fast_plan_decode("
    )
    new = (
        "                self.cache_dtype,\n"
        "                layer._k_scale,\n"
        "                layer._v_scale,\n"
        "            )\n"
        f"            # {_KV_MARKER}\n"
        "            from nemo_rl.models.generation.vllm.quantization import trace_capture\n"
        "            trace_capture.capture_kv(self, layer, kv_cache, slot_mapping)\n"
        "\n"
        "\n"
        "def fast_plan_decode("
    )
    with _locked_file_patch(path) as (content, write_back):
        if _KV_MARKER in content:
            return
        if content.count(old) != 1:
            raise RuntimeError(f"TRACE kv patch anchor not found exactly once in {path}")
        write_back(content.replace(old, new, 1))
    logger.info("Patched flashinfer.py do_kv_cache_update for TRACE kv capture.")


def _patch_flashinfer_out_scale(logger) -> None:
    """Give the NVFP4-KV attention output a scale.

    With an NVFP4 KV cache trtllm-gen only writes FP8 output, and vLLM stores
    it unscaled (``bmm2_scale = v_scale``) before casting back to BF16. Qwen3's
    attention outputs are mostly < 2^-6, i.e. FP8 subnormals (up to ~27%
    error). The output is a convex combination of V rows, so ``|O| <= 448 *
    v_scale``: use ``v_scale`` as the output scale and multiply it back.
    """
    from nemo_rl.models.generation.vllm.patches import _get_vllm_file, _locked_file_patch

    path = _get_vllm_file("v1/attention/backends/flashinfer.py")
    os_expr = "getattr(self, '_nrl_os', 1.0)"
    init_old = (
        "            if is_quantized_kv_cache(self.kv_cache_dtype):\n"
        "                self.bmm2_scale *= layer._v_scale_float\n"
    )
    init_new = (
        init_old
        + f"            # {_OSCALE_MARKER}\n"
        "            if self.is_kvcache_nvfp4:\n"
        "                self._nrl_os = layer._v_scale_float\n"
        "                self.bmm2_scale /= self._nrl_os\n"
    )
    copies = [
        ("].copy_(out_prefill.to(output.dtype))", f"].copy_(out_prefill.to(output.dtype) * {os_expr})", 1),
        ("].copy_(out[:num_prefill_tokens].to(output.dtype))",
         f"].copy_(out[:num_prefill_tokens].to(output.dtype) * {os_expr})", 1),
        ("output[:num_decode_tokens].copy_(out_decode.to(output.dtype))",
         f"output[:num_decode_tokens].copy_(out_decode.to(output.dtype) * {os_expr})", 1),
        ("output[:num_decode_tokens].copy_(out.to(output.dtype))",
         f"output[:num_decode_tokens].copy_(out.to(output.dtype) * {os_expr})", 1),
        # FlashInfer wrapper paths take v_scale directly instead of bmm2_scale.
        ("v_scale=layer._v_scale_float,", f"v_scale=layer._v_scale_float / {os_expr},", 5),
    ]
    with _locked_file_patch(path) as (content, write_back):
        if _OSCALE_MARKER in content:
            return
        if content.count(init_old) != 1:
            raise RuntimeError(f"NVFP4 out-scale anchor not found exactly once in {path}")
        content = content.replace(init_old, init_new, 1)
        for old, new, n in copies:
            if content.count(old) != n:
                raise RuntimeError(f"NVFP4 out-scale anchor {old!r} found {content.count(old)}x (want {n}) in {path}")
            content = content.replace(old, new)
        write_back(content)
    logger.info("Patched flashinfer.py: scaled FP8 attention output for NVFP4 KV.")


def apply_trace_capture_patches(logger) -> None:
    """Install the source patches when ``NRL_TRACE_*`` requests capture."""
    if not trace_layout.trace_requested():
        return
    _patch_routed_experts_capturer(logger)
    if os.environ.get(trace_layout.KV_LAYERS_ENV):
        _patch_flashinfer_kv_update(logger)
    if os.environ.get(KV_OSCALE_ENV) == "1":
        _patch_flashinfer_out_scale(logger)
