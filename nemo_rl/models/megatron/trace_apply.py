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
"""Trainer-side TRACE: payload transport and rollout-guided rounding.

The rollout appends TRACE rows to the routed-experts tensor (see
:mod:`nemo_rl.quantization.trace.layout`). :func:`attach_trace_rows` fetches
them per microbatch, :func:`split_trace_rows` strips them before router replay
consumes the routes and stashes them for the current microbatch: the pre-SP
copy for attention sites (attention sees the full sequence) and the SP-local
copy for MoE sites.

Guidance rewrites the activations TE is about to quantize so that TE's own
round-to-nearest lands on the TRACE code (the element is moved to the centre
of the target code, block maxima are left alone so TE's scales do not move):

* fc1: ``MoELayer.preprocess`` (routing and shared experts already consumed
  the unguided hidden states).
* fc2: a ``linear_fc2`` forward pre-hook; the guidance travels with the tokens
  through the same dispatch (HybridEP ``dispatch_with_permute`` or a replay of
  the alltoall dispatcher's movement) and is matched to rows by expert id.
* kv: NVFP4 quant->dequant of K/V before core attention, mirroring the rollout
  cache (trtllm-gen dequantizes to FP8, ``kv8``); ``kv_rtn`` is plain RTN.

All rewrites are straight-through: the gradient flows to the unguided input.
"""

from __future__ import annotations

import collections
import os
from typing import Any, Optional

import torch

from nemo_rl.quantization.trace import layout as trace_layout

APPLY_ENV = "NRL_TRACE_APPLY"
PROBE_DIR_ENV = "NRL_TRACE_PROBE_DIR"

_LAYOUT: Optional[trace_layout.TraceLayout] = None
_ROWS_FULL: Optional[torch.Tensor] = None  # [T, extra_rows, topk], pre-SP split
_ROWS_LOCAL: Optional[torch.Tensor] = None  # SP-local token chunk
_ROUTES_LOCAL: Optional[torch.Tensor] = None  # SP-local R3 routes [T, L, topk]
_CALL = 0
_APPLY_STATS: dict = {}
# Tests flip this to compare the Triton kernels with the torch references.
USE_FUSED = True


def _modes() -> list[str]:
    return os.environ.get(APPLY_ENV, "").split(",")


def layout_for_megatron_config(cfg: Any) -> trace_layout.TraceLayout:
    num_kv = int(getattr(cfg, "num_query_groups", None) or cfg.num_attention_heads)
    head_dim = int(
        getattr(cfg, "kv_channels", None) or cfg.hidden_size // cfg.num_attention_heads
    )
    # The rollout captures every KV head (vLLM TP=1); trainer TP ranks slice.
    return trace_layout.layout_from_env(
        topk=int(cfg.moe_router_topk),
        hidden_size=int(cfg.hidden_size),
        num_kv_heads=num_kv,
        head_dim=head_dim,
        moe_intermediate_size=int(getattr(cfg, "moe_ffn_hidden_size", 0) or 0),
    )


def attach_trace_rows(data_dict: Any) -> None:
    """Re-attach by-reference TRACE rows onto a microbatch's routed_experts.

    The rollout ships each sample's unpadded ``[tokens, R, topk]`` uint8 trace
    payload as a Ray ObjectRef (``trace_refs``) so it never crosses the driver
    padded. Per microbatch we fetch only our samples and append the rows to
    ``routed_experts`` (``[mb, S, L, K]`` -> ``[mb, S, L + R, K]``); the
    existing padding / CP / SP / router-replay path then carries them and
    :func:`split_trace_rows` strips them again. Missing refs and padded
    positions get zero rows, which fail the marker check (no guidance).
    """
    import ray

    refs = data_dict.pop("trace_refs")
    routes = data_dict.get("routed_experts")
    if routes is None or not trace_layout.trace_requested():
        return
    mb, seq, num_layers, topk = routes.shape
    live = [(i, r) for i, r in enumerate(refs) if r is not None]
    arrays = ray.get([r for _, r in live]) if live else []
    extra = next((a.shape[1] for a in arrays), None)
    if extra is None:
        if _LAYOUT is None:
            return
        extra = _LAYOUT.extra_rows
    rows = torch.zeros((mb, seq, extra, topk), dtype=torch.uint8)
    for (i, _), arr in zip(live, arrays):
        n = min(arr.shape[0], seq)
        rows[i, :n] = torch.from_numpy(arr[:n])
    rows = (
        rows.view(routes.dtype) if routes.element_size() == 1 else rows.to(routes.dtype)
    )
    data_dict["routed_experts"] = torch.cat([routes, rows], dim=2)


def split_trace_rows(
    model: Any,
    model_config: Any,
    routed_experts: torch.Tensor,
    split_for_sp,
) -> torch.Tensor:
    """Strip TRACE rows from ``[T, L + R, K]`` and stash them; return routes."""
    global _LAYOUT, _ROWS_FULL, _ROWS_LOCAL, _ROUTES_LOCAL
    from nemo_rl.models.megatron import trace_probe

    trace_probe.on_microbatch(model)
    if not trace_layout.trace_requested():
        return routed_experts
    if _LAYOUT is None:
        _LAYOUT = layout_for_megatron_config(model_config)
    r = _LAYOUT.extra_rows
    total = int(getattr(model_config, "num_layers"))
    trace_probe.flush_apply_stats(_APPLY_STATS, _CALL)
    _FC2_GUIDE.clear()
    if routed_experts.shape[1] == total:
        # No trace rows were attached (no captured sample in this microbatch).
        _ROWS_FULL = _ROWS_LOCAL = _ROUTES_LOCAL = None
        return routed_experts
    if routed_experts.shape[1] != total + r:
        raise ValueError(
            f"TRACE: routed_experts has {routed_experts.shape[1]} layer rows, "
            f"expected num_layers={total} + extra_rows={r}"
        )
    _ROWS_FULL = routed_experts[:, total:, :]
    _ROWS_LOCAL = split_for_sp(model_config, _ROWS_FULL)
    _ROUTES_LOCAL = split_for_sp(model_config, routed_experts[:, :total, :])
    if os.environ.get(PROBE_DIR_ENV):
        trace_probe.ensure_probe_hooks(model)
    modes = _modes()
    if "fc1" in modes:
        _install_fc1_patch()
    if "kv" in modes or "kv_rtn" in modes:
        _install_attention_hooks(model)
    if _LAYOUT.fc2_records:
        _install_fc2_hooks(model)
    return routed_experts[:, :total, :]


def next_microbatch() -> None:
    global _CALL
    _CALL += 1


def _probing() -> bool:
    return bool(os.environ.get(PROBE_DIR_ENV)) and not torch.is_grad_enabled()


def _nudge(
    x: torch.Tensor, mode: str, payload: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    """Guided rewrite of ``x`` [M, K] from a record payload in ``mode``.

    ``lsb``/``b2`` payloads guide with the 1/2 low code bits; ``full`` records
    (probe layers) guide with their mantissa bit, like ``lsb``.
    """
    from nemo_rl.quantization.trace.codec import fp4_lsb_from_packed

    if x.is_cuda and USE_FUSED:
        from nemo_rl.quantization.trace.nudge_triton import trace_nudge_rowscaled_fast

        if mode == "b2":
            out, st = trace_nudge_rowscaled_fast(x, payload, valid, width=2)
        else:
            lsb = payload if mode == "lsb" else fp4_lsb_from_packed(payload)
            out, st = trace_nudge_rowscaled_fast(x, lsb, valid, width=1)
        stats = {"fast_calls": 1}
        if os.environ.get(PROBE_DIR_ENV):
            stats["flipped"] = int(st["flipped_t"])
            stats["elements"] = x.numel()
        return out, stats
    from nemo_rl.quantization.trace.guided_quant import trace_nudge_rowscaled

    if mode == "b2":
        codes = trace_layout._guide_codes("b2", payload)
        return trace_nudge_rowscaled(
            x, None, valid, guide_codes_packed=codes, guide_bits=2
        )
    lsb = payload if mode == "lsb" else fp4_lsb_from_packed(payload)
    return trace_nudge_rowscaled(x, lsb, valid)


# -- fc1 ------------------------------------------------------------------------


def _apply_fc1_guidance(layer: int, hidden_states: torch.Tensor) -> torch.Tensor:
    """Return ``hidden_states`` whose TE quantization yields the TRACE codes."""
    rows = _ROWS_LOCAL
    shape = hidden_states.shape
    x = hidden_states.reshape(-1, shape[-1])
    if rows is None or rows.shape[0] != x.shape[0] or layer not in _LAYOUT.fc1_records:
        return hidden_states
    dec = trace_layout.decode_fc1(_LAYOUT, rows, layer)
    with torch.no_grad():
        x_new, stats = _nudge(x.detach(), dec.mode, dec.codes, dec.valid)
    acc = _APPLY_STATS.setdefault(("fc1", layer), collections.Counter())
    acc.update(stats)
    acc["calls"] += 1
    if dec.mode == "full" and _probing():
        from nemo_rl.models.megatron import trace_probe

        trace_probe.post_code_stats(acc, x_new, dec.codes, dec.valid)
    # Straight-through: forward uses the guided values, gradient flows to x.
    return (x + (x_new - x).detach()).view(shape)


def _install_fc1_patch() -> None:
    from megatron.core.transformer.moe.moe_layer import MoELayer

    if getattr(MoELayer, "_nrl_trace_fc1_patched", False):
        return
    orig = MoELayer.preprocess

    def preprocess(self, hidden_states, probs, routing_map):
        # Routing (and shared experts) already consumed the unguided hidden
        # states; only the copy dispatched to the routed experts is guided.
        if _LAYOUT is not None:
            hidden_states = _apply_fc1_guidance(self.layer_number - 1, hidden_states)
        return orig(self, hidden_states, probs, routing_map)

    MoELayer.preprocess = preprocess
    MoELayer._nrl_trace_fc1_patched = True
    print("[trace] fc1 guidance applied in MoELayer.preprocess", flush=True)


# -- kv -------------------------------------------------------------------------


def dequant_codes(codes: torch.Tensor) -> torch.Tensor:
    """4-bit E2M1 codes (sign in bit 3) -> float values."""
    from nemo_rl.quantization.trace.guided_quant import E2M1_MAGNITUDES

    mags = torch.tensor(E2M1_MAGNITUDES, device=codes.device)
    v = mags[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -v, v)


_KV_SCALES: Optional[dict] = None


def kv_scales() -> dict:
    """Rollout NVFP4 KV global scales from the calibration file (``{}`` if unset)."""
    global _KV_SCALES
    if _KV_SCALES is None:
        path = os.environ.get("NRL_TRACE_KV_SCALES_FILE")
        _KV_SCALES = {}
        if path:
            from nemo_rl.quantization.trace.kv_calib import kv_scales_from_file

            _KV_SCALES = kv_scales_from_file(
                path, float(os.environ.get("NRL_TRACE_KV_SCALE_HEADROOM", "1.0"))
            )
    return _KV_SCALES


def _nvfp4_kv_fake(
    x: torch.Tensor,  # [n, H, D]
    gscale: float,
    roll_sf: Optional[torch.Tensor],  # [n, H, D/16] e4m3, rollout block scales
    roll_payload: Optional[torch.Tensor],  # [n, H, D*width/8] uint8 record code bits
    valid: Optional[torch.Tensor],  # [n] bool
    width: int = 1,
    fp8_dq: bool = False,
) -> tuple[torch.Tensor, dict]:
    """NVFP4 KV quant->dequant (rollout cache semantics), optionally TRACE-guided.

    Guided rows use the rollout's own E4M3 block scales (the paper's
    "normalise with the rollout scale"; the trainer owns this fake quant, so
    unlike fc1/fc2 it can); other rows recompute ``e4m3(vmax / (6 * gscale))``.
    ``width`` is the record's bits per element: 1 = mantissa bit (bracket
    rule), 2 = two low code bits (nearest code carrying them), 4 = full codes
    (probe layers; guided by their mantissa bit). ``fp8_dq`` mirrors
    trtllm-gen, which dequantizes ``e2m1 * block scale`` to FP8 E4M3 before
    applying the global scale. Straight-through gradient.
    """
    if x.is_cuda and USE_FUSED:
        from nemo_rl.quantization.trace.nudge_triton import kv_fake_fast

        q, st = kv_fake_fast(
            x.detach(), gscale, roll_sf, roll_payload, valid, width, None, fp8_dq
        )
        stats = {"fast_calls": 1}
        if os.environ.get(PROBE_DIR_ENV):
            stats["flipped"] = int(st["flipped_t"])
            stats["elements"] = x.numel()
        return x + (q - x).detach(), stats
    from nemo_rl.quantization.trace.codec import (
        low2_to_packed_codes,
        unpack_bits,
        unpack_fp4_nibbles,
    )
    from nemo_rl.quantization.trace.guided_quant import (
        guided_e2m1,
        nearest_code_with_low_bits,
    )

    n, h, d = x.shape
    xf = x.detach().float().view(n, h, d // 16, 16)
    inv = torch.reciprocal(
        torch.tensor(6.0 * gscale, dtype=torch.float32, device=x.device)
    )
    sf = (
        (xf.abs().amax(-1, keepdim=True) * inv)
        .clamp(max=448.0)
        .to(torch.float8_e4m3fn)
        .float()
    )
    if roll_sf is not None:
        sf = torch.where(
            valid.view(n, 1, 1, 1), roll_sf.float().view(n, h, d // 16, 1), sf
        )
    scale = sf * torch.tensor(gscale, dtype=torch.float32, device=x.device)
    enc = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    z = (xf * enc).clamp(-6.0, 6.0).view(n, h, d)
    codes, rtn = guided_e2m1(z)
    if roll_payload is not None:
        if width == 2:
            low = unpack_fp4_nibbles(low2_to_packed_codes(roll_payload))
            mag = nearest_code_with_low_bits(z, low, 2).to(torch.uint8)
            codes = mag | ((z < 0).to(torch.uint8) << 3)
        else:
            lsb = (
                unpack_bits(roll_payload)
                if width == 1
                else (unpack_fp4_nibbles(roll_payload) & 1).bool()
            )
            codes, _ = guided_e2m1(z, lsb)
        codes = torch.where(valid.view(n, 1, 1), codes, rtn)
    vals = dequant_codes(codes).view(n, h, d // 16, 16)
    if fp8_dq:
        g32 = torch.tensor(gscale, dtype=torch.float32, device=x.device)
        q = ((vals * sf).to(torch.float8_e4m3fn).float() * g32).view(n, h, d)
    else:
        q = (vals * scale).view(n, h, d)
    out = x + (q.to(x.dtype) - x).detach()
    return out, {"elements": n * h * d, "flipped": int((codes != rtn).sum())}


def _tp_rank() -> int:
    from megatron.core import parallel_state

    return parallel_state.get_tensor_model_parallel_rank()


def _apply_attention_quant(layer: int, args: tuple, kwargs: dict):
    """Mirror the rollout's NVFP4 KV numerics in core attention."""
    modes = _modes()
    q, k, v = args[:3]
    scales = kv_scales().get(layer)
    if scales is None:
        return None
    hk, d = k.shape[-2], k.shape[-1]
    k2, v2 = k.reshape(-1, hk, d), v.reshape(-1, hk, d)
    roll = {"k": (None, None, 1), "v": (None, None, 1)}
    valid = None
    rows = _ROWS_FULL
    if (
        "kv" in modes
        and rows is not None
        and rows.shape[0] == k2.shape[0]
        and layer in _LAYOUT.kv_records
    ):
        dec = trace_layout.decode_kv(_LAYOUT, rows, layer)
        hs = slice(_tp_rank() * hk, (_tp_rank() + 1) * hk)
        valid = dec.valid
        width = {"lsb": 1, "b2": 2, "full": 4}[dec.mode]
        roll = {
            "k": (dec.k_scale[:, hs], dec.k_codes[:, hs], width),
            "v": (dec.v_scale[:, hs], dec.v_codes[:, hs], width),
        }
    acc = _APPLY_STATS.setdefault(("kv", layer), collections.Counter())
    (ksf, kp, kw), (vsf, vp, vw) = roll["k"], roll["v"]
    fp8_dq = "kv8" in modes
    k_new, st_k = _nvfp4_kv_fake(k2, scales["k"], ksf, kp, valid, kw, fp8_dq)
    v_new, st_v = _nvfp4_kv_fake(v2, scales["v"], vsf, vp, valid, vw, fp8_dq)
    if kw == 4 and _probing() and valid.any():
        from nemo_rl.models.megatron import trace_probe

        trace_probe.post_kv_code_stats(
            acc, k_new, v_new, ksf, kp, vsf, vp, valid, scales
        )
    acc.update(st_k)
    acc.update({f"v_{a}": b for a, b in st_v.items()})
    acc["calls"] += 1
    return (q, k_new.view(k.shape), v_new.view(v.shape), *args[3:]), kwargs


def _install_attention_hooks(model: Any) -> None:
    from megatron.core.transformer.attention import Attention

    chunks = model if isinstance(model, (list, tuple)) else [model]
    n = 0
    for chunk in chunks:
        for module in chunk.modules():
            if not isinstance(module, Attention) or getattr(
                module, "_nrl_trace_kv_apply", False
            ):
                continue
            module._nrl_trace_kv_apply = True
            layer = module.layer_number - 1

            def pre(mod, args, kwargs, layer=layer):
                return _apply_attention_quant(layer, args, kwargs)

            module.core_attention.register_forward_pre_hook(pre, with_kwargs=True)
            n += 1
    if n:
        print(
            f"[trace] NVFP4 KV hooks installed on {n} attention layers ({os.environ.get(APPLY_ENV)})",
            flush=True,
        )


# -- fc2 (expert side, after token dispatch) ------------------------------------

# layer -> (code rows, valid rows, unpadded tokens per local expert)
_FC2_GUIDE: dict[int, tuple[torch.Tensor, torch.Tensor, list[int]]] = {}
_FC2_REC_BYTES_IDS = 2  # int16 expert id (+1, 0 = none) per top-k slot


def _build_fc2_side_payload(
    layer: int, n_tokens: int, hidden: int, dtype
) -> Optional[torch.Tensor]:
    """Per-token ``[marker | R3 expert ids | fc2 code bits]`` packed into a bf16 view."""
    if _ROWS_LOCAL is None or _ROUTES_LOCAL is None or _ROWS_LOCAL.shape[0] != n_tokens:
        return None
    dec = trace_layout.decode_fc2(_LAYOUT, _ROWS_LOCAL, layer)
    topk = _LAYOUT.topk
    ids = _ROUTES_LOCAL[:, layer, :].to(torch.int32)
    ids = torch.where(ids >= 0, ids + 1, torch.zeros_like(ids)).to(torch.int16)
    nbytes = hidden * torch.finfo(dtype).bits // 8
    g = torch.zeros(n_tokens, nbytes, dtype=torch.uint8, device=_ROWS_LOCAL.device)
    g[:, 0:4] = torch.where(dec.valid.view(-1, 1), trace_layout.MARKER, 0).to(
        torch.uint8
    )
    o = 4
    g[:, o : o + 2 * topk] = ids.contiguous().view(torch.uint8).view(n_tokens, 2 * topk)
    o += 2 * topk
    lsb = dec.lsb.reshape(n_tokens, -1)
    if o + lsb.shape[1] > nbytes:
        raise RuntimeError(
            f"TRACE fc2 payload {o + lsb.shape[1]} B exceeds the {nbytes} B hidden row"
        )
    g[:, o : o + lsb.shape[1]] = lsb
    return g.view(dtype).view(n_tokens, hidden)


def _decode_fc2_side_rows(
    g: torch.Tensor, expert_of_row: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    topk, nb = _LAYOUT.topk, _LAYOUT.fc2_slot_bytes
    # Explicit width: an EP rank can receive zero rows.
    b = (
        g.contiguous()
        .view(torch.uint8)
        .view(g.shape[0], g.shape[-1] * g.element_size())
    )
    marker_ok = (b[:, 0:4] == trace_layout.MARKER).all(dim=1)
    ids = (
        b[:, 4 : 4 + 2 * topk]
        .contiguous()
        .view(torch.int16)
        .view(-1, topk)
        .to(torch.int64)
    )
    hit = ids == (expert_of_row.view(-1, 1) + 1)
    k = hit.to(torch.int8).argmax(dim=1)
    lsb = b[:, 4 + 2 * topk : 4 + 2 * topk + topk * nb].view(-1, topk, nb)
    rows = lsb[torch.arange(lsb.shape[0], device=lsb.device), k]
    return rows, marker_ok & hit.any(dim=1)


def _side_dispatch_fc2(moe_layer, hidden_states: torch.Tensor) -> None:
    layer = moe_layer.layer_number - 1
    if layer not in _LAYOUT.fc2_records:
        return
    from megatron.core.transformer.moe import fused_a2a

    mgr = getattr(moe_layer.token_dispatcher, "_comm_manager", None)
    if mgr is None:
        return  # alltoall: handled in dispatch_postprocess
    if type(mgr).__name__ != "_HybridEPManager" or mgr.handle is None:
        raise RuntimeError("TRACE fc2 supports the HybridEP and alltoall dispatchers")
    n = hidden_states.shape[0]
    g = _build_fc2_side_payload(layer, n, hidden_states.shape[-1], hidden_states.dtype)
    if g is None:
        return
    if mgr._padded_num_tokens is not None and n < mgr._padded_num_tokens:
        g = torch.cat(
            [g, g.new_zeros((mgr._padded_num_tokens - n, g.shape[-1]))], dim=0
        )
    fused = moe_layer.config.moe_permute_fusion_into_hybridep
    with torch.no_grad():
        out = fused_a2a._hybrid_ep_buffer.dispatch_with_permute(
            hidden=g,
            scaling_factor=None,
            handle=mgr.handle,
            pad_multiple=mgr.pad_multiple,
            num_permuted_tokens=mgr.num_permuted_tokens,
            **({"fuse_permute_dispatch": fused} if fused else {}),
        )
    g_rows = out[0]
    tpe = mgr.get_number_of_tokens_per_expert().to(g_rows.device)
    local = torch.repeat_interleave(
        torch.arange(tpe.numel(), device=g_rows.device), tpe
    )
    if local.numel() != g_rows.shape[0]:
        raise RuntimeError(
            f"TRACE fc2: {g_rows.shape[0]} dispatched rows vs {local.numel()} expert slots"
        )
    expert = local + int(moe_layer.local_expert_indices[0])
    _FC2_GUIDE[layer] = (*_decode_fc2_side_rows(g_rows, expert), tpe.tolist())


def _alltoall_fc2_rows(td, layer: int, n_rows: int, tokens_per_expert) -> None:
    """Replay the alltoall dispatcher's token movement on the guidance payload.

    Called after ``dispatch_postprocess`` so every split size is already on
    the host: permute by ``routing_map`` -> EP all-to-all with the same splits
    -> TP gather (SP) -> sort chunks by local expert. Pure data movement, so
    the packed bytes survive bit-exactly.
    """
    from megatron.core.tensor_parallel import (
        all_to_all,
        gather_from_sequence_parallel_region,
    )
    from megatron.core.transformer.moe.moe_utils import permute, sort_chunks_by_idxs

    n_tok = td.hidden_shape_before_permute[0]
    # Only the payload bytes travel (not a full hidden row): pure data movement
    # accepts any width; keep rows 128-byte aligned.
    need = (
        trace_layout.MARKER_BYTES
        + _FC2_REC_BYTES_IDS * _LAYOUT.topk
        + _LAYOUT.topk * _LAYOUT.fc2_slot_bytes
    )
    width = -(-need // 128) * 64
    g = _build_fc2_side_payload(layer, n_tok, width, torch.bfloat16)
    if g is None or td.drop_and_pad:
        return
    with torch.no_grad():
        g = permute(
            g,
            td.routing_map,
            num_out_tokens=td.num_out_tokens,
            fused=td.config.moe_permute_fusion,
        )[0]
        g = all_to_all(td.ep_group, g, td.output_splits, td.input_splits)
        if td.tp_size > 1:
            sizes = (
                None if td.output_splits_tp is None else td.output_splits_tp.tolist()
            )
            g = gather_from_sequence_parallel_region(
                g, group=td.tp_group, output_split_sizes=sizes
            )
        if td.num_local_experts > 1:
            g = sort_chunks_by_idxs(
                g,
                td.num_global_tokens_per_local_expert.ravel(),
                td.sort_input_by_local_experts,
                fused=td.config.moe_permute_fusion,
            )[0]
    if g.shape[0] != n_rows:
        raise RuntimeError(
            f"TRACE fc2 alltoall replay: {g.shape[0]} rows vs {n_rows} expert rows"
        )
    tpe = torch.as_tensor(tokens_per_expert, device=g.device)
    local = torch.repeat_interleave(torch.arange(tpe.numel(), device=g.device), tpe)
    expert = local + int(td.local_expert_indices[0])
    _FC2_GUIDE[layer] = (*_decode_fc2_side_rows(g, expert), tpe.tolist())


def _pad_guide_to_splits(lsb, valid, unpadded: list[int], padded: list[int]):
    """Lay guidance rows out like TEGroupedMLP's per-expert quantization padding."""
    if list(unpadded) == list(padded):
        return lsb, valid
    out_l = lsb.new_zeros((sum(padded), lsb.shape[1]))
    out_v = valid.new_zeros((sum(padded),))
    # Destination row of every source row: its expert's padded start + rank.
    u = torch.tensor(unpadded, dtype=torch.int64)
    shift = (
        torch.tensor(padded, dtype=torch.int64).cumsum(0)
        - u.cumsum(0)
        - (torch.tensor(padded) - u)
    )
    dst = (torch.arange(int(u.sum())) + torch.repeat_interleave(shift, u)).to(
        lsb.device, non_blocking=True
    )
    out_l[dst] = lsb
    out_v[dst] = valid
    return out_l, out_v


def _fc2_pre_hook(layer: int, x: torch.Tensor, m_splits) -> torch.Tensor:
    guide = _FC2_GUIDE.get(layer)
    if guide is None:
        return x
    lsb, valid, unpadded = guide
    padded = list(m_splits) if m_splits is not None else unpadded
    if len(padded) != len(unpadded) or sum(padded) != x.shape[0]:
        raise RuntimeError(
            f"TRACE fc2 L{layer}: fc2 rows {x.shape[0]} / splits {padded} do not match "
            f"guidance splits {unpadded}"
        )
    payload, valid = _pad_guide_to_splits(lsb, valid, unpadded, padded)
    mode = _LAYOUT.fc2_mode
    acc = _APPLY_STATS.setdefault(("fc2", layer), collections.Counter())
    acc["rows"] += x.shape[0]
    if _probing():
        from nemo_rl.models.megatron import trace_probe

        trace_probe.fc2_probe_stats(acc, x, payload, valid, mode)
    if "fc2" not in _modes():
        return x
    with torch.no_grad():
        x_new, stats = _nudge(x.detach(), mode, payload, valid)
    acc.update(stats)
    # Row-scaled NVFP4 is (up to bf16 rounding of p * act) invariant to the
    # per-row router prob, so the prob-weighted fc2 input can be guided with
    # the rollout's unweighted codes directly.
    return x + (x_new - x).detach()


def _install_fc2_hooks(model: Any) -> None:
    from megatron.core.transformer.moe.moe_layer import MoELayer

    if not getattr(MoELayer, "_nrl_trace_fc2_patched", False):
        orig_dispatch = MoELayer.dispatch

        def dispatch(self, hidden_states, probs):
            out = orig_dispatch(self, hidden_states, probs)
            if _LAYOUT is not None and _LAYOUT.fc2_records:
                _side_dispatch_fc2(self, hidden_states)
            return out

        MoELayer.dispatch = dispatch
        MoELayer._nrl_trace_fc2_patched = True

        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAlltoAllTokenDispatcher,
        )

        orig_dpp = MoEAlltoAllTokenDispatcher.dispatch_postprocess

        def dispatch_postprocess(self, global_input_tokens, global_probs):
            out = orig_dpp(self, global_input_tokens, global_probs)
            layer = getattr(self, "_nrl_trace_layer", None)
            if (
                layer is not None
                and _LAYOUT is not None
                and layer in _LAYOUT.fc2_records
            ):
                _alltoall_fc2_rows(self, layer, out[0].shape[0], out[1])
            return out

        MoEAlltoAllTokenDispatcher.dispatch_postprocess = dispatch_postprocess
        print(f"[trace] fc2 hooks installed ({os.environ.get(APPLY_ENV)})", flush=True)

    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for module in chunk.modules():
            if not isinstance(module, MoELayer) or getattr(
                module, "_nrl_trace_fc2_hook", False
            ):
                continue
            module._nrl_trace_fc2_hook = True
            layer = module.layer_number - 1
            module.token_dispatcher._nrl_trace_layer = layer

            def pre(mod, args, layer=layer):
                splits = args[1] if len(args) > 1 else None
                if isinstance(splits, torch.Tensor):
                    splits = splits.tolist()
                return (_fc2_pre_hook(layer, args[0], splits), *args[1:])

            module.experts.linear_fc2.register_forward_pre_hook(pre)
