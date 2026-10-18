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
"""Trainer-side TRACE payload split and rollout/train FP4 alignment probe.

The rollout appends TRACE rows to the routed-experts tensor (see
:mod:`nemo_rl.quantization.trace.layout`). :func:`split_trace_rows` strips them
before router replay consumes the routes and stashes them for the current
microbatch: the pre-SP copy for attention sites (attention sees the full
sequence) and the SP-local copy for MoE sites.

When ``NRL_TRACE_PROBE_DIR`` is set, forward pre-hooks on the selected MoE
layers and core-attention modules compare what training would quantize with
what the rollout recorded, in no-grad (logprob) passes only, and append one
JSON line per (microbatch, site, layer) to ``<dir>/rank<r>.jsonl``.
"""

from __future__ import annotations

import collections
import json
import os
from typing import Any, Optional

import torch

from nemo_rl.quantization.trace import layout as trace_layout
from nemo_rl.quantization.trace.codec import fp4_lsb_from_packed, unpack_fp4_nibbles
from nemo_rl.quantization.trace.guided_quant import (
    E2M1_MAGNITUDES,
    guided_e2m1,
    nvfp4_rowscaled_quantize_ref,
)

PROBE_DIR_ENV = "NRL_TRACE_PROBE_DIR"
PROBE_SAMPLES_ENV = "NRL_TRACE_PROBE_SAVE_SAMPLES"

_LAYOUT: Optional[trace_layout.TraceLayout] = None
_ROWS_FULL: Optional[torch.Tensor] = None  # [T, extra_rows, topk], pre-SP split
_ROWS_LOCAL: Optional[torch.Tensor] = None  # SP-local token chunk
_ROUTES_LOCAL: Optional[torch.Tensor] = None  # SP-local R3 routes [T, L, topk]
_HOOKED: set[int] = set()
_CALL = 0
_SAVED = 0


def layout_for_megatron_config(cfg: Any) -> trace_layout.TraceLayout:
    num_kv = int(getattr(cfg, "num_query_groups", None) or cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "kv_channels", None) or cfg.hidden_size // cfg.num_attention_heads)
    # The rollout captures every KV head (vLLM TP=1); trainer TP ranks slice.
    return trace_layout.layout_from_env(
        topk=int(cfg.moe_router_topk),
        hidden_size=int(cfg.hidden_size),
        num_kv_heads=num_kv,
        head_dim=head_dim,
        moe_intermediate_size=int(getattr(cfg, "moe_ffn_hidden_size", 0) or 0),
        num_experts=int(getattr(cfg, "num_moe_experts", 0) or 0),
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
    data_dict["routed_experts"] = torch.cat([routes, rows.view(routes.dtype) if routes.element_size() == 1 else rows.to(routes.dtype)], dim=2)


def split_trace_rows(
    model: Any,
    model_config: Any,
    routed_experts: torch.Tensor,
    split_for_sp,
) -> torch.Tensor:
    """Strip TRACE rows from ``[T, L + R, K]`` and stash them; return routes."""
    global _LAYOUT, _ROWS_FULL, _ROWS_LOCAL, _ROUTES_LOCAL
    if not trace_layout.trace_requested():
        return routed_experts
    if _LAYOUT is None:
        _LAYOUT = layout_for_megatron_config(model_config)
    r = _LAYOUT.extra_rows
    total = int(getattr(model_config, "num_layers"))
    if routed_experts.shape[1] == total:
        # No trace rows were attached (no captured sample in this microbatch).
        flush_apply_stats()
        _ROWS_FULL = _ROWS_LOCAL = _ROUTES_LOCAL = None
        _FC2_GUIDE.clear()
        return routed_experts
    if routed_experts.shape[1] != total + r:
        raise ValueError(
            f"TRACE: routed_experts has {routed_experts.shape[1]} layer rows, "
            f"expected num_layers={total} + extra_rows={r}"
        )
    flush_apply_stats()
    dump_kv_calib()
    ensure_kv_calib_hooks(model)
    _ROWS_FULL = routed_experts[:, total:, :]
    _ROWS_LOCAL = split_for_sp(model_config, _ROWS_FULL)
    _ROUTES_LOCAL = split_for_sp(model_config, routed_experts[:, :total, :])
    _FC2_GUIDE.clear()
    if os.environ.get(PROBE_DIR_ENV):
        _ensure_probe_hooks(model)
    if _apply_fc1():
        _install_apply_patch()
    modes = os.environ.get(APPLY_ENV, "").split(",")
    if any(m in modes for m in ("kv", "kv_rtn", "q8", "o8")):
        _install_attention_apply_hooks(model)
    if _LAYOUT.fc2_records or any(m in modes for m in ("fc2", "fc2p")):
        _install_fc2_hooks(model)
    return routed_experts[:, :total, :]


# -- probe ------------------------------------------------------------------


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _tp() -> tuple[int, int]:
    from megatron.core import parallel_state

    return (
        parallel_state.get_tensor_model_parallel_rank(),
        parallel_state.get_tensor_model_parallel_world_size(),
    )


def _emit(record: dict) -> None:
    d = os.environ[PROBE_DIR_ENV]
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"rank{_rank()}.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def _maybe_save(name: str, tensors: dict) -> None:
    global _SAVED
    limit = int(os.environ.get(PROBE_SAMPLES_ENV, "4"))
    if _rank() != 0 or _SAVED >= limit:
        return
    _SAVED += 1
    d = os.path.join(os.environ[PROBE_DIR_ENV], "samples")
    os.makedirs(d, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in tensors.items()}, os.path.join(d, f"{name}.pt"))


_MAGS = None


def _mag_index(codes: torch.Tensor) -> torch.Tensor:
    return (codes & 7).to(torch.int16)


def _code_stats(train: torch.Tensor, roll: torch.Tensor) -> dict:
    """train/roll: 4-bit codes, same shape. Signed zeros count as equal."""
    zero = (_mag_index(train) == 0) & (_mag_index(roll) == 0)
    diff = (train != roll) & ~zero
    n = diff.numel()
    nd = int(diff.sum())
    same_sign = ((train & 8) == (roll & 8)) | (_mag_index(train) == 0) | (_mag_index(roll) == 0)
    adj = diff & same_sign & ((_mag_index(train) - _mag_index(roll)).abs() == 1)
    # Codebook distance on the signed E2M1 axis (-6 .. -0.5, 0, 0.5 .. 6):
    # signed index = +-magnitude code, with both zeros at 0.
    def signed_idx(c):
        m = _mag_index(c)
        return torch.where((c & 8) != 0, -m, m)

    dist = (signed_idx(train) - signed_idx(roll)).abs()
    hist = torch.bincount(dist.flatten().to(torch.int64), minlength=15)[:15]
    return {"elems": n, "mismatch": nd, "adjacent": int(adj.sum()),
            **{f"off{k}": int(hist[k]) for k in range(1, 15) if int(hist[k])}}


def _dequant(codes: torch.Tensor) -> torch.Tensor:
    mags = torch.tensor(E2M1_MAGNITUDES, device=codes.device)
    v = mags[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -v, v)


def _probe_fc1(layer: int, x: torch.Tensor) -> None:
    rows = _ROWS_LOCAL
    x = x.reshape(-1, x.shape[-1])
    rec = {"site": "fc1", "layer": layer, "call": _CALL, "tokens": int(x.shape[0]),
           "applied": _apply_fc1()}
    if rows is None or rows.shape[0] != x.shape[0]:
        rec["error"] = f"token mismatch rows={None if rows is None else rows.shape[0]}"
        _emit(rec)
        return
    dec = trace_layout.decode_fc1(_LAYOUT, rows, layer)
    v = dec.valid
    rec["valid_tokens"] = int(v.sum())
    if rec["valid_tokens"] == 0:
        _emit(rec)
        return
    xv = x[v]
    own = nvfp4_rowscaled_quantize_ref(xv)
    train_codes = unpack_fp4_nibbles(own.data)
    lsb = dec.lsb()[v]
    # Mantissa-bit disagreement of the unguided trainer codes: available in
    # every mode and the quantity TRACE drives to zero.
    rtn_lsb = (train_codes & 1).bool()
    from nemo_rl.quantization.trace.codec import unpack_bits

    roll_lsb = unpack_bits(lsb)[:, : xv.shape[1]]
    rec["lsb_mismatch"] = float((rtn_lsb != roll_lsb).float().mean())
    if dec.token_scale is not None:
        tok_roll = dec.token_scale[v]
        tok_train = own.row_amax / (6.0 * 448.0)
        rec["tok_scale_exact"] = float((tok_train == tok_roll).float().mean())
        rec["tok_scale_max_rel"] = float(
            ((tok_train - tok_roll).abs() / tok_roll.clamp_min(1e-30)).max()
        )
        sf_roll = dec.block_scale[v]
        rec["block_scale_exact"] = float(
            (own.decode_scale.view(torch.uint8) == sf_roll.view(torch.uint8)).float().mean()
        )
    if dec.mode == "full":
        roll_packed = dec.codes[v]
        roll_codes = unpack_fp4_nibbles(roll_packed)
        rec["rtn_own_scale"] = _code_stats(train_codes, roll_codes)
        guided_own = nvfp4_rowscaled_quantize_ref(xv, guide_lsb_packed=lsb)
        g_own = unpack_fp4_nibbles(guided_own.data)
        rec["trace_own_scale"] = _code_stats(g_own, roll_codes)
        qr = _dequant(roll_codes)
        rec["d_act_rtn_own"] = float((_dequant(train_codes) - qr).norm() / qr.norm().clamp_min(1e-30))
        rec["d_act_trace_own"] = float((_dequant(g_own) - qr).norm() / qr.norm().clamp_min(1e-30))
        if dec.token_scale is not None:
            # RTN under rollout scales isolates value differences from scale differences.
            under_roll = nvfp4_rowscaled_quantize_ref(
                xv, rollout_decode_scale=sf_roll, rollout_token_scale=tok_roll
            )
            rtn_roll_codes = unpack_fp4_nibbles(under_roll.data)
            rec["rtn_roll_scale"] = _code_stats(rtn_roll_codes, roll_codes)
            guided = nvfp4_rowscaled_quantize_ref(
                xv, guide_lsb_packed=lsb, rollout_decode_scale=sf_roll, rollout_token_scale=tok_roll
            )
            g_codes = unpack_fp4_nibbles(guided.data)
            rec["trace_roll_scale"] = _code_stats(g_codes, roll_codes)
            rec["d_act_rtn"] = float((_dequant(rtn_roll_codes) - qr).norm() / qr.norm().clamp_min(1e-30))
            rec["d_act_trace"] = float((_dequant(g_codes) - qr).norm() / qr.norm().clamp_min(1e-30))
        _maybe_save(f"fc1_L{layer}_call{_CALL}", {"x": xv[:64], "roll_codes": roll_packed[:64]})
    _emit(rec)


# -- apply (training forward) -------------------------------------------------

APPLY_ENV = "NRL_TRACE_APPLY"
_APPLY_STATS: dict = {}


def _apply_fc1() -> bool:
    return "fc1" in os.environ.get(APPLY_ENV, "").split(",")


def _apply_copy() -> bool:
    """Upper-bound mode: reuse whole rollout codes (needs full-mode records).

    ``b2``/``b3`` instead keep only that many low bits of each rollout code.
    """
    modes = os.environ.get(APPLY_ENV, "").split(",")
    return "copy" in modes or "b2" in modes or "b3" in modes


def _copy_bits() -> int:
    modes = os.environ.get(APPLY_ENV, "").split(",")
    return 2 if "b2" in modes else 3 if "b3" in modes else 4


VERIFY_ENV = "NRL_TRACE_VERIFY"


def _nudge(x: torch.Tensor, mode: str, payload: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Guided rewrite of ``x`` from a record payload in ``mode`` (lsb/b2/full).

    Uses the fused Triton kernel unless ``NRL_TRACE_VERIFY=1`` (torch reference
    with a re-quantization check). ``full`` payloads guide with the mantissa bit,
    or with ``copy``/``b2``/``b3`` bits when those apply modes are set.
    """
    copy = mode == "full" and _apply_copy()
    if x.is_cuda and os.environ.get(VERIFY_ENV) != "1":
        from nemo_rl.quantization.trace.nudge_triton import trace_nudge_rowscaled_fast

        if mode == "lsb":
            out, st = trace_nudge_rowscaled_fast(x, payload, valid, width=1)
        elif mode == "b2":
            out, st = trace_nudge_rowscaled_fast(x, payload, valid, width=2)
        elif copy:
            out, st = trace_nudge_rowscaled_fast(x, payload, valid, width=4, match_bits=_copy_bits())
        else:
            out, st = trace_nudge_rowscaled_fast(x, fp4_lsb_from_packed(payload), valid, width=1)
        stats = {"fast_calls": 1}
        if os.environ.get(PROBE_DIR_ENV):
            stats["flipped"] = int(st["flipped_t"])
            stats["elements"] = x.numel()
        return out, stats
    from nemo_rl.quantization.trace.guided_quant import trace_nudge_rowscaled

    if mode == "b2":
        return trace_nudge_rowscaled(x, None, valid, guide_codes_packed=trace_layout._guide_codes("b2", payload), guide_bits=2)
    if copy:
        return trace_nudge_rowscaled(x, None, valid, guide_codes_packed=payload, guide_bits=_copy_bits())
    lsb = payload if mode == "lsb" else fp4_lsb_from_packed(payload)
    return trace_nudge_rowscaled(x, lsb, valid)


def _apply_fc1_guidance(layer: int, hidden_states: torch.Tensor) -> torch.Tensor:
    """Return ``hidden_states`` whose TE quantization yields the TRACE codes."""
    rows = _ROWS_LOCAL
    shape = hidden_states.shape
    x = hidden_states.reshape(-1, shape[-1])
    if rows is None or rows.shape[0] != x.shape[0] or layer not in _LAYOUT.fc1_records:
        return hidden_states
    dec = trace_layout.decode_fc1(_LAYOUT, rows, layer)
    if _apply_copy() and dec.mode != "full":
        raise RuntimeError(f"TRACE copy mode needs full fc1 codes at L{layer}")
    with torch.no_grad():
        x_new, stats = _nudge(x.detach(), dec.mode, dec.codes, dec.valid)
    acc = _APPLY_STATS.setdefault(layer, collections.Counter())
    acc.update(stats)
    acc["calls"] += 1
    if dec.mode == "full" and os.environ.get(PROBE_DIR_ENV) and not torch.is_grad_enabled():
        _post_code_stats(acc, x_new, dec.codes, dec.valid)
    # Straight-through: forward uses the guided values, gradient flows to x.
    out = x + (x_new - x).detach()
    return out.view(shape)


def _post_code_stats(acc, x_new: torch.Tensor, roll_packed: torch.Tensor, valid: torch.Tensor) -> None:
    """Code agreement of the guided values with the rollout (full-code records only)."""
    if not valid.any():
        return
    with torch.no_grad():
        got = unpack_fp4_nibbles(nvfp4_rowscaled_quantize_ref(x_new[valid]).data)
        roll = unpack_fp4_nibbles(roll_packed[valid])[:, : got.shape[1]]
        acc.update({f"post_{k}": v for k, v in _code_stats(got, roll).items()})


def _install_apply_patch() -> None:
    from megatron.core.transformer.moe.moe_layer import MoELayer

    if getattr(MoELayer, "_nrl_trace_apply_patched", False):
        return
    orig = MoELayer.preprocess

    def preprocess(self, hidden_states, probs, routing_map):
        # Routing (and shared experts) already consumed the unguided hidden
        # states; only the copy dispatched to the routed experts is guided.
        if _LAYOUT is not None and _apply_fc1():
            hidden_states = _apply_fc1_guidance(self.layer_number - 1, hidden_states)
        return orig(self, hidden_states, probs, routing_map)

    MoELayer.preprocess = preprocess
    MoELayer._nrl_trace_apply_patched = True
    print("[trace] fc1 guidance applied in MoELayer.preprocess", flush=True)


def _apply_kv_mode() -> Optional[str]:
    """``kv`` (TRACE codes), ``kv_rtn`` (plain RTN fake quant) or None."""
    modes = os.environ.get(APPLY_ENV, "").split(",")
    return "kv" if "kv" in modes else ("kv_rtn" if "kv_rtn" in modes else None)


def _fp8_fake(x: torch.Tensor, scale: float) -> torch.Tensor:
    y = (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    return x + (y.to(x.dtype) - x).detach()


def _nvfp4_kv_fake(
    x: torch.Tensor,  # [n, H, D]
    gscale: float,
    roll_sf: Optional[torch.Tensor],  # [n, H, D/16] e4m3, rollout block scales
    roll_payload: Optional[torch.Tensor],  # [n, H, D*width/8] uint8 record code bits
    valid: Optional[torch.Tensor],  # [n] bool
    width: int = 1,
) -> tuple[torch.Tensor, dict]:
    """NVFP4 KV quant->dequant (rollout cache semantics), optionally TRACE-guided.

    Guided rows use the rollout's own E4M3 block scales (the paper's
    "normalise with the rollout scale"); other rows recompute
    ``e4m3(vmax / (6 * gscale))``. ``width`` is the record's bits per element:
    1 = mantissa bit (bracket rule), 2 = two low code bits (nearest code
    carrying them), 4 = full codes (guided by their mantissa bit).
    Straight-through gradient. Uses the fused Triton kernel on CUDA unless
    ``NRL_TRACE_VERIFY=1``.
    """
    if x.is_cuda and os.environ.get(VERIFY_ENV) != "1":
        from nemo_rl.quantization.trace.nudge_triton import kv_fake_fast

        q, st = kv_fake_fast(x.detach(), gscale, roll_sf, roll_payload, valid, width)
        stats = {"fast_calls": 1}
        if os.environ.get(PROBE_DIR_ENV):
            stats["flipped"] = int(st["flipped_t"])
            stats["elements"] = x.numel()
        return x + (q - x).detach(), stats
    n, h, d = x.shape
    xf = x.detach().float().view(n, h, d // 16, 16)
    inv = torch.reciprocal(torch.tensor(6.0 * gscale, dtype=torch.float32, device=x.device))
    own = (xf.abs().amax(-1, keepdim=True) * inv).clamp(max=448.0)
    own = own.to(torch.float8_e4m3fn).float()
    sf = own
    if roll_sf is not None:
        rsf = roll_sf.float().view(n, h, d // 16, 1)
        sf = torch.where(valid.view(n, 1, 1, 1), rsf, own)
    scale = sf * torch.tensor(gscale, dtype=torch.float32, device=x.device)
    enc = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    z = (xf * enc).clamp(-6.0, 6.0).view(n, h, d)
    codes, rtn = guided_e2m1(z)
    if roll_payload is not None:
        from nemo_rl.quantization.trace.codec import low2_to_packed_codes, unpack_bits
        from nemo_rl.quantization.trace.guided_quant import nearest_code_with_low_bits

        if width == 2:
            low = unpack_fp4_nibbles(low2_to_packed_codes(roll_payload))
            mag = nearest_code_with_low_bits(z, low, 2).to(torch.uint8)
            codes = mag | ((z < 0).to(torch.uint8) << 3)
        else:
            lsb = unpack_bits(roll_payload) if width == 1 else (unpack_fp4_nibbles(roll_payload) & 1).bool()
            codes, _ = guided_e2m1(z, lsb)
        codes = torch.where(valid.view(n, 1, 1), codes, rtn)
    q = (_dequant(codes).view(n, h, d // 16, 16) * scale).view(n, h, d)
    out = x + (q.to(x.dtype) - x).detach()
    stats = {"elements": n * h * d, "flipped": int((codes != rtn).sum())}
    return out, stats


def _apply_attention_quant(layer: int, args: tuple, kwargs: dict):
    """Mirror the rollout's NVFP4 KV (+ FP8 Q/O) numerics in core attention."""
    mode = _apply_kv_mode()
    modes = os.environ.get(APPLY_ENV, "").split(",")
    q, k, v = args[:3]
    scales = _kv_scales().get(layer)
    if scales is None:
        return None
    hk, d = k.shape[-2], k.shape[-1]
    k2, v2 = k.reshape(-1, hk, d), v.reshape(-1, hk, d)
    roll = {"k": (None, None, 1), "v": (None, None, 1)}
    valid = None
    rows = _ROWS_FULL
    if mode == "kv" and rows is not None and rows.shape[0] == k2.shape[0] and layer in _LAYOUT.kv_records:
        dec = trace_layout.decode_kv(_LAYOUT, rows, layer)
        tp_rank, _ = _tp()
        hs = slice(tp_rank * hk, (tp_rank + 1) * hk)
        valid = dec.valid
        width = {"lsb": 1, "b2": 2, "full": 4}[dec.mode]
        roll = {
            "k": (dec.k_scale[:, hs], dec.k_codes[:, hs], width),
            "v": (dec.v_scale[:, hs], dec.v_codes[:, hs], width),
        }
    acc = _APPLY_STATS.setdefault(("kv", layer), collections.Counter())
    if mode is not None:
        (ksf, kp, kw), (vsf, vp, vw) = roll["k"], roll["v"]
        k_new, st_k = _nvfp4_kv_fake(k2, scales["k"], ksf, kp, valid, kw)
        v_new, st_v = _nvfp4_kv_fake(v2, scales["v"], vsf, vp, valid, vw)
        acc.update(st_k)
        acc.update({f"v_{a}": b for a, b in st_v.items()})
        k, v = k_new.view(k.shape), v_new.view(v.shape)
    if "q8" in modes:
        q = _fp8_fake(q, scales["q"])
    acc["calls"] += 1
    return (q, k, v, *args[3:]), kwargs


def _install_attention_apply_hooks(model: Any) -> None:
    from megatron.core.transformer.attention import Attention

    chunks = model if isinstance(model, (list, tuple)) else [model]
    n = 0
    for chunk in chunks:
        for module in chunk.modules():
            if not isinstance(module, Attention) or getattr(module, "_nrl_trace_kv_apply", False):
                continue
            module._nrl_trace_kv_apply = True
            layer = module.layer_number - 1

            def pre(mod, args, kwargs, layer=layer):
                return _apply_attention_quant(layer, args, kwargs)

            module.core_attention.register_forward_pre_hook(pre, with_kwargs=True)
            if "o8" in os.environ.get(APPLY_ENV, "").split(","):
                # Rollout attention writes its output through an FP8 buffer.
                def post(mod, args, output):
                    if isinstance(output, tuple):
                        return (_fp8_fake(output[0], 1.0), *output[1:])
                    return _fp8_fake(output, 1.0)

                module.core_attention.register_forward_hook(post)
            n += 1
    if n:
        print(f"[trace] attention quant hooks installed on {n} layers "
              f"({os.environ.get(APPLY_ENV)})", flush=True)


# -- fc2 (expert-side, after token dispatch) ---------------------------------

# layer -> (lsb rows, valid rows, unpadded tokens per local expert)
_FC2_GUIDE: dict[int, tuple[torch.Tensor, torch.Tensor, list[int]]] = {}
_FC2_REC_BYTES_IDS = 2  # int16 expert id (+1, 0 = none) per top-k slot


def _fc2_modes() -> tuple[bool, bool]:
    modes = os.environ.get(APPLY_ENV, "").split(",")
    # Row-scaled NVFP4 is (up to bf16 rounding of p * act) invariant to the
    # per-row router prob, so guiding the prob-weighted fc2 input with the
    # rollout's unweighted codes works; moving probs after fc2 (``fc2p``) is
    # optional and costs a saved [rows, hidden] tensor per layer.
    return "fc2" in modes, "fc2p" in modes  # (TRACE nudge, probs after fc2)


def _build_fc2_side_payload(layer: int, n_tokens: int, hidden: int, dtype) -> Optional[torch.Tensor]:
    """Per-token ``[marker | R3 expert ids | fc2 LSBs]`` packed into a hidden-shaped bf16 view."""
    if _ROWS_LOCAL is None or _ROUTES_LOCAL is None or _ROWS_LOCAL.shape[0] != n_tokens:
        return None
    dec = trace_layout.decode_fc2(_LAYOUT, _ROWS_LOCAL, layer)
    topk = _LAYOUT.topk
    ids = _ROUTES_LOCAL[:, layer, :].to(torch.int32)
    ids = torch.where(ids >= 0, ids + 1, torch.zeros_like(ids)).to(torch.int16)
    nbytes = hidden * torch.finfo(dtype).bits // 8
    g = torch.zeros(n_tokens, nbytes, dtype=torch.uint8, device=_ROWS_LOCAL.device)
    g[:, 0:4] = torch.where(dec.valid.view(-1, 1), trace_layout.MARKER, 0).to(torch.uint8)
    o = 4
    g[:, o : o + 2 * topk] = ids.contiguous().view(torch.uint8).view(n_tokens, 2 * topk)
    o += 2 * topk
    lsb = dec.lsb.reshape(n_tokens, -1)
    if o + lsb.shape[1] > nbytes:
        raise RuntimeError(f"TRACE fc2 payload {o + lsb.shape[1]} B exceeds the {nbytes} B hidden row")
    g[:, o : o + lsb.shape[1]] = lsb
    return g.view(dtype).view(n_tokens, hidden)


def _decode_fc2_side_rows(g: torch.Tensor, expert_of_row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    topk, nb = _LAYOUT.topk, _LAYOUT.fc2_slot_bytes
    # Explicit width: an EP rank can receive zero rows.
    b = g.contiguous().view(torch.uint8).view(g.shape[0], g.shape[-1] * g.element_size())
    marker_ok = (b[:, 0:4] == trace_layout.MARKER).all(dim=1)
    ids = b[:, 4 : 4 + 2 * topk].contiguous().view(torch.int16).view(-1, topk).to(torch.int64)
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
        raise RuntimeError("TRACE fc2 side dispatch supports the HybridEP and alltoall dispatchers")
    n = hidden_states.shape[0]
    g = _build_fc2_side_payload(layer, n, hidden_states.shape[-1], hidden_states.dtype)
    if g is None:
        return
    if mgr._padded_num_tokens is not None and n < mgr._padded_num_tokens:
        g = torch.cat([g, g.new_zeros((mgr._padded_num_tokens - n, g.shape[-1]))], dim=0)
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
    local = torch.repeat_interleave(torch.arange(tpe.numel(), device=g_rows.device), tpe)
    if local.numel() != g_rows.shape[0]:
        raise RuntimeError(f"TRACE fc2: {g_rows.shape[0]} dispatched rows vs {local.numel()} expert slots")
    expert = local + int(moe_layer.local_expert_indices[0])
    _FC2_GUIDE[layer] = (*_decode_fc2_side_rows(g_rows, expert), tpe.tolist())


def _alltoall_fc2_rows(td, layer: int, n_rows: int, tokens_per_expert) -> None:
    """Replay the alltoall dispatcher's token movement on the guidance payload.

    Called after ``dispatch_postprocess`` so every split size is already on
    the host: permute by ``routing_map`` -> EP all-to-all with the same splits
    -> TP gather (SP) -> sort chunks by local expert. Pure data movement, so
    the packed bytes survive bit-exactly.
    """
    from megatron.core.tensor_parallel import all_to_all, gather_from_sequence_parallel_region
    from megatron.core.transformer.moe.moe_utils import permute, sort_chunks_by_idxs

    n_tok = td.hidden_shape_before_permute[0]
    # Only the payload bytes travel (not a full hidden row): pure data movement
    # accepts any width; keep rows 128-byte aligned.
    need = trace_layout.MARKER_BYTES + _FC2_REC_BYTES_IDS * _LAYOUT.topk + _LAYOUT.topk * _LAYOUT.fc2_slot_bytes
    width = -(-need // 128) * 64
    g = _build_fc2_side_payload(layer, n_tok, width, torch.bfloat16)
    if g is None or td.drop_and_pad:
        return
    with torch.no_grad():
        g = permute(g, td.routing_map, num_out_tokens=td.num_out_tokens, fused=td.config.moe_permute_fusion)[0]
        g = all_to_all(td.ep_group, g, td.output_splits, td.input_splits)
        if td.tp_size > 1:
            sizes = None if td.output_splits_tp is None else td.output_splits_tp.tolist()
            g = gather_from_sequence_parallel_region(g, group=td.tp_group, output_split_sizes=sizes)
        if td.num_local_experts > 1:
            g = sort_chunks_by_idxs(
                g,
                td.num_global_tokens_per_local_expert.ravel(),
                td.sort_input_by_local_experts,
                fused=td.config.moe_permute_fusion,
            )[0]
    if g.shape[0] != n_rows:
        raise RuntimeError(f"TRACE fc2 alltoall replay: {g.shape[0]} rows vs {n_rows} expert rows")
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
    shift = torch.tensor(padded, dtype=torch.int64).cumsum(0) - u.cumsum(0) - (torch.tensor(padded) - u)
    dst = (torch.arange(int(u.sum())) + torch.repeat_interleave(shift, u)).to(lsb.device, non_blocking=True)
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
    guided, _ = _fc2_modes()
    acc = _APPLY_STATS.setdefault(("fc2", layer), collections.Counter())
    probing = bool(os.environ.get(PROBE_DIR_ENV)) and not torch.is_grad_enabled()
    codes = payload if mode == "full" else None
    if probing and valid.any():
        from nemo_rl.quantization.trace.codec import unpack_bits

        lsb = payload if mode == "lsb" else fp4_lsb_from_packed(trace_layout._guide_codes(mode, payload))
        xv = x.detach()[valid]
        own = unpack_fp4_nibbles(nvfp4_rowscaled_quantize_ref(xv).data)
        roll = unpack_bits(lsb[valid])[:, : xv.shape[1]]
        acc["probe_elems"] += own.numel()
        acc["probe_lsb_mismatch"] += int(((own & 1).bool() != roll).sum())
        if mode == "full":
            roll_codes = unpack_fp4_nibbles(codes[valid])[:, : xv.shape[1]]
            acc.update({f"probe_{k}": v for k, v in _code_stats(own, roll_codes).items()})
    acc["rows"] += x.shape[0]
    if probing:
        acc["guided_rows"] += int(valid.sum())
    if not guided:
        return x
    if _apply_copy() and mode != "full":
        raise RuntimeError(f"TRACE copy mode needs NRL_TRACE_FC2_FULL=1 (L{layer})")
    with torch.no_grad():
        x_new, stats = _nudge(x.detach(), mode, payload, valid)
    acc.update(stats)
    if mode == "full" and probing:
        _post_code_stats(acc, x_new, codes, valid)
    return x + (x_new - x).detach()


def _install_fc2_hooks(model: Any) -> None:
    from megatron.core.transformer.moe.experts import TEGroupedMLP
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

        orig_forward = TEGroupedMLP.forward

        def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs, *a, **k):
            if not _fc2_modes()[1]:
                return orig_forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs, *a, **k)
            # Rollout applies router probs after GEMM2; do the same so fc2's
            # input is SwiGLU(fc1) exactly like the rollout's quantized input.
            out, bias = orig_forward(
                self, permuted_local_hidden_states, tokens_per_expert,
                torch.ones_like(permuted_probs), *a, **k,
            )
            # bf16 product: an fp32 copy of [rows, hidden] per layer OOMs
            # imbalanced EP ranks.
            return out * permuted_probs.reshape(-1, 1).to(out.dtype), bias

        TEGroupedMLP.forward = forward

        from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

        orig_dpp = MoEAlltoAllTokenDispatcher.dispatch_postprocess

        def dispatch_postprocess(self, global_input_tokens, global_probs):
            out = orig_dpp(self, global_input_tokens, global_probs)
            layer = getattr(self, "_nrl_trace_layer", None)
            if layer is not None and _LAYOUT is not None and layer in _LAYOUT.fc2_records:
                _alltoall_fc2_rows(self, layer, out[0].shape[0], out[1])
            return out

        MoEAlltoAllTokenDispatcher.dispatch_postprocess = dispatch_postprocess
        print(f"[trace] fc2 hooks installed ({os.environ.get(APPLY_ENV)})", flush=True)

    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for module in chunk.modules():
            if not isinstance(module, MoELayer) or getattr(module, "_nrl_trace_fc2_hook", False):
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


def flush_apply_stats() -> None:
    """Emit and reset accumulated apply counters (one JSON line per layer)."""
    if not _APPLY_STATS or not os.environ.get(PROBE_DIR_ENV):
        _APPLY_STATS.clear()
        return
    for key, c in sorted(_APPLY_STATS.items(), key=lambda kv: str(kv[0])):
        site, layer = key if isinstance(key, tuple) else ("fc1", key)
        _emit({"site": f"{site}_apply", "layer": layer, "call": _CALL, **dict(c)})
    _APPLY_STATS.clear()


def _kv_codes_under_scale(x: torch.Tensor, sf: torch.Tensor, global_scale: float) -> torch.Tensor:
    """x: [n, H, D]; sf: [n, H, D/16] e4m3 -> 4-bit codes [n, H, D] (RTN)."""
    n, h, d = x.shape
    scale = sf.to(torch.float32).view(n, h, d // 16, 1) * global_scale
    enc = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    z = (x.to(torch.float32).view(n, h, d // 16, 16) * enc).clamp(-6, 6).view(n, h, d)
    _, rtn = guided_e2m1(z)
    return rtn


def _kv_own_block_scale(x: torch.Tensor, global_scale: float) -> torch.Tensor:
    n, h, d = x.shape
    vmax = x.abs().to(torch.float32).view(n, h, d // 16, 16).amax(-1)
    return (vmax / (6.0 * global_scale)).clamp(max=448.0).to(torch.float8_e4m3fn)


_KV_SCALES: Optional[dict] = None


def _kv_scales() -> dict:
    """Rollout NVFP4 KV global scales (1.0 unless a calibration file is set)."""
    global _KV_SCALES
    if _KV_SCALES is None:
        path = os.environ.get("NRL_TRACE_KV_SCALES_FILE")
        _KV_SCALES = {}
        if path:
            from nemo_rl.models.generation.vllm.quantization.trace_capture import (
                kv_scales_from_file,
            )

            _KV_SCALES = kv_scales_from_file(
                path, float(os.environ.get("NRL_TRACE_KV_SCALE_HEADROOM", "1.0"))
            )
            which = os.environ.get("NRL_TRACE_KV_SET", "kvq")
            _KV_SCALES = {
                layer: {k: (v if k in which else 1.0) for k, v in d.items()}
                for layer, d in _KV_SCALES.items()
            }
    return _KV_SCALES


def _probe_kv(layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
    rows = _ROWS_FULL
    hk_local, d = key.shape[-2], key.shape[-1]
    k = key.reshape(-1, hk_local, d)
    vv = value.reshape(-1, hk_local, d)
    rec = {"site": "kv", "layer": layer, "call": _CALL, "tokens": int(k.shape[0])}
    if rows is None or rows.shape[0] != k.shape[0]:
        rec["error"] = f"token mismatch rows={None if rows is None else rows.shape[0]}"
        _emit(rec)
        return
    dec = trace_layout.decode_kv(_LAYOUT, rows, layer)
    tp_rank, tp = _tp()
    hs = slice(tp_rank * hk_local, (tp_rank + 1) * hk_local)
    v = dec.valid
    rec["valid_tokens"] = int(v.sum())
    if rec["valid_tokens"] == 0 or dec.mode != "full":
        _emit(rec)
        return
    scales = _kv_scales().get(layer, {})
    for name, x, sf, codes in (
        ("k", k, dec.k_scale, dec.k_codes),
        ("v", vv, dec.v_scale, dec.v_codes),
    ):
        gscale = scales.get(name, 1.0)
        x = x[v]
        sf = sf[v][:, hs]
        roll = unpack_fp4_nibbles(codes[v][:, hs])
        rec[f"{name}_block_scale_exact"] = float(
            (_kv_own_block_scale(x, gscale).view(torch.uint8) == sf.view(torch.uint8)).float().mean()
        )
        rtn = _kv_codes_under_scale(x, sf, gscale)
        rec[f"{name}_rtn_roll_scale"] = _code_stats(rtn, roll)
        n, h, dd = x.shape
        scale = sf.to(torch.float32).view(n, h, dd // 16, 1) * gscale
        enc = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
        z = (x.to(torch.float32).view(n, h, dd // 16, 16) * enc).clamp(-6, 6).view(n, h, dd)
        g, _ = guided_e2m1(z, (roll & 1).bool())
        rec[f"{name}_trace_roll_scale"] = _code_stats(g, roll)
        qr = _dequant(roll)
        rec[f"{name}_d_act_rtn"] = float((_dequant(rtn) - qr).norm() / qr.norm().clamp_min(1e-30))
        rec[f"{name}_d_act_trace"] = float((_dequant(g) - qr).norm() / qr.norm().clamp_min(1e-30))
    _maybe_save(f"kv_L{layer}_call{_CALL}", {"k": k[v][:64], "v": vv[v][:64],
                                            "k_sf": dec.k_scale[v][:64].view(torch.uint8),
                                            "k_codes": dec.k_codes[v][:64],
                                            "v_sf": dec.v_scale[v][:64].view(torch.uint8),
                                            "v_codes": dec.v_codes[v][:64]})
    _emit(rec)


def _ensure_probe_hooks(model: Any) -> None:
    if id(model) in _HOOKED:
        return
    _HOOKED.add(id(model))
    from megatron.core.transformer.attention import Attention
    from megatron.core.transformer.moe.moe_layer import MoELayer

    _install_dbg_hooks(model if isinstance(model, (list, tuple)) else [model])
    fc1_layers = set(_LAYOUT.fc1_layers)
    kv_layers = set(_LAYOUT.kv_layers)
    chunks = model if isinstance(model, (list, tuple)) else [model]
    n_hooks = 0
    for chunk in chunks:
        for module in chunk.modules():
            if getattr(module, "_nrl_trace_probe_hooked", False):
                continue
            if isinstance(module, MoELayer) and (module.layer_number - 1) in fc1_layers:
                module._nrl_trace_probe_hooked = True
                layer = module.layer_number - 1

                def moe_hook(mod, args, kwargs, layer=layer):
                    if torch.is_grad_enabled():
                        return None
                    x = args[0] if args else kwargs["hidden_states"]
                    with torch.no_grad():
                        _probe_fc1(layer, x)
                    return None

                module.register_forward_pre_hook(moe_hook, with_kwargs=True)
                n_hooks += 1
            elif isinstance(module, Attention) and (module.layer_number - 1) in kv_layers:
                module._nrl_trace_probe_hooked = True
                layer = module.layer_number - 1

                def attn_hook(mod, args, kwargs, layer=layer):
                    global _CALL
                    if torch.is_grad_enabled():
                        return None
                    q, k, v = args[:3]
                    with torch.no_grad():
                        _probe_kv(layer, k, v)
                    return None

                module.core_attention.register_forward_pre_hook(attn_hook, with_kwargs=True)
                n_hooks += 1
    print(f"[trace] probe hooks installed: {n_hooks} (layout {_LAYOUT.to_json()})", flush=True)


# -- KV amax calibration ------------------------------------------------------

KV_CALIB_DIR_ENV = "NRL_TRACE_KV_CALIB_DIR"
_KV_AMAX: dict[int, dict[str, float]] = {}


def ensure_kv_calib_hooks(model: Any) -> None:
    """Record per-layer max |q|, |k|, |v| (post-RoPE) in no-grad passes."""
    if not os.environ.get(KV_CALIB_DIR_ENV):
        return
    from megatron.core.transformer.attention import Attention

    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for module in chunk.modules():
            if not isinstance(module, Attention) or getattr(module, "_nrl_kv_calib", False):
                continue
            module._nrl_kv_calib = True
            layer = module.layer_number - 1

            def hook(mod, args, kwargs, layer=layer):
                if torch.is_grad_enabled():
                    return None
                d = _KV_AMAX.setdefault(layer, {"q": 0.0, "k": 0.0, "v": 0.0})
                for name, t in zip("qkv", args[:3]):
                    d[name] = max(d[name], float(t.detach().abs().amax()))
                return None

            module.core_attention.register_forward_pre_hook(hook, with_kwargs=True)


def dump_kv_calib() -> None:
    d = os.environ.get(KV_CALIB_DIR_ENV)
    if not d or not _KV_AMAX:
        return
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"rank{_rank()}.json"), "w") as f:
        json.dump({str(k): v for k, v in sorted(_KV_AMAX.items())}, f)


def _dbg_stats(prefix: str, t: torch.Tensor, r: torch.Tensor) -> dict:
    t, r = t.float(), r.float()
    d = (t - r)
    rel = float(d.norm() / r.norm().clamp_min(1e-30))
    tok_rel = (d.norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30))
    return {
        f"{prefix}_rel_l2": rel,
        f"{prefix}_tok_rel_p50": float(tok_rel.median()),
        f"{prefix}_tok_rel_p99": float(torch.quantile(tok_rel, 0.99)),
        f"{prefix}_tok_rel_max": float(tok_rel.max()),
        f"{prefix}_exact": float((t == r).float().mean()),
    }


def _dbg_step_units(x_t: torch.Tensor, x_r: torch.Tensor) -> dict:
    """|x_t - x_r| in units of the local FP4 code step (rollout row/block scale).

    E2M1 code gaps are 0.5 (|z|<2), 1 (2..4), 2 (4..6) in normalised units, so
    a difference above one local gap means the two sides can land two codes
    apart even with perfect rounding.
    """
    q = nvfp4_rowscaled_quantize_ref(x_r)
    m, k = x_r.shape
    scale = (q.decode_scale.float().view(m, k // 16, 1) * q.global_decode.view(m, 1, 1)).view(m, k // 16, 1)
    zr = (x_r.float().view(m, k // 16, 16) / scale.clamp_min(1e-30)).view(m, k).clamp(-6, 6)
    zt = (x_t.float().view(m, k // 16, 16) / scale.clamp_min(1e-30)).view(m, k).clamp(-6, 6)
    a = zr.abs()
    gap = torch.where(a < 2, 0.5, torch.where(a < 4, 1.0, 2.0))
    u = (zt - zr).abs() / gap
    return {
        "dz_gap_p50": float(u.median()),
        "dz_gap_p99": float(torch.quantile(u.flatten()[:: max(1, u.numel() // 2_000_000)], 0.99)),
        "dz_gt_half_gap": float((u > 0.5).float().mean()),
        "dz_gt_1gap": float((u > 1.0).float().mean()),
        "dz_gt_2gap": float((u > 2.0).float().mean()),
    }


_DBG_IN: dict[int, torch.Tensor] = {}


def _install_dbg_hooks(chunks) -> None:
    if not _LAYOUT.dbg_records:
        return
    from megatron.core.transformer.moe.moe_layer import MoELayer

    for chunk in chunks:
        for module in chunk.modules():
            if not isinstance(module, MoELayer) or getattr(module, "_nrl_trace_dbg", False):
                continue
            layer = module.layer_number - 1
            if layer not in _LAYOUT.dbg_records:
                continue
            module._nrl_trace_dbg = True

            def pre(mod, args, kwargs, layer=layer):
                if not torch.is_grad_enabled():
                    x = args[0] if args else kwargs["hidden_states"]
                    _DBG_IN[layer] = x.detach().reshape(-1, x.shape[-1]).clone()
                return None

            def post(mod, args, kwargs, output, layer=layer):
                x_t = _DBG_IN.pop(layer, None)
                if x_t is None or torch.is_grad_enabled():
                    return None
                with torch.no_grad():
                    y_t = (output[0] if isinstance(output, tuple) else output).reshape(x_t.shape[0], -1)
                    rows = _ROWS_LOCAL
                    rec = {"site": "dbg", "layer": layer, "call": _CALL, "tokens": int(x_t.shape[0])}
                    if rows is None or rows.shape[0] != x_t.shape[0]:
                        rec["error"] = "token mismatch"
                        _emit(rec)
                        return None
                    v, x_r, y_r, lg_r = trace_layout.decode_dbg(_LAYOUT, rows, layer)
                    rec["valid_tokens"] = int(v.sum())
                    if rec["valid_tokens"]:
                        xt, xr = x_t[v], x_r[v]
                        rec.update(_dbg_stats("in", xt, xr))
                        rec.update(_dbg_stats("out", y_t[v], y_r[v]))
                        rec.update(_dbg_step_units(xt, xr))
                        w = mod.router.weight
                        lg_t = torch.nn.functional.linear(xt.float(), w.float())
                        rec.update(_dbg_stats("logit", lg_t, lg_r[v]))
                        # Same-input check: rollout input through the trainer router.
                        lg_tr = torch.nn.functional.linear(xr.float(), w.float())
                        rec.update(_dbg_stats("logit_sameinput", lg_tr, lg_r[v]))
                        routes = _ROUTES_LOCAL[:, layer, :][v].long() if _ROUTES_LOCAL is not None else None
                        if routes is not None and (routes >= 0).all():
                            pt = torch.softmax(lg_t, -1).gather(1, routes)
                            pr = torch.softmax(lg_r[v].float(), -1).gather(1, routes)
                            pt, pr = pt / pt.sum(-1, keepdim=True), pr / pr.sum(-1, keepdim=True)
                            rec["topk_w_absdiff_mean"] = float((pt - pr).abs().mean())
                            rec["topk_w_absdiff_max"] = float((pt - pr).abs().max())
                    _emit(rec)
                return None

            module.register_forward_pre_hook(pre, with_kwargs=True)
            module.register_forward_hook(post, with_kwargs=True)


def next_microbatch() -> None:
    global _CALL
    _CALL += 1
