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
"""TRACE rollout/train FP4 alignment probe and NVFP4 KV calibration.

Probe (``trace.probe_dir``): forward pre-hooks on the TRACE MoE layers and
core-attention modules compare what training would quantize with what the
rollout recorded, in no-grad (logprob) passes only, and append one JSON line
per (microbatch, site, layer) to ``<dir>/rank<r>.jsonl``. Layers listed in
``trace.probe_layers`` ship full 4-bit codes, which adds exact code agreement
and the code-distance histogram (``off1``, ``off2``, ...).

Calibration (``trace.kv.calib_dir``): per-layer q/k/v amax in no-grad passes,
written to ``<dir>/rank<r>.json``; merge with
``python -m nemo_rl.quantization.trace.kv_calib``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from nemo_rl.models.megatron import trace_apply as ta
from nemo_rl.quantization.trace import layout as trace_layout
from nemo_rl.quantization.trace.codec import (
    fp4_lsb_from_packed,
    unpack_bits,
    unpack_fp4_nibbles,
)
from nemo_rl.quantization.trace.guided_quant import (
    guided_e2m1,
    nvfp4_rowscaled_quantize_ref,
)

KV_CALIB_DIR_ENV = "NRL_TRACE_KV_CALIB_DIR"
_HOOKED: set[int] = set()
_KV_AMAX: dict[int, dict[str, float]] = {}


def on_microbatch(model: Any) -> None:
    """Per-microbatch bookkeeping that runs whether or not TRACE is enabled."""
    if os.environ.get(KV_CALIB_DIR_ENV):
        _dump_kv_calib()
        _ensure_kv_calib_hooks(model)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _emit(record: dict) -> None:
    d = os.environ[ta.PROBE_DIR_ENV]
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"rank{_rank()}.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def flush_apply_stats(stats: dict, call: int) -> None:
    """Emit and reset the accumulated apply counters (one JSON line per layer)."""
    if stats and os.environ.get(ta.PROBE_DIR_ENV):
        for (site, layer), c in sorted(stats.items(), key=lambda kv: str(kv[0])):
            _emit({"site": f"{site}_apply", "layer": layer, "call": call, **dict(c)})
    stats.clear()


# -- code statistics --------------------------------------------------------------


def _mag_index(codes: torch.Tensor) -> torch.Tensor:
    return (codes & 7).to(torch.int16)


def code_stats(train: torch.Tensor, roll: torch.Tensor) -> dict:
    """train/roll: 4-bit codes, same shape. Signed zeros count as equal.

    ``offK`` counts elements K codes apart on the signed E2M1 axis
    (-6 .. -0.5, 0, 0.5 .. 6); RTN disagreement is almost all ``off1``,
    1-bit TRACE leaves mostly ``off2`` (rollout code outside the bracket).
    """
    zero = (_mag_index(train) == 0) & (_mag_index(roll) == 0)
    diff = (train != roll) & ~zero

    def signed_idx(c):
        m = _mag_index(c)
        return torch.where((c & 8) != 0, -m, m)

    dist = (signed_idx(train) - signed_idx(roll)).abs()
    hist = torch.bincount(dist.flatten().to(torch.int64), minlength=15)[:15]
    return {
        "elems": diff.numel(),
        "mismatch": int(diff.sum()),
        **{f"off{k}": int(hist[k]) for k in range(1, 15) if int(hist[k])},
    }


def _rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    return float((a - ref).norm() / ref.norm().clamp_min(1e-30))


def post_code_stats(
    acc, x_new: torch.Tensor, roll_packed: torch.Tensor, valid: torch.Tensor
) -> None:
    """Code agreement of the guided values with the rollout (full-code records only)."""
    if not valid.any():
        return
    with torch.no_grad():
        got = unpack_fp4_nibbles(nvfp4_rowscaled_quantize_ref(x_new[valid]).data)
        roll = unpack_fp4_nibbles(roll_packed[valid])[:, : got.shape[1]]
        acc.update({f"post_{k}": v for k, v in code_stats(got, roll).items()})


def fc2_probe_stats(
    acc, x: torch.Tensor, payload: torch.Tensor, valid: torch.Tensor, mode: str
) -> None:
    """Mantissa-bit (and, for full records, code) agreement of unguided fc2 RTN."""
    acc["guided_rows"] += int(valid.sum())
    if not valid.any():
        return
    with torch.no_grad():
        lsb = (
            payload
            if mode == "lsb"
            else fp4_lsb_from_packed(trace_layout._guide_codes(mode, payload))
        )
        xv = x.detach()[valid]
        own = unpack_fp4_nibbles(nvfp4_rowscaled_quantize_ref(xv).data)
        roll = unpack_bits(lsb[valid])[:, : xv.shape[1]]
        acc["probe_elems"] += own.numel()
        acc["probe_lsb_mismatch"] += int(((own & 1).bool() != roll).sum())
        if mode == "full":
            roll_codes = unpack_fp4_nibbles(payload[valid])[:, : xv.shape[1]]
            acc.update(
                {f"probe_{k}": v for k, v in code_stats(own, roll_codes).items()}
            )


def _kv_codes_under_scale(
    x: torch.Tensor, sf: torch.Tensor, global_scale: float
) -> torch.Tensor:
    """x: [n, H, D]; sf: [n, H, D/16] e4m3 -> RTN 4-bit codes [n, H, D]."""
    n, h, d = x.shape
    scale = sf.to(torch.float32).view(n, h, d // 16, 1) * global_scale
    enc = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    z = (x.to(torch.float32).view(n, h, d // 16, 16) * enc).clamp(-6, 6).view(n, h, d)
    return guided_e2m1(z)[1]


def post_kv_code_stats(acc, k_new, v_new, ksf, kp, vsf, vp, valid, scales) -> None:
    """Post-guidance KV code agreement with the rollout (under rollout scales)."""
    for name, new, sf, p in (("k", k_new, ksf, kp), ("v", v_new, vsf, vp)):
        got = _kv_codes_under_scale(new.detach()[valid], sf[valid], scales[name])
        st = code_stats(got, unpack_fp4_nibbles(p[valid]))
        acc.update({f"post_{name}_{a}": b for a, b in st.items()})


# -- probe hooks --------------------------------------------------------------------


def _probe_fc1(layer: int, x: torch.Tensor) -> None:
    rows = ta._ROWS_LOCAL
    x = x.reshape(-1, x.shape[-1])
    rec = {"site": "fc1", "layer": layer, "call": ta._CALL, "tokens": int(x.shape[0])}
    if rows is None or rows.shape[0] != x.shape[0]:
        rec["error"] = f"token mismatch rows={None if rows is None else rows.shape[0]}"
        _emit(rec)
        return
    dec = trace_layout.decode_fc1(ta._LAYOUT, rows, layer)
    v = dec.valid
    rec["valid_tokens"] = int(v.sum())
    if rec["valid_tokens"] == 0:
        _emit(rec)
        return
    xv = x[v]
    own = nvfp4_rowscaled_quantize_ref(xv)
    train_codes = unpack_fp4_nibbles(own.data)
    lsb = dec.lsb()[v]
    # Mantissa-bit disagreement of the unguided trainer codes: available for
    # every record mode and the quantity TRACE drives to zero.
    roll_lsb = unpack_bits(lsb)[:, : xv.shape[1]]
    rec["lsb_mismatch"] = float(((train_codes & 1).bool() != roll_lsb).float().mean())
    if dec.token_scale is not None:
        tok_roll = dec.token_scale[v]
        tok_train = own.row_amax / (6.0 * 448.0)
        sf_roll = dec.block_scale[v]
        rec["tok_scale_exact"] = float((tok_train == tok_roll).float().mean())
        rec["block_scale_exact"] = float(
            (own.decode_scale.view(torch.uint8) == sf_roll.view(torch.uint8))
            .float()
            .mean()
        )
    if dec.mode == "full":
        roll_codes = unpack_fp4_nibbles(dec.codes[v])
        qr = ta.dequant_codes(roll_codes)
        guided = unpack_fp4_nibbles(
            nvfp4_rowscaled_quantize_ref(xv, guide_lsb_packed=lsb).data
        )
        rec["rtn_own_scale"] = code_stats(train_codes, roll_codes)
        rec["trace_own_scale"] = code_stats(guided, roll_codes)
        rec["d_act_rtn_own"] = _rel(ta.dequant_codes(train_codes), qr)
        rec["d_act_trace_own"] = _rel(ta.dequant_codes(guided), qr)
        if dec.token_scale is not None:
            # Under the rollout's scales (the paper's normalisation): isolates
            # value differences from scale differences.
            kw = {"rollout_decode_scale": sf_roll, "rollout_token_scale": tok_roll}
            rtn_r = unpack_fp4_nibbles(nvfp4_rowscaled_quantize_ref(xv, **kw).data)
            tr_r = unpack_fp4_nibbles(
                nvfp4_rowscaled_quantize_ref(xv, guide_lsb_packed=lsb, **kw).data
            )
            rec["rtn_roll_scale"] = code_stats(rtn_r, roll_codes)
            rec["trace_roll_scale"] = code_stats(tr_r, roll_codes)
            rec["d_act_rtn"] = _rel(ta.dequant_codes(rtn_r), qr)
            rec["d_act_trace"] = _rel(ta.dequant_codes(tr_r), qr)
    _emit(rec)


def _probe_kv(layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
    rows = ta._ROWS_FULL
    hk_local, d = key.shape[-2], key.shape[-1]
    k = key.reshape(-1, hk_local, d)
    vv = value.reshape(-1, hk_local, d)
    rec = {"site": "kv", "layer": layer, "call": ta._CALL, "tokens": int(k.shape[0])}
    if rows is None or rows.shape[0] != k.shape[0]:
        rec["error"] = f"token mismatch rows={None if rows is None else rows.shape[0]}"
        _emit(rec)
        return
    dec = trace_layout.decode_kv(ta._LAYOUT, rows, layer)
    hs = slice(ta._tp_rank() * hk_local, (ta._tp_rank() + 1) * hk_local)
    v = dec.valid
    rec["valid_tokens"] = int(v.sum())
    if rec["valid_tokens"] == 0 or dec.mode != "full":
        _emit(rec)
        return
    scales = ta.kv_scales().get(layer, {})
    for name, x, sf, codes in (
        ("k", k, dec.k_scale, dec.k_codes),
        ("v", vv, dec.v_scale, dec.v_codes),
    ):
        gscale = scales.get(name, 1.0)
        x = x[v]
        sf = sf[v][:, hs]
        roll = unpack_fp4_nibbles(codes[v][:, hs])
        n, h, dd = x.shape
        own_sf = (
            x.abs().float().view(n, h, dd // 16, 16).amax(-1) / (6.0 * gscale)
        ).clamp(max=448.0)
        rec[f"{name}_block_scale_exact"] = float(
            (own_sf.to(torch.float8_e4m3fn).view(torch.uint8) == sf.view(torch.uint8))
            .float()
            .mean()
        )
        rtn = _kv_codes_under_scale(x, sf, gscale)
        scale = sf.to(torch.float32).view(n, h, dd // 16, 1) * gscale
        enc = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
        z = (
            (x.to(torch.float32).view(n, h, dd // 16, 16) * enc)
            .clamp(-6, 6)
            .view(n, h, dd)
        )
        g, _ = guided_e2m1(z, (roll & 1).bool())
        qr = ta.dequant_codes(roll)
        rec[f"{name}_rtn_roll_scale"] = code_stats(rtn, roll)
        rec[f"{name}_trace_roll_scale"] = code_stats(g, roll)
        rec[f"{name}_d_act_rtn"] = _rel(ta.dequant_codes(rtn), qr)
        rec[f"{name}_d_act_trace"] = _rel(ta.dequant_codes(g), qr)
    _emit(rec)


def ensure_probe_hooks(model: Any) -> None:
    if id(model) in _HOOKED:
        return
    _HOOKED.add(id(model))
    from megatron.core.transformer.attention import Attention
    from megatron.core.transformer.moe.moe_layer import MoELayer

    fc1_layers = set(ta._LAYOUT.fc1_layers)
    kv_layers = set(ta._LAYOUT.kv_layers)
    chunks = model if isinstance(model, (list, tuple)) else [model]
    n_hooks = 0
    for chunk in chunks:
        for module in chunk.modules():
            if getattr(module, "_nrl_trace_probe_hooked", False):
                continue
            layer = getattr(module, "layer_number", 0) - 1
            if isinstance(module, MoELayer) and layer in fc1_layers:

                def moe_hook(mod, args, kwargs, layer=layer):
                    if not torch.is_grad_enabled():
                        with torch.no_grad():
                            _probe_fc1(
                                layer, args[0] if args else kwargs["hidden_states"]
                            )

                module.register_forward_pre_hook(moe_hook, with_kwargs=True)
            elif isinstance(module, Attention) and layer in kv_layers:

                def attn_hook(mod, args, kwargs, layer=layer):
                    if not torch.is_grad_enabled():
                        with torch.no_grad():
                            _probe_kv(layer, args[1], args[2])

                module.core_attention.register_forward_pre_hook(
                    attn_hook, with_kwargs=True
                )
            else:
                continue
            module._nrl_trace_probe_hooked = True
            n_hooks += 1
    print(
        f"[trace] probe hooks installed: {n_hooks} (layout {ta._LAYOUT.to_json()})",
        flush=True,
    )


# -- NVFP4 KV amax calibration ------------------------------------------------------


def _ensure_kv_calib_hooks(model: Any) -> None:
    """Record per-layer max |q|, |k|, |v| (post-RoPE) in no-grad passes."""
    from megatron.core.transformer.attention import Attention

    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for module in chunk.modules():
            if not isinstance(module, Attention) or getattr(
                module, "_nrl_kv_calib", False
            ):
                continue
            module._nrl_kv_calib = True

            def hook(mod, args, kwargs, layer=module.layer_number - 1):
                if torch.is_grad_enabled():
                    return None
                d = _KV_AMAX.setdefault(layer, {"q": 0.0, "k": 0.0, "v": 0.0})
                for name, t in zip("qkv", args[:3]):
                    d[name] = max(d[name], float(t.detach().abs().amax()))
                return None

            module.core_attention.register_forward_pre_hook(hook, with_kwargs=True)


def _dump_kv_calib() -> None:
    d = os.environ[KV_CALIB_DIR_ENV]
    if not _KV_AMAX:
        return
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"rank{_rank()}.json"), "w") as f:
        json.dump({str(k): v for k, v in sorted(_KV_AMAX.items())}, f)
