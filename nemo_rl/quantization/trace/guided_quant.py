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
"""Rollout-guided E2M1 rounding (TRACE, arXiv 2610.07767), reference version.

TRACE replaces round-to-nearest with: of the two E2M1 codewords bracketing the
scaled training value, take the one closer to the rollout codeword. Adjacent
E2M1 magnitude codes alternate their low bit, so with the 1-bit (mantissa)
guidance the paper ships, "closer to the rollout codeword" is exactly "the
bracket neighbour whose low bit equals the rollout low bit". Values that sit
on a codeword (including the +-6 clip) keep their round-to-nearest code.

The row-scaled NVFP4 math mirrors TE 2.18
``NVFP4QuantizerRef._quantize_blockwise_reference`` (1x16 tiles, no RHT,
no 2D, no stochastic rounding, ``row_scaled_nvfp4=True``) so that
``guide=None`` is bit-identical to TE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from nemo_rl.quantization.trace.codec import pack_fp4_nibbles, unpack_bits

FP4_MAX = 6.0
E4M3_MAX = 448.0
BLOCK = 16
# E2M1 magnitudes indexed by the 3-bit magnitude code.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass
class GuidedRoundStats:
    """Element counts for one guided rounding call (padding rows excluded)."""

    elements: int
    guided_elements: int
    flipped: int  # elements whose code differs from round-to-nearest


def guided_e2m1(
    scaled: torch.Tensor,
    guide_lsb: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round ``scaled`` (fp32, already clipped to [-6, 6]) to 4-bit E2M1 codes.

    Args:
        scaled: fp32 tensor of scaled values.
        guide_lsb: bool tensor broadcastable to ``scaled``; rollout mantissa
            bit per element, or ``None`` for plain round-to-nearest-even.

    Returns:
        ``(codes, rtn_codes)`` as uint8 4-bit codes (sign in bit 3).
    """
    mags = torch.tensor(E2M1_MAGNITUDES, device=scaled.device, dtype=torch.float32)
    mag = scaled.abs()
    # Largest magnitude code with value <= mag.
    lower = (torch.bucketize(mag, mags, right=True) - 1).clamp_(0, 7)
    upper = (lower + 1).clamp_(max=7)
    lo_val = mags[lower]
    hi_val = mags[upper]
    on_grid = lo_val == mag
    mid = (lo_val + hi_val) * 0.5
    # Round half to even code (TE cast_to_fp4x2: 0.25->0, 0.75->2, 1.25->2, ...).
    tie_to_upper = (lower & 1) == 1
    rtn = torch.where(
        on_grid | (mag < mid) | ((mag == mid) & ~tie_to_upper), lower, upper
    )
    if guide_lsb is None:
        mag_code = rtn
    else:
        guided = torch.where((lower & 1) == guide_lsb.to(lower.dtype), lower, upper)
        mag_code = torch.where(on_grid, rtn, guided)
    sign = (scaled < 0).to(torch.uint8) << 3
    return mag_code.to(torch.uint8) | sign, rtn.to(torch.uint8) | sign


def nearest_code_with_low_bits(scaled: torch.Tensor, low: torch.Tensor, bits: int) -> torch.Tensor:
    """Magnitude code (0..7) nearest to ``|scaled|`` whose low ``bits`` equal ``low``.

    Ties go to the smaller code. This is b>=2 TRACE guidance: with two bits an
    off-by-two rollout code is still recoverable. Returns int64 magnitude codes.
    """
    mags = torch.tensor(E2M1_MAGNITUDES, device=scaled.device, dtype=torch.float32)
    mask = (1 << bits) - 1
    codes8 = torch.arange(8, device=scaled.device)
    ok = (codes8 & mask) == (low.to(torch.int64) & mask).unsqueeze(-1)
    dist = (mags - scaled.abs().unsqueeze(-1)).abs()
    return torch.where(ok, dist, torch.full_like(dist, float("inf"))).argmin(-1)


@dataclass
class NVFP4RowScaled:
    """Row-scaled NVFP4 operand in TE's rowwise layout (unswizzled)."""

    data: torch.Tensor  # [M, K/2] uint8
    decode_scale: torch.Tensor  # [M, K/16] float8_e4m3fn
    row_amax: torch.Tensor  # [M] fp32, TE ``_amax_rowwise`` semantics
    stats: GuidedRoundStats
    global_decode: Optional[torch.Tensor] = None  # [M] fp32, exactly as used


def nvfp4_rowscaled_quantize_ref(
    x: torch.Tensor,
    *,
    guide_lsb_packed: Optional[torch.Tensor] = None,
    guide_valid: Optional[torch.Tensor] = None,
    rollout_decode_scale: Optional[torch.Tensor] = None,
    rollout_token_scale: Optional[torch.Tensor] = None,
) -> NVFP4RowScaled:
    """Quantize ``x`` [M, K] to row-scaled NVFP4, optionally rollout-guided.

    Args:
        x: [M, K] bf16/fp32 activations (K % 16 == 0).
        guide_lsb_packed: [M, K/8] uint8 rollout mantissa bits.
        guide_valid: [M] bool; rows without rollout guidance fall back to
            round-to-nearest (prefix-cache misses, padding).
        rollout_decode_scale: [M, K/16] float8_e4m3fn rollout block scales.
            When given (with ``rollout_token_scale``), the training value is
            normalised with the rollout scale as in the paper.
        rollout_token_scale: [M] fp32 rollout per-token global decode scale
            (FlashInfer per-token NVFP4: ``amax / (6 * 448)``).
    """
    if x.ndim != 2 or x.shape[1] % BLOCK:
        raise ValueError(f"expected [M, K] with K % {BLOCK} == 0, got {tuple(x.shape)}")
    if (rollout_decode_scale is None) != (rollout_token_scale is None):
        raise ValueError("rollout_decode_scale and rollout_token_scale go together")
    m, k = x.shape
    dev = x.device
    fp32_max = torch.tensor(torch.finfo(torch.float32).max, device=dev)
    xb = x.view(m, k // BLOCK, BLOCK)

    if rollout_token_scale is None:
        # Keep TE's exact op order: ``scalar / tensor`` lowers to
        # reciprocal-then-multiply in torch, which is not bit-identical.
        fp4_max = torch.tensor(FP4_MAX, device=dev, dtype=torch.float32)
        e4m3_max = torch.tensor(E4M3_MAX, device=dev, dtype=torch.float32)
        row_amax = x.abs().amax(dim=1).to(torch.float32)
        global_encode = torch.minimum(
            torch.div(e4m3_max * fp4_max, row_amax.view(m, 1, 1)), fp32_max
        )
        global_encode = torch.where(
            global_encode == 0.0, torch.ones_like(global_encode), global_encode
        )
        global_decode = torch.div(1.0, global_encode)
        vec_max = xb.abs().amax(dim=-1, keepdim=True).to(torch.float32)
        decode = torch.minimum(vec_max * (global_encode * torch.reciprocal(fp4_max)), fp32_max)
        decode = decode.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    else:
        global_decode = rollout_token_scale.to(torch.float32).view(m, 1, 1)
        row_amax = rollout_token_scale.to(torch.float32) * (E4M3_MAX * FP4_MAX)
        decode = rollout_decode_scale.view(m, k // BLOCK, 1)

    encode = torch.minimum(torch.div(1.0, decode.to(torch.float32) * global_decode), fp32_max)
    scaled = (xb.to(torch.float32) * encode).clamp(-FP4_MAX, FP4_MAX).view(m, k)

    guide = None
    if guide_lsb_packed is not None:
        guide = unpack_bits(guide_lsb_packed)[:, :k]
    codes, rtn = guided_e2m1(scaled, guide)
    if guide is not None and guide_valid is not None:
        codes = torch.where(guide_valid.view(m, 1), codes, rtn)

    n_guided = 0 if guide is None else (
        k * (m if guide_valid is None else int(guide_valid.sum()))
    )
    stats = GuidedRoundStats(
        elements=m * k,
        guided_elements=n_guided,
        flipped=int((codes != rtn).sum()),
    )
    return NVFP4RowScaled(
        data=pack_fp4_nibbles(codes),
        decode_scale=decode.view(m, k // BLOCK),
        row_amax=row_amax,
        stats=stats,
        global_decode=global_decode.reshape(m),
    )


def dequantize_rowscaled_ref(q: NVFP4RowScaled, dtype=torch.float32) -> torch.Tensor:
    """Dequantize a :class:`NVFP4RowScaled` (for tests and STE fake-quant)."""
    from nemo_rl.quantization.trace.codec import unpack_fp4_nibbles

    codes = unpack_fp4_nibbles(q.data)
    mags = torch.tensor(E2M1_MAGNITUDES, device=codes.device, dtype=torch.float32)
    vals = mags[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -vals, vals)
    m, k = vals.shape
    global_decode = (q.row_amax / (E4M3_MAX * FP4_MAX)).view(m, 1, 1)
    scale = q.decode_scale.to(torch.float32).view(m, k // BLOCK, 1) * global_decode
    return (vals.view(m, k // BLOCK, BLOCK) * scale).view(m, k).to(dtype)


def trace_nudge_rowscaled(
    x: torch.Tensor,
    guide_lsb_packed: Optional[torch.Tensor],
    guide_valid: torch.Tensor,
    guide_codes_packed: Optional[torch.Tensor] = None,
    guide_bits: int = 4,
) -> tuple[torch.Tensor, dict]:
    """Rewrite ``x`` so TE's own row-scaled RTN quantization yields TRACE codes.

    TE quantizes routed-expert inputs after the token dispatcher, where the
    rollout guidance is not available. Instead of plumbing guidance through
    dispatch and TE, each element whose TRACE code differs from its RTN code
    is replaced by the centre of its TRACE codeword (far from any rounding
    boundary). Block maxima are never touched, so the per-row amax and every
    E4M3 block scale are unchanged and TE reproduces exactly the guided codes.
    Use as ``x + (x_new - x).detach()`` for the straight-through gradient.

    With ``guide_codes_packed`` (full rollout codes, ``[M, K/2]``) the target
    is the rollout code itself rather than the bracket neighbour with the
    rollout mantissa bit (the "copy" upper bound for b=4), capped at the
    block maximum's code so scales stay unchanged. ``guide_bits`` < 4 keeps
    only the low bits of each rollout magnitude code and picks the nearest
    magnitude code carrying them (emulates b=2/3 guidance; sign from training).

    Returns ``(x_new, stats)``; ``stats["unreproduced"]`` counts elements
    whose re-quantization of ``x_new`` still differs from the TRACE codes
    (expected 0).
    """
    m, k = x.shape
    q = nvfp4_rowscaled_quantize_ref(x)  # TE-identical scales and RTN codes
    global_decode = q.global_decode.view(m, 1, 1)
    scale = q.decode_scale.to(torch.float32).view(m, k // BLOCK, 1) * global_decode
    xb = x.view(m, k // BLOCK, BLOCK)
    is_block_max = xb.abs() == xb.abs().amax(dim=-1, keepdim=True)

    encode = torch.minimum(
        torch.div(1.0, scale), torch.tensor(torch.finfo(torch.float32).max, device=x.device)
    )
    scaled = (xb.to(torch.float32) * encode).clamp(-FP4_MAX, FP4_MAX).view(m, k)
    if guide_codes_packed is None:
        guide = unpack_bits(guide_lsb_packed)[:, :k]
        guided, rtn = guided_e2m1(scaled, guide)
    else:
        from nemo_rl.quantization.trace.codec import unpack_fp4_nibbles

        _, rtn = guided_e2m1(scaled)
        guided = unpack_fp4_nibbles(guide_codes_packed)[:, :k]
        if guide_bits < 4:
            mag_code = nearest_code_with_low_bits(scaled, guided, guide_bits).to(torch.uint8)
            guided = mag_code | ((scaled < 0).to(torch.uint8) << 3)
        # Never exceed the block maximum's code: that would move the amax/scale.
        cap = (rtn & 7).view(m, k // BLOCK, BLOCK).amax(dim=-1, keepdim=True)
        mag = torch.minimum(guided & 7, cap.expand(m, k // BLOCK, BLOCK).reshape(m, k))
        guided = mag | (guided & 8)
    keep = (~guide_valid.view(m, 1)) | is_block_max.view(m, k)
    target = torch.where(keep, rtn, guided)
    flip = target != rtn

    mags = torch.tensor(E2M1_MAGNITUDES, device=x.device, dtype=torch.float32)
    tval = mags[(target & 7).long()]
    tval = torch.where((target & 8) != 0, -tval, tval)
    centre = (tval.view(m, k // BLOCK, BLOCK) * scale).view(m, k).to(x.dtype)
    # A zero target becomes +0: TE encodes -0.0 with the sign bit and the
    # reference cast does not; both mean 0, so pick the unambiguous one.
    centre = torch.where((target & 7) == 0, torch.zeros_like(centre), centre)
    x_new = torch.where(flip, centre, x)

    check = nvfp4_rowscaled_quantize_ref(x_new)
    from nemo_rl.quantization.trace.codec import unpack_fp4_nibbles

    # Signed zeros compare equal numerically; ignore code 0 vs 8.
    got = unpack_fp4_nibbles(check.data)
    unrep = (got != target) & ~(((got & 7) == 0) & ((target & 7) == 0))
    stats = {
        "elements": m * k,
        "guided_tokens": int(guide_valid.sum()),
        "flipped": int(flip.sum()),
        "unreproduced": int(unrep.sum()),
        "scale_changed": int((check.decode_scale.view(torch.uint8) != q.decode_scale.view(torch.uint8)).sum()),
    }
    return x_new, stats
