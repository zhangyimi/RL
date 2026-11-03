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
"""NVFP4 fake quantization for TRACE QAT (Triton, one program per row).

``nvfp4_fakequant_fast`` returns ``dequant(quant(x))`` for ``x`` [M, K]:

* scales: TE row-scaled NVFP4 (per-token global scale from the row amax,
  E4M3 per 16 elements), or one per-tensor global scale (``global_encode``,
  weights; same arithmetic as vLLM's ``scaled_fp4_quant`` refit quantizer);
* rounding: round-to-nearest, except on guided rows (``guide_valid``), which
  quantize under the rollout's own per-token and block scales (the paper's
  normalisation) and pick, of the two E2M1 neighbours, the one whose mantissa
  bit matches the rollout's (TRACE, b=1). Guidance bits are packed
  little-endian, element ``i`` at bit ``i % 8`` of byte ``i // 8``.

Bit-identical to the torch reference in the unit tests.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 16


@triton.jit
def _mag_of(c):
    # E2M1 magnitude of a 3-bit magnitude code.
    v = c.to(tl.float32) * 0.5
    v = tl.where(c == 5, 3.0, v)
    v = tl.where(c == 6, 4.0, v)
    v = tl.where(c == 7, 6.0, v)
    return v


@triton.jit
def _rcp_approx(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=r,r",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fakequant_kernel(
    x_ptr,
    out_ptr,
    guide_ptr,
    valid_ptr,
    flips_ptr,
    genc_ptr,
    rtok_ptr,
    rsf_ptr,
    K,
    guide_row_bytes,
    BLOCK_K: tl.constexpr,
    HAS_GUIDE: tl.constexpr,
    HAS_GENC: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    inb = offs < K
    base = row.to(tl.int64) * K
    x_raw = tl.load(x_ptr + base + offs, mask=inb, other=0.0)
    x = x_raw.to(tl.float32)
    ax = tl.abs(x)
    fp32_max = 3.4028234663852886e38

    if HAS_GENC:
        g_enc = tl.load(genc_ptr).to(tl.float32)
    else:
        # TE row-scaled NVFP4 scales (same op order as the torch reference).
        row_amax = tl.max(ax, axis=0)
        g_enc = tl.minimum(
            tl.math.div_rn(tl.zeros_like(row_amax) + 2688.0, row_amax), fp32_max
        )
        g_enc = tl.where(g_enc == 0.0, 1.0, g_enc)
    g_dec = tl.math.div_rn(tl.zeros_like(g_enc) + 1.0, g_enc)
    ax2 = tl.reshape(ax, (BLOCK_K // 16, 16))
    vmax = tl.max(ax2, axis=1)
    inv6 = 0.1666666716337204  # fp32(1/6) == torch.reciprocal(6.0)
    if HAS_GENC:
        # vLLM ``scaled_fp4_quant`` arithmetic: approximate reciprocals decide
        # exact E2M1 ties the same way.
        ones = tl.zeros_like(vmax) + 1.0
        dec = (ones * g_enc) * (vmax * _rcp_approx(ones * 6.0))
        dec = tl.minimum(dec, 448.0).to(tl.float8e4nv).to(tl.float32)
        enc = tl.where(dec == 0.0, 0.0, _rcp_approx(dec * _rcp_approx(ones * g_enc)))
        scale = dec * g_dec
    else:
        dec = tl.minimum(vmax * (g_enc * inv6), fp32_max)
        dec = tl.minimum(tl.maximum(dec, -448.0), 448.0)
        dec = dec.to(tl.float8e4nv).to(tl.float32)
        scale = dec * g_dec
        enc = tl.minimum(tl.math.div_rn(tl.zeros_like(scale) + 1.0, scale), fp32_max)
    if HAS_GUIDE:
        valid = tl.load(valid_ptr + row).to(tl.int1)
        # Guided rows quantize under the rollout's per-token decode scale and
        # E4M3 block scales.
        boffs = tl.arange(0, BLOCK_K // 16)
        rdec = tl.load(
            rsf_ptr + row.to(tl.int64) * (K // 16) + boffs,
            mask=boffs < K // 16,
            other=0,
        )
        rdec = rdec.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        rscale = rdec * tl.load(rtok_ptr + row)
        renc = tl.minimum(tl.math.div_rn(tl.zeros_like(rscale) + 1.0, rscale), fp32_max)
        scale = tl.where(valid, rscale, scale)
        enc = tl.where(valid, renc, enc)
    else:
        valid = row < 0
    scale_e = tl.reshape(
        tl.broadcast_to(scale[:, None], (BLOCK_K // 16, 16)), (BLOCK_K,)
    )
    enc_e = tl.reshape(tl.broadcast_to(enc[:, None], (BLOCK_K // 16, 16)), (BLOCK_K,))

    s = tl.minimum(tl.maximum(x * enc_e, -6.0), 6.0)
    mag = tl.abs(s)
    neg = s < 0
    lower = (
        (mag >= 0.5).to(tl.int32)
        + (mag >= 1.0).to(tl.int32)
        + (mag >= 1.5).to(tl.int32)
        + (mag >= 2.0).to(tl.int32)
        + (mag >= 3.0).to(tl.int32)
        + (mag >= 4.0).to(tl.int32)
        + (mag >= 6.0).to(tl.int32)
    )
    upper = tl.minimum(lower + 1, 7)
    lo_v = _mag_of(lower)
    hi_v = _mag_of(upper)
    on_grid = lo_v == mag
    mid = (lo_v + hi_v) * 0.5
    tie_up = (lower & 1) == 1
    rtn = tl.where(on_grid | (mag < mid) | ((mag == mid) & ~tie_up), lower, upper)

    if HAS_GUIDE:
        gbyte = tl.load(
            guide_ptr + row.to(tl.int64) * guide_row_bytes + offs // 8,
            mask=inb,
            other=0,
        ).to(tl.int32)
    else:
        gbyte = tl.zeros_like(offs)
    g = (gbyte >> (offs % 8)) & 1
    # Bracket neighbour carrying the rollout's mantissa bit (on-grid values keep RTN).
    guided = tl.where((lower & 1) == g, lower, upper)
    guided = tl.where(on_grid, rtn, guided)

    rtn_code = rtn | (neg.to(tl.int32) << 3)
    g_code = guided | (neg.to(tl.int32) << 3)
    target = tl.where(valid, g_code, rtn_code)
    flip = (target != rtn_code) & inb
    tval = _mag_of(target & 7)
    tval = tl.where((target & 8) != 0, -tval, tval)
    out = (tval * scale_e).to(x_raw.dtype)
    out = tl.where((target & 7) == 0, 0.0, out).to(x_raw.dtype)
    tl.store(out_ptr + base + offs, out, mask=inb)
    tl.atomic_add(flips_ptr, tl.sum(flip.to(tl.int32), axis=0))


def nvfp4_fakequant_fast(
    x: torch.Tensor,
    guide_lsb: torch.Tensor | None = None,
    guide_valid: torch.Tensor | None = None,
    *,
    global_encode: torch.Tensor | None = None,
    rollout_token_scale: torch.Tensor | None = None,
    rollout_block_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """NVFP4 quant->dequant of ``x`` [M, K], optionally TRACE-guided.

    ``guide_lsb``: [M, K/8] packed rollout mantissa bits; ``guide_valid``: [M]
    rows that carry guidance, quantized under ``rollout_token_scale`` ([M]
    fp32 per-token decode) and ``rollout_block_scale`` ([M, K/16] E4M3).
    ``global_encode``: fp32 scalar ``448 * 6 / amax`` for per-tensor scaling.
    Returns the result and ``{"flipped_t": <device count of non-RTN codes>}``.
    """
    m, k = x.shape
    if k % BLOCK:
        raise ValueError(f"K={k} not a multiple of {BLOCK}")
    x = x.contiguous()
    out = torch.empty_like(x)
    flips = torch.zeros(1, dtype=torch.int32, device=x.device)
    has_guide = guide_lsb is not None
    if has_guide != (rollout_token_scale is not None) or has_guide != (
        rollout_block_scale is not None
    ):
        raise ValueError("guidance needs both rollout scales (and vice versa)")
    if m:
        _fakequant_kernel[(m,)](
            x,
            out,
            guide_lsb.contiguous() if has_guide else x,
            guide_valid.to(torch.int8).contiguous() if has_guide else x,
            flips,
            global_encode.to(torch.float32).reshape(1)
            if global_encode is not None
            else flips,
            rollout_token_scale.to(torch.float32).contiguous() if has_guide else flips,
            rollout_block_scale.contiguous().view(torch.uint8) if has_guide else flips,
            k,
            guide_lsb.shape[1] if has_guide else 0,
            BLOCK_K=triton.next_power_of_2(k),
            HAS_GUIDE=has_guide,
            HAS_GENC=global_encode is not None,
        )
    return out, {"flipped_t": flips}
