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
"""Fused Triton version of :func:`guided_quant.trace_nudge_rowscaled`.

One program per row: TE row-scaled NVFP4 scales, round-to-nearest code,
guided target code and the codeword-centre rewrite in a single pass, with no
re-quantization check. Bit-identical to the torch reference (unit-tested).

Guidance is a packed bit field of ``width`` bits per element (element ``i`` at
bit ``(i * width) % 8`` of byte ``(i * width) // 8``):

* ``width=1``: rollout mantissa bit, bracket-neighbour rule (b=1 TRACE).
* ``width=2``: two low code bits, nearest magnitude code carrying them (b=2).
* ``width=4``: packed E2M1 nibbles; ``match_bits`` 4 copies the code (sign
  included), 2/3 emulate b=2/b=3 from full codes.
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
def _nudge_kernel(
    x_ptr, out_ptr, guide_ptr, valid_ptr, flips_ptr,
    K, guide_row_bytes,
    BLOCK_K: tl.constexpr, WIDTH: tl.constexpr, MATCH: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    inb = offs < K
    x_raw = tl.load(x_ptr + row * K + offs, mask=inb, other=0.0)
    x = x_raw.to(tl.float32)
    ax = tl.abs(x)
    fp32_max = 3.4028234663852886e38

    # TE row-scaled NVFP4 scales (same op order as the torch reference).
    row_amax = tl.max(ax, axis=0)
    g_enc = tl.minimum(tl.math.div_rn(tl.zeros_like(row_amax) + 2688.0, row_amax), fp32_max)
    g_enc = tl.where(g_enc == 0.0, 1.0, g_enc)
    g_dec = tl.math.div_rn(tl.zeros_like(g_enc) + 1.0, g_enc)
    ax2 = tl.reshape(ax, (BLOCK_K // 16, 16))
    vmax = tl.max(ax2, axis=1)
    inv6 = 0.1666666716337204  # fp32(1/6) == torch.reciprocal(6.0)
    dec = tl.minimum(vmax * (g_enc * inv6), fp32_max)
    dec = tl.minimum(tl.maximum(dec, -448.0), 448.0)
    dec = dec.to(tl.float8e4nv).to(tl.float32)
    scale = dec * g_dec
    enc = tl.minimum(tl.math.div_rn(tl.zeros_like(scale) + 1.0, scale), fp32_max)
    is_max2 = ax2 == vmax[:, None]
    scale_e = tl.reshape(tl.broadcast_to(scale[:, None], (BLOCK_K // 16, 16)), (BLOCK_K,))
    enc_e = tl.reshape(tl.broadcast_to(enc[:, None], (BLOCK_K // 16, 16)), (BLOCK_K,))
    is_max = tl.reshape(is_max2, (BLOCK_K,))

    s = tl.minimum(tl.maximum(x * enc_e, -6.0), 6.0)
    mag = tl.abs(s)
    neg = s < 0
    lower = ((mag >= 0.5).to(tl.int32) + (mag >= 1.0).to(tl.int32) + (mag >= 1.5).to(tl.int32)
             + (mag >= 2.0).to(tl.int32) + (mag >= 3.0).to(tl.int32) + (mag >= 4.0).to(tl.int32)
             + (mag >= 6.0).to(tl.int32))
    upper = tl.minimum(lower + 1, 7)
    lo_v = _mag_of(lower)
    hi_v = _mag_of(upper)
    on_grid = lo_v == mag
    mid = (lo_v + hi_v) * 0.5
    tie_up = (lower & 1) == 1
    rtn = tl.where(on_grid | (mag < mid) | ((mag == mid) & ~tie_up), lower, upper)

    bit = offs * WIDTH
    gbyte = tl.load(guide_ptr + row * guide_row_bytes + bit // 8, mask=inb, other=0).to(tl.int32)
    g = (gbyte >> (bit % 8)) & ((1 << WIDTH) - 1)

    if WIDTH == 1:
        guided = tl.where((lower & 1) == g, lower, upper)
        guided = tl.where(on_grid, rtn, guided)
        tsign = neg
    else:
        if MATCH == 4:
            guided = g & 7
            tsign = (g & 8) != 0
        else:
            mask = (1 << MATCH) - 1
            best = tl.zeros_like(lower) + 0
            best_d = tl.zeros_like(mag) + 1e30
            for c in tl.static_range(8):
                ok = (c & mask) == (g & mask)
                d = tl.abs(_mag_of(tl.zeros_like(lower) + c) - mag)
                take = ok & (d < best_d)
                best = tl.where(take, c, best)
                best_d = tl.where(take, d, best_d)
            guided = best
            tsign = neg
        # Never exceed the block maximum's code (scales must not move).
        cap2 = tl.max(tl.reshape(rtn, (BLOCK_K // 16, 16)), axis=1)
        cap = tl.reshape(tl.broadcast_to(cap2[:, None], (BLOCK_K // 16, 16)), (BLOCK_K,))
        guided = tl.minimum(guided, cap)

    valid = tl.load(valid_ptr + row).to(tl.int1)
    keep = (~valid) | is_max
    rtn_code = rtn | (neg.to(tl.int32) << 3)
    g_code = guided | (tsign.to(tl.int32) << 3)
    target = tl.where(keep, rtn_code, g_code)
    flip = (target != rtn_code) & inb
    tval = _mag_of(target & 7)
    tval = tl.where((target & 8) != 0, -tval, tval)
    centre = (tval * scale_e).to(x_raw.dtype)
    centre = tl.where((target & 7) == 0, 0.0, centre).to(x_raw.dtype)
    out = tl.where(flip, centre, x_raw)
    tl.store(out_ptr + row * K + offs, out, mask=inb)
    tl.atomic_add(flips_ptr, tl.sum(flip.to(tl.int32), axis=0))


def trace_nudge_rowscaled_fast(
    x: torch.Tensor,
    guide_packed: torch.Tensor,
    guide_valid: torch.Tensor,
    *,
    width: int,
    match_bits: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Fused nudge; see the module docstring for ``width``/``match_bits``."""
    m, k = x.shape
    if k % BLOCK:
        raise ValueError(f"K={k} not a multiple of {BLOCK}")
    if width not in (1, 2, 4):
        raise ValueError(f"width must be 1, 2 or 4, got {width}")
    match = width if match_bits is None else match_bits
    x = x.contiguous()
    out = torch.empty_like(x)
    flips = torch.zeros(1, dtype=torch.int32, device=x.device)
    if m:
        _nudge_kernel[(m,)](
            x, out, guide_packed.contiguous(), guide_valid.to(torch.int8).contiguous(), flips,
            k, guide_packed.shape[1],
            BLOCK_K=triton.next_power_of_2(k), WIDTH=width, MATCH=match,
        )
    # Counters stay on device; callers that log them pay the sync.
    return out, {"flipped_t": flips}
