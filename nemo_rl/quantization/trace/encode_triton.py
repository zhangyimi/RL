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
"""One-kernel rollout encoder for scale-free fc1 records.

Writes ``[marker x4 | code bits]`` for every token straight into the
routed-experts transit buffer, replacing the chain of small torch ops in
:func:`layout.encode_fc1` (which costs ~10 launches per layer per decode
step). Byte-identical to ``encode_fc1`` with ``ship_scales=False`` (padding
bytes past the payload are left untouched; decoders never read them).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from nemo_rl.quantization.trace.layout import MARKER, MARKER_BYTES

_WIDTH = {"lsb": 1, "b2": 2, "full": 4}


@triton.jit
def _encode_kernel(
    src_ptr, src_stride, dst_ptr, dst_stride, n_out,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr, MARKER_V: tl.constexpr,
):
    t = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    m = j < n_out
    src = src_ptr + t.to(tl.int64) * src_stride
    if WIDTH == 4:
        b = tl.load(src + j, mask=m, other=0).to(tl.int32)
    else:
        # Output byte j holds 8/WIDTH elements = 4/WIDTH source bytes.
        per = 4 // WIDTH
        b = tl.zeros([BLOCK], dtype=tl.int32)
        for q in tl.static_range(4 // WIDTH):
            s = tl.load(src + j * per + q, mask=m, other=0).to(tl.int32)
            lo = s & ((1 << WIDTH) - 1)
            hi = (s >> 4) & ((1 << WIDTH) - 1)
            b = b | (lo << (2 * q * WIDTH)) | (hi << ((2 * q + 1) * WIDTH))
    dst = dst_ptr + t.to(tl.int64) * dst_stride
    tl.store(dst + 4 + j, b.to(tl.uint8), mask=m)
    k = tl.arange(0, 4)
    tl.store(dst + k, tl.full([4], MARKER_V, tl.uint8))


def encode_fc1_into(buf: torch.Tensor, byte_offset: int, packed_codes: torch.Tensor, mode: str, hidden: int) -> None:
    """Encode ``packed_codes`` ``[n, >=H/2]`` uint8 into ``buf`` (uint8, ``[T, R, topk]``).

    The record of token ``t`` starts at ``buf[t].view(-1)[byte_offset]``.
    """
    if buf.dtype != torch.uint8 or not buf.is_contiguous():
        raise ValueError("transit buffer must be contiguous uint8")
    if MARKER_BYTES != 4:
        raise AssertionError("kernel writes a 4-byte marker")
    n = packed_codes.shape[0]
    width = _WIDTH[mode]
    n_out = hidden * width // 8
    if n:
        flat = buf.view(buf.shape[0], -1)
        _encode_kernel[(n,)](
            packed_codes, packed_codes.stride(0), flat[:, byte_offset:], flat.stride(0), n_out,
            WIDTH=width, BLOCK=triton.next_power_of_2(n_out), MARKER_V=MARKER,
        )


@triton.jit
def _bits_from_nibbles(src, j, m, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    """Output byte j of a WIDTH-bit field packing, from packed nibbles at ``src``."""
    if WIDTH == 4:
        return tl.load(src + j, mask=m, other=0).to(tl.int32)
    per = 4 // WIDTH
    b = tl.zeros([BLOCK], dtype=tl.int32)
    for q in tl.static_range(4 // WIDTH):
        s = tl.load(src + j * per + q, mask=m, other=0).to(tl.int32)
        b = b | ((s & ((1 << WIDTH) - 1)) << (2 * q * WIDTH)) | (((s >> 4) & ((1 << WIDTH) - 1)) << ((2 * q + 1) * WIDTH))
    return b


@triton.jit
def _kv_capture_kernel(
    cache_ptr, s_blk, s1d, s2d, s1s, s2s, side_v_off, data_per_side, slot_ptr, block_size,
    dst_ptr, dst_stride,
    HK: tl.constexpr, D: tl.constexpr, WIDTH: tl.constexpr, MARKER_V: tl.constexpr,
):
    # Page layout per side (vLLM nvfp4_split_data_scale): all heads' data
    # [h, t, D/2] (strides s1d, s2d) then all scales [h, t, D/16] (s1s, s2s).
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    valid = slot >= 0
    slot = tl.maximum(slot, 0)
    blk = slot // block_size
    off = slot % block_size
    S: tl.constexpr = D // 16
    NB: tl.constexpr = D * WIDTH // 8
    dst = dst_ptr + t.to(tl.int64) * dst_stride
    k4 = tl.arange(0, 4)
    tl.store(dst + k4, tl.where(valid, MARKER_V, 0).to(tl.uint8) + tl.zeros([4], tl.uint8))
    js = tl.arange(0, S)
    jb = tl.arange(0, NB)
    side_bytes = HK * S + HK * NB
    for side in tl.static_range(2):
        sbase = cache_ptr + blk * s_blk + side * side_v_off
        for h in tl.static_range(HK):
            row = sbase + h * s1d + off * s2d
            scl = sbase + data_per_side + h * s1s
            if side == 0:
                sf = tl.load(scl + off * s2s + js, mask=valid, other=0)
            else:
                # V scales: TRT-LLM 4-token interleave within the (page, head) [N, S] tile.
                flat = (off // 4) * 4 * S + js * 4 + off % 4
                sf = tl.load(scl + (flat // S) * s2s + flat % S, mask=valid, other=0)
            base = dst + 4 + side * side_bytes
            tl.store(base + h * S + js, sf)
            b = _bits_from_nibbles(row, jb, valid & (jb < NB), WIDTH, NB)
            tl.store(base + HK * S + h * NB + jb, b.to(tl.uint8))


def capture_kv_into(buf: torch.Tensor, byte_offset: int, kv_cache: torch.Tensor, slot_mapping: torch.Tensor,
                    num_kv_heads: int, head_dim: int, mode: str) -> None:
    """``layout.encode_kv`` records for this step's tokens, read straight from the
    NVFP4 paged cache ``(num_blocks, 2 * Hkv, block_size, D/2 + D/16)`` (K heads
    first), addressed exactly like ``vllm.utils.torch_utils.nvfp4_split_data_scale``."""
    n = slot_mapping.shape[0]
    if not n:
        return
    c = kv_cache.view(torch.uint8)
    full = head_dim // 2 + head_dim // 16
    if c.shape[-1] != full or c.stride(-1) != 1:
        raise ValueError(f"unexpected NVFP4 KV cache layout {tuple(c.shape)} / {c.stride()}")
    data_dim, scale_dim = full * 8 // 9, full - full * 8 // 9
    s1, s2 = c.stride(1), c.stride(2)
    flat = buf.view(buf.shape[0], -1)
    _kv_capture_kernel[(n,)](
        c, c.stride(0), s1 * data_dim // full, s2 * data_dim // full,
        s1 * scale_dim // full, s2 * scale_dim // full,
        num_kv_heads * s1, num_kv_heads * c.shape[2] * data_dim,
        slot_mapping, c.shape[2],
        flat[:, byte_offset:], flat.stride(0),
        HK=num_kv_heads, D=head_dim, WIDTH=_WIDTH[mode], MARKER_V=MARKER,
    )
