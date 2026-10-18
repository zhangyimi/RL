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
