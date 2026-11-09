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
"""One-kernel fc1 record encoder for the rollout capture.

Writes the same bytes as ``layout.encode_fc1`` followed by the buffer write,
``[marker | fp32 token scale | E4M3 block scales | mantissa-bit mask | 0 pad]``,
straight into the token's record. The torch version launches ~13 small kernels
per MoE layer; this launches one.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from nemo_rl.quantization.trace.layout import MARKER, MARKER_BYTES


@triton.jit
def _encode_fc1_kernel(
    out_ptr,
    out_row_stride,
    tok_ptr,
    sf_ptr,
    sf_row_stride,
    code_ptr,
    code_row_stride,
    SF_BYTES: tl.constexpr,
    LSB_BYTES: tl.constexpr,
    REC_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
    MARK: tl.constexpr,
    MARK_BYTES: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, BLOCK)
    sf0 = MARK_BYTES + 4
    lsb0 = sf0 + SF_BYTES
    end = lsb0 + LSB_BYTES

    is_tok = (j >= MARK_BYTES) & (j < sf0)
    tok = tl.load(tok_ptr + t * 4 + (j - MARK_BYTES), mask=is_tok, other=0)

    is_sf = (j >= sf0) & (j < lsb0)
    sf = tl.load(sf_ptr + t * sf_row_stride + (j - sf0), mask=is_sf, other=0)

    # LSB byte k packs the mantissa bit (bit 0 of each nibble) of elements
    # 8k..8k+7 = packed bytes 4k..4k+3, low nibble first.
    is_lsb = (j >= lsb0) & (j < end)
    base = code_ptr + t * code_row_stride + (j - lsb0) * 4
    lsb = tl.zeros([BLOCK], dtype=tl.int32)
    for q in tl.static_range(4):
        p = tl.load(base + q, mask=is_lsb, other=0).to(tl.int32)
        lsb |= (p & 1) << (2 * q)
        lsb |= ((p >> 4) & 1) << (2 * q + 1)

    val = tl.where(j < MARK_BYTES, MARK, 0)
    val = tl.where(is_tok, tok.to(tl.int32), val)
    val = tl.where(is_sf, sf.to(tl.int32), val)
    val = tl.where(is_lsb, lsb, val)
    tl.store(out_ptr + t * out_row_stride + j, val.to(tl.uint8), mask=j < REC_BYTES)


def encode_fc1_into(
    out: torch.Tensor,
    packed_codes: torch.Tensor,
    block_scale: torch.Tensor,
    token_scale: torch.Tensor,
    hidden_size: int,
) -> None:
    """Write fc1 records for ``n`` tokens.

    ``out``: ``[n, rec_bytes]`` uint8 view (row stride may exceed rec_bytes).
    ``packed_codes``: ``[n, >=H/2]`` uint8; ``block_scale``: ``[n, >=H/16]``
    uint8; ``token_scale``: ``[n]`` fp32.
    """
    n, rec_bytes = out.shape
    if n == 0:
        return
    sf_bytes, lsb_bytes = hidden_size // 16, hidden_size // 8
    assert rec_bytes >= MARKER_BYTES + 4 + sf_bytes + lsb_bytes
    assert out.dtype == torch.uint8 and out.stride(1) == 1
    assert packed_codes.stride(1) == 1 and block_scale.stride(1) == 1
    tok = token_scale.reshape(n)
    if tok.dtype != torch.float32:
        tok = tok.to(torch.float32)
    tok = tok.contiguous().view(torch.uint8)
    _encode_fc1_kernel[(n,)](
        out,
        out.stride(0),
        tok,
        block_scale,
        block_scale.stride(0),
        packed_codes,
        packed_codes.stride(0),
        SF_BYTES=sf_bytes,
        LSB_BYTES=lsb_bytes,
        REC_BYTES=rec_bytes,
        BLOCK=triton.next_power_of_2(rec_bytes),
        MARK=MARKER,
        MARK_BYTES=MARKER_BYTES,
        num_warps=4,
    )


@triton.jit
def _encode_fc2_kernel(
    out_ptr,
    out_row_stride,
    code_ptr,
    sf_ptr,
    tok_ptr,
    TOPK: tl.constexpr,
    CODE_BYTES: tl.constexpr,
    SF_BYTES: tl.constexpr,
    LSB_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
    MARK: tl.constexpr,
    MARK_BYTES: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1).to(tl.int64)
    row = t * TOPK + k
    rec = out_ptr + t * out_row_stride
    j = tl.arange(0, BLOCK)
    if k == 0:
        m = tl.arange(0, MARK_BYTES)
        tl.store(rec + m, tl.full([MARK_BYTES], MARK, tl.uint8))
    slot_bytes = LSB_BYTES + 4 + SF_BYTES
    sf0 = LSB_BYTES + 4

    is_lsb = j < LSB_BYTES
    base = code_ptr + row * CODE_BYTES + j * 4
    lsb = tl.zeros([BLOCK], dtype=tl.int32)
    for q in tl.static_range(4):
        p = tl.load(base + q, mask=is_lsb, other=0).to(tl.int32)
        lsb |= (p & 1) << (2 * q)
        lsb |= ((p >> 4) & 1) << (2 * q + 1)

    is_tok = (j >= LSB_BYTES) & (j < sf0)
    tok = tl.load(tok_ptr + row * 4 + (j - LSB_BYTES), mask=is_tok, other=0)
    is_sf = (j >= sf0) & (j < slot_bytes)
    sf = tl.load(sf_ptr + row * (CODE_BYTES // 8) + (j - sf0), mask=is_sf, other=0)

    val = tl.where(is_lsb, lsb, 0)
    val = tl.where(is_tok, tok.to(tl.int32), val)
    val = tl.where(is_sf, sf.to(tl.int32), val)
    dst = rec + MARK_BYTES + k * slot_bytes + j
    tl.store(dst, val.to(tl.uint8), mask=j < slot_bytes)


def encode_fc2_into(
    out: torch.Tensor,
    codes: torch.Tensor,
    block_scale: torch.Tensor,
    token_scale: torch.Tensor,
    width: int,
) -> None:
    """Write fc2 records ``[marker | topk x (LSB mask | fp32 scale | E4M3 scales)]``.

    Inputs are FlashInfer's ``fc2_input_out`` buffers for ``n`` tokens:
    ``codes`` ``[n, topk, I/2]`` uint8, ``block_scale`` ``[n, topk, I/16]``,
    ``token_scale`` ``[n, topk]`` fp32, where ``I`` may be padded beyond the
    record ``width``. ``out``: ``[n, >= record bytes]`` uint8 view. Only the
    marker and the slots are written.
    """
    n, topk, code_bytes = codes.shape
    if n == 0:
        return
    assert width % 16 == 0 and width <= code_bytes * 2
    lsb_bytes, sf_bytes = width // 8, width // 16
    assert out.shape[1] >= MARKER_BYTES + topk * (lsb_bytes + 4 + sf_bytes)
    assert out.dtype == torch.uint8 and out.stride(1) == 1
    assert codes.is_contiguous() and block_scale.is_contiguous()
    assert token_scale.is_contiguous() and token_scale.dtype == torch.float32
    _encode_fc2_kernel[(n, topk)](
        out,
        out.stride(0),
        codes,
        block_scale.view(torch.uint8),
        token_scale.view(torch.uint8),
        TOPK=topk,
        CODE_BYTES=code_bytes,
        SF_BYTES=sf_bytes,
        LSB_BYTES=lsb_bytes,
        BLOCK=triton.next_power_of_2(lsb_bytes + 4 + sf_bytes),
        MARK=MARKER,
        MARK_BYTES=MARKER_BYTES,
        num_warps=2,
    )
