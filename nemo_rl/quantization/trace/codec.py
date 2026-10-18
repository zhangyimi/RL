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
"""Bit codecs for TRACE guidance.

NVFP4 data is packed two E2M1 nibbles per byte: element ``2i`` in the low
nibble, element ``2i + 1`` in the high nibble (TE ``cast_to_fp4x2`` and the
FlashInfer/vLLM kernels agree on this). A nibble is ``s e1 e0 m``; TRACE with
``b = 1`` ships only bit 0 (the mantissa bit) of every nibble.

LSB masks are bit-packed little-endian: element ``8j + t`` is bit ``t`` of
byte ``j``.
"""

from __future__ import annotations

import torch


def _shifts(device) -> torch.Tensor:
    # Created on-device (no host copy) so this is safe under CUDA-graph capture.
    return torch.arange(8, dtype=torch.uint8, device=device)


def unpack_fp4_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """``[..., K/2]`` uint8 -> ``[..., K]`` uint8 4-bit codes."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"expected uint8 packed FP4, got {packed.dtype}")
    lo = packed & 0x0F
    hi = packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def pack_fp4_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """``[..., K]`` uint8 4-bit codes -> ``[..., K/2]`` uint8."""
    if codes.shape[-1] % 2:
        raise ValueError(f"last dim must be even, got {codes.shape[-1]}")
    codes = codes.to(torch.uint8)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """``[..., K]`` bool/0-1 -> ``[..., K/8]`` uint8 (little-endian in byte)."""
    if bits.shape[-1] % 8:
        raise ValueError(f"last dim must be a multiple of 8, got {bits.shape[-1]}")
    b = bits.to(torch.uint8).unflatten(-1, (-1, 8))
    return (b << _shifts(b.device)).sum(dim=-1, dtype=torch.uint8)


def unpack_bits(packed: torch.Tensor) -> torch.Tensor:
    """``[..., K/8]`` uint8 -> ``[..., K]`` bool."""
    return ((packed.unsqueeze(-1) >> _shifts(packed.device)) & 1).flatten(-2).bool()


def fp4_lsb_from_packed(packed: torch.Tensor) -> torch.Tensor:
    """Packed NVFP4 data ``[..., K/2]`` -> packed mantissa-bit mask ``[..., K/8]``."""
    return pack_bits(unpack_fp4_nibbles(packed) & 1)


def fp4_low2_from_packed(packed: torch.Tensor) -> torch.Tensor:
    """Packed NVFP4 ``[..., K/2]`` -> low two code bits ``[..., K/4]``.

    Element ``4j + t`` occupies bits ``2t, 2t+1`` of byte ``j`` (b=2 TRACE).
    """
    c = (unpack_fp4_nibbles(packed) & 3).unflatten(-1, (-1, 4))
    return (c << (2 * _shifts(c.device)[:4])).sum(dim=-1, dtype=torch.uint8)


def low2_to_packed_codes(low2: torch.Tensor) -> torch.Tensor:
    """Inverse view of :func:`fp4_low2_from_packed`: ``[..., K/4]`` -> packed
    nibbles ``[..., K/2]`` whose codes carry only the two low bits."""
    c = ((low2.unsqueeze(-1) >> (2 * _shifts(low2.device)[:4])) & 3).flatten(-2)
    return pack_fp4_nibbles(c)
