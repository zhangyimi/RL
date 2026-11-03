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
"""Per-token TRACE payload layout carried on the router-replay side channel.

Rollout FP4 records travel back to the trainer as extra rows appended to the
vLLM routed-experts capture buffer ``[tokens, num_layers + extra_rows, topk]``:
each row holds ``topk`` bytes, so every per-token R3 code path (scheduler slot
store, prefix-cache reuse, NeMo-RL padding, packing, CP/SP sharding) moves the
payload bit-for-bit next to the routes. The trainer splits the extra rows off
before router replay sees them.

Only this module decides the byte layout; vLLM capture and the trainer both
derive it from the model dimensions and ``NRL_TRACE_LAYERS`` (resolved from
``nvfp4_pertoken_rollout.trace`` on the driver, see :mod:`.config`). It imports
nothing heavier than torch because the vLLM scheduler process imports it to
size its slot store.

Per guided MoE layer there are two records, each starting with a 4-byte marker:

* fc1 (routed-expert input, per token, shared by the token's top-k experts):
  ``[marker | fp32 per-token decode scale | E4M3 block scales (H/16) | mantissa bits (H/8)]``
* fc2 (routed-expert fc2 input, one slot per top-k expert in rollout order):
  ``[marker | topk x (mantissa bits (I/8) | fp32 per-token decode scale | E4M3 block scales (I/16))]``

Records are padded to whole rows.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

LAYERS_ENV = "NRL_TRACE_LAYERS"
MARKER = 0xA5
MARKER_BYTES = 4


def _parse_layers(value: Optional[str]) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(
        sorted({int(v) for v in re.split(r"[,:;]", value.replace(" ", "")) if v})
    )


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class SiteRecord:
    """Row offsets of one layer's record inside the token's payload."""

    layer: int
    row_offset: int  # relative to the first trace row
    rows: int


@dataclass(frozen=True)
class TraceLayout:
    topk: int
    hidden_size: int
    moe_intermediate_size: int
    layers: tuple[int, ...] = ()
    fc1_records: dict[int, SiteRecord] = field(default_factory=dict, compare=False)
    fc2_records: dict[int, SiteRecord] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.hidden_size % 16 or self.moe_intermediate_size % 16:
            raise ValueError(
                "TRACE needs hidden_size and moe_intermediate_size multiples of 16"
            )
        offset = 0
        for layer in self.layers:
            rows = _ceil_div(self.fc1_bytes, self.topk)
            self.fc1_records[layer] = SiteRecord(layer, offset, rows)
            offset += rows
            # The FlashInfer overlay writes the fc2 record directly.
            rows = _ceil_div(self.fc2_bytes, self.topk)
            self.fc2_records[layer] = SiteRecord(layer, offset, rows)
            offset += rows
        object.__setattr__(self, "_extra_rows", offset)

    # -- sizes -------------------------------------------------------------
    @staticmethod
    def scale_bytes(n: int) -> int:
        """fp32 per-token decode scale + one E4M3 block scale per 16 elements."""
        return 4 + n // 16

    @property
    def fc1_bytes(self) -> int:
        h = self.hidden_size
        return MARKER_BYTES + self.scale_bytes(h) + h // 8

    @property
    def fc2_code_bytes(self) -> int:
        return self.moe_intermediate_size // 8

    @property
    def fc2_slot_bytes(self) -> int:
        return self.fc2_code_bytes + self.scale_bytes(self.moe_intermediate_size)

    @property
    def fc2_bytes(self) -> int:
        return MARKER_BYTES + self.topk * self.fc2_slot_bytes

    @property
    def extra_rows(self) -> int:
        return self._extra_rows  # type: ignore[attr-defined]

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("fc1_records")
        d.pop("fc2_records")
        return json.dumps(d, sort_keys=True)


def layout_from_env(
    *, topk: int, hidden_size: int, moe_intermediate_size: int
) -> TraceLayout:
    return TraceLayout(
        topk=topk,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        layers=_parse_layers(os.environ.get(LAYERS_ENV)),
    )


def trace_requested() -> bool:
    return bool(_parse_layers(os.environ.get(LAYERS_ENV)))


def layout_for_hf_config(hf_text_config) -> TraceLayout:
    """Layout from a HF text config (vLLM side)."""
    return layout_from_env(
        topk=int(hf_text_config.num_experts_per_tok),
        hidden_size=int(hf_text_config.hidden_size),
        moe_intermediate_size=int(hf_text_config.moe_intermediate_size),
    )


def extra_rows_for_vllm_config(vllm_config) -> int:
    """Extra routed-experts rows for a vLLM engine (0 when TRACE is off)."""
    if not trace_requested():
        return 0
    tp = int(vllm_config.parallel_config.tensor_parallel_size)
    if tp != 1:
        # Only TP rank 0's routed-experts reach the scheduler; fc2 shards on
        # other ranks would be silently dropped.
        raise ValueError(
            f"TRACE capture requires vLLM tensor_parallel_size=1, got {tp}"
        )
    return layout_for_hf_config(vllm_config.model_config.hf_text_config).extra_rows


# -- payload encode (rollout) / decode (trainer) ----------------------------


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """Flatten the trailing dims of ``t`` to raw bytes ``[n, nbytes]``."""
    n = t.shape[0]
    return t.contiguous().view(torch.uint8).reshape(n, -1)


def _pad_to_rows(
    layout: TraceLayout, parts: list[torch.Tensor], rows: int
) -> torch.Tensor:
    n = parts[0].shape[0]
    rec = torch.cat(parts, dim=1)
    pad = rows * layout.topk - rec.shape[1]
    if pad:
        rec = torch.nn.functional.pad(rec, (0, pad))
    return rec.view(n, rows, layout.topk)


def _marker(n: int, device) -> torch.Tensor:
    return torch.full((n, MARKER_BYTES), MARKER, dtype=torch.uint8, device=device)


def encode_fc1(
    layout: TraceLayout,
    rec: SiteRecord,
    packed_codes: torch.Tensor,  # [n, H/2] uint8 E2M1 nibbles
    block_scale: torch.Tensor,  # [n, H/16] e4m3 bytes
    token_scale: torch.Tensor,  # [n] fp32 per-token decode scale
) -> torch.Tensor:
    from nemo_rl.quantization.trace.codec import fp4_lsb_from_packed

    n = packed_codes.shape[0]
    parts = [
        _marker(n, packed_codes.device),
        _bytes(token_scale.to(torch.float32).reshape(n, 1)),
        _bytes(block_scale),
        fp4_lsb_from_packed(_bytes(packed_codes)),
    ]
    return _pad_to_rows(layout, parts, rec.rows)


def _record_bytes(
    layout: TraceLayout, trace_rows: torch.Tensor, rec: SiteRecord
) -> torch.Tensor:
    n = trace_rows.shape[0]
    rows = trace_rows[:, rec.row_offset : rec.row_offset + rec.rows, :]
    # Each element carries one byte; the carrier may be int8/int16/int32/long
    # (signed R3 dtypes), and a modular cast recovers the byte exactly.
    return rows.to(torch.uint8).reshape(n, rec.rows * layout.topk)


def _is_valid(b: torch.Tensor) -> torch.Tensor:
    return (b[:, :MARKER_BYTES] == MARKER).all(dim=1)


@dataclass
class DecodedFc1:
    valid: torch.Tensor  # [n] bool
    token_scale: torch.Tensor  # [n] fp32
    block_scale: torch.Tensor  # [n, H/16] float8_e4m3fn
    lsb: torch.Tensor  # [n, H/8] packed mantissa bits


def decode_fc1(layout: TraceLayout, trace_rows: torch.Tensor, layer: int) -> DecodedFc1:
    """``trace_rows``: ``[n, extra_rows, topk]`` (any 1-byte integer dtype)."""
    rec = layout.fc1_records[layer]
    b = _record_bytes(layout, trace_rows, rec)
    h = layout.hidden_size
    o = MARKER_BYTES
    tok = b[:, o : o + 4].contiguous().view(torch.float32).view(-1)
    o += 4
    sf = b[:, o : o + h // 16].contiguous().view(torch.float8_e4m3fn)
    o += h // 16
    lsb = b[:, o : o + h // 8].contiguous()
    return DecodedFc1(_is_valid(b), tok, sf, lsb)


def encode_fc2(
    layout: TraceLayout, rec: SiteRecord, slots: torch.Tensor
) -> torch.Tensor:
    """``slots``: ``[n, topk * slot_bytes]`` uint8 in rollout top-k order."""
    n = slots.shape[0]
    return _pad_to_rows(layout, [_marker(n, slots.device), slots], rec.rows)


@dataclass
class DecodedFc2:
    valid: torch.Tensor  # [n] bool
    slots: torch.Tensor  # [n, topk, slot_bytes] uint8, rollout top-k order


def decode_fc2(layout: TraceLayout, trace_rows: torch.Tensor, layer: int) -> DecodedFc2:
    rec = layout.fc2_records[layer]
    b = _record_bytes(layout, trace_rows, rec)
    n, nb = b.shape[0], layout.fc2_slot_bytes
    slots = (
        b[:, MARKER_BYTES : MARKER_BYTES + layout.topk * nb]
        .contiguous()
        .view(n, layout.topk, nb)
    )
    return DecodedFc2(_is_valid(b), slots)


def split_fc2_slot(
    layout: TraceLayout, slots: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[m, slot_bytes]`` -> (mantissa bits, fp32 decode scale [m], E4M3 block scales)."""
    cb = layout.fc2_code_bytes
    lsb = slots[:, :cb]
    tok = slots[:, cb : cb + 4].contiguous().view(torch.float32).view(-1)
    blk = slots[:, cb + 4 :].contiguous().view(torch.float8_e4m3fn)
    return lsb, tok, blk
