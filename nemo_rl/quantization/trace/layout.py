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

Rollout FP4 codes travel back to the trainer as extra rows appended to the
vLLM routed-experts capture buffer ``[tokens, num_layers + extra_rows, topk]``:
each row holds ``topk`` bytes, so every per-token R3 code path (scheduler slot
store, prefix-cache reuse, NeMo-RL padding, packing, CP/SP sharding) moves the
payload bit-for-bit next to the routes. The trainer splits the extra rows off
before router replay sees them.

Only this module decides the byte layout; vLLM capture and the trainer both
derive it from the same model dimensions plus the ``NRL_TRACE_*`` env vars
(resolved from ``nvfp4_pertoken_rollout.trace`` on the driver, see
:mod:`.config`), so it must stay a pure function of its inputs. It imports nothing heavier than
torch because the vLLM scheduler process imports it to size its slot store.

Per selected layer, a site record is ``[marker u8 x4 | scales | codes]``:

* fc1 (routed-expert input, per token, shared by the token's top-k experts):
  ``[per-token fp32 scale (4B) | E4M3 block scales (H/16)] | codes`` (scales
  only with ``NRL_TRACE_SHIP_SCALES=1``; read by the probe only)
* fc2 (routed-expert fc2 input, one slot per top-k expert in rollout order):
  ``codes`` per slot
* kv (attention cache, rollout-local KV heads; static per-layer global scale):
  ``K block scales | K codes | V block scales | V codes``

``codes`` are the packed E2M1 nibbles (``mode=full``, used by the alignment
probe), the packed mantissa bits (``mode=lsb``, b=1 TRACE) or the two low code
bits (``mode=b2``); fc2 follows ``NRL_TRACE_FC2_BITS`` and KV
``NRL_TRACE_KV_BITS``. ``NRL_TRACE_FULL_LAYERS`` forces full fc1/KV codes on
probe layers.
Records are padded to whole rows.
"""

from __future__ import annotations

import json
import re
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

FC1_LAYERS_ENV = "NRL_TRACE_FC1_LAYERS"
KV_LAYERS_ENV = "NRL_TRACE_KV_LAYERS"
FC2_LAYERS_ENV = "NRL_TRACE_FC2_LAYERS"
FC2_BITS_ENV = "NRL_TRACE_FC2_BITS"
KV_BITS_ENV = "NRL_TRACE_KV_BITS"
MODE_ENV = "NRL_TRACE_MODE"
# Layers that always ship full 4-bit codes (alignment probe), whatever MODE is.
FULL_LAYERS_ENV = "NRL_TRACE_FULL_LAYERS"
# fc1 records carry the rollout scales only when this is "1". The training
# forward always uses its own (TE) scales; the probe compares the two.
SHIP_SCALES_ENV = "NRL_TRACE_SHIP_SCALES"
MARKER = 0xA5
MARKER_BYTES = 4
MODES = ("full", "lsb", "b2")
_FC2_MODES = {1: "lsb", 2: "b2", 4: "full"}


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
    """Byte offsets of one layer's record inside the token's payload."""

    layer: int
    row_offset: int  # relative to the first trace row
    rows: int
    mode: str  # "full", "lsb" or "b2"


@dataclass(frozen=True)
class TraceLayout:
    topk: int
    hidden_size: int
    num_kv_heads: int
    head_dim: int
    fc1_layers: tuple[int, ...] = ()
    kv_layers: tuple[int, ...] = ()
    fc2_layers: tuple[int, ...] = ()
    moe_intermediate_size: int = 0
    mode: str = "full"
    full_layers: tuple[int, ...] = ()
    ship_scales: bool = False
    fc2_bits: int = 1
    kv_bits: int = 1
    fc1_records: dict[int, SiteRecord] = field(default_factory=dict, compare=False)
    kv_records: dict[int, SiteRecord] = field(default_factory=dict, compare=False)
    fc2_records: dict[int, SiteRecord] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"{MODE_ENV} must be one of {MODES}, got {self.mode!r}")
        if self.fc2_bits not in _FC2_MODES:
            raise ValueError(
                f"{FC2_BITS_ENV} must be one of {tuple(_FC2_MODES)}, got {self.fc2_bits}"
            )
        if self.hidden_size % 16 or self.head_dim % 16:
            raise ValueError("hidden_size and head_dim must be multiples of 16")
        offset = 0
        for layer in self.fc1_layers:
            mode = self.mode_for(layer)
            rows = _ceil_div(self.fc1_bytes(mode), self.topk)
            self.fc1_records[layer] = SiteRecord(layer, offset, rows, mode)
            offset += rows
        for layer in self.kv_layers:
            mode = (
                "full"
                if self.mode_for(layer) == "full"
                else {1: "lsb", 2: "b2", 4: "full"}[self.kv_bits]
            )
            rows = _ceil_div(self.kv_bytes(mode), self.topk)
            self.kv_records[layer] = SiteRecord(layer, offset, rows, mode)
            offset += rows
        if self.fc2_layers and self.moe_intermediate_size % 16:
            raise ValueError("fc2 capture needs moe_intermediate_size % 16 == 0")
        for layer in self.fc2_layers:
            # The FlashInfer overlay writes the record's code bits directly.
            rows = _ceil_div(self.fc2_bytes, self.topk)
            self.fc2_records[layer] = SiteRecord(layer, offset, rows, self.fc2_mode)
            offset += rows
        object.__setattr__(self, "_extra_rows", offset)

    # -- sizes -------------------------------------------------------------
    def mode_for(self, layer: int) -> str:
        return "full" if layer in self.full_layers else self.mode

    @staticmethod
    def code_bytes(n: int, mode: str) -> int:
        return {"full": n // 2, "b2": n // 4, "lsb": n // 8}[mode]

    def fc1_scale_bytes(self) -> int:
        return 4 + self.hidden_size // 16 if self.ship_scales else 0

    def fc1_bytes(self, mode: str) -> int:
        h = self.hidden_size
        return MARKER_BYTES + self.fc1_scale_bytes() + self.code_bytes(h, mode)

    @property
    def kv_side_elems(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def fc2_mode(self) -> str:
        return _FC2_MODES[self.fc2_bits]

    @property
    def fc2_slot_bytes(self) -> int:
        return self.code_bytes(self.moe_intermediate_size, self.fc2_mode)

    @property
    def fc2_bytes(self) -> int:
        """Marker + one row of mantissa bits (or codes) per top-k slot (rollout k order)."""
        return MARKER_BYTES + self.topk * self.fc2_slot_bytes

    def kv_bytes(self, mode: str) -> int:
        n = self.kv_side_elems
        return MARKER_BYTES + 2 * (n // 16 + self.code_bytes(n, mode))

    @property
    def extra_rows(self) -> int:
        return self._extra_rows  # type: ignore[attr-defined]

    @property
    def enabled(self) -> bool:
        return self.extra_rows > 0

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("fc1_records")
        d.pop("kv_records")
        d.pop("fc2_records")
        return json.dumps(d, sort_keys=True)


def layout_from_env(
    *,
    topk: int,
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    moe_intermediate_size: int = 0,
) -> TraceLayout:
    return TraceLayout(
        fc2_layers=_parse_layers(os.environ.get(FC2_LAYERS_ENV)),
        moe_intermediate_size=moe_intermediate_size,
        topk=topk,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        fc1_layers=_parse_layers(os.environ.get(FC1_LAYERS_ENV)),
        kv_layers=_parse_layers(os.environ.get(KV_LAYERS_ENV)),
        mode=os.environ.get(MODE_ENV, "full"),
        full_layers=_parse_layers(os.environ.get(FULL_LAYERS_ENV)),
        ship_scales=os.environ.get(SHIP_SCALES_ENV, "0") == "1",
        fc2_bits=int(os.environ.get(FC2_BITS_ENV) or 1),
        kv_bits=int(os.environ.get(KV_BITS_ENV) or 1),
    )


def trace_requested() -> bool:
    return bool(
        _parse_layers(os.environ.get(FC1_LAYERS_ENV))
        or _parse_layers(os.environ.get(KV_LAYERS_ENV))
        or _parse_layers(os.environ.get(FC2_LAYERS_ENV))
    )


def layout_for_hf_config(hf_text_config, *, kv_tp_size: int = 1) -> TraceLayout:
    """Layout from a HF text config (vLLM side; rollout-local KV heads)."""
    num_heads = int(hf_text_config.num_attention_heads)
    head_dim = int(
        getattr(hf_text_config, "head_dim", None)
        or hf_text_config.hidden_size // num_heads
    )
    num_kv = int(getattr(hf_text_config, "num_key_value_heads", num_heads))
    if num_kv % kv_tp_size:
        raise ValueError(f"num_kv_heads={num_kv} not divisible by TP={kv_tp_size}")
    return layout_from_env(
        topk=int(hf_text_config.num_experts_per_tok),
        hidden_size=int(hf_text_config.hidden_size),
        num_kv_heads=num_kv // kv_tp_size,
        head_dim=head_dim,
        moe_intermediate_size=int(
            getattr(hf_text_config, "moe_intermediate_size", 0) or 0
        ),
    )


def extra_rows_for_vllm_config(vllm_config) -> int:
    """Extra routed-experts rows for a vLLM engine (0 when TRACE is off)."""
    if not trace_requested():
        return 0
    tp = int(vllm_config.parallel_config.tensor_parallel_size)
    if tp != 1:
        # Only TP rank 0's routed-experts reach the scheduler; KV heads and
        # fc2 shards on other ranks would be silently dropped.
        raise ValueError(
            f"TRACE capture requires vLLM tensor_parallel_size=1, got {tp}"
        )
    return layout_for_hf_config(vllm_config.model_config.hf_text_config).extra_rows


# -- payload encode (rollout) / decode (trainer) ----------------------------


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """Flatten the trailing dims of ``t`` to raw bytes ``[n, nbytes]``."""
    n = t.shape[0]
    return t.contiguous().view(torch.uint8).reshape(n, -1)


def _code_payload(mode: str, packed_codes: torch.Tensor) -> torch.Tensor:
    if mode == "full":
        return _bytes(packed_codes)
    from nemo_rl.quantization.trace.codec import (
        fp4_lsb_from_packed,
        fp4_low2_from_packed,
    )

    if mode == "b2":
        return fp4_low2_from_packed(_bytes(packed_codes))
    return fp4_lsb_from_packed(_bytes(packed_codes))


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
    packed_codes: torch.Tensor,  # [n, H/2] uint8
    block_scale: torch.Tensor,  # [n, H/16] e4m3 bytes
    token_scale: torch.Tensor,  # [n] fp32
) -> torch.Tensor:
    n = packed_codes.shape[0]
    parts = [_marker(n, packed_codes.device)]
    if layout.ship_scales:
        parts += [
            _bytes(token_scale.to(torch.float32).reshape(n, 1)),
            _bytes(block_scale),
        ]
    parts.append(_code_payload(rec.mode, packed_codes))
    return _pad_to_rows(layout, parts, rec.rows)


def encode_kv(
    layout: TraceLayout,
    rec: SiteRecord,
    k_codes: torch.Tensor,  # [n, Hkv, D/2]
    k_scale: torch.Tensor,  # [n, Hkv, D/16]
    v_codes: torch.Tensor,
    v_scale: torch.Tensor,
    valid: torch.Tensor,  # [n] bool (padding slots carry no record)
) -> torch.Tensor:
    n = k_codes.shape[0]
    marker = _marker(n, k_codes.device) * valid.view(n, 1).to(torch.uint8)
    return _pad_to_rows(
        layout,
        [
            marker,
            _bytes(k_scale),
            _code_payload(rec.mode, k_codes),
            _bytes(v_scale),
            _code_payload(rec.mode, v_codes),
        ],
        rec.rows,
    )


@dataclass
class DecodedFc1:
    valid: torch.Tensor  # [n] bool
    token_scale: Optional[torch.Tensor]  # [n] fp32 (None unless ship_scales)
    block_scale: Optional[torch.Tensor]  # [n, H/16] float8_e4m3fn
    codes: (
        torch.Tensor
    )  # [n, H/2] nibbles (full), [n, H/4] low-2 bits (b2) or [n, H/8] bits (lsb)
    mode: str

    def guide_codes(self) -> torch.Tensor:
        """Packed nibbles ``[n, H/2]`` (b2: only the two low bits are meaningful)."""
        return _guide_codes(self.mode, self.codes)

    def lsb(self) -> torch.Tensor:
        """Packed mantissa bits ``[n, H/8]`` whatever the record mode."""
        if self.mode == "lsb":
            return self.codes
        from nemo_rl.quantization.trace.codec import fp4_lsb_from_packed

        return fp4_lsb_from_packed(self.guide_codes())


def _guide_codes(mode: str, codes: torch.Tensor) -> torch.Tensor:
    if mode == "full":
        return codes
    if mode == "b2":
        from nemo_rl.quantization.trace.codec import low2_to_packed_codes

        return low2_to_packed_codes(codes)
    raise ValueError(f"no code bits beyond the LSB in mode {mode!r}")


@dataclass
class DecodedKV:
    valid: torch.Tensor  # [n] bool
    k_scale: torch.Tensor  # [n, Hkv, D/16] float8_e4m3fn
    k_codes: torch.Tensor  # [n, Hkv, D/2] (full) or [n, Hkv, D/8] (lsb)
    v_scale: torch.Tensor
    v_codes: torch.Tensor
    mode: str


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


def decode_fc1(layout: TraceLayout, trace_rows: torch.Tensor, layer: int) -> DecodedFc1:
    """``trace_rows``: ``[n, extra_rows, topk]`` (any 1-byte integer dtype)."""
    rec = layout.fc1_records[layer]
    b = _record_bytes(layout, trace_rows, rec)
    h = layout.hidden_size
    o = MARKER_BYTES
    tok = sf = None
    if layout.ship_scales:
        tok = b[:, o : o + 4].contiguous().view(torch.float32).view(-1)
        o += 4
        sf = b[:, o : o + h // 16].contiguous().view(torch.float8_e4m3fn)
        o += h // 16
    codes = b[:, o : o + layout.code_bytes(h, rec.mode)].contiguous()
    return DecodedFc1(_is_valid(b), tok, sf, codes, rec.mode)


def decode_kv(layout: TraceLayout, trace_rows: torch.Tensor, layer: int) -> DecodedKV:
    rec = layout.kv_records[layer]
    b = _record_bytes(layout, trace_rows, rec)
    n, hk, d = b.shape[0], layout.num_kv_heads, layout.head_dim
    o = MARKER_BYTES
    out = []
    for _ in range(2):
        sf = (
            b[:, o : o + hk * d // 16]
            .contiguous()
            .view(torch.float8_e4m3fn)
            .view(n, hk, d // 16)
        )
        o += hk * d // 16
        cb = layout.code_bytes(d, rec.mode)
        codes = b[:, o : o + hk * cb].contiguous().view(n, hk, cb)
        o += hk * cb
        out += [sf, codes]
    return DecodedKV(_is_valid(b), out[0], out[1], out[2], out[3], rec.mode)


def encode_fc2(layout: TraceLayout, rec: SiteRecord, lsb: torch.Tensor) -> torch.Tensor:
    """``lsb``: ``[n, topk * slot_bytes]`` uint8 in rollout top-k order."""
    n = lsb.shape[0]
    return _pad_to_rows(layout, [_marker(n, lsb.device), lsb], rec.rows)


@dataclass
class DecodedFc2:
    valid: torch.Tensor  # [n] bool
    lsb: torch.Tensor  # [n, topk, slot_bytes] uint8 in record mode, rollout top-k order
    mode: str = "lsb"


def decode_fc2(layout: TraceLayout, trace_rows: torch.Tensor, layer: int) -> DecodedFc2:
    rec = layout.fc2_records[layer]
    b = _record_bytes(layout, trace_rows, rec)
    n, nb = b.shape[0], layout.fc2_slot_bytes
    lsb = (
        b[:, MARKER_BYTES : MARKER_BYTES + layout.topk * nb]
        .contiguous()
        .view(n, layout.topk, nb)
    )
    return DecodedFc2(_is_valid(b), lsb, rec.mode)
