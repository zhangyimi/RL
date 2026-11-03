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

import pytest
import torch

from nemo_rl.quantization.trace import layout as L
from nemo_rl.quantization.trace.codec import fp4_lsb_from_packed


def _layout(monkeypatch):
    monkeypatch.setenv(L.LAYERS_ENV, "4:16")
    return L.layout_from_env(topk=8, hidden_size=2048, moe_intermediate_size=768)


def test_layout_sizes(monkeypatch):
    lay = _layout(monkeypatch)
    assert lay.fc1_bytes == 4 + 4 + 128 + 256
    # 768 mantissa bits + fp32 per-token scale + 48 E4M3 block scales per slot.
    assert lay.fc2_code_bytes == 96 and lay.fc2_slot_bytes == 96 + 4 + 48
    assert lay.fc2_bytes == 4 + 8 * 148
    recs = [*lay.fc1_records.values(), *lay.fc2_records.values()]
    assert set(lay.fc1_records) == set(lay.fc2_records) == {4, 16}
    # Records tile the extra rows without overlap.
    spans = sorted((r.row_offset, r.row_offset + r.rows) for r in recs)
    assert spans[0][0] == 0 and spans[-1][1] == lay.extra_rows
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_trace_off_without_layers(monkeypatch):
    monkeypatch.delenv(L.LAYERS_ENV, raising=False)
    assert not L.trace_requested()
    lay = L.layout_from_env(topk=8, hidden_size=2048, moe_intermediate_size=768)
    assert lay.extra_rows == 0


@pytest.mark.parametrize("carrier", [torch.int8, torch.int16, torch.int32])
def test_roundtrip_through_signed_carrier(monkeypatch, carrier):
    lay = _layout(monkeypatch)
    g = torch.Generator().manual_seed(0)
    n = 5
    codes = torch.randint(0, 256, (n, 1024), generator=g, dtype=torch.uint8)
    sf = torch.randint(0, 127, (n, 128), generator=g, dtype=torch.uint8)
    tok = torch.rand(n, generator=g)
    slots = torch.randint(
        0, 256, (n, 8 * lay.fc2_slot_bytes), generator=g, dtype=torch.uint8
    )
    rows = torch.zeros(n, lay.extra_rows, 8, dtype=torch.uint8)
    for layer in (4, 16):
        r1, r2 = lay.fc1_records[layer], lay.fc2_records[layer]
        rows[:, r1.row_offset : r1.row_offset + r1.rows] = L.encode_fc1(
            lay, r1, codes, sf, tok
        )
        rows[:, r2.row_offset : r2.row_offset + r2.rows] = L.encode_fc2(lay, r2, slots)
    carried = rows.view(torch.int8).to(carrier)  # R3 dtypes are signed
    for layer in (4, 16):
        d1 = L.decode_fc1(lay, carried, layer)
        assert d1.valid.all()
        assert torch.equal(d1.lsb, fp4_lsb_from_packed(codes))
        assert torch.equal(d1.token_scale, tok)
        assert torch.equal(d1.block_scale.view(torch.uint8), sf)
        d2 = L.decode_fc2(lay, carried, layer)
        assert d2.valid.all() and torch.equal(d2.slots.reshape(n, -1), slots)
    lsb, t, b = L.split_fc2_slot(lay, d2.slots[:, 0])
    s0 = slots[:, : lay.fc2_slot_bytes]
    assert torch.equal(lsb, s0[:, :96])
    assert torch.equal(t.view(torch.uint8).view(n, 4), s0[:, 96:100])
    assert torch.equal(b.view(torch.uint8), s0[:, 100:])


def test_missing_rows_are_invalid(monkeypatch):
    lay = _layout(monkeypatch)
    rows = torch.full((3, lay.extra_rows, 8), -1, dtype=torch.int8)  # R3 sentinel
    assert not L.decode_fc1(lay, rows, 4).valid.any()
    pad = torch.arange(8, dtype=torch.int8).expand(3, lay.extra_rows, 8)  # CP padding
    assert not L.decode_fc2(lay, pad, 16).valid.any()


def test_fc2_guide_follows_quantization_padding():
    pytest.importorskip("megatron.core")
    from nemo_rl.models.megatron.trace_apply import _pad_guide_to_splits

    rows = torch.arange(5 * 2, dtype=torch.uint8).view(5, 2)
    valid = torch.ones(5, dtype=torch.bool)
    out_r, out_v = _pad_guide_to_splits(rows, valid, [2, 0, 3], [16, 0, 16])
    assert out_r.shape == (32, 2)
    assert torch.equal(out_r[:2], rows[:2]) and torch.equal(out_r[16:19], rows[2:])
    assert out_v.sum() == 5 and not out_v[2:16].any() and not out_v[19:].any()
