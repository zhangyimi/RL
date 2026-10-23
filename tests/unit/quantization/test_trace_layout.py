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


def _layout(monkeypatch, mode):
    monkeypatch.setenv(L.FC1_LAYERS_ENV, "4:16")
    monkeypatch.setenv(L.KV_LAYERS_ENV, "0,47")
    monkeypatch.setenv(L.MODE_ENV, mode)
    monkeypatch.setenv(L.SHIP_SCALES_ENV, "1")
    return L.layout_from_env(topk=8, hidden_size=2048, num_kv_heads=4, head_dim=128)


@pytest.mark.parametrize("mode", ["full", "lsb"])
def test_layout_sizes(monkeypatch, mode):
    lay = _layout(monkeypatch, mode)
    code = 1024 if mode == "full" else 256
    assert lay.fc1_bytes(mode) == 4 + 4 + 128 + code
    assert lay.extra_rows == sum(r.rows for r in lay.fc1_records.values()) + sum(
        r.rows for r in lay.kv_records.values()
    )
    # Records tile the extra rows without overlap.
    spans = sorted(
        (r.row_offset, r.row_offset + r.rows)
        for r in [*lay.fc1_records.values(), *lay.kv_records.values()]
    )
    assert spans[0][0] == 0 and spans[-1][1] == lay.extra_rows
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


@pytest.mark.parametrize("mode", ["full", "lsb"])
@pytest.mark.parametrize("carrier", [torch.int8, torch.int16, torch.int32])
def test_roundtrip_through_signed_carrier(monkeypatch, mode, carrier):
    lay = _layout(monkeypatch, mode)
    g = torch.Generator().manual_seed(0)
    n = 5
    codes = torch.randint(0, 256, (n, 1024), generator=g, dtype=torch.uint8)
    sf = torch.randint(0, 127, (n, 128), generator=g, dtype=torch.uint8)
    tok = torch.rand(n, generator=g)
    kc = torch.randint(0, 256, (n, 4, 64), generator=g, dtype=torch.uint8)
    ks = torch.randint(0, 127, (n, 4, 8), generator=g, dtype=torch.uint8)
    vc = torch.randint(0, 256, (n, 4, 64), generator=g, dtype=torch.uint8)
    vs = torch.randint(0, 127, (n, 4, 8), generator=g, dtype=torch.uint8)
    valid = torch.tensor([True, True, False, True, True])

    buf = torch.full((n, 48 + lay.extra_rows, 8), -1, dtype=torch.int32)
    for layer, rec in lay.fc1_records.items():
        buf[:, 48 + rec.row_offset : 48 + rec.row_offset + rec.rows] = L.encode_fc1(
            lay, rec, codes, sf, tok
        ).to(torch.int32)
    for layer, rec in lay.kv_records.items():
        buf[:, 48 + rec.row_offset : 48 + rec.row_offset + rec.rows] = L.encode_kv(
            lay, rec, kc, ks, vc, vs, valid
        ).to(torch.int32)
    # vLLM narrows to uint8 in the slot store; NeMo-RL then casts to a signed dtype.
    carried = buf.to(torch.uint8).to(carrier)[:, 48:]

    exp_codes = codes if mode == "full" else fp4_lsb_from_packed(codes)
    d = L.decode_fc1(lay, carried, 16)
    assert d.valid.all()
    assert torch.equal(d.token_scale, tok)
    assert torch.equal(d.block_scale.view(torch.uint8), sf)
    assert torch.equal(d.codes, exp_codes)

    k = L.decode_kv(lay, carried, 47)
    assert torch.equal(k.valid, valid)
    assert torch.equal(k.k_scale.view(torch.uint8), ks)
    assert torch.equal(k.v_scale.view(torch.uint8), vs)
    if mode == "full":
        assert torch.equal(k.k_codes, kc)
        assert torch.equal(k.v_codes, vc)
    else:
        assert torch.equal(k.k_codes, fp4_lsb_from_packed(kc))


def test_missing_rows_are_invalid(monkeypatch):
    lay = _layout(monkeypatch, "full")
    rows = torch.full((3, lay.extra_rows, 8), -1, dtype=torch.int8)  # R3 sentinel
    assert not L.decode_fc1(lay, rows, 4).valid.any()
    pad = torch.arange(8, dtype=torch.int8).expand(3, lay.extra_rows, 8)  # CP padding
    assert not L.decode_kv(lay, pad, 0).valid.any()


def test_full_layers_and_no_scales(monkeypatch):
    monkeypatch.setenv(L.FC1_LAYERS_ENV, "2:3:4")
    monkeypatch.setenv(L.FULL_LAYERS_ENV, "3")
    monkeypatch.setenv(L.MODE_ENV, "lsb")
    monkeypatch.setenv(L.SHIP_SCALES_ENV, "0")
    lay = L.layout_from_env(topk=8, hidden_size=2048, num_kv_heads=4, head_dim=128)
    assert lay.fc1_records[2].mode == "lsb" and lay.fc1_records[3].mode == "full"
    assert lay.fc1_records[2].rows == -(-(4 + 256) // 8)
    g = torch.Generator().manual_seed(3)
    codes = torch.randint(0, 256, (4, 1024), generator=g, dtype=torch.uint8)
    buf = torch.zeros((4, lay.extra_rows, 8), dtype=torch.int32)
    for rec in lay.fc1_records.values():
        buf[:, rec.row_offset : rec.row_offset + rec.rows] = L.encode_fc1(
            lay, rec, codes, None, None
        ).to(torch.int32)
    rows = buf.to(torch.uint8).to(torch.int8)
    d3, d2 = L.decode_fc1(lay, rows, 3), L.decode_fc1(lay, rows, 2)
    assert d3.token_scale is None and d3.block_scale is None
    assert torch.equal(d3.codes, codes)
    assert torch.equal(d2.lsb(), d3.lsb())


def test_fc2_guide_follows_quantization_padding():
    pytest.importorskip("megatron.core")
    from nemo_rl.models.megatron.trace_apply import _pad_guide_to_splits

    lsb = torch.arange(5 * 2, dtype=torch.uint8).view(5, 2)
    valid = torch.ones(5, dtype=torch.bool)
    out_l, out_v = _pad_guide_to_splits(lsb, valid, [2, 0, 3], [16, 0, 16])
    assert out_l.shape == (32, 2)
    assert torch.equal(out_l[:2], lsb[:2]) and torch.equal(out_l[16:19], lsb[2:])
    assert out_v.sum() == 5 and not out_v[2:16].any() and not out_v[19:].any()


def test_fc2_full_roundtrip(monkeypatch):
    monkeypatch.setenv(L.FC2_LAYERS_ENV, "5")
    monkeypatch.setenv(L.FC2_BITS_ENV, "4")
    lay = L.layout_from_env(
        topk=8,
        hidden_size=2048,
        num_kv_heads=4,
        head_dim=128,
        moe_intermediate_size=768,
    )
    rec = lay.fc2_records[5]
    assert rec.mode == "full" and lay.fc2_slot_bytes == 384
    codes = torch.randint(0, 256, (3, 8 * 384), dtype=torch.uint8)
    rows = L.encode_fc2(lay, rec, codes).to(torch.int32).to(torch.uint8).to(torch.int8)
    d = L.decode_fc2(lay, rows, 5)
    assert d.valid.all() and d.mode == "full"
    assert torch.equal(d.lsb.reshape(3, -1), codes)


def test_b2_mode_roundtrip(monkeypatch):
    from nemo_rl.quantization.trace.codec import (
        fp4_lsb_from_packed,
        fp4_low2_from_packed,
    )

    monkeypatch.setenv(L.FC1_LAYERS_ENV, "3")
    monkeypatch.setenv(L.KV_LAYERS_ENV, "3")
    monkeypatch.setenv(L.FC2_LAYERS_ENV, "3")
    monkeypatch.setenv(L.MODE_ENV, "b2")
    monkeypatch.setenv(L.SHIP_SCALES_ENV, "0")
    monkeypatch.setenv(L.FC2_BITS_ENV, "2")
    lay = L.layout_from_env(
        topk=8,
        hidden_size=2048,
        num_kv_heads=4,
        head_dim=128,
        moe_intermediate_size=768,
    )
    assert lay.fc1_records[3].mode == "b2" and lay.kv_records[3].mode == "lsb"
    assert lay.fc2_records[3].mode == "b2" and lay.fc2_slot_bytes == 192
    codes = torch.randint(0, 256, (4, 1024), dtype=torch.uint8)
    rows = torch.zeros((4, lay.extra_rows, 8), dtype=torch.int32)
    r = lay.fc1_records[3]
    rows[:, r.row_offset : r.row_offset + r.rows] = L.encode_fc1(
        lay, r, codes, None, None
    ).to(torch.int32)
    d = L.decode_fc1(lay, rows.to(torch.uint8).to(torch.int8), 3)
    assert torch.equal(d.codes, fp4_low2_from_packed(codes))
    assert torch.equal(d.lsb(), fp4_lsb_from_packed(codes))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("mode", ["lsb", "b2", "full"])
def test_encode_fc1_into_matches_torch_encoder(monkeypatch, mode):
    from nemo_rl.quantization.trace.encode_triton import encode_fc1_into

    monkeypatch.setenv(L.FC1_LAYERS_ENV, "1:5")
    monkeypatch.setenv(L.MODE_ENV, mode)
    monkeypatch.setenv(L.SHIP_SCALES_ENV, "0")
    lay = L.layout_from_env(topk=8, hidden_size=2048, num_kv_heads=4, head_dim=128)
    n, nl = 37, 48
    codes = torch.randint(0, 256, (n, 1024 + 32), dtype=torch.uint8, device="cuda")[
        :, :1024
    ]
    buf = torch.zeros((64, nl + lay.extra_rows, 8), dtype=torch.uint8, device="cuda")
    for rec in lay.fc1_records.values():
        encode_fc1_into(buf, (nl + rec.row_offset) * 8, codes, rec.mode, 2048)
        ref = L.encode_fc1(lay, rec, codes, None, None).reshape(n, -1)
        need = L.MARKER_BYTES + lay.code_bytes(2048, rec.mode)
        got = buf[:n, nl + rec.row_offset : nl + rec.row_offset + rec.rows].reshape(
            n, -1
        )
        assert torch.equal(got[:, :need], ref[:, :need])
    assert not buf[n:].any() and not buf[:, :nl].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("bits", [1, 2])
def test_capture_kv_into_matches_torch_encoder(monkeypatch, bits):
    from nemo_rl.quantization.trace.encode_triton import capture_kv_into

    monkeypatch.setenv(L.KV_LAYERS_ENV, "3")
    monkeypatch.setenv(L.MODE_ENV, "lsb")
    monkeypatch.setenv(L.KV_BITS_ENV, str(bits))
    hk, d, bs, nb = 4, 128, 16, 9
    lay = L.layout_from_env(topk=8, hidden_size=2048, num_kv_heads=hk, head_dim=d)
    rec = lay.kv_records[3]
    cache = torch.randint(
        0, 256, (nb, 2 * hk, bs, d // 2 + d // 16), dtype=torch.uint8, device="cuda"
    )
    slots = torch.tensor([5, 17, -1, 100, 33, 143, 0, 66], device="cuda")
    n, nl = slots.shape[0], 48
    buf = torch.zeros((16, nl + lay.extra_rows, 8), dtype=torch.uint8, device="cuda")
    capture_kv_into(buf, (nl + rec.row_offset) * 8, cache, slots, hk, d, rec.mode)
    # Torch replica of trace_capture.capture_kv's gather.
    valid = slots >= 0
    sl = slots.clamp(min=0)
    blk, off = sl // bs, sl % bs

    def split(side):  # copy of vllm.utils.torch_utils.nvfp4_split_data_scale
        full = side.shape[3]
        dd, sd = full * 8 // 9, full - full * 8 // 9
        b, h1, n2 = side.shape[:3]
        data = torch.as_strided(
            side,
            (b, h1, n2, dd),
            (
                side.stride(0),
                side.stride(1) * dd // full,
                side.stride(2) * dd // full,
                1,
            ),
            storage_offset=side.storage_offset(),
        )
        sc = torch.as_strided(
            side,
            (b, h1, n2, sd),
            (
                side.stride(0),
                side.stride(1) * sd // full,
                side.stride(2) * sd // full,
                1,
            ),
            storage_offset=side.storage_offset() + h1 * n2 * dd,
        )
        return data, sc

    s_dim = d // 16
    kd, ks = split(cache[:, :hk])
    vd, vs = split(cache[:, hk:])
    k_d, k_s = kd[blk, :, off], ks[blk, :, off]
    flat = (
        (off // 4).unsqueeze(1) * (4 * s_dim)
        + torch.arange(s_dim, device="cuda") * 4
        + (off % 4).unsqueeze(1)
    )
    vsf_t = vs.permute(0, 2, 3, 1)
    v_s = vsf_t[blk.unsqueeze(1), flat // s_dim, flat % s_dim].permute(0, 2, 1)
    v_d = vd[blk, :, off]
    ref = L.encode_kv(lay, rec, k_d, k_s, v_d, v_s, valid).reshape(n, -1)
    got = buf[:n, nl + rec.row_offset : nl + rec.row_offset + rec.rows].reshape(n, -1)
    need = lay.kv_bytes(rec.mode)
    assert torch.equal(got[valid, :need], ref[valid, :need])
    assert torch.equal(got[~valid, :4], ref[~valid, :4])
