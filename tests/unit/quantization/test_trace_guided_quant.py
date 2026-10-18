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

from nemo_rl.quantization.trace.codec import (
    fp4_lsb_from_packed,
    pack_bits,
    pack_fp4_nibbles,
    unpack_bits,
    unpack_fp4_nibbles,
)
from nemo_rl.quantization.trace.guided_quant import (
    E2M1_MAGNITUDES,
    guided_e2m1,
    nvfp4_rowscaled_quantize_ref,
    trace_nudge_rowscaled,
)

MAGS = torch.tensor(E2M1_MAGNITUDES)


def _decode(codes: torch.Tensor) -> torch.Tensor:
    v = MAGS.to(codes.device)[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -v, v)


def test_codec_roundtrip():
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 16, (5, 64), generator=g, dtype=torch.uint8)
    packed = pack_fp4_nibbles(codes)
    assert torch.equal(unpack_fp4_nibbles(packed), codes)
    bits = torch.randint(0, 2, (5, 64), generator=g).bool()
    assert torch.equal(unpack_bits(pack_bits(bits)), bits)
    assert torch.equal(unpack_bits(fp4_lsb_from_packed(packed)), (codes & 1).bool())


def test_adjacent_codes_alternate_lsb():
    # The b=1 reduction relies on this property of E2M1.
    for c in range(7):
        assert (c & 1) != ((c + 1) & 1)
        assert E2M1_MAGNITUDES[c] < E2M1_MAGNITUDES[c + 1]


def test_rtn_matches_te_cast_to_fp4x2():
    ref = pytest.importorskip(
        "transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4"
    )
    grid = torch.linspace(-6, 6, 4801)
    special = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.0, -0.0, 6.0])
    x = torch.cat([grid, special, -special]).view(1, -1)
    x = x[:, : x.shape[1] // 2 * 2]
    _, rtn = guided_e2m1(x)
    assert torch.equal(pack_fp4_nibbles(rtn), ref.cast_to_fp4x2(x))


@pytest.mark.parametrize("lsb", [0, 1])
def test_guided_picks_bracket_neighbor_with_rollout_lsb(lsb):
    x = torch.linspace(-6, 6, 2401).view(1, -1)
    guide = torch.full_like(x, lsb, dtype=torch.bool)
    codes, rtn = guided_e2m1(x, guide)
    on_grid = torch.isin(x.abs(), MAGS)
    # Off-grid values must take the requested LSB and stay within the bracket.
    assert torch.all(((codes & 1) == lsb) | on_grid)
    lo = MAGS[(torch.bucketize(x.abs(), MAGS, right=True) - 1).clamp(0, 7)]
    hi = MAGS[(torch.bucketize(x.abs(), MAGS, right=True)).clamp(0, 7)]
    dq = _decode(codes).abs()
    assert torch.all((dq == lo) | (dq == hi))
    assert torch.equal(codes[on_grid], rtn[on_grid])
    # Sign always follows the training value.
    assert torch.equal((codes & 8) != 0, x < 0)


def test_guided_with_rtn_lsb_is_identity():
    g = torch.Generator().manual_seed(1)
    x = (torch.rand(64, 256, generator=g) * 12 - 6)
    _, rtn = guided_e2m1(x)
    codes, _ = guided_e2m1(x, (rtn & 1).bool())
    assert torch.equal(codes, rtn)


def test_paper_boundary_example():
    # 60.24 vs 59.76 straddle the 2.5 boundary under a 24.0 scale.
    train = torch.tensor([[60.24 / 24.0, 0.0]])
    roll = torch.tensor([[59.76 / 24.0, 0.0]])
    _, roll_codes = guided_e2m1(roll)
    _, train_rtn = guided_e2m1(train)
    guided, _ = guided_e2m1(train, (roll_codes & 1).bool())
    assert int(train_rtn[0, 0]) != int(roll_codes[0, 0])
    assert int(guided[0, 0]) == int(roll_codes[0, 0])


def _make_x(m, k, seed, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g) * torch.logspace(-2, 2, m).view(m, 1)
    x[0] = 0.0  # all-zero row
    x[1, :16] = 0.0  # all-zero block
    return x.to(torch.bfloat16).to(device)


def test_ref_matches_te_reference_quantizer():
    ref = pytest.importorskip(
        "transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4"
    )
    x = _make_x(32, 256, 2)
    q = nvfp4_rowscaled_quantize_ref(x)
    row_amax = x.abs().amax(dim=1).float()
    data, dec = ref.NVFP4QuantizerRef._quantize_blockwise_reference(
        x, row_amax, 16, 1, pow_2_scales=False, row_scaled_nvfp4=True, eps=0.0
    )
    assert torch.equal(q.data, data)
    assert torch.equal(q.decode_scale.view(torch.uint8), dec.view(torch.uint8))
    assert torch.equal(q.row_amax, row_amax)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_ref_matches_te_kernel_rowscaled():
    te_nvfp4 = pytest.importorskip("transformer_engine.pytorch.tensor.nvfp4_tensor")
    m, k = 256, 2048
    x = _make_x(m, k, 3, "cuda")
    quantizer = te_nvfp4.NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_rht=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=True,
    )
    qt = quantizer(x)
    q = nvfp4_rowscaled_quantize_ref(x)
    assert torch.equal(qt._rowwise_data.view(torch.uint8)[:m, : k // 2], q.data)
    sf = qt._rowwise_scale_inv.view(torch.uint8)
    assert not qt._with_gemm_swizzled_scales
    assert torch.equal(sf[:m, : k // 16], q.decode_scale.view(torch.uint8))
    assert torch.equal(qt._amax_rowwise.view(-1)[:m], q.row_amax)


def test_rollout_scale_path_and_valid_mask():
    x = _make_x(16, 128, 4)
    base = nvfp4_rowscaled_quantize_ref(x)
    tok_scale = base.row_amax / (6 * 448)
    # Rollout saw slightly different values: perturb, then guide training.
    roll_x = (x.float() * (1 + 1e-2 * torch.randn_like(x.float()))).to(torch.bfloat16)
    roll = nvfp4_rowscaled_quantize_ref(roll_x)
    roll_lsb = fp4_lsb_from_packed(roll.data)
    valid = torch.ones(16, dtype=torch.bool)
    valid[3] = False
    g = nvfp4_rowscaled_quantize_ref(
        x,
        guide_lsb_packed=roll_lsb,
        guide_valid=valid,
        rollout_decode_scale=base.decode_scale,
        rollout_token_scale=tok_scale,
    )
    assert g.stats.guided_elements == 15 * 128
    assert g.stats.flipped > 0
    # The invalid row keeps round-to-nearest.
    assert torch.equal(g.data[3], base.data[3])
    # Guided codes agree with rollout codes at least as often as RTN does.
    rc = unpack_fp4_nibbles(roll.data)
    agree_g = (unpack_fp4_nibbles(g.data) == rc)[valid].float().mean()
    agree_b = (unpack_fp4_nibbles(base.data) == rc)[valid].float().mean()
    assert agree_g >= agree_b


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nudge_reproduces_trace_codes_under_te(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs GPU")
    m, k = 128, 2048
    x = _make_x(m, k, 7, device)
    g = torch.Generator().manual_seed(8)
    lsb = torch.randint(0, 256, (m, k // 8), generator=g, dtype=torch.uint8).to(device)
    valid = torch.ones(m, dtype=torch.bool, device=device)
    valid[5] = False
    x_new, stats = trace_nudge_rowscaled(x, lsb, valid)
    assert stats["flipped"] > 0
    assert stats["unreproduced"] == 0
    assert stats["scale_changed"] == 0
    # Expected codes: TRACE on valid rows except block maxima; RTN elsewhere.
    q = nvfp4_rowscaled_quantize_ref(x_new)
    assert torch.equal(q.data[5], nvfp4_rowscaled_quantize_ref(x).data[5])
    if device == "cuda":
        te_nvfp4 = pytest.importorskip("transformer_engine.pytorch.tensor.nvfp4_tensor")
        quantizer = te_nvfp4.NVFP4Quantizer(
            rowwise=True, columnwise=False, with_rht=False,
            with_2d_quantization=False, stochastic_rounding=False, row_scaled_nvfp4=True,
        )
        qt = quantizer(x_new)
        assert torch.equal(qt._rowwise_data.view(torch.uint8)[:m, : k // 2], q.data)
        # Off-max elements of valid rows now carry the rollout mantissa bit,
        # except values already sitting on a codeword (zeros included).
        codes = unpack_fp4_nibbles(q.data)
        xb = x.view(m, k // 16, 16).abs()
        is_max = (xb == xb.amax(-1, keepdim=True)).view(m, k)
        ok = ((codes & 1).bool() == unpack_bits(lsb)) | is_max | ~valid.view(m, 1)
        assert ok.float().mean() > 0.99


def test_nudge_copy_reproduces_rollout_codes():
    m, k = 64, 512
    x = _make_x(m, k, 9)
    roll_x = (x.float() * (1 + 0.2 * torch.randn(m, k, generator=torch.Generator().manual_seed(10)))).to(torch.bfloat16)
    roll = nvfp4_rowscaled_quantize_ref(roll_x)
    valid = torch.ones(m, dtype=torch.bool)
    x_new, stats = trace_nudge_rowscaled(x, None, valid, guide_codes_packed=roll.data)
    assert stats["unreproduced"] == 0 and stats["scale_changed"] == 0
    got = unpack_fp4_nibbles(nvfp4_rowscaled_quantize_ref(x_new).data)
    rc = unpack_fp4_nibbles(roll.data)
    xb = x.view(m, k // 16, 16).abs()
    is_max = (xb == xb.amax(-1, keepdim=True)).view(m, k)
    same = (got == rc) | (((got & 7) == 0) & ((rc & 7) == 0))
    # Everything but block maxima and codes capped at the block max matches.
    assert same[~is_max].float().mean() > 0.97
    base = unpack_fp4_nibbles(nvfp4_rowscaled_quantize_ref(x).data)
    assert same.float().mean() > ((base == rc).float().mean() + 0.05)


def test_nudge_b2_corrects_off_by_two():
    # Rollout two codes above the training value: b=1 cannot reach it, b=2 can.
    x = torch.zeros(1, 16)
    x[0, 0] = 6.0  # block max -> scale 1
    x[0, 1] = 1.1  # RTN code 2 (1.0)
    roll = torch.zeros(1, 16)
    roll[0, 0] = 6.0
    roll[0, 1] = 2.0  # code 4, two codes up
    rp = nvfp4_rowscaled_quantize_ref(roll.to(torch.bfloat16)).data
    valid = torch.ones(1, dtype=torch.bool)
    x2, _ = trace_nudge_rowscaled(x.to(torch.bfloat16), None, valid, guide_codes_packed=rp, guide_bits=2)
    got = unpack_fp4_nibbles(nvfp4_rowscaled_quantize_ref(x2).data)
    assert int(got[0, 1]) == 4


def test_b2_transport_matches_full_code_emulation():
    from nemo_rl.quantization.trace.codec import fp4_low2_from_packed, low2_to_packed_codes

    m, k = 32, 256
    x = _make_x(m, k, 11)
    roll_x = (x.float() * (1 + 0.3 * torch.randn(m, k, generator=torch.Generator().manual_seed(12)))).to(torch.bfloat16)
    rp = nvfp4_rowscaled_quantize_ref(roll_x).data
    low2 = fp4_low2_from_packed(rp)
    assert low2.shape == (m, k // 4)
    assert torch.equal(unpack_fp4_nibbles(low2_to_packed_codes(low2)), unpack_fp4_nibbles(rp) & 3)
    valid = torch.ones(m, dtype=torch.bool)
    a, _ = trace_nudge_rowscaled(x, None, valid, guide_codes_packed=rp, guide_bits=2)
    b, _ = trace_nudge_rowscaled(x, None, valid, guide_codes_packed=low2_to_packed_codes(low2), guide_bits=2)
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("k", [2048, 768])
@pytest.mark.parametrize("mode", ["b1", "b2", "copy", "b2emu"])
def test_fast_nudge_bit_identical(k, mode):
    from nemo_rl.quantization.trace.codec import fp4_low2_from_packed, low2_to_packed_codes
    from nemo_rl.quantization.trace.nudge_triton import trace_nudge_rowscaled_fast

    m = 300
    x = _make_x(m, k, 21, "cuda")
    roll_x = (x.float() * (1 + 0.3 * torch.randn(m, k, device="cuda", generator=torch.Generator("cuda").manual_seed(22)))).to(torch.bfloat16)
    rp = nvfp4_rowscaled_quantize_ref(roll_x).data
    valid = torch.rand(m, device="cuda", generator=torch.Generator("cuda").manual_seed(23)) > 0.1
    if mode == "b1":
        lsb = fp4_lsb_from_packed(rp)
        ref, st = trace_nudge_rowscaled(x, lsb, valid)
        got, fs = trace_nudge_rowscaled_fast(x, lsb, valid, width=1)
    elif mode == "b2":
        low2 = fp4_low2_from_packed(rp)
        ref, st = trace_nudge_rowscaled(x, None, valid, guide_codes_packed=low2_to_packed_codes(low2), guide_bits=2)
        got, fs = trace_nudge_rowscaled_fast(x, low2, valid, width=2)
    elif mode == "copy":
        ref, st = trace_nudge_rowscaled(x, None, valid, guide_codes_packed=rp)
        got, fs = trace_nudge_rowscaled_fast(x, rp, valid, width=4)
    else:
        ref, st = trace_nudge_rowscaled(x, None, valid, guide_codes_packed=rp, guide_bits=2)
        got, fs = trace_nudge_rowscaled_fast(x, rp, valid, width=4, match_bits=2)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    assert int(fs["flipped_t"]) == st["flipped"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("width", [0, 1, 2, 4])
def test_fast_kv_fake_bit_identical(monkeypatch, width):
    pytest.importorskip("megatron.core")
    from nemo_rl.models.megatron import trace_probe as tp
    from nemo_rl.quantization.trace.codec import fp4_lsb_from_packed, fp4_low2_from_packed

    n, h, d, gs = 257, 4, 128, 0.0123
    g = torch.Generator("cuda").manual_seed(5)
    x = (torch.randn(n, h, d, device="cuda", generator=g) * 3).to(torch.bfloat16)
    valid = torch.rand(n, device="cuda", generator=g) > 0.2
    xr = (x.float() * (1 + 0.2 * torch.randn(n, h, d, device="cuda", generator=g))).view(n, h, d // 16, 16)
    sf = (xr.abs().amax(-1) / (6 * gs)).clamp(max=448).to(torch.float8_e4m3fn)
    z = (xr / (sf.float().unsqueeze(-1) * gs)).clamp(-6, 6).view(n * h, d)
    from nemo_rl.quantization.trace.guided_quant import guided_e2m1
    codes = pack_fp4_nibbles(guided_e2m1(z)[1]).view(n, h, d // 2)
    payload = {0: None, 1: fp4_lsb_from_packed(codes), 2: fp4_low2_from_packed(codes), 4: codes}[width]
    rsf = sf if width else None
    monkeypatch.setenv(tp.VERIFY_ENV, "1")
    ref, st = tp._nvfp4_kv_fake(x, gs, rsf, payload, valid, max(width, 1))
    monkeypatch.setenv(tp.VERIFY_ENV, "0")
    got, _ = tp._nvfp4_kv_fake(x, gs, rsf, payload, valid, max(width, 1))
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    if width:
        assert st["flipped"] > 0
