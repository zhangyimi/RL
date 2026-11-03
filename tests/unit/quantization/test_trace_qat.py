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

from nemo_rl.quantization.trace.codec import fp4_lsb_from_packed, unpack_fp4_nibbles
from tests.unit.quantization.trace_reference import (
    dequantize_rowscaled_ref,
    nvfp4_rowscaled_quantize_ref,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _x(m, k, seed, scale=3.0):
    g = torch.Generator("cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=g) * scale
    x = x * torch.exp(torch.randn(m, 1, device="cuda", generator=g))  # per-row range
    x[:3, :32] = 0.0  # all-zero blocks
    return x.to(torch.bfloat16)


def _nz(packed: torch.Tensor) -> torch.Tensor:
    """Unpacked codes with -0 folded into +0 (dequantized zero has no sign)."""
    c = unpack_fp4_nibbles(packed)
    return torch.where((c & 7) == 0, torch.zeros_like(c), c)


@pytest.mark.parametrize("k", [2048, 768])
def test_fakequant_rtn_matches_reference(k):
    from nemo_rl.quantization.trace.nudge_triton import nvfp4_fakequant_fast

    x = _x(257, k, 1)
    got, _ = nvfp4_fakequant_fast(x)
    q = nvfp4_rowscaled_quantize_ref(x)
    ref = dequantize_rowscaled_ref(q, torch.float32)
    torch.testing.assert_close(got.float(), ref, rtol=2**-8, atol=0)
    # Re-quantizing the fake-quantized values reproduces the codes and scales.
    q2 = nvfp4_rowscaled_quantize_ref(got)
    assert torch.equal(_nz(q2.data), _nz(q.data))
    assert torch.equal(
        q2.decode_scale.view(torch.uint8), q.decode_scale.view(torch.uint8)
    )


@pytest.mark.parametrize("shape", [(1536, 2048), (2048, 768)])
def test_weight_fakequant_keeps_amax_and_is_idempotent(shape):
    from nemo_rl.quantization.trace.nudge_triton import nvfp4_fakequant_fast

    g = torch.Generator("cuda").manual_seed(4)
    w = (torch.randn(*shape, device="cuda", generator=g) * 0.02).to(torch.bfloat16)

    def fq(t):
        amax = t.abs().amax().float().clamp_min(1e-8)
        return nvfp4_fakequant_fast(t, global_encode=torch.reciprocal(amax / 2688.0))[0]

    once = fq(w)
    rel = (once.float() - w.float()).abs().amax() / w.float().abs().amax()
    assert 0 < rel < 0.2  # NVFP4 error is at most a block max / 6
    # The refit re-quantizes q(w): the global amax and every value must survive.
    assert once.abs().amax() == w.abs().amax()
    twice = fq(once)
    assert torch.equal(twice.view(torch.int16), once.view(torch.int16))


@pytest.mark.parametrize("k", [2048, 768])
def test_fakequant_with_rollout_scales_matches_reference(k):
    from nemo_rl.quantization.trace.nudge_triton import nvfp4_fakequant_fast

    m = 300
    x = _x(m, k, 5)
    g = torch.Generator("cuda").manual_seed(6)
    roll = (x.float() * (1 + 0.05 * torch.randn(m, k, device="cuda", generator=g))).to(
        torch.bfloat16
    )
    rq = nvfp4_rowscaled_quantize_ref(roll)
    lsb = fp4_lsb_from_packed(rq.data)
    valid = torch.rand(m, device="cuda", generator=g) > 0.1
    got, _ = nvfp4_fakequant_fast(
        x,
        lsb,
        valid,
        rollout_token_scale=rq.global_decode,
        rollout_block_scale=rq.decode_scale,
    )
    ref = dequantize_rowscaled_ref(
        nvfp4_rowscaled_quantize_ref(
            x,
            guide_lsb_packed=lsb,
            guide_valid=valid,
            rollout_decode_scale=rq.decode_scale,
            rollout_token_scale=rq.global_decode,
        )
    )
    torch.testing.assert_close(got.float()[valid], ref[valid], rtol=2**-8, atol=0)
    # Guidance moved some codes away from round-to-nearest.
    rtn = dequantize_rowscaled_ref(
        nvfp4_rowscaled_quantize_ref(
            x,
            rollout_decode_scale=rq.decode_scale,
            rollout_token_scale=rq.global_decode,
        )
    )
    assert (rtn[valid] != ref[valid]).any()
    # Unguided rows keep the trainer's own round-to-nearest.
    own, _ = nvfp4_fakequant_fast(x)
    assert torch.equal(got[~valid].view(torch.int16), own[~valid].view(torch.int16))


def test_guidance_needs_rollout_scales():
    from nemo_rl.quantization.trace.nudge_triton import nvfp4_fakequant_fast

    x = _x(8, 64, 7)
    lsb = torch.zeros(8, 8, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError):
        nvfp4_fakequant_fast(x, lsb, torch.ones(8, dtype=torch.bool, device="cuda"))
