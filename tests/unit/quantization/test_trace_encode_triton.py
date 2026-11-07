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
"""The fused fc1 encoder writes exactly the bytes of the torch encoder."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("hidden,topk", [(2048, 8), (2688, 6)])
@pytest.mark.parametrize("n", [1, 48, 777])
def test_fused_fc1_matches_torch(hidden, topk, n):
    from nemo_rl.quantization.trace import layout as L
    from nemo_rl.quantization.trace.encode_triton import encode_fc1_into

    g = torch.Generator(device="cuda").manual_seed(n + hidden)
    rows = -(-(L.MARKER_BYTES + 4 + hidden // 16 + hidden // 8) // topk)
    rec = L.SiteRecord(layer=0, row_offset=3, rows=rows)
    layout = type("Lay", (), {"topk": topk})()
    # Producer tensors padded past H like the quantize kernel's outputs.
    codes = torch.randint(0, 256, (n, hidden // 2 + 64), dtype=torch.uint8, device="cuda", generator=g)
    sf = torch.randint(0, 256, (n, hidden // 16 + 8), dtype=torch.uint8, device="cuda", generator=g)
    tok = torch.rand(n, device="cuda", generator=g) * 7 + 1e-3

    want = L.encode_fc1(layout, rec, codes[:, : hidden // 2], sf[:, : hidden // 16], tok)

    total_rows = rec.row_offset + rows + 5
    buf = torch.full((n + 3, total_rows, topk), 0x3C, dtype=torch.uint8, device="cuda")
    ref = buf.clone()
    ref[:n, rec.row_offset : rec.row_offset + rows] = want
    out = buf[:n].view(n, -1)[:, rec.row_offset * topk : (rec.row_offset + rows) * topk]
    encode_fc1_into(out, codes, sf, tok, hidden)
    torch.testing.assert_close(buf, ref, rtol=0, atol=0)


def test_fused_fc1_under_cuda_graph():
    from nemo_rl.quantization.trace.encode_triton import encode_fc1_into

    n, hidden, rec_bytes = 16, 2048, 392
    codes = torch.randint(0, 256, (n, hidden // 2), dtype=torch.uint8, device="cuda")
    sf = torch.randint(0, 256, (n, hidden // 16), dtype=torch.uint8, device="cuda")
    tok = torch.rand(n, device="cuda")
    out = torch.zeros((n, rec_bytes), dtype=torch.uint8, device="cuda")
    encode_fc1_into(out, codes, sf, tok, hidden)  # compile outside capture
    first = out.clone()
    out.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        encode_fc1_into(out, codes, sf, tok, hidden)
    graph.replay()
    torch.testing.assert_close(out, first, rtol=0, atol=0)
    codes.random_(0, 256)
    graph.replay()
    expect = torch.zeros_like(out)
    encode_fc1_into(expect, codes, sf, tok, hidden)
    torch.testing.assert_close(out, expect, rtol=0, atol=0)
