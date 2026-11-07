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
"""TRACE side transport: rows filed per engine step come back per request."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

R, K = 3, 8  # extra rows x topk
ROW = R * K


def _truth(seed: int, n: int) -> np.ndarray:
    """Payload a request "computes" at positions [0, n): depends on the prompt
    seed for prompt positions (shared prompts give equal rows) and is random
    elsewhere."""
    g = np.random.default_rng(seed)
    return g.integers(0, 256, size=(n, ROW), dtype=np.uint8)


def _run(monkeypatch, steps, outputs, truth):
    from nemo_rl.models.generation.vllm.quantization import trace_capture as tc

    buf = torch.zeros((4096, R, K), dtype=torch.uint8, device="cuda")
    monkeypatch.setattr(tc, "_TRACE_BUF", buf)
    store = tc.SideStore(ROW)
    for batch in steps:  # batch: list of (internal id, start, count)
        qsl = [0]
        for rid, start, cnt in batch:
            rows = truth[rid][start : start + cnt]
            buf[qsl[-1] : qsl[-1] + cnt] = torch.from_numpy(rows).view(cnt, R, K).cuda()
            qsl.append(qsl[-1] + cnt)
        store.on_step([b[0] for b in batch], [b[1] for b in batch], qsl)
        # The next forward overwrites the buffer right away.
        buf.fill_(0xEE)
    return store.collect(outputs)


def _out(ext, prompt, n_gen):
    return SimpleNamespace(
        request_id=ext,
        prompt_token_ids=list(prompt),
        outputs=[SimpleNamespace(token_ids=list(range(n_gen)))],
    )


def test_chunked_prefill_decode_and_prefix_donor(monkeypatch):
    prompt = list(range(300))  # 300-token prompt shared by a and b
    p_rows = _truth(0, 300)
    a = np.concatenate([p_rows, _truth(1, 50)])
    b = np.concatenate([p_rows, _truth(2, 50)])
    c = _truth(3, 40)  # different 10-token prompt
    truth = {"0-aaaa": a, "1-bbbb": b, "2-cccc": c}
    steps = [
        [("0-aaaa", 0, 200)],  # chunked prefill
        [("0-aaaa", 200, 100), ("2-cccc", 0, 10)],
        # b hits the prefix cache for its first 288 prompt tokens.
        [("1-bbbb", 288, 12), ("0-aaaa", 300, 1), ("2-cccc", 10, 1)],
    ]
    for t in range(1, 49):
        steps.append([("0-aaaa", 300 + t, 1), ("1-bbbb", 300 + t - 1, 1)])
    steps.append([("1-bbbb", 348, 1)])
    for t in range(11, 39):
        steps.append([("2-cccc", t, 1)])
    outputs = [_out("0", prompt, 50), _out("1", prompt, 50), _out("2", range(10), 30)]
    arrays, stats = _run(monkeypatch, steps, outputs, truth)
    assert stats["missing"] == 0
    assert stats["prefix_filled"] == 288
    np.testing.assert_array_equal(arrays[0], a[:349])
    np.testing.assert_array_equal(arrays[1], b[:349])
    np.testing.assert_array_equal(arrays[2], c[:39])


def test_preemption_recompute_overwrites(monkeypatch):
    t = _truth(5, 80)
    stale = t.copy()
    stale[40:60] ^= 0xFF
    from nemo_rl.models.generation.vllm.quantization import trace_capture as tc

    buf = torch.zeros((4096, R, K), dtype=torch.uint8, device="cuda")
    monkeypatch.setattr(tc, "_TRACE_BUF", buf)
    store = tc.SideStore(ROW)

    def step(rid, src, start, cnt):
        buf[:cnt] = torch.from_numpy(src[start : start + cnt]).view(cnt, R, K).cuda()
        store.on_step([rid], [start], [0, cnt])

    step("7-x", t, 0, 40)
    for i in range(20):
        step("7-x", stale, 40 + i, 1)
    step("7-x", t, 0, 79)  # preempted, recomputed from position 0
    arrays, stats = store.collect([_out("7", range(30), 50)])
    assert stats["missing"] == 0
    np.testing.assert_array_equal(arrays[0], t[:79])


def test_uncaptured_positions_are_zero_and_counted(monkeypatch):
    t = _truth(9, 20)
    arrays, stats = _run(
        monkeypatch, [[("3-z", 0, 10)]], [_out("3", range(5), 16)], {"3-z": t}
    )
    assert stats["missing"] == 10
    np.testing.assert_array_equal(arrays[0][:10], t[:10])
    assert not arrays[0][10:].any()
