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
import torch

# The store runs on CPU too (unpinned ring, synchronous copies).
DEV = "cuda" if torch.cuda.is_available() else "cpu"

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

    buf = torch.zeros((4096, R, K), dtype=torch.uint8, device=DEV)
    monkeypatch.setattr(tc, "_TRACE_BUF", buf)
    store = tc.SideStore(ROW)
    for batch in steps:  # batch: list of (internal id, start, count)
        qsl = [0]
        for rid, start, cnt in batch:
            rows = truth[rid][start : start + cnt]
            buf[qsl[-1] : qsl[-1] + cnt] = (
                torch.from_numpy(rows).view(cnt, R, K).to(DEV)
            )
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

    buf = torch.zeros((4096, R, K), dtype=torch.uint8, device=DEV)
    monkeypatch.setattr(tc, "_TRACE_BUF", buf)
    store = tc.SideStore(ROW)

    def step(rid, src, start, cnt):
        buf[:cnt] = torch.from_numpy(src[start : start + cnt]).view(cnt, R, K).to(DEV)
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


def test_shared_template_prefix_across_prompts(monkeypatch):
    """Different prompts sharing a 32-token chat template: the second prompt's
    template positions come from the prefix cache and are filled from the first."""
    tmpl = _truth(10, 32)
    a = np.concatenate([tmpl, _truth(11, 30)])  # prompt 40 = 32 template + 8
    b = np.concatenate([tmpl, _truth(12, 30)])
    truth = {"0-a": a, "1-b": b}
    pa = list(range(32)) + [100 + i for i in range(8)]
    pb = list(range(32)) + [200 + i for i in range(8)]
    steps = [[("0-a", 0, 40), ("1-b", 32, 8)]]
    steps += [[("0-a", 40 + t, 1), ("1-b", 40 + t, 1)] for t in range(19)]
    arrays, stats = _run(
        monkeypatch, steps, [_out("0", pa, 20), _out("1", pb, 20)], truth
    )
    assert stats == {"prefix_filled": 32, "missing": 0}
    np.testing.assert_array_equal(arrays[0], a[:59])
    np.testing.assert_array_equal(arrays[1], b[:59])


def test_prefix_donor_from_earlier_call_until_reset(monkeypatch):
    from nemo_rl.models.generation.vllm.quantization import trace_capture as tc

    buf = torch.zeros((4096, R, K), dtype=torch.uint8, device=DEV)
    monkeypatch.setattr(tc, "_TRACE_BUF", buf)
    store = tc.SideStore(ROW)
    p_rows = _truth(20, 16)
    first = np.concatenate([p_rows, _truth(21, 8)])
    second = np.concatenate([p_rows, _truth(22, 8)])

    def step(rid, src, start, cnt):
        buf[:cnt] = torch.from_numpy(src[start : start + cnt]).view(cnt, R, K).to(DEV)
        store.on_step([rid], [start], [0, cnt])

    prompt = list(range(16))
    step("0-a", first, 0, 23)
    arrays, stats = store.collect([_out("0", prompt, 8)])
    assert stats["missing"] == 0
    # Next call: same prompt served from the prefix cache.
    step("1-b", second, 16, 7)
    arrays, stats = store.collect([_out("1", prompt, 8)])
    assert stats == {"prefix_filled": 16, "missing": 0}
    np.testing.assert_array_equal(arrays[0], second[:23])
    # After a prefix-cache reset the old rows must not be reused.
    store.reset_prefix_cache()
    step("2-c", second, 16, 7)
    arrays, stats = store.collect([_out("2", prompt, 8)])
    assert stats == {"prefix_filled": 0, "missing": 16}


def test_pinned_ring_wraps_and_blocks():
    import threading

    from nemo_rl.models.generation.vllm.quantization.trace_capture import _PinnedRing

    ring = _PinnedRing(100)
    a, sa = ring.alloc(40)
    b, sb = ring.alloc(40)
    big, sbig = ring.alloc(60)  # over half the ring: own buffer
    assert sbig is None and big.numel() == 60
    ring.release(sa)
    c, sc = ring.alloc(30)  # 20 left at the end, wraps to the freed front
    assert sc == (0, 30)
    got = []
    t = threading.Thread(target=lambda: got.append(ring.alloc(30)[1]))
    t.start()
    t.join(0.2)
    assert t.is_alive()  # [30, 40) and [80, 100) are too small
    ring.release(sb)
    t.join(5)
    assert got == [(30, 60)]


# -- async engine: per-request collection ----------------------------------


class _Engine:
    """Drives a SideStore the way the engine's prepare_output hook does."""

    def __init__(self, monkeypatch, **store_attrs):
        from nemo_rl.models.generation.vllm.quantization import trace_capture as tc

        self.buf = torch.zeros((4096, R, K), dtype=torch.uint8, device=DEV)
        monkeypatch.setattr(tc, "_TRACE_BUF", self.buf)
        for k, v in store_attrs.items():
            monkeypatch.setattr(tc.SideStore, k, v)
        self.store = tc.SideStore(ROW)

    def step(self, batch):
        """batch: list of (internal id, source rows, start, count)."""
        qsl = [0]
        for _, src, start, cnt in batch:
            rows = torch.from_numpy(src[start : start + cnt]).view(cnt, R, K)
            self.buf[qsl[-1] : qsl[-1] + cnt] = rows.to(DEV)
            qsl.append(qsl[-1] + cnt)
        self.store.on_step([b[0] for b in batch], [b[2] for b in batch], qsl)
        self.buf.fill_(0xEE)

    def collect(self, *outputs):
        return self.store.collect(list(outputs), release_all=False)


def test_partial_collect_keeps_running_requests(monkeypatch):
    eng = _Engine(monkeypatch)
    a, b = _truth(30, 40), _truth(31, 200)
    eng.step([("0-a", a, 0, 20), ("1-b", b, 0, 20)])
    for t in range(19):
        eng.step([("0-a", a, 20 + t, 1), ("1-b", b, 20 + t, 1)])
    arrays, stats = eng.collect(_out("0", range(20), 20))
    assert stats["missing"] == 0
    np.testing.assert_array_equal(arrays[0], a[:39])
    # b keeps running after a was released.
    for t in range(19, 179):
        eng.step([("1-b", b, 20 + t, 1)])
    arrays, stats = eng.collect(_out("1", range(20), 180))
    assert stats["missing"] == 0
    np.testing.assert_array_equal(arrays[0], b[:199])
    # Everything released: the arena is gone.
    assert eng.store._rows == 0 and not eng.store._segs and not eng.store._ext_ix


def test_live_donor_from_running_group_sibling(monkeypatch):
    """GRPO siblings: b hits the cache of a's prompt and finishes first."""
    eng = _Engine(monkeypatch)
    prompt = list(range(300))
    p_rows = _truth(40, 300)
    a = np.concatenate([p_rows, _truth(41, 100)])
    b = np.concatenate([p_rows, _truth(42, 10)])
    eng.store.register_prompts([("0", prompt), ("1", prompt)])
    eng.step([("0-a", a, 0, 300)])
    eng.step([("0-a", a, 300, 1), ("1-b", b, 288, 12)])
    for t in range(1, 10):
        eng.step([("0-a", a, 300 + t, 1), ("1-b", b, 300 + t - 1, 1)])
    eng.step([("1-b", b, 309, 1)])
    arrays, stats = eng.collect(_out("1", prompt, 11))
    assert stats == {"prefix_filled": 288, "missing": 0}
    np.testing.assert_array_equal(arrays[0], b[:310])
    for t in range(10, 99):
        eng.step([("0-a", a, 300 + t, 1)])
    arrays, stats = eng.collect(_out("0", prompt, 100))
    assert stats["missing"] == 0
    np.testing.assert_array_equal(arrays[0], a[:399])


def test_live_donor_shared_template_prefix(monkeypatch):
    eng = _Engine(monkeypatch)
    tmpl = _truth(50, 32)
    a = np.concatenate([tmpl, _truth(51, 60)])
    b = np.concatenate([tmpl, _truth(52, 20)])
    pa = list(range(32)) + [100 + i for i in range(8)]
    pb = list(range(32)) + [200 + i for i in range(8)]
    eng.store.register_prompts([("0", pa), ("1", pb)])
    eng.step([("0-a", a, 0, 40), ("1-b", b, 32, 8)])
    for t in range(9):
        eng.step([("0-a", a, 40 + t, 1), ("1-b", b, 40 + t, 1)])
    arrays, stats = eng.collect(_out("1", pb, 10))
    assert stats == {"prefix_filled": 32, "missing": 0}
    np.testing.assert_array_equal(arrays[0], b[:49])


def test_generation_bump_supersedes_rows_and_donors(monkeypatch):
    """A prefix-cache reset (refit) preempts running requests: rows from
    before it are superseded and donors must come from the new generation."""
    eng = _Engine(monkeypatch)
    prompt = list(range(32))
    old_x = _truth(60, 60)  # x under the old weights
    new_p = _truth(61, 32)  # prompt rows under the new weights
    new_x = np.concatenate([new_p, _truth(62, 28)])
    y = np.concatenate([new_p, _truth(63, 50)])
    eng.store.register_prompts([("0", prompt)])
    eng.step([("0-x", old_x, 0, 32)])
    for t in range(10):
        eng.step([("0-x", old_x, 32 + t, 1)])
    eng.store.reset_prefix_cache()
    eng.store.register_prompts([("1", prompt)])
    # y recomputes the prompt under the new weights; x is recomputed from the
    # new cache (hit on [0, 32)) and continues.
    eng.step([("1-y", y, 0, 32)])
    eng.step([("0-x", new_x, 32, 11), ("1-y", y, 32, 1)])
    for t in range(16):
        eng.step([("0-x", new_x, 43 + t, 1), ("1-y", y, 33 + t, 1)])
    arrays, stats = eng.collect(_out("0", prompt, 28))
    assert stats == {"prefix_filled": 32, "missing": 0}
    np.testing.assert_array_equal(arrays[0], new_x[:59])


def test_old_generation_pool_not_reused(monkeypatch):
    eng = _Engine(monkeypatch)
    prompt = list(range(16))
    first = _truth(70, 24)
    eng.step([("0-a", first, 0, 23)])
    eng.collect(_out("0", prompt, 8))
    eng.store.reset_prefix_cache()
    second = _truth(71, 24)
    eng.step([("1-b", second, 16, 7)])
    arrays, stats = eng.collect(_out("1", prompt, 8))
    assert stats == {"prefix_filled": 0, "missing": 16}


def test_compaction_keeps_live_rows_exact(monkeypatch):
    # Tiny segments and threshold: compaction and segment turnover happen
    # many times while requests start and finish at different steps.
    eng = _Engine(monkeypatch, SEG_BYTES=ROW * 64, COMPACT_BYTES=ROW * 32)
    rng = np.random.default_rng(0)
    truth, length, pos, started = {}, {}, {}, 0
    running: list[str] = []
    done = 0
    for step in range(400):
        while len(running) < 6:
            ext = str(started)
            truth[ext] = _truth(100 + started, 90)
            length[ext] = int(rng.integers(10, 80))
            pos[ext] = 0
            running.append(ext)
            started += 1
        batch = []
        for ext in running:
            cnt = 10 if pos[ext] == 0 else 1  # prefill, then decode
            batch.append((f"{ext}-r", truth[ext], pos[ext], cnt))
            pos[ext] += cnt
        eng.step(batch)
        for ext in list(running):
            if pos[ext] >= 10 + length[ext] - 1:
                arrays, stats = eng.collect(_out(ext, range(10), length[ext]))
                assert stats["missing"] == 0
                np.testing.assert_array_equal(arrays[0], truth[ext][: pos[ext]])
                running.remove(ext)
                done += 1
    assert done > 20
    live = sum(pos[e] for e in running)
    eng.store._sync()  # append the steps still staged
    assert eng.store._rows - eng.store._dead_rows == live
    assert eng.store._rows <= 2 * live + 64  # compacted, not growing


def test_release_aborted_and_late_steps(monkeypatch):
    eng = _Engine(monkeypatch)
    a, b = _truth(80, 30), _truth(81, 30)
    eng.store.register_prompts([("0", range(10)), ("1", range(10))])
    eng.step([("0-a", a, 0, 10), ("1-b", b, 0, 10)])
    eng.store.release(["0"])
    eng.step([("0-a", a, 10, 1), ("1-b", b, 10, 1)])  # late step of a
    assert "0" not in eng.store._ext_ix and "0" not in eng.store._live_prompts
    arrays, stats = eng.collect(_out("1", range(10), 2))
    assert stats["missing"] == 0
    np.testing.assert_array_equal(arrays[0], b[:11])
    assert eng.store._rows == 0 and not eng.store._ext_ix


def test_trace_sync_on_worker(monkeypatch):
    from nemo_rl.models.generation.vllm.quantization import trace_capture as tc

    eng = _Engine(monkeypatch)
    monkeypatch.setattr(tc, "_SIDE", eng.store)
    monkeypatch.setattr(tc, "_LAYOUT", SimpleNamespace(extra_rows=R, topk=K))
    monkeypatch.setattr(tc, "_TP_MERGE", None)
    a = _truth(90, 20)
    assert tc.trace_sync_on_worker([], [("0", list(range(8)))], []) == []
    eng.step([("0-a", a, 0, 12)])
    rows = tc.trace_sync_on_worker(
        [("0", list(range(8)), list(range(5))), ("9", [1], [])], [], []
    )
    assert rows[0].shape == (12, R, K) and rows[1] is None
    np.testing.assert_array_equal(rows[0].reshape(12, ROW), a[:12])
    assert "0" not in eng.store._live_prompts
