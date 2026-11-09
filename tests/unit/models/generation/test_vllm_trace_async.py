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
"""TRACE rows on the async vLLM worker: batching of trace_sync RPCs."""

import asyncio

import numpy as np
import pytest

from nemo_rl.models.generation.vllm.quantization.trace_async import (
    AsyncTraceCollector,
)


class _FakeEngine:
    """Records trace_sync calls; returns one array per collected request."""

    def __init__(self, ranks: int = 1, fail: bool = False):
        self.calls: list[tuple[list, list, list]] = []
        self.gate = asyncio.Event()
        self.ranks = ranks
        self.fail = fail

    async def rpc(self, collect, register, release):
        self.calls.append((collect, register, release))
        await self.gate.wait()
        if self.fail:
            return [None] * self.ranks
        rows = [
            np.full((len(p) + len(t) - 1, 2, 4), len(p), np.uint8)
            for _, p, t in collect
        ]
        rows = [r if len(r) else None for r in rows]
        # Output rank first, the others return None.
        return [rows] + [None] * (self.ranks - 1)


def _put(a):
    return ("ref", a.shape)


async def _spin():
    for _ in range(5):
        await asyncio.sleep(0)


def test_finished_requests_batch_into_one_rpc():
    async def main():
        eng = _FakeEngine(ranks=2)
        trace = AsyncTraceCollector(eng.rpc, put=_put)
        trace.register("a", [1, 2, 3])
        first = asyncio.ensure_future(trace.collect("a", [1, 2, 3], [7, 8]))
        await _spin()
        assert len(eng.calls) == 1
        # Finished while the first RPC is in flight: batched into the next.
        trace.register("b", [4, 5])
        trace.register("c", [6])
        second = asyncio.ensure_future(trace.collect("b", [4, 5], [9]))
        third = asyncio.ensure_future(trace.collect("c", [6], []))
        trace.release("d")
        await _spin()
        assert len(eng.calls) == 1
        eng.gate.set()
        refs = await asyncio.gather(first, second, third)
        assert refs == [("ref", (4, 2, 4)), ("ref", (2, 2, 4)), None]
        assert eng.calls[0] == ([("a", [1, 2, 3], [7, 8])], [("a", [1, 2, 3])], [])
        assert eng.calls[1] == (
            [("b", [4, 5], [9]), ("c", [6], [])],
            [("b", [4, 5]), ("c", [6])],
            ["d"],
        )

    asyncio.run(main())


def test_release_alone_is_sent():
    async def main():
        eng = _FakeEngine()
        eng.gate.set()
        trace = AsyncTraceCollector(eng.rpc, put=_put)
        trace.release("x")
        await trace._task
        assert eng.calls == [([], [], ["x"])]

    asyncio.run(main())


def test_missing_rows_raise_clear_error():
    async def main():
        eng = _FakeEngine(fail=True)
        eng.gate.set()
        trace = AsyncTraceCollector(eng.rpc, put=_put)
        with pytest.raises(RuntimeError, match="no vLLM worker returned trace rows"):
            await trace.collect("a", [1], [2])
        # The collector keeps serving later requests.
        eng.fail = False
        assert await trace.collect("b", [1], [2]) == ("ref", (1, 2, 4))

    asyncio.run(main())


def test_cancelled_waiter_does_not_break_batch():
    async def main():
        eng = _FakeEngine()
        trace = AsyncTraceCollector(eng.rpc, put=_put)
        a = asyncio.ensure_future(trace.collect("a", [1, 2], [3]))
        b = asyncio.ensure_future(trace.collect("b", [1, 2], [3]))
        await _spin()
        a.cancel()
        eng.gate.set()
        assert await b == ("ref", (2, 2, 4))
        with pytest.raises(asyncio.CancelledError):
            await a

    asyncio.run(main())
