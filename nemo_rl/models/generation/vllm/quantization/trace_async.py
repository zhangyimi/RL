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
"""TRACE rows for the async vLLM worker (AsyncLLM).

AsyncLLM always runs the engine core in another process, so the side store
(:mod:`trace_capture`) is never in the worker's process: rows come back
through ``collective_rpc("trace_sync", ...)``. Requests finish one at a time;
this client batches every request that finished while the previous RPC was in
flight into the next one, piggybacking the prompts of newly submitted
requests (live prefix donors) and the ids of aborted ones. The rows are
``ray.put`` here, so ``trace_ref`` matches the sync worker's output.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

TraceRpc = Callable[[list, list, list], Awaitable[Any]]


class AsyncTraceCollector:
    """Batches TRACE collection of finished async-engine requests.

    ``rpc(collect_items, register_items, release_ids)`` returns the
    per-worker results of ``trace_sync``; the output rank returns one array
    (or None) per collect item, the others None.
    """

    def __init__(self, rpc: TraceRpc, put: Optional[Callable[[Any], Any]] = None):
        if put is None:
            import ray

            put = ray.put
        self._rpc = rpc
        self._put = put
        self._collect: list[tuple[tuple, asyncio.Future]] = []
        self._register: list[tuple[str, list[int]]] = []
        self._release: list[str] = []
        self._task: Optional[asyncio.Task] = None

    def register(self, request_id: str, prompt_token_ids) -> None:
        """Announce a request before it is submitted (sent with the next RPC)."""
        self._register.append((request_id, list(prompt_token_ids)))

    async def collect(self, request_id: str, prompt_token_ids, token_ids) -> Any:
        """Ray ref of the finished request's rows (None when it has none)."""
        fut = asyncio.get_running_loop().create_future()
        item = (request_id, list(prompt_token_ids), list(token_ids))
        self._collect.append((item, fut))
        self._kick()
        return await fut

    def release(self, request_id: str) -> None:
        """Drop the rows of a request that will not be collected (aborted)."""
        self._release.append(request_id)
        self._kick()

    def _kick(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._drain())

    async def _drain(self) -> None:
        while self._collect or self._release:
            batch, self._collect = self._collect, []
            register, self._register = self._register, []
            release, self._release = self._release, []
            try:
                results = await self._rpc([i for i, _ in batch], register, release)
                if not batch:
                    continue
                rows = next((r for r in results if r is not None), None)
                if rows is None:
                    raise RuntimeError(
                        "TRACE is enabled but no vLLM worker returned trace rows"
                    )
                if len(rows) != len(batch):
                    raise RuntimeError(
                        f"TRACE: {len(rows)} row sets for {len(batch)} requests"
                    )
                put = self._put
                refs = await asyncio.to_thread(
                    lambda: [put(a) if a is not None and len(a) else None for a in rows]
                )
                del rows, results
                for (_, fut), ref in zip(batch, refs):
                    if not fut.done():
                        fut.set_result(ref)
            except BaseException as e:
                for _, fut in batch:
                    if fut.done():
                        continue
                    if isinstance(e, Exception):
                        fut.set_exception(e)
                    else:
                        fut.cancel()
                if not isinstance(e, Exception):
                    raise
