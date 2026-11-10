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
"""Rollout-side TRACE capture (vLLM worker process).

Writes each selected layer's rollout FP4 record (see
:mod:`nemo_rl.quantization.trace.layout`) into a TRACE-only device buffer
``[max_tokens, extra_rows, topk]`` and hands every engine step's rows to a
host :class:`SideStore`, which returns them per request after ``generate``
(sync engine) or per batch of finished requests (async engine,
:func:`trace_sync_on_worker`).

Hooks (installed by :func:`apply_trace_capture_patches` / :func:`register_capturer`):

* ``RoutedExpertsCapturer.__init__``: registers the capturer here (sizes the
  buffer) and wraps ``MoERunner._forward_impl`` to remember which MoE layer is
  running; the fc1/fc2 hooks inside the kernel call cannot tell otherwise.
* fc1: vLLM's ``quantize_nvfp4_per_token_input`` as bound in the TRT-LLM
  NVFP4 experts module (no vLLM source patch).
* fc2: ``TraceFc2CaptureMixin.apply`` on NeMo-RL's per-token experts class
  (FlashInfer's ``fc2_input_out`` when available, encoded by a Triton kernel;
  otherwise the FlashInfer source overlay writes the record inside the kernel).
* ``AuxOutputWorkerConnector.prepare_output`` / ``begin_step``: per-step
  hand-off to the side store and prefix-cache reset tracking.

All writes are fixed-shape device ops into a persistent buffer, so they are
recorded correctly when vLLM captures CUDA graphs.
"""

from __future__ import annotations

import collections
import concurrent.futures
import queue
import threading
from typing import Any, Optional

import numpy as np
import torch

from nemo_rl.quantization.trace import layout as trace_layout

_LAYOUT: Optional[trace_layout.TraceLayout] = None
_CURRENT_MOE_LAYER: Optional[int] = None
# TRACE capture buffer [max_tokens, extra_rows, topk].
_TRACE_BUF: Optional[torch.Tensor] = None
_SIDE: Optional["SideStore"] = None

# fc2 capture through FlashInfer's ``fc2_input_out`` (when the installed
# FlashInfer has it): scratch shared by all layers, the request flag set by
# TraceFc2CaptureMixin around one layer's kernel call, and that call's token
# count. Falls back to the source overlay in trace_fi_fc2 otherwise.
FC2_ENV = "NRL_TRACE_FC2"  # auto (default) | api | overlay
_FC2_SCRATCH: Optional[tuple] = None
_TP_MERGE: Any = None  # vLLM TP group when TP>1 (expert parallel)
_FC2_WANTED = False
_FC2_TOKENS = 0


def fc2_capture_mode() -> str:
    """``"api"`` when FlashInfer accepts ``fc2_input_out``, else ``"overlay"``."""
    import inspect
    import os

    want = os.environ.get(FC2_ENV, "auto")
    if want not in ("auto", "api", "overlay"):
        raise ValueError(f"{FC2_ENV} must be auto, api or overlay, got {want!r}")
    if want == "overlay":
        return "overlay"
    from flashinfer import fused_moe

    has_api = (
        "fc2_input_out"
        in inspect.signature(fused_moe.trtllm_fp4_block_scale_routed_moe).parameters
    )
    if want == "api" and not has_api:
        raise RuntimeError(
            f"{FC2_ENV}=api but the installed FlashInfer has no fc2_input_out"
        )
    return "api" if has_api else "overlay"


def _install_fc2_api() -> None:
    """Pass ``fc2_input_out`` on the MoE calls TraceFc2CaptureMixin asks for.

    vLLM looks the entry points up on ``flashinfer.fused_moe`` at call time.
    """
    import functools

    from flashinfer import fused_moe

    def wrap(fn):
        if getattr(fn, "_nrl_trace", False):
            return fn

        @functools.wraps(fn)
        def call(*args, **kwargs):
            global _FC2_SCRATCH, _FC2_TOKENS
            if not _FC2_WANTED:
                return fn(*args, **kwargs)
            hidden = kwargs.get("hidden_states", args[2] if len(args) > 2 else None)
            topk, inter = kwargs["top_k"], kwargs["intermediate_size"]
            rows = _TRACE_BUF.shape[0]
            if _FC2_SCRATCH is None or _FC2_SCRATCH[0].shape[1:] != (topk, inter // 2):
                dev = _TRACE_BUF.device
                _FC2_SCRATCH = (
                    torch.empty(rows, topk, inter // 2, dtype=torch.uint8, device=dev),
                    torch.empty(rows, topk, inter // 16, dtype=torch.uint8, device=dev),
                    torch.empty(rows, topk, dtype=torch.float32, device=dev),
                )
            _FC2_TOKENS = int(hidden.shape[0])
            return fn(*args, fc2_input_out=_FC2_SCRATCH, **kwargs)

        call._nrl_trace = True  # type: ignore[attr-defined]
        return call

    for name in ("trtllm_fp4_block_scale_routed_moe", "trtllm_fp4_block_scale_moe"):
        setattr(fused_moe, name, wrap(getattr(fused_moe, name)))


def register_capturer(capturer: Any, vllm_config: Any) -> None:
    """Called from the patched ``RoutedExpertsCapturer.__init__``."""
    global _LAYOUT, _TRACE_BUF, _SIDE
    if not trace_layout.trace_requested():
        return
    global _TP_MERGE
    pc = vllm_config.parallel_config
    tp = int(pc.tensor_parallel_size)
    if tp > 1 and not pc.enable_expert_parallel:
        # TP-sharded experts quantize each fc2 row slice with its own per-token
        # scale, which the trainer (whole experts) does not reproduce.
        raise ValueError(
            "TRACE capture with vLLM tensor_parallel_size>1 requires expert "
            f"parallelism (expert_parallel_size == tensor_parallel_size), got TP={tp}"
        )
    if pc.data_parallel_size != 1:
        raise RuntimeError("TRACE capture supports vLLM data_parallel_size=1 only")
    if tp > 1:
        # Each rank fills the fc2 slots of its local experts (others stay zero);
        # rows are max-reduced onto the output rank every step.
        from vllm.distributed.parallel_state import get_tp_group

        _TP_MERGE = get_tp_group()
    from nemo_rl.models.generation.vllm.quantization import trace_fi_fc2

    fc2_mode = fc2_capture_mode()
    if fc2_mode == "overlay" and not trace_fi_fc2._INSTALLED:
        raise RuntimeError("TRACE fc2 capture requires the FlashInfer overlay")
    if fc2_mode == "api":
        _install_fc2_api()
    layout = trace_layout.layout_for_hf_config(vllm_config.model_config.hf_text_config)
    _LAYOUT = layout
    _TRACE_BUF = torch.zeros(
        (capturer.device_buffer.shape[0], layout.extra_rows, layout.topk),
        dtype=torch.uint8,
        device=capturer.device_buffer.device,
    )
    _SIDE = SideStore(layout.extra_rows * layout.topk)
    _install_side_store_hook()
    _install_layer_tracking()
    _install_fc1_capture()
    print(
        f"[trace] rollout capture active: layout={layout.to_json()} "
        f"extra_rows={layout.extra_rows} fc2={fc2_mode}",
        flush=True,
    )


def collect_trace_rows(outputs) -> Optional[list]:
    """TRACE rows ``[prompt+response-1, extra_rows, topk]`` per vLLM output.

    Returns ``None`` when capture is off.
    """
    if _SIDE is None:
        return None
    rows, _ = _SIDE.collect(outputs)
    e, k = _LAYOUT.extra_rows, _LAYOUT.topk
    return [a.reshape(a.shape[0], e, k) for a in rows]


def collect_trace_rows_on_worker(worker, items) -> Optional[list]:
    """``collective_rpc`` target for multi-process vLLM (TP>1, Ray executor).

    The side store lives in the vLLM GPU workers; TP rank 0 holds the merged
    rows. ``items``: ``(request_id, prompt_token_ids, token_ids)`` per output.
    Returns the row arrays on rank 0, None on the other ranks. The caller
    ``ray.put``s them: a ref made here and passed back through
    ``collective_rpc`` would lose its Ray ownership.
    """
    import types

    del worker
    if _SIDE is None or (_TP_MERGE is not None and not _TP_MERGE.is_first_rank):
        return None
    outs = [
        types.SimpleNamespace(
            request_id=r,
            prompt_token_ids=p,
            outputs=[types.SimpleNamespace(token_ids=t)],
        )
        for r, p, t in items
    ]
    return collect_trace_rows(outs)


def trace_sync_on_worker(collect_items, register_items, release_ids) -> Optional[list]:
    """Async-engine counterpart of :func:`collect_trace_rows_on_worker`.

    Runs in the process that owns the side store (the AsyncLLM engine core
    with TP=1, the Ray GPU workers with TP>1), reached by name through the
    worker extension's ``trace_sync``. One call per batch of requests that
    finished since the last call; it also registers the prompts of newly
    submitted requests (live prefix donors) and drops aborted requests.
    ``collect_items``: ``(request_id, prompt_token_ids, token_ids)``.

    Returns, on the output rank, one ``[n, extra_rows, topk]`` uint8 array (or
    None when empty) per collect item; the AsyncLLM process ``ray.put``s them.
    None on the other ranks or when capture is off.
    """
    if _SIDE is None or (_TP_MERGE is not None and not _TP_MERGE.is_first_rank):
        return None
    if register_items:
        _SIDE.register_prompts(register_items)
    rows: list = []
    if collect_items:
        outs = _as_outputs(collect_items)
        arrays, _ = _SIDE.collect(outs, release_all=False)
        e, k = _LAYOUT.extra_rows, _LAYOUT.topk
        rows = [a.reshape(a.shape[0], e, k) if len(a) else None for a in arrays]
    if release_ids:
        _SIDE.release(release_ids)
    return rows


def _as_outputs(items) -> list:
    import types

    return [
        types.SimpleNamespace(
            request_id=r,
            prompt_token_ids=p,
            outputs=[types.SimpleNamespace(token_ids=t)],
        )
        for r, p, t in items
    ]


def _install_side_store_hook() -> None:
    """Hand every engine step's TRACE rows to the side store.

    ``prepare_output`` runs on the main stream right after the forward, with the
    batch's request ids, per-request start positions and row boundaries -- the
    same bookkeeping vLLM uses to file its routed-experts rows.
    """
    from vllm.distributed.aux_output_connector import worker as aux_worker

    cls = aux_worker.AuxOutputWorkerConnector
    if getattr(cls, "_nrl_trace_side", False):
        return
    orig = cls.prepare_output

    def prepare_output(self, input_batch):
        if _TP_MERGE is not None and _TRACE_BUF is not None:
            n = int(input_batch.query_start_loc_np[len(input_batch.req_ids)])
            if n:
                torch.distributed.reduce(
                    _TRACE_BUF[:n],
                    dst=_TP_MERGE.ranks[0],
                    op=torch.distributed.ReduceOp.MAX,
                    group=_TP_MERGE.device_group,
                )
        pending = orig(self, input_batch)
        if pending is not None and _SIDE is not None:
            _SIDE.on_step(
                pending.request_ids, pending.token_starts, pending.query_start_loc
            )
        return pending

    orig_begin = cls.begin_step

    def begin_step(self, metadata):
        # A generation bump follows a prefix-cache reset (e.g. after refit):
        # cached prompt rows from older weights must not serve as donors.
        if (
            metadata is not None
            and _SIDE is not None
            and metadata.generation > getattr(self, "_generation", 0)
        ):
            _SIDE.reset_prefix_cache()
        return orig_begin(self, metadata)

    cls.prepare_output = prepare_output
    cls.begin_step = begin_step
    cls._nrl_trace_side = True


class _Spans:
    """Growable span table in step order: one row per (engine step, request).

    Columns: request index, first position, row count, arena segment, first
    arena row, prefix-cache generation of the step.
    """

    COLS = ("ix", "start", "count", "seg", "row", "gen")

    def __init__(self) -> None:
        self.n = 0
        self._c = {k: np.empty(1024, np.int64) for k in self.COLS}

    def col(self, k: str) -> np.ndarray:
        return self._c[k][: self.n]

    def extend(self, **cols) -> None:
        m = len(cols["ix"])
        cap = len(self._c["ix"])
        if self.n + m > cap:
            cap = max(2 * cap, self.n + m)
            for k in self.COLS:
                grown = np.empty(cap, np.int64)
                grown[: self.n] = self._c[k][: self.n]
                self._c[k] = grown
        for k in self.COLS:
            self._c[k][self.n : self.n + m] = cols[k]
        self.n += m

    def keep(self, mask: np.ndarray) -> None:
        m = int(mask.sum())
        for k in self.COLS:
            self._c[k][:m] = self._c[k][: self.n][mask]
        self.n = m


# Prompts are indexed for live prefix donors by their first tokens; a prefix
# cache hit covers at least one KV block, which is never shorter than this.
_DONOR_KEY_TOKENS = 16


def _lcp(a: np.ndarray, b: np.ndarray) -> int:
    m = min(len(a), len(b))
    neq = np.nonzero(a[:m] != b[:m])[0]
    return int(neq[0]) if len(neq) else m


class SideStore:
    """Host store of TRACE rows keyed by (vLLM request id, token position).

    Per engine step: one async D2H into a pinned ring on the engine stream.
    Every ``BATCH_STEPS`` steps a background thread appends those steps' rows
    to a host arena (one bulk copy per step) and records (request, start,
    count) spans. Rows are filed by request and position in one vectorized
    pass at ``collect``, so per-step host work is constant in the batch size.
    Positions served from the prefix cache are filled at collection time from
    a request whose prompt computed them.

    ``collect(outputs)`` (sync engine) files a whole finished batch and clears
    the arena. ``collect(outputs, release_all=False)`` (async engine) files
    and releases only those requests while others are still running; the
    arena is compacted once released rows outweigh live ones. Requests still
    running can then donate prefix-cache rows if their prompts were passed to
    ``register_prompts``.

    Every step is tagged with the prefix-cache generation (bumped by
    ``reset_prefix_cache``). A reset preempts running requests, so a request's
    rows from before its latest generation are superseded, and donors must be
    from the same generation (same weights, same cache).
    """

    SEG_BYTES = 1 << 30  # host arena segment
    BATCH_STEPS = 32  # engine steps per store-thread wakeup
    PINNED_BYTES = 512 << 20  # D2H staging ring
    FILE_THREADS = 8  # collect-time row copies (numpy releases the GIL)
    COMPACT_BYTES = 1 << 30  # released rows before the arena is compacted
    RETIRED_IDS = 1 << 16  # recently released internal ids remembered

    def __init__(self, row_bytes: int) -> None:
        self.row_bytes = row_bytes
        self._q: queue.Queue = queue.Queue()
        # Serializes the engine thread (on_step) with collection calls, which
        # may come from another thread (Ray actor RPCs with TP>1).
        self._lock = threading.RLock()
        self._segs: list[np.ndarray] = []  # arena segments [rows, row_bytes]
        self._seg_used = 0
        self._spans = _Spans()
        self._rid_ix: dict[str, int] = {}  # vLLM internal request id -> index
        self._ix_rid: dict[int, str] = {}
        self._ext_ix: dict[str, list[int]] = {}  # RequestOutput id -> indices
        self._next_ix = 0
        self._retired: collections.OrderedDict[str, None] = collections.OrderedDict()
        self._rows = 0  # arena rows, live + released
        self._dead_rows = 0
        self._gen = 0
        # Per generation, per distinct prompt: (prompt tokens, rows, have) of
        # computed prompts of collected requests -- donors for later
        # prefix-cache hits.
        self._prompt_rows: dict[
            int, dict[bytes, tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = {}
        # Prompts of running requests (async engine), indexed by first tokens.
        self._live_prompts: dict[str, np.ndarray] = {}
        self._live_index: dict[bytes, set[str]] = {}
        self._live_exact: dict[bytes, set[str]] = {}
        self._err: Optional[BaseException] = None
        self._ring = _PinnedRing(self.PINNED_BYTES)
        self._pending: list = []
        self._pool = concurrent.futures.ThreadPoolExecutor(
            self.FILE_THREADS, thread_name_prefix="trace-file"
        )
        threading.Thread(target=self._run, name="trace-side-store", daemon=True).start()

    def on_step(self, request_ids, token_starts, query_start_loc) -> None:
        n_req = len(request_ids)
        qsl = np.array(query_start_loc[: n_req + 1], dtype=np.int64)
        n = int(qsl[-1])
        if n <= 0:
            return
        starts = np.array(token_starts[:n_req], dtype=np.int64)
        src = _TRACE_BUF[:n]
        nbytes = src.numel()
        with self._lock:
            got = self._ring.alloc(nbytes, block=False)
            if got is None:  # ring full: hand over what we hold, then wait
                self._flush()
                got = self._ring.alloc(nbytes)
            host, slot = got
            # D2H on the engine stream: one op per step, ordered before the
            # next forward overwrites the buffer (1.5 MB/decode step over
            # NVLink-C2C is a few us of GPU time; a side stream costs more host
            # time than that).
            host.copy_(src.reshape(-1), non_blocking=True)
            # Recorded here (engine thread, engine stream) so a flush from a
            # collecting thread waits on the right stream.
            done = None
            if src.is_cuda:
                done = torch.cuda.Event()
                done.record()
            self._pending.append(
                (host, slot, list(request_ids), starts, qsl, self._gen, done)
            )
            # Wake the store thread once per batch of steps, not every step.
            if len(self._pending) >= self.BATCH_STEPS:
                self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        self._q.put(self._pending)
        self._pending = []

    def reset_prefix_cache(self) -> None:
        with self._lock:
            self._gen += 1
            if not self._pending and not self._q.unfinished_tasks and not self._ext_ix:
                # Nothing in flight can still need older donors.
                self._prompt_rows.clear()

    def _run(self) -> None:
        while True:
            steps = self._q.get()
            try:
                done = steps[-1][-1]
                if done is not None:
                    done.synchronize()
                for host, slot, request_ids, starts, qsl, gen, _ in steps:
                    try:
                        self._append(host, request_ids, starts, qsl, gen)
                    finally:
                        self._ring.release(slot)
            except BaseException as e:  # surfaced by collect()
                self._err = e
            finally:
                self._q.task_done()

    def _ix_for(self, rid: str) -> int:
        k = self._rid_ix.get(rid)
        if k is None:
            k = self._rid_ix[rid] = self._next_ix
            self._ix_rid[k] = rid
            self._next_ix += 1
            self._ext_ix.setdefault(rid.rsplit("-", 1)[0], []).append(k)
        return k

    def _append(self, host: torch.Tensor, request_ids, starts, qsl, gen: int) -> None:
        n = int(qsl[-1])
        seg_rows = max(self.SEG_BYTES // self.row_bytes, 1)
        if not self._segs or self._seg_used + n > len(self._segs[-1]):
            self._segs.append(np.empty((max(seg_rows, n), self.row_bytes), np.uint8))
            self._seg_used = 0
        r0 = self._seg_used
        dst = torch.from_numpy(self._segs[-1][r0 : r0 + n])
        dst.copy_(host.view(n, self.row_bytes))  # bulk copy, GIL released
        self._seg_used += n
        self._rows += n
        counts = np.diff(qsl)
        rows = r0 + qsl[:-1]
        if self._retired:
            # A step that still covers a released request (late output): its
            # rows are dead on arrival.
            late = np.array([r in self._retired for r in request_ids], bool)
            if late.any():
                self._dead_rows += int(counts[late].sum())
                request_ids = [r for r, x in zip(request_ids, late) if not x]
                starts, counts, rows = starts[~late], counts[~late], rows[~late]
        ids = np.array([self._ix_for(r) for r in request_ids], np.int64)
        self._spans.extend(
            ix=ids,
            start=starts,
            count=counts,
            seg=len(self._segs) - 1,
            row=rows,
            gen=gen,
        )

    def _file(self, outputs, lens: list[int]):
        """Scatter arena rows into per-output arrays; later writes win.

        Only rows from each request's latest generation are used. Returns the
        arrays, their have-masks, each request's latest generation (-1 if it
        has no rows) and the number of arena rows the requests hold.
        """
        arrays = [np.zeros((max(L, 0), self.row_bytes), np.uint8) for L in lens]
        haves = [np.zeros(max(L, 0), bool) for L in lens]
        req_gen = np.full(len(outputs), -1, np.int64)
        t = self._spans
        if not t.n:
            return arrays, haves, req_gen, 0
        lut = np.full(self._next_ix, -1, np.int64)
        for i, o in enumerate(outputs):
            lut[self._ext_ix.get(o.request_id, [])] = i
        req = lut[t.col("ix")]
        sel = np.nonzero(req >= 0)[0]
        if not len(sel):
            return arrays, haves, req_gen, 0
        req = req[sel]
        starts, counts = t.col("start")[sel], t.col("count")[sel]
        segs, rows, gens = t.col("seg")[sel], t.col("row")[sel], t.col("gen")[sel]
        held = int(counts.sum())
        np.maximum.at(req_gen, req, gens)
        cur = gens == req_gen[req]
        req, starts, counts = req[cur], starts[cur], counts[cur]
        segs, rows = segs[cur], rows[cur]
        tot = int(counts.sum())
        first = np.cumsum(counts) - counts
        off = np.arange(tot, dtype=np.int64) - np.repeat(first, counts)
        req = np.repeat(req, counts)
        pos = np.repeat(starts, counts) + off
        loc = np.repeat(rows, counts) + off
        seg = np.repeat(segs, counts)
        lens_a = np.array(lens, np.int64)
        ok = pos < lens_a[req]
        req, pos, loc, seg = req[ok], pos[ok], loc[ok], seg[ok]
        # Keep the last write of each (request, position) (preemption recompute).
        key = req * (int(lens_a.max(initial=0)) + 1) + pos
        _, last = np.unique(key[::-1], return_index=True)
        keep = len(key) - 1 - last
        req, pos, loc, seg = req[keep], pos[keep], loc[keep], seg[keep]
        order = np.lexsort((pos, seg, req))
        req, pos, loc, seg = req[order], pos[order], loc[order], seg[order]
        cut = np.nonzero((np.diff(req) != 0) | (np.diff(seg) != 0))[0] + 1
        bounds = zip(np.concatenate(([0], cut)), np.concatenate((cut, [len(req)])))

        def copy(ab) -> None:
            a, b = ab
            r, g, p = int(req[a]), int(seg[a]), pos[a:b]
            haves[r][p] = True
            if p[-1] - p[0] == b - a - 1:  # contiguous positions: one copy
                np.take(self._segs[g], loc[a:b], 0, arrays[r][p[0] : p[-1] + 1], "clip")
            else:
                arrays[r][p] = self._segs[g][loc[a:b]]

        list(self._pool.map(copy, bounds))
        return arrays, haves, req_gen, held

    def _fill_from_live(
        self, toks: np.ndarray, rows: np.ndarray, have: np.ndarray, gen: int, skip
    ) -> int:
        """Fill missing prompt positions from running requests' arena rows.

        Requests with the same prompt (GRPO group) first; then any running
        request sharing a prefix (e.g. the chat template).
        """
        filled = 0
        exact = self._live_exact.get(toks.tobytes(), set())
        for tier in (exact, None):
            miss = ~have
            if not miss.any():
                break
            hi = int(np.nonzero(miss)[0][-1]) + 1
            lcp_by_ix = np.zeros(self._next_ix, np.int64)
            if tier is not None:
                for ext in tier - skip:
                    lcp_by_ix[self._ext_ix.get(ext, [])] = hi
            else:
                key = toks[:_DONOR_KEY_TOKENS].tobytes()
                for ext in self._live_index.get(key, set()) - skip - exact:
                    lcp = min(_lcp(toks[:hi], self._live_prompts[ext]), hi)
                    lcp_by_ix[self._ext_ix.get(ext, [])] = lcp
            if lcp_by_ix.any():
                filled += self._fill_from_spans(lcp_by_ix, rows, have, gen)
        return filled

    def _fill_from_spans(
        self, lcp_by_ix: np.ndarray, rows: np.ndarray, have: np.ndarray, gen: int
    ) -> int:
        """Copy rows at missing positions < lcp of each donor index."""
        miss = ~have
        t = self._spans
        lc = lcp_by_ix[t.col("ix")]
        starts = t.col("start")
        sel = np.nonzero((lc > 0) & (t.col("gen") == gen) & (starts < lc))[0]
        if not len(sel):
            return 0
        starts, lc = starts[sel], lc[sel]
        counts = np.minimum(t.col("count")[sel], lc - starts)
        first = np.cumsum(counts) - counts
        off = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(first, counts)
        pos = np.repeat(starts, counts) + off
        loc = np.repeat(t.col("row")[sel], counts) + off
        seg = np.repeat(t.col("seg")[sel], counts)
        ok = miss[pos]
        pos, loc, seg = pos[ok], loc[ok], seg[ok]
        pos, idx = np.unique(pos, return_index=True)  # one source per position
        loc, seg = loc[idx], seg[idx]
        for g in np.unique(seg):
            m = seg == g
            rows[pos[m]] = self._segs[int(g)][loc[m]]
        have[pos] = True
        return len(pos)

    def register_prompts(self, items) -> None:
        """Prompts ``(request_id, prompt_token_ids)`` of submitted requests.

        Lets a request that finishes first take prefix-cache rows from a
        running one (async engine). Removed when the request is released.
        """
        with self._lock:
            for rid, toks in items:
                toks = np.asarray(toks, np.int64)
                self._live_prompts[rid] = toks
                key = toks[:_DONOR_KEY_TOKENS].tobytes()
                self._live_index.setdefault(key, set()).add(rid)
                self._live_exact.setdefault(toks.tobytes(), set()).add(rid)

    def release(self, request_ids) -> None:
        """Drop the rows of requests that will not be collected (aborted)."""
        with self._lock:
            self._sync()
            ixs = [k for r in request_ids for k in self._ext_ix.get(r, [])]
            if ixs:
                ix = self._spans.col("ix")
                self._dead_rows += int(self._spans.col("count")[np.isin(ix, ixs)].sum())
            self._retire(request_ids)
            self._maybe_compact()

    def _retire(self, request_ids) -> None:
        for ext in request_ids:
            toks = self._live_prompts.pop(ext, None)
            if toks is not None:
                for index, key in (
                    (self._live_index, toks[:_DONOR_KEY_TOKENS].tobytes()),
                    (self._live_exact, toks.tobytes()),
                ):
                    live = index.get(key)
                    if live is not None:
                        live.discard(ext)
                        if not live:
                            del index[key]
            for k in self._ext_ix.pop(ext, []):
                rid = self._ix_rid.pop(k)
                del self._rid_ix[rid]
                self._retired[rid] = None
        while len(self._retired) > self.RETIRED_IDS:
            self._retired.popitem(last=False)

    def _clear(self, prompts: bool) -> None:
        self._segs.clear()
        self._seg_used = 0
        self._spans.n = 0
        self._rid_ix.clear()
        self._ix_rid.clear()
        self._ext_ix.clear()
        self._next_ix = 0
        self._rows = self._dead_rows = 0
        if prompts:
            self._live_prompts.clear()
            self._live_index.clear()
            self._live_exact.clear()

    def _maybe_compact(self) -> None:
        if not self._ext_ix:
            self._clear(prompts=False)  # everything released: drop the arena
            return
        live_rows = self._rows - self._dead_rows
        min_dead = max(self.COMPACT_BYTES // self.row_bytes, 1)
        if self._dead_rows >= max(live_rows, min_dead):
            self._compact()

    def _compact(self) -> None:
        """Copy the live rows into fresh segments and renumber the requests."""
        t = self._spans
        live_ix = np.fromiter(self._ix_rid, np.int64, len(self._ix_rid))
        remap = np.full(self._next_ix, -1, np.int64)
        remap[live_ix] = np.arange(len(live_ix))
        t.keep(remap[t.col("ix")] >= 0)
        t.col("ix")[:] = remap[t.col("ix")]
        segs, rows, counts = t.col("seg"), t.col("row"), t.col("count")
        new_segs, new_rows = np.empty_like(segs), np.empty_like(rows)
        old, self._segs = self._segs, []
        for g in np.unique(segs):
            m = segs == g
            c = counts[m]
            first = np.cumsum(c) - c
            n = int(c.sum())
            src = np.repeat(rows[m], c) + (np.arange(n) - np.repeat(first, c))
            self._segs.append(np.take(old[int(g)], src, axis=0))
            new_segs[m] = len(self._segs) - 1
            new_rows[m] = first
        segs[:] = new_segs
        rows[:] = new_rows
        del old
        # Later steps start a fresh segment.
        self._seg_used = len(self._segs[-1]) if self._segs else 0
        self._rows = int(counts.sum())
        self._dead_rows = 0
        self._rid_ix = {rid: int(remap[k]) for rid, k in self._rid_ix.items()}
        self._ix_rid = {k: rid for rid, k in self._rid_ix.items()}
        self._ext_ix = {
            ext: [int(remap[k]) for k in ks] for ext, ks in self._ext_ix.items()
        }
        self._next_ix = len(live_ix)

    def _sync(self) -> None:
        """Wait until every captured step is in the arena."""
        self._flush()
        self._q.join()
        if self._err is not None:
            err, self._err = self._err, None
            raise RuntimeError("TRACE side store failed") from err

    def collect(
        self, outputs, release_all: bool = True
    ) -> tuple[list[Optional[np.ndarray]], dict[str, int]]:
        """Rows ``[prompt+response-1, extra_rows*topk]`` per vLLM RequestOutput.

        ``release_all`` drops the whole arena afterwards (every request of the
        engine is in ``outputs``); otherwise only these requests are released.
        """
        with self._lock:
            return self._collect(outputs, release_all)

    def _collect(self, outputs, release_all: bool):
        self._sync()
        lens = [
            len(o.prompt_token_ids) + len(o.outputs[0].token_ids) - 1 for o in outputs
        ]
        prompts = [len(o.prompt_token_ids) for o in outputs]
        arrays, haves, req_gen, held = self._file(outputs, lens)
        req_gen[req_gen < 0] = self._gen
        toks_all = [np.asarray(o.prompt_token_ids, np.int64) for o in outputs]
        # Prefix-cache hits: position p of a prompt takes the rows of any
        # request (this call, an earlier one, or a running one) of the same
        # generation whose tokens agree on [0, p] -- same tokens, same
        # weights => same codes. Shared chat-template prefixes are reused
        # across prompts.
        this_call: dict[int, list] = {}
        for i in range(len(outputs)):
            p = min(prompts[i], len(haves[i]))
            this_call.setdefault(int(req_gen[i]), []).append(
                (toks_all[i][:p], arrays[i][:p], haves[i][:p])
            )
        skip = {o.request_id for o in outputs}
        filled = 0
        for i in range(len(outputs)):
            p = min(prompts[i], len(haves[i]))
            miss = ~haves[i][:p]
            if not miss.any():
                continue
            g = int(req_gen[i])
            toks = toks_all[i][:p]
            donors = list(self._prompt_rows.get(g, {}).values()) + this_call[g]
            for d_toks, d_rows, d_have in donors:
                lcp = _lcp(toks, d_toks)
                if lcp == 0:
                    continue
                take = miss[:lcp] & d_have[:lcp]
                if take.any():
                    idx = np.nonzero(take)[0]
                    arrays[i][idx] = d_rows[idx]
                    haves[i][idx] = True
                    filled += len(idx)
                    miss[:lcp] &= ~take
                if not miss.any():
                    break
            if miss.any() and self._live_prompts:
                filled += self._fill_from_live(
                    toks, arrays[i][:p], haves[i][:p], g, skip
                )
        # Keep each distinct computed prompt for later calls (until reset).
        for i in range(len(outputs)):
            p = min(prompts[i], len(haves[i]))
            pool = self._prompt_rows.setdefault(int(req_gen[i]), {})
            key = toks_all[i][:p].tobytes()
            if p and key not in pool and haves[i][:p].all():
                pool[key] = (toks_all[i][:p], arrays[i][:p].copy(), haves[i][:p].copy())
        missing = int(sum((~h).sum() for h in haves))
        if release_all:
            self._clear(prompts=True)
        else:
            self._dead_rows += held
            self._retire(list(skip))
            self._maybe_compact()
        # Donor pools stay while a running request may still be in their
        # generation (it was admitted before the reset and not preempted yet).
        oldest = self._gen
        if self._spans.n:
            oldest = min(oldest, int(self._spans.col("gen").min()))
        for g in [g for g in self._prompt_rows if g < oldest]:
            del self._prompt_rows[g]
        stats = {"prefix_filled": filled, "missing": missing}
        return arrays, stats


class _PinnedRing:
    """FIFO pinned host buffer for the side store's D2H copies.

    Slots are released in allocation order by the store thread; ``alloc``
    blocks while the ring is full. Requests larger than the ring get their own
    pinned tensor.
    """

    def __init__(self, capacity: int) -> None:
        self._pin = torch.cuda.is_available()
        self._buf = torch.empty(capacity, dtype=torch.uint8, pin_memory=self._pin)
        self._cap = capacity
        self._live: "collections.deque[tuple[int, int]]" = collections.deque()
        self._cv = threading.Condition()

    def _fit(self, nbytes: int) -> Optional[int]:
        if not self._live:
            return 0
        head, tail = self._live[-1][1], self._live[0][0]
        if head >= tail:
            if self._cap - head >= nbytes:
                return head
            return 0 if tail >= nbytes else None
        return head if tail - head >= nbytes else None

    def alloc(
        self, nbytes: int, block: bool = True
    ) -> Optional[tuple[torch.Tensor, Any]]:
        if nbytes > self._cap // 2:
            return torch.empty(nbytes, dtype=torch.uint8, pin_memory=self._pin), None
        with self._cv:
            while (start := self._fit(nbytes)) is None:
                if not block:
                    return None
                self._cv.wait()
            slot = (start, start + nbytes)
            self._live.append(slot)
        return self._buf[slot[0] : slot[1]], slot

    def release(self, slot: Any) -> None:
        if slot is None:
            return
        with self._cv:
            assert self._live and self._live[0] == slot
            self._live.popleft()
            self._cv.notify_all()


def _install_layer_tracking() -> None:
    """Record the running MoE layer around each runner call.

    ``_forward_impl`` runs eagerly inside the opaque ``moe_forward`` custom op,
    so under CUDA-graph capture this host code runs once per captured layer and
    the per-layer record offsets are baked into the graph.
    """
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    if getattr(MoERunner, "_nrl_trace_layer", False):
        return
    orig = MoERunner._forward_impl

    def _forward_impl(self, *args, **kwargs):
        global _CURRENT_MOE_LAYER
        _CURRENT_MOE_LAYER = int(self.layer_id)
        try:
            return orig(self, *args, **kwargs)
        finally:
            _CURRENT_MOE_LAYER = None

    MoERunner._forward_impl = _forward_impl
    MoERunner._nrl_trace_layer = True


def _install_fc1_capture() -> None:
    """Capture fc1 codes from the per-token input quantize of the TRT-LLM experts."""
    from vllm.model_executor.layers.fused_moe.experts import trtllm_nvfp4_moe

    orig = trtllm_nvfp4_moe.quantize_nvfp4_per_token_input
    if getattr(orig, "_nrl_trace", False):
        return

    def quantize_nvfp4_per_token_input(hidden_states: torch.Tensor):
        out = orig(hidden_states)
        capture_fc1(*out)
        return out

    quantize_nvfp4_per_token_input._nrl_trace = True  # type: ignore[attr-defined]
    trtllm_nvfp4_moe.quantize_nvfp4_per_token_input = quantize_nvfp4_per_token_input


def capture_fc1(
    packed_codes: torch.Tensor, block_scale: torch.Tensor, token_scale: torch.Tensor
) -> None:
    if _LAYOUT is None or _CURRENT_MOE_LAYER is None:
        return
    rec = _LAYOUT.fc1_records.get(_CURRENT_MOE_LAYER)
    if rec is None:
        return
    from nemo_rl.quantization.trace.encode_triton import encode_fc1_into

    n = packed_codes.shape[0]
    k = _LAYOUT.topk
    out = _TRACE_BUF[:n].view(n, -1)[
        :, rec.row_offset * k : (rec.row_offset + rec.rows) * k
    ]
    encode_fc1_into(
        out,
        packed_codes.view(torch.uint8).reshape(n, -1),
        block_scale.view(torch.uint8).reshape(n, -1),
        token_scale,
        _LAYOUT.hidden_size,
    )


class TraceFc2CaptureMixin:
    """Mixed into NeMo-RL's per-token TRT-LLM experts class (fc2 capture)."""

    def apply(self, hidden_states: torch.Tensor, *args, **kwargs):
        rec = None
        if _LAYOUT is not None and _CURRENT_MOE_LAYER is not None:
            rec = _LAYOUT.fc2_records.get(_CURRENT_MOE_LAYER)
        if rec is None:
            return super().apply(hidden_states, *args, **kwargs)  # type: ignore[misc]
        from nemo_rl.models.generation.vllm.quantization import trace_fi_fc2

        buf = _TRACE_BUF
        if not trace_fi_fc2._INSTALLED:
            return self._apply_fc2_api(rec, hidden_states, *args, **kwargs)
        # The kernel writes [marker | slots] straight into this layer's record
        # (layout.encode_fc2 format); the pointer is baked per layer under CUDA graphs.
        trace_fi_fc2.set_fc2_out(
            buf,
            byte_offset=rec.row_offset * _LAYOUT.topk,
            row_stride=buf.shape[1] * buf.shape[2],
            record_width=_LAYOUT.moe_intermediate_size,
        )
        try:
            out = super().apply(hidden_states, *args, **kwargs)  # type: ignore[misc]
        finally:
            trace_fi_fc2.set_fc2_out(None)
        return out

    def _apply_fc2_api(self, rec, hidden_states, *args, **kwargs):
        global _FC2_WANTED
        from nemo_rl.quantization.trace.encode_triton import encode_fc2_into

        _FC2_WANTED = True
        try:
            out = super().apply(hidden_states, *args, **kwargs)  # type: ignore[misc]
        finally:
            _FC2_WANTED = False
        n, k = _FC2_TOKENS, _LAYOUT.topk
        codes, scales, tokens = (t[:n] for t in _FC2_SCRATCH)  # type: ignore[index]
        dst = _TRACE_BUF[:n].view(n, -1)[
            :, rec.row_offset * k : (rec.row_offset + rec.rows) * k
        ]
        encode_fc2_into(dst, codes, scales, tokens, _LAYOUT.moe_intermediate_size)
        return out


# -- source patches ---------------------------------------------------------

_INIT_MARKER = "NeMo-RL patch (TRACE capturer registration)"


def _patch_routed_experts_capturer(logger) -> None:
    from nemo_rl.models.generation.vllm.patches import (
        _get_vllm_file,
        _locked_file_patch,
    )

    path = _get_vllm_file("model_executor/layers/fused_moe/routed_experts_capturer.py")
    init_old = (
        "        self.tp_size = vllm_config.parallel_config.tensor_parallel_size\n"
    )
    init_new = (
        init_old + f"        # {_INIT_MARKER}\n"
        "        from nemo_rl.models.generation.vllm.quantization import trace_capture\n"
        "        trace_capture.register_capturer(self, vllm_config)\n"
    )
    with _locked_file_patch(path) as (content, write_back):
        if _INIT_MARKER in content:
            return
        if content.count(init_old) != 1:
            raise RuntimeError(
                f"TRACE patch anchor not found exactly once in {path}: {init_old!r}"
            )
        write_back(content.replace(init_old, init_new, 1))
    logger.info("Patched routed_experts_capturer.py for TRACE capture.")


def apply_trace_capture_patches(logger) -> None:
    """Install the source patches when ``NRL_TRACE_*`` requests capture."""
    if not trace_layout.trace_requested():
        return
    _patch_routed_experts_capturer(logger)
