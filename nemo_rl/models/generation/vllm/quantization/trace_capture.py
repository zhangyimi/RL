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
host :class:`SideStore`, which returns them per request after ``generate``.

Hooks (installed by :func:`apply_trace_capture_patches` / :func:`register_capturer`):

* ``RoutedExpertsCapturer.__init__``: registers the capturer here (sizes the
  buffer) and wraps ``MoERunner._forward_impl`` to remember which MoE layer is
  running; the fc1/fc2 hooks inside the kernel call cannot tell otherwise.
* fc1: vLLM's ``quantize_nvfp4_per_token_input`` as bound in the TRT-LLM
  NVFP4 experts module (no vLLM source patch).
* fc2: ``TraceFc2CaptureMixin.apply`` on NeMo-RL's per-token experts class
  (the FlashInfer overlay writes the record from inside the MoE kernel).
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


def register_capturer(capturer: Any, vllm_config: Any) -> None:
    """Called from the patched ``RoutedExpertsCapturer.__init__``."""
    global _LAYOUT, _TRACE_BUF, _SIDE
    if not trace_layout.trace_requested():
        return
    tp = int(vllm_config.parallel_config.tensor_parallel_size)
    if tp != 1:
        # fc2 shards on TP>1 would each see only part of the record.
        raise ValueError(
            f"TRACE capture requires vLLM tensor_parallel_size=1, got {tp}"
        )
    if vllm_config.parallel_config.data_parallel_size != 1:
        raise RuntimeError("TRACE capture supports vLLM data_parallel_size=1 only")
    from nemo_rl.models.generation.vllm.quantization import trace_fi_fc2

    if not trace_fi_fc2._INSTALLED:
        raise RuntimeError("TRACE fc2 capture requires the FlashInfer overlay")
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
        f"extra_rows={layout.extra_rows}",
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


class SideStore:
    """Host store of TRACE rows keyed by (vLLM request id, token position).

    Per engine step: one async D2H into a pinned ring on the engine stream.
    Every ``BATCH_STEPS`` steps a background thread appends those steps' rows
    to a host arena (one bulk copy per step) and records (request, start,
    count) spans. Rows are filed by request and position in one vectorized
    pass at ``collect``, so per-step host work is constant in the batch size.
    Positions served from the prefix cache are filled at collection time from
    a request whose prompt computed them.
    """

    SEG_BYTES = 1 << 30  # host arena segment
    BATCH_STEPS = 32  # engine steps per store-thread wakeup
    PINNED_BYTES = 512 << 20  # D2H staging ring
    FILE_THREADS = 8  # collect-time row copies (numpy releases the GIL)

    def __init__(self, row_bytes: int) -> None:
        self.row_bytes = row_bytes
        self._q: queue.Queue = queue.Queue()
        self._segs: list[np.ndarray] = []  # arena segments [rows, row_bytes]
        self._seg_used = 0
        # Per step: (segment, first row, request ids, starts, counts).
        self._records: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
        self._rid_ix: dict[str, int] = {}
        # (prompt tokens, rows, have) of computed prompts since the last
        # prefix-cache reset: donors for later prefix-cache hits.
        self._prompt_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
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
        got = self._ring.alloc(nbytes, block=False)
        if got is None:  # ring full: hand over what we hold, then wait
            self._flush()
            got = self._ring.alloc(nbytes)
        host, slot = got
        # D2H on the engine stream: one op per step, ordered before the next
        # forward overwrites the buffer (1.5 MB/decode step over NVLink-C2C is
        # a few us of GPU time; a side stream costs more host time than that).
        host.copy_(src.reshape(-1), non_blocking=True)
        self._pending.append((host, slot, list(request_ids), starts, qsl))
        # Wake the store thread once per batch of steps, not every step.
        if len(self._pending) >= self.BATCH_STEPS:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        done = torch.cuda.Event()
        done.record()
        self._q.put((done, self._pending))
        self._pending = []

    def reset_prefix_cache(self) -> None:
        self._prompt_rows.clear()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                done, steps = item
                done.synchronize()
                for host, slot, request_ids, starts, qsl in steps:
                    try:
                        self._append(host, request_ids, starts, qsl)
                    finally:
                        self._ring.release(slot)
            except BaseException as e:  # surfaced by collect()
                self._err = e
            finally:
                self._q.task_done()

    def _append(self, host: torch.Tensor, request_ids, starts, qsl) -> None:
        n = int(qsl[-1])
        seg_rows = max(self.SEG_BYTES // self.row_bytes, 1)
        if not self._segs or self._seg_used + n > len(self._segs[-1]):
            self._segs.append(np.empty((max(seg_rows, n), self.row_bytes), np.uint8))
            self._seg_used = 0
        r0 = self._seg_used
        dst = torch.from_numpy(self._segs[-1][r0 : r0 + n])
        dst.copy_(host.view(n, self.row_bytes))  # bulk copy, GIL released
        self._seg_used += n
        ix = self._rid_ix
        ids = np.array([ix.setdefault(r, len(ix)) for r in request_ids], np.int64)
        self._records.append((len(self._segs) - 1, r0, ids, starts, np.diff(qsl)))

    def _file(self, outputs, lens: list[int]):
        """Scatter arena rows into per-output arrays; later writes win."""
        arrays = [np.zeros((max(L, 0), self.row_bytes), np.uint8) for L in lens]
        haves = [np.zeros(max(L, 0), bool) for L in lens]
        if not self._records:
            return arrays, haves
        ext = {o.request_id: i for i, o in enumerate(outputs)}
        lut = np.full(len(self._rid_ix), -1, np.int64)
        for rid, k in self._rid_ix.items():
            lut[k] = ext.get(rid.rsplit("-", 1)[0], -1)
        segs, r0s, ids, starts, counts = zip(*self._records)
        nspan = np.array([len(c) for c in counts])
        counts = np.concatenate(counts).astype(np.int64)
        ids = np.concatenate(ids)
        starts = np.concatenate(starts).astype(np.int64)
        span_seg = np.repeat(np.array(segs, np.int64), nspan)
        # First arena row of each span: record row + offset within the step.
        span_first = np.cumsum(counts) - counts
        rec_first = span_first[np.cumsum(nspan) - nspan]
        within = span_first - np.repeat(rec_first, nspan)
        span_row = np.repeat(np.array(r0s, np.int64), nspan) + within
        tot = int(counts.sum())
        off = np.arange(tot, dtype=np.int64) - np.repeat(span_first, counts)
        req = np.repeat(lut[ids], counts)
        pos = np.repeat(starts, counts) + off
        loc = np.repeat(span_row, counts) + off
        seg = np.repeat(span_seg, counts)
        lens_a = np.array(lens, np.int64)
        ok = req >= 0
        ok[ok] &= pos[ok] < lens_a[req[ok]]
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
        return arrays, haves

    def collect(self, outputs) -> tuple[list[Optional[np.ndarray]], dict[str, int]]:
        """Rows ``[prompt+response-1, extra_rows*topk]`` per vLLM RequestOutput."""
        self._flush()
        self._q.join()
        if self._err is not None:
            err, self._err = self._err, None
            raise RuntimeError("TRACE side store failed") from err
        lens = [
            len(o.prompt_token_ids) + len(o.outputs[0].token_ids) - 1 for o in outputs
        ]
        prompts = [len(o.prompt_token_ids) for o in outputs]
        arrays, haves = self._file(outputs, lens)
        # Prefix-cache hits: position p of a prompt takes the rows of any
        # request (this call, or an earlier one since the last prefix-cache
        # reset) whose tokens agree on [0, p] -- same tokens, same weights =>
        # same codes. Shared chat-template prefixes are reused across prompts.
        donors = list(self._prompt_rows)
        for i, out in enumerate(outputs):
            p = min(prompts[i], len(haves[i]))
            donors.append(
                (np.asarray(out.prompt_token_ids[:p]), arrays[i][:p], haves[i][:p])
            )
        filled = 0
        for i, out in enumerate(outputs):
            p = min(prompts[i], len(haves[i]))
            miss = ~haves[i][:p]
            if not miss.any():
                continue
            toks = np.asarray(out.prompt_token_ids[:p])
            for d_toks, d_rows, d_have in donors:
                m = min(p, len(d_toks))
                neq = np.nonzero(toks[:m] != d_toks[:m])[0]
                lcp = int(neq[0]) if len(neq) else m
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
        # Keep each distinct computed prompt for later calls (until reset).
        seen = {d[0].tobytes() for d in self._prompt_rows}
        for i, out in enumerate(outputs):
            p = min(prompts[i], len(haves[i]))
            toks = np.asarray(out.prompt_token_ids[:p])
            key = toks.tobytes()
            if p and key not in seen and haves[i][:p].all():
                seen.add(key)
                self._prompt_rows.append(
                    (toks, arrays[i][:p].copy(), haves[i][:p].copy())
                )
        missing = int(sum((~h).sum() for h in haves))
        self._segs.clear()
        self._seg_used = 0
        self._records.clear()
        self._rid_ix.clear()
        stats = {"prefix_filled": filled, "missing": missing}
        return arrays, stats


class _PinnedRing:
    """FIFO pinned host buffer for the side store's D2H copies.

    Slots are released in allocation order by the store thread; ``alloc``
    blocks while the ring is full. Requests larger than the ring get their own
    pinned tensor.
    """

    def __init__(self, capacity: int) -> None:
        self._buf = torch.empty(capacity, dtype=torch.uint8, pin_memory=True)
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
            return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True), None
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
