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
"""Trainer-side TRACE QAT: payload transport and rollout-guided fake quantization.

The routed experts train in BF16 on NVFP4 fake-quantized operands
(``NRL_TRACE_QAT_LAYERS``): the fc1 and fc2 inputs per token and the expert
weights per tensor, mirroring what the NVFP4 rollout quantizes. On the guided
layers (``NRL_TRACE_LAYERS``) the activation fake quantization follows the
rollout: each element takes the E2M1 neighbour whose mantissa bit matches the
rollout's code, under the rollout's own per-token and block scales.

Transport: the rollout appends TRACE rows to the routed-experts tensor (see
:mod:`nemo_rl.quantization.trace.layout`). :func:`attach_trace_rows` fetches
them per microbatch and :func:`split_trace_rows` strips them before router
replay consumes the routes, keeping the SP-local copy for the MoE layers.

* fc1: ``MoELayer.preprocess`` (routing and shared experts already consumed
  the unquantized hidden states).
* fc2: a ``linear_fc2`` forward pre-hook; the guidance travels with the tokens
  through the same dispatch (HybridEP ``dispatch_with_permute`` or a replay of
  the alltoall dispatcher's movement) and is matched to rows by expert id.
* weights: replaced in place by their fake quantization once per optimizer
  step, inside ``TEGroupedLinear.forward`` (after the parameter all-gather).

Activations use a straight-through estimator; for the weights the FP32 main
parameters take the gradient.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from nemo_rl.quantization.trace import layout as trace_layout

QAT_LAYERS_ENV = "NRL_TRACE_QAT_LAYERS"

_LAYOUT: Optional[trace_layout.TraceLayout] = None
_ROWS_LOCAL: Optional[torch.Tensor] = None  # SP-local trace rows [T, R, topk]
_ROUTES_LOCAL: Optional[torch.Tensor] = None  # SP-local R3 routes [T, L, topk]
# layer -> (fc2 slot rows, valid rows, unpadded tokens per local expert)
_FC2_GUIDE: dict[int, tuple[torch.Tensor, torch.Tensor, list[int]]] = {}
# layer -> router probs folded into the fc2 input rows
_FC2_PROBS: dict[int, torch.Tensor] = {}
_FC2_REC_BYTES_IDS = 2  # int16 expert id (+1, 0 = none) per top-k slot
_QAT_EPOCH = 0
_ATTACH_STATS = [0, 0.0, 0.0, 0, 0]
_QAT_INSTALLED = False


def qat_layers() -> tuple[int, ...]:
    return trace_layout._parse_layers(os.environ.get(QAT_LAYERS_ENV))


def _qat_layer(layer: int) -> bool:
    return layer in qat_layers()


def layout_for_megatron_config(cfg: Any) -> trace_layout.TraceLayout:
    return trace_layout.layout_from_env(
        topk=int(cfg.moe_router_topk),
        hidden_size=int(cfg.hidden_size),
        moe_intermediate_size=int(cfg.moe_ffn_hidden_size),
    )


class _Prefetcher:
    """Fetch a rank's TRACE payloads in the background, once per batch.

    The logprob and training passes of a step consume the same samples, so the
    fetched arrays are kept until a batch with different refs arrives. Fetching
    starts when the worker receives its data, overlapping the transfer with the
    forward of the first microbatches.
    """

    def __init__(self) -> None:
        import concurrent.futures

        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("NRL_TRACE_FETCH_THREADS", "4")),
            thread_name_prefix="trace-fetch",
        )
        self._futs: dict = {}

    def prefetch(self, refs) -> None:
        import ray

        live = [r for r in refs if r is not None]
        keep = set(live)
        for r in list(self._futs):
            if r not in keep:
                del self._futs[r]
        new = [r for r in live if r not in self._futs]
        # Small groups: one transfer request each, several in flight.
        for i in range(0, len(new), 4):
            chunk = new[i : i + 4]
            fut = self._pool.submit(ray.get, chunk)
            for j, r in enumerate(chunk):
                self._futs[r] = (fut, j)

    def get(self, refs) -> list:
        import ray

        out = []
        for r in refs:
            hit = self._futs.get(r)
            out.append(hit[0].result()[hit[1]] if hit is not None else ray.get(r))
        return out


_PREFETCHER: Optional[_Prefetcher] = None


def prefetch_trace_refs(refs) -> None:
    """Start fetching all TRACE payloads of this rank's batch (non-blocking)."""
    global _PREFETCHER
    if refs is None or not trace_layout.trace_requested():
        return
    if _PREFETCHER is None:
        _PREFETCHER = _Prefetcher()
    _PREFETCHER.prefetch(refs)


def attach_trace_rows(data_dict: Any) -> None:
    """Re-attach by-reference TRACE rows onto a microbatch's routed_experts.

    The rollout ships each sample's unpadded ``[tokens, R, topk]`` uint8 trace
    payload as a Ray ObjectRef (``trace_refs``) so it never crosses the driver
    padded. Per microbatch we fetch only our samples and append the rows to
    ``routed_experts`` (``[mb, S, L, K]`` -> ``[mb, S, L + R, K]``); the
    existing padding / CP / SP / router-replay path then carries them and
    :func:`split_trace_rows` strips them again. Missing refs and padded
    positions get zero rows, which fail the marker check (no guidance).
    """
    import ray

    refs = data_dict.pop("trace_refs")
    routes = data_dict.get("routed_experts")
    if routes is None or not trace_layout.trace_requested():
        return
    mb, seq, num_layers, topk = routes.shape
    live = [(i, r) for i, r in enumerate(refs) if r is not None]
    _prof = os.environ.get("NRL_TRACE_PROF") == "1"
    if _prof:
        import time

        _t0 = time.perf_counter()
    wanted = [r for _, r in live]
    if _PREFETCHER is not None:
        arrays = _PREFETCHER.get(wanted)
    else:
        arrays = ray.get(wanted) if live else []
    if _prof:
        _t1 = time.perf_counter()
    extra = next((a.shape[1] for a in arrays), None)
    if extra is None:
        if _LAYOUT is None:
            return
        extra = _LAYOUT.extra_rows
    rows = torch.empty((mb, seq, extra, topk), dtype=torch.uint8)
    filled = [0] * mb
    for (i, _), arr in zip(live, arrays):
        n = min(arr.shape[0], seq)
        rows[i, :n] = torch.from_numpy(arr[:n])
        filled[i] = n
    for i, n in enumerate(filled):
        rows[i, n:] = 0
    rows = (
        rows.view(routes.dtype) if routes.element_size() == 1 else rows.to(routes.dtype)
    )
    data_dict["routed_experts"] = torch.cat([routes, rows], dim=2)
    if _prof:
        _ATTACH_STATS[0] += 1
        _ATTACH_STATS[1] += _t1 - _t0
        _ATTACH_STATS[2] += time.perf_counter() - _t1
        _ATTACH_STATS[3] += sum(a.nbytes for a in arrays)
        _ATTACH_STATS[4] += data_dict["routed_experts"].numel()
        if _ATTACH_STATS[0] % 8 == 0:
            n, g, b, nb, nel = _ATTACH_STATS
            print(
                f"[trace-prof] attach calls={int(n)} ray_get_s={g:.2f} build_s={b:.2f} "
                f"fetched_GB={nb / 1e9:.3f} padded_GB={nel / 1e9:.3f}",
                flush=True,
            )


def split_trace_rows(
    model: Any,
    model_config: Any,
    routed_experts: torch.Tensor,
    split_for_sp,
) -> torch.Tensor:
    """Strip TRACE rows from ``[T, L + R, K]`` and stash them; return routes."""
    global _LAYOUT, _ROWS_LOCAL, _ROUTES_LOCAL
    if qat_layers():
        _install_qat(model)
    if not trace_layout.trace_requested():
        return routed_experts
    if _LAYOUT is None:
        _LAYOUT = layout_for_megatron_config(model_config)
    r = _LAYOUT.extra_rows
    total = int(getattr(model_config, "num_layers"))
    _FC2_GUIDE.clear()
    _FC2_PROBS.clear()
    if routed_experts.shape[1] == total:
        # No trace rows were attached (no captured sample in this microbatch).
        _ROWS_LOCAL = _ROUTES_LOCAL = None
        return routed_experts
    if routed_experts.shape[1] != total + r:
        raise ValueError(
            f"TRACE: routed_experts has {routed_experts.shape[1]} layer rows, "
            f"expected num_layers={total} + extra_rows={r}"
        )
    _ROWS_LOCAL = split_for_sp(model_config, routed_experts[:, total:, :])
    _ROUTES_LOCAL = split_for_sp(model_config, routed_experts[:, :total, :])
    return routed_experts[:, :total, :]


def _fake_quant(
    x: torch.Tensor,
    lsb: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
    token_scale: Optional[torch.Tensor] = None,
    block_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-token NVFP4 fake quantization of ``x`` [M, K], straight-through.

    Rows with ``valid`` guidance round toward the rollout's mantissa bits
    under the rollout's scales; the others round to nearest.
    """
    from nemo_rl.quantization.trace.nudge_triton import nvfp4_fakequant_fast

    with torch.no_grad():
        q, _ = nvfp4_fakequant_fast(
            x.detach(),
            lsb,
            valid,
            rollout_token_scale=token_scale,
            rollout_block_scale=block_scale,
        )
    return x + (q - x).detach()


# -- fc1 ------------------------------------------------------------------------


def _fc1(layer: int, hidden_states: torch.Tensor) -> torch.Tensor:
    shape = hidden_states.shape
    x = hidden_states.reshape(-1, shape[-1])
    rows = _ROWS_LOCAL
    if (
        _LAYOUT is None
        or rows is None
        or rows.shape[0] != x.shape[0]
        or layer not in _LAYOUT.fc1_records
    ):
        return _fake_quant(x).view(shape)
    dec = trace_layout.decode_fc1(_LAYOUT, rows, layer)
    return _fake_quant(x, dec.lsb, dec.valid, dec.token_scale, dec.block_scale).view(
        shape
    )


def _install_fc1_patch() -> None:
    from megatron.core.transformer.moe.moe_layer import MoELayer

    orig = MoELayer.preprocess

    def preprocess(self, hidden_states, probs, routing_map):
        # Routing (and shared experts) already consumed the unquantized hidden
        # states; only the copy dispatched to the routed experts is quantized.
        layer = self.layer_number - 1
        if _qat_layer(layer):
            hidden_states = _fc1(layer, hidden_states)
        return orig(self, hidden_states, probs, routing_map)

    MoELayer.preprocess = preprocess


# -- fc2 (expert side, after token dispatch) ------------------------------------


def _build_fc2_side_payload(
    layer: int, n_tokens: int, hidden: int, dtype
) -> Optional[torch.Tensor]:
    """Per-token ``[marker | R3 expert ids | fc2 slots]`` packed into a bf16 view."""
    if _ROWS_LOCAL is None or _ROUTES_LOCAL is None or _ROWS_LOCAL.shape[0] != n_tokens:
        return None
    dec = trace_layout.decode_fc2(_LAYOUT, _ROWS_LOCAL, layer)
    topk = _LAYOUT.topk
    ids = _ROUTES_LOCAL[:, layer, :].to(torch.int32)
    ids = torch.where(ids >= 0, ids + 1, torch.zeros_like(ids)).to(torch.int16)
    nbytes = hidden * torch.finfo(dtype).bits // 8
    g = torch.zeros(n_tokens, nbytes, dtype=torch.uint8, device=_ROWS_LOCAL.device)
    g[:, 0:4] = torch.where(dec.valid.view(-1, 1), trace_layout.MARKER, 0).to(
        torch.uint8
    )
    o = 4
    g[:, o : o + 2 * topk] = ids.contiguous().view(torch.uint8).view(n_tokens, 2 * topk)
    o += 2 * topk
    slots = dec.slots.reshape(n_tokens, -1)
    if o + slots.shape[1] > nbytes:
        raise RuntimeError(
            f"TRACE fc2 payload {o + slots.shape[1]} B exceeds the {nbytes} B hidden row"
        )
    g[:, o : o + slots.shape[1]] = slots
    return g.view(dtype).view(n_tokens, hidden)


def _decode_fc2_side_rows(
    g: torch.Tensor, expert_of_row: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    topk, nb = _LAYOUT.topk, _LAYOUT.fc2_slot_bytes
    # Explicit width: an EP rank can receive zero rows.
    b = (
        g.contiguous()
        .view(torch.uint8)
        .view(g.shape[0], g.shape[-1] * g.element_size())
    )
    marker_ok = (b[:, 0:4] == trace_layout.MARKER).all(dim=1)
    ids = (
        b[:, 4 : 4 + 2 * topk]
        .contiguous()
        .view(torch.int16)
        .view(-1, topk)
        .to(torch.int64)
    )
    hit = ids == (expert_of_row.view(-1, 1) + 1)
    k = hit.to(torch.int8).argmax(dim=1)
    slots = b[:, 4 + 2 * topk : 4 + 2 * topk + topk * nb].view(-1, topk, nb)
    rows = slots[torch.arange(slots.shape[0], device=slots.device), k]
    return rows, marker_ok & hit.any(dim=1)


def _side_dispatch_fc2(moe_layer, hidden_states: torch.Tensor) -> None:
    layer = moe_layer.layer_number - 1
    if layer not in _LAYOUT.fc2_records:
        return
    from megatron.core.transformer.moe import fused_a2a

    mgr = getattr(moe_layer.token_dispatcher, "_comm_manager", None)
    if mgr is None:
        return  # alltoall: handled in dispatch_postprocess
    if type(mgr).__name__ != "_HybridEPManager" or mgr.handle is None:
        raise RuntimeError("TRACE fc2 supports the HybridEP and alltoall dispatchers")
    n = hidden_states.shape[0]
    g = _build_fc2_side_payload(layer, n, hidden_states.shape[-1], hidden_states.dtype)
    if g is None:
        return
    if mgr._padded_num_tokens is not None and n < mgr._padded_num_tokens:
        g = torch.cat(
            [g, g.new_zeros((mgr._padded_num_tokens - n, g.shape[-1]))], dim=0
        )
    fused = moe_layer.config.moe_permute_fusion_into_hybridep
    with torch.no_grad():
        out = fused_a2a._hybrid_ep_buffer.dispatch_with_permute(
            hidden=g,
            scaling_factor=None,
            handle=mgr.handle,
            pad_multiple=mgr.pad_multiple,
            num_permuted_tokens=mgr.num_permuted_tokens,
            **({"fuse_permute_dispatch": fused} if fused else {}),
        )
    g_rows = out[0]
    tpe = mgr.get_number_of_tokens_per_expert().to(g_rows.device)
    local = torch.repeat_interleave(
        torch.arange(tpe.numel(), device=g_rows.device), tpe
    )
    if local.numel() != g_rows.shape[0]:
        raise RuntimeError(
            f"TRACE fc2: {g_rows.shape[0]} dispatched rows vs {local.numel()} expert slots"
        )
    expert = local + int(moe_layer.local_expert_indices[0])
    _FC2_GUIDE[layer] = (*_decode_fc2_side_rows(g_rows, expert), tpe.tolist())


def _alltoall_fc2_rows(td, layer: int, n_rows: int, tokens_per_expert) -> None:
    """Replay the alltoall dispatcher's token movement on the guidance payload.

    Called after ``dispatch_postprocess`` so every split size is already on
    the host: permute by ``routing_map`` -> EP all-to-all with the same splits
    -> TP gather (SP) -> sort chunks by local expert. Pure data movement, so
    the packed bytes survive bit-exactly.
    """
    from megatron.core.tensor_parallel import (
        all_to_all,
        gather_from_sequence_parallel_region,
    )
    from megatron.core.transformer.moe.moe_utils import permute, sort_chunks_by_idxs

    n_tok = td.hidden_shape_before_permute[0]
    # Only the payload bytes travel (not a full hidden row): pure data movement
    # accepts any width; keep rows 128-byte aligned.
    need = (
        trace_layout.MARKER_BYTES
        + _FC2_REC_BYTES_IDS * _LAYOUT.topk
        + _LAYOUT.topk * _LAYOUT.fc2_slot_bytes
    )
    width = -(-need // 128) * 64
    g = _build_fc2_side_payload(layer, n_tok, width, torch.bfloat16)
    if g is None or td.drop_and_pad:
        return
    with torch.no_grad():
        g = permute(
            g,
            td.routing_map,
            num_out_tokens=td.num_out_tokens,
            fused=td.config.moe_permute_fusion,
        )[0]
        g = all_to_all(td.ep_group, g, td.output_splits, td.input_splits)
        if td.tp_size > 1:
            sizes = (
                None if td.output_splits_tp is None else td.output_splits_tp.tolist()
            )
            g = gather_from_sequence_parallel_region(
                g, group=td.tp_group, output_split_sizes=sizes
            )
        if td.num_local_experts > 1:
            g = sort_chunks_by_idxs(
                g,
                td.num_global_tokens_per_local_expert.ravel(),
                td.sort_input_by_local_experts,
                fused=td.config.moe_permute_fusion,
            )[0]
    if g.shape[0] != n_rows:
        raise RuntimeError(
            f"TRACE fc2 alltoall replay: {g.shape[0]} rows vs {n_rows} expert rows"
        )
    tpe = torch.as_tensor(tokens_per_expert, device=g.device)
    local = torch.repeat_interleave(torch.arange(tpe.numel(), device=g.device), tpe)
    expert = local + int(td.local_expert_indices[0])
    _FC2_GUIDE[layer] = (*_decode_fc2_side_rows(g, expert), tpe.tolist())


def _pad_guide_to_splits(rows, valid, unpadded: list[int], padded: list[int]):
    """Lay guidance rows out like TEGroupedMLP's per-expert quantization padding."""
    if list(unpadded) == list(padded):
        return rows, valid
    out_r = rows.new_zeros((sum(padded), rows.shape[1]))
    out_v = valid.new_zeros((sum(padded),))
    # Destination row of every source row: its expert's padded start + rank.
    u = torch.tensor(unpadded, dtype=torch.int64)
    shift = (
        torch.tensor(padded, dtype=torch.int64).cumsum(0)
        - u.cumsum(0)
        - (torch.tensor(padded) - u)
    )
    dst = (torch.arange(int(u.sum())) + torch.repeat_interleave(shift, u)).to(
        rows.device, non_blocking=True
    )
    out_r[dst] = rows
    out_v[dst] = valid
    return out_r, out_v


def _fc2(layer: int, x: torch.Tensor, m_splits) -> torch.Tensor:
    guide = _FC2_GUIDE.get(layer)
    if guide is None:
        return _fake_quant(x)
    slots, valid, unpadded = guide
    padded = list(m_splits) if m_splits is not None else unpadded
    if len(padded) != len(unpadded) or sum(padded) != x.shape[0]:
        raise RuntimeError(
            f"TRACE fc2 L{layer}: fc2 rows {x.shape[0]} / splits {padded} do not match "
            f"guidance splits {unpadded}"
        )
    slots, valid = _pad_guide_to_splits(slots, valid, unpadded, padded)
    lsb, tok, blk = trace_layout.split_fc2_slot(_LAYOUT, slots)
    probs = _FC2_PROBS.get(layer)
    if probs is not None:
        # The rollout quantizes SwiGLU(fc1) and applies the router prob after
        # GEMM2; training feeds p * SwiGLU(fc1) to fc2, so scale the rollout's
        # per-token decode scale by p.
        probs = probs.reshape(-1, 1).float()
        if probs.shape[0] != x.shape[0]:
            ones = torch.ones_like(probs[:, 0], dtype=torch.bool)
            probs, _ = _pad_guide_to_splits(probs, ones, unpadded, padded)
        tok = tok * probs.view(-1)
    return _fake_quant(x, lsb, valid, tok, blk)


def _install_fc2_hooks(model: Any) -> None:
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )

    orig_dispatch = MoELayer.dispatch

    def dispatch(self, hidden_states, probs):
        out = orig_dispatch(self, hidden_states, probs)
        if _LAYOUT is not None:
            _side_dispatch_fc2(self, hidden_states)
        return out

    MoELayer.dispatch = dispatch

    orig_dpp = MoEAlltoAllTokenDispatcher.dispatch_postprocess

    def dispatch_postprocess(self, global_input_tokens, global_probs):
        out = orig_dpp(self, global_input_tokens, global_probs)
        layer = getattr(self, "_nrl_trace_layer", None)
        if layer is not None and _LAYOUT is not None and layer in _LAYOUT.fc2_records:
            _alltoall_fc2_rows(self, layer, out[0].shape[0], out[1])
        return out

    MoEAlltoAllTokenDispatcher.dispatch_postprocess = dispatch_postprocess

    # Router probs of the fc2 input rows (rescale the rollout's fc2 scales).
    orig_mlp_fwd = TEGroupedMLP.forward

    def mlp_forward(self, hidden, tokens_per_expert, permuted_probs, *a, **k):
        layer = getattr(self, "_nrl_trace_layer", None)
        if layer is not None:
            _FC2_PROBS[layer] = permuted_probs.detach()
        return orig_mlp_fwd(self, hidden, tokens_per_expert, permuted_probs, *a, **k)

    TEGroupedMLP.forward = mlp_forward

    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for module in chunk.modules():
            if not isinstance(module, MoELayer):
                continue
            layer = module.layer_number - 1
            if not _qat_layer(layer):
                continue
            module.token_dispatcher._nrl_trace_layer = layer
            module.experts._nrl_trace_layer = layer

            def pre(mod, args, layer=layer):
                splits = args[1] if len(args) > 1 else None
                if isinstance(splits, torch.Tensor):
                    splits = splits.tolist()
                return (_fc2(layer, args[0], splits), *args[1:])

            module.experts.linear_fc2.register_forward_pre_hook(pre)


# -- weights --------------------------------------------------------------------


@torch.no_grad()
def _quantize_weights(module: Any) -> None:
    """Replace each expert weight by its per-tensor NVFP4 fake quantization.

    Same scales as the rollout refit quantizer (``amax / (6 * 448)`` global,
    E4M3 per 16 input elements); a gated fc1 is gate||up, matching the shared
    W1/W3 global scale.
    """
    from nemo_rl.quantization.trace.nudge_triton import nvfp4_fakequant_fast

    for name, p in module.named_parameters(recurse=False):
        if not name.startswith("weight") or p.dim() != 2:
            continue
        amax = p.detach().abs().amax().float().clamp_min(1e-8)
        q, _ = nvfp4_fakequant_fast(
            p.detach(), global_encode=torch.reciprocal(amax / 2688.0)
        )
        p.data.copy_(q)


def _install_weight_quant(model: Any) -> int:
    from megatron.core.extensions.transformer_engine import TEGroupedLinear
    from megatron.core.optimizer.optimizer import MegatronOptimizer
    from megatron.core.transformer.moe.moe_layer import MoELayer

    def bump(cls):
        orig = cls.__dict__["step"]

        def step(self, *a, **k):
            global _QAT_EPOCH
            out = orig(self, *a, **k)
            _QAT_EPOCH += 1
            return out

        cls.step = step

    todo = [MegatronOptimizer]
    while todo:
        cls = todo.pop()
        todo.extend(cls.__subclasses__())
        if "step" in cls.__dict__:
            bump(cls)

    # Weights are fake-quantized inside forward: after the DDP pre-hook has
    # finished this module's (overlapped) parameter all-gather.
    orig_fwd = TEGroupedLinear.forward

    def forward(self, x, m_splits, *a, **k):
        if getattr(self, "_nrl_qat_epoch", None) not in (None, _QAT_EPOCH):
            _quantize_weights(self)
            self._nrl_qat_epoch = _QAT_EPOCH
        return orig_fwd(self, x, m_splits, *a, **k)

    TEGroupedLinear.forward = forward

    n = 0
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for module in chunk.modules():
            if isinstance(module, MoELayer) and _qat_layer(module.layer_number - 1):
                for lin in (module.experts.linear_fc1, module.experts.linear_fc2):
                    lin._nrl_qat_epoch = -1
                n += 1
    return n


def _install_qat(model: Any) -> None:
    global _QAT_INSTALLED
    if _QAT_INSTALLED:
        return
    n = _install_weight_quant(model)
    _install_fc1_patch()
    _install_fc2_hooks(model)
    _QAT_INSTALLED = True
    guided = trace_layout._parse_layers(os.environ.get(trace_layout.LAYERS_ENV))
    print(
        f"[trace] QAT fake quantization on {n} MoE layers, rollout guidance on "
        f"{len(guided)}",
        flush=True,
    )
