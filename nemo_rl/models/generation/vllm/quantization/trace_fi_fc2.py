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
"""FlashInfer TRT-LLM MoE overlay that exposes the fc2 input code bits.

With per-token NVFP4 activations the fc2 input (SwiGLU output) is quantized
inside ``MoE::Runner::run`` (``trtllm_fused_moe_runner.cu``) into permuted rows
that never reach Python. The overlay adds a small kernel after finalize that
writes, for every expanded index ``e = token * top_k + k``, the packed code
bits of that row straight into the token's TRACE record in the routed-experts
transit buffer (marker included), so no further encode ops run per layer.

The record pointer is passed through a process-global set by the exported
``nrl_trace_set_fc2_out`` before each layer's kernel call (and cleared
after). Under CUDA-graph capture the host code runs once per captured layer,
so the per-layer persistent pointer is baked into the graph; the FFI
signature of ``trtllm_fp4_block_scale_moe`` is untouched.

The patched sources are generated from the installed FlashInfer at runtime
and built under a new module name, because the shipped
``flashinfer_jit_cache`` AOT ``.so`` would otherwise shadow any source change.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

MODULE_SUFFIX = "_nrltrace"
_MODULE: Any = None
_INSTALLED = False

_RUNNER_INCLUDE_ANCHOR = '#include "tensorrt_llm/kernels/quantization.h"\n'
_RUNNER_GLOBALS = r'''
// ---- NeMo-RL TRACE: fc2 input code-bit capture ------------------------------
namespace nrl_trace {
uint8_t* g_fc2_out = nullptr;  // record base (transit buffer + byte offset)
int64_t g_fc2_row_stride = 0;  // bytes between consecutive tokens' records
int g_fc2_bits = 1;            // 1: mantissa bit, 2: two low bits, 4: full code

// One block per expanded index e = token * topk + k. Row p of the permuted FP4
// activation holds n/2 bytes (element 2i in the low nibble). The record of a
// token is [marker x4 | slot k=0 | ... | slot k=topk-1], each slot n*bits/8
// bytes; element i sits at bit (i*bits)%8 of byte (i*bits)/8. Non-local
// experts (p < 0) get zeros.
__global__ void fc2CaptureKernel(int32_t numExpanded, int32_t topk, int32_t n, int bits,
                                 int64_t rowStride, int32_t const* e2p,
                                 uint8_t const* act, uint8_t* out) {
  int32_t e = blockIdx.x;
  if (e >= numExpanded) return;
  int32_t tok = e / topk, k = e % topk;
  int32_t p = e2p[e];
  int32_t nb = n * bits / 8;
  uint8_t* rec = out + int64_t(tok) * rowStride;
  if (k == 0 && threadIdx.x < 4) rec[threadIdx.x] = 0xA5;
  uint8_t* dst = rec + 4 + int64_t(k) * nb;
  uint8_t const* src = act + int64_t(p) * (n / 2);
  for (int32_t j = threadIdx.x; j < nb; j += blockDim.x) {
    uint8_t b = 0;
    if (p >= 0) {
      if (bits == 4) {
        b = src[j];
      } else if (bits == 2) {
        uint16_t w = *reinterpret_cast<uint16_t const*>(src + int64_t(j) * 2);
#pragma unroll
        for (int i = 0; i < 4; ++i) b |= uint8_t(((w >> (4 * i)) & 3u) << (2 * i));
      } else {
        uint32_t w = *reinterpret_cast<uint32_t const*>(src + int64_t(j) * 4);
#pragma unroll
        for (int i = 0; i < 8; ++i) b |= uint8_t(((w >> (4 * i)) & 1u) << i);
      }
    }
    dst[j] = b;
  }
}
}  // namespace nrl_trace

extern "C" void nrl_trace_set_fc2_out(void* p, int64_t stride, int bits) {
  nrl_trace::g_fc2_out = static_cast<uint8_t*>(p);
  nrl_trace::g_fc2_row_stride = stride;
  nrl_trace::g_fc2_bits = bits;
}
// ---------------------------------------------------------------------------
'''
_RUNNER_FINALIZE_ANCHOR = (
    "  // Run finalize\n"
    "  if (args.do_finalize) {\n"
    "    // Run finalize\n"
    "    moe::dev::finalize::run(finalizeData, stream);\n"
    "    sync_check_cuda_error(stream);\n"
    "  }\n"
)
_RUNNER_CAPTURE = r'''
  // NeMo-RL TRACE: capture fc2 input code bits (per-token NVFP4 path only).
  if (mUsePerTokenScalingGemm2 && ::nrl_trace::g_fc2_out != nullptr) {
    int32_t numExpanded = args.num_tokens * totalExpertsPerToken;
    ::nrl_trace::fc2CaptureKernel<<<numExpanded, 128, 0, stream>>>(
        numExpanded, totalExpertsPerToken, args.intermediate_size, ::nrl_trace::g_fc2_bits,
        ::nrl_trace::g_fc2_row_stride,
        reinterpret_cast<int32_t const*>(workspace.expanded_idx_to_permuted_idx),
        reinterpret_cast<uint8_t const*>(workspace.activation_output),
        ::nrl_trace::g_fc2_out);
  }
'''
_LAUNCHER_TAIL_ANCHOR = "}  // namespace flashinfer\n"
_LAUNCHER_EXPORT = r'''
extern "C" void nrl_trace_set_fc2_out(void* p, int64_t stride, int bits);
void nrl_trace_set_fc2_out_ffi(Optional<TensorView> buf, int64_t byte_offset, int64_t row_stride,
                               int64_t bits) {
  nrl_trace_set_fc2_out(
      buf.has_value() ? static_cast<uint8_t*>(buf.value().data_ptr()) + byte_offset : nullptr,
      row_stride, static_cast<int>(bits));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nrl_trace_set_fc2_out, nrl_trace_set_fc2_out_ffi);

'''


def _patch_once(text: str, anchor: str, new: str, *, before: bool = False, last: bool = False) -> str:
    count = text.count(anchor)
    if count == 0 or (count > 1 and not last):
        raise RuntimeError(f"TRACE FlashInfer overlay anchor found {count}x: {anchor[:60]!r}")
    idx = text.rfind(anchor) if last else text.find(anchor)
    if before:
        return text[:idx] + new + text[idx:]
    return text[: idx + len(anchor)] + new + text[idx + len(anchor) :]


def _write_overlay(csrc_dir, out_dir) -> tuple:
    from flashinfer.jit.utils import write_if_different

    out_dir.mkdir(parents=True, exist_ok=True)
    runner = (csrc_dir / "trtllm_fused_moe_runner.cu").read_text()
    runner = _patch_once(runner, _RUNNER_INCLUDE_ANCHOR, _RUNNER_GLOBALS)
    runner = _patch_once(runner, _RUNNER_FINALIZE_ANCHOR, _RUNNER_CAPTURE)
    launcher = (csrc_dir / "trtllm_fused_moe_kernel_launcher.cu").read_text()
    launcher = _patch_once(launcher, _LAUNCHER_TAIL_ANCHOR, _LAUNCHER_EXPORT, before=True, last=True)
    runner_path = out_dir / "trtllm_fused_moe_runner.cu"
    launcher_path = out_dir / "trtllm_fused_moe_kernel_launcher.cu"
    write_if_different(runner_path, runner)
    write_if_different(launcher_path, launcher)
    return runner_path, launcher_path


def install() -> None:
    """Route FlashInfer's TRT-LLM MoE module through the TRACE overlay.

    Must run before the first ``get_trtllm_moe_sm100_module()`` call in the
    process (that factory is ``functools.cache``-d).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    import flashinfer.fused_moe.core as fi_core
    from flashinfer.jit import env as jit_env

    if fi_core._get_trtllm_moe_sm100_module_impl.cache_info().currsize:
        raise RuntimeError(
            "TRACE fc2 overlay installed after FlashInfer loaded its MoE module"
        )
    orig_gen = fi_core.gen_trtllm_gen_fused_moe_sm100_module

    def gen(enable_rubin: bool = False):
        spec = orig_gen(enable_rubin=enable_rubin)
        csrc = jit_env.FLASHINFER_CSRC_DIR
        overlay = jit_env.FLASHINFER_GEN_SRC_DIR / f"{spec.name}{MODULE_SUFFIX}"
        runner_path, launcher_path = _write_overlay(csrc, overlay)
        swap = {
            "trtllm_fused_moe_runner.cu": runner_path,
            "trtllm_fused_moe_kernel_launcher.cu": launcher_path,
        }
        sources = [swap.get(p.name, p) for p in spec.sources]
        # The originals resolve some quoted includes relative to csrc/.
        include_dirs = [csrc, *(spec.extra_include_dirs or [])]
        new_spec = dataclasses.replace(
            spec,
            name=spec.name + MODULE_SUFFIX,
            sources=sources,
            extra_include_dirs=include_dirs,
        )
        orig_build_and_load = new_spec.build_and_load

        def build_and_load():
            global _MODULE
            _MODULE = orig_build_and_load()
            return _MODULE

        new_spec.build_and_load = build_and_load  # type: ignore[method-assign]
        return new_spec

    fi_core.gen_trtllm_gen_fused_moe_sm100_module = gen
    _INSTALLED = True
    print("[trace] FlashInfer TRT-LLM MoE routed through the fc2-capture overlay", flush=True)


def set_fc2_out(buf: Optional[Any], byte_offset: int = 0, row_stride: int = 0, bits: int = 1) -> None:
    """Make the next MoE kernel launches write fc2 records into ``buf`` (or stop: None).

    ``buf`` is the contiguous uint8 transit buffer; token ``t``'s record starts
    at byte ``byte_offset + t * row_stride``: ``[marker x4 | topk slots of
    I*bits/8 bytes]`` (``layout.encode_fc2`` format).
    """
    if _MODULE is None:
        if buf is None:
            return
        raise RuntimeError("TRACE fc2 overlay module is not loaded yet")
    _MODULE.nrl_trace_set_fc2_out(buf, int(byte_offset), int(row_stride), int(bits))
