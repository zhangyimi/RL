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
"""Calibrated NVFP4 KV-cache scales for TRACE.

vLLM leaves NVFP4 KV scales at 1.0 without a calibrated checkpoint, which
pushes small V blocks into E4M3 subnormals or zero. A calibration pass
(``trace.kv.calib_dir``, BF16 KV rollout) records per-layer post-RoPE q/k/v
amax on every trainer rank; merge them into the ``trace.kv.scales_file``::

    python -m nemo_rl.quantization.trace.kv_calib <calib_dir> <out.json>
"""

from __future__ import annotations

import glob
import json
import os
import sys


def merge_calib_dir(calib_dir: str) -> dict[str, dict[str, float]]:
    """Max over ranks of ``rank<r>.json`` ``{layer: {q, k, v: amax}}``."""
    merged: dict[str, dict[str, float]] = {}
    paths = sorted(glob.glob(os.path.join(calib_dir, "rank*.json")))
    if not paths:
        raise FileNotFoundError(f"no rank*.json under {calib_dir}")
    for path in paths:
        with open(path) as f:
            for layer, d in json.load(f).items():
                cur = merged.setdefault(layer, {"q": 0.0, "k": 0.0, "v": 0.0})
                for name in cur:
                    cur[name] = max(cur[name], float(d[name]))
    return dict(sorted(merged.items(), key=lambda kv: int(kv[0])))


def kv_scales_from_file(
    path: str, headroom: float = 1.0
) -> dict[int, dict[str, float]]:
    """Per-layer dequant scales from a ``{layer: {q, k, v: amax}}`` JSON.

    NVFP4 K/V: ``amax / 448``, i.e. block scales stay in ``[0, 448 / 6]``.
    The trtllm-gen attention kernel dequantizes NVFP4 K/V to FP8 E4M3
    internally, so ``code * block_scale`` (up to ``6 * block_scale``) must fit
    in 448 (FlashInfer ``nvfp4_quantize_paged_kv_cache`` uses the same
    ``global_sf = 448 / amax``). ``amax / (6 * 448)`` overflows that FP8
    intermediate and breaks the rollout.
    FP8 query (vLLM quantizes Q to e4m3 with NVFP4 KV): ``amax / 448``.
    """
    with open(path) as f:
        amax = json.load(f)
    return {
        int(layer): {name: d[name] * headroom / 448.0 for name in ("q", "k", "v")}
        for layer, d in amax.items()
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "usage: python -m nemo_rl.quantization.trace.kv_calib <calib_dir> <out.json>"
        )
    with open(sys.argv[2], "w") as f:
        json.dump(merge_calib_dir(sys.argv[1]), f, indent=0)
