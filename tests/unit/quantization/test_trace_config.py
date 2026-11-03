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
from types import SimpleNamespace

import pytest

from nemo_rl.quantization.trace.config import (
    TraceConfig,
    layer_types,
    resolve_trace_env,
)

QWEN = SimpleNamespace(num_hidden_layers=6, mlp_only_layers=[], decoder_sparse_step=1)
HYBRID = SimpleNamespace(
    layers_block_type=["mamba", "moe", "attention", "moe", "mamba", "moe"],
    num_hidden_layers=6,
)


def test_layer_types():
    assert layer_types(QWEN) == ["moe"] * 6
    sparse = SimpleNamespace(
        num_hidden_layers=4, mlp_only_layers=[3], decoder_sparse_step=2
    )
    assert layer_types(sparse) == ["dense", "moe", "dense", "dense"]
    assert layer_types(SimpleNamespace(text_config=HYBRID)) == HYBRID.layers_block_type


def test_disabled_is_empty():
    assert resolve_trace_env(TraceConfig(), QWEN, set()) == {}


def test_latter_half_skips_bf16_boundary():
    env = resolve_trace_env(TraceConfig(enabled=True), QWEN, {0, 5})
    # QAT on every NVFP4 MoE layer, guidance on the latter half of them.
    assert env == {"NRL_TRACE_LAYERS": "3:4", "NRL_TRACE_QAT_LAYERS": "1:2:3:4"}


def test_hybrid_and_explicit_layers():
    env = resolve_trace_env(TraceConfig(enabled=True, layers="all"), HYBRID, set())
    assert env["NRL_TRACE_LAYERS"] == env["NRL_TRACE_QAT_LAYERS"] == "1:3:5"
    env = resolve_trace_env(TraceConfig(enabled=True, layers=[2, 3]), HYBRID, set())
    assert env["NRL_TRACE_LAYERS"] == "3"
    with pytest.raises(ValueError):
        resolve_trace_env(TraceConfig(enabled=True, layers=[2]), HYBRID, set())
