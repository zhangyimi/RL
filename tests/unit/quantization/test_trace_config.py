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
import json
from types import SimpleNamespace

import pytest

from nemo_rl.quantization.trace.config import (
    TraceConfig,
    layer_types,
    resolve_trace_env,
)
from nemo_rl.quantization.trace.kv_calib import kv_scales_from_file, merge_calib_dir

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


def test_default_fc1_fc2_skips_bf16_boundary():
    env = resolve_trace_env(TraceConfig(enabled=True), QWEN, {0, 5})
    assert env["NRL_TRACE_FC1_LAYERS"] == env["NRL_TRACE_FC2_LAYERS"] == "1:2:3:4"
    assert env["NRL_TRACE_MODE"] == "lsb" and env["NRL_TRACE_FC2_BITS"] == "1"
    assert env["NRL_TRACE_APPLY"] == "fc1,fc2"
    assert env["NRL_TRACE_SHIP_SCALES"] == "0"
    assert "NRL_TRACE_KV_LAYERS" not in env


def test_hybrid_layers_and_kv():
    cfg = TraceConfig(
        enabled=True,
        sites=["fc1", "fc2", "kv"],
        bits=2,
        kv={"scales_file": "/x.json"},
        probe_dir="/p",
        probe_layers=[3],
    )
    env = resolve_trace_env(cfg, HYBRID, set())
    assert env["NRL_TRACE_FC1_LAYERS"] == "1:3:5"
    assert env["NRL_TRACE_KV_LAYERS"] == "2"
    assert env["NRL_TRACE_MODE"] == "b2" and env["NRL_TRACE_KV_BITS"] == "2"
    assert env["NRL_TRACE_APPLY"] == "fc1,fc2,kv,kv8"
    assert env["NRL_TRACE_KV_OSCALE"] == "1"
    assert env["NRL_TRACE_FULL_LAYERS"] == "3" and env["NRL_TRACE_PROBE_DIR"] == "/p"


def test_explicit_layers_and_kv_rtn():
    cfg = TraceConfig(
        enabled=True,
        sites=["kv"],
        layers=[0, 1],
        kv={"mode": "rtn", "scales_file": "/x"},
    )
    env = resolve_trace_env(cfg, QWEN, set())
    assert (
        env["NRL_TRACE_KV_LAYERS"] == "0:1" and env["NRL_TRACE_APPLY"] == "kv_rtn,kv8"
    )


def test_kv_needs_scales():
    with pytest.raises(ValueError, match="scales_file"):
        TraceConfig(enabled=True, sites=["kv"])


def test_calibration_without_trace():
    env = resolve_trace_env(TraceConfig(kv={"calib_dir": "/c"}), QWEN, set())
    assert env == {"NRL_TRACE_KV_CALIB_DIR": "/c"}


def test_kv_calib_merge(tmp_path):
    (tmp_path / "rank0.json").write_text(
        json.dumps({"2": {"q": 1.0, "k": 4.0, "v": 0.5}})
    )
    (tmp_path / "rank1.json").write_text(
        json.dumps({"2": {"q": 3.0, "k": 2.0, "v": 0.25}})
    )
    merged = merge_calib_dir(str(tmp_path))
    assert merged == {"2": {"q": 3.0, "k": 4.0, "v": 0.5}}
    out = tmp_path / "scales.json"
    out.write_text(json.dumps(merged))
    sc = kv_scales_from_file(str(out), headroom=2.0)
    assert sc[2]["k"] == pytest.approx(8.0 / 448.0)
