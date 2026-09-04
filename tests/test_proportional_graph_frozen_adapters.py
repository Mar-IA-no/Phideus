from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py"
SPEC = importlib.util.spec_from_file_location("frozen_adapters", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_declares_cpu_scale_and_primary_arms() -> None:
    config = json.loads(MODULE.DEFAULT_CONFIG.read_text())
    assert config["arms"] == ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]
    assert config["training"]["torch_threads"] == 1
    assert config["training"]["epochs"] == 5
    assert config["training"]["max_seconds"] == 1800


def test_deterministic_npz_is_byte_exact(tmp_path: Path) -> None:
    arrays = {"b": np.arange(5, dtype=np.int64), "a": np.asarray([0.5], dtype=np.float32)}
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    MODULE.save_npz(first, arrays)
    MODULE.save_npz(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as loaded:
        assert loaded.files == ["a", "b"]
        np.testing.assert_array_equal(loaded["b"], arrays["b"])


def test_effects_are_seed_averaged_and_strict() -> None:
    config = {"arms": ["raw_generic"], "seeds": [1, 2], "bootstrap_seed": 3, "bootstrap_replicates": 50}
    rows = []
    for seed, adapter in ((1, 0.8), (2, 0.6)):
        for variant, value in (("wls_adapter", adapter), ("wls_static", 1.0),
                               ("irls_adapter", 1.0), ("irls_static", 1.0)):
            rows.append({"arm": "raw_generic", "seed": seed, "variant": variant,
                         "split": "test", "mechanism": "iid", "master_id": "m0",
                         "wls_quotient_rmse": value, "irls_quotient_rmse": value})
    result = MODULE.effects(rows, config)["raw_generic"]["test_iid"]
    assert result["wls_adapter_minus_wls_static"]["mean_delta"] == pytest.approx(-0.3)
    assert result["irls_adapter_minus_irls_static"]["mean_delta"] == 0.0
    assert MODULE.effects(rows, config)["raw_generic"]["test_grouped"]["wls_adapter_minus_wls_static"]["status"] == "NOT_EVALUABLE"
