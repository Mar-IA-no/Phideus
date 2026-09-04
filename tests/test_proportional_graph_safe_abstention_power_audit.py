from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_safe_abstention_power_audit.py"
SPEC = importlib.util.spec_from_file_location("safe_abstention_power_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_and_all_opened_inputs_are_hash_frozen() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    hashes = MODULE.verify_inputs(cfg)
    assert len(hashes) == 15
    assert cfg["execution"]["threads"] == 1


def test_projection_matches_fixed_effect_inverse_sqrt_n_formula() -> None:
    value = MODULE.projected_sample_size(250, -0.002, 0.004, 0.1)
    assert value == 1000.0
    assert MODULE.projected_sample_size(250, 0.001, 0.004, 0.1) is None
    assert MODULE.projected_sample_size(250, -0.002, 0.004, 0.0) is None


def test_paired_effect_vectors_average_seeds_before_pairing() -> None:
    quotient = np.zeros((2, 3, 4), dtype=np.float64)
    quotient[:, 1, :] = np.asarray([1.0, 2.0, 3.0, 4.0])
    action = np.asarray([1, 0, 1, 0])
    result = MODULE.paired_effect_vectors(quotient, action)
    np.testing.assert_array_equal(result["iid"], np.asarray([1.0, 3.0]))
    np.testing.assert_array_equal(result["grouped"], np.asarray([0.0, 0.0]))
    np.testing.assert_array_equal(result["balanced"], np.asarray([0.5, 1.5]))


def test_action_fractions_distinguish_empirical_identity() -> None:
    identity = MODULE.action_fractions(np.zeros(6, dtype=np.int64))
    assert identity == {
        "active_total": 0.0,
        "active_iid": 0.0,
        "active_grouped": 0.0,
        "empirical_identity": True,
    }
    active = MODULE.action_fractions(np.asarray([1, 0, 0, 2]))
    assert active["active_iid"] == 0.5
    assert active["active_grouped"] == 0.5
    assert active["empirical_identity"] is False
