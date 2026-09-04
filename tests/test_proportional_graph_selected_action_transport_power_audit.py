from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_selected_action_transport_power_audit.py"
SPEC = importlib.util.spec_from_file_location("selected_action_transport_power", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_and_four_opened_sources_are_frozen_cpu_contracts() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    verified = MODULE.verify_sources(cfg)
    assert set(verified["manifests"]) == {
        "cohort_a_data", "cohort_a_models", "cohort_b_data", "cohort_b_selected",
    }
    assert cfg["execution"] == {"max_seconds": 300, "max_rss_gib": 4.0}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_transport_classification_respects_numerical_identity() -> None:
    tol = 1e-12
    assert MODULE.classify_transport(-1.0, -2.0, tol) == "FAVORABLE_BOTH"
    assert MODULE.classify_transport(1.0, 2.0, tol) == "ADVERSE_BOTH"
    assert MODULE.classify_transport(-1.0, 2.0, tol) == "SIGN_UNSTABLE"
    assert MODULE.classify_transport(0.5e-12, -2.0, tol) == "IDENTITY_OR_NUMERICAL_ZERO"


def test_fixed_effect_projection_never_shrinks_a_resolved_sample() -> None:
    resolved = MODULE.project_fixed_effect({"mean": -2.0, "ci95": [-3.0, -1.0]}, 100)
    unresolved = MODULE.project_fixed_effect({"mean": -1.0, "ci95": [-3.0, 1.0]}, 100)
    adverse = MODULE.project_fixed_effect({"mean": 1.0, "ci95": [-1.0, 3.0]}, 100)
    assert resolved["n_projected"] == 100
    assert unresolved["n_projected"] == 400
    assert adverse["n_projected"] is None


def test_balanced_cell_pairs_iid_and_grouped_before_seed_average() -> None:
    raw = np.asarray([[1.0, 5.0, 3.0, 7.0], [3.0, 7.0, 5.0, 9.0]])
    np.testing.assert_allclose(MODULE.cell_values(raw, "iid"), [2.0, 4.0])
    np.testing.assert_allclose(MODULE.cell_values(raw, "grouped"), [6.0, 8.0])
    np.testing.assert_allclose(MODULE.cell_values(raw, "balanced"), [4.0, 6.0])


def test_cohort_b_reconstruction_is_exactly_r364() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    reconstructed = MODULE.reconstruct_b(cfg)
    assert reconstructed["exact_reproduction"] is True
    expected = MODULE.read_npz(MODULE.roots(cfg)["cohort_b_selected"] / "adjudication_diagnostics.npz")
    for family in MODULE.FAMILIES:
        np.testing.assert_array_equal(
            reconstructed["adjudication_pack"][f"{family}|deployed"],
            expected[f"{family}|deployed"],
        )
