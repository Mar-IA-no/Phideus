from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_safe_abstention_gate.py"
SPEC = importlib.util.spec_from_file_location("safe_abstention_gate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def zero_report(width: int, intercept: float = -0.1) -> dict:
    model = {
        "intercept": intercept,
        "mean": [0.0] * width,
        "scale": [1.0] * width,
        "coefficients": [0.0] * width,
    }
    return {"models": [model for _ in range(4)]}


def fake_models(cfg: dict) -> dict:
    return {
        "feature_order": list(MODULE.FEATURE_ORDER),
        "arms": {
            arm: {
                "models": {
                    "public_mixed_ridge_gate": zero_report(len(MODULE.FEATURE_ORDER)),
                    "correction_scale_ridge": zero_report(len(MODULE.REDUCED_COLUMNS)),
                },
                "shuffled_target_mixed_ridge": [
                    {"model": zero_report(len(MODULE.FEATURE_ORDER))}
                    for _ in range(16)
                ],
            }
            for arm in cfg["arms"]
        },
    }


def fake_freeze(cfg: dict, threshold: float | None = None) -> dict:
    return {
        "arms": {
            arm: {
                "families": {
                    family: {"selected_threshold": threshold}
                    for family in MODULE.FAMILIES
                },
                "shuffled": [
                    {"selected_threshold": threshold} for _ in range(16)
                ],
            }
            for arm in cfg["arms"]
        }
    }


def test_config_and_source_manifest_are_frozen_cpu_contracts() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    assert cfg["execution"]["torch_threads"] == 1
    assert cfg["realizations"] == {
        "selection_seed": 2026090529,
        "adjudication_seed": 2026090537,
        "min_eligible_masters": 220,
    }
    MODULE.verify_source(cfg)


def test_threshold_grid_is_sorted_unique_and_ends_in_identity() -> None:
    advantage = np.asarray([0.0, 0.1, 0.1, 0.2, 0.4])
    grid = MODULE.threshold_grid(advantage, [0.5, 0.5, 0.75])
    assert grid[-1] is None
    assert grid[:-1] == sorted(set(grid[:-1]))
    assert grid[0] == 0.0


def test_threshold_action_uses_strict_margin_and_exact_identity() -> None:
    base = np.asarray([1, 2, 3, 4])
    advantage = np.asarray([0.0, 0.1, 0.2, 0.3])
    np.testing.assert_array_equal(
        MODULE.threshold_action(base, advantage, 0.2), np.asarray([0, 0, 0, 4])
    )
    np.testing.assert_array_equal(
        MODULE.threshold_action(base, advantage, None), np.zeros(4, dtype=np.int64)
    )


def test_no_harm_selector_retains_exact_identity_when_candidates_harm() -> None:
    # Two seeds, two alphas used, three paired masters. Alpha 1 harms IID and grouped.
    quotient = np.zeros((2, 5, 6), dtype=np.float64)
    quotient[:, 1, :] = 1.0
    actions = [np.ones(6, dtype=np.int64), np.zeros(6, dtype=np.int64)]
    bootstrap = np.tile(np.arange(3), (20, 1))
    stats = MODULE.candidate_statistics(quotient, actions, bootstrap, 95.0)
    assert stats["admissible"].tolist() == [False, True]
    assert stats["selected_index"] == 1
    assert stats["upper_iid"][-1] == 0.0


def test_deployable_actions_do_not_depend_on_adjudication_targets() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    rng = np.random.default_rng(11)
    features = rng.normal(size=(4, 2, 10, len(MODULE.FEATURE_ORDER)))
    first = {"features": features, "quotient_rmse": rng.uniform(size=(4, 2, 5, 10))}
    second = {"features": features, "quotient_rmse": rng.uniform(size=(4, 2, 5, 10)) * 100}
    models, freeze = fake_models(cfg), fake_freeze(cfg, 0.05)
    actions_a, shuffle_a, advantage_a = MODULE.apply_frozen_policies(cfg, first, models, freeze)
    actions_b, shuffle_b, advantage_b = MODULE.apply_frozen_policies(cfg, second, models, freeze)
    for policy in actions_a:
        if policy != "oracle_per_view":
            np.testing.assert_array_equal(actions_a[policy], actions_b[policy])
    np.testing.assert_array_equal(shuffle_a, shuffle_b)
    for family in advantage_a:
        np.testing.assert_array_equal(advantage_a[family], advantage_b[family])


def test_selection_phase_never_requests_adjudication_seed(tmp_path, monkeypatch) -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    called: list[int] = []
    monkeypatch.setattr(MODULE, "source_context", lambda _cfg: ({}, {}, {}))

    def spy_views(_cfg, _source_cfg, seed):
        called.append(seed)
        return []

    monkeypatch.setattr(MODULE.mixed, "fresh_views", spy_views)
    monkeypatch.setattr(MODULE.mixed, "view_index", lambda _views, _seed: [])
    monkeypatch.setattr(MODULE.mixed, "run_universe", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(MODULE, "source_feature_means", lambda _cfg: np.empty((4, 0, 15)))
    monkeypatch.setattr(MODULE, "select_thresholds", lambda *_args: ({}, {}))
    output = tmp_path / "selection_only"
    MODULE.selection_phase(cfg, output, development=True)
    assert called == [cfg["realizations"]["selection_seed"]]
    receipt = json.loads((output / "phase_receipt.json").read_text())
    assert receipt["adjudication_seed_materialized"] is False
    assert not (output / "adjudication").exists()
