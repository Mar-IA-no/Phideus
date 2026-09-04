from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py"
SPEC = importlib.util.spec_from_file_location("fresh_mixed_gate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_config_freezes_disjoint_cpu_realizations_and_sources() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    assert cfg["execution"]["torch_threads"] == 1
    assert cfg["realizations"]["calibration_seed"] != cfg["realizations"]["adjudication_seed"]
    assert cfg["shuffle_replicates"] == 16
    MODULE.verify_sources(cfg)


def test_all_frozen_post_irls_adapters_load_on_cpu() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    _, scale, models = MODULE.source_context(cfg)
    assert scale > 0
    assert set(models) == set(cfg["arms"])
    assert all(set(by_seed) == set(cfg["seeds"]) for by_seed in models.values())


def fake_paired_views(n_per_stratum: int = 4) -> list[SimpleNamespace]:
    views = []
    for n_nodes in (8, 9):
        for index in range(n_per_stratum):
            master = f"n{n_nodes}-m{index}"
            for mechanism in MODULE.MECHANISMS:
                views.append(SimpleNamespace(
                    private=SimpleNamespace(master_id=master, corruption_mechanism=mechanism),
                    public=SimpleNamespace(n_nodes=n_nodes),
                ))
    return views


def test_paired_shuffle_is_deranged_stratified_and_master_atomic() -> None:
    views = fake_paired_views()
    permutation = MODULE.paired_master_permutation(views, 71)
    for index in range(0, len(views), 2):
        donor_iid, donor_grouped = permutation[index : index + 2]
        assert donor_grouped == donor_iid + 1
        assert donor_iid % 2 == 0
        assert donor_iid != index
        assert views[donor_iid].public.n_nodes == views[index].public.n_nodes
        assert views[donor_iid].private.master_id == views[donor_grouped].private.master_id


def zero_report(width: int) -> dict:
    model = {
        "intercept": 0.0,
        "mean": [0.0] * width,
        "scale": [1.0] * width,
        "coefficients": [0.0] * width,
    }
    return {"models": [model for _ in range(4)]}


def frozen_models(cfg: dict) -> dict:
    result = {"feature_order": list(MODULE.FEATURE_ORDER), "arms": {}}
    for arm in cfg["arms"]:
        result["arms"][arm] = {
            "constant_alpha_index": 0,
            "models": {
                "correction_scale_ridge": zero_report(len(MODULE.REDUCED_COLUMNS)),
                "public_mixed_ridge_gate": zero_report(len(MODULE.FEATURE_ORDER)),
                "historical_iid_ridge_gate": zero_report(len(MODULE.FEATURE_ORDER)),
            },
            "shuffled_target_mixed_ridge": [
                {"model": zero_report(len(MODULE.FEATURE_ORDER))}
                for _ in range(cfg["shuffle_replicates"])
            ],
        }
    return result


def test_deployable_actions_ignore_future_targets_and_private_labels() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    rng = np.random.default_rng(5)
    features = rng.normal(size=(len(cfg["arms"]), len(cfg["seeds"]), 12, len(MODULE.FEATURE_ORDER)))
    first_quotient = rng.uniform(size=(len(cfg["arms"]), len(cfg["seeds"]), len(cfg["alphas"]), 12))
    second_quotient = rng.uniform(size=first_quotient.shape) * 100.0
    models = frozen_models(cfg)
    first, first_shuffle, _ = MODULE.decisions_from_freeze(models, features, first_quotient, cfg)
    second, second_shuffle, _ = MODULE.decisions_from_freeze(models, features, second_quotient, cfg)
    for policy in first:
        if policy != "oracle_per_view":
            np.testing.assert_array_equal(first[policy], second[policy])
    np.testing.assert_array_equal(first_shuffle, second_shuffle)


def test_identity_wins_prediction_ties() -> None:
    predicted = np.zeros((7, 4))
    np.testing.assert_array_equal(
        MODULE.historical_gate.choose_alpha(predicted), np.zeros(7, dtype=np.int64)
    )


def test_calibration_phase_never_requests_adjudication_seed(tmp_path, monkeypatch) -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    called: list[int] = []

    monkeypatch.setattr(MODULE, "source_context", lambda _: ({}, 1.0, {}))

    def spy_fresh_views(_cfg, _source, seed):
        called.append(seed)
        return []

    monkeypatch.setattr(MODULE, "fresh_views", spy_fresh_views)
    monkeypatch.setattr(MODULE, "view_index", lambda _views, _seed: [])
    monkeypatch.setattr(MODULE, "run_universe", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(MODULE, "calibration_fit", lambda *_args: ({}, {}))
    monkeypatch.setattr(MODULE, "calibration_diagnostics", lambda *_args: {})
    output = tmp_path / "calibration_only"
    MODULE.calibration_phase(cfg, output, development=True)
    assert called == [cfg["realizations"]["calibration_seed"]]
    receipt = json.loads((output / "phase_receipt.json").read_text())
    assert receipt["adjudication_seed_materialized"] is False
    assert not (output / "adjudication").exists()


def test_fresh_realizations_are_paired_and_disjoint() -> None:
    cfg = MODULE.load_config(MODULE.DEFAULT_CONFIG)
    cfg["realizations"]["min_eligible_masters"] = 2
    source_cfg = MODULE.smoke._load_config((ROOT / cfg["source_smoke_config"]).resolve())
    # Non-canonical seeds only: the frozen realizations remain unmaterialized.
    calibration = MODULE.fresh_views(cfg, source_cfg, 2026090917)
    adjudication = MODULE.fresh_views(cfg, source_cfg, 2026090923)
    assert len(calibration) % 2 == len(adjudication) % 2 == 0
    assert {v.private.master_id for v in calibration}.isdisjoint(
        {v.private.master_id for v in adjudication}
    )
    for views in (calibration, adjudication):
        for index in range(0, len(views), 2):
            assert [v.private.corruption_mechanism for v in views[index : index + 2]] == list(MODULE.MECHANISMS)
