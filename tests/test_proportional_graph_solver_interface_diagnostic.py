"""Contract tests for the CPU-only solver-interface diagnostic."""

from __future__ import annotations

import ast
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.geometria_proporcional import (  # noqa: E402
    run_proportional_graph_solver_interface_diagnostic as runner,
)


CONFIG_PATH = (
    REPO_ROOT
    / "experiments/geometria_proporcional/configs/proportional_graph_solver_interface_diagnostic_v1.json"
)


@pytest.fixture(scope="module")
def frozen_sources():
    config = runner._load_config(CONFIG_PATH)
    verified = runner._verify_sources(config, time.monotonic())
    return config, verified


def test_config_and_sources_are_exact(frozen_sources):
    config, verified = frozen_sources
    _, _, canonical, neural, metrics, solver_raws, bundle = verified
    assert config["execution"] == runner.EXPECTED_EXECUTION
    assert tuple(config["analysis"]["feature_order"]) == runner.FEATURE_ORDER
    assert len(canonical["view_id"]) == 631
    assert len(neural) == len(solver_raws) == 16
    assert metrics["quotient_rmse"].shape == (8, 2, 8, 631)
    assert bundle["attestation"]["validation_masters"] == 127
    assert bundle["attestation"]["test_masters"] == 252


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x["analysis"]["alphas"].reverse(),
        lambda x: x["analysis"].__setitem__("shuffle_replicates", 9),
        lambda x: x["analysis"].__setitem__("ridge_lambdas", [1.0]),
        lambda x: x["analysis"]["feature_order"].pop(),
        lambda x: x["solver"].__setitem__("huber_delta", 2.0),
        lambda x: x["execution"].__setitem__("max_seconds", 901),
        lambda x: x["sources"]["smoke"].__setitem__("manifest_sha256", "0" * 64),
        lambda x: x["analysis"].__setitem__("extra", True),
    ],
)
def test_protocol_mutations_are_rejected(tmp_path: Path, mutation):
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutation(data)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        runner._load_config(path)


def test_output_guard_rejects_both_source_trees(frozen_sources, tmp_path: Path):
    _, verified = frozen_sources
    smoke, disentangle = verified[:2]
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        runner._prepare_output(existing, (smoke, disentangle))
    with pytest.raises(ValueError):
        runner._prepare_output(smoke / "child", (smoke, disentangle))
    with pytest.raises(ValueError):
        runner._prepare_output(disentangle.parent, (smoke, disentangle))


def test_public_feature_boundary_and_path_validation(frozen_sources):
    _, verified = frozen_sources
    _, _, canonical, neural, *_ = verified
    raw = neural["raw_generic|seed=104729"]
    edge_slice, _, path_slice = runner.dis._view_slices(canonical, 0)
    kwargs = {
        "n_nodes": int(canonical["n_nodes"][0]),
        "edge_index": canonical["edge_index"][edge_slice],
        "edge_valid": canonical["edge_valid"][edge_slice],
        "edge_variance": canonical["edge_variance"][edge_slice],
        "observed_log_ratio": canonical["observed_log_ratio"][edge_slice],
        "path_index": raw["path_index"][path_slice],
        "path_sign": raw["path_sign"][path_slice],
        "path_valid": raw["path_valid"][path_slice],
        "corrected_log_ratio": raw["corrected_log_ratio"][edge_slice],
        "reliability": raw["reliability"][edge_slice],
    }
    features = runner._public_features(**kwargs)
    assert features.shape == (len(runner.FEATURE_ORDER),)
    assert np.all(np.isfinite(features))
    assert "causal_corruption_mask" not in inspect.signature(runner._public_features).parameters
    with pytest.raises(TypeError):
        runner._public_features(**kwargs, causal_corruption_mask=np.zeros(len(kwargs["edge_valid"])))
    broken = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in kwargs.items()}
    broken["path_index"][0, 0] = 100_000
    with pytest.raises(ValueError, match="path indices"):
        runner._public_features(**broken)


def test_temperature_endpoints_copy_source_weights(frozen_sources):
    _, verified = frozen_sources
    _, _, canonical, neural, *_ = verified
    raw = neural["raw_typed|seed=104729"]
    edge_slice, _, _ = runner.dis._view_slices(canonical, 0)
    learned = raw["reliability"][edge_slice]
    valid = canonical["edge_valid"][edge_slice]
    unit = runner._tempered_weights(learned, valid, 0.0)
    copied = runner._tempered_weights(learned, valid, 1.0)
    assert np.array_equal(unit[valid], np.ones(valid.sum()))
    assert np.array_equal(copied[valid], learned[valid])


def test_shuffles_are_deterministic_and_preserve_multiset(frozen_sources):
    _, verified = frozen_sources
    _, _, canonical, neural, *_ = verified
    raw = neural["raw_typed|seed=104729"]
    edge_slice, _, _ = runner.dis._view_slices(canonical, 0)
    learned = raw["reliability"][edge_slice]
    valid = canonical["edge_valid"][edge_slice]
    kwargs = dict(arm="raw_typed", seed=104729, view_index=0, seed_root=350903)
    first = runner._shuffle_weights(learned, valid, replicate=0, **kwargs)
    repeat = runner._shuffle_weights(learned, valid, replicate=0, **kwargs)
    other = runner._shuffle_weights(learned, valid, replicate=1, **kwargs)
    assert np.array_equal(first, repeat)
    assert np.array_equal(np.sort(first[valid]), np.sort(learned[valid]))
    assert not np.array_equal(first, other)


def test_static_selection_is_invariant_to_test_metrics_and_mechanism(frozen_sources):
    _, verified = frozen_sources
    _, _, canonical, _, metrics, *_ = verified
    baseline = runner._validation_selection(canonical, metrics)
    mutated_metrics = {key: np.array(value, copy=True) for key, value in metrics.items()}
    test = canonical["split"] == "test"
    mutated_metrics["quotient_rmse"][..., test] = 1e9
    mutated_canonical = {key: np.array(value, copy=True) for key, value in canonical.items()}
    mutated_canonical["mechanism"] = mutated_canonical["mechanism"][::-1]
    assert runner._validation_selection(mutated_canonical, mutated_metrics) == baseline


def test_static_tie_within_tolerance_prefers_simpler_cell(frozen_sources):
    _, verified = frozen_sources
    _, _, canonical, _, metrics, *_ = verified
    tied = {key: np.array(value, copy=True) for key, value in metrics.items()}
    validation = canonical["split"] == "validation"
    arm_index = runner.ARMS.index("raw_generic")
    for order, cell_name in enumerate(runner.STATIC_CELLS):
        relation, weight = cell_name.split("|")
        cell = runner.dis._cell_index(relation, weight, "wls")
        tied["quotient_rmse"][arm_index, :, cell, validation] = 0.2 + order * 2e-13
    selected = runner._validation_selection(canonical, tied)
    assert selected["arms"]["raw_generic"]["wls"]["selected_cell"] == "observed|unit"


def test_temperature_selection_ignores_test_values(frozen_sources):
    _, verified = frozen_sources
    _, _, canonical, *_ = verified
    rng = np.random.default_rng(1)
    quotient = rng.uniform(
        0.1, 0.3, size=(len(runner.PRIMARY_ARMS), 2, len(runner.VARIANTS), 631)
    )
    baseline = runner._temperature_selection(canonical, quotient)
    mutated = quotient.copy()
    mutated[..., canonical["split"] == "test"] = 999.0
    assert runner._temperature_selection(canonical, mutated) == baseline


def test_temperature_tie_within_tolerance_prefers_alpha_zero(frozen_sources):
    _, verified = frozen_sources
    canonical = verified[2]
    quotient = np.full(
        (len(runner.PRIMARY_ARMS), 2, len(runner.VARIANTS), 631), 0.2
    )
    validation = canonical["split"] == "validation"
    for order, alpha in enumerate(runner.ALPHAS):
        variant = runner.VARIANTS.index(f"observed|alpha={alpha:.2f}|wls")
        quotient[0, :, variant, validation] += order * 2e-13
    selected = runner._temperature_selection(canonical, quotient)
    assert selected["arms"]["raw_generic"]["wls"]["observed"]["selected_alpha"] == 0.0


def test_ridge_models_are_invariant_to_test_targets_and_mechanism(frozen_sources):
    config, verified = frozen_sources
    _, _, canonical, _, metrics, *_ = verified
    rng = np.random.default_rng(2)
    features = rng.normal(size=(8, 2, 631, len(runner.FEATURE_ORDER)))
    report, predictions, decisions = runner._fit_transport_models(
        canonical, features, metrics, config
    )
    mutated_metrics = {key: np.array(value, copy=True) for key, value in metrics.items()}
    mutated_metrics["quotient_rmse"][..., canonical["split"] == "test"] = 12345.0
    changed = {key: np.array(value, copy=True) for key, value in canonical.items()}
    changed["mechanism"] = changed["mechanism"][::-1]
    report_2, predictions_2, decisions_2 = runner._fit_transport_models(
        changed, features, mutated_metrics, config
    )
    assert report == report_2
    assert np.array_equal(predictions, predictions_2, equal_nan=True)
    assert np.array_equal(decisions, decisions_2)


def test_fold_assignment_keeps_master_groups_together():
    masters = np.asarray(["a", "a", "b", "c", "c", "d"])
    folds = runner._fold_ids(masters, 5)
    assert folds[0] == folds[1]
    assert folds[3] == folds[4]


def test_ridge_treats_numerical_dust_as_constant():
    x = np.column_stack([np.arange(6.0), np.linspace(0.0, 5e-19, 6)])
    y = np.arange(6.0)
    model = runner._ridge_fit(x, y, 1.0)
    assert model["scale"][1] == 1.0


def test_bootstrap_master_mapping_uses_preserved_key(frozen_sources):
    _, verified = frozen_sources
    _, _, canonical, _, _, _, bundle = verified
    iid, grouped, indices = runner._test_pair_indices(
        canonical, bundle["indices"]
    )
    assert iid.shape == grouped.shape == (252,)
    assert indices.shape == (2000, 252)
    assert np.array_equal(canonical["master_id"][iid], bundle["indices"]["master_id"])
    assert np.array_equal(canonical["master_id"][grouped], bundle["indices"]["master_id"])


def test_endpoint_recomputation_matches_disentanglement(frozen_sources):
    config, verified = frozen_sources
    _, _, canonical, neural, _, solver_raws, _ = verified
    raw = neural["raw_generic|seed=104729"]
    stored = solver_raws["raw_generic|seed=104729"]
    edge_slice, node_slice, _ = runner.dis._view_slices(canonical, 0)
    observation = runner.dis._public_view(canonical, 0)
    for relation_name, values in (
        ("observed", canonical["observed_log_ratio"][edge_slice]),
        ("corrected", raw["corrected_log_ratio"][edge_slice]),
    ):
        for weight_name, weights in (
            ("unit", np.ones(edge_slice.stop - edge_slice.start)),
            ("learned", raw["reliability"][edge_slice]),
        ):
            for solver in runner.SOLVERS:
                state = runner.dis._solve_cell(
                    observation,
                    values,
                    weights,
                    solver,
                    config,
                    canonical["x_true"][node_slice],
                    canonical["clean_log_ratio"][edge_slice],
                )
                cell = runner.dis._cell_index(relation_name, weight_name, solver)
                runner._assert_endpoint(
                    state,
                    stored,
                    cell,
                    0,
                    edge_slice,
                    node_slice,
                    1e-12,
                    "test",
                )


def test_weight_semantics_distinguishes_effective_multiplier(frozen_sources):
    _, verified = frozen_sources
    _, _, canonical, neural, _, solver_raws, _ = verified
    report, arrays = runner._weight_semantics(canonical, neural, solver_raws)
    assert set(arrays) == {
        "base_effective_fraction",
        "unit_irls_effective_fraction",
        "learned_irls_effective_fraction",
        "tail_overlap",
        "base_corrupt_mass",
        "unit_irls_corrupt_mass",
        "learned_irls_corrupt_mass",
        "learned_minus_unit_irls_rmse",
        "effective_fraction_change",
    }
    assert "test|grouped" in report["arms"]["raw_generic"]
    assert np.nanmin(arrays["tail_overlap"]) >= 0.0
    assert np.nanmax(arrays["tail_overlap"]) <= 1.0


def test_failure_diagnostics_preserve_variant_and_slice_axes(frozen_sources):
    _, verified = frozen_sources
    canonical = verified[2]
    converged = np.ones(
        (len(runner.PRIMARY_ARMS), 2, len(runner.VARIANTS), 631), dtype=bool
    )
    converged[0, 0, 0, 0] = False
    payload = runner._failure_diagnostics(canonical, converged)
    assert len(payload["records"]) == len(runner.PRIMARY_ARMS) * 2 * len(runner.VARIANTS) * 3
    assert sum(record["n_failed"] for record in payload["records"]) == 1


def test_runner_has_no_torch_or_gpu_imports():
    tree = ast.parse(runner.RUNNER_PATH.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imports
    assert "cupy" not in imports


def test_report_and_replay_lifecycle(tmp_path: Path):
    output = tmp_path / "artifact"
    output.mkdir()
    runner.dis._write_json(output / "effects.json", {"x": 1})
    estimate = {"status": "ESTIMATED", "mean": 0.0}
    static = {
        "arms": {
            arm: {solver: {"selected_cell": "observed|unit"} for solver in runner.SOLVERS}
            for arm in runner.ARMS
        }
    }
    temperature = {
        "arms": {
            arm: {
                "wls": {
                    relation: {"selected_alpha": 0.0}
                    for relation in runner.RELATIONS
                },
                "irls": {"observed": {"selected_alpha": 0.0}},
            }
            for arm in runner.PRIMARY_ARMS
        }
    }
    effects = {"arms": {}}
    for arm in runner.ARMS:
        effects["arms"][arm] = {
            "static": {
                solver: {
                    "delta_vs_unit": {
                        "test|iid": estimate,
                        "test|grouped": estimate,
                    }
                }
                for solver in runner.SOLVERS
            },
            "temperature": {},
            "shuffle": {},
        }
        if arm in runner.PRIMARY_ARMS:
            effects["arms"][arm]["temperature"] = {
                "wls": {
                    relation: {
                        "selected_alpha": 0.0,
                        "delta_vs_unit": {
                            "test|iid": estimate,
                            "test|grouped": estimate,
                        },
                    }
                    for relation in runner.RELATIONS
                },
                "irls": {
                    "observed": {
                        "selected_alpha": 0.0,
                        "delta_vs_unit": {
                            "test|iid": estimate,
                            "test|grouped": estimate,
                        },
                    }
                },
            }
            effects["arms"][arm]["shuffle"] = {
                "wls": {
                    name: {"test|iid": estimate, "test|grouped": estimate}
                    for name in ("learned_minus_shuffled", "shuffled_minus_unit")
                }
            }
    transport = {
        "arms": {
            arm: {
                solver: {
                    "selected_fraction": {"test|iid": 0.5, "test|grouped": 0.5},
                    "prediction": {
                        "test|iid": {"correlation": 0.0},
                        "test|grouped": {"correlation": 0.0},
                    },
                    "policy_delta_vs_unit": {
                        "test|iid": estimate,
                        "test|grouped": estimate,
                    },
                }
                for solver in runner.SOLVERS
            }
            for arm in runner.PRIMARY_ARMS
        }
    }
    semantics = {
        "arms": {
            arm: {
                split: {
                    "unit_irls_effective_fraction": 1.0,
                    "learned_irls_effective_fraction": 0.9,
                    "tail_overlap": 0.5,
                    "learned_minus_unit_irls_rmse": 0.0,
                }
                for split in ("test|iid", "test|grouped")
            }
            for arm in runner.PRIMARY_ARMS
        }
    }
    text = runner._report_text(static, temperature, effects, transport, semantics)
    assert "no declara GO/NO-GO" in text
    assert "Ubicación del peso" in text
    relative_config = str(CONFIG_PATH.relative_to(REPO_ROOT))
    source_records = {
        relative_config: {
            "sha256": runner.dis._sha256_file(CONFIG_PATH),
            "bytes": CONFIG_PATH.stat().st_size,
            "tracked": True,
            "dirty": False,
        }
    }
    runner._write_replay(
        output, CONFIG_PATH, source_records, development=True
    )
    replay = (output / "replay.sh").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=''" in replay
    assert "venv/bin/python" in replay
    assert "usage: replay.sh OUTPUT_DIR" in replay
    assert '  --output "$1" --development' in replay
    assert source_records[relative_config]["sha256"] in replay
    assert "--development" in replay
