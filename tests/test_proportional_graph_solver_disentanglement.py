"""Contract tests for the CPU-only frozen-output solver disentanglement."""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.geometria_proporcional import (  # noqa: E402
    run_proportional_graph_solver_disentanglement as runner,
)
from geometria_proporcional.proportional_graph_contract import SolverResult  # noqa: E402


CONFIG_PATH = (
    REPO_ROOT
    / "experiments/geometria_proporcional/configs/proportional_graph_solver_disentanglement_v1.json"
)
SOURCE = REPO_ROOT / "data/geometria_proporcional/proportional_graph_neural_smoke_v1"


@pytest.fixture(scope="module")
def frozen_source():
    config = runner._load_config(CONFIG_PATH)
    canonical, raws, bootstrap, attestation = runner._verify_source(SOURCE, config)
    return config, canonical, raws, bootstrap, attestation


def test_config_and_frozen_source_contract_are_strict(frozen_source, tmp_path: Path):
    config, canonical, raws, bootstrap, attestation = frozen_source
    assert config["execution"] == {
        "device": "cpu",
        "numeric_threads": 1,
        "max_seconds": 7200,
        "max_rss_gib": 8.0,
    }
    assert len(raws) == 16
    assert len(attestation["used_files"]) == 19
    assert len(canonical["view_id"]) == 631
    assert bootstrap["complete_n_252"].shape == (2000, 252)
    modified = json.loads(CONFIG_PATH.read_text())
    modified["analysis"]["relations"].reverse()
    bad = tmp_path / "modified.json"
    bad.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="relation order"):
        runner._load_config(bad)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["solver"].__setitem__("tolerance", 1e-3),
        lambda data: data["solver"].__setitem__("reuse_atol", 1e6),
        lambda data: data["analysis"].__setitem__("reuse_verification_indices", []),
        lambda data: data["analysis"].__setitem__("primary_arms", data["arms"][1:5]),
        lambda data: data["analysis"].__setitem__("unplanned", True),
        lambda data: data["execution"].__setitem__("max_seconds", 999999),
        lambda data: data["execution"].__setitem__("extra", 1),
    ],
)
def test_every_protocol_mutation_is_rejected(tmp_path: Path, mutation):
    data = json.loads(CONFIG_PATH.read_text())
    mutation(data)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        runner._load_config(path)


def test_output_guard_rejects_existing_and_source_descendants(tmp_path: Path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        runner._prepare_output(existing, SOURCE)
    with pytest.raises(ValueError, match="descend"):
        runner._prepare_output(SOURCE / "forbidden-output", SOURCE)
    with pytest.raises(ValueError):
        runner._prepare_output(SOURCE, SOURCE)
    alias = tmp_path / "source-alias"
    alias.symlink_to(SOURCE, target_is_directory=True)
    with pytest.raises(ValueError):
        runner._prepare_output(alias, SOURCE)
    with pytest.raises(ValueError, match="contain"):
        runner._prepare_output(SOURCE.parent, SOURCE)


def test_raw_alignment_enforces_float_projection_and_shuffle_exception(frozen_source):
    _, canonical, raws, _, _ = frozen_source
    raw = {key: value.copy() for key, value in raws["raw_generic|seed=104729"].items()}
    assert runner._validate_raw_alignment(canonical, raw, "raw_generic")[
        "edge_variance_projection"
    ] == "float64_to_float32_exact"
    raw["edge_variance"][0] = np.nextafter(raw["edge_variance"][0], np.float32(1.0))
    with pytest.raises(ValueError, match="edge_variance"):
        runner._validate_raw_alignment(canonical, raw, "raw_generic")
    shuffled = raws["closure_typed_path_shuffle|seed=104729"]
    result = runner._validate_raw_alignment(
        canonical, shuffled, "closure_typed_path_shuffle"
    )
    assert result["path_policy"] == "intervened_shuffle"


@pytest.mark.parametrize("corruption", ["offset", "index", "sign"])
def test_path_shuffle_ragged_validation_rejects_internal_corruption(
    frozen_source, corruption
):
    _, canonical, raws, _, _ = frozen_source
    raw = {
        key: value.copy()
        for key, value in raws["closure_typed_path_shuffle|seed=104729"].items()
    }
    if corruption == "offset":
        raw["path_offsets"][1] = -3
    elif corruption == "index":
        raw["path_index"][0, 0] = 10_000
    else:
        raw["path_sign"][0, 0] = np.nan
    with pytest.raises(ValueError, match="path"):
        runner._validate_raw_alignment(
            canonical, raw, "closure_typed_path_shuffle"
        )


def test_solver_boundary_passes_only_public_observation(monkeypatch, frozen_source):
    config, canonical, raws, _, _ = frozen_source
    raw = raws["raw_generic|seed=104729"]
    edge_slice, node_slice, _ = runner._view_slices(canonical, 0)
    observation = runner._public_view(canonical, 0)
    captured = {}

    def spy(public, values=None, weights=None, **kwargs):
        captured["fields"] = set(public.arrays())
        captured["values"] = values.copy()
        captured["weights"] = weights.copy()
        n = public.n_nodes
        incidence = runner.incidence_matrix(n, public.edge_index)
        x = np.zeros(n)
        return SolverResult(
            x_hat=x,
            reconstructed_log_ratio=incidence @ x,
            weights=np.ones(len(values)),
            quotient_rmse=np.nan,
            relation_rmse=np.nan,
            weighted_residual_rmse=0.0,
            laplacian_rank=n - 1,
            laplacian_condition=1.0,
            converged=True,
            iterations=1,
        )

    monkeypatch.setattr(runner, "solve_weighted_least_squares", spy)
    runner._solve_cell(
        observation,
        raw["corrected_log_ratio"][edge_slice],
        raw["reliability"][edge_slice],
        "wls",
        config,
        canonical["x_true"][node_slice],
        canonical["clean_log_ratio"][edge_slice],
    )
    assert captured["fields"] == {
        "n_nodes",
        "edge_index",
        "observed_log_ratio",
        "edge_valid",
        "path_index",
        "path_sign",
        "path_valid",
        "edge_variance",
    }
    assert np.array_equal(captured["values"], raw["corrected_log_ratio"][edge_slice])
    assert np.array_equal(captured["weights"], raw["reliability"][edge_slice])


@pytest.mark.parametrize("weight_name", ["unit", "learned"])
def test_irls_receives_the_declared_base_weight(monkeypatch, frozen_source, weight_name):
    config, canonical, raws, _, _ = frozen_source
    raw = raws["raw_generic|seed=104729"]
    edge_slice, node_slice, _ = runner._view_slices(canonical, 0)
    observation = runner._public_view(canonical, 0)
    expected = (
        np.ones(edge_slice.stop - edge_slice.start)
        if weight_name == "unit"
        else raw["reliability"][edge_slice]
    )
    captured = {}

    def spy(public, *, values, base_weights, **kwargs):
        captured["base_weights"] = base_weights.copy()
        n = public.n_nodes
        return SolverResult(
            x_hat=np.zeros(n),
            reconstructed_log_ratio=np.zeros(len(values)),
            weights=np.ones(len(values)),
            quotient_rmse=np.nan,
            relation_rmse=np.nan,
            weighted_residual_rmse=0.0,
            laplacian_rank=n - 1,
            laplacian_condition=1.0,
            converged=True,
            iterations=1,
        )

    monkeypatch.setattr(runner, "solve_huber_irls", spy)
    runner._solve_cell(
        observation,
        canonical["observed_log_ratio"][edge_slice],
        expected,
        "irls",
        config,
        canonical["x_true"][node_slice],
        canonical["clean_log_ratio"][edge_slice],
    )
    assert np.array_equal(captured["base_weights"], expected)


@pytest.mark.parametrize(
    "relation,weight,solver",
    [
        (relation, weight, solver)
        for solver in runner.SOLVERS
        for relation in runner.RELATIONS
        for weight in runner.BASE_WEIGHTS
    ],
)
def test_all_eight_factor_cells_have_a_unique_stable_index(relation, weight, solver):
    index = runner._cell_index(relation, weight, solver)
    assert runner.CELL_IDS[index] == f"{relation}|{weight}|{solver}"
    assert len(set(runner.CELL_IDS)) == 8


def test_reused_source_states_match_canonical_float64_solver(frozen_source):
    config, canonical, raws, _, _ = frozen_source
    raw = raws["raw_generic|seed=104729"]
    for index in (0, 64):
        edge_slice, node_slice, _ = runner._view_slices(canonical, index)
        observation = runner._public_view(canonical, index)
        for relation, weight, source_raw in (
            (canonical["observed_log_ratio"][edge_slice], np.ones(edge_slice.stop - edge_slice.start), canonical),
            (raw["corrected_log_ratio"][edge_slice], raw["reliability"][edge_slice], raw),
        ):
            actual = runner._solve_cell(
                observation,
                relation,
                weight,
                "irls",
                config,
                canonical["x_true"][node_slice],
                canonical["clean_log_ratio"][edge_slice],
            )
            expected = runner._stored_state(
                canonical, source_raw, index, "irls", relation
            )
            runner._assert_state_close(actual, expected, 1e-12, f"view={index}")


@pytest.mark.parametrize(
    "field",
    ["x_hat_wls", "wls_quotient_rmse", "wls_laplacian_rank", "wls_condition"],
)
def test_wls_attestation_rejects_every_preserved_field_corruption(
    frozen_source, field
):
    config, canonical, _, _, _ = frozen_source
    edge_slice, node_slice, _ = runner._view_slices(canonical, 0)
    state = runner._solve_cell(
        runner._public_view(canonical, 0),
        canonical["observed_log_ratio"][edge_slice],
        np.ones(edge_slice.stop - edge_slice.start),
        "wls",
        config,
        canonical["x_true"][node_slice],
        canonical["clean_log_ratio"][edge_slice],
    )
    corrupted = {key: value.copy() for key, value in canonical.items()}
    if field == "x_hat_wls":
        corrupted[field][node_slice.start] += 1e-4
    elif field == "wls_laplacian_rank":
        corrupted[field][0] += 1
    else:
        corrupted[field][0] += 1e-4
    with pytest.raises(AssertionError, match="stored WLS"):
        runner._assert_wls_source(
            state, corrupted, 0, node_slice, 1e-12, "corrupted"
        )


def test_factorial_algebra_and_total_identity_are_exact():
    cells = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [7.0, 9.0]],
        ]
    )
    effects = runner._factorial_effects(cells)
    assert np.array_equal(effects["relation_at_unit_weight"], [1.0, 1.0])
    assert np.array_equal(effects["weight_on_observed_relation"], [2.0, 2.0])
    assert np.array_equal(effects["relation_weight_interaction"], [3.0, 4.0])
    assert np.array_equal(effects["total_delivered_effect"], [6.0, 7.0])


def test_seed_mean_is_strict_before_master_effects():
    values = np.asarray([[1.0, 3.0, 5.0], [3.0, 5.0, np.nan]])
    result = runner._strict_seed_mean(values)
    assert np.array_equal(result[:2], [2.0, 4.0])
    assert np.isnan(result[2])


def test_master_mapping_uses_ids_not_raw_position():
    canonical = {
        "master_id": np.asarray(["m2", "m1", "m1", "m2"]),
        "split": np.asarray(["test"] * 4),
        "mechanism": np.asarray(["iid", "grouped", "iid", "grouped"]),
    }
    mapping = runner._master_view_indices(canonical, np.asarray(["m1", "m2"]))
    assert np.array_equal(mapping["iid"], [2, 0])
    assert np.array_equal(mapping["grouped"], [1, 3])


def test_strict_irls_failure_suppresses_level_effect_and_ci():
    indices = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    result = runner._bootstrap_summary(
        np.asarray([0.1, np.nan]), indices, strict=True
    )
    assert result["status"] == "NOT_EVALUABLE_SOLVER_FAILURE"
    assert result["mean"] is None
    assert result["ci95_marginal"] == [None, None]
    assert result["diagnostic_finite_survivors"]["n_finite"] == 1


def test_build_effects_propagates_failures_through_levels_effects_and_solver():
    config = {
        "arms": ["a"],
        "seeds": [1, 2],
        "analysis": {
            "primary_arms": ["a"],
            "expected_test_masters": 2,
        },
    }
    canonical = {
        "view_id": np.asarray(["v0", "v1", "v2", "v3"]),
        "master_id": np.asarray(["m2", "m1", "m2", "m1"]),
        "split": np.asarray(["test"] * 4),
        "mechanism": np.asarray(["grouped", "iid", "iid", "grouped"]),
    }
    bootstrap = {
        "master_id": np.asarray(["m1", "m2"]),
        "complete_n_2": np.asarray([[0, 1], [1, 0], [0, 0]], dtype=np.int32),
    }
    quotient = np.ones((1, 2, 8, 4), dtype=np.float64)
    irls_cells = {
        ("observed", "unit"): 0,
        ("corrected", "unit"): 1,
        ("observed", "learned"): 2,
        ("corrected", "learned"): 3,
    }
    for (relation, weight), view_index in irls_cells.items():
        quotient[0, 1, runner._cell_index(relation, weight, "irls"), view_index] = np.nan
    effects = runner._build_effects(config, canonical, bootstrap, quotient)
    arm = effects["arms"]["a"]
    assert arm["levels"]["irls"]["observed|unit"]["test|grouped"][
        "status"
    ] == "NOT_EVALUABLE_SOLVER_FAILURE"
    for effect_name in runner.EFFECTS:
        assert arm["effects"]["irls"][effect_name]["test|grouped"][
            "status"
        ] == "NOT_EVALUABLE_SOLVER_FAILURE"
        assert arm["effects"]["irls"][effect_name]["grouped_minus_iid"][
            "status"
        ] == "NOT_EVALUABLE_SOLVER_FAILURE"
        assert arm["solver_interactions"][effect_name]["test|grouped"][
            "status"
        ] == "NOT_EVALUABLE_SOLVER_FAILURE"
    assert arm["effects"]["wls"]["total_delivered_effect"]["test|grouped"][
        "status"
    ] == "ESTIMATED"


def test_nonconverged_last_state_is_preserved_but_inference_is_nan():
    result = SolverResult(
        x_hat=np.asarray([-0.5, 0.5]),
        reconstructed_log_ratio=np.asarray([1.0]),
        weights=np.asarray([0.25]),
        quotient_rmse=np.nan,
        relation_rmse=np.nan,
        weighted_residual_rmse=0.4,
        laplacian_rank=1,
        laplacian_condition=1.0,
        converged=False,
        iterations=99,
    )
    state = runner._score_state(result, np.asarray([-0.4, 0.4]), np.asarray([0.8]))
    assert np.array_equal(state.x_hat, result.x_hat)
    assert np.array_equal(state.weights, result.weights)
    assert np.isnan(state.quotient_rmse)
    assert np.isnan(state.relation_rmse)
    assert state.iterations == 99


def _tiny_matrix_fixture():
    edge_index = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    x_true = np.asarray([-0.5, 0.0, 0.5])
    clean = runner.incidence_matrix(3, edge_index) @ x_true
    observed = clean + np.asarray([0.01, -0.02, 0.03])
    canonical = {
        "edge_offsets": np.asarray([0, 3], dtype=np.int64),
        "node_offsets": np.asarray([0, 3], dtype=np.int64),
        "path_offsets": np.asarray([0, 0], dtype=np.int64),
        "n_nodes": np.asarray([3], dtype=np.int64),
        "edge_index": edge_index,
        "edge_valid": np.ones(3, dtype=bool),
        "observed_log_ratio": observed,
        "clean_log_ratio": clean,
        "x_true": x_true,
        "path_index": np.empty((0, 3), dtype=np.int64),
        "path_sign": np.empty((0, 3), dtype=np.float64),
        "path_valid": np.empty(0, dtype=bool),
        "edge_variance": np.full(3, 0.0016),
        "view_id": np.asarray(["v"]),
        "master_id": np.asarray(["m"]),
        "split": np.asarray(["validation"]),
        "mechanism": np.asarray(["iid"]),
    }
    config = json.loads(CONFIG_PATH.read_text())
    config["solver"]["irls_iterations"] = 100
    config["analysis"]["reuse_verification_indices"] = [0]
    config["execution"]["max_seconds"] = 30
    observation = runner._public_view(canonical, 0)

    def source_arrays(values, weights):
        wls = runner._solve_cell(
            observation, values, weights, "wls", config, x_true, clean
        )
        irls = runner._solve_cell(
            observation, values, weights, "irls", config, x_true, clean
        )
        return {
            "x_hat_wls": wls.x_hat,
            "x_hat_irls": irls.x_hat,
            "irls_weights": irls.weights,
            "wls_quotient_rmse": np.asarray([wls.quotient_rmse]),
            "wls_laplacian_rank": np.asarray([wls.laplacian_rank]),
            "wls_condition": np.asarray([wls.laplacian_condition]),
            "irls_laplacian_rank": np.asarray([irls.laplacian_rank]),
            "irls_condition": np.asarray([irls.laplacian_condition]),
            "irls_converged": np.asarray([irls.converged]),
            "irls_iterations": np.asarray([irls.iterations]),
        }

    control = {**canonical, **source_arrays(observed, np.ones(3))}
    corrected = clean + np.asarray([0.005, -0.005, 0.0])
    reliability = np.asarray([1.0, 0.2, 1.0])
    raw = {
        **canonical,
        "corrected_log_ratio": corrected,
        "reliability": reliability,
        **source_arrays(corrected, reliability),
    }
    return config, canonical, control, raw


def test_tiny_cpu_matrix_runs_all_cells_without_neural_forward():
    config, canonical, control, raw = _tiny_matrix_fixture()
    torch_before = "torch" in sys.modules
    payload, metrics, reuse = runner._run_model_matrix(
        canonical,
        control,
        raw,
        "tiny",
        7,
        config,
        runner.time.monotonic(),
    )
    assert payload["x_hat"].shape == (8, 3)
    assert payload["final_weights"].shape == (8, 3)
    assert metrics["quotient_rmse"].shape == (8, 1)
    assert np.all(metrics["converged"])
    assert reuse == {"wls_verified": 2, "irls_verified": 2}
    assert ("torch" in sys.modules) == torch_before


def test_deterministic_npz_writer_is_byte_exact(tmp_path: Path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    payload = {"z": np.arange(7), "a": np.asarray(["x", "y"])}
    runner._write_deterministic_npz(first, **payload)
    runner._write_deterministic_npz(second, **payload)
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as saved:
        assert np.array_equal(saved["z"], payload["z"])


def test_runner_source_has_no_torch_import_and_cpu_guard_is_explicit():
    source = runner.RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported |= {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "torch" not in imported
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "threadpool_limits" in source


def test_replay_embeds_cpu_environment_head_and_executable_hashes(tmp_path: Path):
    output = tmp_path / "artifact"
    output.mkdir()
    record = runner._tracked_clean_records([runner.PLAN_PATH], development=False)
    runner._write_replay(output, CONFIG_PATH, SOURCE, record, development=True)
    replay = (output / "replay.sh").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=''" in replay
    assert runner._git_output("rev-parse", "HEAD") in replay
    assert record[str(runner.PLAN_PATH.relative_to(REPO_ROOT))]["sha256"] in replay
    assert 'if [[ $# -ne 1 ]]' in replay


def test_tiny_main_writes_complete_byte_exact_artifact_lifecycle(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    outputs = iter((tmp_path / "first", tmp_path / "replay"))
    monkeypatch.setattr(
        runner,
        "_parse_args",
        lambda: argparse.Namespace(
            config=CONFIG_PATH,
            source=SOURCE,
            output=next(outputs),
            development=True,
        ),
    )
    canonical = {
        "view_id": np.asarray(["v"]),
        "master_id": np.asarray(["m"]),
        "split": np.asarray(["validation"]),
        "mechanism": np.asarray(["iid"]),
    }
    config = json.loads(CONFIG_PATH.read_text())
    raw_by_key = {
        f"{arm}|seed={seed}": {}
        for arm in config["arms"]
        for seed in config["seeds"]
    }
    monkeypatch.setattr(runner, "_tracked_clean_records", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runner,
        "_verify_source",
        lambda *args, **kwargs: (
            canonical,
            raw_by_key,
            {"master_id": np.asarray(["m"]), "complete_n_252": np.zeros((1, 1), dtype=np.int32)},
            {"source": "tiny"},
        ),
    )

    def fake_matrix(*args, **kwargs):
        metrics = {
            metric: np.full(
                (8, 1),
                True if metric == "converged" else 1,
                dtype=bool if metric == "converged" else np.int64 if metric in {"laplacian_rank", "iterations"} else np.float64,
            )
            for metric in runner.METRICS
        }
        return (
            {
                "cell_id": np.asarray(runner.CELL_IDS),
                "x_hat": np.zeros((8, 1)),
                "reconstructed_log_ratio": np.zeros((8, 1)),
                "final_weights": np.ones((8, 1)),
            },
            metrics,
            {"wls_verified": 2, "irls_verified": 2},
        )

    monkeypatch.setattr(runner, "_run_model_matrix", fake_matrix)
    result = {
        "status": "ESTIMATED",
        "mean": 0.0,
        "ci95_marginal": [0.0, 0.0],
        "n": 1,
        "diagnostic_finite_survivors": {
            "n_total": 1,
            "n_finite": 1,
            "failure_rate": 0.0,
            "finite_mean": 0.0,
        },
    }
    fake_effects = {
        "arms": {
            arm: {
                "effects": {
                    solver: {
                        "total_delivered_effect": {
                            slice_name: result for slice_name in runner.SLICES
                        }
                    }
                    for solver in runner.SOLVERS
                }
            }
            for arm in config["analysis"]["primary_arms"]
        }
    }
    monkeypatch.setattr(runner, "_build_summary", lambda *args: {"records": []})
    monkeypatch.setattr(runner, "_build_effects", lambda *args: fake_effects)
    monkeypatch.setattr(
        runner, "_build_failure_diagnostics", lambda *args: {"records": []}
    )
    runner.main()
    runner.main()

    first = tmp_path / "first"
    replay = tmp_path / "replay"
    first_manifest = json.loads((first / "manifest.json").read_text())
    replay_manifest = json.loads((replay / "manifest.json").read_text())
    assert first_manifest == replay_manifest
    assert len(first_manifest["files"]) == 25
    for relative in first_manifest["files"]:
        assert (first / relative).read_bytes() == (replay / relative).read_bytes()
    assert "runtime_observation.json" not in first_manifest["files"]
    assert len(list((first / "raw_solver").glob("*.npz"))) == 16


def test_official_source_checker_rejects_untracked_file(tmp_path: Path):
    candidate = tmp_path / "outside-repository.txt"
    candidate.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="live in repository"):
        runner._tracked_clean_records([candidate], development=False)
    records = runner._tracked_clean_records([candidate], development=True)
    record = next(iter(records.values()))
    assert record["tracked"] is False
    assert record["sha256"] == hashlib.sha256(b"x").hexdigest()
