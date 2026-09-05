from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from geometria_proporcional.wave58_open_diagnostic import (
    HGB_CLASSIFIER_KWARGS,
    HGB_REGRESSOR_KWARGS,
    LOGISTIC_KWARGS,
    Q_GUARD_8,
    Q_GUARD_9,
    Q_GUARD_WAVE57,
    Q_PROPOSER,
    RIDGE_KWARGS,
    apply_selection,
    canonical_candidate_ids,
    derive_targets,
    fit_hgb_state,
    fit_logistic_state,
    fit_ridge_state,
    paired_bootstrap_indices,
    paired_delta_ci,
    score_hgb_state,
    score_linear_state,
    select_candidate,
    select_sequential,
    validate_runtime_contract,
)


REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "experiments/geometria_proporcional/run_wave58_open_diagnostic.py"
CONFIG_PATH = REPO / "experiments/geometria_proporcional/configs/wave58_open_model_class_diagnostic.json"
spec = importlib.util.spec_from_file_location("wave58_runner", RUNNER_PATH)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_runtime_contract_and_rosters_are_exact() -> None:
    validate_runtime_contract()
    config = load_config()
    assert len(canonical_candidate_ids()) == 36
    assert len(set(canonical_candidate_ids())) == 36
    assert len(config["historical_probe_ids"]) == 24
    assert len(set(config["historical_probe_ids"])) == 24
    assert tuple(config["proposer_quantiles"]) == Q_PROPOSER
    assert tuple(config["guard_quantiles_8"]) == Q_GUARD_8
    assert tuple(config["guard_quantiles_9"]) == Q_GUARD_9
    assert tuple(config["legacy_guard_quantiles"]) == Q_GUARD_WAVE57
    assert RIDGE_KWARGS["solver"] == "svd"
    assert LOGISTIC_KWARGS["solver"] == "lbfgs"
    assert HGB_REGRESSOR_KWARGS["random_state"] == 5801
    assert HGB_CLASSIFIER_KWARGS["max_iter"] == 100


def test_config_rejects_bound_input_hash_drift() -> None:
    config = load_config()
    config["inputs"]["train_bundle"][1] = "0" * 64
    with pytest.raises(RuntimeError, match="input hash drifted: train_bundle"):
        runner.validate_config(config, require_frozen_sources=False)


def manual_target_fixture() -> tuple[dict[str, np.ndarray], np.ndarray]:
    target = np.asarray(
        [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=bool
    )
    utilities = np.asarray([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])
    hard = np.asarray([[1, 0], [1, 1], [2, 2]], dtype=np.int64)
    posterior = np.asarray([[0, 2], [2, 3], [3, 1]], dtype=np.int64)
    gain = np.asarray(
        [[-1.0 / 3.0, -1.25], [1.0 / 3.0, -1.25], [1.0 / 3.0, -1.25]]
    )
    return {
        "target": target,
        "hard_actions": hard,
        "posterior_actions": posterior,
        "gain": gain,
    }, utilities


def test_all_six_targets_match_manual_formula_domain_and_polarity() -> None:
    data, utilities = manual_target_fixture()
    targets = derive_targets(data, utilities, 1.25)
    assert set(targets) == {
        "gain",
        "harm",
        "compatibility_loss",
        "posterior_incompatibility",
        "accuracy_loss",
        "tail_breach",
    }
    np.testing.assert_array_equal(targets["harm"], data["gain"] < -1e-12)
    np.testing.assert_array_equal(
        targets["compatibility_loss"],
        [[False, True], [False, True], [False, True]],
    )
    np.testing.assert_array_equal(
        targets["posterior_incompatibility"],
        [[False, True], [False, True], [False, True]],
    )
    np.testing.assert_array_equal(
        targets["accuracy_loss"],
        [[True, True], [False, True], [False, True]],
    )
    np.testing.assert_array_equal(
        targets["tail_breach"],
        [[True, True], [False, True], [False, True]],
    )
    assert all(value.shape == (3, 2) for value in targets.values())


def weighted_model_fixture(seed: int = 58):
    rng = np.random.Generator(np.random.PCG64(seed))
    design = rng.normal(size=(240, 17))
    weights = rng.uniform(0.05, 1.0, size=len(design))
    regression = 0.7 * design[:, 0] - 0.3 * design[:, 4]
    binary = regression > np.median(regression)
    return design, weights, regression, binary


def test_linear_states_reconstruct_scores_and_preserve_objective_contract() -> None:
    design, weights, regression, binary = weighted_model_fixture()
    ridge_state, ridge = fit_ridge_state(design, regression, weights)
    np.testing.assert_allclose(
        score_linear_state(ridge_state, design), ridge.predict((design - ridge_state["mean"]) / ridge_state["scale"]),
        rtol=0.0,
        atol=2e-15,
    )
    log_state, logistic = fit_logistic_state(design, binary, weights)
    assert logistic is not None and log_state["status"] == "PASS"
    np.testing.assert_allclose(
        score_linear_state(log_state, design),
        logistic.predict_proba((design - log_state["mean"]) / log_state["scale"])[:, 1],
        rtol=0.0,
        atol=2e-15,
    )
    balanced, _ = fit_logistic_state(
        design, binary, weights, class_weight="balanced"
    )
    assert log_state["kwargs"]["class_weight"] is None
    assert balanced["kwargs"]["class_weight"] == "balanced"


def test_hgb_typed_state_reconstructs_exact_scores() -> None:
    design, weights, regression, binary = weighted_model_fixture()
    state, arrays = fit_hgb_state(
        "test_regressor", design, regression, weights, classifier=False, seed=5801
    )
    first = score_hgb_state(state, arrays, design)
    second = score_hgb_state(state, arrays, design)
    np.testing.assert_array_equal(first, second)
    assert state["n_iter"] == 100
    assert len(state["tree_keys"]) == 100
    assert state["transport_only"] is True
    assert state["score_authority"] == "preserved_float64_scores_per_split"
    assert all(values.dtype != object for values in arrays.values())

    classifier_state, classifier_arrays = fit_hgb_state(
        "test_classifier", design, binary, weights, classifier=True, seed=5802
    )
    probability = score_hgb_state(classifier_state, classifier_arrays, design)
    assert np.all((probability > 0.0) & (probability < 1.0))


def test_single_class_is_not_evaluable() -> None:
    design, weights, _, _ = weighted_model_fixture()
    linear, model = fit_logistic_state(design, np.zeros(len(design)), weights)
    assert model is None
    assert linear["status"] == "NOT_EVALUABLE"
    hgb, arrays = fit_hgb_state(
        "single", design, np.zeros(len(design), dtype=bool), weights,
        classifier=True, seed=5802,
    )
    assert hgb["status"] == "NOT_EVALUABLE"
    assert arrays == {}
    invalid = design.copy()
    invalid[0, 0] = np.inf
    ridge, model = fit_ridge_state(invalid, np.zeros(len(invalid)), weights)
    assert model is None and ridge["status"] == "NOT_EVALUABLE"
    logistic, model = fit_logistic_state(invalid, np.zeros(len(invalid)), weights)
    assert model is None and logistic["status"] == "NOT_EVALUABLE"


def selection_fixture(
    tokens: int = 300, policies: int = 1, *, posterior_better: bool = False
):
    pair_token = np.asarray([f"token-{index:04d}" for index in range(tokens)])
    target = np.tile(np.asarray([[1, 0, 0, 0]], dtype=bool), (tokens, 1))
    hard = np.full((tokens, policies), 1 if posterior_better else 0, dtype=np.int64)
    posterior = np.zeros_like(hard)
    disagreement = np.ones((tokens, policies), dtype=bool)
    utilities = np.tile(np.asarray([[3.0, 2.0, 1.0, 0.0]]), (policies, 1))
    data = {
        "pair_token": pair_token,
        "target": target,
        "hard_actions": hard,
        "posterior_actions": posterior,
        "disagreement": disagreement,
        "primary": np.ones(tokens, dtype=bool),
    }
    proposer = np.linspace(0.0, 1.0, tokens * policies).reshape(tokens, policies)
    guard = np.linspace(1.0, 0.0, tokens * policies).reshape(tokens, policies)
    return data, utilities, proposer, guard


def test_hard_terminal_identity_and_sequential_short_circuit() -> None:
    data, utilities, proposer, guard = selection_fixture(tokens=80)
    result = select_sequential(
        np.ones_like(proposer), [("harm", guard)], data, utilities, 1.25
    )
    assert result["selected"]["terminal"] == "HARD_ONLY"
    assert result["grid"] == []
    actions, proposals, authorized = apply_selection(
        result, proposer, {"harm": guard}, data
    )
    np.testing.assert_array_equal(actions, data["hard_actions"])
    assert not proposals.any() and not authorized.any()


def test_nary_hct_cartesian_grid_keeps_duplicates_and_active_tie_break() -> None:
    data, utilities, proposer, _ = selection_fixture(
        tokens=300, posterior_better=True
    )
    guard = np.where(proposer > 0.5, 0.0, 1.0)
    result = select_candidate(
        proposer,
        [("harm", guard), ("compatibility_loss", guard), ("tail_breach", guard)],
        data,
        utilities,
        1.25,
        selector="FIXED-W57-PROPOSER",
        guard_quantiles=Q_GUARD_8,
        fixed_proposer_threshold=0.2,
    )
    assert len(result["grid"]) == len(Q_GUARD_8) ** 3 + 1
    assert result["selected"].get("terminal") is None
    assert result["selected"]["guard_qs"] == [0.7, 0.7, 0.7]
    assert result["selected"]["authorization_support"]["tokens"] == 150
    thresholds = [
        tuple(row["guard_thresholds"])
        for row in result["grid"][:-1]
    ]
    assert len(thresholds) == len(Q_GUARD_8) ** 3


def test_joint_and_shard_robust_have_total_deterministic_selection() -> None:
    data, utilities, proposer, _ = selection_fixture(
        tokens=300, posterior_better=True
    )
    guard = np.where(proposer > 0.6, 0.0, 1.0)
    joint = select_candidate(
        proposer, [("harm", guard)], data, utilities, 1.25, selector="JOINT"
    )
    robust = select_candidate(
        proposer,
        [("harm", guard)],
        data,
        utilities,
        1.25,
        selector="JOINT_SHARD_ROBUST",
    )
    assert len(joint["grid"]) == len(Q_PROPOSER) * len(Q_GUARD_9) + 1
    assert len(robust["grid"]) == len(joint["grid"])
    assert joint["selected"].get("terminal") is None
    assert robust["selected"].get("terminal") is None
    assert all(set(row["shards"]) == {"0", "1"} for row in robust["grid"][:-1])


def test_bootstrap_is_deterministic_ordered_bounded_and_paired() -> None:
    tokens = np.asarray(["a", "b", "c", "d"])
    left = paired_bootstrap_indices(tokens, replicates=25)
    right = paired_bootstrap_indices(tokens, replicates=25)
    np.testing.assert_array_equal(left, right)
    assert left.shape == (25, 4)
    assert left.min() >= 0 and left.max() < 4
    with pytest.raises(ValueError, match="lexicographically ordered"):
        paired_bootstrap_indices(tokens[::-1], replicates=2)
    ci = paired_delta_ci(np.arange(4.0), np.zeros(4), left)
    assert set(ci) == {"mean_diff", "ci95_low", "ci95_high"}


def test_replay_comparator_accepts_structured_and_text_arrays(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    for root in (primary, replay):
        for relative in runner.SCIENTIFIC_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative.endswith(".npz"):
                structured = np.asarray(
                    [(1.0, 2)], dtype=[("value", "<f8"), ("index", "<i8")]
                )
                np.savez(path, structured=structured, text=np.asarray(["x"]), floating=np.asarray([np.nan]))
            else:
                path.write_text("exact\n", encoding="utf-8")
    assert runner.compare_outputs(primary, replay)["all_exact"] is True


def test_canonical_id_parser_does_not_confuse_proposer_and_risk() -> None:
    assert runner.canonical_components(
        "C-HGB-LOGISTIC-HARM-JOINT"
    ) == ("HGB", "LOGISTIC", "HARM", "JOINT")
    assert runner.canonical_components(
        "C-RIDGE-HGB-INCOMPATIBILITY-SEQUENTIAL"
    ) == ("RIDGE", "HGB", "INCOMPATIBILITY", "SEQUENTIAL")


def test_historical_p3_uses_qg8_then_qg9() -> None:
    shape = (300, 1)
    scores = {
        name: np.zeros(shape, dtype=np.float64)
        for name in (
            "ridge_gain",
            "hgb_gain",
            "log_harm",
            "log_compatibility",
            "log_compatibility_balanced",
            "log_tail",
            "log_incompatibility",
            "log_incompatibility_balanced",
            "log_accuracy",
            "log_accuracy_balanced",
            "hgb_harm",
            "hgb_incompatibility",
        )
    }
    grids = runner.historical_specs(scores)["P3-I-N"][3]
    assert grids == (Q_GUARD_8, Q_GUARD_9)


def test_not_evaluable_models_propagate_without_pruning_roster() -> None:
    data, utilities, proposer, guard = selection_fixture(tokens=300)
    scores = {
        name: guard.copy()
        for name in (
            "ridge_gain",
            "hgb_gain",
            "log_harm",
            "log_compatibility",
            "log_compatibility_balanced",
            "log_tail",
            "log_incompatibility",
            "log_incompatibility_balanced",
            "log_accuracy",
            "log_accuracy_balanced",
            "hgb_harm",
            "hgb_incompatibility",
        )
    }
    scores["hgb_harm"][:] = np.nan
    results = runner.canonical_selections(scores, data, utilities, 1.25)
    assert len(results) == 36
    affected = [
        row
        for candidate_id, row in results.items()
        if runner.canonical_components(candidate_id)[1] == "HGB"
        and "HARM" in runner.canonical_components(candidate_id)[2]
    ]
    assert affected and all(row["status"] == "NOT_EVALUABLE" for row in affected)


def test_fit_models_passes_identical_sample_weights_to_every_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    data = runner.load_bundle(REPO / config["inputs"]["train_bundle"][0])
    utilities, _ = runner.load_utilities(REPO / config["inputs"]["policy_manifest"][0])
    expected = data["weights"][data["primary"][:, None] & data["disagreement"]]
    observed: list[tuple[str, np.ndarray, str | None]] = []

    def ridge(_x, _y, weights):
        observed.append(("ridge", np.asarray(weights).copy(), None))
        return {"kind": "ridge", "status": "NOT_EVALUABLE"}, None

    def logistic(_x, _y, weights, *, class_weight=None):
        observed.append(("logistic", np.asarray(weights).copy(), class_weight))
        return {"kind": "logistic", "status": "NOT_EVALUABLE"}, None

    def hgb(_name, _x, _y, weights, *, classifier, seed):
        observed.append(("hgb", np.asarray(weights).copy(), None))
        return {
            "kind": "hgb_classifier" if classifier else "hgb_regressor",
            "status": "NOT_EVALUABLE",
        }, {}

    monkeypatch.setattr(runner, "fit_ridge_state", ridge)
    monkeypatch.setattr(runner, "fit_logistic_state", logistic)
    monkeypatch.setattr(runner, "fit_hgb_state", hgb)
    monkeypatch.setattr(
        runner,
        "score_grid",
        lambda _state, _arrays, bundle: np.full(
            bundle["disagreement"].shape, np.nan, dtype=np.float64
        ),
    )
    runner.fit_models(data, utilities, config["penalty"])
    assert len(observed) == 12
    assert all(np.array_equal(weights, expected) for _, weights, _ in observed)
    assert sum(class_weight == "balanced" for _, _, class_weight in observed) == 3


def test_runner_has_cpu_airlock_and_no_cuda_import() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    module_source = (
        REPO / "src/geometria_proporcional/wave58_open_diagnostic.py"
    ).read_text(encoding="utf-8")
    combined = source + module_source
    assert "import torch" not in combined
    assert "import cupy" not in combined
    assert "CUDA_VISIBLE_DEVICES" in source
    assert '"OMP_NUM_THREADS": "4"' in source
    assert '"MKL_NUM_THREADS": "4"' in source
    assert '"OPENBLAS_NUM_THREADS": "4"' in source
    assert "threadpool_limits(limits=4)" in source
    assert '"monitor_bundle.npz"' not in source.split('"select": {', 1)[1].split("},", 1)[0]


def test_worker_config_contains_no_original_or_future_paths() -> None:
    config = load_config()
    for phase in ("fit", "select", "monitor"):
        staged = runner.worker_config(config, phase)
        encoded = json.dumps(staged, sort_keys=True)
        assert "inputs" not in staged
        assert "source_paths" not in staged
        assert "data/geometria_proporcional" not in encoded
        assert staged["phase"] == phase


def test_phase_freeze_validation_precedes_future_bundle_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    runner.write_json(stage / "config.json", runner.worker_config(load_config(), "select"))

    def forbidden_bundle(_path):
        raise AssertionError("future bundle opened before freeze validation")

    monkeypatch.setattr(runner, "load_bundle", forbidden_bundle)
    monkeypatch.setattr(
        runner,
        "validate_fit_freeze",
        lambda _path: (_ for _ in ()).throw(RuntimeError("bad FIT freeze")),
    )
    with pytest.raises(RuntimeError, match="bad FIT freeze"):
        runner.run_select_phase(stage, tmp_path / "select-output")

    runner.write_json(stage / "config.json", runner.worker_config(load_config(), "monitor"))
    monkeypatch.setattr(
        runner,
        "validate_selection_freeze",
        lambda *_: (_ for _ in ()).throw(RuntimeError("bad SELECT freeze")),
    )
    with pytest.raises(RuntimeError, match="bad SELECT freeze"):
        runner.run_monitor_phase(stage, tmp_path / "monitor-output")


@pytest.mark.skipif(
    os.geteuid() != 0 or shutil.which("setpriv") is None,
    reason="physical Wave 58 worker requires root and setpriv",
)
def test_fit_worker_crosses_nobody_readonly_airlock(tmp_path: Path) -> None:
    config = load_config()
    inputs = config["inputs"]
    output = tmp_path / "fit"
    phase_config = tmp_path / "fit-config.json"
    runner.write_json(phase_config, runner.worker_config(config, "fit"))
    runner.run_worker(
        "fit",
        {
            "config.json": phase_config,
            "train_bundle.npz": REPO / inputs["train_bundle"][0],
            "policy_manifest.json": REPO / inputs["policy_manifest"][0],
        },
        output,
    )
    assert (output / "fit_freeze.json").is_file()
    assert json.loads((output / "model_states.json").read_text())["status"] == "FIT_COMPLETE"


def test_real_legacy_wave57_replay_is_exact(tmp_path: Path) -> None:
    config = load_config()
    train = runner.load_bundle(REPO / config["inputs"]["train_bundle"][0])
    validation = runner.load_bundle(REPO / config["inputs"]["validation_bundle"][0])
    monitor = runner.load_bundle(REPO / config["inputs"]["monitor_bundle"][0])
    utilities, _ = runner.load_utilities(REPO / config["inputs"]["policy_manifest"][0])
    states, arrays = runner.fit_models(train, utilities, config["penalty"])
    validation_scores = runner.score_all(states, arrays, validation)
    monitor_scores = runner.score_all(states, arrays, monitor)
    frozen = select_sequential(
        validation_scores["ridge_gain"],
        [("harm", validation_scores["log_harm"])],
        validation,
        utilities,
        config["penalty"],
        guard_quantiles=Q_GUARD_WAVE57,
    )["selected"]
    staged = tmp_path / "legacy-stage"
    staged.mkdir()
    shutil.copy2(REPO / config["inputs"]["legacy_selection"][0], staged / "legacy_selection.npz")
    shutil.copy2(REPO / config["inputs"]["legacy_results"][0], staged / "legacy_results.npz")
    result = runner.verify_legacy(
        staged,
        states,
        validation,
        monitor,
        utilities,
        config["penalty"],
        validation_scores,
        monitor_scores,
        frozen,
    )
    assert result["all_exact"] is True
    assert all(result["checks"].values())


def test_source_inventory_has_no_unexpected_gpu_or_colab_path() -> None:
    config = load_config()
    assert all("colab" not in path.lower() for path in config["source_paths"])
    assert all("cuda" not in path.lower() for path in config["source_paths"])
    assert config["device"] == "cpu"


@pytest.mark.skipif(
    os.geteuid() != 0 or shutil.which("setpriv") is None,
    reason="full physical Wave 58 integration requires root and setpriv",
)
def test_full_physical_primary_replay_from_empty_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "validate_config",
        lambda _config, *, require_frozen_sources: validate_runtime_contract(),
    )
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    runner.coordinator(CONFIG_PATH, primary, None)
    runner.coordinator(CONFIG_PATH, replay, primary)
    comparison = json.loads((replay / "runtime.json").read_text())["replay"]
    assert comparison["all_exact"] is True
    assert len(comparison["checks"]) == len(runner.SCIENTIFIC_FILES)
    analysis = json.loads((primary / "analysis.json").read_text())
    assert analysis["candidate_counts"] == {"canonical": 36, "historical": 24}
    assert analysis["legacy_wave57"]["all_exact"] is True
    report = (primary / "REPORT.md").read_text(encoding="utf-8")
    assert all(candidate_id in report for candidate_id in canonical_candidate_ids())
    assert all(candidate_id in report for candidate_id in load_config()["historical_probe_ids"])
    with np.load(primary / "scores_and_masks.npz", allow_pickle=False) as arrays:
        assert "validation__C__HGB__HGB__INCOMPATIBILITY__JOINT__authorized" in arrays
        assert "monitor__C__HGB__HGB__INCOMPATIBILITY__JOINT__authorized" in arrays
        assert "validation__target__harm" in arrays
        assert "monitor__target__harm" in arrays
    runner.validate_fit_freeze(primary / "fit")
    runner.validate_selection_freeze(primary / "select", primary / "fit")
