from __future__ import annotations

import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments/geometria_proporcional"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import _wave57_phase_worker as worker  # noqa: E402
import prepare_wave56_fresh as preparer  # noqa: E402
import run_wave56_contextual_gate as coordinator  # noqa: E402
from geometria_proporcional.wave52_policy import constrained_regret  # noqa: E402
from geometria_proporcional.wave56_contextual_gate import disagreement_weights  # noqa: E402
from geometria_proporcional.wave57_tail_guard import validate_wave57_frozen_config  # noqa: E402


CONFIG_PATH = EXPERIMENTS / "configs/wave57_contextual_tail_guard_fresh.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def synthetic_data(n_tokens: int = 420, seed: int = 5700) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    permutations = np.asarray(list(itertools.permutations(range(4))), dtype=np.int64)
    utilities = np.asarray([-0.3, 0.2, 0.8, 1.5], dtype=np.float64)[permutations]
    token_index = np.arange(n_tokens)[:, None]
    policy_index = np.arange(24)[None, :]
    hard = ((token_index + policy_index) % 4).astype(np.int64)
    disagreement = ((token_index + 2 * policy_index) % 5) != 0
    posterior = np.where(disagreement, (hard + 1 + policy_index % 2) % 4, hard)
    target = np.zeros((n_tokens, 4), dtype=bool)
    target[np.arange(n_tokens), np.arange(n_tokens) % 4] = True
    target[np.arange(n_tokens), (np.arange(n_tokens) + 1 + np.arange(n_tokens) % 2) % 4] = True
    hard_regret = constrained_regret(hard, target, utilities, 1.25)
    posterior_regret = constrained_regret(posterior, target, utilities, 1.25)
    gain = hard_regret - posterior_regret
    design = rng.normal(size=(n_tokens, 24, 17))
    # Make mean gain and harm separately learnable without copying truth exactly.
    design[..., 0] = gain + rng.normal(scale=0.18, size=gain.shape)
    design[..., 1] = (gain < -1e-12) + rng.normal(scale=0.35, size=gain.shape)
    ensemble = rng.normal(size=(n_tokens, 4))
    per_seed = ensemble[None, ...] + rng.normal(scale=0.1, size=(3, n_tokens, 4))
    hard_set = np.zeros((n_tokens, 4), dtype=bool)
    hard_set[np.arange(n_tokens), np.argmax(ensemble, axis=1)] = True
    tokens = np.asarray([f"wave57-{seed}-{index:05d}" for index in range(n_tokens)])
    return {
        "pair_token": tokens,
        "cluster_id": tokens.copy(),
        "target": target,
        "per_seed_logits": per_seed,
        "ensemble_logits": ensemble,
        "design_stratum": np.asarray(["NEAR_RIVAL"] * n_tokens),
        "cardinality": np.full(n_tokens, 2, dtype=np.int64),
        "split_role": np.asarray(["synthetic"] * n_tokens),
        "design": design,
        "gain": gain,
        "disagreement": disagreement,
        "weights": disagreement_weights(disagreement),
        "primary": np.ones(n_tokens, dtype=bool),
        "hard_actions": hard,
        "posterior_actions": posterior,
        "hard_set": hard_set,
        "advantage": np.maximum(design[..., 0], 0.0),
        "absent_support": np.zeros(n_tokens, dtype=bool),
    }, utilities


def test_frozen_config_is_accepted_by_preparer_and_worker() -> None:
    prospective = config()
    preparer.validate_prospective_config(prospective)
    validate_wave57_frozen_config(prospective)
    worker.base.validate_frozen_config(prospective)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("hard_set_tau",), 0.25),
        (("incompatible_regret_penalty",), 9.0),
        (("selection", "accuracy_noninferiority_margin"), 0.2),
        (("diagnostic_criteria", "regret_reduction_vs_hard_min"), 0.2),
        (("quantile_method",), "linear"),
        (("harm_epsilon",), 1e-6),
        (("bootstrap", "interval"), [5.0, 95.0]),
        (("shards", "salt"), "wrong-salt"),
        (("shared_boundary_interface", "compatibility_semantics"), "silent-alias"),
        (
            ("phase_worker_relative",),
            "experiments/geometria_proporcional/_wave56_phase_worker.py",
        ),
        (("scalar_gamma_grid",), [0.0, "hard_only"]),
        (("phase_runtime_sources",), []),
    ),
)
def test_complete_frozen_config_rejects_each_drift_family(
    path: tuple[str, ...], replacement: object
) -> None:
    prospective = config()
    target = prospective
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(RuntimeError, match="complete frozen config drifted"):
        validate_wave57_frozen_config(prospective)
    with pytest.raises(RuntimeError, match="complete frozen config drifted"):
        preparer.validate_prospective_config(prospective)


def test_runtime_hook_stages_only_bound_sources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    worker_path = coordinator.build_phase_runtime(source, config())
    assert worker_path.name == "_wave57_phase_worker.py"
    assert (source / "_wave56_phase_worker.py").is_file()
    assert (source / "geometria_proporcional/wave57_tail_guard.py").is_file()
    staged = {path.name for path in source.iterdir() if path.is_file()}
    assert staged == {
        "_wave56_phase_worker.py",
        "_wave57_phase_worker.py",
        "run_wave56_retrospective.py",
    }
    completed = subprocess.run(
        [sys.executable, str(worker_path), "--help"],
        cwd=source,
        env={"PATH": str(Path(sys.executable).parent), "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_unbound_runtime_hook_is_rejected(tmp_path: Path) -> None:
    prospective = config()
    prospective["phase_runtime_sources"].append(
        "experiments/geometria_proporcional/run_wave55_policy_bridge.py"
    )
    try:
        coordinator.build_phase_runtime(tmp_path / "source", prospective)
    except RuntimeError as error:
        assert "not bound" in str(error)
    else:
        raise AssertionError("unbound runtime source was accepted")


def test_proposer_hard_only_terminates_selection_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, utilities = synthetic_data(n_tokens=20)
    terminal = {
        "selected": {"threshold": "hard_only", "support": {"tokens": 0}},
        "grid": [],
    }
    monkeypatch.setattr(worker, "select_proposer", lambda *args, **kwargs: terminal)
    result = worker._selection_sequence(
        np.zeros_like(data["gain"]),
        np.zeros_like(data["gain"]),
        [np.zeros_like(data["gain"])] * 5,
        ["PASS"] * 5,
        data,
        utilities,
        config(),
        data["primary"],
        40,
        25,
    )
    assert result["status"] == "NOT_EVALUABLE"
    assert result["reason"] == "proposer_identity_or_low_support"


def test_guard_cells_fail_individually_on_authorized_token_support() -> None:
    data, utilities = synthetic_data(n_tokens=60)
    data["disagreement"][:, 0] = True
    proposals = np.zeros_like(data["disagreement"], dtype=bool)
    proposals[:, 0] = True
    probabilities = np.full(proposals.shape, np.nan)
    probabilities[:, 0] = np.linspace(0.0, 1.0, len(data["primary"]))
    result = worker.select_guard(
        probabilities,
        proposals,
        data,
        utilities,
        config(),
        data["primary"],
        30,
    )
    nonterminal = result["grid"][:-1]
    assert any(not row["evaluable"] for row in nonterminal)
    assert any(row["evaluable"] for row in nonterminal)
    assert result["grid"][-1]["threshold"] == "hard_only"


def test_one_invalid_sham_does_not_terminate_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prospective = config()
    data, utilities = synthetic_data(seed=5711)
    original = worker.conditional_harm_shuffle
    calls = 0

    def one_invalid(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        if calls == 0:
            result["diagnostics"]["permutable_fraction"] = 0.0
            result["diagnostics"]["hamming_global"] = 0.0
            result["diagnostics"]["hamming_weighted"] = 0.0
        calls += 1
        return result

    monkeypatch.setattr(worker, "conditional_harm_shuffle", one_invalid)
    stage = tmp_path / "stage"
    output = tmp_path / "output"
    stage.mkdir()
    output.mkdir()
    (stage / "phase_request.json").write_text(
        json.dumps({"phase": "fit", "synthetic": True}), encoding="utf-8"
    )
    assert worker.run_fit(stage, output, prospective, utilities, data) == "FIT_COMPLETE"
    rows = json.loads((output / "fit_core.json").read_text(encoding="utf-8"))["shams"]
    assert [row["status"] for row in rows] == ["NOT_EVALUABLE"] + ["PASS"] * 4
    assert (output / "fit_arrays.npz").is_file()

    select_data, _ = synthetic_data(seed=5712)
    select_stage = tmp_path / "select-stage"
    (select_stage / "previous").mkdir(parents=True)
    (select_stage / "phase_request.json").write_text(
        json.dumps({"phase": "select", "synthetic": True}), encoding="utf-8"
    )
    shutil.copy2(output / "fit_core.json", select_stage / "previous/fit_core.json")
    select_output = tmp_path / "select-output"
    select_output.mkdir()
    assert worker.run_select(
        select_stage, select_output, prospective, utilities, select_data
    ) == "SELECT_COMPLETE"
    selection = json.loads(
        (select_output / "selection_core.json").read_text(encoding="utf-8")
    )
    assert selection["sequence"]["guard"]["selected"] is not None
    assert [row["status"] for row in selection["sequence"]["shams"]] == [
        "NOT_EVALUABLE",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
    ]

    monitor_data, _ = synthetic_data(seed=5713)
    adjudicate_stage = tmp_path / "adjudicate-stage"
    (adjudicate_stage / "previous").mkdir(parents=True)
    for name in ("fit_core.json", "fit_arrays.npz"):
        shutil.copy2(output / name, adjudicate_stage / "previous" / name)
    for name in ("selection_core.json", "selection_freeze.json", "selection_arrays.npz"):
        shutil.copy2(select_output / name, adjudicate_stage / "previous" / name)
    adjudicate_output = tmp_path / "adjudicate-output"
    adjudicate_output.mkdir()
    assert worker.run_adjudicate(
        adjudicate_stage, adjudicate_output, prospective, utilities, monitor_data
    ) == "COMPLETE"
    analysis = json.loads(
        (adjudicate_output / "analysis_core.json").read_text(encoding="utf-8")
    )
    assert analysis["arms"]["mean_plus_harm_guard"]["status"] == "PASS"
    assert analysis["arms"]["mean_plus_shuffled_harm_guard"] == {
        "status": "NOT_EVALUABLE",
        "summary": None,
        "replicate_statuses": ["NOT_EVALUABLE"] + ["PASS"] * 4,
    }
    assert analysis["contrasts"]["main_minus_shuffled"]["status"] == "NOT_EVALUABLE"
    assert analysis["diagnostic_pattern"]["conditions"]["diagnostic_condition_5"] == "NOT_EVALUABLE"
    with np.load(adjudicate_output / "result_arrays.npz", allow_pickle=False) as arrays:
        assert not any(name.startswith("sham__average_metric__") for name in arrays.files)


def test_one_low_support_shard_is_granular_and_uses_shard_minima(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prospective = config()
    fit_data, utilities = synthetic_data(seed=5721)
    select_data, _ = synthetic_data(seed=5722)
    tokens_by_shard = {0: [], 1: []}
    candidate = 0
    while len(tokens_by_shard[0]) < 39 or len(tokens_by_shard[1]) < 381:
        token = f"imbalanced-{candidate:06d}"
        shard = int(worker._shard_assignment(np.asarray([token]))[0])
        limit = 39 if shard == 0 else 381
        if len(tokens_by_shard[shard]) < limit:
            tokens_by_shard[shard].append(token)
        candidate += 1
    tokens = np.asarray(tokens_by_shard[0] + tokens_by_shard[1])
    select_data["pair_token"] = tokens
    select_data["cluster_id"] = tokens.copy()

    fit_stage = tmp_path / "fit-stage"
    fit_output = tmp_path / "fit-output"
    fit_stage.mkdir()
    fit_output.mkdir()
    (fit_stage / "phase_request.json").write_text(
        json.dumps({"phase": "fit", "synthetic": True}), encoding="utf-8"
    )
    assert worker.run_fit(
        fit_stage, fit_output, prospective, utilities, fit_data
    ) == "FIT_COMPLETE"

    calls: list[tuple[int, int]] = []
    original = worker._selection_sequence

    def recording_sequence(*args, **kwargs):
        calls.append((int(args[-2]), int(args[-1])))
        return original(*args, **kwargs)

    monkeypatch.setattr(worker, "_selection_sequence", recording_sequence)
    select_stage = tmp_path / "select-stage"
    (select_stage / "previous").mkdir(parents=True)
    (select_stage / "phase_request.json").write_text(
        json.dumps({"phase": "select", "synthetic": True}), encoding="utf-8"
    )
    shutil.copy2(fit_output / "fit_core.json", select_stage / "previous/fit_core.json")
    select_output = tmp_path / "select-output"
    select_output.mkdir()
    assert worker.run_select(
        select_stage, select_output, prospective, utilities, select_data
    ) == "SELECT_COMPLETE"
    core = json.loads((select_output / "selection_core.json").read_text(encoding="utf-8"))
    assert core["shards"]["0"]["status"] == "NOT_EVALUABLE"
    assert core["shards"]["1"]["status"] == "PASS"
    assert calls[0] == (40, 25)
    assert calls[1:] == [(20, 12)]


def run_synthetic_package(root: Path) -> None:
    prospective = config()
    fit_data, utilities = synthetic_data(seed=5701)
    select_data, _ = synthetic_data(seed=5702)
    monitor_data, _ = synthetic_data(seed=5703)
    monitor_data["target"][:40] = np.asarray([True, False, True, False])
    hard_regret = constrained_regret(
        monitor_data["hard_actions"], monitor_data["target"], utilities, 1.25
    )
    posterior_regret = constrained_regret(
        monitor_data["posterior_actions"], monitor_data["target"], utilities, 1.25
    )
    monitor_data["gain"] = hard_regret - posterior_regret
    fit_stage = root / "fit-stage"
    fit_stage.mkdir(parents=True)
    (fit_stage / "phase_request.json").write_text(
        json.dumps({"phase": "fit", "synthetic": True}), encoding="utf-8"
    )
    fit = root / "phases/fit.complete"
    fit.mkdir(parents=True)
    assert worker.run_fit(fit_stage, fit, prospective, utilities, fit_data) == "FIT_COMPLETE"

    select_stage = root / "select-stage"
    (select_stage / "previous").mkdir(parents=True)
    (select_stage / "phase_request.json").write_text(
        json.dumps({"phase": "select", "synthetic": True}), encoding="utf-8"
    )
    shutil.copy2(fit / "fit_core.json", select_stage / "previous/fit_core.json")
    select = root / "phases/select.complete"
    select.mkdir(parents=True)
    assert worker.run_select(select_stage, select, prospective, utilities, select_data) == "SELECT_COMPLETE"
    freeze = json.loads((select / "selection_freeze.json").read_text(encoding="utf-8"))["selected"]
    assert freeze["proposer"]["threshold"] != "hard_only"

    adjudicate_stage = root / "adjudicate-stage"
    (adjudicate_stage / "previous").mkdir(parents=True)
    for name in ("fit_core.json", "fit_arrays.npz"):
        shutil.copy2(fit / name, adjudicate_stage / "previous" / name)
    for name in ("selection_core.json", "selection_freeze.json", "selection_arrays.npz"):
        shutil.copy2(select / name, adjudicate_stage / "previous" / name)
    adjudicate = root / "phases/adjudicate.complete"
    adjudicate.mkdir(parents=True)
    assert worker.run_adjudicate(
        adjudicate_stage, adjudicate, prospective, utilities, monitor_data
    ) == "COMPLETE"


def test_synthetic_fit_select_adjudicate_and_exact_replay(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    run_synthetic_package(primary)
    run_synthetic_package(replay)
    report = json.loads(
        (primary / "phases/adjudicate.complete/analysis_core.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["estimand"]["unit"] == "pair_token"
    assert len(report["diagnostic_pattern"]["conditions"]) == 6
    assert set(report["absent_support_by_set"]) == {"0", "4", "8", "10", "12"}
    support = report["absent_support_by_set"]["4"]
    assert support["status"] == "EVALUABLE"
    assert support["arm_statuses"]["mean_plus_shuffled_harm_guard"] == "PASS"
    assert support["summaries"]["mean_plus_shuffled_harm_guard"] is not None
    assert support["sham_replicate_statuses"] == ["PASS"] * 5
    assert "main_minus_shuffled" in support["contrasts"]
    assert "scalar_advantage_gate" in report["secondary_diagnostics"]
    assert "all_in_catalog_by_observational_slice" in report["secondary_diagnostics"][
        "mean_plus_harm_guard_sensitivities"
    ]

    for relative in (
        "phases/fit.complete/fit_core.json",
        "phases/fit.complete/fit_freeze.json",
        "phases/fit.complete/fit_arrays.npz",
        "phases/select.complete/selection_core.json",
        "phases/select.complete/selection_freeze.json",
        "phases/select.complete/selection_arrays.npz",
        "phases/adjudicate.complete/analysis_core.json",
        "phases/adjudicate.complete/result_arrays.npz",
    ):
        assert (primary / relative).read_bytes() == (replay / relative).read_bytes()
    preparation = {
        "config_sha256": coordinator.sha256_file(CONFIG_PATH),
        "prospective_config": config(),
    }
    for root in (primary, replay):
        (root / "preparation_freeze.json").write_text(
            json.dumps(preparation, sort_keys=True), encoding="utf-8"
        )
    comparison = coordinator.compare_reference(replay, primary)
    assert comparison["all_exact"] is True
    assert comparison["checks"]["preparation/prospective_config"] is True
    assert comparison["checks"]["fit/fit_freeze.json"] is True
    assert comparison["checks"]["select/selection_freeze.json"] is True


@pytest.mark.parametrize("relative", ("fit/fit_freeze.json", "select/selection_freeze.json"))
def test_replay_rejects_each_mutated_phase_freeze(tmp_path: Path, relative: str) -> None:
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    run_synthetic_package(primary)
    shutil.copytree(primary, replay)
    path = replay / "phases" / f"{relative.split('/')[0]}.complete" / relative.split('/')[1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["adversarial_mutation"] = relative
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="replay mismatch"):
        coordinator.compare_reference(replay, primary)


def test_monitor_without_disagreement_is_not_evaluable(tmp_path: Path) -> None:
    data, utilities = synthetic_data()
    data["disagreement"][:] = False
    data["weights"][:] = 0.0
    output = tmp_path / "output"
    output.mkdir()
    failure = worker._global_failure(data, config(), "adjudicate")
    assert failure is not None
    assert "disagreement_tokens" in failure["failed"]
