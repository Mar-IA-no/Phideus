from __future__ import annotations

import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np


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


def test_synthetic_fit_select_adjudicate_and_exact_replay(tmp_path: Path) -> None:
    prospective = config()
    fit_data, utilities = synthetic_data(seed=5701)
    select_data, _ = synthetic_data(seed=5702)
    monitor_data, _ = synthetic_data(seed=5703)

    def run(root: Path) -> None:
        fit_stage = root / "fit-stage"
        fit_stage.mkdir(parents=True)
        (fit_stage / "phase_request.json").write_text(
            json.dumps({"phase": "fit", "synthetic": True}), encoding="utf-8"
        )
        fit = root / "fit"
        fit.mkdir()
        assert worker.run_fit(fit_stage, fit, prospective, utilities, fit_data) == "FIT_COMPLETE"

        select_stage = root / "select-stage"
        (select_stage / "previous").mkdir(parents=True)
        (select_stage / "phase_request.json").write_text(
            json.dumps({"phase": "select", "synthetic": True}), encoding="utf-8"
        )
        shutil.copy2(fit / "fit_core.json", select_stage / "previous/fit_core.json")
        select = root / "select"
        select.mkdir()
        assert worker.run_select(select_stage, select, prospective, utilities, select_data) == "SELECT_COMPLETE"
        freeze = json.loads((select / "selection_freeze.json").read_text(encoding="utf-8"))["selected"]
        assert freeze["proposer"]["threshold"] != "hard_only"

        adjudicate_stage = root / "adjudicate-stage"
        (adjudicate_stage / "previous").mkdir(parents=True)
        for name in ("fit_core.json", "fit_arrays.npz"):
            shutil.copy2(fit / name, adjudicate_stage / "previous" / name)
        for name in ("selection_core.json", "selection_freeze.json", "selection_arrays.npz"):
            shutil.copy2(select / name, adjudicate_stage / "previous" / name)
        adjudicate = root / "adjudicate"
        adjudicate.mkdir()
        assert worker.run_adjudicate(
            adjudicate_stage, adjudicate, prospective, utilities, monitor_data
        ) == "COMPLETE"
        report = json.loads((adjudicate / "analysis_core.json").read_text(encoding="utf-8"))
        assert report["estimand"]["unit"] == "pair_token"
        assert len(report["diagnostic_pattern"]["conditions"]) == 6
        assert set(report["absent_support_by_set"]) == {"0", "4", "8", "10", "12"}

    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    run(primary)
    run(replay)
    for relative in (
        "fit/fit_core.json",
        "fit/fit_arrays.npz",
        "select/selection_core.json",
        "select/selection_arrays.npz",
        "adjudicate/analysis_core.json",
        "adjudicate/result_arrays.npz",
    ):
        assert (primary / relative).read_bytes() == (replay / relative).read_bytes()


def test_monitor_without_disagreement_is_not_evaluable(tmp_path: Path) -> None:
    data, utilities = synthetic_data()
    data["disagreement"][:] = False
    data["weights"][:] = 0.0
    output = tmp_path / "output"
    output.mkdir()
    failure = worker._global_failure(data, config(), "adjudicate")
    assert failure is not None
    assert "disagreement_tokens" in failure["failed"]
