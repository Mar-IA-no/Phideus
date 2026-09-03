from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from geometria_proporcional.wave52_policy import constrained_regret
from geometria_proporcional.wave56_contextual_gate import disagreement_weights


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = REPO_ROOT / "experiments/geometria_proporcional/_wave56_phase_worker.py"
RUNNER_PATH = REPO_ROOT / "experiments/geometria_proporcional/run_wave56_contextual_gate.py"
PREP_PATH = REPO_ROOT / "experiments/geometria_proporcional/prepare_wave56_fresh.py"
CONFIG_PATH = REPO_ROOT / "experiments/geometria_proporcional/configs/wave56_contextual_gate_fresh.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load_module(WORKER_PATH, "wave56_phase_worker_test")
runner = load_module(RUNNER_PATH, "wave56_runner_test")
prep = load_module(PREP_PATH, "wave56_prep_test")


def prospective_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def synthetic_data(n_tokens: int = 120, *, seed: int = 5600) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    permutations = np.asarray(list(itertools.permutations(range(4))), dtype=np.int64)
    utilities = np.asarray([-0.3, 0.2, 0.8, 1.5], dtype=np.float64)[permutations]
    token_index = np.arange(n_tokens)[:, None]
    policy_index = np.arange(24)[None, :]
    hard = ((token_index + policy_index) % 4).astype(np.int64)
    disagreement = ((token_index + 2 * policy_index) % 4) != 0
    posterior = np.where(disagreement, (hard + 1 + policy_index % 2) % 4, hard)
    target = np.zeros((n_tokens, 4), dtype=bool)
    target[np.arange(n_tokens), np.arange(n_tokens) % 4] = True
    target[np.arange(n_tokens), (np.arange(n_tokens) + 1 + np.arange(n_tokens) % 2) % 4] = True
    penalty = 1.25
    hard_regret = constrained_regret(hard, target, utilities, penalty)
    posterior_regret = constrained_regret(posterior, target, utilities, penalty)
    gain = hard_regret - posterior_regret
    design = rng.normal(size=(n_tokens, 24, 17))
    design[..., 0] = np.abs(design[..., 0])
    design[..., 1] = gain + rng.normal(scale=0.05, size=gain.shape)
    ensemble = rng.normal(size=(n_tokens, 4))
    per_seed = ensemble[None, ...] + rng.normal(scale=0.1, size=(3, n_tokens, 4))
    hard_set = np.zeros((n_tokens, 4), dtype=bool)
    hard_set[np.arange(n_tokens), np.argmax(ensemble, axis=1)] = True
    tokens = np.asarray([f"pair-{index:04d}" for index in range(n_tokens)])
    data = {
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
        "advantage": design[..., 0],
        "absent_support": np.arange(n_tokens) % 3 == 0,
    }
    return data, utilities


def run_synthetic_pipeline(root: Path) -> None:
    config = prospective_config()
    fit_data, utilities = synthetic_data(seed=5601)
    select_data, _ = synthetic_data(seed=5602)
    monitor_data, _ = synthetic_data(seed=5603)
    fit = root / "phases/fit.complete"
    select = root / "phases/select.complete"
    adjudicate = root / "phases/adjudicate.complete"
    root.mkdir(parents=True)
    (root / "phase_request.json").write_text(
        json.dumps({"phase": "fit", "synthetic": True}), encoding="utf-8"
    )
    fit.mkdir(parents=True)
    assert worker.run_fit(root, fit, config, utilities, fit_data) == "FIT_COMPLETE"
    select_stage = root / "select-stage/previous"
    select_stage.mkdir(parents=True)
    (select_stage.parent / "phase_request.json").write_text(
        json.dumps({"phase": "select", "synthetic": True}), encoding="utf-8"
    )
    shutil.copy2(fit / "fit_core.json", select_stage / "fit_core.json")
    assert worker.run_select(select_stage.parent, select, config, utilities, select_data) == "SELECT_COMPLETE"
    adjudicate_stage = root / "adjudicate-stage/previous"
    adjudicate_stage.mkdir(parents=True)
    shutil.copy2(fit / "fit_core.json", adjudicate_stage / "fit_core.json")
    shutil.copy2(fit / "fit_arrays.npz", adjudicate_stage / "fit_arrays.npz")
    shutil.copy2(select / "selection_freeze.json", adjudicate_stage / "selection_freeze.json")
    shutil.copy2(select / "selection_core.json", adjudicate_stage / "selection_core.json")
    shutil.copy2(select / "selection_arrays.npz", adjudicate_stage / "selection_arrays.npz")
    assert worker.run_adjudicate(
        adjudicate_stage.parent, adjudicate, config, utilities, monitor_data
    ) == "COMPLETE"


def test_frozen_config_rejects_model_or_rng_drift() -> None:
    config = prospective_config()
    prep.validate_prospective_config(config)
    worker.validate_frozen_config(config)
    config["primary_model"]["alpha"] = 10.0
    with pytest.raises(RuntimeError, match="alphas drifted"):
        worker.validate_frozen_config(config)


def test_frozen_config_rejects_absent_support_drift() -> None:
    config = prospective_config()
    config["absent_support"]["set_indices"] = [0, 4, 8, 10, 13]
    with pytest.raises(RuntimeError, match="absent-support indices drifted"):
        worker.validate_frozen_config(config)


def test_preparation_rejects_incomplete_diagnostic_contract() -> None:
    config = prospective_config()
    del config["diagnostic_criteria"]["replay_exact_required"]
    with pytest.raises(RuntimeError, match="diagnostic criteria are incomplete"):
        prep.validate_prospective_config(config)


def test_execution_source_manifest_covers_local_runtime_closure() -> None:
    config = prospective_config()
    required = set(config["required_execution_sources"])
    expected = {
        "experiments/geometria_proporcional/run_wave56_retrospective.py",
        "src/geometria_proporcional/wave49_attestation.py",
        "src/geometria_proporcional/wave49_checker.py",
        "src/geometria_proporcional/wave49_generator.py",
        "src/geometria_proporcional/wave49_logic_oracle.py",
        "src/geometria_proporcional/wave49_oracle.py",
        "src/geometria_proporcional/wave49_oracle_reference.py",
        "src/geometria_proporcional/wave49_schema.py",
        "src/geometria_proporcional/wave50_model.py",
        "src/geometria_proporcional/wave50_neural.py",
        "src/geometria_proporcional/wave51_factored.py",
        "src/geometria_proporcional/wave52_policy.py",
        "src/geometria_proporcional/wave53_uncertainty.py",
        "src/geometria_proporcional/wave54_joint_set.py",
        "src/geometria_proporcional/wave55_policy_bridge.py",
        "src/geometria_proporcional/wave56_contextual_gate.py",
    }
    assert expected <= required


def test_atomic_escrow_publication_preserves_mode_and_payload(tmp_path: Path) -> None:
    path = tmp_path / "generation_escrow.json"
    payload = {"phase": "test", "keys": {"not-a-real-key": "00"}}
    digest = prep.atomic_write_json(path, payload, mode=0o600)
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_primary_cannot_redraw_even_with_force(tmp_path: Path) -> None:
    config = prospective_config()
    config["output_parent_relative"] = "."
    output = tmp_path / config["primary_output_name"]
    output.mkdir()
    args = SimpleNamespace(
        replay_secrets_from=None,
        recovery_secrets_from=None,
        reference_dir=None,
        force=True,
    )
    with pytest.raises(FileExistsError, match="unique primary already exists"):
        prep.validate_invocation(args, output, config, repo_root=tmp_path)


@pytest.mark.skipif(os.geteuid() != 0 or shutil.which("setpriv") is None, reason="requires root and setpriv")
def test_setpriv_contract_yields_zero_capabilities() -> None:
    raw = Path(tempfile.mkdtemp(prefix="wave56-security-", dir="/tmp"))
    raw.chmod(0o755)
    try:
        source = raw / "source"
        runner.build_phase_runtime(source)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(source)
        completed = subprocess.run(
            [
                shutil.which("setpriv") or "setpriv",
                "--reuid=65534",
                "--regid=65534",
                "--clear-groups",
                "--no-new-privs",
                sys.executable,
                "-c",
                (
                    "import json;"
                    "import _wave56_phase_worker as worker;"
                    "print(json.dumps(worker.process_security_state(),sort_keys=True))"
                ),
            ],
            cwd=source,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
    finally:
        shutil.rmtree(raw)
    state = json.loads(completed.stdout)
    assert state == {
        "effective_capabilities_hex": "0000000000000000",
        "no_new_privileges": 1,
        "supplementary_groups": [],
    }


def test_shards_use_sha256_least_significant_bit() -> None:
    tokens = np.asarray(["alpha", "beta", "gamma"])
    expected = np.asarray([
        hashlib.sha256((token + "wave56-shard").encode()).digest()[-1] & 1
        for token in tokens
    ])
    np.testing.assert_array_equal(worker.shard_assignment(tokens), expected)


def test_phase_minimums_produce_terminal_not_evaluable(tmp_path: Path) -> None:
    config = prospective_config()
    data, utilities = synthetic_data(20)
    output = tmp_path / "fit"
    output.mkdir()
    assert worker.run_fit(tmp_path, output, config, utilities, data) == "FIT_NOT_EVALUABLE"
    payload = json.loads((output / "fit_not_evaluable.json").read_text(encoding="utf-8"))
    assert set(payload["failed"]) == {"tokens", "disagreement_rows"}
    assert not (output / "fit_core.json").exists()


def test_monitor_not_evaluable_does_not_create_analysis_core(tmp_path: Path) -> None:
    config = prospective_config()
    data, utilities = synthetic_data(20)
    output = tmp_path / "monitor"
    output.mkdir()
    assert worker.run_adjudicate(
        tmp_path, output, config, utilities, data
    ) == "MONITOR_NOT_EVALUABLE"
    assert (output / "monitor_not_evaluable.json").is_file()
    assert not (output / "analysis_core.json").exists()


def test_diagnostic_outcome_has_exactly_six_conditions(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    pending.mkdir()
    conditions = {f"diagnostic_condition_{index}": True for index in range(1, 6)}
    conditions["diagnostic_condition_6_without_replay"] = True
    (pending / "analysis_core.json").write_text(
        json.dumps(
            {
                "diagnostic_pattern": {"conditions": conditions},
                "selector_stability": {"selector_sensitive": False},
            }
        ),
        encoding="utf-8",
    )
    runner.write_diagnostic_outcome(pending, True)
    outcome = json.loads((pending / "diagnostic_outcome.json").read_text(encoding="utf-8"))
    assert set(outcome["conditions"]) == {
        f"diagnostic_condition_{index}" for index in range(1, 7)
    }
    assert outcome["prospective_pattern_observed"] is True


def test_transaction_state_machine_blocks_after_not_evaluable(tmp_path: Path) -> None:
    (tmp_path / "preparation_freeze.json").write_text("{}", encoding="utf-8")
    assert runner.current_state(tmp_path) == "PREPARED"
    terminal = tmp_path / "phases/fit.not_evaluable"
    terminal.mkdir(parents=True)
    assert runner.current_state(tmp_path) == "FIT_NOT_EVALUABLE"
    with pytest.raises(RuntimeError, match="cannot run select"):
        runner.validate_transition("FIT_NOT_EVALUABLE", "select")


def test_worker_stage_rejects_future_oracle_material(tmp_path: Path) -> None:
    required = [
        "phase_request.json",
        "config.json",
        "protocol_config.json",
        "visible/train.jsonl",
        "labels/train.jsonl",
        "frozen/policy_manifest.json",
        "frozen/wave54_selection_freeze.json",
        "frozen/historical_pair_tokens.json",
        "inference/logits/seed17__train.npz",
    ]
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "phase_request.json").write_text(
        json.dumps({"phase": "fit", "split": "train", "role": "gate_fit"}), encoding="utf-8"
    )
    future = tmp_path / "labels/lockbox.jsonl"
    future.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected files"):
        worker.validate_stage(tmp_path, "fit")


def test_synthetic_pipeline_and_exact_replay(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    run_synthetic_pipeline(primary)
    run_synthetic_pipeline(replay)

    comparison = runner.compare_reference(replay, primary)
    assert comparison["all_exact"] is True
    result = replay / "phases/adjudicate.complete/result_arrays.npz"
    with np.load(result, allow_pickle=False) as arrays:
        assert arrays["shuffle_id"].shape == (5,)
        assert arrays["shuffle_score"].shape == (5, 120, 24)
        assert arrays["shuffle_actions"].shape == (5, 120, 24)
        assert arrays["shuffle_metric__regret_by_policy"].shape == (5, 120, 24)
        assert arrays["bootstrap_indices"].shape == (5000, 120)
    core = json.loads(
        (replay / "phases/adjudicate.complete/analysis_core.json").read_text(encoding="utf-8")
    )
    assert set(core["arms"]) == {
        "hard_set_policy",
        "pure_joint_full",
        "scalar_advantage_gate",
        "contextual_value_gate",
        "advantage_only_value_gate",
        "contextual_shuffled_gain",
        "oracle_positive_gain",
    }
    assert core["contextual_shuffled_gain"]["replicate_statuses"] == ["PASS"] * 5


def test_coordinator_cannot_draw_fresh_keys_or_load_truth() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "import secrets" not in source
    assert "generate_benchmark" not in source
    assert "compute_oracle_splits" not in source
    assert "load_labeled_records" not in source
