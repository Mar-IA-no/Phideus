from __future__ import annotations

import json
from pathlib import Path
import sys
import shutil

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
EXPERIMENTS = REPO_ROOT / "experiments/geometria_proporcional"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _wave59_phase_worker as worker  # noqa: E402
import prepare_wave56_fresh as preparer  # noqa: E402
import run_wave59_hgb_guard_bracket as runner  # noqa: E402
from geometria_proporcional.wave59_hgb_guard_bracket import (  # noqa: E402
    inference_safe_view,
)


WAVE57 = (
    REPO_ROOT
    / "data/geometria_proporcional/wave57_contextual_tail_guard_fresh_v1/phases"
)
POLICY_MANIFEST = (
    REPO_ROOT
    / "data/geometria_proporcional/wave52_policy_transport_v1/policy_manifest.json"
)
CONFIG = EXPERIMENTS / "configs/wave59_fresh_hgb_guard_bracket.json"


def test_shared_preparer_dispatches_wave59_and_blocks_unfrozen_config() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert preparer.preparation_phase_prefix(config) == "wave59"
    with pytest.raises(RuntimeError, match="not frozen"):
        preparer.validate_prospective_config(config)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_npz(path: Path, arrays: dict[str, np.ndarray], *, secret: bool = False) -> None:
    np.savez(path, **arrays)
    if secret:
        path.chmod(0o600)


@pytest.fixture(scope="module")
def historical_physical_pipeline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not WAVE57.is_dir() or not POLICY_MANIFEST.is_file():
        pytest.skip("canonical Wave 57 artifacts are unavailable")
    root = tmp_path_factory.mktemp("wave59-physical")
    prepared = root / "prepared"
    prepared.mkdir()
    fit = load_npz(WAVE57 / "fit.complete/analytics.complete/gate_fit_bundle.npz")
    validation = load_npz(
        WAVE57 / "select.complete/analytics.complete/gate_select_bundle.npz"
    )
    monitor = load_npz(
        WAVE57 / "adjudicate.complete/analytics.complete/sealed_monitor_bundle.npz"
    )
    save_npz(prepared / "gate_fit_bundle.npz", fit, secret=True)
    save_npz(prepared / "gate_select_truth_bundle.npz", validation, secret=True)
    save_npz(prepared / "sealed_monitor_truth_bundle.npz", monitor, secret=True)
    save_npz(
        prepared / "gate_select_inference_bundle.npz",
        inference_safe_view(validation),
    )
    save_npz(
        prepared / "sealed_monitor_inference_bundle.npz",
        inference_safe_view(monitor),
    )
    return runner.execute(prepared, POLICY_MANIFEST, root / "output", CONFIG)


def test_physical_workers_are_unprivileged_and_truth_is_denied(
    historical_physical_pipeline: Path,
) -> None:
    for phase in ("fit", "calibrate_scores", "validate", "monitor_apply", "monitor_evaluate"):
        receipt = json.loads(
            (historical_physical_pipeline / "journals" / f"{phase}.json").read_text(
                encoding="utf-8"
            )
        )["access_receipt"]
        assert receipt["effective_uid"] == 65534
        assert receipt["effective_gid"] == 65534
        assert receipt["process_security"] == {
            "effective_capabilities_hex": "0000000000000000",
            "no_new_privileges": 1,
            "supplementary_groups": [],
        }
        assert all(row["denied"] for row in receipt["forbidden_probes"])


def test_historical_main_policies_reproduce_wave58_metrics(
    historical_physical_pipeline: Path,
) -> None:
    analysis = json.loads(
        (
            historical_physical_pipeline / "analysis.json"
        ).read_text(encoding="utf-8")
    )
    assert len(analysis["contrasts"]["factorial"]) == 36
    assert analysis["scientific_decision"] is None
    mean_regret = analysis["contrasts"]["mean_vs_hard"]["regret"]
    tail_worst = analysis["contrasts"]["tail_vs_hard"]["worst_regret"]
    assert mean_regret["mean_diff"] == pytest.approx(-0.014490286855482936)
    assert tail_worst["mean_diff"] == pytest.approx(-0.02342047930283224)
    manifest = json.loads(
        (
            historical_physical_pipeline / "fit/model_states/manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert len(manifest["models"]) == 16
    assert "ORACLE-POSITIVE-GAIN" in analysis["summaries"]
    assert set(analysis["prospective_patterns"]["harm"]["conditions"]) >= {
        "authorized_pair_tokens_at_least_25",
        "replay_exact",
    }
    validation = json.loads(
        (historical_physical_pipeline / "validation/validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(validation["shards"]) == {"0", "1"}
    for shard in validation["shards"].values():
        if shard["status"] == "PASS":
            assert len(shard["directional_deltas"]) == 7
            assert set(shard["nonidentity_vs_global"]) == {"mean", "tail"}


def test_isolated_replay_is_scientifically_exact(
    historical_physical_pipeline: Path,
) -> None:
    prepared = historical_physical_pipeline.parent / "prepared"
    replay = runner.execute(
        prepared,
        POLICY_MANIFEST,
        historical_physical_pipeline.parent / "replay-output",
        CONFIG,
    )
    comparison = runner.compare_runs(replay, historical_physical_pipeline)
    assert comparison["all_exact"] is True
    assert all(comparison["scientific_exact"].values())
    assert all(comparison["scientific_array_exact"].values())


def test_all_joblib_copies_match_portable_scores_exactly(
    historical_physical_pipeline: Path,
) -> None:
    prepared = historical_physical_pipeline.parent / "prepared"
    staged = historical_physical_pipeline / "prepared"
    shutil.copytree(prepared, staged)
    try:
        checks = runner._portable_joblib_check(historical_physical_pipeline)
    finally:
        shutil.rmtree(staged)
    assert len(checks) == 16
    assert all(checks.values())


def test_identical_hash_resume_reuses_completed_journals(
    historical_physical_pipeline: Path,
) -> None:
    prepared = historical_physical_pipeline.parent / "prepared"
    staged = historical_physical_pipeline / "prepared"
    shutil.copytree(prepared, staged)
    fit_hash = runner.sha256_file(
        historical_physical_pipeline / "fit/model_state_arrays.npz"
    )
    archived = runner.archive_failed_attempt(
        historical_physical_pipeline,
        RuntimeError("injected after monitor promotion"),
        run_role="primary",
        recovery_context=False,
    )
    restored = runner.restore_identical_hash_attempt(
        archived, historical_physical_pipeline, CONFIG
    )
    resumed = runner.execute(restored, POLICY_MANIFEST, restored, CONFIG)
    assert runner.sha256_file(resumed / "fit/model_state_arrays.npz") == fit_hash
    assert json.loads((resumed / "runtime.json").read_text())["status"] == "COMPLETE"


def test_stage_allowlist_rejects_truth_in_calibration(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in worker.PHASE_FILES["calibrate_scores"] - {"phase_request.json"}:
        (stage / name).write_bytes(b"placeholder")
    (stage / "truth_bundle.npz").write_bytes(b"forbidden")
    worker.write_json(
        stage / "phase_request.json",
        {
            "phase": "calibrate_scores",
            "allowed_files": sorted(path.name for path in stage.iterdir()),
            "sha256": {},
        },
    )
    with pytest.raises(RuntimeError, match="allowlist"):
        worker.validate_stage(stage, "calibrate_scores")


@pytest.mark.parametrize(
    ("run_role", "recovery_context"),
    [("primary", False), ("replay", False), ("primary", True), ("replay", True)],
)
def test_closed_artifact_matrix_covers_all_four_run_contexts(
    tmp_path: Path, run_role: str, recovery_context: bool
) -> None:
    root = tmp_path / f"{run_role}-{int(recovery_context)}"
    root.mkdir()
    classes = runner._artifact_classes(
        root, run_role=run_role, recovery_context=recovery_context
    )
    for relative in sorted({path for values in classes.values() for path in values}):
        if relative == "artifact_manifest.json":
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "config.snapshot.json":
            shutil.copyfile(CONFIG, path)
        else:
            path.write_bytes(relative.encode("utf-8"))
    manifest = runner.write_artifact_manifest(root, run_role=run_role)
    assert manifest["coverage"]["missing"] == []
    assert manifest["coverage"]["extra"] == []
    assert manifest["recovery_context"] is recovery_context
    assert (root / "artifact_manifest.json").is_file()
    (root / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="extra"):
        runner.write_artifact_manifest(root, run_role=run_role)


def test_failed_attempt_is_archived_with_redacted_error_and_closed_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wave59-primary"
    (root / "journals").mkdir(parents=True)
    shutil.copyfile(CONFIG, root / "config.snapshot.json")
    runner.write_json(root / "source_bindings.json", {"bound": True})
    runner.write_json(
        root / "journals/prepare.json",
        {
            "phase": "prepare",
            "status": "PREPARED",
            "maximum_truth_materialized": "prepared_all_splits_root_only",
        },
    )
    archived = runner.archive_failed_attempt(
        root,
        RuntimeError("sensitive diagnostic text"),
        run_role="primary",
        recovery_context=False,
    )
    assert not root.exists()
    failure = json.loads((archived / "FAILURE.json").read_text(encoding="utf-8"))
    assert "sensitive diagnostic text" not in json.dumps(failure)
    assert failure["error_message_sha256"]
    inventory = json.loads(
        (archived / "failure_inventory.json").read_text(encoding="utf-8")
    )
    assert inventory["unclassified"] == []
    assert inventory["failure_records"] == ["FAILURE.json", "failure_inventory.json"]
