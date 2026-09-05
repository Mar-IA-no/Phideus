from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys
import shutil
import subprocess
import time
from types import SimpleNamespace

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
    validate_pre_draw_config,
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


def test_shared_preparer_dispatches_wave59_and_accepts_frozen_config() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert preparer.preparation_phase_prefix(config) == "wave59"
    preparer.validate_prospective_config(config)
    config["status"] = "IMPLEMENTATION_PRE_DRAW"
    with pytest.raises(RuntimeError, match="not frozen"):
        preparer.validate_prospective_config(config)


def test_frozen_status_alone_cannot_bypass_implementation_binding() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["implementation_binding"] = {
        "status": "PENDING_IMPLEMENTATION_AUDIT",
        "commit": None,
        "audit_path": None,
        "audit_sha256": None,
    }
    with pytest.raises(RuntimeError, match="implementation audit"):
        validate_pre_draw_config(config)
    with pytest.raises(RuntimeError, match="implementation audit"):
        preparer.validate_prospective_config(config)


def test_execution_source_must_remain_the_clean_head_blob(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "wave59@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Wave 59 Test"], cwd=repo, check=True
    )
    source = repo / "source.json"
    source.write_text('{"value":1}\n', encoding="utf-8")
    subprocess.run(["git", "add", "source.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=repo, check=True)
    runner.require_clean_head_source(repo, "source.json", source)
    source.write_text('{"value":2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="is dirty"):
        runner.require_clean_head_source(repo, "source.json", source)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_npz(path: Path, arrays: dict[str, np.ndarray], *, secret: bool = False) -> None:
    np.savez(path, **arrays)
    if secret:
        path.chmod(0o600)


def write_frozen_test_preparation_authority(root: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    bundle_names = (
        "gate_fit_bundle.npz",
        "gate_select_truth_bundle.npz",
        "gate_select_inference_bundle.npz",
        "sealed_monitor_truth_bundle.npz",
        "sealed_monitor_inference_bundle.npz",
    )
    runner.write_json(
        root / "source_bindings.json", config["source_binding"], mode=0o444
    )
    runner.write_json(
        root / "preparation_freeze.json",
        {
            "schema_version": config["schema_version"],
            "phase": "prepared-with-blind-inference-before-any-oracle",
            "config_sha256": runner.sha256_file(CONFIG),
            "prospective_config": config,
            "sources": {
                relative: runner.sha256_file(REPO_ROOT / relative)
                for relative in config["required_execution_sources"]
            },
            "source_bindings": config["source_binding"],
            "prepared_bundle_hashes": {
                f"prepared/{name}": runner.sha256_file(root / "prepared" / name)
                for name in bundle_names
            },
        },
        mode=0o444,
    )


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
    write_frozen_test_preparation_authority(root)
    return runner.execute(root, POLICY_MANIFEST, root, CONFIG)


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
    assert set(analysis["contrasts"]) >= {
        "mean_vs_hgb_proposer_only",
        "tail_vs_hgb_proposer_only",
    }
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
        historical_physical_pipeline,
    )
    assert (replay / "replay_comparison.json").is_file()
    comparison = runner.compare_runs(replay, historical_physical_pipeline)
    assert comparison["all_exact"] is True
    assert all(comparison["scientific_exact"].values())
    assert all(comparison["scientific_array_exact"].values())


def test_replay_rejects_unbound_or_different_recovery_amendments(
    historical_physical_pipeline: Path, tmp_path: Path
) -> None:
    replay = historical_physical_pipeline.parent / "replay-output"
    primary_copy = tmp_path / "primary"
    replay_copy = tmp_path / "replay"
    shutil.copytree(historical_physical_pipeline, primary_copy)
    shutil.copytree(replay, replay_copy)
    (primary_copy / "recovery_amendment.json").write_bytes(b"primary")
    (replay_copy / "recovery_amendment.json").write_bytes(b"replay")
    with pytest.raises(RuntimeError, match="metadata and amendment presence"):
        runner.compare_runs(replay_copy, primary_copy)


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


def _stage_resume_copy(
    historical_physical_pipeline: Path, destination: Path
) -> Path:
    shutil.copytree(
        historical_physical_pipeline,
        destination,
        ignore=shutil.ignore_patterns("prepared"),
    )
    shutil.copytree(
        historical_physical_pipeline.parent / "prepared",
        destination / "prepared",
    )
    for phase, relatives in runner.PHASE_PROBE_RELATIVES.items():
        journal_path = destination / "journals" / f"{phase}.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["access_receipt"]["forbidden_probes"] = [
            {
                "path_sha256": hashlib.sha256(
                    str((destination / relative).resolve()).encode("utf-8")
                ).hexdigest(),
                "denied": True,
                "error_type": "PermissionError",
            }
            for relative in relatives
        ]
        runner.write_json(journal_path, journal, mode=0o444)
    return destination


def test_identical_hash_resume_reuses_completed_journals(
    historical_physical_pipeline: Path, tmp_path: Path
) -> None:
    working = tmp_path / "canonical"
    _stage_resume_copy(historical_physical_pipeline, working)
    fit_hash = runner.sha256_file(
        working / "fit/model_state_arrays.npz"
    )
    archived = runner.archive_failed_attempt(
        working,
        RuntimeError("injected after monitor promotion"),
        run_role="primary",
        recovery_context=False,
    )
    restored = runner.restore_identical_hash_attempt(
        archived, working, CONFIG, POLICY_MANIFEST
    )
    assert not any((restored / name).exists() for name in runner.FAILURE_METADATA)
    resumed = runner.execute(restored, POLICY_MANIFEST, restored, CONFIG)
    assert runner.sha256_file(resumed / "fit/model_state_arrays.npz") == fit_hash
    assert json.loads((resumed / "runtime.json").read_text())["status"] == "COMPLETE"


def test_identical_hash_resume_rejects_post_failure_tamper(
    historical_physical_pipeline: Path, tmp_path: Path
) -> None:
    working = tmp_path / "canonical"
    _stage_resume_copy(historical_physical_pipeline, working)
    archived = runner.archive_failed_attempt(
        working,
        RuntimeError("injected after monitor promotion"),
        run_role="primary",
        recovery_context=False,
    )
    analysis_path = archived / "analysis.json"
    journal_path = archived / "journals/monitor_evaluate.json"
    inventory_path = archived / "failure_inventory.json"
    original_bytes = {
        path: path.read_bytes() for path in (analysis_path, journal_path, inventory_path)
    }
    original = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis = dict(original)
    analysis["scientific_decision"] = "TAMPERED_AFTER_FAILURE"
    runner.write_json(analysis_path, analysis, mode=0o444)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["output_sha256"]["analysis.json"] = runner.sha256_file(analysis_path)
    runner.write_json(journal_path, journal, mode=0o444)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    records = {row["path"]: row for row in inventory["records"]}
    for relative, path in {
        "analysis.json": analysis_path,
        "journals/monitor_evaluate.json": journal_path,
    }.items():
        records[relative]["sha256"] = runner.sha256_file(path)
        records[relative]["bytes"] = path.stat().st_size
    runner.write_json(inventory_path, inventory, mode=0o600)
    with pytest.raises(RuntimeError, match="attestation payload drifted"):
        runner.restore_identical_hash_attempt(
            archived, working, CONFIG, POLICY_MANIFEST
        )
    for path, payload in original_bytes.items():
        path.write_bytes(payload)
    runner.restore_identical_hash_attempt(
        archived, working, CONFIG, POLICY_MANIFEST
    )
    runner.execute(
        working,
        POLICY_MANIFEST,
        working,
        CONFIG,
    )


def _resign_failed_attempt(archived: Path) -> None:
    failure_path = archived / "FAILURE.json"
    inventory_path = archived / "failure_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for row in inventory["records"]:
        if row["path"] == "FAILURE.json":
            row["bytes"] = failure_path.stat().st_size
            row["sha256"] = runner.sha256_file(failure_path)
    runner.write_json(inventory_path, inventory, mode=0o600)
    attestation = runner.sign_attestation(
        {
            "schema_version": "wave59-failure-anchor-v1",
            "failure_inventory_sha256": runner.sha256_file(inventory_path),
            "failure_sha256": runner.sha256_file(failure_path),
            "archived_path": str(archived),
        },
        runner.DEFAULT_PRIVATE_KEY,
        runner.TRUSTED_PUBLIC_KEY,
    )
    runner.write_json(
        archived / "failure_attestation.json", attestation, mode=0o600
    )


@pytest.mark.parametrize(
    "target", ["attestation", "payload", "inventory", "record", "failure"]
)
def test_resume_rejects_resigned_schema_extensions(
    tmp_path: Path, target: str
) -> None:
    root = tmp_path / "canonical"
    root.mkdir()
    shutil.copyfile(CONFIG, root / "config.snapshot.json")
    runner.write_json(root / "source_bindings.json", {"bound": True})
    archived = runner.archive_failed_attempt(
        root,
        RuntimeError("schema extension"),
        run_role="primary",
        recovery_context=False,
    )
    if target == "attestation":
        path = archived / "failure_attestation.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["extension"] = "authenticated-but-forbidden"
    elif target == "payload":
        path = archived / "failure_attestation.json"
        current = json.loads(path.read_text(encoding="utf-8"))["payload"]
        current["extension"] = "authenticated-but-forbidden"
        payload = runner.sign_attestation(
            current, runner.DEFAULT_PRIVATE_KEY, runner.TRUSTED_PUBLIC_KEY
        )
    elif target == "failure":
        path = archived / "FAILURE.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["extension"] = "authenticated-but-forbidden"
    else:
        path = archived / "failure_inventory.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if target == "inventory":
            payload["extension"] = "authenticated-but-forbidden"
        else:
            payload["records"][0]["extension"] = "authenticated-but-forbidden"
    runner.write_json(path, payload, mode=0o600)
    if target in {"inventory", "record", "failure"}:
        _resign_failed_attempt(archived)
    with pytest.raises(RuntimeError, match="keys drifted"):
        runner.restore_identical_hash_attempt(archived, root, CONFIG)


def test_resume_rejects_resigned_journal_extension(
    historical_physical_pipeline: Path, tmp_path: Path
) -> None:
    working = tmp_path / "canonical"
    _stage_resume_copy(historical_physical_pipeline, working)
    archived = runner.archive_failed_attempt(
        working,
        RuntimeError("journal extension"),
        run_role="primary",
        recovery_context=False,
    )
    journal_path = archived / "journals/fit.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["extension"] = "authenticated-but-forbidden"
    runner.write_json(journal_path, journal, mode=0o444)
    inventory_path = archived / "failure_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for row in inventory["records"]:
        if row["path"] == "journals/fit.json":
            row["bytes"] = journal_path.stat().st_size
            row["sha256"] = runner.sha256_file(journal_path)
    runner.write_json(inventory_path, inventory, mode=0o600)
    _resign_failed_attempt(archived)
    with pytest.raises(RuntimeError, match="fit journal keys drifted"):
        runner.restore_identical_hash_attempt(
            archived, working, CONFIG, POLICY_MANIFEST
        )


def test_resume_rejects_resigned_empty_receipt_coverage(
    historical_physical_pipeline: Path, tmp_path: Path
) -> None:
    working = _stage_resume_copy(
        historical_physical_pipeline, tmp_path / "canonical"
    )
    archived = runner.archive_failed_attempt(
        working,
        RuntimeError("receipt coverage"),
        run_role="primary",
        recovery_context=False,
    )
    journal_path = archived / "journals/fit.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["access_receipt"]["stage_hashes"] = {}
    journal["access_receipt"]["forbidden_probes"] = []
    runner.write_json(journal_path, journal, mode=0o444)
    inventory_path = archived / "failure_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    journal_record = next(
        row for row in inventory["records"] if row["path"] == "journals/fit.json"
    )
    journal_record["bytes"] = journal_path.stat().st_size
    journal_record["sha256"] = runner.sha256_file(journal_path)
    runner.write_json(inventory_path, inventory, mode=0o600)
    _resign_failed_attempt(archived)
    with pytest.raises(RuntimeError, match="stage coverage drifted"):
        runner.restore_identical_hash_attempt(
            archived, working, CONFIG, POLICY_MANIFEST
        )


def test_resume_rejects_resigned_missing_required_phase_output(
    historical_physical_pipeline: Path, tmp_path: Path
) -> None:
    working = _stage_resume_copy(
        historical_physical_pipeline, tmp_path / "canonical"
    )
    archived = runner.archive_failed_attempt(
        working,
        RuntimeError("output coverage"),
        run_role="primary",
        recovery_context=False,
    )
    missing_path = archived / "fit/feature_schema.json"
    missing_path.unlink()
    journal_path = archived / "journals/fit.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["output_sha256"].pop("feature_schema.json")
    journal["access_receipt"]["output_inventory_before_receipt"].pop(
        "feature_schema.json"
    )
    runner.write_json(journal_path, journal, mode=0o444)
    inventory_path = archived / "failure_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["records"] = [
        row for row in inventory["records"] if row["path"] != "fit/feature_schema.json"
    ]
    journal_record = next(
        row for row in inventory["records"] if row["path"] == "journals/fit.json"
    )
    journal_record["bytes"] = journal_path.stat().st_size
    journal_record["sha256"] = runner.sha256_file(journal_path)
    runner.write_json(inventory_path, inventory, mode=0o600)
    _resign_failed_attempt(archived)
    with pytest.raises(RuntimeError, match="derived coverage drifted"):
        runner.restore_identical_hash_attempt(
            archived, working, CONFIG, POLICY_MANIFEST
        )


def test_resume_rejects_resigned_false_input_values_before_copy(
    historical_physical_pipeline: Path, tmp_path: Path
) -> None:
    working = _stage_resume_copy(
        historical_physical_pipeline, tmp_path / "canonical"
    )
    archived = runner.archive_failed_attempt(
        working,
        RuntimeError("input values"),
        run_role="primary",
        recovery_context=False,
    )
    journal_path = archived / "journals/fit.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    false_hash = "1" * 64
    journal["input_sha256"]["bundle.npz"] = false_hash
    journal["access_receipt"]["stage_hashes"]["bundle.npz"] = false_hash
    journal["access_receipt"]["stage_hashes"][
        "phase_request.json"
    ] = runner._json_payload_sha256(
        {
            "phase": "fit",
            "allowed_files": sorted(runner.PHASE_FILES["fit"]),
            "sha256": journal["input_sha256"],
        }
    )
    runner.write_json(journal_path, journal, mode=0o444)
    inventory_path = archived / "failure_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    journal_record = next(
        row for row in inventory["records"] if row["path"] == "journals/fit.json"
    )
    journal_record["bytes"] = journal_path.stat().st_size
    journal_record["sha256"] = runner.sha256_file(journal_path)
    runner.write_json(inventory_path, inventory, mode=0o600)
    _resign_failed_attempt(archived)
    with pytest.raises(RuntimeError, match="input values drifted"):
        runner.restore_identical_hash_attempt(
            archived, working, CONFIG, POLICY_MANIFEST
        )
    assert not working.exists()


def test_resume_rejects_resigned_source_binding_drift_before_copy(
    historical_physical_pipeline: Path, tmp_path: Path
) -> None:
    working = _stage_resume_copy(
        historical_physical_pipeline, tmp_path / "canonical"
    )
    archived = runner.archive_failed_attempt(
        working,
        RuntimeError("source binding drift"),
        run_role="primary",
        recovery_context=False,
    )
    binding_path = archived / "source_bindings.json"
    runner.write_json(binding_path, {"forged": True}, mode=0o444)
    binding_hash = runner.sha256_file(binding_path)
    changed_paths = {"source_bindings.json": binding_path}
    for phase in (
        "fit",
        "calibrate_scores",
        "validate",
        "monitor_apply",
        "monitor_evaluate",
    ):
        journal_path = archived / "journals" / f"{phase}.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["input_sha256"]["source_bindings.json"] = binding_hash
        journal["access_receipt"]["stage_hashes"][
            "source_bindings.json"
        ] = binding_hash
        journal["access_receipt"]["stage_hashes"][
            "phase_request.json"
        ] = runner._json_payload_sha256(
            {
                "phase": phase,
                "allowed_files": sorted(runner.PHASE_FILES[phase]),
                "sha256": journal["input_sha256"],
            }
        )
        runner.write_json(journal_path, journal, mode=0o444)
        changed_paths[f"journals/{phase}.json"] = journal_path
    inventory_path = archived / "failure_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for row in inventory["records"]:
        if row["path"] in changed_paths:
            changed = changed_paths[row["path"]]
            row["bytes"] = changed.stat().st_size
            row["sha256"] = runner.sha256_file(changed)
    runner.write_json(inventory_path, inventory, mode=0o600)
    _resign_failed_attempt(archived)
    with pytest.raises(RuntimeError, match="source bindings drifted"):
        runner.restore_identical_hash_attempt(
            archived, working, CONFIG, POLICY_MANIFEST
        )
    assert not working.exists()


def test_resume_rejects_unbound_policy_manifest_before_copy(
    historical_physical_pipeline: Path, tmp_path: Path
) -> None:
    unbound_manifest = tmp_path / "unbound_policy_manifest.json"
    unbound_manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from the bound upstream"):
        runner._expected_resumed_input_hashes(
            historical_physical_pipeline,
            "fit",
            CONFIG,
            unbound_manifest,
        )


def _convert_complete_copy_to_not_evaluable(
    root: Path, phase: str, sentinel_relative: str, success_extra: str
) -> None:
    phase_order = [
        "fit",
        "calibrate_scores",
        "validate",
        "monitor_apply",
        "monitor_evaluate",
    ]
    destination = {
        "fit": root / "fit",
        "calibrate_scores": root / "calibration",
        "monitor_apply": root / "adjudication",
    }[phase]
    shutil.rmtree(destination)
    sentinel = root / sentinel_relative
    runner.write_json(sentinel, {"status": "NOT_EVALUABLE"}, mode=0o444)
    extra = root / success_extra
    runner.write_json(extra, {"contradictory_success": True}, mode=0o444)
    terminal_index = phase_order.index(phase)
    for later_phase in phase_order[terminal_index + 1 :]:
        (root / "journals" / f"{later_phase}.json").unlink(missing_ok=True)
    for later_directory in {
        "fit": ("calibration", "validation", "adjudication"),
        "calibrate_scores": ("validation", "adjudication"),
        "monitor_apply": (),
    }[phase]:
        shutil.rmtree(root / later_directory, ignore_errors=True)
    for relative in ("analysis.json", "REPORT.md", "runtime.json", "replay_comparison.json"):
        (root / relative).unlink(missing_ok=True)
    journal_path = root / "journals" / f"{phase}.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    terminal_name = Path(sentinel_relative).name
    terminal_hash = runner.sha256_file(sentinel)
    journal["status"] = "NOT_EVALUABLE"
    journal["output_sha256"] = {terminal_name: terminal_hash}
    journal["access_receipt"]["status"] = "NOT_EVALUABLE"
    journal["access_receipt"]["output_inventory_before_receipt"] = {
        terminal_name: terminal_hash
    }
    runner.write_json(journal_path, journal, mode=0o444)


@pytest.mark.parametrize(
    ("phase", "sentinel", "success_extra"),
    [
        ("fit", "fit/fit_not_evaluable.json", "fit/feature_schema.json"),
        (
            "calibrate_scores",
            "calibration/calibration_not_evaluable.json",
            "calibration/calibration_freeze.json",
        ),
        (
            "monitor_apply",
            "adjudication/monitor_not_evaluable.json",
            "adjudication/monitor_action_freeze.json",
        ),
    ],
)
def test_resume_rejects_not_evaluable_with_same_phase_success_output(
    historical_physical_pipeline: Path,
    tmp_path: Path,
    phase: str,
    sentinel: str,
    success_extra: str,
) -> None:
    working = _stage_resume_copy(
        historical_physical_pipeline, tmp_path / "canonical"
    )
    _convert_complete_copy_to_not_evaluable(
        working, phase, sentinel, success_extra
    )
    archived = runner.archive_failed_attempt(
        working,
        RuntimeError("mutually exclusive phase outputs"),
        run_role="primary",
        recovery_context=False,
    )
    inventory = json.loads((archived / "failure_inventory.json").read_text())
    assert success_extra in inventory["extra"]
    with pytest.raises(RuntimeError, match="extra is not empty"):
        runner.restore_identical_hash_attempt(
            archived, working, CONFIG, POLICY_MANIFEST
        )


def test_resume_rejects_complete_history_with_terminal_sentinels(
    historical_physical_pipeline: Path, tmp_path: Path
) -> None:
    working = _stage_resume_copy(
        historical_physical_pipeline, tmp_path / "canonical"
    )
    sentinels = {
        "fit/fit_not_evaluable.json",
        "calibration/calibration_not_evaluable.json",
        "adjudication/monitor_not_evaluable.json",
    }
    for relative in sentinels:
        runner.write_json(
            working / relative, {"status": "NOT_EVALUABLE"}, mode=0o444
        )
    archived = runner.archive_failed_attempt(
        working,
        RuntimeError("complete plus terminal sentinels"),
        run_role="primary",
        recovery_context=False,
    )
    inventory = json.loads((archived / "failure_inventory.json").read_text())
    assert sentinels.issubset(inventory["extra"])
    with pytest.raises(RuntimeError, match="extra is not empty"):
        runner.restore_identical_hash_attempt(
            archived, working, CONFIG, POLICY_MANIFEST
        )


def test_resume_rejects_unsigned_nested_failure_metadata(tmp_path: Path) -> None:
    root = tmp_path / "canonical"
    root.mkdir()
    shutil.copyfile(CONFIG, root / "config.snapshot.json")
    runner.write_json(root / "source_bindings.json", {"bound": True})
    archived = runner.archive_failed_attempt(
        root,
        RuntimeError("nested metadata"),
        run_role="primary",
        recovery_context=False,
    )
    runner.write_json(
        archived / "nested/failure_attestation.json", {"unsigned": True}
    )
    with pytest.raises(RuntimeError, match="inventory coverage drifted"):
        runner.restore_identical_hash_attempt(archived, root, CONFIG)


def test_resume_rejects_resigned_semantic_value_drift(tmp_path: Path) -> None:
    root = tmp_path / "canonical"
    root.mkdir()
    shutil.copyfile(CONFIG, root / "config.snapshot.json")
    runner.write_json(root / "source_bindings.json", {"bound": True})
    archived = runner.archive_failed_attempt(
        root,
        RuntimeError("semantic drift"),
        run_role="primary",
        recovery_context=False,
    )
    inventory_path = archived / "failure_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    original_class = inventory["records"][0]["class"]
    inventory["records"][0]["class"] = "operational_semantic"
    runner.write_json(inventory_path, inventory, mode=0o600)
    _resign_failed_attempt(archived)
    with pytest.raises(RuntimeError, match="class drifted"):
        runner.restore_identical_hash_attempt(archived, root, CONFIG)

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["records"][0]["class"] = original_class
    runner.write_json(inventory_path, inventory, mode=0o600)
    failure_path = archived / "FAILURE.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    failure["last_state"] = "COMPLETE"
    runner.write_json(failure_path, failure, mode=0o600)
    _resign_failed_attempt(archived)
    with pytest.raises(RuntimeError, match="durable journals differ"):
        runner.restore_identical_hash_attempt(archived, root, CONFIG)


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


def test_validate_rejects_policy_arrays_mutated_after_freeze(
    historical_physical_pipeline: Path, tmp_path: Path
) -> None:
    mutated = load_npz(
        historical_physical_pipeline / "calibration/validation_policy_arrays.npz"
    )
    key = next(name for name in sorted(mutated) if name.startswith("actions__"))
    mutated[key] = np.asarray(mutated[key]).copy()
    mutated[key].flat[0] = (int(mutated[key].flat[0]) + 1) % 24
    mutated_path = tmp_path / "validation_policy_arrays.npz"
    save_npz(mutated_path, mutated)
    utilities = tmp_path / "utilities.npy"
    np.save(utilities, runner.load_utilities(POLICY_MANIFEST))
    with pytest.raises(RuntimeError, match="calibration_freeze.json hash mismatch"):
        runner._run_phase(
            tmp_path / "validate-run",
            "validate",
            {
                "config.json": CONFIG,
                "source_bindings.json": historical_physical_pipeline
                / "source_bindings.json",
                "preparation_freeze.json": historical_physical_pipeline
                / "preparation_freeze.json",
                "inference_bundle.npz": historical_physical_pipeline.parent
                / "prepared/gate_select_inference_bundle.npz",
                "truth_bundle.npz": historical_physical_pipeline.parent
                / "prepared/gate_select_truth_bundle.npz",
                "validation_scores.npz": historical_physical_pipeline
                / "calibration/validation_scores.npz",
                "validation_policy_arrays.npz": mutated_path,
                "calibration_freeze.json": historical_physical_pipeline
                / "calibration/calibration_freeze.json",
                "utilities.npy": utilities,
            },
            [],
        )


def test_monitor_evaluate_rejects_actions_mutated_after_freeze(
    historical_physical_pipeline: Path, tmp_path: Path
) -> None:
    mutated = load_npz(
        historical_physical_pipeline / "adjudication/monitor_policy_arrays.npz"
    )
    key = next(name for name in sorted(mutated) if name.startswith("actions__"))
    mutated[key] = np.asarray(mutated[key]).copy()
    mutated[key].flat[0] = (int(mutated[key].flat[0]) + 1) % 24
    mutated_path = tmp_path / "monitor_policy_arrays.npz"
    save_npz(mutated_path, mutated)
    utilities = tmp_path / "utilities.npy"
    np.save(utilities, runner.load_utilities(POLICY_MANIFEST))
    with pytest.raises(RuntimeError, match="monitor_action_freeze.json hash mismatch"):
        runner._run_phase(
            tmp_path / "monitor-run",
            "monitor_evaluate",
            {
                "config.json": CONFIG,
                "source_bindings.json": historical_physical_pipeline
                / "source_bindings.json",
                "preparation_freeze.json": historical_physical_pipeline
                / "preparation_freeze.json",
                "truth_bundle.npz": historical_physical_pipeline.parent
                / "prepared/sealed_monitor_truth_bundle.npz",
                "monitor_policy_arrays.npz": mutated_path,
                "monitor_action_freeze.json": historical_physical_pipeline
                / "adjudication/monitor_action_freeze.json",
                "utilities.npy": utilities,
            },
            [],
        )


def test_worker_wall_time_and_rss_budgets_are_enforced(
    historical_physical_pipeline: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utilities = tmp_path / "utilities.npy"
    np.save(utilities, runner.load_utilities(POLICY_MANIFEST))
    inputs = {
        "config.json": CONFIG,
        "source_bindings.json": historical_physical_pipeline / "source_bindings.json",
        "preparation_freeze.json": historical_physical_pipeline
        / "preparation_freeze.json",
        "bundle.npz": historical_physical_pipeline.parent
        / "prepared/gate_fit_bundle.npz",
        "utilities.npy": utilities,
    }
    with pytest.raises(RuntimeError, match="wall-time budget"):
        runner._run_phase(
            tmp_path / "deadline-run",
            "fit",
            inputs,
            [],
            deadline=time.monotonic() - 1.0,
        )
    monkeypatch.setattr(runner, "_process_rss_bytes", lambda pid: 9 * 1024**3)
    with pytest.raises(RuntimeError, match="RSS budget"):
        runner._run_phase(tmp_path / "rss-run", "fit", inputs, [])


def test_preparation_coordinator_hides_cuda_and_enforces_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="can see CUDA"):
        with preparer.wave59_coordinator_budget(config):
            pass
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    config["runtime_budget"]["max_seconds_per_run"] = 0.01
    with pytest.raises(RuntimeError, match="wall-time budget"):
        with preparer.wave59_coordinator_budget(config):
            time.sleep(0.1)


def test_analytical_coordinator_enforces_budget_during_manifest_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = tmp_path / "canonical"
    (output / "benchmark").mkdir(parents=True)
    runner.write_json(output / "runtime.json", {"status": "COMPLETE"})

    def slow_manifest(*args: object, **kwargs: object) -> None:
        time.sleep(0.4)

    monkeypatch.setattr(runner, "write_artifact_manifest", slow_manifest)
    with pytest.raises(RuntimeError, match="wall-time budget"):
        with runner.analytical_coordinator_budget(config, 0.3) as state:
            runner._finalize_accounted_runtime(
                output,
                config,
                state,
                0.0,
                None,
                run_role="primary",
            )


def test_analytical_coordinator_enforces_rss_during_manifest_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = tmp_path / "canonical"
    (output / "benchmark").mkdir(parents=True)
    runner.write_json(output / "runtime.json", {"status": "COMPLETE"})
    observed_rss = {"value": 1}
    monkeypatch.setattr(
        runner, "_coordinator_rss_bytes", lambda: observed_rss["value"]
    )

    def high_rss_manifest(*args: object, **kwargs: object) -> None:
        observed_rss["value"] = 9 * 1024**3
        time.sleep(0.15)

    monkeypatch.setattr(runner, "write_artifact_manifest", high_rss_manifest)
    with pytest.raises(RuntimeError, match="RSS budget"):
        with runner.analytical_coordinator_budget(config, 1.0) as state:
            runner._finalize_accounted_runtime(
                output,
                config,
                state,
                0.0,
                None,
                run_role="primary",
            )


def test_runner_overrides_inherited_thread_caps_before_import() -> None:
    script = f"""
import json
import os
import sys
sys.path.insert(0, {str(SRC)!r})
sys.path.insert(0, {str(EXPERIMENTS)!r})
import run_wave59_hgb_guard_bracket as runner
from threadpoolctl import threadpool_info
print(json.dumps({{
    "cuda": os.environ["CUDA_VISIBLE_DEVICES"],
    "caps": {{name: os.environ[name] for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    )}},
    "threads": [row["num_threads"] for row in threadpool_info()],
}}))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "17",
            "OPENBLAS_NUM_THREADS": "17",
            "MKL_NUM_THREADS": "17",
            "NUMEXPR_NUM_THREADS": "17",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(completed.stdout)
    assert observed["cuda"] == ""
    assert set(observed["caps"].values()) == {"4"}
    assert observed["threads"]
    assert max(observed["threads"]) <= 4


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
        elif relative == "runtime.json":
            runner.write_json(path, {"status": "COMPLETE"})
        elif relative != "preparation_freeze.json":
            path.write_bytes(relative.encode("utf-8"))
    amendment = root / "recovery_amendment.json"
    runner.write_json(
        root / "preparation_freeze.json",
        {
            "schema_version": "test-preparation-v1",
            **(
                {
                    "recovery_provenance": {
                        "amendment_sha256": runner.sha256_file(amendment)
                    }
                }
                if recovery_context
                else {}
            ),
        },
    )
    manifest = runner.write_artifact_manifest(root, run_role=run_role)
    assert manifest["coverage"]["missing"] == []
    assert manifest["coverage"]["extra"] == []
    assert manifest["recovery_context"] is recovery_context
    assert (root / "artifact_manifest.json").is_file()
    (root / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="extra"):
        runner.write_artifact_manifest(root, run_role=run_role)


@pytest.mark.parametrize(
    ("terminal_phase", "terminal_path"),
    [
        ("fit", "fit/fit_not_evaluable.json"),
        ("calibrate_scores", "calibration/calibration_not_evaluable.json"),
        ("monitor_apply", "adjudication/monitor_not_evaluable.json"),
    ],
)
def test_canonical_not_evaluable_manifests_are_explicit_and_closed(
    tmp_path: Path, terminal_phase: str, terminal_path: str
) -> None:
    root = tmp_path / terminal_phase
    root.mkdir()
    classes = runner._artifact_classes(
        root, run_role="primary", recovery_context=False
    )
    classes["scientific_exact"] = sorted(
        set(classes["scientific_exact"])
        | {
            "fit/fit_not_evaluable.json",
            "calibration/calibration_not_evaluable.json",
            "adjudication/monitor_not_evaluable.json",
        }
    )
    terminal_classes = runner._terminal_artifact_classes(classes, terminal_phase)
    expected = {path for paths in terminal_classes.values() for path in paths}
    for relative in sorted(expected - {"artifact_manifest.json"}):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "config.snapshot.json":
            shutil.copyfile(CONFIG, path)
        elif relative == "preparation_freeze.json":
            runner.write_json(path, {"schema_version": "test-preparation-v1"})
        elif relative == "runtime.json":
            runner.write_json(path, {"status": "NOT_EVALUABLE"})
        elif relative.startswith("journals/"):
            phase = Path(relative).stem
            status = "NOT_EVALUABLE" if phase == terminal_phase else "COMPLETE"
            runner.write_json(
                path,
                {
                    "phase": phase,
                    "status": status,
                    "output_sha256": {},
                    "maximum_truth_materialized": "test",
                },
            )
        else:
            path.write_bytes(relative.encode("utf-8"))
    runner.write_json(root / terminal_path, {"status": "NOT_EVALUABLE"})
    manifest = runner.write_artifact_manifest(root, run_role="primary")
    assert manifest["terminal_status"] == "NOT_EVALUABLE"
    assert manifest["coverage"]["extra"] == []
    success_extra = {
        "fit": "fit/feature_schema.json",
        "calibrate_scores": "calibration/validation_scores.npz",
        "monitor_apply": "adjudication/monitor_scores.npz",
    }[terminal_phase]
    extra = root / success_extra
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"forbidden-success-output")
    with pytest.raises(RuntimeError, match="extra"):
        runner.write_artifact_manifest(root, run_role="primary")


def test_canonical_complete_restore_rebuilds_closed_manifest(tmp_path: Path) -> None:
    root = tmp_path / "canonical"
    root.mkdir()
    classes = runner._artifact_classes(
        root, run_role="primary", recovery_context=False
    )
    expected = {relative for paths in classes.values() for relative in paths}
    journal_paths = {relative for relative in expected if relative.startswith("journals/")}
    for relative in sorted(expected - journal_paths - {"artifact_manifest.json"}):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "config.snapshot.json":
            shutil.copyfile(CONFIG, path)
        elif relative == "preparation_freeze.json":
            runner.write_json(path, {"schema_version": "test-preparation-v1"})
        elif relative == "runtime.json":
            runner.write_json(path, {"status": "COMPLETE"})
        else:
            path.write_bytes(relative.encode("utf-8"))
    write_frozen_test_preparation_authority(root)
    journals = root / "journals"
    journals.mkdir(exist_ok=True)
    runner.write_json(
        journals / "prepare.json",
        {
            "schema_version": "wave59-phase-journal-v1",
            "phase": "prepare",
            "status": "PREPARED",
            "execution_mode": "fresh",
            "preparation_freeze_sha256": runner.sha256_file(
                root / "preparation_freeze.json"
            ),
            "prepared_bundle_hashes": {
                relative: runner.sha256_file(root / relative)
                for relative in (
                    "prepared/gate_fit_bundle.npz",
                    "prepared/gate_select_truth_bundle.npz",
                    "prepared/gate_select_inference_bundle.npz",
                    "prepared/sealed_monitor_truth_bundle.npz",
                    "prepared/sealed_monitor_inference_bundle.npz",
                )
            },
            "maximum_truth_materialized": "prepared_all_splits_root_only",
            "next_state": "PREPARED",
        },
    )
    statuses = {
        "fit": "FIT_COMPLETE",
        "calibrate_scores": "CALIBRATION_FROZEN",
        "validate": "VALIDATION_COMPLETE",
        "monitor_apply": "MONITOR_ACTIONS_FROZEN",
        "monitor_evaluate": "COMPLETE",
    }
    truths = {
        "fit": "train",
        "calibrate_scores": "train",
        "validate": "validation",
        "monitor_apply": "validation",
        "monitor_evaluate": "monitor",
    }
    inputs = {
        phase: set(files) - {"phase_request.json"}
        for phase, files in runner.PHASE_FILES.items()
    }
    destinations = {
        "fit": root / "fit",
        "calibrate_scores": root / "calibration",
        "validate": root / "validation",
        "monitor_apply": root / "adjudication",
    }
    for phase, status in statuses.items():
        input_hashes = runner._expected_resumed_input_hashes(
            root, phase, CONFIG, POLICY_MANIFEST
        )
        stage_files = set(inputs[phase]) | {"phase_request.json"}
        stage_hashes = dict(input_hashes)
        stage_hashes["phase_request.json"] = runner._json_payload_sha256(
            {
                "phase": phase,
                "allowed_files": sorted(stage_files),
                "sha256": input_hashes,
            }
        )
        forbidden_probes = [
            {
                "path_sha256": hashlib.sha256(
                    str((root / relative).resolve()).encode("utf-8")
                ).hexdigest(),
                "denied": True,
                "error_type": "PermissionError",
            }
            for relative in runner.PHASE_PROBE_RELATIVES[phase]
        ]
        if phase == "monitor_evaluate":
            output_hashes = {
                "analysis.json": runner.sha256_file(root / "analysis.json"),
                "analysis_arrays.npz": runner.sha256_file(
                    root / "adjudication/analysis_arrays.npz"
                ),
                "bootstrap_indices.npz": runner.sha256_file(
                    root / "adjudication/bootstrap_indices.npz"
                ),
            }
        else:
            phase_number = {
                "fit": 1,
                "calibrate_scores": 2,
                "validate": 3,
                "monitor_apply": 4,
            }[phase]
            destination = destinations[phase]
            output_hashes = {
                str(path.relative_to(destination)): runner.sha256_file(path)
                for path in destination.rglob("*")
                if path.is_file()
                and runner._artifact_phase(str(path.relative_to(root)))
                == phase_number
            }
        runner.write_json(
            journals / f"{phase}.json",
            {
                "schema_version": "wave59-phase-journal-v1",
                "phase": phase,
                "status": status,
                "input_sha256": input_hashes,
                "output_sha256": output_hashes,
                "access_receipt": {
                    "phase": phase,
                    "status": status,
                    "effective_uid": 65534,
                    "effective_gid": 65534,
                    "process_security": {
                        "effective_capabilities_hex": "0000000000000000",
                        "no_new_privileges": 1,
                        "supplementary_groups": [],
                    },
                    "threadpools": [
                        {
                            "filepath": "/test/libgomp.so",
                            "internal_api": "openmp",
                            "num_threads": 4,
                            "prefix": "libgomp",
                            "user_api": "openmp",
                            "version": None,
                        }
                    ],
                    "stage_hashes": stage_hashes,
                    "forbidden_probes": forbidden_probes,
                    "output_inventory_before_receipt": output_hashes,
                    "benchmark_root_received": False,
                },
                "duration_seconds": 0.0,
                "max_rss_bytes": 0,
                "maximum_truth_materialized": truths[phase],
            },
        )
    runner.write_artifact_manifest(root, run_role="primary")
    archived = runner.archive_failed_attempt(
        root,
        RuntimeError("injected after canonical completion"),
        run_role="primary",
        recovery_context=False,
    )
    restored = runner.restore_identical_hash_attempt(
        archived, root, CONFIG, POLICY_MANIFEST
    )
    assert not any((restored / name).exists() for name in runner.FAILURE_METADATA)
    manifest = runner.write_artifact_manifest(restored, run_role="primary")
    assert manifest["coverage"]["missing"] == []
    assert manifest["coverage"]["extra"] == []


def test_failed_attempt_is_archived_with_redacted_error_and_closed_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wave59-primary"
    (root / "journals").mkdir(parents=True)
    shutil.copyfile(CONFIG, root / "config.snapshot.json")
    runner.write_json(root / "source_bindings.json", {"bound": True})
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
    assert inventory["missing_required_through_last_journal"] == []
    assert inventory["failure_records"] == [
        "FAILURE.json",
        "failure_inventory.json",
        "failure_attestation.json",
    ]
    assert (archived / "failure_attestation.json").is_file()


def test_failed_prepare_inventory_exposes_missing_required_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wave59-missing-prepare"
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
    runner.write_json(root / "fit/feature_schema.json", {"future": True})
    archived = runner.archive_failed_attempt(
        root, RuntimeError("missing"), run_role="primary", recovery_context=False
    )
    inventory = json.loads((archived / "failure_inventory.json").read_text())
    assert "pre_generation_freeze.json" in inventory[
        "missing_required_through_last_journal"
    ]
    assert "fit/feature_schema.json" in inventory["extra"]
    with pytest.raises(RuntimeError, match="missing_required"):
        runner._validate_failure_inventory(archived)


def test_wave59_preparer_uses_redacted_failure_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "fresh"
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("sensitive preparation diagnostic")

    monkeypatch.setattr(preparer, "execute_preparation", fail)
    with pytest.raises(RuntimeError, match="sensitive preparation diagnostic"):
        preparer.run_preparation_transaction(
            SimpleNamespace(attestation_private_key=runner.DEFAULT_PRIVATE_KEY),
            output,
            CONFIG,
            config,
            "fresh",
            {},
            None,
            force=False,
        )
    [archived] = list(tmp_path.glob("fresh.failed_*"))
    failure = json.loads((archived / "FAILURE.json").read_text())
    assert failure["schema_version"] == "wave59-failed-attempt-v1"
    assert "sensitive preparation diagnostic" not in json.dumps(failure)
    assert (archived / "failure_inventory.json").is_file()
    assert (archived / "failure_attestation.json").is_file()
    runner._validate_failure_inventory(archived)
