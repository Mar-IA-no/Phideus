"""Boundary tests for the Wave 50 prospective runner recovery path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.geometria_proporcional import run_wave50_prospective as runner
from geometria_proporcional.wave49_schema import sha256_file


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _recovery_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "authorized-failed-attempt"
    config = tmp_path / "prospective.json"
    authorization = tmp_path / "recovery.json"
    config.write_text("{}", encoding="utf-8")
    root.mkdir()
    (root / "prospective_config.json").write_text("{}", encoding="utf-8")
    (root / "benchmark/visible").mkdir(parents=True)
    (root / "benchmark/visible/lockbox.jsonl").write_text("lockbox\n", encoding="utf-8")
    for relative in (
        "benchmark/manifest.json",
        "training_manifest.json",
        "neural_prediction_manifest.json",
        "benchmark/prediction_manifest.json",
    ):
        _write_json(root / relative, {})
    shutil_source = runner.REPO_ROOT / "src/geometria_proporcional/__init__.py"
    source_commit = "a" * 40
    _write_json(
        root / "source_snapshot_manifest.json",
        {
            "git_commit": source_commit,
            "files": {
                "source_snapshots/preexecution/src/geometria_proporcional/__init__.py": {
                    "sha256": sha256_file(shutil_source),
                    "bytes": shutil_source.stat().st_size,
                }
            },
        },
    )
    failure = {
        "status": "TECHNICAL_FAILURE_BEFORE_CANONICAL_ADJUDICATION",
        "frozen_predictions_completed": True,
        "architecture_or_protocol_changed": False,
        "source_commit": source_commit,
    }
    _write_json(root / "technical_failure.json", failure)
    identity = {
        "status": "TECHNICAL_RECOVERY_AUTHORIZED_AFTER_CODE_AUDIT",
        "failed_attempt_directory": root.name,
        "source_commit": source_commit,
        "allowed_source_deltas": {},
    }
    for field, relative in {
        "technical_failure_sha256": "technical_failure.json",
        "lockbox_visible_sha256": "benchmark/visible/lockbox.jsonl",
        "benchmark_manifest_sha256": "benchmark/manifest.json",
        "training_manifest_sha256": "training_manifest.json",
        "neural_prediction_manifest_sha256": "neural_prediction_manifest.json",
        "classical_prediction_manifest_sha256": "benchmark/prediction_manifest.json",
    }.items():
        identity[field] = sha256_file(root / relative)
    _write_json(authorization, identity)
    monkeypatch.setattr(runner, "validate_manifest", lambda *_: None)
    monkeypatch.setattr(runner, "validate_prediction_manifest", lambda *_: None)
    monkeypatch.setattr(runner, "validate_frozen_files", lambda *_: None)
    monkeypatch.setattr(runner, "_replay_keys", lambda *_: (b"a", b"b", b"c"))
    return root, config, authorization


def test_recovery_source_is_bound_to_authorized_identity(tmp_path, monkeypatch):
    root, config, authorization = _recovery_fixture(tmp_path, monkeypatch)
    result = runner._validate_recovery_source(root, config, authorization)
    assert result["architecture_or_protocol_changed"] is False

    (root / "benchmark/visible/lockbox.jsonl").write_text("different\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lockbox_visible_sha256"):
        runner._validate_recovery_source(root, config, authorization)


def test_recovery_source_rejects_a_different_attempt(tmp_path, monkeypatch):
    root, config, authorization = _recovery_fixture(tmp_path, monkeypatch)
    alternate = root.with_name("different-failed-attempt")
    root.rename(alternate)
    with pytest.raises(RuntimeError, match="not the authorized failed attempt"):
        runner._validate_recovery_source(alternate, config, authorization)
