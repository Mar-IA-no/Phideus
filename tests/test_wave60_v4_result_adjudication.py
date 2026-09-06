from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments/geometria_proporcional"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import adjudicate_wave60_v4_result as adjudicator


def _file_identity(root: Path) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISREG(metadata.st_mode):
            result[path.relative_to(root).as_posix()] = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_uid,
                metadata.st_gid,
                metadata.st_mode & 0o777,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
    return result


@pytest.fixture(scope="module")
def real_evidence() -> dict:
    return adjudicator.adjudicate_attempt()


def _candidate_payload(evidence: dict) -> dict:
    chain = adjudicator.validate_static_authorities()
    implementation_commit = "a" * 40
    files = {
        relative: str(index + 1) * 64
        for index, relative in enumerate(adjudicator.IMPLEMENTATION_PATHS)
    }
    r521_json = adjudicator._audit_payload(
        "R521",
        "R509_REPLAY_NORMALIZATION_RESOLUTION_IMPLEMENTATION",
        {"implementation_commit": implementation_commit, "files": files},
        "PASS",
        {"high": 0, "medium": 0, "low": 0},
    )
    chain.update(
        {
            "r520_implementation": {
                "commit": implementation_commit,
                "files": files,
            },
            "r521_implementation_audit": {
                "commit": "b" * 40,
                "path": adjudicator.IMPLEMENTATION_AUDIT_PATH,
                "sha256": "c" * 64,
                "authority_json": r521_json,
            },
        }
    )
    return {
        "schema_version": "wave60-v4-replay-normalization-correction-v1",
        "artifact_status": "CANDIDATE_PENDING_R523_AUDIT",
        "activation_condition": {
            "required_audit_id": "R523",
            "required_audit_path": adjudicator.ARTIFACT_AUDIT_PATH,
            "required_scope": "WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION",
            "required_verdict": "PASS",
            "required_findings": {"high": 0, "medium": 0, "low": 0},
            "authority_effect": "ACTIVATES_CONDITIONAL_CORRECTED_VIEW",
        },
        "authority_chain": chain,
        "attempt_binding": deepcopy(evidence["attempt_binding"]),
        "config_binding": deepcopy(adjudicator.CONFIG_BINDING),
        "source_bindings": deepcopy(adjudicator.SOURCE_BINDINGS),
        "original_observation": deepcopy(evidence["original_observation"]),
        "normalized_evidence": deepcopy(evidence["normalized_evidence"]),
        "conditional_corrected_view": deepcopy(
            evidence["conditional_corrected_view"]
        ),
        "metrics_binding": deepcopy(evidence["metrics_binding"]),
        "limitations": {
            "original": list(adjudicator.EXPECTED_LIMITATIONS),
            "if_activated": list(adjudicator.ACTIVATED_LIMITATIONS),
            "replaced": {
                "from": "replay_exact_pending_pair_finalize",
                "to": "replay_exact_adjudicated_by_r509_findings_resolution_chain",
            },
        },
        "scientific_decision": None,
        "decision_authority": "user",
        "architecture_promoted": False,
        "gpu_used_or_queried": False,
    }


def test_real_attempt_is_read_only_and_uses_historical_commit() -> None:
    before = _file_identity(adjudicator.ATTEMPT)
    original_git_commit = adjudicator.wave60_runner.git_commit
    evidence = adjudicator.adjudicate_attempt()
    after = _file_identity(adjudicator.ATTEMPT)
    assert before == after
    assert adjudicator.wave60_runner.git_commit is original_git_commit
    assert evidence["attempt_binding"]["physical_inventory"] == {
        "primary_files": 65,
        "replay_files": 66,
        "pair_files": 10,
        "total_files": 141,
        "manifested_files": 138,
        "self_manifest_files": 3,
        "regular_files": 141,
        "nlink_one_files": 141,
        "unique_device_inode_pairs": 141,
        "metadata_and_hashes_match": True,
    }
    assert evidence["original_observation"]["patterns"] == {
        "incompatibility": False,
        "harm": False,
    }
    assert evidence["conditional_corrected_view"]["patterns"] == {
        "incompatibility": False,
        "harm": False,
    }


def test_static_git_config_and_source_authorities_are_exact() -> None:
    chain = adjudicator.validate_static_authorities()
    assert set(chain) == {
        "r509_result_audit",
        "r510_base_plan",
        "r511_base_plan_audit",
        "r512_first_resolution_plan",
        "r513_first_resolution_plan_audit",
        "r514_final_resolution_plan",
        "r515_final_resolution_plan_audit",
        "r516_false_source_hash_plan",
        "r517_false_source_hash_plan_audit",
        "r518_false_attribution_resolution_plan",
        "r519_false_attribution_resolution_plan_audit",
    }
    config, sources = adjudicator.validate_config_and_sources()
    assert config == adjudicator.CONFIG_BINDING
    assert sources == adjudicator.SOURCE_BINDINGS
    module_path = "src/geometria_proporcional/wave60_frozen_policy_transport.py"
    assert sources[module_path] == (
        "46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65"
    )
    assert sources[module_path] != (
        "46e31fa1096ab4e5a0c5115fc4f16922e165a21dbe47016c42953bf345116c65"
    )


def test_unique_mismatch_rejects_any_second_failure() -> None:
    comparison = adjudicator.read_json(
        adjudicator.ATTEMPT / "pair/replay_comparison.json"
    )
    adjudicator.validate_unique_historical_mismatch(comparison)
    extra = deepcopy(comparison)
    extra["exact_json_md"]["evaluation/analysis.json"] = False
    extra["mismatches"].append("evaluation/analysis.json")
    with pytest.raises(adjudicator.AdjudicationError, match="unique R509"):
        adjudicator.validate_unique_historical_mismatch(extra)
    renamed = deepcopy(comparison)
    renamed["mismatches"] = ["operational:anything_else.json"]
    with pytest.raises(adjudicator.AdjudicationError, match="unique R509"):
        adjudicator.validate_unique_historical_mismatch(renamed)


def _receipt_fixture() -> tuple[dict, dict, dict, dict]:
    generations = {
        "primary": {"execution_mode": "recovery", "shared": {"value": 1}},
        "replay": {"execution_mode": "replay", "shared": {"value": 1}},
    }
    generation_hashes = {"primary": "1" * 64, "replay": "2" * 64}
    freeze_hashes = {"primary": "3" * 64, "replay": "3" * 64}
    preparations = {
        role: {
            "generation_receipt_sha256": generation_hashes[role],
            "preparation_freeze_sha256": freeze_hashes[role],
            "next_state": "PREPARED",
        }
        for role in ("primary", "replay")
    }
    return preparations, generations, generation_hashes, freeze_hashes


def test_receipts_validate_local_links_and_only_role_difference() -> None:
    values = _receipt_fixture()
    assert adjudicator.validate_receipt_normalization(*values) == {
        "primary": "recovery",
        "replay": "replay",
    }
    preparations, generations, hashes, freezes = values
    crossed = deepcopy(preparations)
    crossed["primary"]["generation_receipt_sha256"] = hashes["replay"]
    with pytest.raises(adjudicator.AdjudicationError, match="local generation"):
        adjudicator.validate_receipt_normalization(
            crossed, generations, hashes, freezes
        )
    broken_freeze = deepcopy(freezes)
    broken_freeze["replay"] = "4" * 64
    with pytest.raises(adjudicator.AdjudicationError, match="preparation freeze"):
        adjudicator.validate_receipt_normalization(
            preparations, generations, hashes, broken_freeze
        )
    extra = deepcopy(generations)
    extra["replay"]["extra"] = True
    with pytest.raises(adjudicator.AdjudicationError, match="execution_mode"):
        adjudicator.validate_receipt_normalization(
            preparations, extra, hashes, freezes
        )
    wrong_role = deepcopy(generations)
    wrong_role["primary"]["execution_mode"] = "primary"
    with pytest.raises(adjudicator.AdjudicationError, match="roles"):
        adjudicator.validate_receipt_normalization(
            preparations, wrong_role, hashes, freezes
        )


@pytest.mark.parametrize(
    "text",
    [
        "no authority",
        "```json\n{}\n```\n```json\n{}\n```",
        "```json\n{broken}\n```",
        "```json\n{}\n```",
    ],
)
def test_audit_parser_is_fail_closed(text: str) -> None:
    with pytest.raises(adjudicator.AdjudicationError):
        adjudicator.parse_single_audit_json(text, "synthetic audit")


def test_commit_authority_rejects_parent_status_and_pathset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit, parent, path = "a" * 40, "b" * 40, "one.md"

    def valid_git(*args: str) -> str:
        if args[0] == "rev-list":
            return f"{commit} {parent}"
        return f"A\t{path}"

    monkeypatch.setattr(adjudicator, "_git", valid_git)
    adjudicator._require_commit(commit, parent, [path], "synthetic")
    with pytest.raises(adjudicator.AdjudicationError, match="pathset"):
        adjudicator._require_commit(
            commit, parent, [path], "synthetic", statuses={path: "M"}
        )

    def wrong_parent(*args: str) -> str:
        if args[0] == "rev-list":
            return f"{commit} {'c' * 40}"
        return f"A\t{path}"

    monkeypatch.setattr(adjudicator, "_git", wrong_parent)
    with pytest.raises(adjudicator.AdjudicationError, match="parent"):
        adjudicator._require_commit(commit, parent, [path], "synthetic")


def test_physical_record_rejects_alias_special_and_metadata_drift(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    original.write_bytes(b"identity")
    os.chmod(original, 0o444)
    record = adjudicator._physical_record(original)
    expected = {
        "bytes": 8,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "mode": "0444",
        "sha256": adjudicator.file_sha256(original),
    }
    adjudicator.require_physical_match(record, expected, "original")
    os.chmod(original, 0o400)
    with pytest.raises(adjudicator.AdjudicationError, match="metadata"):
        adjudicator.require_physical_match(
            adjudicator._physical_record(original), expected, "original"
        )
    os.chmod(original, 0o444)
    if os.geteuid() == 0:
        os.chown(original, 65534, 65534)
        with pytest.raises(adjudicator.AdjudicationError, match="metadata"):
            adjudicator.require_physical_match(
                adjudicator._physical_record(original), expected, "original"
            )
        os.chown(original, os.getuid(), os.getgid())
    linked = tmp_path / "linked"
    os.link(original, linked)
    with pytest.raises(adjudicator.AdjudicationError, match="hardlink"):
        adjudicator.require_single_link(
            adjudicator._physical_record(original), "original"
        )
    linked.unlink()
    symlink = tmp_path / "symlink"
    symlink.symlink_to(original)
    with pytest.raises(adjudicator.AdjudicationError, match="aliased"):
        adjudicator._physical_record(symlink)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(adjudicator.AdjudicationError, match="non-regular"):
        adjudicator._physical_record(fifo)


def _nested_dicts(payload: dict) -> list[dict]:
    chain = payload["authority_chain"]
    return [
        payload,
        payload["activation_condition"],
        chain,
        chain["r509_result_audit"],
        chain["r509_result_audit"]["authority_json"],
        chain["r520_implementation"],
        chain["r520_implementation"]["files"],
        payload["attempt_binding"],
        payload["attempt_binding"]["target_sha256"],
        payload["attempt_binding"]["physical_inventory"],
        payload["attempt_binding"]["self_manifest_metadata"],
        payload["attempt_binding"]["self_manifest_metadata"][
            "primary/artifact_manifest.json"
        ],
        payload["config_binding"],
        payload["config_binding"]["audit"],
        payload["source_bindings"],
        payload["original_observation"],
        payload["normalized_evidence"],
        payload["normalized_evidence"]["local_generation_receipt_sha256"],
        payload["normalized_evidence"]["generation_execution_modes"],
        payload["conditional_corrected_view"],
        payload["metrics_binding"],
        payload["metrics_binding"]["r509_numeric_recomputation"],
        payload["limitations"],
        payload["limitations"]["replaced"],
    ]


def test_payload_keysets_are_closed_at_every_level(real_evidence: dict) -> None:
    payload = _candidate_payload(real_evidence)
    adjudicator.validate_correction_payload(payload)
    for index, target in enumerate(_nested_dicts(payload)):
        key = next(iter(target))
        missing = deepcopy(payload)
        del _nested_dicts(missing)[index][key]
        with pytest.raises(adjudicator.AdjudicationError):
            adjudicator.validate_correction_payload(missing)
        extra = deepcopy(payload)
        _nested_dicts(extra)[index]["unexpected_key"] = None
        with pytest.raises(adjudicator.AdjudicationError):
            adjudicator.validate_correction_payload(extra)


@pytest.mark.parametrize(
    "mutation",
    [
        "original_replay",
        "corrected_pattern",
        "normalized_count",
        "metric_count",
        "source_hash",
        "historical_path",
        "r517_target",
    ],
)
def test_payload_rejects_semantic_or_authority_drift(
    real_evidence: dict, mutation: str
) -> None:
    payload = _candidate_payload(real_evidence)
    if mutation == "original_replay":
        payload["original_observation"]["replay_exact"] = True
    elif mutation == "corrected_pattern":
        payload["conditional_corrected_view"]["patterns"]["harm"] = True
    elif mutation == "normalized_count":
        payload["normalized_evidence"]["historical_check_count"] = 35
    elif mutation == "metric_count":
        payload["metrics_binding"]["r509_numeric_recomputation"]["actions"] = 13
    elif mutation == "source_hash":
        key = next(iter(payload["source_bindings"]))
        payload["source_bindings"][key] = "0" * 64
    elif mutation == "historical_path":
        payload["authority_chain"]["r510_base_plan"]["path"] = "wrong"
    elif mutation == "r517_target":
        payload["authority_chain"]["r521_implementation_audit"][
            "authority_json"
        ]["target"]["implementation_commit"] = "d" * 40
    with pytest.raises(adjudicator.AdjudicationError):
        adjudicator.validate_correction_payload(payload)


def test_implementation_target_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    implementation, audit = "a" * 40, "b" * 40
    captured: dict = {}
    monkeypatch.setattr(adjudicator, "_require_commit", lambda *args, **kwargs: None)
    monkeypatch.setattr(adjudicator, "file_sha256", lambda path: "c" * 64)
    monkeypatch.setattr(adjudicator, "git_blob_sha256", lambda *args: "c" * 64)

    def validate(spec: dict, label: str) -> dict:
        captured.update(spec)
        return {
            key: spec[key]
            for key in ("commit", "path", "sha256", "authority_json")
        }

    monkeypatch.setattr(adjudicator, "_validate_audit", validate)
    result = adjudicator.validate_implementation_authority(
        implementation, audit, "d" * 64
    )
    assert set(result["r520_implementation"]["files"]) == set(
        adjudicator.IMPLEMENTATION_PATHS
    )
    assert captured["authority_json"]["target"] == {
        "implementation_commit": implementation,
        "files": {
            relative: "c" * 64 for relative in adjudicator.IMPLEMENTATION_PATHS
        },
    }


def test_publication_is_canonical_exclusive_and_external(
    real_evidence: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _candidate_payload(real_evidence)
    output = tmp_path / "candidate.json"
    fake_attempt = tmp_path / "attempt"
    monkeypatch.setattr(adjudicator, "OUTPUT", output)
    monkeypatch.setattr(adjudicator, "ATTEMPT", fake_attempt)
    assert adjudicator.publish_correction(payload, output) == output
    assert output.read_bytes() == adjudicator.canonical_bytes(payload)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        adjudicator.publish_correction(payload, output)

    inside = fake_attempt / "candidate.json"
    monkeypatch.setattr(adjudicator, "OUTPUT", inside)
    with pytest.raises(adjudicator.AdjudicationError, match="external"):
        adjudicator.publish_correction(payload, inside)
    assert not inside.exists()


def test_gpu_is_forced_invisible() -> None:
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
