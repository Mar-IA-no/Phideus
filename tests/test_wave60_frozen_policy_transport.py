from __future__ import annotations

from copy import deepcopy
import importlib.util
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

from geometria_proporcional.wave56_contextual_gate import FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments/geometria_proporcional"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from geometria_proporcional.wave60_frozen_policy_transport import (
    BOOTSTRAP_REPLICATES,
    CONTROL_MODELS,
    FROZEN_THRESHOLDS,
    MAIN_POLICIES,
    PLAN_COMMIT,
    PLAN_AUDIT_COMMIT,
    PLAN_AUDIT_SHA256,
    PLAN_SHA256,
    REFERENCE_ACTIONS,
    SOURCE_HASHES,
    SOURCE_LAW_RECOVERY_BINDING,
    SOURCE_LAW_RECOVERY_IMPLEMENTATION_SCOPE,
    SOURCE_LAW_RECOVERY_REQUEST_SCHEMA,
    UNUSED_MODELS,
    USED_MODELS,
    apply_transport_policies,
    evaluate_transport_actions,
    expected_policy_array_keys,
    file_sha256,
    finalize_patterns,
    frozen_policy_spec,
    project_transport_law,
    retrospective_source_verification,
    score_transport_models,
    selected_wave59_array_keys,
    validate_pre_draw_config,
    validate_frozen_policy_spec,
    validate_transport_manifest,
    wave60_bootstrap_indices,
)
from run_wave60_frozen_policy_transport import (
    IntegrityDriftError,
    PRIOR_SOURCE_AUTHORITY,
    SOURCE_ALIASES,
    canonical_source_output_path,
    array_exact,
    copy_regular,
    execute_prepared_pair,
    initialize_attempt_container,
    pair_status,
    publish_pair_failure,
    publish_source_law_authority,
    publish_source_law_recovery,
    run_worker,
    seal_root_failure,
    stage_phase_request,
    validate_new_draw_pair,
    validate_pair_failure_package,
    validate_pair_status_against_roots,
    validate_prior_source_law_failure,
    validate_recovery_implementation_lineage,
    validate_implementation_audit_authority,
    validate_source_law_recovery_request,
    validate_evaluated_root,
)
from run_wave59_hgb_guard_bracket import load_utilities
import prepare_wave56_fresh as preparer
import run_wave60_frozen_policy_transport as wave60_runner

WAVE59 = (
    REPO_ROOT
    / "data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def recovery_source_request(output: Path | None = None) -> dict:
    target = output or wave60_runner.SOURCE_AUTHORITY_DEFAULT
    return {
        "schema_version": SOURCE_LAW_RECOVERY_REQUEST_SCHEMA,
        "plan_commit": PLAN_COMMIT,
        "plan_sha256": PLAN_SHA256,
        "implementation_commit": "1" * 40,
        "implementation_audit_commit": "2" * 40,
        "implementation_audit_sha256": "3" * 64,
        "source_paths": {
            name: str(path.relative_to(REPO_ROOT))
            for name, path in SOURCE_ALIASES.items()
        },
        "source_sha256": dict(SOURCE_HASHES),
        "output_path": str(target.relative_to(REPO_ROOT)),
        "runtime_budget": {"max_seconds": 900, "max_rss_bytes": 1610612736},
        "recovery": dict(SOURCE_LAW_RECOVERY_BINDING),
    }


def invalid_preparation_hard_set_contract() -> dict:
    return {
        "authority": "wave59_config_snapshot_transitively_bound_by_source_law_v2",
        "source_authority_path": (
            "data/geometria_proporcional/"
            "wave60_frozen_policy_transport_source_law_v2"
        ),
        "source_authority_manifest_sha256": (
            "9c69745a0661994049530e917e59e0a68b99d5f15a7c0d2bae3da66f1df43dc2"
        ),
        "source_law_request_relative": "source_law_request.json",
        "source_law_request_sha256": (
            "983af4bb024f966b60b4e79fe753ebd38665747eab21b95db27ec3f0ab889a99"
        ),
        "request_alias": "wave59_config_snapshot.json",
        "source_path": (
            "data/geometria_proporcional/"
            "wave59_fresh_hgb_guard_bracket_replay_normalized_v1/"
            "config.snapshot.json"
        ),
        "source_sha256": (
            "f6edfd2106fe87c8150562d096469e29b64a108a73de2dae0d371bd689a4a9b6"
        ),
        "hard_set_tau": 0.5,
    }


def invalid_preparation_origin_authority(
    repo_root: Path, prior_relative: Path
) -> tuple[dict, dict, Path, Path]:
    prior = repo_root / prior_relative
    primary = prior / "primary"
    draw = primary / "failed_preparation"
    config = load_json(primary / "config.snapshot.json")
    manifest = load_json(draw / "benchmark/manifest.json")
    preserved_relatives = [
        preparer.ESCROW_NAME,
        preparer.FREEZE_NAME,
        "benchmark/manifest.json",
        *(f"benchmark/{relative}" for relative in manifest["files"]),
    ]
    preserved = {
        relative: file_sha256(draw / relative) for relative in preserved_relatives
    }
    config["attempt"] = {
        "version": 2,
        "container": "data/geometria_proporcional/wave60_test_attempt_v2",
        "primary": "primary",
        "replay": "replay",
        "pair": "pair",
        "recovery": {
            "prior_attempt_container": prior_relative.as_posix(),
            "amendment_path": "synthetic-amendment.json",
            "amendment_sha256": "0" * 64,
            "preserved_draw_sha256": preserved,
        },
    }
    amendment = {
        "schema_version": (
            preparer.WAVE60_INVALID_PREPARATION_RECOVERY_AMENDMENT_SCHEMA
        ),
        "status": "APPROVED",
        "recovery_kind": "INVALID_PREPARATION",
        "prior_attempt_container": prior_relative.as_posix(),
        "prior_pair_failure_sha256": file_sha256(prior / "pair/FAILURE.json"),
        "unledgered_preparation_debit": {
            "seconds": 60.0,
            "regime": "CONSERVATIVE_UNSIGNED_PREPARATION_DEBIT",
            "observed_external_wall_seconds": 48.51,
            "observed_external_record_authority": (
                "TRANSCRIPT_ONLY_NOT_SIGNED_LEDGER"
            ),
            "applied_once": True,
        },
        "escrow_origin": {
            "contract_sha256": preparer.compact_json_sha256(
                preparer.read_escrow(draw)["contract"]
            ),
            "escrow_sha256": file_sha256(draw / preparer.ESCROW_NAME),
            "pre_generation_freeze_sha256": file_sha256(
                draw / preparer.FREEZE_NAME
            ),
            "benchmark_manifest_sha256": file_sha256(
                draw / "benchmark/manifest.json"
            ),
        },
        "preserved_draw_sha256": preserved,
        "origin_inventory": preparer.physical_tree_inventory(primary),
    }
    return config, amendment, primary, draw


def valid_config() -> dict:
    config_path = (
        "experiments/geometria_proporcional/configs/"
        "wave60_frozen_policy_transport.json"
    )
    implementation_audit = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "000_wave60_implementation_audit.md"
    )
    required = [
        config_path,
        "src/geometria_proporcional/wave60_frozen_policy_transport.py",
        "experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py",
        "experiments/geometria_proporcional/_wave60_phase_worker.py",
        "experiments/geometria_proporcional/prepare_wave56_fresh.py",
        "tests/test_wave60_frozen_policy_transport.py",
        implementation_audit,
        "Biblioteca/source_law_audit.md",
    ]
    return {
        "schema_version": "wave60-frozen-policy-transport-v1",
        "status": "FROZEN_PROSPECTIVE_PROTOCOL_PRE_KEY_DRAW",
        "device": "cpu",
        "cpu_threads": 4,
        "penalty": 1.25,
        "bootstrap": {
            "replicates": 5000,
            "seed": 6007,
            "interval": [2.5, 97.5],
            "unit": "pair_token_in_T_primary",
        },
        "runtime_budget": {
            "gpu_allowed": False,
            "max_seconds_total": 900,
            "max_rss_bytes_per_process": 1610612736,
        },
        "main_policies": dict(MAIN_POLICIES),
        "source_law_authority": {
            "path": "data/geometria_proporcional/wave60_frozen_policy_transport_source_law_v2",
            "source_law_freeze_sha256": "0" * 64,
            "source_law_attestation_sha256": "1" * 64,
            "transport_law_manifest_sha256": "2" * 64,
            "transport_law_arrays_sha256": "3" * 64,
            "frozen_policy_spec_sha256": "4" * 64,
            "feature_schema_sha256": "5" * 64,
            "source_authority_manifest_sha256": "6" * 64,
            "audit_commit": "7" * 40,
            "audit_path": "Biblioteca/source_law_audit.md",
            "audit_sha256": "8" * 64,
        },
        "attempt": {
            "version": 1,
            "container": "data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v1",
            "primary": "primary",
            "replay": "replay",
            "pair": "pair",
            "recovery": None,
        },
        "implementation_binding": {
            "status": "ACCEPTED_IMPLEMENTATION_AUDIT",
            "commit": "9" * 40,
            "audit_commit": "a" * 40,
            "audit_path": implementation_audit,
            "audit_sha256": "b" * 64,
        },
        "final_audit": {
            "audit_id": "R999",
            "audit_path": (
                "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
                "999_wave60_config_final_audit.md"
            ),
        },
        "plan_binding": {
            "commit": PLAN_COMMIT,
            "sha256": PLAN_SHA256,
            "audit_commit": PLAN_AUDIT_COMMIT,
            "audit_path": (
                "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
                "463_wave60_plan_final_focal_pass.md"
            ),
            "audit_sha256": PLAN_AUDIT_SHA256,
        },
        "fresh_benchmark": {
            "protocol": "wave49-relational-benchmark-v2",
            "expected_visible_fixtures_per_split": 4992,
            "expected_eligible_pair_tokens_per_split": 768,
            "pair_token_count_basis": "eligible_unique_pair_tokens",
            "no_redraw_after_escrow": True,
            "sealed_directory_mode": "0700",
            "escrow_file_mode": "0600",
            "inference_uid": 65534,
            "inference_gid": 65534,
            "inference_user": "nobody",
            "staging_parent": "/tmp",
        },
        "physical_splits": {
            "train": "unused_train",
            "val": "unused_validation",
            "lockbox": "sealed_monitor",
        },
        "seeds": [17, 29, 43],
        "inference_batch_size": 256,
        "feature_names": list(FEATURE_NAMES),
        "source_binding": {"upstream": "frozen"},
        "required_execution_sources": required,
        "source_sha256": {relative: "c" * 64 for relative in required},
        "primary_output": "data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v1/primary",
        "primary_output_name": "primary",
        "replay_output": "data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v1/replay",
        "replay_output_name": "replay",
        "output_parent_relative": "data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v1",
    }


@pytest.fixture(scope="module")
def source_material() -> dict:
    return {
        "manifest": load_json(WAVE59 / "fit/model_states/manifest.json"),
        "arrays": load_npz(WAVE59 / "fit/model_state_arrays.npz"),
        "calibration": load_json(WAVE59 / "calibration/calibration_freeze.json"),
        "inference": load_npz(WAVE59 / "prepared/sealed_monitor_inference_bundle.npz"),
        "truth": load_npz(WAVE59 / "prepared/sealed_monitor_truth_bundle.npz"),
        "scores": load_npz(WAVE59 / "adjudication/monitor_scores.npz"),
        "policies": load_npz(WAVE59 / "adjudication/monitor_policy_arrays.npz"),
        "utilities": load_utilities(
            REPO_ROOT
            / "data/geometria_proporcional/wave52_policy_transport_v1/policy_manifest.json"
        ),
    }


@pytest.fixture(scope="module")
def verified(source_material: dict) -> dict:
    manifest, arrays, verification = retrospective_source_verification(
        source_material["manifest"],
        source_material["arrays"],
        source_material["calibration"],
        source_material["inference"],
        source_material["scores"],
        source_material["policies"],
    )
    scores = score_transport_models(manifest, arrays, source_material["inference"])
    policies = apply_transport_policies(
        source_material["inference"], scores, verification["spec"]
    )
    return {
        "manifest": manifest,
        "arrays": arrays,
        "verification": verification,
        "scores": scores,
        "policies": policies,
    }


@pytest.fixture(scope="module")
def source_authority() -> dict:
    with tempfile.TemporaryDirectory(
        prefix=".wave60-e2e-source-", dir=REPO_ROOT
    ) as raw:
        base = Path(raw)
        output = base / "authority"
        request_path = base / "request.json"
        request = {
            "schema_version": "wave60-source-law-v1",
            "plan_commit": PLAN_COMMIT,
            "plan_sha256": PLAN_SHA256,
            "implementation_commit": "1" * 40,
            "implementation_audit_commit": "2" * 40,
            "implementation_audit_sha256": "3" * 64,
            "source_paths": {name: str(path) for name, path in SOURCE_ALIASES.items()},
            "source_sha256": dict(SOURCE_HASHES),
            "output_path": str(output.relative_to(REPO_ROOT)),
            "runtime_budget": {"max_seconds": 900, "max_rss_bytes": 1610612736},
        }
        request_path.write_text(
            json.dumps(request, sort_keys=True) + "\n", encoding="utf-8"
        )
        workspace = base / "worker"
        stage = workspace / "stage"
        stage.mkdir(parents=True)
        copy_regular(request_path, stage / "source_law_request.json")
        for alias, source in SOURCE_ALIASES.items():
            copy_regular(source, stage / alias)
        worker_output, _, duration, peak = run_worker(
            workspace,
            stage,
            "verify_source_law",
            [output],
            max_seconds=900,
            max_rss=1610612736,
        )
        output.mkdir()
        for name in (
            "source_law_freeze.json",
            "transport_law_manifest.json",
            "transport_law_arrays.npz",
            "frozen_policy_spec.json",
            "feature_schema.json",
            "verify_source_law_receipt.json",
        ):
            copy_regular(worker_output / name, output / name, mode=0o444)
        copy_regular(request_path, output / "source_law_request.json", mode=0o444)
        wave60_runner.write_json(
            output / "journals/verify_source_law.json",
            {
                "schema_version": "wave60-source-law-v1",
                "phase": "verify_source_law",
                "status": "SOURCE_LAW_VERIFIED",
                "input_sha256": file_sha256(output / "source_law_request.json"),
                "duration_seconds": duration,
                "max_rss_bytes": peak,
                "truth_accessed": False,
            },
            mode=0o444,
        )
        attestation = wave60_runner.make_attestation(
            "verify_source_law",
            {
                "scope": "pre-draw-source-law",
                "request_sha256": file_sha256(output / "source_law_request.json"),
                "implementation_commit": request["implementation_commit"],
                "freeze_sha256": file_sha256(output / "source_law_freeze.json"),
                "receipt_sha256": file_sha256(
                    output / "verify_source_law_receipt.json"
                ),
                "journal_sha256": file_sha256(
                    output / "journals/verify_source_law.json"
                ),
            },
            wave60_runner.DEFAULT_PRIVATE_KEY,
        )
        wave60_runner.write_json(
            output / "source_law_attestation.json", attestation, mode=0o444
        )
        files = wave60_runner.inventory(output)
        wave60_runner.write_json(
            output / "source_authority_manifest.json",
            {
                "schema_version": "wave60-source-law-v1",
                "terminal": "SOURCE_LAW_VERIFIED",
                "files": files,
                "classes": {
                    relative: (
                        "OPERATIONAL_JOURNAL"
                        if relative.startswith("journals/")
                        or relative
                        in {
                            "verify_source_law_receipt.json",
                            "source_law_attestation.json",
                        }
                        else "SOURCE_LAW_FROZEN"
                    )
                    for relative in files
                },
                "self_reference": {
                    "path": "source_authority_manifest.json",
                    "hashes_omitted": True,
                },
            },
            mode=0o444,
        )
        published = output
        binding = {
            "source_law_freeze_sha256": file_sha256(
                published / "source_law_freeze.json"
            ),
            "source_law_attestation_sha256": file_sha256(
                published / "source_law_attestation.json"
            ),
            "transport_law_manifest_sha256": file_sha256(
                published / "transport_law_manifest.json"
            ),
            "transport_law_arrays_sha256": file_sha256(
                published / "transport_law_arrays.npz"
            ),
            "frozen_policy_spec_sha256": file_sha256(
                published / "frozen_policy_spec.json"
            ),
            "feature_schema_sha256": file_sha256(published / "feature_schema.json"),
            "source_authority_manifest_sha256": file_sha256(
                published / "source_authority_manifest.json"
            ),
        }
        original_validator = wave60_runner.validate_source_authority

        def validate_injected_test_authority(
            authority: Path, authority_binding: dict, config: dict
        ) -> None:
            injected_binding = dict(authority_binding)
            injected_binding["path"] = str(
                authority.resolve(strict=True).relative_to(wave60_runner.REPO_ROOT)
            )
            original_validator(authority, injected_binding, config)

        patcher = pytest.MonkeyPatch()
        patcher.setattr(
            wave60_runner,
            "validate_source_authority",
            validate_injected_test_authority,
        )
        try:
            yield {"path": published, "binding": binding, "request": request}
        finally:
            patcher.undo()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _repacked_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **dict(reversed(list(arrays.items()))))


def _build_prepared_primary(
    root: Path,
    config: dict,
    source_material: dict,
    *,
    recovery_amendment: Path | None = None,
) -> None:
    root.mkdir(parents=True)
    _write_json(root / "config.snapshot.json", config)
    _write_json(root / "source_bindings.json", config["source_binding"])
    benchmark = root / "benchmark"
    files = {
        "protocol_config.json": {"protocol": "wave60-test"},
        "attestations/semantic_root.json": {"root": "fresh-wave60"},
    }
    for split in ("train", "val", "lockbox"):
        path = benchmark / f"visible/{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"split":"{split}","draw":"fresh-wave60"}}\n')
        sealed = benchmark / f"sealed/{split}.jsonl"
        sealed.parent.mkdir(parents=True, exist_ok=True)
        sealed.write_text(f'{{"sealed":"{split}","draw":"fresh-wave60"}}\n')
    commitment = benchmark / "commitments/semantic.jsonl"
    commitment.parent.mkdir(parents=True, exist_ok=True)
    commitment.write_text('{"commitment":"fresh-wave60"}\n')
    for relative, payload in files.items():
        _write_json(benchmark / relative, payload)
    declared = {}
    for path in sorted(benchmark.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(benchmark))
            declared[relative] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    key_commitments = {
        "generation_secret.json": "fresh-generation",
        "identity_secret.json": "fresh-identity",
        "semantic_commitment_key.json": "fresh-semantic",
    }
    manifest = {
        "generation_key_commitment": "fresh-generation",
        "identity_key_commitment": "fresh-identity",
        "semantic_commitment_key_commitment": "fresh-semantic",
        "files": declared,
    }
    _write_json(benchmark / "manifest.json", manifest)
    _write_json(root / "generation_escrow.json", {"keys": "opaque-fresh"})
    _write_json(
        root / "pre_generation_freeze.json", {"key_commitments": key_commitments}
    )
    _write_json(root / "generation_receipt.json", {"phase": "generated"})
    for name in (
        "gate_select_inference_bundle.npz",
        "sealed_monitor_inference_bundle.npz",
    ):
        _repacked_npz(root / "prepared" / name, source_material["inference"])
    for name in (
        "gate_fit_bundle.npz",
        "gate_select_truth_bundle.npz",
        "sealed_monitor_truth_bundle.npz",
    ):
        _repacked_npz(root / "prepared" / name, source_material["truth"])
    for seed in (17, 29, 43):
        for split in ("train", "val", "lockbox"):
            _repacked_npz(
                root / f"inference/logits/seed{seed}__{split}.npz",
                {"marker": np.asarray([seed], dtype=np.int64)},
            )
    _write_json(root / "inference/access_receipt.json", {"uid": 65534})
    bundle_hashes = {
        f"prepared/{path.name}": file_sha256(path)
        for path in sorted((root / "prepared").iterdir())
    }
    _write_json(
        root / "preparation_freeze.json",
        {
            "schema_version": "wave60-frozen-policy-transport-v1",
            "git_commit": wave60_runner.git_commit(),
            "config_sha256": file_sha256(root / "config.snapshot.json"),
            "bundles_present": True,
            "fit_operations": False,
            "oracle_materialized": False,
            "authorized_labels_present": False,
            "prepared_bundle_hashes": bundle_hashes,
        },
    )
    _write_json(
        root / "preparation_receipt.json",
        {
            "phase": "wave60-stage1-preparation-complete",
            "execution_mode": "primary",
            "preparation_freeze_sha256": file_sha256(root / "preparation_freeze.json"),
            "generation_receipt_sha256": file_sha256(root / "generation_receipt.json"),
            "replay_exact": None,
            "next_state": "PREPARED",
            "timestamp_utc": "2000-01-01T00:00:00Z",
            "superseded_output": None,
            "coordinator_budget": {
                "duration_seconds": 1.0,
                "cumulative_duration_seconds": 1.0,
                "prior_elapsed_seconds": 0.0,
                "max_rss_bytes": 1,
                "max_seconds": 900.0,
                "max_seconds_total": 900.0,
                "max_rss_allowed_bytes": 1610612736,
                "cuda_visible_devices": "",
                "budget_enforced": True,
            },
        },
    )
    _write_json(root / "journals/prepare.json", {"phase": "prepare"})
    if recovery_amendment is not None:
        shutil.copyfile(
            recovery_amendment,
            root / preparer.RECOVERY_AMENDMENT_COPY_NAME,
        )
    preparer.publish_wave60_preparation_attestation(
        root, "primary", wave60_runner.DEFAULT_PRIVATE_KEY
    )


def _build_prepared_pair(attempt: Path, config: dict, source_material: dict) -> None:
    primary = attempt / "primary"
    replay = attempt / "replay"
    _build_prepared_primary(primary, config, source_material)
    shutil.copytree(primary, replay)
    receipt = load_json(replay / "preparation_receipt.json")
    receipt["execution_mode"] = "replay"
    receipt["replay_exact"] = True
    receipt["coordinator_budget"].update(
        {
            "duration_seconds": 1.0,
            "prior_elapsed_seconds": 1.0,
            "cumulative_duration_seconds": 2.0,
            "max_seconds": 899.0,
        }
    )
    _write_json(replay / "preparation_receipt.json", receipt)
    _write_json(
        replay / "preparation_replay.json",
        {"phase": "wave60-preparation-exact-replay", "all_exact": True},
    )
    preparer.publish_wave60_preparation_attestation(
        replay, "replay", wave60_runner.DEFAULT_PRIVATE_KEY
    )


def test_physical_source_hashes_and_aliases_are_exact() -> None:
    assert set(SOURCE_HASHES).issubset(SOURCE_ALIASES)
    for alias, expected in SOURCE_HASHES.items():
        assert file_sha256(SOURCE_ALIASES[alias]) == expected


def test_projection_is_closed_and_retrospectively_exact(verified: dict) -> None:
    manifest = verified["manifest"]
    assert manifest["used_models"] == list(USED_MODELS)
    assert manifest["unused_models"] == list(UNUSED_MODELS)
    assert len(manifest["array_keys"]) == 3900
    assert (
        len(
            {
                key
                for state in manifest["model_states"].values()
                for key in state["tree_keys"]
            }
        )
        == 1300
    )
    assert verified["verification"]["diagnostics"] == {
        "full_model_scores_exact": 16,
        "transport_model_scores_exact": 13,
        "selected_policy_arrays_exact": 26,
        "hard_reference_exact": True,
        "tree_keys": 1300,
        "transport_arrays": 3900,
        "score_mask": "disagreement",
        "decision_mask": "primary AND disagreement",
    }
    assert set(verified["policies"]) == set(expected_policy_array_keys())
    assert len(selected_wave59_array_keys()) == 26


def test_bit_changes_to_state_calibration_scores_and_policy_are_rejected(
    source_material: dict,
) -> None:
    manifest = deepcopy(source_material["manifest"])
    manifest["portable_states"][USED_MODELS[0]]["n_features"] += 1
    with pytest.raises(RuntimeError, match="feature count"):
        project_transport_law(manifest, source_material["arrays"])

    calibration = deepcopy(source_material["calibration"])
    calibration["calibration"]["controls"].pop(CONTROL_MODELS[0])
    with pytest.raises(RuntimeError, match="control calibration roster"):
        frozen_policy_spec(calibration)

    scores = {key: value.copy() for key, value in source_material["scores"].items()}
    scores[USED_MODELS[0]][0, 0] = np.nextafter(scores[USED_MODELS[0]][0, 0], np.inf)
    with pytest.raises(RuntimeError, match="score reproduction"):
        retrospective_source_verification(
            source_material["manifest"],
            source_material["arrays"],
            source_material["calibration"],
            source_material["inference"],
            scores,
            source_material["policies"],
        )

    policies = {key: value.copy() for key, value in source_material["policies"].items()}
    policies["actions__HARD-SET"][0, 0] ^= 1
    with pytest.raises(RuntimeError, match="full policy reproduction"):
        retrospective_source_verification(
            source_material["manifest"],
            source_material["arrays"],
            source_material["calibration"],
            source_material["inference"],
            source_material["scores"],
            policies,
        )


def test_strict_thresholds_and_nonprimary_disagreement_remain_hard(
    source_material: dict,
) -> None:
    data = {
        key: value[:2].copy() for key, value in source_material["inference"].items()
    }
    data["primary"][:] = [True, False]
    data["disagreement"][:] = True
    spec = frozen_policy_spec(source_material["calibration"])
    scores = {
        identifier: np.zeros(data["disagreement"].shape, dtype=np.float64)
        for identifier in USED_MODELS
    }
    scores["proposer-hgb"][:] = FROZEN_THRESHOLDS["proposer-hgb"]
    arrays = apply_transport_policies(data, scores, spec)
    assert not arrays["proposal__hgb"].any()  # equality does not pass strict >

    scores["proposer-hgb"][:] = np.nextafter(FROZEN_THRESHOLDS["proposer-hgb"], np.inf)
    scores["guard-hgb-incompatibility"][:] = FROZEN_THRESHOLDS[MAIN_POLICIES["mean"]]
    scores["guard-hgb-harm"][:] = FROZEN_THRESHOLDS[MAIN_POLICIES["tail"]]
    for identifier in CONTROL_MODELS:
        scores[identifier][:] = FROZEN_THRESHOLDS[identifier]
    arrays = apply_transport_policies(data, scores, spec)
    assert arrays["proposal__hgb"][0].all()
    assert not any(
        arrays[f"authorized__{identifier}"][0].any()
        for identifier in (*MAIN_POLICIES.values(), *CONTROL_MODELS)
    )
    assert not arrays["proposal__hgb"][1].any()
    for identifier in REFERENCE_ACTIONS + (*MAIN_POLICIES.values(),) + CONTROL_MODELS:
        np.testing.assert_array_equal(
            arrays[f"actions__{identifier}"][1], data["hard_actions"][1]
        )


def test_closed_world_rejects_extra_model_policy_and_state_array(
    source_material: dict, verified: dict
) -> None:
    scores = dict(verified["scores"])
    scores["proposer-ridge"] = source_material["scores"]["proposer-ridge"]
    with pytest.raises(RuntimeError, match="not closed-world"):
        apply_transport_policies(
            source_material["inference"], scores, verified["verification"]["spec"]
        )

    spec = deepcopy(verified["verification"]["spec"])
    spec["controls"]["EXTRA"] = deepcopy(next(iter(spec["controls"].values())))
    with pytest.raises(RuntimeError, match="control roster"):
        validate_frozen_policy_spec(spec)

    arrays = dict(verified["arrays"])
    arrays["extra"] = np.asarray([1])
    with pytest.raises(RuntimeError, match="closed-world"):
        validate_transport_manifest(verified["manifest"], arrays)


def test_transport_does_not_call_fit_or_quantile(
    monkeypatch: pytest.MonkeyPatch, source_material: dict, verified: dict
) -> None:
    import numpy as numpy_module
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    import geometria_proporcional.wave59_hgb_guard_bracket as wave59_module

    def forbidden(*args, **kwargs):
        raise AssertionError("learning operation called")

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", forbidden)
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(Ridge, "fit", forbidden)
    monkeypatch.setattr(numpy_module, "quantile", forbidden)
    monkeypatch.setattr(wave59_module, "quantile_higher", forbidden)
    monkeypatch.setattr(wave59_module, "calibrate_policies", forbidden)
    score_transport_models(
        verified["manifest"], verified["arrays"], source_material["inference"]
    )
    apply_transport_policies(
        source_material["inference"],
        verified["scores"],
        verified["verification"]["spec"],
    )


def test_bootstrap_is_shared_lexicographic_and_full_size(source_material: dict) -> None:
    tokens = source_material["truth"]["pair_token"][source_material["truth"]["primary"]]
    first = wave60_bootstrap_indices(tokens)
    second = wave60_bootstrap_indices(tokens[::-1])
    assert first.shape == (BOOTSTRAP_REPLICATES, len(tokens))
    np.testing.assert_array_equal(first, second)


def test_evaluation_has_fourteen_actions_four_metrics_and_seven_core_conditions(
    source_material: dict, verified: dict
) -> None:
    analysis, arrays, bootstrap = evaluate_transport_actions(
        source_material["truth"],
        source_material["utilities"],
        1.25,
        verified["policies"],
    )
    assert len(analysis["metrics"]) == 14
    assert all(
        set(row) == {"accuracy", "compatible", "regret", "worst_regret"}
        for row in analysis["metrics"].values()
    )
    assert set(analysis["core_patterns"]) == {"incompatibility", "harm"}
    assert all(len(row) == 7 for row in analysis["core_conditions"].values())
    assert analysis["scientific_decision"] is None
    assert analysis["decision_authority"] == "user"
    assert bootstrap["indices"].shape[0] == 5000
    assert len(arrays) == 14 * 4


def test_final_pattern_adds_replay_without_mutating_local_analysis() -> None:
    analysis = {
        "core_conditions": {
            "incompatibility": {"authorized_pair_tokens_at_least_25": True, "x": True},
            "harm": {"authorized_pair_tokens_at_least_25": False, "x": True},
        },
        "core_patterns": {"incompatibility": True, "harm": "NOT_EVALUABLE"},
    }
    before = deepcopy(analysis)
    conditions, patterns = finalize_patterns(analysis, True)
    assert analysis == before
    assert conditions["incompatibility"]["replay_exact"] is True
    assert patterns == {"incompatibility": True, "harm": "NOT_EVALUABLE"}

    false_analysis = deepcopy(analysis)
    false_analysis["core_conditions"]["incompatibility"]["support"] = False
    false_analysis["core_patterns"]["incompatibility"] = False
    _, false_patterns = finalize_patterns(false_analysis, True)
    assert false_patterns["incompatibility"] is False


def test_pair_status_covers_success_pretruth_and_posttruth() -> None:
    digest = "0" * 64
    assert (
        pair_status(
            "EVALUATED_IMMUTABLE",
            "EVALUATED_IMMUTABLE",
            digest,
            digest,
            any_truth_accessed=True,
        )["terminal"]
        == "COMPLETE"
    )
    pre = pair_status(
        "SCORE_APPLY_FAILED_PRE_TRUTH",
        "PEER_ABORTED_PRE_TRUTH",
        digest,
        digest,
        any_truth_accessed=False,
    )
    assert (
        pre["terminal"] == "PAIR_ABORTED_PRE_TRUTH" and pre["recovery_allowed"] is True
    )
    post = pair_status(
        "EVALUATION_FAILED_POST_TRUTH",
        "EVALUATED_IMMUTABLE",
        digest,
        digest,
        any_truth_accessed=True,
    )
    assert (
        post["terminal"] == "PAIR_ABORTED_POST_TRUTH"
        and post["recovery_allowed"] is False
    )
    finalize_failure = pair_status(
        "EVALUATED_IMMUTABLE",
        "EVALUATED_IMMUTABLE",
        digest,
        digest,
        any_truth_accessed=True,
        pair_failed=True,
    )
    assert finalize_failure["terminal"] == "PAIR_ABORTED_POST_TRUTH"


def test_pair_finalize_resume_preserves_science_and_accumulates_signed_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt"
    for role in ("primary", "replay"):
        root = attempt / role
        (root / "evaluation").mkdir(parents=True)
        (root / "artifact_manifest.json").write_text(
            json.dumps({"terminal": "EVALUATED_IMMUTABLE"}) + "\n",
            encoding="utf-8",
        )
        (root / "evaluation/analysis.json").write_text(
            json.dumps(
                {
                    "core_conditions": {
                        "incompatibility": {"support": True},
                        "harm": {"support": True},
                    },
                    "core_patterns": {"incompatibility": True, "harm": True},
                    "limitations": ["synthetic_generator_only"],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "evaluation/evaluation_attestation.json").write_text(
            "{}\n", encoding="utf-8"
        )
    comparison = {
        "schema_version": "wave60-replay-finalize-v1",
        "status": "EXACT",
        "primary_scientific_hashes": {},
        "replay_scientific_hashes": {},
        "exact_json_md": {},
        "exact_npz": {},
        "functional_states": {},
        "secret_hashes": {},
        "operational_semantic": {},
        "mismatches": [],
    }
    monkeypatch.setattr(wave60_runner, "compare_evaluated_roots", lambda *_: comparison)
    primary_binding = file_sha256(attempt / "primary/artifact_manifest.json")
    replay_binding = file_sha256(attempt / "replay/artifact_manifest.json")
    monkeypatch.setattr(
        wave60_runner,
        "validate_evaluated_root",
        lambda _root, role: (primary_binding if role == "primary" else replay_binding),
    )
    monkeypatch.setattr(
        wave60_runner,
        "pair_durable_elapsed",
        lambda *_: {
            "preparation_seconds": 1.0,
            "worker_seconds": 1.0,
            "coordinator_seconds": 1.0,
            "durable_seconds": 2.0,
        },
    )
    staged = attempt / "pair.initializing"
    staged.mkdir()
    prior_status = pair_status(
        "EVALUATED_IMMUTABLE",
        "EVALUATED_IMMUTABLE",
        primary_binding,
        replay_binding,
        any_truth_accessed=True,
    )
    prior_status["created_at"] = "2000-01-01T00:00:00Z"
    (staged / "pair_status.json").write_text(
        json.dumps(prior_status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prior_bytes = (staged / "pair_status.json").read_bytes()
    original_replace = wave60_runner.os.replace

    def interrupt_pair_publish(source: object, destination: object) -> None:
        if Path(source) == staged and Path(destination) == attempt / "pair":
            raise OSError("synthetic rename interruption")
        original_replace(source, destination)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(wave60_runner.os, "replace", interrupt_pair_publish)
        with pytest.raises(OSError, match="rename interruption"):
            wave60_runner.finalize_pair(attempt)
    prior_runtime = load_json(staged / "runtime.json")
    prior_static_hashes = {
        relative: file_sha256(staged / relative)
        for relative in (
            "pair_status.json",
            "replay_comparison.json",
            "final_analysis.json",
            "replay_finalize_freeze.json",
            "journals/replay_finalize.json",
            "replay_finalize_receipt.json",
            "REPORT.md",
        )
    }
    result = wave60_runner.finalize_pair(attempt)
    assert result == attempt / "pair"
    assert (result / "pair_status.json").read_bytes() == prior_bytes
    assert not staged.exists()
    assert {
        relative: file_sha256(result / relative) for relative in prior_static_hashes
    } == prior_static_hashes
    runtime = load_json(result / "runtime.json")
    assert runtime["budget"]["finalize_invocations"] == 2
    assert runtime["budget"]["finalize_seconds"] > prior_runtime["budget"][
        "finalize_seconds"
    ]
    assert runtime["budget"]["observed_total_seconds"] == pytest.approx(
        runtime["budget"]["elapsed_before_finalize_seconds"]
        + runtime["budget"]["finalize_seconds"]
    )
    assert load_json(result / "replay_finalize_attestation.json")["payload"][
        "runtime_sha256"
    ] == file_sha256(result / "runtime.json")


def test_direct_finalize_rejects_exhausted_durable_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt"
    for role in ("primary", "replay"):
        (attempt / role).mkdir(parents=True)
    monkeypatch.setattr(
        wave60_runner,
        "pair_durable_elapsed",
        lambda *_: {
            "preparation_seconds": 450.0,
            "worker_seconds": 450.0,
            "coordinator_seconds": 450.0,
            "durable_seconds": 900.0,
        },
    )
    with pytest.raises(IntegrityDriftError, match="budget is exhausted"):
        wave60_runner.finalize_pair(attempt)
    assert not (attempt / "pair").exists()


def test_resumed_finalize_charges_prior_899_seconds_before_atomic_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt"
    for role in ("primary", "replay"):
        root = attempt / role
        (root / "evaluation").mkdir(parents=True)
        _write_json(root / "artifact_manifest.json", {"role": role})
        _write_json(
            root / "evaluation/analysis.json",
            {
                "core_conditions": {
                    "incompatibility": {"support": True},
                    "harm": {"support": True},
                },
                "core_patterns": {"incompatibility": True, "harm": True},
                "limitations": ["synthetic_generator_only"],
            },
        )
        _write_json(root / "evaluation/evaluation_attestation.json", {})
    bindings = {
        role: file_sha256(attempt / role / "artifact_manifest.json")
        for role in ("primary", "replay")
    }
    monkeypatch.setattr(
        wave60_runner,
        "validate_evaluated_root",
        lambda _root, role: bindings[role],
    )
    monkeypatch.setattr(
        wave60_runner,
        "compare_evaluated_roots",
        lambda *_: {
            "schema_version": "wave60-replay-finalize-v1",
            "status": "EXACT",
            "primary_scientific_hashes": {},
            "replay_scientific_hashes": {},
            "exact_json_md": {},
            "exact_npz": {},
            "functional_states": {},
            "secret_hashes": {},
            "operational_semantic": {},
            "mismatches": [],
        },
    )
    monkeypatch.setattr(
        wave60_runner,
        "pair_durable_elapsed",
        lambda *_: {
            "preparation_seconds": 449.0,
            "worker_seconds": 449.0,
            "coordinator_seconds": 449.0,
            "durable_seconds": 898.0,
        },
    )
    staging = attempt / "pair.initializing"
    staging.mkdir()
    prior_runtime = {
        "schema_version": "wave60-pair-final-v1",
        "terminal": "COMPLETE",
        "cuda_visible_devices": "",
        "cpu_threads": 4,
        "budget": {
            "max_seconds_total": 900.0,
            "preparation_seconds": 449.0,
            "worker_seconds": 449.0,
            "coordinator_seconds": 449.0,
            "durable_seconds": 898.0,
            "elapsed_before_finalize_seconds": 898.0,
            "finalize_seconds": 1.0,
            "finalize_invocations": 1,
            "observed_total_seconds": 899.0,
            "budget_enforced": True,
        },
    }
    _write_json(staging / "runtime.json", prior_runtime)
    prior_runtime_bytes = (staging / "runtime.json").read_bytes()
    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(wave60_runner.time, "monotonic", lambda: next(ticks))
    with pytest.raises(IntegrityDriftError, match="finalize exceeded"):
        wave60_runner.finalize_pair(attempt)
    assert not (attempt / "pair").exists()
    assert (staging / "runtime.json").read_bytes() == prior_runtime_bytes


def test_transient_finalize_error_preserves_recoverable_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt"
    for role in ("primary", "replay"):
        (attempt / role).mkdir(parents=True)
    staging = attempt / "pair.initializing"
    staging.mkdir()
    (staging / "marker").write_text("recoverable\n", encoding="utf-8")
    monkeypatch.setattr(wave60_runner, "validate_prepared_root", lambda *_a, **_k: {})
    monkeypatch.setattr(wave60_runner, "pair_preparation_elapsed", lambda *_: 0.0)
    monkeypatch.setattr(
        wave60_runner,
        "validate_new_draw_pair",
        lambda *_a, **_k: {"status": "PASS"},
    )
    monkeypatch.setattr(wave60_runner, "bind_source_law", lambda *_a, **_k: {})
    monkeypatch.setattr(wave60_runner, "score_root", lambda *_a, **_k: None)
    monkeypatch.setattr(wave60_runner, "evaluate_root", lambda *_a, **_k: None)
    monkeypatch.setattr(
        wave60_runner, "seal_evaluated_root", lambda *_a, **_k: "0" * 64
    )
    monkeypatch.setattr(
        wave60_runner,
        "finalize_pair",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("temporary I/O")),
    )
    with pytest.raises(OSError, match="temporary I/O"):
        execute_prepared_pair(attempt, {"runtime_budget": {}})
    assert (staging / "marker").read_text(encoding="utf-8") == "recoverable\n"
    assert not (attempt / "pair").exists()


def test_failure_bindings_are_directional_and_pair_package_is_atomic(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt_v1"
    primary = attempt / "primary"
    replay = attempt / "replay"
    primary.mkdir(parents=True)
    replay.mkdir()
    for root in (primary, replay):
        (root / "config.snapshot.json").write_text("{}\n", encoding="utf-8")
        (root / "source_bindings.json").write_text("{}\n", encoding="utf-8")
    (primary / "failed_preparation").mkdir()
    (primary / "failed_preparation/preparation_error.json").write_text(
        "{}\n", encoding="utf-8"
    )
    own_binding = seal_root_failure(
        primary,
        terminal="INVALID_PREPARATION",
        phase="prepare",
        role="primary",
        truth_accessed=False,
        error=RuntimeError("synthetic failure"),
        authority_binding_sha256=file_sha256(primary / "config.snapshot.json"),
        last_complete_phase="INITIALIZED",
    )
    own = load_json(primary / "FAILURE.json")
    assert own["peer_terminal"] is None
    assert own["peer_terminal_binding_sha256"] is None
    replay_binding = seal_root_failure(
        replay,
        terminal="PEER_ABORTED_PRE_TRUTH",
        phase="peer_abort",
        role="replay",
        truth_accessed=False,
        error=RuntimeError("peer failed"),
        authority_binding_sha256=file_sha256(replay / "config.snapshot.json"),
        last_complete_phase="INITIALIZED",
        peer_terminal="INVALID_PREPARATION",
        peer_terminal_binding_sha256=own_binding,
    )
    peer = load_json(replay / "FAILURE.json")
    assert peer["peer_terminal_binding_sha256"] == own_binding
    status = pair_status(
        own["terminal"],
        peer["terminal"],
        own_binding,
        replay_binding,
        any_truth_accessed=False,
    )
    published = publish_pair_failure(
        attempt, status, error=RuntimeError("synthetic pair failure")
    )
    assert published == attempt / "pair"
    assert not (attempt / "pair.initializing").exists()
    assert load_json(published / "pair_status.json")["recovery_allowed"] is True
    assert load_json(published / "FAILURE.json")["schema_version"] == (
        "wave60-pair-failure-v1"
    )
    assert validate_pair_failure_package(attempt)["terminal"] == (
        "PAIR_ABORTED_PRE_TRUTH"
    )
    replay_attestation = replay / "failure_attestation.json"
    replay_attestation_bytes = replay_attestation.read_bytes()
    replay_attestation.write_bytes(replay_attestation_bytes + b"\n")
    with pytest.raises(IntegrityDriftError, match="root binding"):
        validate_pair_failure_package(attempt)
    replay_attestation.write_bytes(replay_attestation_bytes)
    validate_pair_failure_package(attempt)


def test_pair_failure_publication_requires_two_physical_root_terminals(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt_v1"
    attempt.mkdir()
    status = pair_status(
        "INVALID_NEW_DRAW_IDENTITY",
        "PEER_ABORTED_PRE_TRUTH",
        "1" * 64,
        "2" * 64,
        any_truth_accessed=False,
    )
    with pytest.raises(IntegrityDriftError, match="root is absent"):
        publish_pair_failure(
            attempt, status, error=RuntimeError("unbacked declaration")
        )
    assert not (attempt / "pair").exists()
    assert not (attempt / "pair.initializing").exists()
    physical = tmp_path / "physical-attempt"
    physical.mkdir()
    alias = tmp_path / "attempt-alias"
    alias.symlink_to(physical, target_is_directory=True)
    with pytest.raises(IntegrityDriftError, match="attempt container.*physical"):
        validate_pair_status_against_roots(alias, status)


def test_success_finalize_rejects_attempt_and_root_symlink_aliases(
    tmp_path: Path,
) -> None:
    physical = tmp_path / "physical-attempt"
    for role in ("primary", "replay"):
        (physical / role).mkdir(parents=True)
    alias = tmp_path / "attempt-alias"
    alias.symlink_to(physical, target_is_directory=True)
    with pytest.raises(IntegrityDriftError, match="attempt container.*physical"):
        wave60_runner.finalize_pair(alias)

    for aliased_role in ("primary", "replay"):
        attempt = tmp_path / f"attempt-{aliased_role}-alias"
        attempt.mkdir()
        outside = tmp_path / f"outside-{aliased_role}"
        outside.mkdir()
        other_role = "replay" if aliased_role == "primary" else "primary"
        (attempt / other_role).mkdir()
        (attempt / aliased_role).symlink_to(outside, target_is_directory=True)
        with pytest.raises(
            IntegrityDriftError,
            match=rf"{aliased_role} root.*physical",
        ):
            wave60_runner.finalize_pair(attempt)


def test_failure_presence_matrix_rejects_claimed_completed_phases(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    (root / "journals").mkdir(parents=True)
    (root / "config.snapshot.json").write_text("{}\n", encoding="utf-8")
    (root / "source_bindings.json").write_text("{}\n", encoding="utf-8")
    (root / "journals/score_apply.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="INVALID_PREPARATION"):
        seal_root_failure(
            root,
            terminal="SCORE_APPLY_FAILED_PRE_TRUTH",
            phase="score_apply",
            role="primary",
            truth_accessed=False,
            error=RuntimeError("synthetic failure"),
            authority_binding_sha256=file_sha256(root / "config.snapshot.json"),
            last_complete_phase="SOURCE_LAW_BOUND",
        )


def test_pretruth_failure_terminals_inventory_exact_common(
    tmp_path: Path, source_material: dict
) -> None:
    config = valid_config()
    draw_root = tmp_path / "invalid-draw"
    _build_prepared_primary(draw_root, config, source_material)
    binding = seal_root_failure(
        draw_root,
        terminal="INVALID_NEW_DRAW_IDENTITY",
        phase="new_draw_identity",
        role="primary",
        truth_accessed=False,
        error=RuntimeError("identity collision"),
        authority_binding_sha256=file_sha256(draw_root / "config.snapshot.json"),
        last_complete_phase="PREPARED",
    )
    assert len(binding) == 64
    inventory = load_json(draw_root / "failure_inventory.json")
    assert inventory["missing_expected"] == []
    assert inventory["forbidden_present"] == []


@pytest.mark.parametrize(
    ("phase", "truth_accessed"),
    (("WRONG_PHASE", False), ("new_draw_identity", True)),
)
def test_failure_terminal_rejects_contradictory_semantics(
    tmp_path: Path,
    source_material: dict,
    phase: str,
    truth_accessed: bool,
) -> None:
    root = tmp_path / f"root-{phase}-{truth_accessed}"
    config = valid_config()
    _build_prepared_primary(root, config, source_material)
    with pytest.raises(RuntimeError, match="terminal semantics"):
        seal_root_failure(
            root,
            terminal="INVALID_NEW_DRAW_IDENTITY",
            phase=phase,
            role="primary",
            truth_accessed=truth_accessed,
            error=RuntimeError("contradiction"),
            authority_binding_sha256=file_sha256(root / "config.snapshot.json"),
            last_complete_phase="PREPARED",
        )
    assert not (root / "FAILURE.json").exists()

    source_root = tmp_path / "source-failure"
    _build_prepared_primary(source_root, config, source_material)
    wave60_runner.write_failure_journal(
        source_root,
        "source_bind",
        RuntimeError("source unavailable"),
        input_sha256=file_sha256(source_root / "config.snapshot.json"),
        truth_accessed=False,
        duration_seconds=0.1,
    )
    seal_root_failure(
        source_root,
        terminal="SOURCE_BINDING_FAILED_PRE_TRUTH",
        phase="source_bind",
        role="primary",
        truth_accessed=False,
        error=RuntimeError("source unavailable"),
        authority_binding_sha256=file_sha256(source_root / "config.snapshot.json"),
        last_complete_phase="PREPARED",
    )
    inventory = load_json(source_root / "failure_inventory.json")
    assert inventory["missing_expected"] == []
    assert inventory["forbidden_present"] == []


def test_replay_budget_inherits_signed_primary_duration(
    tmp_path: Path, source_material: dict
) -> None:
    root = tmp_path / "primary"
    config = valid_config()
    _build_prepared_primary(root, config, source_material)
    args = SimpleNamespace(
        replay_secrets_from=root,
        recovery_secrets_from=None,
    )
    assert preparer.wave60_prior_preparation_elapsed(args, config, "replay") == 1.0
    receipt = load_json(root / "preparation_receipt.json")
    receipt["coordinator_budget"]["duration_seconds"] = 2.0
    _write_json(root / "preparation_receipt.json", receipt)
    with pytest.raises(RuntimeError, match="not signed"):
        preparer.wave60_prior_preparation_elapsed(args, config, "replay")


def test_attempt_initialization_is_atomic_and_requires_absent_target(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    attempt = tmp_path / "attempt_v1"
    initialize_attempt_container(attempt, config, {"binding": "x"})
    assert sorted(path.name for path in attempt.iterdir()) == ["primary", "replay"]
    for role in ("primary", "replay"):
        assert {path.name for path in (attempt / role).iterdir()} == {
            "config.snapshot.json",
            "source_bindings.json",
        }
    with pytest.raises(FileExistsError):
        initialize_attempt_container(attempt, config, {})


def test_incomplete_initialized_pair_terminates_as_invalid_preparation(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    attempt = tmp_path / "attempt_v1"
    initialize_attempt_container(attempt, config_path, {"binding": "x"})
    pair = execute_prepared_pair(attempt, {})
    assert load_json(attempt / "primary/FAILURE.json")["terminal"] == (
        "INVALID_PREPARATION"
    )
    assert load_json(attempt / "replay/FAILURE.json")["terminal"] == (
        "INVALID_PREPARATION"
    )
    assert load_json(pair / "pair_status.json")["terminal"] == (
        "PAIR_ABORTED_PRE_TRUTH"
    )


@pytest.mark.parametrize("failed_role", ("primary", "replay"))
@pytest.mark.parametrize(
    ("failed_phase", "failed_terminal", "pair_terminal", "truth_accessed"),
    (
        ("prepare", "INVALID_PREPARATION", "PAIR_ABORTED_PRE_TRUTH", False),
        (
            "source",
            "SOURCE_BINDING_FAILED_PRE_TRUTH",
            "PAIR_ABORTED_PRE_TRUTH",
            False,
        ),
        (
            "score",
            "SCORE_APPLY_FAILED_PRE_TRUTH",
            "PAIR_ABORTED_PRE_TRUTH",
            False,
        ),
        (
            "evaluate",
            "EVALUATION_FAILED_POST_TRUTH",
            "PAIR_ABORTED_POST_TRUTH",
            True,
        ),
    ),
)
def test_asymmetric_phase_failures_publish_exact_terminals(
    tmp_path: Path,
    source_material: dict,
    source_authority: dict,
    monkeypatch: pytest.MonkeyPatch,
    failed_role: str,
    failed_phase: str,
    failed_terminal: str,
    pair_terminal: str,
    truth_accessed: bool,
) -> None:
    config = valid_config()
    config["implementation_binding"]["commit"] = source_authority["request"][
        "implementation_commit"
    ]
    config["implementation_binding"]["audit_sha256"] = source_authority[
        "request"
    ]["implementation_audit_sha256"]
    config["source_law_authority"].update(source_authority["binding"])
    config["source_binding"] = load_json(WAVE59 / "source_bindings.json")
    attempt = tmp_path / f"attempt-{failed_phase}-{failed_role}"
    _build_prepared_pair(attempt, config, source_material)

    if failed_phase == "prepare":
        path = attempt / failed_role / "preparation_attestation.json"
        path.unlink()
    else:
        attribute = {
            "source": "bind_source_law",
            "score": "score_root",
            "evaluate": "evaluate_root",
        }[failed_phase]
        original = getattr(wave60_runner, attribute)

        def fail_selected(
            root: Path, role: str, *args: object, **kwargs: object
        ) -> object:
            if role == failed_role:
                raise RuntimeError(f"injected {failed_phase} failure")
            return original(root, role, *args, **kwargs)

        monkeypatch.setattr(wave60_runner, attribute, fail_selected)

    pair = execute_prepared_pair(
        attempt, config, authority=source_authority["path"]
    )
    peer_role = "replay" if failed_role == "primary" else "primary"
    status = load_json(pair / "pair_status.json")
    assert status["terminal"] == pair_terminal
    assert status["any_truth_accessed"] is truth_accessed
    assert status["recovery_allowed"] is (not truth_accessed)
    assert status[f"{failed_role}_terminal"] == failed_terminal

    failed = load_json(attempt / failed_role / "FAILURE.json")
    assert failed["terminal"] == failed_terminal
    assert failed["truth_accessed"] is truth_accessed
    assert failed["recovery_allowed"] is (not truth_accessed)
    assert failed["peer_terminal"] is None
    assert failed["peer_terminal_binding_sha256"] is None
    assert status[f"{failed_role}_terminal_binding_sha256"] == file_sha256(
        attempt / failed_role / "failure_attestation.json"
    )

    if failed_phase == "evaluate":
        assert status[f"{peer_role}_terminal"] == "EVALUATED_IMMUTABLE"
        assert not (attempt / peer_role / "FAILURE.json").exists()
        assert status[f"{peer_role}_terminal_binding_sha256"] == file_sha256(
            attempt / peer_role / "artifact_manifest.json"
        )
    else:
        assert status[f"{peer_role}_terminal"] == "PEER_ABORTED_PRE_TRUTH"
        peer = load_json(attempt / peer_role / "FAILURE.json")
        assert peer["phase"] == "peer_abort"
        assert peer["truth_accessed"] is False
        assert peer["recovery_allowed"] is True
        assert peer["peer_terminal"] == failed_terminal
        assert peer["peer_terminal_binding_sha256"] == status[
            f"{failed_role}_terminal_binding_sha256"
        ]
        assert status[f"{peer_role}_terminal_binding_sha256"] == file_sha256(
            attempt / peer_role / "failure_attestation.json"
        )


def test_source_law_failure_publishes_only_the_closed_invalid_terminal(
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".wave60-source-failure-test-", dir=REPO_ROOT
    ) as raw:
        workspace = Path(raw)
        request = workspace / "source_law_request.json"
        request.write_text(
            json.dumps({"output_path": "deliberately-invalid"}) + "\n",
            encoding="utf-8",
        )
        output = workspace / "source_authority"
        result = publish_source_law_authority(request, output)
        assert result == output
        assert {
            str(path.relative_to(output))
            for path in output.rglob("*")
            if path.is_file()
        } == {
            "source_law_request.json",
            "journals/verify_source_law.json",
            "FAILURE.json",
            "failure_inventory.json",
            "failure_attestation.json",
        }
        assert load_json(output / "FAILURE.json")["terminal"] == (
            "SOURCE_LAW_INVALID"
        )


def test_source_law_publisher_refuses_unaudited_implementation() -> None:
    with tempfile.TemporaryDirectory(
        prefix=".wave60-source-publish-test-", dir=REPO_ROOT
    ) as raw:
        workspace = Path(raw)
        output = workspace / "authority"
        request_path = workspace / "request.json"
        request = {
            "schema_version": "wave60-source-law-v1",
            "plan_commit": PLAN_COMMIT,
            "plan_sha256": PLAN_SHA256,
            "implementation_commit": "1" * 40,
            "implementation_audit_commit": "2" * 40,
            "implementation_audit_sha256": "3" * 64,
            "source_paths": {name: str(path) for name, path in SOURCE_ALIASES.items()},
            "source_sha256": dict(SOURCE_HASHES),
            "output_path": str(output.relative_to(REPO_ROOT)),
            "runtime_budget": {
                "max_seconds": 900,
                "max_rss_bytes": 1610612736,
            },
        }
        request_path.write_text(
            json.dumps(request, sort_keys=True) + "\n", encoding="utf-8"
        )
        published = publish_source_law_authority(request_path, output)
        assert load_json(published / "FAILURE.json")["terminal"] == (
            "SOURCE_LAW_INVALID"
        )
        assert not (published / "source_authority_manifest.json").exists()


def test_source_law_recovery_request_and_prior_terminal_are_authentic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = recovery_source_request()
    validate_source_law_recovery_request(request)
    before = {
        str(path.relative_to(PRIOR_SOURCE_AUTHORITY)): file_sha256(path)
        for path in PRIOR_SOURCE_AUTHORITY.rglob("*")
        if path.is_file()
    }
    real_iterdir = Path.iterdir

    def pre_attempt_view(path: Path):
        entries = real_iterdir(path)
        if path == PRIOR_SOURCE_AUTHORITY.parent:
            return (
                entry
                for entry in entries
                if not entry.name.startswith(
                    "wave60_frozen_policy_transport_attempt_v"
                )
            )
        return entries

    # The v1 validator deliberately models the historical pre-attempt instant.
    # The canonical attempt now exists, so isolate only that final temporal guard.
    monkeypatch.setattr(Path, "iterdir", pre_attempt_view)
    duration = validate_prior_source_law_failure(request["recovery"])
    after = {
        str(path.relative_to(PRIOR_SOURCE_AUTHORITY)): file_sha256(path)
        for path in PRIOR_SOURCE_AUTHORITY.rglob("*")
        if path.is_file()
    }
    assert duration == 0.00401783362030983
    assert before == after
    assert canonical_source_output_path(
        Path(request["output_path"])
    ) == wave60_runner.SOURCE_AUTHORITY_DEFAULT


@pytest.mark.parametrize(
    "mutation",
    (
        "alias_absent",
        "alias_duplicated",
        "path_crossed",
        "same_content_path",
        "hash_crossed",
    ),
)
def test_invalid_preparation_hard_set_request_alias_rejects_drift(
    mutation: str,
) -> None:
    request = load_json(
        REPO_ROOT
        / "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_source_law_v2/"
        "source_law_request.json"
    )
    contract = invalid_preparation_hard_set_contract()
    if mutation == "alias_absent":
        request["source_paths"].pop(contract["request_alias"])
        request["source_sha256"].pop(contract["request_alias"])
    elif mutation == "alias_duplicated":
        request["source_paths"]["duplicate_snapshot.json"] = contract["source_path"]
        request["source_sha256"]["duplicate_snapshot.json"] = contract[
            "source_sha256"
        ]
    elif mutation == "path_crossed":
        request["source_paths"][contract["request_alias"]] = request[
            "source_paths"
        ]["wave59_artifact_manifest.json"]
    elif mutation == "same_content_path":
        request["source_paths"][contract["request_alias"]] = (
            contract["source_path"] + ".byte_identical_copy"
        )
    else:
        request["source_sha256"][contract["request_alias"]] = request[
            "source_sha256"
        ]["wave59_artifact_manifest.json"]
    with pytest.raises(RuntimeError, match="hard-set request alias binding"):
        preparer.validate_wave60_hard_set_request_alias(request, contract)


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    (
        ("manifest", "append", "authority manifest"),
        ("request", "append", "request is not manifest-bound"),
        ("snapshot", "append", "snapshot hash"),
        ("snapshot", "symlink", "snapshot path escaped"),
    ),
)
def test_invalid_preparation_hard_set_physical_chain_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    mutation: str,
    message: str,
) -> None:
    authority_relative = Path(
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_source_law_v2"
    )
    snapshot_relative = Path(
        "data/geometria_proporcional/"
        "wave59_fresh_hgb_guard_bracket_replay_normalized_v1/"
        "config.snapshot.json"
    )
    authority = tmp_path / authority_relative
    snapshot = tmp_path / snapshot_relative
    shutil.copytree(REPO_ROOT / authority_relative, authority)
    snapshot.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / snapshot_relative, snapshot)
    targets = {
        "manifest": authority / "source_authority_manifest.json",
        "request": authority / "source_law_request.json",
        "snapshot": snapshot,
    }
    path = targets[target]
    if mutation == "append":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        original = snapshot.with_name("config.snapshot.original.json")
        snapshot.replace(original)
        snapshot.symlink_to(original.name)
    config = load_json(
        REPO_ROOT
        / "experiments/geometria_proporcional/configs/"
        "wave60_frozen_policy_transport.json"
    )
    monkeypatch.setattr(wave60_runner, "REPO_ROOT", tmp_path)
    # Each negative targets this adapter's additional physical chain. The real
    # source-law validator is exercised without substitution in the positive.
    monkeypatch.setattr(wave60_runner, "validate_source_authority", lambda *_: None)
    with pytest.raises(RuntimeError, match=message):
        preparer.validate_wave60_invalid_preparation_hard_set_contract(
            {"hard_set_contract": invalid_preparation_hard_set_contract()},
            config,
            repo_root=tmp_path,
        )


def test_invalid_preparation_hard_set_manifest_must_be_config_bound() -> None:
    config = load_json(
        REPO_ROOT
        / "experiments/geometria_proporcional/configs/"
        "wave60_frozen_policy_transport.json"
    )
    config["source_law_authority"]["source_authority_manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="authority is not config-bound"):
        preparer.validate_wave60_invalid_preparation_hard_set_contract(
            {"hard_set_contract": invalid_preparation_hard_set_contract()}, config
        )


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("absent", None),
        ("different", 0.6),
        ("non_finite", float("nan")),
        ("text", "0.5"),
        ("boolean", True),
    ),
)
def test_invalid_preparation_hard_set_snapshot_value_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    value: object,
) -> None:
    authority_relative = Path(
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_source_law_v2"
    )
    snapshot_relative = Path(
        "data/geometria_proporcional/"
        "wave59_fresh_hgb_guard_bracket_replay_normalized_v1/"
        "config.snapshot.json"
    )
    authority = tmp_path / authority_relative
    snapshot = tmp_path / snapshot_relative
    shutil.copytree(REPO_ROOT / authority_relative, authority)
    snapshot.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / snapshot_relative, snapshot)
    payload = load_json(snapshot)
    if mutation == "absent":
        payload.pop("hard_set_tau")
    else:
        payload["hard_set_tau"] = value
    snapshot.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    config = load_json(
        REPO_ROOT
        / "experiments/geometria_proporcional/configs/"
        "wave60_frozen_policy_transport.json"
    )
    contract = invalid_preparation_hard_set_contract()
    monkeypatch.setattr(wave60_runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(wave60_runner, "validate_source_authority", lambda *_: None)
    physical_sha256 = preparer.sha256_file

    def digest_with_authenticated_snapshot(path: Path) -> str:
        if path.resolve() == snapshot.resolve():
            return contract["source_sha256"]
        return physical_sha256(path)

    # Isolate the semantic guard after the independently tested physical hash
    # boundary; this models a digest match without weakening the positive test.
    monkeypatch.setattr(preparer, "sha256_file", digest_with_authenticated_snapshot)
    with pytest.raises(RuntimeError, match="snapshot value drifted"):
        preparer.validate_wave60_invalid_preparation_hard_set_contract(
            {"hard_set_contract": contract}, config, repo_root=tmp_path
        )


def test_invalid_preparation_hard_set_snapshot_path_rejects_traversal(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="path is not canonical"):
        preparer.require_canonical_repo_file(
            tmp_path,
            "../config.snapshot.json",
            "Wave 60 hard-set snapshot",
        )


def test_invalid_preparation_hard_set_real_chain_and_materializer(
    tmp_path: Path,
) -> None:
    from run_wave59_hgb_guard_bracket import materialize_prepared_bundles

    config = load_json(
        REPO_ROOT
        / "experiments/geometria_proporcional/configs/"
        "wave60_frozen_policy_transport.json"
    )
    tau = preparer.validate_wave60_invalid_preparation_hard_set_contract(
        {"hard_set_contract": invalid_preparation_hard_set_contract()}, config
    )
    assert tau == 0.5
    assert "hard_set_tau" not in config
    source = (
        REPO_ROOT
        / "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v1/primary/failed_preparation"
    )
    run = tmp_path / "run"
    run.mkdir()
    shutil.copytree(source / "benchmark", run / "benchmark")
    shutil.copytree(source / "inference", run / "inference")
    hashes = materialize_prepared_bundles(
        run,
        {**config, "hard_set_tau": tau},
        policy_manifest=(
            REPO_ROOT
            / "data/geometria_proporcional/wave52_policy_transport_v1/"
            "policy_manifest.json"
        ),
        wave54_selection_freeze=(
            REPO_ROOT
            / "data/geometria_proporcional/wave54_joint_set_v1/"
            "selection_freeze.json"
        ),
    )
    assert set(hashes) == {
        "prepared/gate_fit_bundle.npz",
        "prepared/gate_select_truth_bundle.npz",
        "prepared/gate_select_inference_bundle.npz",
        "prepared/sealed_monitor_truth_bundle.npz",
        "prepared/sealed_monitor_inference_bundle.npz",
    }
    assert all(file_sha256(run / relative) == digest for relative, digest in hashes.items())


def test_invalid_preparation_transaction_wires_tau_and_signed_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = (
        REPO_ROOT
        / "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v1/primary"
    )
    draw = primary / "failed_preparation"
    manifest = load_json(draw / "benchmark/manifest.json")
    preserved_relatives = [
        preparer.ESCROW_NAME,
        preparer.FREEZE_NAME,
        "benchmark/manifest.json",
        *(f"benchmark/{relative}" for relative in manifest["files"]),
    ]
    preserved = {
        relative: file_sha256(draw / relative) for relative in preserved_relatives
    }
    population = {
        split: preparer.sealed_population_counts(
            draw / "benchmark/sealed" / f"{split}.jsonl"
        )
        for split in preparer.SPLITS
    }
    config = load_json(
        REPO_ROOT
        / "experiments/geometria_proporcional/configs/"
        "wave60_frozen_policy_transport.json"
    )
    amendment_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_INVALID_PREPARATION_RECOVERY_V2_AMENDMENT.json"
    )
    config["attempt"] = {
        "version": 2,
        "container": (
            "data/geometria_proporcional/"
            "wave60_frozen_policy_transport_attempt_v2"
        ),
        "primary": "primary",
        "replay": "replay",
        "pair": "pair",
        "recovery": {
            "schema_version": "wave60-pretruth-recovery-v1",
            "prior_attempt_container": str(primary.parent.relative_to(REPO_ROOT)),
            "prior_pair_failure_sha256": file_sha256(
                primary.parent / "pair/FAILURE.json"
            ),
            "prior_config_audit_commit": "2" * 40,
            "prior_config_audit_path": "prior_config_audit.md",
            "prior_config_audit_sha256": "3" * 64,
            "amendment_path": amendment_relative,
            "amendment_sha256": "4" * 64,
            "amendment_audit_commit": "5" * 40,
            "amendment_audit_path": "amendment_audit.md",
            "amendment_audit_sha256": "6" * 64,
            "preserved_draw_sha256": preserved,
        },
    }
    config["output_parent_relative"] = config["attempt"]["container"]
    config["primary_output"] = f"{config['attempt']['container']}/primary"
    config["replay_output"] = f"{config['attempt']['container']}/replay"
    assert "hard_set_tau" not in config

    amendment = {
        "schema_version": (
            preparer.WAVE60_INVALID_PREPARATION_RECOVERY_AMENDMENT_SCHEMA
        ),
        "recovery_kind": "INVALID_PREPARATION",
        "prior_attempt_container": str(primary.parent.relative_to(REPO_ROOT)),
        "prior_pair_failure_sha256": file_sha256(
            primary.parent / "pair/FAILURE.json"
        ),
        "hard_set_contract": invalid_preparation_hard_set_contract(),
        "unledgered_preparation_debit": {
            "seconds": 60.0,
            "regime": "CONSERVATIVE_UNSIGNED_PREPARATION_DEBIT",
            "observed_external_wall_seconds": 48.51,
            "observed_external_record_authority": (
                "TRANSCRIPT_ONLY_NOT_SIGNED_LEDGER"
            ),
            "applied_once": True,
        },
        "escrow_origin": {
            "contract_sha256": preparer.compact_json_sha256(
                preparer.read_escrow(draw)["contract"]
            ),
            "escrow_sha256": file_sha256(draw / preparer.ESCROW_NAME),
            "pre_generation_freeze_sha256": file_sha256(
                draw / preparer.FREEZE_NAME
            ),
            "benchmark_manifest_sha256": file_sha256(
                draw / "benchmark/manifest.json"
            ),
        },
        "preserved_draw_sha256": preserved,
        "population_contract": {
            "eligibility_predicate": {
                "is_out_of_catalog": False,
                "calibration_population": "canonical_preserving",
                "filter_rows_before_deduplicating_pair_tokens": True,
            },
            "counts_by_split": population,
        },
        "origin_inventory": preparer.physical_tree_inventory(primary),
    }
    amendment_sha256 = preparer.canonical_json_sha256(amendment)
    config["attempt"]["recovery"]["amendment_sha256"] = amendment_sha256

    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    contract = deepcopy(preparer.read_escrow(draw)["contract"])
    contract.update(
        {
            "git_commit": "7" * 40,
            "config_sha256": file_sha256(config_path),
            "prospective_config": config,
        }
    )
    output = tmp_path / "primary"
    output.mkdir(mode=0o700)
    _write_json(output / "config.snapshot.json", config)
    _write_json(output / "source_bindings.json", contract["source_bindings"])
    origin_before = preparer.physical_tree_inventory(primary)
    context = {
        "amendment": amendment,
        "amendment_sha256": amendment_sha256,
        "amendment_path": amendment_relative,
        "implementation_commit": "8" * 40,
        "implementation_audit": {"audit_id": "R493"},
        "final_audit": config["final_audit"],
        "escrow_origin_contract_sha256": amendment["escrow_origin"][
            "contract_sha256"
        ],
        "failed_attempt": primary,
        "failed_attempt_basename": primary.name,
        "benchmark_manifest_sha256": amendment["escrow_origin"][
            "benchmark_manifest_sha256"
        ],
        "origin_inventory": origin_before,
        "reuse_source": draw,
        "repo_root": REPO_ROOT,
    }
    reused_escrow = preparer.read_escrow(draw)
    args = SimpleNamespace(
        wave51_dir=(
            REPO_ROOT / "data/geometria_proporcional/wave51_factored_smoke_v1"
        ),
        wave52_dir=REPO_ROOT / "data/geometria_proporcional/wave52_policy_transport_v1",
        wave54_dir=REPO_ROOT / "data/geometria_proporcional/wave54_joint_set_v1",
        replay_secrets_from=None,
        recovery_secrets_from=primary,
        recovery_amendment=tmp_path / "unused-amendment.json",
        reference_dir=None,
        force=False,
        attestation_private_key=wave60_runner.DEFAULT_PRIVATE_KEY,
    )

    def copy_pretruth_inference(
        destination_root: Path, *_args: object, **_kwargs: object
    ) -> dict[str, object]:
        shutil.copytree(draw / "inference", destination_root / "inference")
        hashes = preparer.inventory_hashes(destination_root / "inference")
        return {
            "staging_input_hashes": {},
            "runtime_hashes": {},
            "checkpoint_receipts": [],
            "inference_hashes": hashes,
            "effective_uid": 65534,
            "effective_gid": 65534,
            "negative_truth_probe": "PermissionError",
            "fit_operations": False,
        }

    materializer_module = sys.modules["run_wave59_hgb_guard_bracket"]
    real_materializer = materializer_module.materialize_prepared_bundles
    observed_materializer_configs: list[dict] = []

    def observe_real_materializer(
        destination_root: Path,
        materializer_config: dict,
        **kwargs: object,
    ) -> dict[str, str]:
        observed_materializer_configs.append(deepcopy(materializer_config))
        return real_materializer(destination_root, materializer_config, **kwargs)

    monkeypatch.setattr(preparer, "stage_and_infer", copy_pretruth_inference)
    monkeypatch.setattr(
        materializer_module,
        "materialize_prepared_bundles",
        observe_real_materializer,
    )
    preparer.run_preparation_transaction(
        args,
        output,
        config_path,
        config,
        "recovery",
        contract,
        reused_escrow,
        force=False,
        recovery_context=context,
    )
    budget = {
        "duration_seconds": 1.0,
        "cumulative_duration_seconds": 61.0,
        "prior_elapsed_seconds": 60.0,
        "max_rss_bytes": 0,
        "max_seconds": 840.0,
        "max_seconds_total": 900.0,
        "max_rss_allowed_bytes": 1610612736,
        "cuda_visible_devices": "",
        "budget_enforced": True,
    }
    preparer.finalize_preparation_budget_authority(
        output,
        config,
        "recovery",
        budget,
        wave60_runner.DEFAULT_PRIVATE_KEY,
    )

    assert len(observed_materializer_configs) == 1
    assert observed_materializer_configs[0]["hard_set_tau"] == 0.5
    assert "hard_set_tau" not in config
    assert load_json(config_path) == config
    expected_provenance = preparer.recovery_provenance(context, contract)
    assert expected_provenance["implementation_audit"] == {"audit_id": "R493"}
    assert "R483" not in json.dumps(expected_provenance, sort_keys=True)
    assert "R485" not in json.dumps(expected_provenance, sort_keys=True)
    assert "R489" not in json.dumps(expected_provenance, sort_keys=True)
    assert "R491" not in json.dumps(expected_provenance, sort_keys=True)
    for relative in (
        "generation_receipt.json",
        "preparation_freeze.json",
        "preparation_receipt.json",
    ):
        assert load_json(output / relative)["recovery_provenance"] == expected_provenance
    assert file_sha256(output / preparer.RECOVERY_AMENDMENT_COPY_NAME) == amendment_sha256

    for relative in preserved_relatives:
        source = draw / relative
        copied = output / relative
        assert source.read_bytes() == copied.read_bytes()
        assert (source.stat().st_dev, source.stat().st_ino) != (
            copied.stat().st_dev,
            copied.stat().st_ino,
        )
    attestation = load_json(output / preparer.WAVE59_PREPARATION_ATTESTATION_NAME)
    preparer.verify_attestation(
        {
            "algorithm": "Ed25519",
            "payload": attestation["payload"],
            "signature_base64": attestation["signature_base64"],
            "trusted_public_key_sha256": attestation["public_key_fingerprint"],
        },
        preparer.PUBLIC_KEY,
    )
    preparation_receipt = output / "preparation_receipt.json"
    assert attestation["payload"]["records"]["preparation_receipt.json"] == {
        "path": "preparation_receipt.json",
        "bytes": preparation_receipt.stat().st_size,
        "sha256": file_sha256(preparation_receipt),
    }
    replay_args = SimpleNamespace(
        replay_secrets_from=output,
        recovery_secrets_from=None,
    )
    assert preparer.wave60_prior_preparation_elapsed(
        replay_args, config, "replay"
    ) == 61.0
    assert preparer.physical_tree_inventory(primary) == origin_before
    assert not (output / "benchmark/sealed/oracle").exists()
    assert not (output / "authorized_labels").exists()


def test_invalid_preparation_plan_lineage_through_r480_is_exact() -> None:
    rejected = {
        "commit": "ba193d52cd46f23d57c1a0b811433d2cbcfedb6d",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
            "WAVE_60_INVALID_PREPARATION_RECOVERY_PLAN.md"
        ),
        "sha256": (
            "0ae881499221845d309e66dba9da42821b9e841002f8f8c4329790e87896965f"
        ),
    }
    preparer.validate_wave60_bound_document(
        REPO_ROOT,
        rejected,
        label="rejected plan",
        expected_parent="2a10b6cb5a88fd2af4ca2f5f8230a296f59c8948",
    )
    r478_path = preparer.validate_wave60_bound_document(
        REPO_ROOT,
        {
            "commit": "979c835bc2b2f08182e23e919f265f4fe2bc480a",
            "path": (
                "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
                "478_wave60_invalid_preparation_recovery_plan_audit.md"
            ),
            "sha256": (
                "ef4be8f02f2bf40f005c63c9cf61e5932ddd52bb36fb343aab15f457dceb1eb8"
            ),
        },
        label="R478",
        expected_parent=rejected["commit"],
    )
    preparer.parse_wave60_revise_audit_report(
        r478_path,
        audit_id="R478",
        scope="RECOVERY_PLAN",
        target={
            "plan_commit": rejected["commit"],
            "plan_sha256": rejected["sha256"],
        },
        findings={"high": 0, "medium": 2, "low": 0},
    )
    r478_plan = {
        "commit": "0c21db44ebb428647dbcd7921713b593895fc07a",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
            "WAVE_60_INVALID_PREPARATION_RECOVERY_R478_RESOLUTION_PLAN.md"
        ),
        "sha256": (
            "39f13cd749083ed581b967166e823ff66e872b9519013c3f1e66523ee58e9c8c"
        ),
    }
    preparer.validate_wave60_bound_document(
        REPO_ROOT,
        r478_plan,
        label="R478 resolution",
        expected_parent="979c835bc2b2f08182e23e919f265f4fe2bc480a",
    )
    r479_path = preparer.validate_wave60_bound_document(
        REPO_ROOT,
        {
            "commit": "3f20c79db2311cfefebc90dd049b4862d2a04a11",
            "path": (
                "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
                "479_wave60_invalid_preparation_recovery_r478_resolution_plan_audit.md"
            ),
            "sha256": (
                "4089ef7cd4812714fa731efae45719fc1685dafb767e1e92c26574abdddb9fa4"
            ),
        },
        label="R479",
        expected_parent=r478_plan["commit"],
    )
    preparer.parse_wave60_revise_audit_report(
        r479_path,
        audit_id="R479",
        scope="INVALID_PREPARATION_RECOVERY_R478_RESOLUTION_PLAN",
        target={
            "plan_commit": r478_plan["commit"],
            "plan_sha256": r478_plan["sha256"],
        },
        findings={"high": 0, "medium": 1, "low": 0},
    )
    r479_plan = {
        "commit": "305cdfcc47a0d5f537019a4a5fa8cf3ca271fd2c",
        "path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
            "WAVE_60_INVALID_PREPARATION_RECOVERY_R479_RESOLUTION_PLAN.md"
        ),
        "sha256": (
            "ec82136d826564effb50b848c61d674c950e9e83a3b6371496a24756efc92081"
        ),
    }
    preparer.validate_wave60_bound_document(
        REPO_ROOT,
        r479_plan,
        label="R479 resolution",
        expected_parent="3f20c79db2311cfefebc90dd049b4862d2a04a11",
    )
    preparer.validate_wave60_audit_commit(
        REPO_ROOT,
        {
            "audit_commit": "abb8fd2e8c9119e46fabee2aa15405ceb146d4b2",
            "audit_path": (
                "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
                "480_wave60_invalid_preparation_recovery_r479_resolution_plan_audit.md"
            ),
            "audit_sha256": (
                "d4860501a98acfe71659f49b0fbeb89c1c8fec7d3fcdd1fa4444664c7e6aaad3"
            ),
        },
        scope="INVALID_PREPARATION_RECOVERY_R479_RESOLUTION_PLAN",
        target={
            "plan_commit": r479_plan["commit"],
            "plan_sha256": r479_plan["sha256"],
        },
        expected_parent=r479_plan["commit"],
    )
    with pytest.raises(RuntimeError, match="directly descend"):
        preparer.validate_wave60_bound_document(
            REPO_ROOT,
            rejected,
            label="rejected plan",
            expected_parent=r478_plan["commit"],
        )


def test_invalid_preparation_nested_origin_and_unsigned_debit_are_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_relative = Path(
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v1"
    )
    prior = tmp_path / prior_relative
    shutil.copytree(REPO_ROOT / prior_relative, prior)
    primary = prior / "primary"
    draw = primary / "failed_preparation"
    config = load_json(primary / "config.snapshot.json")
    manifest = load_json(draw / "benchmark/manifest.json")
    preserved_relatives = [
        preparer.ESCROW_NAME,
        preparer.FREEZE_NAME,
        "benchmark/manifest.json",
        *(f"benchmark/{relative}" for relative in manifest["files"]),
    ]
    preserved = {
        relative: file_sha256(draw / relative) for relative in preserved_relatives
    }
    amendment = {
        "schema_version": (
            preparer.WAVE60_INVALID_PREPARATION_RECOVERY_AMENDMENT_SCHEMA
        ),
        "status": "APPROVED",
        "recovery_kind": "INVALID_PREPARATION",
        "prior_attempt_container": str(prior_relative),
        "prior_pair_failure_sha256": file_sha256(prior / "pair/FAILURE.json"),
        "unledgered_preparation_debit": {
            "seconds": 60.0,
            "regime": "CONSERVATIVE_UNSIGNED_PREPARATION_DEBIT",
            "observed_external_wall_seconds": 48.51,
            "observed_external_record_authority": (
                "TRANSCRIPT_ONLY_NOT_SIGNED_LEDGER"
            ),
            "applied_once": True,
        },
        "escrow_origin": {
            "contract_sha256": preparer.compact_json_sha256(
                preparer.read_escrow(draw)["contract"]
            ),
            "escrow_sha256": file_sha256(draw / preparer.ESCROW_NAME),
            "pre_generation_freeze_sha256": file_sha256(
                draw / preparer.FREEZE_NAME
            ),
            "benchmark_manifest_sha256": file_sha256(
                draw / "benchmark/manifest.json"
            ),
        },
        "preserved_draw_sha256": preserved,
        "origin_inventory": preparer.physical_tree_inventory(primary),
    }
    amendment_relative = Path(
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_INVALID_PREPARATION_RECOVERY_V2_AMENDMENT.json"
    )
    amendment_path = tmp_path / amendment_relative
    amendment_path.parent.mkdir(parents=True)
    amendment_path.write_text(
        json.dumps(amendment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config["attempt"] = {
        "version": 2,
        "container": (
            "data/geometria_proporcional/"
            "wave60_frozen_policy_transport_attempt_v2"
        ),
        "primary": "primary",
        "replay": "replay",
        "pair": "pair",
        "recovery": {
            "prior_attempt_container": str(prior_relative),
            "amendment_path": str(amendment_relative),
            "amendment_sha256": file_sha256(amendment_path),
            "preserved_draw_sha256": preserved,
        },
    }
    config["output_parent_relative"] = config["attempt"]["container"]
    config["primary_output"] = f"{config['attempt']['container']}/primary"
    config["replay_output"] = f"{config['attempt']['container']}/replay"
    args = SimpleNamespace(
        recovery_secrets_from=primary,
        replay_secrets_from=None,
        recovery_amendment=amendment_path,
        reference_dir=None,
        force=False,
    )
    monkeypatch.setattr(preparer, "REPO_ROOT", tmp_path)
    assert preparer.validate_invocation(
        args,
        tmp_path / config["primary_output"],
        config,
        repo_root=tmp_path,
    ) == "recovery"
    assert preparer.wave60_prior_preparation_elapsed(args, config, "recovery") == 60.0
    (prior / "replay/preparation_receipt.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="presence matrix|cannot mix with signed"):
        preparer.wave60_prior_preparation_elapsed(args, config, "recovery")


@pytest.mark.parametrize(
    "mutation",
    (
        "escrow_bytes",
        "freeze_bytes",
        "manifest_bytes",
        "benchmark_bytes",
        "draw_mode",
        "draw_owner",
        "sensitive_mode",
        "sensitive_owner",
        "symlink",
        "hardlink",
        "special_node",
        "extra_file_authorized_by_inventory",
        "primary_inventory",
        "primary_signature",
        "replay_inventory",
        "replay_signature",
        "pair_signature",
        "preserved_map_extra",
    ),
)
def test_invalid_preparation_nested_origin_rejects_physical_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    prior_relative = Path(
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v1"
    )
    prior = tmp_path / prior_relative
    shutil.copytree(REPO_ROOT / prior_relative, prior)
    config, amendment, primary, draw = invalid_preparation_origin_authority(
        tmp_path, prior_relative
    )

    if mutation == "escrow_bytes":
        path = draw / preparer.ESCROW_NAME
        path.write_bytes(path.read_bytes() + b"\n")
        amendment["origin_inventory"] = preparer.physical_tree_inventory(primary)
    elif mutation == "freeze_bytes":
        path = draw / preparer.FREEZE_NAME
        path.write_bytes(path.read_bytes() + b"\n")
        amendment["origin_inventory"] = preparer.physical_tree_inventory(primary)
    elif mutation == "manifest_bytes":
        path = draw / "benchmark/manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
        amendment["origin_inventory"] = preparer.physical_tree_inventory(primary)
    elif mutation == "benchmark_bytes":
        path = draw / "benchmark/visible/train.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        amendment["origin_inventory"] = preparer.physical_tree_inventory(primary)
    elif mutation == "draw_mode":
        draw.chmod(0o755)
        amendment["origin_inventory"] = preparer.physical_tree_inventory(primary)
    elif mutation == "draw_owner":
        os.chown(draw, 65534, 65534)
        amendment["origin_inventory"] = preparer.physical_tree_inventory(primary)
    elif mutation == "sensitive_mode":
        (draw / preparer.ESCROW_NAME).chmod(0o644)
        amendment["origin_inventory"] = preparer.physical_tree_inventory(primary)
    elif mutation == "sensitive_owner":
        os.chown(draw / preparer.ESCROW_NAME, 65534, 65534)
        amendment["origin_inventory"] = preparer.physical_tree_inventory(primary)
    elif mutation == "symlink":
        path = draw / "preparation_error.json"
        path.unlink()
        path.symlink_to("generation_receipt.json")
    elif mutation == "hardlink":
        path = draw / "preparation_error.json"
        path.unlink()
        path.hardlink_to(draw / "generation_receipt.json")
        amendment["origin_inventory"] = preparer.physical_tree_inventory(primary)
    elif mutation == "special_node":
        path = draw / "preparation_error.json"
        path.unlink()
        os.mkfifo(path)
    elif mutation == "extra_file_authorized_by_inventory":
        (draw / "extra.json").write_text("{}\n", encoding="utf-8")
        amendment["origin_inventory"] = preparer.physical_tree_inventory(primary)
    elif mutation == "primary_inventory":
        path = primary / "failure_inventory.json"
        path.write_bytes(path.read_bytes() + b"\n")
        amendment["origin_inventory"] = preparer.physical_tree_inventory(primary)
    elif mutation == "primary_signature":
        path = primary / "failure_attestation.json"
        path.write_bytes(path.read_bytes() + b"\n")
        amendment["origin_inventory"] = preparer.physical_tree_inventory(primary)
    elif mutation == "replay_inventory":
        path = prior / "replay/failure_inventory.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutation == "replay_signature":
        path = prior / "replay/failure_attestation.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutation == "pair_signature":
        path = prior / "pair/failure_attestation.json"
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        amendment["preserved_draw_sha256"]["extra.json"] = "0" * 64
        config["attempt"]["recovery"]["preserved_draw_sha256"] = amendment[
            "preserved_draw_sha256"
        ]

    with pytest.raises((RuntimeError, PermissionError, json.JSONDecodeError)):
        preparer._validate_wave60_invalid_preparation_recovery_origin(
            amendment,
            config,
            repo_root=tmp_path,
            trusted_public_key_path=preparer.PUBLIC_KEY,
        )


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("seconds", 0.0),
        ("seconds", -1.0),
        ("seconds", 59.0),
        ("applied_once", False),
        ("regime", "UNDECLARED"),
        ("observed_external_wall_seconds", 48.5),
        ("observed_external_record_authority", "SIGNED_LEDGER"),
        ("version", 3),
        ("prior_attempt_container", "wave60_frozen_policy_transport_attempt_v0"),
        ("source", "replay"),
        ("amendment_path", True),
        ("amendment_hash", True),
        ("primary_receipt_only", True),
        ("primary_attestation_only", True),
        ("replay_receipt_only", True),
        ("replay_attestation_only", True),
        ("primary_signed", True),
        ("replay_signed", True),
    ),
)
def test_invalid_preparation_unsigned_debit_rejects_each_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    value: object,
) -> None:
    prior_relative = Path(
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v1"
    )
    prior = tmp_path / prior_relative
    shutil.copytree(REPO_ROOT / prior_relative, prior)
    config, amendment, primary, _ = invalid_preparation_origin_authority(
        tmp_path, prior_relative
    )
    amendment_relative = Path("synthetic-amendment.json")
    amendment_path = tmp_path / amendment_relative
    source = primary
    if mutation in amendment["unledgered_preparation_debit"]:
        amendment["unledgered_preparation_debit"][mutation] = value
    elif mutation == "version":
        config["attempt"]["version"] = value
    elif mutation == "prior_attempt_container":
        config["attempt"]["recovery"]["prior_attempt_container"] = (
            f"data/geometria_proporcional/{value}"
        )
    elif mutation == "source":
        source = prior / str(value)
    elif mutation == "primary_receipt_only":
        (primary / "preparation_receipt.json").write_text("{}\n", encoding="utf-8")
    elif mutation == "primary_attestation_only":
        (primary / preparer.WAVE59_PREPARATION_ATTESTATION_NAME).write_text(
            "{}\n", encoding="utf-8"
        )
    elif mutation == "replay_receipt_only":
        (prior / "replay/preparation_receipt.json").write_text(
            "{}\n", encoding="utf-8"
        )
    elif mutation == "replay_attestation_only":
        (prior / "replay" / preparer.WAVE59_PREPARATION_ATTESTATION_NAME).write_text(
            "{}\n", encoding="utf-8"
        )
    elif mutation == "primary_signed":
        (primary / "preparation_receipt.json").write_text("{}\n", encoding="utf-8")
        (primary / preparer.WAVE59_PREPARATION_ATTESTATION_NAME).write_text(
            "{}\n", encoding="utf-8"
        )
    elif mutation == "replay_signed":
        (prior / "replay/preparation_receipt.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (prior / "replay" / preparer.WAVE59_PREPARATION_ATTESTATION_NAME).write_text(
            "{}\n", encoding="utf-8"
        )
    _write_json(amendment_path, amendment)
    config["attempt"]["recovery"]["amendment_path"] = amendment_relative.as_posix()
    config["attempt"]["recovery"]["amendment_sha256"] = file_sha256(
        amendment_path
    )
    if mutation == "amendment_path":
        alternative_path = tmp_path / "alternative-amendment.json"
        _write_json(alternative_path, amendment)
        config["attempt"]["recovery"]["amendment_path"] = alternative_path.name
        config["attempt"]["recovery"]["amendment_sha256"] = file_sha256(
            alternative_path
        )
    elif mutation == "amendment_hash":
        config["attempt"]["recovery"]["amendment_sha256"] = "0" * 64
    args = SimpleNamespace(recovery_amendment=amendment_path)
    monkeypatch.setattr(preparer, "REPO_ROOT", tmp_path)
    with pytest.raises(RuntimeError):
        preparer._wave60_invalid_preparation_unsigned_debit(
            args, config, source
        )


def test_invalid_preparation_unsigned_debit_flows_once_through_signed_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_material: dict,
) -> None:
    prior_relative = Path(
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v1"
    )
    prior = tmp_path / prior_relative
    shutil.copytree(REPO_ROOT / prior_relative, prior)
    config, amendment, origin_primary, _ = invalid_preparation_origin_authority(
        tmp_path, prior_relative
    )
    amendment_relative = Path("synthetic-amendment.json")
    amendment_path = tmp_path / amendment_relative
    _write_json(amendment_path, amendment)
    recovery = config["attempt"]["recovery"]
    recovery.update(
        {
            "schema_version": "wave60-pretruth-recovery-v1",
            "prior_pair_failure_sha256": amendment["prior_pair_failure_sha256"],
            "prior_config_audit_commit": "1" * 40,
            "prior_config_audit_path": "prior-config-audit.md",
            "prior_config_audit_sha256": "2" * 64,
            "amendment_path": amendment_relative.as_posix(),
            "amendment_sha256": file_sha256(amendment_path),
            "amendment_audit_commit": "3" * 40,
            "amendment_audit_path": "amendment-audit.md",
            "amendment_audit_sha256": "4" * 64,
        }
    )
    v2_relative = Path(
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v2"
    )
    config["attempt"]["container"] = v2_relative.as_posix()
    config["output_parent_relative"] = v2_relative.as_posix()
    config["primary_output"] = (v2_relative / "primary").as_posix()
    config["replay_output"] = (v2_relative / "replay").as_posix()
    preparer.validate_prospective_config(config)
    monkeypatch.setattr(preparer, "REPO_ROOT", tmp_path)

    unsigned_args = SimpleNamespace(
        replay_secrets_from=None,
        recovery_secrets_from=origin_primary,
        recovery_amendment=amendment_path,
    )
    assert preparer.wave60_prior_preparation_elapsed(
        unsigned_args, config, "recovery"
    ) == pytest.approx(60.0)

    def publish_signed_boundary(
        root: Path,
        *,
        execution_mode: str,
        prior_seconds: float,
        duration_seconds: float,
    ) -> None:
        _build_prepared_primary(
            root,
            config,
            source_material,
            recovery_amendment=amendment_path,
        )
        receipt = load_json(root / "preparation_receipt.json")
        receipt["execution_mode"] = execution_mode
        receipt["replay_exact"] = execution_mode == "replay"
        receipt["coordinator_budget"] = {
            "duration_seconds": duration_seconds,
            "cumulative_duration_seconds": prior_seconds + duration_seconds,
            "prior_elapsed_seconds": prior_seconds,
            "max_rss_bytes": 0,
            "max_seconds": 900.0 - prior_seconds,
            "max_seconds_total": 900.0,
            "max_rss_allowed_bytes": 1610612736,
            "cuda_visible_devices": "",
            "budget_enforced": True,
        }
        _write_json(root / "preparation_receipt.json", receipt)
        (root / preparer.WAVE59_PREPARATION_ATTESTATION_NAME).unlink()
        if execution_mode == "replay":
            _write_json(
                root / "preparation_replay.json",
                {"phase": "wave60-preparation-exact-replay", "all_exact": True},
            )
        preparer.publish_wave60_preparation_attestation(
            root,
            execution_mode,
            wave60_runner.DEFAULT_PRIVATE_KEY,
        )

    v2 = tmp_path / v2_relative
    primary = v2 / "primary"
    replay = v2 / "replay"
    publish_signed_boundary(
        primary,
        execution_mode="recovery",
        prior_seconds=60.0,
        duration_seconds=1.0,
    )
    replay_args = SimpleNamespace(
        replay_secrets_from=primary,
        recovery_secrets_from=None,
        recovery_amendment=amendment_path,
    )
    assert preparer.wave60_prior_preparation_elapsed(
        replay_args, config, "replay"
    ) == pytest.approx(61.0)
    publish_signed_boundary(
        replay,
        execution_mode="replay",
        prior_seconds=61.0,
        duration_seconds=2.0,
    )
    assert wave60_runner.pair_preparation_elapsed(primary, replay) == pytest.approx(
        63.0
    )

    replay_error = RuntimeError("synthetic replay failure after signed preparation")
    wave60_runner.write_failure_journal(
        replay,
        "source_bind",
        replay_error,
        input_sha256="6" * 64,
        truth_accessed=False,
        duration_seconds=0.5,
    )
    replay_binding = seal_root_failure(
        replay,
        terminal="SOURCE_BINDING_FAILED_PRE_TRUTH",
        phase="source_bind",
        role="replay",
        truth_accessed=False,
        error=replay_error,
        authority_binding_sha256=file_sha256(replay / "config.snapshot.json"),
        last_complete_phase="PREPARED",
    )
    primary_binding = seal_root_failure(
        primary,
        terminal="PEER_ABORTED_PRE_TRUTH",
        phase="peer_abort",
        role="primary",
        truth_accessed=False,
        error=RuntimeError("synthetic peer abort after signed preparation"),
        authority_binding_sha256=file_sha256(primary / "config.snapshot.json"),
        last_complete_phase="PREPARED",
        peer_terminal="SOURCE_BINDING_FAILED_PRE_TRUTH",
        peer_terminal_binding_sha256=replay_binding,
    )
    status = pair_status(
        "PEER_ABORTED_PRE_TRUTH",
        "SOURCE_BINDING_FAILED_PRE_TRUTH",
        primary_binding,
        replay_binding,
        any_truth_accessed=False,
    )
    publish_pair_failure(
        v2,
        status,
        error=RuntimeError("synthetic signed pair failure"),
        private_key=wave60_runner.DEFAULT_PRIVATE_KEY,
    )
    durable = wave60_runner.recovery_pair_durable_elapsed(
        v2,
        public_key=preparer.PUBLIC_KEY,
    )
    assert durable == pytest.approx(
        {
            "preparation_seconds": 63.0,
            "source_binding_seconds": 0.5,
            "score_apply_seconds": 0.0,
            "phase_seconds": 0.5,
            "durable_seconds": 63.5,
        }
    )

    v3 = deepcopy(config)
    v3["attempt"]["version"] = 3
    v3["attempt"]["container"] = (
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v3"
    )
    v3["attempt"]["recovery"]["prior_attempt_container"] = v2_relative.as_posix()
    v3_args = SimpleNamespace(
        replay_secrets_from=None,
        recovery_secrets_from=primary,
        recovery_amendment=amendment_path,
    )
    with monkeypatch.context() as debit_guard:
        debit_guard.setattr(
            preparer,
            "_wave60_invalid_preparation_unsigned_debit",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("unsigned debit was applied more than once")
            ),
        )
        assert preparer.wave60_prior_preparation_elapsed(
            v3_args, v3, "recovery"
        ) == pytest.approx(63.5)


@pytest.mark.parametrize("field", sorted(SOURCE_LAW_RECOVERY_BINDING))
def test_source_law_recovery_request_rejects_each_binding_drift(field: str) -> None:
    request = recovery_source_request()
    value = request["recovery"][field]
    if isinstance(value, bool):
        request["recovery"][field] = not value
    else:
        request["recovery"][field] = f"{value}-drift"
    with pytest.raises(RuntimeError, match="recovery request drifted"):
        validate_source_law_recovery_request(request)


@pytest.mark.parametrize(
    "relative",
    (
        "source_law_request.json",
        "journals/verify_source_law.json",
        "FAILURE.json",
        "failure_inventory.json",
        "failure_attestation.json",
    ),
)
def test_prior_source_law_failure_rejects_each_file_tamper(
    relative: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".wave60-prior-failure-tamper-", dir=REPO_ROOT
    ) as raw:
        workspace = Path(raw)
        copied = workspace / "prior"
        shutil.copytree(PRIOR_SOURCE_AUTHORITY, copied)
        binding = deepcopy(SOURCE_LAW_RECOVERY_BINDING)
        binding["prior_authority_path"] = str(copied.relative_to(REPO_ROOT))
        monkeypatch.setattr(wave60_runner, "SOURCE_LAW_RECOVERY_BINDING", binding)
        target = copied / relative
        target.chmod(0o644)
        target.write_bytes(target.read_bytes() + b"\n")
        target.chmod(0o444)
        with pytest.raises(RuntimeError, match="hash drifted"):
            validate_prior_source_law_failure(binding, copied)


@pytest.mark.parametrize(
    "mutation",
    (
        "mode",
        "special_mode",
        "owner",
        "symlink",
        "root_symlink",
        "hardlink",
        "extra",
    ),
)
def test_prior_source_law_failure_rejects_physical_shape_drift(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".wave60-prior-failure-shape-", dir=REPO_ROOT
    ) as raw:
        workspace = Path(raw)
        copied = workspace / "prior"
        shutil.copytree(PRIOR_SOURCE_AUTHORITY, copied)
        prior_argument = copied
        binding = deepcopy(SOURCE_LAW_RECOVERY_BINDING)
        target = copied / "FAILURE.json"
        if mutation == "mode":
            target.chmod(0o644)
        elif mutation == "special_mode":
            target.chmod(0o2444)
        elif mutation == "owner":
            os.chown(target, 65534, 65534)
        elif mutation == "symlink":
            target.unlink()
            target.symlink_to(copied / "source_law_request.json")
        elif mutation == "root_symlink":
            prior_argument = workspace / "prior-alias"
            prior_argument.symlink_to(copied, target_is_directory=True)
        elif mutation == "hardlink":
            target.unlink()
            target.hardlink_to(copied / "source_law_request.json")
        else:
            extra = copied / "extra.json"
            extra.write_text("{}\n", encoding="utf-8")
            extra.chmod(0o444)
        binding["prior_authority_path"] = str(
            prior_argument.relative_to(REPO_ROOT)
        )
        monkeypatch.setattr(wave60_runner, "SOURCE_LAW_RECOVERY_BINDING", binding)
        with pytest.raises(RuntimeError):
            validate_prior_source_law_failure(binding, prior_argument)


def test_source_law_recovery_semantic_failure_seals_nonrecoverable_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_before = {
        str(path.relative_to(PRIOR_SOURCE_AUTHORITY)): file_sha256(path)
        for path in PRIOR_SOURCE_AUTHORITY.rglob("*")
        if path.is_file()
    }
    with tempfile.TemporaryDirectory(
        prefix=".wave60-source-recovery-failure-", dir=REPO_ROOT
    ) as raw:
        workspace = Path(raw)
        output = workspace / "source-law-v2"
        request_path = workspace / "request.json"
        request_path.write_text(
            json.dumps(recovery_source_request(output), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(wave60_runner, "SOURCE_AUTHORITY_DEFAULT", output)

        def reject_prior(*_args: object, **_kwargs: object) -> float:
            raise RuntimeError("injected prior authority mismatch")

        monkeypatch.setattr(
            wave60_runner, "validate_prior_source_law_failure", reject_prior
        )
        published = publish_source_law_recovery(
            request_path, Path(output.relative_to(REPO_ROOT))
        )
        assert published == output
        failure = load_json(output / "FAILURE.json")
        assert failure["terminal"] == "SOURCE_LAW_INVALID"
        assert failure["truth_accessed"] is False
        assert failure["recovery_allowed"] is False
        assert not (output / "source_authority_manifest.json").exists()
        assert not output.with_name(output.name + ".initializing").exists()
    prior_after = {
        str(path.relative_to(PRIOR_SOURCE_AUTHORITY)): file_sha256(path)
        for path in PRIOR_SOURCE_AUTHORITY.rglob("*")
        if path.is_file()
    }
    assert prior_after == prior_before


@pytest.mark.parametrize("entrypoint", ("api_relative", "api_absolute", "cli_relative"))
def test_source_law_recovery_publisher_positive_end_to_end(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canonical_prior_before = {
        str(path.relative_to(PRIOR_SOURCE_AUTHORITY)): file_sha256(path)
        for path in PRIOR_SOURCE_AUTHORITY.rglob("*")
        if path.is_file()
    }
    with tempfile.TemporaryDirectory(
        prefix=f".wave60-recovery-positive-{entrypoint}-", dir=REPO_ROOT.parent
    ) as raw:
        workspace = Path(raw)
        repo = workspace / "repo"
        subprocess.run(
            ["git", "clone", "-q", "--shared", str(REPO_ROOT), str(repo)],
            check=True,
        )
        (repo / "venv").symlink_to(REPO_ROOT / "venv", target_is_directory=True)
        original_aliases = dict(SOURCE_ALIASES)
        isolated_aliases: dict[str, Path] = {}
        for alias, source in original_aliases.items():
            destination = repo / source.relative_to(REPO_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            isolated_aliases[alias] = destination
            monkeypatch.setitem(wave60_runner.SOURCE_ALIASES, alias, destination)
        isolated_prior = (
            repo
            / "data/geometria_proporcional/"
            "wave60_frozen_policy_transport_source_law_v1"
        )
        shutil.copytree(PRIOR_SOURCE_AUTHORITY, isolated_prior)
        output = (
            repo
            / "data/geometria_proporcional/"
            "wave60_frozen_policy_transport_source_law_v2"
        )
        attempt = (
            repo
            / "data/geometria_proporcional/"
            "wave60_frozen_policy_transport_attempt_v1"
        )
        monkeypatch.setattr(wave60_runner, "REPO_ROOT", repo)
        monkeypatch.setattr(wave60_runner, "SOURCE_AUTHORITY_DEFAULT", output)
        monkeypatch.setattr(
            wave60_runner, "PRIOR_SOURCE_AUTHORITY", isolated_prior
        )
        monkeypatch.setattr(wave60_runner, "ATTEMPT_DEFAULT", attempt)
        request = {
            "schema_version": SOURCE_LAW_RECOVERY_REQUEST_SCHEMA,
            "plan_commit": PLAN_COMMIT,
            "plan_sha256": PLAN_SHA256,
            "implementation_commit": "1" * 40,
            "implementation_audit_commit": "2" * 40,
            "implementation_audit_sha256": "3" * 64,
            "source_paths": {
                name: str(path.relative_to(repo))
                for name, path in isolated_aliases.items()
            },
            "source_sha256": dict(SOURCE_HASHES),
            "output_path": str(output.relative_to(repo)),
            "runtime_budget": {
                "max_seconds": 900,
                "max_rss_bytes": 1610612736,
            },
            "recovery": dict(SOURCE_LAW_RECOVERY_BINDING),
        }
        request_path = repo / "source_law_recovery_request.json"
        request_path.write_text(
            json.dumps(request, sort_keys=True) + "\n", encoding="utf-8"
        )
        audit_scopes: list[str] = []

        def accept_exact_audit(
            implementation_commit: str,
            audit_commit: str,
            audit_sha256: str,
            *,
            expected_scope: str,
        ) -> Path:
            audit_scopes.append(expected_scope)
            if implementation_commit == request["implementation_commit"]:
                assert audit_commit == request["implementation_audit_commit"]
                assert audit_sha256 == request["implementation_audit_sha256"]
                assert expected_scope == SOURCE_LAW_RECOVERY_IMPLEMENTATION_SCOPE
            else:
                assert expected_scope == "IMPLEMENTATION"
            return workspace / "injected-audit-boundary"

        def accept_exact_lineage(implementation_commit: str) -> None:
            assert implementation_commit == request["implementation_commit"]

        monkeypatch.setattr(
            wave60_runner,
            "validate_implementation_audit_authority",
            accept_exact_audit,
        )
        monkeypatch.setattr(
            wave60_runner,
            "validate_recovery_implementation_lineage",
            accept_exact_lineage,
        )
        if entrypoint == "api_relative":
            published = publish_source_law_recovery(
                request_path.relative_to(repo), output.relative_to(repo)
            )
        elif entrypoint == "api_absolute":
            published = publish_source_law_recovery(request_path, output)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "run_wave60_frozen_policy_transport.py",
                    "verify-source-law",
                    "--request",
                    str(request_path.relative_to(repo)),
                    "--output",
                    str(output.relative_to(repo)),
                ],
            )
            wave60_runner.main()
            emitted = json.loads(capsys.readouterr().out)
            assert emitted == {
                "status": "SOURCE_LAW_VERIFIED",
                "path": str(output),
            }
            published = output
        assert published == output
        assert not output.with_name(output.name + ".initializing").exists()
        assert not (output / "FAILURE.json").exists()
        assert file_sha256(output / "source_law_request.json") == file_sha256(
            request_path
        )
        journal = load_json(output / "journals/verify_source_law.json")
        assert journal["schema_version"] == "wave60-source-law-recovery-journal-v1"
        assert journal["status"] == "SOURCE_LAW_VERIFIED"
        assert journal["prior_durable_elapsed_seconds"] == 0.00401783362030983
        assert journal["cumulative_duration_seconds"] == (
            journal["prior_durable_elapsed_seconds"] + journal["duration_seconds"]
        )
        assert journal["cumulative_duration_seconds"] < 900
        assert journal["max_rss_bytes"] < 1610612736
        assert journal["truth_accessed"] is False
        receipt = load_json(output / "verify_source_law_receipt.json")
        assert receipt["uid"] == receipt["gid"] == 65534
        assert all(row["denied"] is True for row in receipt["denied_path_probes"])
        manifest = load_json(output / "source_authority_manifest.json")
        observed = wave60_runner.inventory(output)
        observed.pop("source_authority_manifest.json")
        assert manifest["terminal"] == "SOURCE_LAW_VERIFIED"
        assert manifest["files"] == observed
        binding = {
            "path": str(output.relative_to(repo)),
            "source_law_freeze_sha256": file_sha256(
                output / "source_law_freeze.json"
            ),
            "source_law_attestation_sha256": file_sha256(
                output / "source_law_attestation.json"
            ),
            "transport_law_manifest_sha256": file_sha256(
                output / "transport_law_manifest.json"
            ),
            "transport_law_arrays_sha256": file_sha256(
                output / "transport_law_arrays.npz"
            ),
            "frozen_policy_spec_sha256": file_sha256(
                output / "frozen_policy_spec.json"
            ),
            "feature_schema_sha256": file_sha256(output / "feature_schema.json"),
            "source_authority_manifest_sha256": file_sha256(
                output / "source_authority_manifest.json"
            ),
        }
        config = valid_config()
        config["implementation_binding"].update(
            {
                "commit": request["implementation_commit"],
                "audit_commit": request["implementation_audit_commit"],
                "audit_sha256": request["implementation_audit_sha256"],
            }
        )
        wave60_runner.validate_source_authority(output, binding, config)
        assert SOURCE_LAW_RECOVERY_IMPLEMENTATION_SCOPE in audit_scopes
        assert "IMPLEMENTATION" in audit_scopes
        assert not attempt.exists()
        assert not attempt.with_name(attempt.name + ".initializing").exists()
    canonical_prior_after = {
        str(path.relative_to(PRIOR_SOURCE_AUTHORITY)): file_sha256(path)
        for path in PRIOR_SOURCE_AUTHORITY.rglob("*")
        if path.is_file()
    }
    assert canonical_prior_after == canonical_prior_before


def test_source_law_namespace_rejections_do_not_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-authority"
    with pytest.raises(RuntimeError, match="outside the repository"):
        canonical_source_output_path(outside)
    assert not outside.exists()

    with tempfile.TemporaryDirectory(
        prefix=".wave60-source-namespace-test-", dir=REPO_ROOT
    ) as raw:
        workspace = Path(raw)
        request = workspace / "request.json"
        request.write_text("{}\n", encoding="utf-8")

        traversal = Path("data/geometria_proporcional/../unsafe-source-law")
        with pytest.raises(RuntimeError, match="traversal"):
            publish_source_law_authority(request, traversal)
        assert not (REPO_ROOT / "data/unsafe-source-law").exists()

        real = workspace / "real"
        real.mkdir()
        alias = workspace / "alias"
        alias.symlink_to(real, target_is_directory=True)
        with pytest.raises(RuntimeError, match="symlink"):
            publish_source_law_authority(request, alias / "authority")
        assert not (real / "authority").exists()

        existing = workspace / "existing"
        existing.mkdir()
        sentinel = existing / "sentinel"
        sentinel.write_text("preserve\n", encoding="utf-8")
        with pytest.raises(FileExistsError):
            publish_source_law_authority(request, existing)
        assert sentinel.read_text(encoding="utf-8") == "preserve\n"

        output = workspace / "reserved"
        staging = output.with_name(output.name + ".initializing")
        staging.mkdir()
        sentinel = staging / "sentinel"
        sentinel.write_text("preserve\n", encoding="utf-8")
        with pytest.raises(FileExistsError):
            publish_source_law_authority(request, output)
        assert not output.exists()
        assert sentinel.read_text(encoding="utf-8") == "preserve\n"


@pytest.mark.parametrize("absolute", (False, True))
def test_recovery_output_drift_and_legacy_dispatch_reject_without_writes(
    absolute: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".wave60-recovery-namespace-test-", dir=REPO_ROOT
    ) as raw:
        workspace = Path(raw)
        canonical = workspace / "canonical-v2"
        wrong = workspace / "wrong-v2"
        monkeypatch.setattr(wave60_runner, "SOURCE_AUTHORITY_DEFAULT", canonical)
        request = workspace / "request.json"
        request.write_text(
            json.dumps(recovery_source_request(canonical), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        argument = wrong if absolute else wrong.relative_to(REPO_ROOT)
        with pytest.raises(RuntimeError, match="canonical v2 root"):
            publish_source_law_recovery(request, argument)
        assert not wrong.exists()
        assert not wrong.with_name(wrong.name + ".initializing").exists()
        with pytest.raises(RuntimeError, match="requires the recovery publisher"):
            publish_source_law_authority(request, wrong)
        assert not wrong.exists()
        assert not wrong.with_name(wrong.name + ".initializing").exists()


def test_wave60_config_branch_is_typed_without_changing_older_dispatch() -> None:
    config = valid_config()
    validate_pre_draw_config(config)
    preparer.validate_prospective_config(config)
    assert preparer.preparation_phase_prefix(config) == "wave60"
    assert (
        preparer.preparation_phase_prefix(
            {"schema_version": preparer.WAVE59_CONFIG_SCHEMA}
        )
        == "wave59"
    )
    counts = {
        split: {
            "rows": 4992,
            "total_unique_pair_tokens": 1152,
            "eligible_unique_pair_tokens": 768,
        }
        for split in ("train", "val", "lockbox")
    }
    assert (
        preparer.validate_wave59_population_contract(
            config, counts, recovery_context=None
        )
        == "eligible_unique_pair_tokens"
    )


def test_recovery_v2_requires_new_namespace_and_bound_amendment() -> None:
    config = valid_config()
    recovery = {
        "schema_version": "wave60-pretruth-recovery-v1",
        "prior_attempt_container": (
            "data/geometria_proporcional/" "wave60_frozen_policy_transport_attempt_v1"
        ),
        "prior_pair_failure_sha256": "1" * 64,
        "prior_config_audit_commit": "a" * 40,
        "prior_config_audit_path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/"
            "agent_reports/998_wave60_v1_config_audit.md"
        ),
        "prior_config_audit_sha256": "b" * 64,
        "amendment_path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/"
            "waves/WAVE_60_RECOVERY_V2_AMENDMENT.md"
        ),
        "amendment_sha256": "2" * 64,
        "amendment_audit_commit": "3" * 40,
        "amendment_audit_path": (
            "Biblioteca/Geometria_Proporcional_Ground_Truth/"
            "agent_reports/000_wave60_recovery_audit.md"
        ),
        "amendment_audit_sha256": "4" * 64,
        "preserved_draw_sha256": {
            "generation_escrow.json": "5" * 64,
            "benchmark/manifest.json": "6" * 64,
        },
    }
    config["attempt"] = {
        "version": 2,
        "container": (
            "data/geometria_proporcional/" "wave60_frozen_policy_transport_attempt_v2"
        ),
        "primary": "primary",
        "replay": "replay",
        "pair": "pair",
        "recovery": recovery,
    }
    config["primary_output"] = f"{config['attempt']['container']}/primary"
    config["replay_output"] = f"{config['attempt']['container']}/replay"
    config["output_parent_relative"] = config["attempt"]["container"]
    validate_pre_draw_config(config)
    traversal = deepcopy(config)
    traversal["attempt"]["recovery"]["prior_attempt_container"] = (
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v1/../../../escaped/"
        "wave60_frozen_policy_transport_attempt_v9"
    )
    with pytest.raises(RuntimeError, match="recovery authority"):
        validate_pre_draw_config(traversal)
    future = deepcopy(config)
    future["attempt"]["recovery"]["prior_attempt_container"] = (
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v3"
    )
    with pytest.raises(RuntimeError, match="recovery authority"):
        validate_pre_draw_config(future)
    config["attempt"]["container"] = recovery["prior_attempt_container"]
    with pytest.raises(RuntimeError, match="namespace"):
        validate_pre_draw_config(config)


def test_recovery_prior_attempt_rejects_symlink_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    namespace = repo / "data/geometria_proporcional"
    namespace.mkdir(parents=True)
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    relative = (
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v1"
    )
    (repo / relative).symlink_to(escaped, target_is_directory=True)
    with pytest.raises(RuntimeError, match="escaped its namespace"):
        preparer.require_canonical_repo_directory(
            repo, relative, "Wave 60 prior attempt"
        )


def test_recovery_v2_and_v3_execute_with_cumulative_signed_ledgers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_material: dict,
    source_authority: dict,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "wave60@test.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Wave 60 Test"], cwd=repo, check=True
    )

    def commit_all(message: str) -> str:
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", message], cwd=repo, check=True)
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()

    def write_audit(
        relative: str,
        *,
        audit_id: str,
        scope: str,
        target: dict[str, str],
    ) -> None:
        authority = {
            "schema_version": "wave60-audit-authority-v1",
            "audit_id": audit_id,
            "scope": scope,
            "target": target,
            "technical_verdict": "PASS",
            "findings": {"high": 0, "medium": 0, "low": 0},
            "files_modified": False,
            "gpu_used_or_queried": False,
        }
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# Independent audit\n\n```json\n"
            + json.dumps(authority, sort_keys=True)
            + "\n```\n",
            encoding="utf-8",
        )

    prior_container_relative = (
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v1"
    )
    prior = repo / prior_container_relative
    v1 = valid_config()
    v1["source_binding"] = {
        "wave50_generation_key_commitment": "historical-generation",
        "wave50_visible_val_sha256": "f" * 64,
    }
    v1["source_law_authority"].update(source_authority["binding"])
    implementation_sources = {
        "src/geometria_proporcional/wave60_frozen_policy_transport.py",
        "experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py",
        "experiments/geometria_proporcional/_wave60_phase_worker.py",
        "experiments/geometria_proporcional/prepare_wave56_fresh.py",
        "tests/test_wave60_frozen_policy_transport.py",
    }
    (repo / ".gitignore").write_text("data/\n", encoding="utf-8")
    for relative in implementation_sources:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"audited:{relative}\n", encoding="utf-8")
    implementation_commit = commit_all("implementation")
    source_audit_path = repo / v1["source_law_authority"]["audit_path"]
    source_audit_path.parent.mkdir(parents=True, exist_ok=True)
    source_audit_path.write_text("source law accepted\n", encoding="utf-8")
    source_audit_commit = commit_all("source law audit")
    v1["implementation_binding"]["commit"] = implementation_commit
    v1["source_law_authority"]["audit_commit"] = source_audit_commit
    v1["source_sha256"].update(
        {
            relative: file_sha256(repo / relative)
            for relative in implementation_sources
        }
    )
    v1_audit_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "998_wave60_v1_config_audit.md"
    )
    v1["final_audit"] = {
        "audit_id": "R998",
        "audit_path": v1_audit_relative,
    }
    config_path = (
        repo
        / "experiments/geometria_proporcional/configs/"
        "wave60_frozen_policy_transport.json"
    )
    _write_json(config_path, v1)
    v1_config_commit = commit_all("v1 config")
    v1_config_sha256 = file_sha256(config_path)
    write_audit(
        v1_audit_relative,
        audit_id="R998",
        scope="CONFIG",
        target={
            "config_commit": v1_config_commit,
            "config_sha256": v1_config_sha256,
        },
    )
    v1_audit_commit = commit_all("v1 config audit")
    v1_audit_sha256 = file_sha256(repo / v1_audit_relative)
    origin_contract = {
        "git_commit": v1_audit_commit,
        "config_sha256": v1_config_sha256,
        "prospective_config": v1,
        "sources": dict(v1["source_sha256"]),
        "upstream": [],
        "historical_preflight": {},
        "source_bindings": v1["source_binding"],
    }
    _build_prepared_pair(prior, v1, source_material)
    primary_origin = prior / "primary"
    replay_origin = prior / "replay"
    keys = (b"g" * 32, b"i" * 32, b"s" * 32)
    escrow = preparer.make_escrow(origin_contract, keys)
    _write_json(primary_origin / "generation_escrow.json", escrow)
    (primary_origin / "generation_escrow.json").chmod(0o600)
    _write_json(
        primary_origin / "pre_generation_freeze.json",
        preparer.public_freeze_from_escrow(escrow),
    )
    (primary_origin / "pre_generation_freeze.json").chmod(0o644)
    benchmark = primary_origin / "benchmark"
    for name, key in zip(preparer.SECRET_FILES, keys, strict=True):
        _write_json(benchmark / "sealed" / name, {"key_hex": key.hex()})
    manifest = load_json(benchmark / "manifest.json")
    manifest.update(
        {
            "generation_key_commitment": preparer.sha256_bytes(keys[0]),
            "identity_key_commitment": preparer.sha256_bytes(keys[1]),
            "semantic_commitment_key_commitment": preparer.sha256_bytes(keys[2]),
            "counts": {split: 4992 for split in ("train", "val", "lockbox")},
        }
    )
    manifest["files"] = {
        str(path.relative_to(benchmark)): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(benchmark.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(benchmark / "manifest.json", manifest)
    preparer.publish_wave60_preparation_attestation(
        primary_origin, "primary", wave60_runner.DEFAULT_PRIVATE_KEY
    )
    primary_binding = seal_root_failure(
        primary_origin,
        terminal="INVALID_NEW_DRAW_IDENTITY",
        phase="new_draw_identity",
        role="primary",
        truth_accessed=False,
        error=RuntimeError("new draw identity collision"),
        authority_binding_sha256=file_sha256(
            primary_origin / "config.snapshot.json"
        ),
        last_complete_phase="PREPARED",
    )
    replay_binding = seal_root_failure(
        replay_origin,
        terminal="PEER_ABORTED_PRE_TRUTH",
        phase="peer_abort",
        role="replay",
        truth_accessed=False,
        error=RuntimeError("peer draw identity failed"),
        authority_binding_sha256=file_sha256(
            replay_origin / "config.snapshot.json"
        ),
        last_complete_phase="PREPARED",
        peer_terminal="INVALID_NEW_DRAW_IDENTITY",
        peer_terminal_binding_sha256=primary_binding,
    )
    status = pair_status(
        "INVALID_NEW_DRAW_IDENTITY",
        "PEER_ABORTED_PRE_TRUTH",
        primary_binding,
        replay_binding,
        any_truth_accessed=False,
    )
    pair = publish_pair_failure(
        prior,
        status,
        error=RuntimeError("pre-truth abort"),
        private_key=wave60_runner.DEFAULT_PRIVATE_KEY,
    )
    preserved = {
        "generation_escrow.json": file_sha256(
            primary_origin / "generation_escrow.json"
        ),
        "pre_generation_freeze.json": file_sha256(
            primary_origin / "pre_generation_freeze.json"
        ),
        "benchmark/manifest.json": file_sha256(benchmark / "manifest.json"),
        **{
            f"benchmark/{relative}": record["sha256"]
            for relative, record in manifest["files"].items()
        },
    }
    counts = {
        "rows": 4992,
        "total_unique_pair_tokens": 1152,
        "eligible_unique_pair_tokens": 768,
        "out_of_catalog_unique_pair_tokens": 384,
        "noncanonical_unique_pair_tokens": 192,
        "eligible_intersection_noncanonical_unique_pair_tokens": 192,
    }
    amendment_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_RECOVERY_V2_AMENDMENT.json"
    )
    amendment = {
        "schema_version": "wave60-pretruth-recovery-amendment-v1",
        "status": "APPROVED",
        "prior_attempt_container": prior_container_relative,
        "prior_pair_failure_sha256": file_sha256(pair / "FAILURE.json"),
        "prior_config_audit": {
            "commit": v1_audit_commit,
            "path": v1_audit_relative,
            "sha256": v1_audit_sha256,
        },
        "escrow_origin": {
            "contract_sha256": preparer.compact_json_sha256(origin_contract),
            "escrow_sha256": preserved["generation_escrow.json"],
            "pre_generation_freeze_sha256": preserved[
                "pre_generation_freeze.json"
            ],
            "benchmark_manifest_sha256": preserved["benchmark/manifest.json"],
        },
        "preserved_draw_sha256": preserved,
        "population_contract": {
            "eligibility_predicate": {
                "is_out_of_catalog": False,
                "calibration_population": "canonical_preserving",
                "filter_rows_before_deduplicating_pair_tokens": True,
            },
            "counts_by_split": {
                split: counts for split in ("train", "val", "lockbox")
            },
        },
        "origin_inventory": preparer.physical_tree_inventory(primary_origin),
    }
    amendment_path = repo / amendment_relative
    _write_json(amendment_path, amendment)
    amendment_sha256 = file_sha256(amendment_path)
    commit_all("v2 recovery amendment")
    amendment_audit_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "997_wave60_recovery_audit.md"
    )
    write_audit(
        amendment_audit_relative,
        audit_id="R997",
        scope="RECOVERY_AMENDMENT",
        target={"amendment_sha256": amendment_sha256},
    )
    amendment_audit_commit = commit_all("v2 recovery amendment audit")
    amendment_audit_sha256 = file_sha256(repo / amendment_audit_relative)
    v2 = deepcopy(v1)
    v2_container_relative = (
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v2"
    )
    v2["attempt"] = {
        "version": 2,
        "container": v2_container_relative,
        "primary": "primary",
        "replay": "replay",
        "pair": "pair",
        "recovery": {
            "schema_version": "wave60-pretruth-recovery-v1",
            "prior_attempt_container": prior_container_relative,
            "prior_pair_failure_sha256": amendment["prior_pair_failure_sha256"],
            "prior_config_audit_commit": v1_audit_commit,
            "prior_config_audit_path": amendment["prior_config_audit"]["path"],
            "prior_config_audit_sha256": v1_audit_sha256,
            "amendment_path": amendment_relative,
            "amendment_sha256": amendment_sha256,
            "amendment_audit_commit": amendment_audit_commit,
            "amendment_audit_path": amendment_audit_relative,
            "amendment_audit_sha256": amendment_audit_sha256,
            "preserved_draw_sha256": preserved,
        },
    }
    v2_audit_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "996_wave60_v2_config_audit.md"
    )
    v2["final_audit"] = {
        "audit_id": "R996",
        "audit_path": v2_audit_relative,
    }
    v2["primary_output"] = f"{v2_container_relative}/primary"
    v2["replay_output"] = f"{v2_container_relative}/replay"
    v2["output_parent_relative"] = v2_container_relative
    validate_pre_draw_config(v2)
    _write_json(config_path, v2)
    v2_config_commit = commit_all("v2 config")
    v2_config_sha256 = file_sha256(config_path)
    write_audit(
        v2_audit_relative,
        audit_id="R996",
        scope="CONFIG",
        target={
            "config_commit": v2_config_commit,
            "config_sha256": v2_config_sha256,
        },
    )
    v2_audit_commit = commit_all("v2 config audit")
    preparer.validate_wave60_final_config_authority(
        repo, config_path, v2, v2_audit_commit
    )
    execution_contract = {
        **origin_contract,
        "git_commit": v2_audit_commit,
        "config_sha256": v2_config_sha256,
        "prospective_config": v2,
        "sources": dict(v2["source_sha256"]),
    }
    v2_container = repo / v2_container_relative
    v2_container.mkdir(parents=True)
    wave51_dir = repo / "wave51"
    wave52_dir = repo / "wave52"
    wave54_dir = repo / "wave54"
    for source_dir in (wave51_dir, wave52_dir, wave54_dir):
        source_dir.mkdir()
    primary_output = v2_container / "primary"
    primary_args = SimpleNamespace(
        wave51_dir=wave51_dir,
        wave52_dir=wave52_dir,
        wave54_dir=wave54_dir,
        replay_secrets_from=None,
        recovery_secrets_from=primary_origin,
        recovery_amendment=amendment_path,
        reference_dir=None,
        force=False,
        attestation_private_key=wave60_runner.DEFAULT_PRIVATE_KEY,
    )
    rogue_source = repo / "rogue-recovery-source"
    rogue_source.mkdir()
    rogue_args = deepcopy(primary_args)
    rogue_args.recovery_secrets_from = rogue_source
    with pytest.raises(ValueError, match="recovery escrow must come"):
        preparer.validate_invocation(
            rogue_args, primary_output, v2, repo_root=repo
        )
    assert (
        preparer.validate_invocation(
            primary_args, primary_output, v2, repo_root=repo
        )
        == "recovery"
    )
    primary_context = preparer.validate_recovery_amendment(
        amendment_path,
        primary_origin,
        execution_contract,
        "recovery",
        repo_root=repo,
    )
    pair_failure_path = pair / "FAILURE.json"
    pair_failure_bytes = pair_failure_path.read_bytes()
    pair_failure_path.write_bytes(pair_failure_bytes + b"\n")
    with pytest.raises(RuntimeError, match="pair failure"):
        preparer.validate_recovery_amendment(
            amendment_path,
            primary_origin,
            execution_contract,
            "recovery",
            repo_root=repo,
        )
    pair_failure_path.write_bytes(pair_failure_bytes)

    replay_failure_path = replay_origin / "FAILURE.json"
    replay_failure_bytes = replay_failure_path.read_bytes()
    replay_failure_path.write_bytes(replay_failure_bytes + b"\n")
    with pytest.raises(RuntimeError, match="root failure inventory"):
        preparer.validate_recovery_amendment(
            amendment_path,
            primary_origin,
            execution_contract,
            "recovery",
            repo_root=repo,
        )
    replay_failure_path.write_bytes(replay_failure_bytes)

    amendment_bytes = amendment_path.read_bytes()
    amendment_path.write_bytes(amendment_bytes + b"\n")
    with pytest.raises(RuntimeError, match="amendment hash drifted"):
        preparer.validate_recovery_amendment(
            amendment_path,
            primary_origin,
            execution_contract,
            "recovery",
            repo_root=repo,
        )
    amendment_path.write_bytes(amendment_bytes)

    def hardlink_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.link(source, destination)

    with monkeypatch.context() as hardlink_patch:
        hardlink_patch.setattr(preparer, "copy_regular", hardlink_copy)
        with pytest.raises(RuntimeError, match="benchmark hardlink"):
            preparer.copy_wave60_preserved_benchmark(
                primary_origin,
                repo / "hardlinked-benchmark",
                preserved,
            )
    shutil.rmtree(repo / "hardlinked-benchmark")
    primary_escrow = preparer.validate_reused_escrow(
        primary_origin, execution_contract, primary_context
    )

    def fake_stage(
        output: Path, *_args: object, **_kwargs: object
    ) -> dict[str, object]:
        for seed in (17, 29, 43):
            for split in ("train", "val", "lockbox"):
                _repacked_npz(
                    output / f"inference/logits/seed{seed}__{split}.npz",
                    {"marker": np.asarray([seed], dtype=np.int64)},
                )
        _write_json(output / "inference/access_receipt.json", {"uid": 65534})
        return {
            "staging_input_hashes": {},
            "runtime_hashes": {},
            "checkpoint_receipts": [],
            "inference_hashes": {},
            "effective_uid": 65534,
            "effective_gid": 65534,
            "negative_truth_probe": True,
        }

    def fake_bundles(output: Path, *_args: object, **_kwargs: object) -> dict[str, str]:
        hashes = {}
        for name in wave60_runner.PREPARED_BUNDLES:
            path = output / "prepared" / name
            arrays = (
                source_material["inference"]
                if "inference" in name
                else source_material["truth"]
            )
            _repacked_npz(path, arrays)
            hashes[f"prepared/{name}"] = file_sha256(path)
        return hashes

    monkeypatch.setattr(preparer, "stage_and_infer", fake_stage)
    monkeypatch.setattr(preparer, "validate_manifest", lambda *_: None)
    monkeypatch.setattr(preparer, "validate_visible_package", lambda *_: None)
    monkeypatch.setattr(preparer, "validate_semantic_attestation", lambda *_: None)
    monkeypatch.setattr(preparer, "assert_prepared_boundary", lambda *_: None)
    monkeypatch.setattr(preparer, "sealed_population_counts", lambda *_: counts)
    monkeypatch.setattr(
        sys.modules["run_wave59_hgb_guard_bracket"],
        "materialize_prepared_bundles",
        fake_bundles,
    )
    primary_output.mkdir()
    _write_json(primary_output / "config.snapshot.json", v2)
    _write_json(primary_output / "source_bindings.json", v2["source_binding"])
    primary_prior = preparer.wave60_prior_preparation_elapsed(
        primary_args, v2, "recovery"
    )
    assert primary_prior == pytest.approx(
        wave60_runner.pair_preparation_elapsed(primary_origin, replay_origin)
    )
    with preparer.wave59_coordinator_budget(
        v2, prior_elapsed_seconds=primary_prior
    ) as primary_budget:
        preparer.run_preparation_transaction(
            primary_args,
            primary_output,
            config_path,
            v2,
            "recovery",
            execution_contract,
            primary_escrow,
            force=False,
            recovery_context=primary_context,
            protocol_override={},
        )
    preparer.finalize_preparation_budget_authority(
        primary_output,
        v2,
        "recovery",
        primary_budget,
        wave60_runner.DEFAULT_PRIVATE_KEY,
    )
    wave60_runner.validate_prepared_root(primary_output, "primary", v2)
    primary_budget_record = wave60_runner.preparation_budget_record(
        primary_output, "primary"
    )
    assert primary_budget_record["prior_elapsed_seconds"] == primary_prior
    assert primary_budget_record["cumulative_duration_seconds"] == pytest.approx(
        primary_prior + primary_budget_record["duration_seconds"]
    )
    for relative in manifest["files"]:
        source = primary_origin / "benchmark" / relative
        copied = primary_output / "benchmark" / relative
        assert source.read_bytes() == copied.read_bytes()
        assert (source.stat().st_dev, source.stat().st_ino) != (
            copied.stat().st_dev,
            copied.stat().st_ino,
        )

    replay_output = v2_container / "replay"
    replay_args = SimpleNamespace(
        wave51_dir=wave51_dir,
        wave52_dir=wave52_dir,
        wave54_dir=wave54_dir,
        replay_secrets_from=primary_output,
        recovery_secrets_from=None,
        recovery_amendment=amendment_path,
        reference_dir=primary_output,
        force=False,
        attestation_private_key=wave60_runner.DEFAULT_PRIVATE_KEY,
    )
    assert (
        preparer.validate_invocation(
            replay_args, replay_output, v2, repo_root=repo
        )
        == "replay"
    )
    replay_context = preparer.validate_recovery_amendment(
        amendment_path,
        primary_output,
        execution_contract,
        "replay",
        repo_root=repo,
    )
    replay_escrow = preparer.validate_reused_escrow(
        primary_output, execution_contract, replay_context
    )
    replay_prior = preparer.wave60_prior_preparation_elapsed(
        replay_args, v2, "replay"
    )
    assert replay_prior == pytest.approx(
        primary_budget_record["cumulative_duration_seconds"]
    )
    replay_output.mkdir()
    _write_json(replay_output / "config.snapshot.json", v2)
    _write_json(replay_output / "source_bindings.json", v2["source_binding"])
    with preparer.wave59_coordinator_budget(
        v2, prior_elapsed_seconds=replay_prior
    ) as replay_budget:
        preparer.run_preparation_transaction(
            replay_args,
            replay_output,
            config_path,
            v2,
            "replay",
            execution_contract,
            replay_escrow,
            force=False,
            recovery_context=replay_context,
            protocol_override={},
        )
    preparer.finalize_preparation_budget_authority(
        replay_output,
        v2,
        "replay",
        replay_budget,
        wave60_runner.DEFAULT_PRIVATE_KEY,
    )
    wave60_runner.validate_prepared_root(replay_output, "replay", v2)
    replay_budget_record = wave60_runner.preparation_budget_record(
        replay_output, "replay"
    )
    assert replay_budget_record["prior_elapsed_seconds"] == pytest.approx(
        primary_budget_record["cumulative_duration_seconds"]
    )
    assert wave60_runner.pair_preparation_elapsed(
        primary_output, replay_output
    ) == pytest.approx(replay_budget_record["cumulative_duration_seconds"])
    assert file_sha256(primary_output / "generation_escrow.json") == file_sha256(
        replay_output / "generation_escrow.json"
    )
    assert (primary_output / "generation_escrow.json").stat().st_ino != (
        replay_output / "generation_escrow.json"
    ).stat().st_ino

    original_score_root = wave60_runner.score_root

    def score_replay_then_fail(
        root: Path, role: str, *args: object, **kwargs: object
    ) -> object:
        result = original_score_root(root, role, *args, **kwargs)
        if role == "replay":
            raise RuntimeError("synthetic failure after replay score_apply")
        return result

    with monkeypatch.context() as phase_patch:
        phase_patch.setattr(
            wave60_runner, "validate_source_authority", lambda *_args: None
        )
        phase_patch.setattr(wave60_runner, "score_root", score_replay_then_fail)
        v2_pair = execute_prepared_pair(
            v2_container,
            v2,
            authority=source_authority["path"],
        )
    v2_status = load_json(v2_pair / "pair_status.json")
    assert v2_status["terminal"] == "PAIR_ABORTED_PRE_TRUTH"
    assert v2_status["primary_terminal"] == "PEER_ABORTED_PRE_TRUTH"
    assert v2_status["replay_terminal"] == "SCORE_APPLY_FAILED_PRE_TRUTH"
    v2_durable = wave60_runner.recovery_pair_durable_elapsed(v2_container)
    v2_journals = {
        (role, phase): load_json(
            v2_container / role / f"journals/{phase}.json"
        )
        for role in ("primary", "replay")
        for phase in ("source_bind", "score_apply")
    }
    assert v2_journals[("primary", "score_apply")]["status"] == (
        "LOCKBOX_ACTIONS_FROZEN"
    )
    assert v2_journals[("replay", "score_apply")]["status"] == "FAILED"
    expected_source_seconds = sum(
        journal["duration_seconds"]
        for (role, phase), journal in v2_journals.items()
        if phase == "source_bind"
    )
    expected_score_seconds = sum(
        journal["duration_seconds"]
        for (role, phase), journal in v2_journals.items()
        if phase == "score_apply"
    )
    assert v2_durable["preparation_seconds"] == pytest.approx(
        replay_budget_record["cumulative_duration_seconds"]
    )
    assert v2_durable["source_binding_seconds"] == pytest.approx(
        expected_source_seconds
    )
    assert v2_durable["score_apply_seconds"] == pytest.approx(
        expected_score_seconds
    )
    assert expected_source_seconds > 0.0
    assert expected_score_seconds > 0.0
    assert v2_durable["durable_seconds"] == pytest.approx(
        v2_durable["preparation_seconds"] + v2_durable["phase_seconds"]
    )
    static_recovery_sources = dict(
        preparer.WAVE60_STATIC_PROTOCOL_RECOVERY_SOURCES
    )
    static_old_hashes = {
        relative: v2["source_sha256"][relative]
        for relative in static_recovery_sources.values()
    }
    for relative in static_recovery_sources.values():
        (repo / relative).write_text(
            f"static-v3:{relative}\n", encoding="utf-8"
        )
    static_implementation_commit = commit_all(
        "static-protocol recovery implementation"
    )
    assert preparer.git_changed_paths(
        repo, static_implementation_commit
    ) == set(static_recovery_sources.values())
    static_implementation_audit_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "498_wave60_static_protocol_implementation_audit.md"
    )
    write_audit(
        static_implementation_audit_relative,
        audit_id="R498",
        scope="STATIC_PROTOCOL_IDENTITY_GUARD_RECOVERY_IMPLEMENTATION",
        target={"implementation_commit": static_implementation_commit},
    )
    static_implementation_audit_commit = commit_all(
        "static-protocol implementation audit"
    )
    static_recovery_implementation = {
        "commit": static_implementation_commit,
        "audit_commit": static_implementation_audit_commit,
        "audit_path": static_implementation_audit_relative,
        "audit_sha256": file_sha256(
            repo / static_implementation_audit_relative
        ),
        "audit_id": "R498",
        "scope": "STATIC_PROTOCOL_IDENTITY_GUARD_RECOVERY_IMPLEMENTATION",
        "changed_sources": {
            label: {
                "path": relative,
                "old_sha256": static_old_hashes[relative],
                "new_sha256": file_sha256(repo / relative),
            }
            for label, relative in static_recovery_sources.items()
        },
        "unchanged_source_law_sources": list(
            preparer.WAVE60_STATIC_PROTOCOL_UNCHANGED_SOURCES
        ),
    }
    v2_manifest = load_json(primary_output / "benchmark/manifest.json")
    v3_preserved = {
        "generation_escrow.json": file_sha256(
            primary_output / "generation_escrow.json"
        ),
        "pre_generation_freeze.json": file_sha256(
            primary_output / "pre_generation_freeze.json"
        ),
        "benchmark/manifest.json": file_sha256(
            primary_output / "benchmark/manifest.json"
        ),
        **{
            f"benchmark/{relative}": record["sha256"]
            for relative, record in v2_manifest["files"].items()
        },
    }
    v2_audit_sha256 = file_sha256(repo / v2_audit_relative)
    v3_amendment_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_STATIC_PROTOCOL_RECOVERY_V3_AMENDMENT.json"
    )
    terminal_relatives = (
        "pair/pair_status.json",
        "pair/FAILURE.json",
        "pair/failure_inventory.json",
        "pair/failure_attestation.json",
        "pair/artifact_manifest.json",
        "primary/FAILURE.json",
        "primary/failure_inventory.json",
        "primary/failure_attestation.json",
        "primary/preparation_receipt.json",
        "primary/preparation_attestation.json",
        "replay/FAILURE.json",
        "replay/failure_inventory.json",
        "replay/failure_attestation.json",
        "replay/preparation_receipt.json",
        "replay/preparation_attestation.json",
    )
    v3_amendment = {
        "schema_version": preparer.WAVE60_STATIC_PROTOCOL_RECOVERY_AMENDMENT_SCHEMA,
        "status": "APPROVED",
        "recovery_kind": "STATIC_PROTOCOL_IDENTITY_GUARD",
        "prior_attempt_container": v2_container_relative,
        "prior_pair_failure_sha256": file_sha256(v2_pair / "FAILURE.json"),
        "prior_source_sha256": dict(v2["source_sha256"]),
        "prior_terminal": {
            relative: file_sha256(v2_container / relative)
            for relative in terminal_relatives
        },
        "prior_durable_budget": v2_durable,
        "prior_config_audit": {
            "commit": v2_audit_commit,
            "path": v2_audit_relative,
            "sha256": v2_audit_sha256,
        },
        "initial_plan": {
            "commit": "1" * 40,
            "path": "synthetic-initial-plan.md",
            "sha256": "1" * 64,
        },
        "initial_plan_audit": {
            "commit": "2" * 40,
            "path": "synthetic-initial-audit.md",
            "sha256": "2" * 64,
            "audit_id": "R496",
            "verdict": "REVISE",
            "findings": {"high": 0, "medium": 0, "low": 1},
        },
        "correction_plan": {
            "commit": "3" * 40,
            "path": "synthetic-correction-plan.md",
            "sha256": "3" * 64,
        },
        "correction_plan_audit": {
            "commit": v2_audit_commit,
            "path": "synthetic-correction-audit.md",
            "sha256": "4" * 64,
            "audit_id": "R497",
            "verdict": "PASS",
            "findings": {"high": 0, "medium": 0, "low": 0},
        },
        "recovery_implementation": static_recovery_implementation,
        "escrow_origin": {
            "contract_sha256": preparer.compact_json_sha256(origin_contract),
            "escrow_sha256": v3_preserved["generation_escrow.json"],
            "pre_generation_freeze_sha256": v3_preserved[
                "pre_generation_freeze.json"
            ],
            "benchmark_manifest_sha256": v3_preserved[
                "benchmark/manifest.json"
            ],
        },
        "preserved_draw_sha256": v3_preserved,
        "population_contract": {
            "eligibility_predicate": {
                "is_out_of_catalog": False,
                "calibration_population": "canonical_preserving",
                "filter_rows_before_deduplicating_pair_tokens": True,
            },
            "counts_by_split": {
                split: counts for split in ("train", "val", "lockbox")
            },
        },
        "origin_inventory": preparer.physical_tree_inventory(primary_output),
    }
    v3_amendment_path = repo / v3_amendment_relative
    _write_json(v3_amendment_path, v3_amendment)
    v3_amendment_sha256 = file_sha256(v3_amendment_path)
    commit_all("v3 recovery amendment")
    v3_amendment_audit_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "499_wave60_v3_static_protocol_recovery_audit.md"
    )
    write_audit(
        v3_amendment_audit_relative,
        audit_id="R499",
        scope="STATIC_PROTOCOL_IDENTITY_GUARD_RECOVERY_AMENDMENT",
        target={"amendment_sha256": v3_amendment_sha256},
    )
    v3_amendment_audit_commit = commit_all("v3 recovery amendment audit")
    v3_amendment_audit_sha256 = file_sha256(
        repo / v3_amendment_audit_relative
    )
    v3_container_relative = (
        "data/geometria_proporcional/"
        "wave60_frozen_policy_transport_attempt_v3"
    )
    v3 = deepcopy(v2)
    v3["source_sha256"].update(
        {
            relative: file_sha256(repo / relative)
            for relative in static_recovery_sources.values()
        }
    )
    v3["attempt"] = {
        "version": 3,
        "container": v3_container_relative,
        "primary": "primary",
        "replay": "replay",
        "pair": "pair",
        "recovery": {
            "schema_version": "wave60-pretruth-recovery-v1",
            "prior_attempt_container": v2_container_relative,
            "prior_pair_failure_sha256": v3_amendment[
                "prior_pair_failure_sha256"
            ],
            "prior_config_audit_commit": v2_audit_commit,
            "prior_config_audit_path": v2_audit_relative,
            "prior_config_audit_sha256": v2_audit_sha256,
            "amendment_path": v3_amendment_relative,
            "amendment_sha256": v3_amendment_sha256,
            "amendment_audit_commit": v3_amendment_audit_commit,
            "amendment_audit_path": v3_amendment_audit_relative,
            "amendment_audit_sha256": v3_amendment_audit_sha256,
            "preserved_draw_sha256": v3_preserved,
        },
    }
    v3_audit_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "500_wave60_v3_config_audit.md"
    )
    v3["final_audit"] = {
        "audit_id": "R500",
        "audit_path": v3_audit_relative,
    }
    v3["primary_output"] = f"{v3_container_relative}/primary"
    v3["replay_output"] = f"{v3_container_relative}/replay"
    v3["output_parent_relative"] = v3_container_relative
    validate_pre_draw_config(v3)
    _write_json(config_path, v3)
    v3_config_commit = commit_all("v3 config")
    v3_config_sha256 = file_sha256(config_path)
    write_audit(
        v3_audit_relative,
        audit_id="R500",
        scope="CONFIG",
        target={
            "config_commit": v3_config_commit,
            "config_sha256": v3_config_sha256,
        },
    )
    v3_audit_commit = commit_all("v3 config audit")
    preparer.validate_wave60_final_config_authority(
        repo, config_path, v3, v3_audit_commit
    )
    v3_contract = {
        **origin_contract,
        "git_commit": v3_audit_commit,
        "config_sha256": v3_config_sha256,
        "prospective_config": v3,
        "sources": dict(v3["source_sha256"]),
    }
    v3_container = repo / v3_container_relative
    v3_container.mkdir(parents=True)
    v3_primary = v3_container / "primary"
    v3_primary_args = SimpleNamespace(
        wave51_dir=wave51_dir,
        wave52_dir=wave52_dir,
        wave54_dir=wave54_dir,
        replay_secrets_from=None,
        recovery_secrets_from=primary_output,
        recovery_amendment=v3_amendment_path,
        reference_dir=None,
        force=False,
        attestation_private_key=wave60_runner.DEFAULT_PRIVATE_KEY,
    )
    assert preparer.validate_invocation(
        v3_primary_args, v3_primary, v3, repo_root=repo
    ) == "recovery"
    monkeypatch.setattr(
        preparer,
        "_validate_wave60_static_protocol_plan_chain",
        lambda *_args, **_kwargs: None,
    )
    bad_terminal = deepcopy(v3_amendment)
    bad_terminal["prior_terminal"]["pair/pair_status.json"] = "0" * 64
    with pytest.raises(RuntimeError, match="terminal binding drifted"):
        preparer.validate_wave60_static_protocol_terminal_binding(
            repo, bad_terminal
        )
    bad_budget = deepcopy(v3_amendment)
    bad_budget["prior_durable_budget"]["durable_seconds"] += 1.0
    with pytest.raises(RuntimeError, match="durable budget drifted"):
        preparer.validate_wave60_static_protocol_terminal_binding(
            repo, bad_budget
        )
    bad_prior_sources = deepcopy(v3_amendment)
    bad_prior_sources["prior_source_sha256"][
        static_recovery_sources["preparer"]
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="prior source map drifted"):
        preparer.validate_wave60_static_protocol_recovery_authority(
            repo, bad_prior_sources, v3, v2
        )
    bad_lineage = deepcopy(v3_amendment)
    bad_lineage["correction_plan_audit"]["commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="directly descend"):
        preparer.validate_wave60_static_protocol_recovery_authority(
            repo, bad_lineage, v3, v2
        )
    bad_source_delta = deepcopy(v3_amendment)
    bad_source_delta["recovery_implementation"]["changed_sources"][
        "runner"
    ]["old_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="runner source binding drifted"):
        preparer.validate_wave60_static_protocol_recovery_authority(
            repo, bad_source_delta, v3, v2
        )
    v3_primary_context = preparer.validate_recovery_amendment(
        v3_amendment_path,
        primary_output,
        v3_contract,
        "recovery",
        repo_root=repo,
    )
    v3_primary_escrow = preparer.validate_reused_escrow(
        primary_output, v3_contract, v3_primary_context
    )
    v3_primary.mkdir()
    _write_json(v3_primary / "config.snapshot.json", v3)
    _write_json(v3_primary / "source_bindings.json", v3["source_binding"])
    v3_primary_prior = preparer.wave60_prior_preparation_elapsed(
        v3_primary_args, v3, "recovery"
    )
    assert v3_primary_prior == pytest.approx(v2_durable["durable_seconds"])
    assert v3_primary_prior > replay_budget_record["cumulative_duration_seconds"]
    with preparer.wave59_coordinator_budget(
        v3, prior_elapsed_seconds=v3_primary_prior
    ) as v3_primary_budget:
        preparer.run_preparation_transaction(
            v3_primary_args,
            v3_primary,
            config_path,
            v3,
            "recovery",
            v3_contract,
            v3_primary_escrow,
            force=False,
            recovery_context=v3_primary_context,
            protocol_override={},
        )
    preparer.finalize_preparation_budget_authority(
        v3_primary,
        v3,
        "recovery",
        v3_primary_budget,
        wave60_runner.DEFAULT_PRIVATE_KEY,
    )
    wave60_runner.validate_prepared_root(v3_primary, "primary", v3)
    v3_primary_record = wave60_runner.preparation_budget_record(
        v3_primary, "primary"
    )
    assert v3_primary_record["prior_elapsed_seconds"] == pytest.approx(
        v2_durable["durable_seconds"]
    )

    v3_replay = v3_container / "replay"
    v3_replay_args = SimpleNamespace(
        wave51_dir=wave51_dir,
        wave52_dir=wave52_dir,
        wave54_dir=wave54_dir,
        replay_secrets_from=v3_primary,
        recovery_secrets_from=None,
        recovery_amendment=v3_amendment_path,
        reference_dir=v3_primary,
        force=False,
        attestation_private_key=wave60_runner.DEFAULT_PRIVATE_KEY,
    )
    assert preparer.validate_invocation(
        v3_replay_args, v3_replay, v3, repo_root=repo
    ) == "replay"
    v3_replay_context = preparer.validate_recovery_amendment(
        v3_amendment_path,
        v3_primary,
        v3_contract,
        "replay",
        repo_root=repo,
    )
    v3_replay_escrow = preparer.validate_reused_escrow(
        v3_primary, v3_contract, v3_replay_context
    )
    v3_replay.mkdir()
    _write_json(v3_replay / "config.snapshot.json", v3)
    _write_json(v3_replay / "source_bindings.json", v3["source_binding"])
    v3_replay_prior = preparer.wave60_prior_preparation_elapsed(
        v3_replay_args, v3, "replay"
    )
    assert v3_replay_prior == pytest.approx(
        v3_primary_record["cumulative_duration_seconds"]
    )
    with preparer.wave59_coordinator_budget(
        v3, prior_elapsed_seconds=v3_replay_prior
    ) as v3_replay_budget:
        preparer.run_preparation_transaction(
            v3_replay_args,
            v3_replay,
            config_path,
            v3,
            "replay",
            v3_contract,
            v3_replay_escrow,
            force=False,
            recovery_context=v3_replay_context,
            protocol_override={},
        )
    preparer.finalize_preparation_budget_authority(
        v3_replay,
        v3,
        "replay",
        v3_replay_budget,
        wave60_runner.DEFAULT_PRIVATE_KEY,
    )
    wave60_runner.validate_prepared_root(v3_replay, "replay", v3)
    v3_replay_record = wave60_runner.preparation_budget_record(
        v3_replay, "replay"
    )
    assert v3_replay_record["prior_elapsed_seconds"] == pytest.approx(
        v3_primary_record["cumulative_duration_seconds"]
    )
    assert wave60_runner.pair_preparation_elapsed(
        v3_primary, v3_replay
    ) == pytest.approx(v3_replay_record["cumulative_duration_seconds"])
    assert v3_replay_record["cumulative_duration_seconds"] > (
        replay_budget_record["cumulative_duration_seconds"]
    )
    assert file_sha256(v3_primary / "generation_escrow.json") == file_sha256(
        v3_replay / "generation_escrow.json"
    )
    assert (v3_primary / "generation_escrow.json").stat().st_ino != (
        v3_replay / "generation_escrow.json"
    ).stat().st_ino
    assert preparer.read_escrow(v3_primary)["contract"]["prospective_config"][
        "attempt"
    ]["version"] == 1
    assert load_json(v3_primary / "config.snapshot.json")["attempt"][
        "version"
    ] == 3


def test_canonical_audit_parser_ignores_incidental_prose_pass(tmp_path: Path) -> None:
    target = {"implementation_commit": "1" * 40}
    authority = {
        "schema_version": "wave60-audit-authority-v1",
        "audit_id": "R999",
        "scope": "IMPLEMENTATION",
        "target": target,
        "technical_verdict": "PASS",
        "findings": {"high": 0, "medium": 0, "low": 0},
        "files_modified": False,
        "gpu_used_or_queried": False,
    }
    report = tmp_path / "audit.md"
    report.write_text(
        "PASS incidental in prose.\n\n```json\n"
        + json.dumps(authority, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    assert (
        preparer.parse_wave60_audit_report(
            report, scope="IMPLEMENTATION", target=target
        )["technical_verdict"]
        == "PASS"
    )
    authority["technical_verdict"] = "REVISE"
    report.write_text(
        "PASS incidental in prose.\n\n```json\n"
        + json.dumps(authority, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not grant exact PASS"):
        preparer.parse_wave60_audit_report(
            report, scope="IMPLEMENTATION", target=target
        )


def test_recovery_implementation_scope_is_authenticated_by_runner_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "wave60@test.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Wave 60 Test"], cwd=repo, check=True
    )
    implementation = repo / "implementation.py"
    implementation.write_text("accepted = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "implementation"], cwd=repo, check=True)
    implementation_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    relative_audit = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "999_wave60_recovery_implementation_audit.md"
    )
    audit = repo / relative_audit
    audit.parent.mkdir(parents=True)
    authority = {
        "schema_version": "wave60-audit-authority-v1",
        "audit_id": "R999",
        "scope": SOURCE_LAW_RECOVERY_IMPLEMENTATION_SCOPE,
        "target": {"implementation_commit": implementation_commit},
        "technical_verdict": "PASS",
        "findings": {"high": 0, "medium": 0, "low": 0},
        "files_modified": False,
        "gpu_used_or_queried": False,
    }
    audit.write_text(
        "# Recovery implementation audit\n\n```json\n"
        + json.dumps(authority, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "audit"], cwd=repo, check=True)
    audit_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    audit_sha256 = file_sha256(audit)
    binding = {
        "status": "ACCEPTED_IMPLEMENTATION_AUDIT",
        "commit": implementation_commit,
        "audit_commit": audit_commit,
        "audit_path": relative_audit,
        "audit_sha256": audit_sha256,
    }
    monkeypatch.setattr(wave60_runner, "REPO_ROOT", repo)
    validate_implementation_audit_authority(
        implementation_commit,
        audit_commit,
        audit_sha256,
        expected_scope=SOURCE_LAW_RECOVERY_IMPLEMENTATION_SCOPE,
    )
    preparer.validate_wave60_implementation_audit_commit(repo, binding)
    with pytest.raises(RuntimeError, match="does not grant PASS"):
        validate_implementation_audit_authority(
            implementation_commit,
            audit_commit,
            audit_sha256,
            expected_scope="IMPLEMENTATION",
        )
    with pytest.raises(RuntimeError, match="exact PASS"):
        preparer.validate_wave60_audit_commit(
            repo,
            binding,
            scope="IMPLEMENTATION",
            target={"implementation_commit": implementation_commit},
            expected_parent=implementation_commit,
        )


def test_recovery_implementation_lineage_accepts_only_r474_child_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "wave60@test.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Wave 60 Test"], cwd=repo, check=True
    )
    anchor = repo / "r474.md"
    anchor.write_text("R474 PASS\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "R474"], cwd=repo, check=True)
    r474_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    implementation = (
        repo / "experiments/geometria_proporcional/prepare_wave56_fresh.py"
    )
    implementation.parent.mkdir(parents=True)
    implementation.write_text("recovery_scope = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "correct implementation"], cwd=repo, check=True
    )
    implementation_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    monkeypatch.setattr(wave60_runner, "REPO_ROOT", repo)
    monkeypatch.setattr(
        wave60_runner, "SOURCE_LAW_RECOVERY_RESOLUTION_AUDIT_COMMIT", r474_commit
    )
    validate_recovery_implementation_lineage(implementation_commit)
    unrelated = repo / "unrelated.txt"
    unrelated.write_text("drift\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "unrelated"], cwd=repo, check=True)
    unrelated_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    with pytest.raises(RuntimeError, match="lineage drifted"):
        validate_recovery_implementation_lineage(unrelated_commit)


def test_final_config_authority_requires_config_only_then_audit_at_head(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "wave60@test.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Wave 60 Test"], cwd=repo, check=True
    )
    implementation_sources = {
        "src/geometria_proporcional/wave60_frozen_policy_transport.py",
        "experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py",
        "experiments/geometria_proporcional/_wave60_phase_worker.py",
        "experiments/geometria_proporcional/prepare_wave56_fresh.py",
        "tests/test_wave60_frozen_policy_transport.py",
    }
    for relative in implementation_sources:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"audited:{relative}\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "implementation"], cwd=repo, check=True)
    implementation_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    source_audit = repo / "Biblioteca/source-law-audit.md"
    source_audit.parent.mkdir(parents=True)
    source_audit.write_text("source law accepted\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "source audit"], cwd=repo, check=True)
    source_audit_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    config = valid_config()
    config["implementation_binding"]["commit"] = implementation_commit
    config["source_law_authority"]["audit_commit"] = source_audit_commit
    config["source_sha256"].update(
        {
            relative: file_sha256(repo / relative)
            for relative in implementation_sources
        }
    )
    config_path = (
        repo
        / "experiments/geometria_proporcional/configs/"
        "wave60_frozen_policy_transport.json"
    )
    _write_json(config_path, config)
    subprocess.run(["git", "add", str(config_path)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "config"], cwd=repo, check=True)
    config_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    audit_path = repo / config["final_audit"]["audit_path"]
    authority = {
        "schema_version": "wave60-audit-authority-v1",
        "audit_id": config["final_audit"]["audit_id"],
        "scope": "CONFIG",
        "target": {
            "config_commit": config_commit,
            "config_sha256": file_sha256(config_path),
        },
        "technical_verdict": "PASS",
        "findings": {"high": 0, "medium": 0, "low": 0},
        "files_modified": False,
        "gpu_used_or_queried": False,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        "# Final config audit\n\n```json\n"
        + json.dumps(authority, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", str(audit_path)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "config audit"], cwd=repo, check=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    preparer.validate_wave60_final_config_authority(
        repo, config_path, config, head
    )
    (repo / next(iter(implementation_sources))).write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="executed blob differs"):
        preparer.validate_wave60_final_config_authority(
            repo, config_path, config, head
        )


def test_invalid_preparation_amendment_audit_requires_r494(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "wave60@test.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Wave 60 Test"], cwd=repo, check=True
    )

    def commit(paths: list[str], message: str) -> str:
        subprocess.run(["git", "add", "-f", "--", *paths], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", message], cwd=repo, check=True)
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()

    amendment_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_INVALID_PREPARATION_RECOVERY_V2_AMENDMENT.json"
    )
    amendment_path = repo / amendment_relative
    amendment_path.parent.mkdir(parents=True)
    amendment_path.write_text('{"status":"APPROVED"}\n', encoding="utf-8")
    amendment_commit = commit([amendment_relative], "amendment")
    amendment_sha256 = file_sha256(amendment_path)
    audit_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "494_wave60_invalid_preparation_recovery_amendment_audit.md"
    )

    def materialize_audit(audit_id: str) -> dict[str, str]:
        audit_path = repo / audit_relative
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        authority = {
            "schema_version": "wave60-audit-authority-v1",
            "audit_id": audit_id,
            "scope": "INVALID_PREPARATION_RECOVERY_AMENDMENT",
            "target": {"amendment_sha256": amendment_sha256},
            "technical_verdict": "PASS",
            "findings": {"high": 0, "medium": 0, "low": 0},
            "files_modified": False,
            "gpu_used_or_queried": False,
        }
        audit_path.write_text(
            f"# {audit_id}\n\n```json\n"
            + json.dumps(authority, sort_keys=True)
            + "\n```\n",
            encoding="utf-8",
        )
        audit_commit = commit([audit_relative], f"{audit_id} amendment audit")
        assert preparer.git_changed_paths(repo, audit_commit) == {audit_relative}
        assert subprocess.check_output(
            ["git", "rev-parse", f"{audit_commit}^"], cwd=repo, text=True
        ).strip() == amendment_commit
        return {
            "amendment_audit_commit": audit_commit,
            "amendment_audit_path": audit_relative,
            "amendment_audit_sha256": file_sha256(audit_path),
        }

    recovery = materialize_audit("R494")
    preparer.validate_wave60_invalid_preparation_amendment_audit(
        repo,
        recovery,
        amendment_sha256,
        amendment_commit,
    )

    for stale_audit_id in ("R490", "R492"):
        subprocess.run(
            ["git", "checkout", "-q", "--detach", amendment_commit],
            cwd=repo,
            check=True,
        )
        stale_recovery = materialize_audit(stale_audit_id)
        with pytest.raises(RuntimeError, match="audit id drifted"):
            preparer.validate_wave60_invalid_preparation_amendment_audit(
                repo,
                stale_recovery,
                amendment_sha256,
                amendment_commit,
            )


def test_invalid_preparation_implementation_suffix_preserves_revise_then_pass(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "wave60@test.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Wave 60 Test"], cwd=repo, check=True
    )

    def commit(paths: list[str], message: str) -> str:
        subprocess.run(["git", "add", "-f", "--", *paths], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", message], cwd=repo, check=True)
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()

    def write_audit(
        relative: str,
        *,
        audit_id: str,
        scope: str,
        target: dict[str, str],
        verdict: str,
        findings: dict[str, int],
    ) -> None:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        authority = {
            "schema_version": "wave60-audit-authority-v1",
            "audit_id": audit_id,
            "scope": scope,
            "target": target,
            "technical_verdict": verdict,
            "findings": findings,
            "files_modified": False,
            "gpu_used_or_queried": False,
        }
        path.write_text(
            f"# {audit_id}\n\n```json\n"
            + json.dumps(authority, sort_keys=True)
            + "\n```\n",
            encoding="utf-8",
        )

    recovery_sources = dict(preparer.WAVE60_RECOVERY_IMPLEMENTATION_SOURCES)
    source_law_sources = list(preparer.WAVE60_SOURCE_LAW_SOURCES)
    implementation_paths = set(recovery_sources.values())
    for relative in [*source_law_sources, *recovery_sources.values()]:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"r475:{relative}\n", encoding="utf-8")
    anchor_relative = "r480_anchor.md"
    (repo / anchor_relative).write_text("R480\n", encoding="utf-8")
    r480 = commit(
        [anchor_relative, *source_law_sources, *recovery_sources.values()],
        "synthetic R475/R480 anchor",
    )
    r475 = r480
    old_hashes = {
        relative: file_sha256(repo / relative)
        for relative in recovery_sources.values()
    }

    for relative in recovery_sources.values():
        (repo / relative).write_text(f"rejected:{relative}\n", encoding="utf-8")
    rejected_commit = commit(
        list(recovery_sources.values()), "rejected recovery implementation"
    )
    rejected = {
        "commit": rejected_commit,
        "parent": r480,
        "changed_sources": {
            label: {
                "path": relative,
                "old_sha256": old_hashes[relative],
                "new_sha256": file_sha256(repo / relative),
            }
            for label, relative in recovery_sources.items()
        },
    }
    r481_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "481_wave60_invalid_preparation_recovery_implementation_audit.md"
    )
    write_audit(
        r481_relative,
        audit_id="R481",
        scope="INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        target={"implementation_commit": rejected_commit},
        verdict="REVISE",
        findings={"high": 0, "medium": 1, "low": 0},
    )
    r481_commit = commit([r481_relative], "R481 REVISE")
    r481 = {
        "commit": r481_commit,
        "path": r481_relative,
        "sha256": file_sha256(repo / r481_relative),
        "audit_id": "R481",
        "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        "verdict": "REVISE",
        "findings": {"high": 0, "medium": 1, "low": 0},
    }

    resolution_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_INVALID_PREPARATION_RECOVERY_R481_RESOLUTION_PLAN.md"
    )
    resolution_path = repo / resolution_relative
    resolution_path.parent.mkdir(parents=True, exist_ok=True)
    resolution_path.write_text("R481 resolution\n", encoding="utf-8")
    resolution_commit = commit([resolution_relative], "R481 resolution")
    resolution = {
        "commit": resolution_commit,
        "path": resolution_relative,
        "sha256": file_sha256(resolution_path),
    }
    r482_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "482_wave60_invalid_preparation_recovery_r481_resolution_plan_audit.md"
    )
    write_audit(
        r482_relative,
        audit_id="R482",
        scope="INVALID_PREPARATION_RECOVERY_R481_RESOLUTION_PLAN",
        target={
            "plan_commit": resolution_commit,
            "plan_sha256": resolution["sha256"],
        },
        verdict="PASS",
        findings={"high": 0, "medium": 0, "low": 0},
    )
    r482_commit = commit([r482_relative], "R482 PASS")
    r482 = {
        "commit": r482_commit,
        "path": r482_relative,
        "sha256": file_sha256(repo / r482_relative),
        "audit_id": "R482",
        "verdict": "PASS",
        "findings": {"high": 0, "medium": 0, "low": 0},
    }

    for relative in recovery_sources.values():
        (repo / relative).write_text(f"r483-revise:{relative}\n", encoding="utf-8")
    intermediate_commit = commit(
        list(recovery_sources.values()), "R481 resolution implementation"
    )
    intermediate = {
        "commit": intermediate_commit,
        "parent": r482_commit,
        "changed_sources": {
            label: {
                "path": relative,
                "old_sha256": old_hashes[relative],
                "new_sha256": file_sha256(repo / relative),
            }
            for label, relative in recovery_sources.items()
        },
    }
    r483_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "483_wave60_invalid_preparation_recovery_implementation_reaudit.md"
    )
    write_audit(
        r483_relative,
        audit_id="R483",
        scope="INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        target={"implementation_commit": intermediate_commit},
        verdict="REVISE",
        findings={"high": 0, "medium": 1, "low": 0},
    )
    r483_commit = commit([r483_relative], "R483 REVISE")
    r483 = {
        "commit": r483_commit,
        "path": r483_relative,
        "sha256": file_sha256(repo / r483_relative),
        "audit_id": "R483",
        "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        "verdict": "REVISE",
        "findings": {"high": 0, "medium": 1, "low": 0},
    }

    r483_resolution_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN.md"
    )
    r483_resolution_path = repo / r483_resolution_relative
    r483_resolution_path.write_text("R483 resolution\n", encoding="utf-8")
    r483_resolution_commit = commit(
        [r483_resolution_relative], "R483 resolution"
    )
    r483_resolution = {
        "commit": r483_resolution_commit,
        "path": r483_resolution_relative,
        "sha256": file_sha256(r483_resolution_path),
    }
    r484_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "484_wave60_invalid_preparation_recovery_r483_resolution_plan_audit.md"
    )
    write_audit(
        r484_relative,
        audit_id="R484",
        scope="INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN",
        target={
            "plan_commit": r483_resolution_commit,
            "plan_sha256": r483_resolution["sha256"],
        },
        verdict="PASS",
        findings={"high": 0, "medium": 0, "low": 0},
    )
    r484_commit = commit([r484_relative], "R484 PASS")
    r484 = {
        "commit": r484_commit,
        "path": r484_relative,
        "sha256": file_sha256(repo / r484_relative),
        "audit_id": "R484",
        "verdict": "PASS",
        "findings": {"high": 0, "medium": 0, "low": 0},
    }

    for relative in recovery_sources.values():
        (repo / relative).write_text(f"r485-revise:{relative}\n", encoding="utf-8")
    r483_resolution_implementation_commit = commit(
        list(recovery_sources.values()), "R483 resolution implementation"
    )
    r483_resolution_implementation = {
        "commit": r483_resolution_implementation_commit,
        "parent": r484_commit,
        "changed_sources": {
            label: {
                "path": relative,
                "old_sha256": old_hashes[relative],
                "new_sha256": file_sha256(repo / relative),
            }
            for label, relative in recovery_sources.items()
        },
    }
    r485_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "485_wave60_invalid_preparation_recovery_implementation_final_reaudit.md"
    )
    write_audit(
        r485_relative,
        audit_id="R485",
        scope="INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        target={"implementation_commit": r483_resolution_implementation_commit},
        verdict="REVISE",
        findings={"high": 0, "medium": 1, "low": 0},
    )
    r485_commit = commit([r485_relative], "R485 REVISE")
    r485 = {
        "commit": r485_commit,
        "path": r485_relative,
        "sha256": file_sha256(repo / r485_relative),
        "audit_id": "R485",
        "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        "verdict": "REVISE",
        "findings": {"high": 0, "medium": 1, "low": 0},
    }

    r485_resolution_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN.md"
    )
    r485_resolution_path = repo / r485_resolution_relative
    r485_resolution_path.write_text("R485 resolution\n", encoding="utf-8")
    r485_resolution_commit = commit(
        [r485_resolution_relative], "R485 resolution"
    )
    r485_resolution = {
        "commit": r485_resolution_commit,
        "path": r485_resolution_relative,
        "sha256": file_sha256(r485_resolution_path),
    }
    r486_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "486_wave60_invalid_preparation_recovery_r485_resolution_plan_audit.md"
    )
    write_audit(
        r486_relative,
        audit_id="R486",
        scope="INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN",
        target={
            "plan_commit": r485_resolution_commit,
            "plan_sha256": r485_resolution["sha256"],
        },
        verdict="REVISE",
        findings={"high": 0, "medium": 2, "low": 0},
    )
    r486_commit = commit([r486_relative], "R486 REVISE")
    r486 = {
        "commit": r486_commit,
        "path": r486_relative,
        "sha256": file_sha256(repo / r486_relative),
        "audit_id": "R486",
        "verdict": "REVISE",
        "findings": {"high": 0, "medium": 2, "low": 0},
    }

    r486_resolution_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN.md"
    )
    r486_resolution_path = repo / r486_resolution_relative
    r486_resolution_path.write_text("R486 resolution\n", encoding="utf-8")
    r486_resolution_commit = commit(
        [r486_resolution_relative], "R486 resolution"
    )
    r486_resolution = {
        "commit": r486_resolution_commit,
        "path": r486_resolution_relative,
        "sha256": file_sha256(r486_resolution_path),
    }
    r487_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "487_wave60_invalid_preparation_recovery_r486_resolution_plan_audit.md"
    )
    write_audit(
        r487_relative,
        audit_id="R487",
        scope="INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN",
        target={
            "plan_commit": r486_resolution_commit,
            "plan_sha256": r486_resolution["sha256"],
        },
        verdict="REVISE",
        findings={"high": 0, "medium": 1, "low": 0},
    )
    r487_commit = commit([r487_relative], "R487 REVISE")
    r487 = {
        "commit": r487_commit,
        "path": r487_relative,
        "sha256": file_sha256(repo / r487_relative),
        "audit_id": "R487",
        "verdict": "REVISE",
        "findings": {"high": 0, "medium": 1, "low": 0},
    }

    r487_resolution_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_INVALID_PREPARATION_RECOVERY_R487_RESOLUTION_PLAN.md"
    )
    r487_resolution_path = repo / r487_resolution_relative
    r487_resolution_path.write_text("R487 resolution\n", encoding="utf-8")
    r487_resolution_commit = commit(
        [r487_resolution_relative], "R487 resolution"
    )
    r487_resolution = {
        "commit": r487_resolution_commit,
        "path": r487_resolution_relative,
        "sha256": file_sha256(r487_resolution_path),
    }
    r488_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "488_wave60_invalid_preparation_recovery_r487_resolution_plan_audit.md"
    )
    write_audit(
        r488_relative,
        audit_id="R488",
        scope="INVALID_PREPARATION_RECOVERY_R487_RESOLUTION_PLAN",
        target={
            "plan_commit": r487_resolution_commit,
            "plan_sha256": r487_resolution["sha256"],
        },
        verdict="PASS",
        findings={"high": 0, "medium": 0, "low": 0},
    )
    r488_commit = commit([r488_relative], "R488 PASS")
    r488 = {
        "commit": r488_commit,
        "path": r488_relative,
        "sha256": file_sha256(repo / r488_relative),
        "audit_id": "R488",
        "verdict": "PASS",
        "findings": {"high": 0, "medium": 0, "low": 0},
    }

    for relative in recovery_sources.values():
        (repo / relative).write_text(f"r489-revise:{relative}\n", encoding="utf-8")
    r487_resolution_implementation_commit = commit(
        list(recovery_sources.values()), "R487 resolution implementation"
    )
    r487_resolution_implementation = {
        "commit": r487_resolution_implementation_commit,
        "parent": r488_commit,
        "changed_sources": {
            label: {
                "path": relative,
                "old_sha256": old_hashes[relative],
                "new_sha256": file_sha256(repo / relative),
            }
            for label, relative in recovery_sources.items()
        },
    }
    r489_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "489_wave60_invalid_preparation_recovery_implementation_acceptance_audit.md"
    )
    write_audit(
        r489_relative,
        audit_id="R489",
        scope="INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        target={"implementation_commit": r487_resolution_implementation_commit},
        verdict="REVISE",
        findings={"high": 0, "medium": 1, "low": 0},
    )
    r489_commit = commit([r489_relative], "R489 REVISE")
    r489 = {
        "commit": r489_commit,
        "path": r489_relative,
        "sha256": file_sha256(repo / r489_relative),
        "audit_id": "R489",
        "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        "verdict": "REVISE",
        "findings": {"high": 0, "medium": 1, "low": 0},
    }

    r489_resolution_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_INVALID_PREPARATION_RECOVERY_R489_RESOLUTION_PLAN.md"
    )
    r489_resolution_path = repo / r489_resolution_relative
    r489_resolution_path.write_text("R489 resolution\n", encoding="utf-8")
    r489_resolution_commit = commit(
        [r489_resolution_relative], "R489 resolution"
    )
    r489_resolution = {
        "commit": r489_resolution_commit,
        "path": r489_resolution_relative,
        "sha256": file_sha256(r489_resolution_path),
    }
    r490_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "490_wave60_invalid_preparation_recovery_r489_resolution_plan_audit.md"
    )
    write_audit(
        r490_relative,
        audit_id="R490",
        scope="INVALID_PREPARATION_RECOVERY_R489_RESOLUTION_PLAN",
        target={
            "plan_commit": r489_resolution_commit,
            "plan_sha256": r489_resolution["sha256"],
        },
        verdict="PASS",
        findings={"high": 0, "medium": 0, "low": 0},
    )
    r490_commit = commit([r490_relative], "R490 PASS")
    r490 = {
        "commit": r490_commit,
        "path": r490_relative,
        "sha256": file_sha256(repo / r490_relative),
        "audit_id": "R490",
        "verdict": "PASS",
        "findings": {"high": 0, "medium": 0, "low": 0},
    }

    for relative in recovery_sources.values():
        (repo / relative).write_text(f"r491-revise:{relative}\n", encoding="utf-8")
    r489_resolution_implementation_commit = commit(
        list(recovery_sources.values()), "R489 resolution implementation"
    )
    r489_resolution_implementation = {
        "commit": r489_resolution_implementation_commit,
        "parent": r490_commit,
        "changed_sources": {
            label: {
                "path": relative,
                "old_sha256": old_hashes[relative],
                "new_sha256": file_sha256(repo / relative),
            }
            for label, relative in recovery_sources.items()
        },
    }
    r491_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "491_wave60_invalid_preparation_recovery_implementation_acceptance_audit.md"
    )
    write_audit(
        r491_relative,
        audit_id="R491",
        scope="INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        target={"implementation_commit": r489_resolution_implementation_commit},
        verdict="REVISE",
        findings={"high": 0, "medium": 1, "low": 0},
    )
    r491_commit = commit([r491_relative], "R491 REVISE")
    r491 = {
        "commit": r491_commit,
        "path": r491_relative,
        "sha256": file_sha256(repo / r491_relative),
        "audit_id": "R491",
        "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        "verdict": "REVISE",
        "findings": {"high": 0, "medium": 1, "low": 0},
    }

    r491_resolution_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_INVALID_PREPARATION_RECOVERY_R491_RESOLUTION_PLAN.md"
    )
    r491_resolution_path = repo / r491_resolution_relative
    r491_resolution_path.write_text("R491 resolution\n", encoding="utf-8")
    r491_resolution_commit = commit(
        [r491_resolution_relative], "R491 resolution"
    )
    r491_resolution = {
        "commit": r491_resolution_commit,
        "path": r491_resolution_relative,
        "sha256": file_sha256(r491_resolution_path),
    }
    r492_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "492_wave60_invalid_preparation_recovery_r491_resolution_plan_audit.md"
    )
    write_audit(
        r492_relative,
        audit_id="R492",
        scope="INVALID_PREPARATION_RECOVERY_R491_RESOLUTION_PLAN",
        target={
            "plan_commit": r491_resolution_commit,
            "plan_sha256": r491_resolution["sha256"],
        },
        verdict="PASS",
        findings={"high": 0, "medium": 0, "low": 0},
    )
    r492_commit = commit([r492_relative], "R492 PASS")
    r492 = {
        "commit": r492_commit,
        "path": r492_relative,
        "sha256": file_sha256(repo / r492_relative),
        "audit_id": "R492",
        "verdict": "PASS",
        "findings": {"high": 0, "medium": 0, "low": 0},
    }

    for relative in recovery_sources.values():
        (repo / relative).write_text(f"r493-pass:{relative}\n", encoding="utf-8")
    final_commit = commit(
        list(recovery_sources.values()), "accepted recovery implementation"
    )
    r493_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "493_wave60_invalid_preparation_recovery_implementation_acceptance_audit.md"
    )
    write_audit(
        r493_relative,
        audit_id="R493",
        scope="INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        target={"implementation_commit": final_commit},
        verdict="PASS",
        findings={"high": 0, "medium": 0, "low": 0},
    )
    r493_commit = commit([r493_relative], "R493 PASS")
    final = {
        "commit": final_commit,
        "audit_commit": r493_commit,
        "audit_path": r493_relative,
        "audit_sha256": file_sha256(repo / r493_relative),
        "audit_id": "R493",
        "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        "changed_sources": {
            label: {
                "path": relative,
                "old_sha256": old_hashes[relative],
                "new_sha256": file_sha256(repo / relative),
            }
            for label, relative in recovery_sources.items()
        },
        "unchanged_source_law_sources": list(preparer.WAVE60_SOURCE_LAW_SOURCES),
    }

    arguments = {
        "repo_root": repo,
        "r475_implementation_commit": r475,
        "r480_anchor": r480,
        "rejected_implementation": rejected,
        "r481_audit": r481,
        "resolution_plan": resolution,
        "r482_audit": r482,
        "r481_resolution_implementation": intermediate,
        "r483_audit": r483,
        "r483_resolution_plan": r483_resolution,
        "r484_audit": r484,
        "r483_resolution_implementation": r483_resolution_implementation,
        "r485_audit": r485,
        "r485_resolution_plan": r485_resolution,
        "r486_audit": r486,
        "r486_resolution_plan": r486_resolution,
        "r487_audit": r487,
        "r487_resolution_plan": r487_resolution,
        "r488_audit": r488,
        "r487_resolution_implementation": r487_resolution_implementation,
        "r489_audit": r489,
        "r489_resolution_plan": r489_resolution,
        "r490_audit": r490,
        "r489_resolution_implementation": r489_resolution_implementation,
        "r491_audit": r491,
        "r491_resolution_plan": r491_resolution,
        "r492_audit": r492,
        "final_implementation": final,
    }
    preparer.validate_wave60_invalid_preparation_implementation_suffix(**arguments)

    rejected_pass = deepcopy(arguments)
    rejected_pass["r481_audit"]["verdict"] = "PASS"
    with pytest.raises(RuntimeError, match="R481 REVISE"):
        preparer.validate_wave60_invalid_preparation_implementation_suffix(
            **rejected_pass
        )
    r483_pass = deepcopy(arguments)
    r483_pass["r483_audit"]["verdict"] = "PASS"
    with pytest.raises(RuntimeError, match="R483 REVISE"):
        preparer.validate_wave60_invalid_preparation_implementation_suffix(
            **r483_pass
        )
    for audit_key in (
        "r481_audit",
        "r483_audit",
        "r485_audit",
        "r486_audit",
        "r487_audit",
        "r489_audit",
        "r491_audit",
    ):
        findings_drift = deepcopy(arguments)
        findings_drift[audit_key]["findings"] = {
            "high": 0,
            "medium": 0,
            "low": 0,
        }
        with pytest.raises(RuntimeError, match="REVISE authority"):
            preparer.validate_wave60_invalid_preparation_implementation_suffix(
                **findings_drift
            )
    crossed = deepcopy(arguments)
    crossed["final_implementation"]["changed_sources"]["preparer"][
        "old_sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="accepted recovery preparer"):
        preparer.validate_wave60_invalid_preparation_implementation_suffix(**crossed)

    audit_field_drifts = (
        ("r481_audit", "audit_id", "R999"),
        ("r481_audit", "scope", "WRONG_SCOPE"),
        ("r481_audit", "verdict", "PASS"),
        ("r481_audit", "sha256", "0" * 64),
        ("r481_audit", "path", "wrong-r481.md"),
        ("r482_audit", "audit_id", "R999"),
        ("r482_audit", "findings", {"high": 0, "medium": 1, "low": 0}),
        ("r482_audit", "sha256", "0" * 64),
        ("r482_audit", "path", "wrong-r482.md"),
        ("r483_audit", "audit_id", "R999"),
        ("r483_audit", "scope", "WRONG_SCOPE"),
        ("r483_audit", "verdict", "PASS"),
        ("r483_audit", "sha256", "0" * 64),
        ("r483_audit", "path", "wrong-r483.md"),
        ("r484_audit", "audit_id", "R999"),
        ("r484_audit", "findings", {"high": 0, "medium": 1, "low": 0}),
        ("r484_audit", "sha256", "0" * 64),
        ("r484_audit", "path", "wrong-r484.md"),
        ("r485_audit", "audit_id", "R999"),
        ("r485_audit", "scope", "WRONG_SCOPE"),
        ("r485_audit", "verdict", "PASS"),
        ("r485_audit", "findings", {"high": 0, "medium": 0, "low": 0}),
        ("r485_audit", "sha256", "0" * 64),
        ("r485_audit", "path", "wrong-r485.md"),
        ("r486_audit", "audit_id", "R999"),
        ("r486_audit", "verdict", "PASS"),
        ("r486_audit", "findings", {"high": 0, "medium": 0, "low": 0}),
        ("r486_audit", "sha256", "0" * 64),
        ("r486_audit", "path", "wrong-r486.md"),
        ("r487_audit", "audit_id", "R999"),
        ("r487_audit", "verdict", "PASS"),
        ("r487_audit", "findings", {"high": 0, "medium": 0, "low": 0}),
        ("r487_audit", "sha256", "0" * 64),
        ("r487_audit", "path", "wrong-r487.md"),
        ("r488_audit", "audit_id", "R999"),
        ("r488_audit", "findings", {"high": 0, "medium": 1, "low": 0}),
        ("r488_audit", "sha256", "0" * 64),
        ("r488_audit", "path", "wrong-r488.md"),
        ("r489_audit", "audit_id", "R999"),
        ("r489_audit", "scope", "WRONG_SCOPE"),
        ("r489_audit", "verdict", "PASS"),
        ("r489_audit", "findings", {"high": 0, "medium": 0, "low": 0}),
        ("r489_audit", "sha256", "0" * 64),
        ("r489_audit", "path", "wrong-r489.md"),
        ("r490_audit", "audit_id", "R999"),
        ("r490_audit", "findings", {"high": 0, "medium": 1, "low": 0}),
        ("r490_audit", "sha256", "0" * 64),
        ("r490_audit", "path", "wrong-r490.md"),
        ("r491_audit", "audit_id", "R999"),
        ("r491_audit", "scope", "WRONG_SCOPE"),
        ("r491_audit", "verdict", "PASS"),
        ("r491_audit", "findings", {"high": 0, "medium": 0, "low": 0}),
        ("r491_audit", "sha256", "0" * 64),
        ("r491_audit", "path", "wrong-r491.md"),
        ("r492_audit", "audit_id", "R999"),
        ("r492_audit", "findings", {"high": 0, "medium": 1, "low": 0}),
        ("r492_audit", "sha256", "0" * 64),
        ("r492_audit", "path", "wrong-r492.md"),
        ("final_implementation", "audit_id", "R999"),
        ("final_implementation", "scope", "WRONG_SCOPE"),
        ("final_implementation", "audit_sha256", "0" * 64),
        ("final_implementation", "audit_path", "wrong-r493.md"),
    )
    for binding, field, value in audit_field_drifts:
        drift = deepcopy(arguments)
        drift[binding][field] = value
        with pytest.raises((RuntimeError, FileNotFoundError)):
            preparer.validate_wave60_invalid_preparation_implementation_suffix(
                **drift
            )

    document_drifts = (
        ("resolution_plan", "path", "wrong-r481-resolution.md"),
        ("resolution_plan", "sha256", "0" * 64),
        ("r483_resolution_plan", "path", "wrong-r483-resolution.md"),
        ("r483_resolution_plan", "sha256", "0" * 64),
        ("r485_resolution_plan", "path", "wrong-r485-resolution.md"),
        ("r485_resolution_plan", "sha256", "0" * 64),
        ("r486_resolution_plan", "path", "wrong-r486-resolution.md"),
        ("r486_resolution_plan", "sha256", "0" * 64),
        ("r487_resolution_plan", "path", "wrong-r487-resolution.md"),
        ("r487_resolution_plan", "sha256", "0" * 64),
        ("r489_resolution_plan", "path", "wrong-r489-resolution.md"),
        ("r489_resolution_plan", "sha256", "0" * 64),
        ("r491_resolution_plan", "path", "wrong-r491-resolution.md"),
        ("r491_resolution_plan", "sha256", "0" * 64),
    )
    for binding, field, value in document_drifts:
        drift = deepcopy(arguments)
        drift[binding][field] = value
        with pytest.raises((RuntimeError, FileNotFoundError)):
            preparer.validate_wave60_invalid_preparation_implementation_suffix(
                **drift
            )

    implementation_bindings = (
        "rejected_implementation",
        "r481_resolution_implementation",
        "r483_resolution_implementation",
        "r487_resolution_implementation",
        "r489_resolution_implementation",
        "final_implementation",
    )
    assert len(implementation_bindings) == 6
    for binding in implementation_bindings:
        for label in recovery_sources:
            for field, value in (
                ("old_sha256", "0" * 64),
                ("new_sha256", "1" * 64),
                ("path", "wrong-implementation-source.py"),
            ):
                drift = deepcopy(arguments)
                drift[binding]["changed_sources"][label][field] = value
                with pytest.raises(RuntimeError, match="source binding drifted"):
                    preparer.validate_wave60_invalid_preparation_implementation_suffix(
                        **drift
                    )
            for crossed_binding in implementation_bindings:
                if crossed_binding == binding:
                    continue
                crossed_sha = arguments[crossed_binding]["changed_sources"][label][
                    "new_sha256"
                ]
                own_sha = arguments[binding]["changed_sources"][label]["new_sha256"]
                assert crossed_sha != own_sha
                drift = deepcopy(arguments)
                drift[binding]["changed_sources"][label]["new_sha256"] = crossed_sha
                with pytest.raises(RuntimeError, match="source binding drifted"):
                    preparer.validate_wave60_invalid_preparation_implementation_suffix(
                        **drift
                    )
    scientific_cross = deepcopy(arguments)
    scientific_cross["final_implementation"][
        "unchanged_source_law_sources"
    ] = list(preparer.WAVE60_SOURCE_LAW_SOURCES[:-1])
    with pytest.raises(RuntimeError, match="accepted recovery implementation"):
        preparer.validate_wave60_invalid_preparation_implementation_suffix(
            **scientific_cross
        )

    def audit_authority(
        audit_id: str,
        scope: str,
        target: dict[str, str],
        verdict: str,
        findings: dict[str, int],
    ) -> dict[str, object]:
        return {
            "schema_version": "wave60-audit-authority-v1",
            "audit_id": audit_id,
            "scope": scope,
            "target": target,
            "technical_verdict": verdict,
            "findings": findings,
            "files_modified": False,
            "gpu_used_or_queried": False,
        }

    audit_specs = (
        (
            "r481_audit",
            rejected_commit,
            r481_relative,
            "commit",
            "path",
            "sha256",
            audit_authority(
                "R481",
                "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
                {"implementation_commit": rejected_commit},
                "REVISE",
                {"high": 0, "medium": 1, "low": 0},
            ),
        ),
        (
            "r482_audit",
            resolution_commit,
            r482_relative,
            "commit",
            "path",
            "sha256",
            audit_authority(
                "R482",
                "INVALID_PREPARATION_RECOVERY_R481_RESOLUTION_PLAN",
                {
                    "plan_commit": resolution_commit,
                    "plan_sha256": resolution["sha256"],
                },
                "PASS",
                {"high": 0, "medium": 0, "low": 0},
            ),
        ),
        (
            "r483_audit",
            intermediate_commit,
            r483_relative,
            "commit",
            "path",
            "sha256",
            audit_authority(
                "R483",
                "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
                {"implementation_commit": intermediate_commit},
                "REVISE",
                {"high": 0, "medium": 1, "low": 0},
            ),
        ),
        (
            "r484_audit",
            r483_resolution_commit,
            r484_relative,
            "commit",
            "path",
            "sha256",
            audit_authority(
                "R484",
                "INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN",
                {
                    "plan_commit": r483_resolution_commit,
                    "plan_sha256": r483_resolution["sha256"],
                },
                "PASS",
                {"high": 0, "medium": 0, "low": 0},
            ),
        ),
        (
            "r485_audit",
            r483_resolution_implementation_commit,
            r485_relative,
            "commit",
            "path",
            "sha256",
            audit_authority(
                "R485",
                "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
                {
                    "implementation_commit": (
                        r483_resolution_implementation_commit
                    )
                },
                "REVISE",
                {"high": 0, "medium": 1, "low": 0},
            ),
        ),
        (
            "r486_audit",
            r485_resolution_commit,
            r486_relative,
            "commit",
            "path",
            "sha256",
            audit_authority(
                "R486",
                "INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN",
                {
                    "plan_commit": r485_resolution_commit,
                    "plan_sha256": r485_resolution["sha256"],
                },
                "REVISE",
                {"high": 0, "medium": 2, "low": 0},
            ),
        ),
        (
            "r487_audit",
            r486_resolution_commit,
            r487_relative,
            "commit",
            "path",
            "sha256",
            audit_authority(
                "R487",
                "INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN",
                {
                    "plan_commit": r486_resolution_commit,
                    "plan_sha256": r486_resolution["sha256"],
                },
                "REVISE",
                {"high": 0, "medium": 1, "low": 0},
            ),
        ),
        (
            "r488_audit",
            r487_resolution_commit,
            r488_relative,
            "commit",
            "path",
            "sha256",
            audit_authority(
                "R488",
                "INVALID_PREPARATION_RECOVERY_R487_RESOLUTION_PLAN",
                {
                    "plan_commit": r487_resolution_commit,
                    "plan_sha256": r487_resolution["sha256"],
                },
                "PASS",
                {"high": 0, "medium": 0, "low": 0},
            ),
        ),
        (
            "r489_audit",
            r487_resolution_implementation_commit,
            r489_relative,
            "commit",
            "path",
            "sha256",
            audit_authority(
                "R489",
                "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
                {
                    "implementation_commit": (
                        r487_resolution_implementation_commit
                    )
                },
                "REVISE",
                {"high": 0, "medium": 1, "low": 0},
            ),
        ),
        (
            "r490_audit",
            r489_resolution_commit,
            r490_relative,
            "commit",
            "path",
            "sha256",
            audit_authority(
                "R490",
                "INVALID_PREPARATION_RECOVERY_R489_RESOLUTION_PLAN",
                {
                    "plan_commit": r489_resolution_commit,
                    "plan_sha256": r489_resolution["sha256"],
                },
                "PASS",
                {"high": 0, "medium": 0, "low": 0},
            ),
        ),
        (
            "r491_audit",
            r489_resolution_implementation_commit,
            r491_relative,
            "commit",
            "path",
            "sha256",
            audit_authority(
                "R491",
                "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
                {
                    "implementation_commit": (
                        r489_resolution_implementation_commit
                    )
                },
                "REVISE",
                {"high": 0, "medium": 1, "low": 0},
            ),
        ),
        (
            "r492_audit",
            r491_resolution_commit,
            r492_relative,
            "commit",
            "path",
            "sha256",
            audit_authority(
                "R492",
                "INVALID_PREPARATION_RECOVERY_R491_RESOLUTION_PLAN",
                {
                    "plan_commit": r491_resolution_commit,
                    "plan_sha256": r491_resolution["sha256"],
                },
                "PASS",
                {"high": 0, "medium": 0, "low": 0},
            ),
        ),
        (
            "final_implementation",
            final_commit,
            r493_relative,
            "audit_commit",
            "audit_path",
            "audit_sha256",
            audit_authority(
                "R493",
                "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
                {"implementation_commit": final_commit},
                "PASS",
                {"high": 0, "medium": 0, "low": 0},
            ),
        ),
    )
    assert len(audit_specs) == 13

    semantic_case_count = 0
    for (
        binding,
        parent,
        relative,
        commit_field,
        path_field,
        sha_field,
        authority,
    ) in audit_specs:
        for semantic_field in (
            "audit_id",
            "target",
            "scope",
            "technical_verdict",
            "findings",
        ):
            subprocess.run(
                ["git", "checkout", "-q", "--detach", parent],
                cwd=repo,
                check=True,
            )
            bad_authority = deepcopy(authority)
            if semantic_field == "audit_id":
                bad_authority[semantic_field] = "R999"
            elif semantic_field == "target":
                bad_authority[semantic_field] = {"wrong_target": "1"}
            elif semantic_field == "scope":
                bad_authority[semantic_field] = "WRONG_SCOPE"
            elif semantic_field == "technical_verdict":
                bad_authority[semantic_field] = (
                    "PASS"
                    if authority[semantic_field] == "REVISE"
                    else "REVISE"
                )
            else:
                bad_authority[semantic_field] = {
                    "high": 1,
                    "medium": 0,
                    "low": 0,
                }
            assert {
                key
                for key in authority
                if authority[key] != bad_authority[key]
            } == {semantic_field}
            write_audit(
                relative,
                audit_id=str(bad_authority["audit_id"]),
                scope=str(bad_authority["scope"]),
                target=bad_authority["target"],
                verdict=str(bad_authority["technical_verdict"]),
                findings=bad_authority["findings"],
            )
            bad_commit = commit(
                [relative], f"{authority['audit_id']} bad {semantic_field}"
            )
            assert preparer.git_changed_paths(repo, bad_commit) == {relative}
            assert subprocess.check_output(
                ["git", "rev-parse", f"{bad_commit}^"], cwd=repo, text=True
            ).strip() == parent
            case = deepcopy(arguments)
            case[binding][commit_field] = bad_commit
            case[binding][sha_field] = file_sha256(repo / relative)
            duplicate_field = {
                "audit_id": "audit_id",
                "scope": "scope",
                "technical_verdict": "verdict",
                "findings": "findings",
            }.get(semantic_field)
            if duplicate_field in case[binding]:
                case[binding][duplicate_field] = bad_authority[semantic_field]
            assert case[binding][path_field] == relative
            physical_sha256 = file_sha256(repo / relative)
            blob_sha256 = preparer.git_blob_sha256(repo, bad_commit, relative)
            assert blob_sha256 == physical_sha256 == case[binding][sha_field]
            with pytest.raises(RuntimeError):
                preparer.validate_wave60_invalid_preparation_implementation_suffix(
                    **case
                )
            semantic_case_count += 1
    assert semantic_case_count == 65

    subprocess.run(
        ["git", "checkout", "-q", "--detach", r493_commit], cwd=repo, check=True
    )

    alternate = deepcopy(arguments)
    subprocess.run(
        ["git", "checkout", "-q", "--detach", r475], cwd=repo, check=True
    )
    mutated_source = source_law_sources[0]
    (repo / mutated_source).write_text(
        "scientific source mutation\n", encoding="utf-8"
    )
    alternate_anchor = commit([mutated_source], "mutated scientific anchor")
    alternate["r480_anchor"] = alternate_anchor

    def alternate_implementation(
        binding: str, parent: str, source_commit: str
    ) -> str:
        subprocess.run(
            [
                "git",
                "checkout",
                source_commit,
                "--",
                *sorted(implementation_paths),
            ],
            cwd=repo,
            check=True,
        )
        new_commit = commit(
            sorted(implementation_paths), f"alternate {binding}"
        )
        alternate[binding]["commit"] = new_commit
        if "parent" in alternate[binding]:
            alternate[binding]["parent"] = parent
        for label, relative in recovery_sources.items():
            alternate[binding]["changed_sources"][label] = {
                "path": relative,
                "old_sha256": old_hashes[relative],
                "new_sha256": file_sha256(repo / relative),
            }
        return new_commit

    def alternate_plan(binding: str, source_commit: str) -> str:
        relative = alternate[binding]["path"]
        subprocess.run(
            ["git", "checkout", source_commit, "--", relative],
            cwd=repo,
            check=True,
        )
        new_commit = commit([relative], f"alternate {binding}")
        alternate[binding]["commit"] = new_commit
        alternate[binding]["sha256"] = file_sha256(repo / relative)
        return new_commit

    def alternate_audit(
        binding: str,
        *,
        relative: str,
        audit_id: str,
        scope: str,
        target: dict[str, str],
        verdict: str,
        findings: dict[str, int],
        final_audit: bool = False,
    ) -> str:
        write_audit(
            relative,
            audit_id=audit_id,
            scope=scope,
            target=target,
            verdict=verdict,
            findings=findings,
        )
        new_commit = commit([relative], f"alternate {audit_id}")
        if final_audit:
            alternate[binding]["audit_commit"] = new_commit
            alternate[binding]["audit_path"] = relative
            alternate[binding]["audit_sha256"] = file_sha256(repo / relative)
        else:
            alternate[binding]["commit"] = new_commit
            alternate[binding]["path"] = relative
            alternate[binding]["sha256"] = file_sha256(repo / relative)
            alternate[binding]["audit_id"] = audit_id
            alternate[binding]["verdict"] = verdict
            alternate[binding]["findings"] = findings
            if "scope" in alternate[binding]:
                alternate[binding]["scope"] = scope
        return new_commit

    alt_rejected = alternate_implementation(
        "rejected_implementation", alternate_anchor, rejected_commit
    )
    alt_r481 = alternate_audit(
        "r481_audit",
        relative=r481_relative,
        audit_id="R481",
        scope="INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        target={"implementation_commit": alt_rejected},
        verdict="REVISE",
        findings={"high": 0, "medium": 1, "low": 0},
    )
    alt_resolution = alternate_plan("resolution_plan", resolution_commit)
    alt_r482 = alternate_audit(
        "r482_audit",
        relative=r482_relative,
        audit_id="R482",
        scope="INVALID_PREPARATION_RECOVERY_R481_RESOLUTION_PLAN",
        target={
            "plan_commit": alt_resolution,
            "plan_sha256": alternate["resolution_plan"]["sha256"],
        },
        verdict="PASS",
        findings={"high": 0, "medium": 0, "low": 0},
    )
    alt_intermediate = alternate_implementation(
        "r481_resolution_implementation", alt_r482, intermediate_commit
    )
    alt_r483 = alternate_audit(
        "r483_audit",
        relative=r483_relative,
        audit_id="R483",
        scope="INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        target={"implementation_commit": alt_intermediate},
        verdict="REVISE",
        findings={"high": 0, "medium": 1, "low": 0},
    )
    alt_r483_resolution = alternate_plan(
        "r483_resolution_plan", r483_resolution_commit
    )
    alt_r484 = alternate_audit(
        "r484_audit",
        relative=r484_relative,
        audit_id="R484",
        scope="INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN",
        target={
            "plan_commit": alt_r483_resolution,
            "plan_sha256": alternate["r483_resolution_plan"]["sha256"],
        },
        verdict="PASS",
        findings={"high": 0, "medium": 0, "low": 0},
    )
    alt_r483_implementation = alternate_implementation(
        "r483_resolution_implementation",
        alt_r484,
        r483_resolution_implementation_commit,
    )
    alt_r485 = alternate_audit(
        "r485_audit",
        relative=r485_relative,
        audit_id="R485",
        scope="INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        target={"implementation_commit": alt_r483_implementation},
        verdict="REVISE",
        findings={"high": 0, "medium": 1, "low": 0},
    )
    alt_r485_resolution = alternate_plan(
        "r485_resolution_plan", r485_resolution_commit
    )
    alt_r486 = alternate_audit(
        "r486_audit",
        relative=r486_relative,
        audit_id="R486",
        scope="INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN",
        target={
            "plan_commit": alt_r485_resolution,
            "plan_sha256": alternate["r485_resolution_plan"]["sha256"],
        },
        verdict="REVISE",
        findings={"high": 0, "medium": 2, "low": 0},
    )
    alt_r486_resolution = alternate_plan(
        "r486_resolution_plan", r486_resolution_commit
    )
    alt_r487 = alternate_audit(
        "r487_audit",
        relative=r487_relative,
        audit_id="R487",
        scope="INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN",
        target={
            "plan_commit": alt_r486_resolution,
            "plan_sha256": alternate["r486_resolution_plan"]["sha256"],
        },
        verdict="REVISE",
        findings={"high": 0, "medium": 1, "low": 0},
    )
    alt_r487_resolution = alternate_plan(
        "r487_resolution_plan", r487_resolution_commit
    )
    alt_r488 = alternate_audit(
        "r488_audit",
        relative=r488_relative,
        audit_id="R488",
        scope="INVALID_PREPARATION_RECOVERY_R487_RESOLUTION_PLAN",
        target={
            "plan_commit": alt_r487_resolution,
            "plan_sha256": alternate["r487_resolution_plan"]["sha256"],
        },
        verdict="PASS",
        findings={"high": 0, "medium": 0, "low": 0},
    )
    alt_r487_implementation = alternate_implementation(
        "r487_resolution_implementation",
        alt_r488,
        r487_resolution_implementation_commit,
    )
    alt_r489 = alternate_audit(
        "r489_audit",
        relative=r489_relative,
        audit_id="R489",
        scope="INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        target={"implementation_commit": alt_r487_implementation},
        verdict="REVISE",
        findings={"high": 0, "medium": 1, "low": 0},
    )
    alt_r489_resolution = alternate_plan(
        "r489_resolution_plan", r489_resolution_commit
    )
    alt_r490 = alternate_audit(
        "r490_audit",
        relative=r490_relative,
        audit_id="R490",
        scope="INVALID_PREPARATION_RECOVERY_R489_RESOLUTION_PLAN",
        target={
            "plan_commit": alt_r489_resolution,
            "plan_sha256": alternate["r489_resolution_plan"]["sha256"],
        },
        verdict="PASS",
        findings={"high": 0, "medium": 0, "low": 0},
    )
    alt_r489_implementation = alternate_implementation(
        "r489_resolution_implementation",
        alt_r490,
        r489_resolution_implementation_commit,
    )
    alt_r491 = alternate_audit(
        "r491_audit",
        relative=r491_relative,
        audit_id="R491",
        scope="INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        target={"implementation_commit": alt_r489_implementation},
        verdict="REVISE",
        findings={"high": 0, "medium": 1, "low": 0},
    )
    alt_r491_resolution = alternate_plan(
        "r491_resolution_plan", r491_resolution_commit
    )
    alt_r492 = alternate_audit(
        "r492_audit",
        relative=r492_relative,
        audit_id="R492",
        scope="INVALID_PREPARATION_RECOVERY_R491_RESOLUTION_PLAN",
        target={
            "plan_commit": alt_r491_resolution,
            "plan_sha256": alternate["r491_resolution_plan"]["sha256"],
        },
        verdict="PASS",
        findings={"high": 0, "medium": 0, "low": 0},
    )
    alt_final = alternate_implementation(
        "final_implementation", alt_r492, final_commit
    )
    alternate_audit(
        "final_implementation",
        relative=r493_relative,
        audit_id="R493",
        scope="INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        target={"implementation_commit": alt_final},
        verdict="PASS",
        findings={"high": 0, "medium": 0, "low": 0},
        final_audit=True,
    )
    assert preparer.git_changed_paths(repo, alt_final) == implementation_paths
    assert preparer.git_blob_sha256(
        repo, alt_final, mutated_source
    ) != preparer.git_blob_sha256(repo, r475, mutated_source)
    with pytest.raises(RuntimeError, match="crossed scientific source"):
        preparer.validate_wave60_invalid_preparation_implementation_suffix(
            **alternate
        )
    subprocess.run(
        ["git", "checkout", "-q", "--detach", r493_commit], cwd=repo, check=True
    )
    alternate_steps = (
        {
            "kind": "implementation",
            "binding": "rejected_implementation",
            "source_commit": rejected_commit,
        },
        {
            "kind": "audit",
            "binding": "r481_audit",
            "relative": r481_relative,
            "audit_id": "R481",
            "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
            "target_binding": "rejected_implementation",
            "target_kind": "implementation",
            "verdict": "REVISE",
            "findings": {"high": 0, "medium": 1, "low": 0},
        },
        {
            "kind": "plan",
            "binding": "resolution_plan",
            "source_commit": resolution_commit,
        },
        {
            "kind": "audit",
            "binding": "r482_audit",
            "relative": r482_relative,
            "audit_id": "R482",
            "scope": "INVALID_PREPARATION_RECOVERY_R481_RESOLUTION_PLAN",
            "target_binding": "resolution_plan",
            "target_kind": "plan",
            "verdict": "PASS",
            "findings": {"high": 0, "medium": 0, "low": 0},
        },
        {
            "kind": "implementation",
            "binding": "r481_resolution_implementation",
            "source_commit": intermediate_commit,
        },
        {
            "kind": "audit",
            "binding": "r483_audit",
            "relative": r483_relative,
            "audit_id": "R483",
            "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
            "target_binding": "r481_resolution_implementation",
            "target_kind": "implementation",
            "verdict": "REVISE",
            "findings": {"high": 0, "medium": 1, "low": 0},
        },
        {
            "kind": "plan",
            "binding": "r483_resolution_plan",
            "source_commit": r483_resolution_commit,
        },
        {
            "kind": "audit",
            "binding": "r484_audit",
            "relative": r484_relative,
            "audit_id": "R484",
            "scope": "INVALID_PREPARATION_RECOVERY_R483_RESOLUTION_PLAN",
            "target_binding": "r483_resolution_plan",
            "target_kind": "plan",
            "verdict": "PASS",
            "findings": {"high": 0, "medium": 0, "low": 0},
        },
        {
            "kind": "implementation",
            "binding": "r483_resolution_implementation",
            "source_commit": r483_resolution_implementation_commit,
        },
        {
            "kind": "audit",
            "binding": "r485_audit",
            "relative": r485_relative,
            "audit_id": "R485",
            "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
            "target_binding": "r483_resolution_implementation",
            "target_kind": "implementation",
            "verdict": "REVISE",
            "findings": {"high": 0, "medium": 1, "low": 0},
        },
        {
            "kind": "plan",
            "binding": "r485_resolution_plan",
            "source_commit": r485_resolution_commit,
        },
        {
            "kind": "audit",
            "binding": "r486_audit",
            "relative": r486_relative,
            "audit_id": "R486",
            "scope": "INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN",
            "target_binding": "r485_resolution_plan",
            "target_kind": "plan",
            "verdict": "REVISE",
            "findings": {"high": 0, "medium": 2, "low": 0},
        },
        {
            "kind": "plan",
            "binding": "r486_resolution_plan",
            "source_commit": r486_resolution_commit,
        },
        {
            "kind": "audit",
            "binding": "r487_audit",
            "relative": r487_relative,
            "audit_id": "R487",
            "scope": "INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN",
            "target_binding": "r486_resolution_plan",
            "target_kind": "plan",
            "verdict": "REVISE",
            "findings": {"high": 0, "medium": 1, "low": 0},
        },
        {
            "kind": "plan",
            "binding": "r487_resolution_plan",
            "source_commit": r487_resolution_commit,
        },
        {
            "kind": "audit",
            "binding": "r488_audit",
            "relative": r488_relative,
            "audit_id": "R488",
            "scope": "INVALID_PREPARATION_RECOVERY_R487_RESOLUTION_PLAN",
            "target_binding": "r487_resolution_plan",
            "target_kind": "plan",
            "verdict": "PASS",
            "findings": {"high": 0, "medium": 0, "low": 0},
        },
        {
            "kind": "implementation",
            "binding": "r487_resolution_implementation",
            "source_commit": r487_resolution_implementation_commit,
        },
        {
            "kind": "audit",
            "binding": "r489_audit",
            "relative": r489_relative,
            "audit_id": "R489",
            "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
            "target_binding": "r487_resolution_implementation",
            "target_kind": "implementation",
            "verdict": "REVISE",
            "findings": {"high": 0, "medium": 1, "low": 0},
        },
        {
            "kind": "plan",
            "binding": "r489_resolution_plan",
            "source_commit": r489_resolution_commit,
        },
        {
            "kind": "audit",
            "binding": "r490_audit",
            "relative": r490_relative,
            "audit_id": "R490",
            "scope": "INVALID_PREPARATION_RECOVERY_R489_RESOLUTION_PLAN",
            "target_binding": "r489_resolution_plan",
            "target_kind": "plan",
            "verdict": "PASS",
            "findings": {"high": 0, "medium": 0, "low": 0},
        },
        {
            "kind": "implementation",
            "binding": "r489_resolution_implementation",
            "source_commit": r489_resolution_implementation_commit,
        },
        {
            "kind": "audit",
            "binding": "r491_audit",
            "relative": r491_relative,
            "audit_id": "R491",
            "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
            "target_binding": "r489_resolution_implementation",
            "target_kind": "implementation",
            "verdict": "REVISE",
            "findings": {"high": 0, "medium": 1, "low": 0},
        },
        {
            "kind": "plan",
            "binding": "r491_resolution_plan",
            "source_commit": r491_resolution_commit,
        },
        {
            "kind": "audit",
            "binding": "r492_audit",
            "relative": r492_relative,
            "audit_id": "R492",
            "scope": "INVALID_PREPARATION_RECOVERY_R491_RESOLUTION_PLAN",
            "target_binding": "r491_resolution_plan",
            "target_kind": "plan",
            "verdict": "PASS",
            "findings": {"high": 0, "medium": 0, "low": 0},
        },
        {
            "kind": "implementation",
            "binding": "final_implementation",
            "source_commit": final_commit,
        },
        {
            "kind": "audit",
            "binding": "final_implementation",
            "relative": r493_relative,
            "audit_id": "R493",
            "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
            "target_binding": "final_implementation",
            "target_kind": "implementation",
            "verdict": "PASS",
            "findings": {"high": 0, "medium": 0, "low": 0},
            "final_audit": True,
        },
    )
    assert len(alternate_steps) == 26
    for mutated_source in source_law_sources[1:]:
        alternate = deepcopy(arguments)
        subprocess.run(
            ["git", "checkout", "-q", "--detach", r475],
            cwd=repo,
            check=True,
        )
        (repo / mutated_source).write_text(
            f"scientific source mutation:{mutated_source}\n", encoding="utf-8"
        )
        current_parent = commit(
            [mutated_source], f"mutated scientific anchor {mutated_source}"
        )
        alternate["r480_anchor"] = current_parent
        for step in alternate_steps:
            if step["kind"] == "implementation":
                current_parent = alternate_implementation(
                    step["binding"], current_parent, step["source_commit"]
                )
            elif step["kind"] == "plan":
                current_parent = alternate_plan(
                    step["binding"], step["source_commit"]
                )
            else:
                target_binding = alternate[step["target_binding"]]
                if step["target_kind"] == "implementation":
                    target = {
                        "implementation_commit": target_binding["commit"]
                    }
                else:
                    target = {
                        "plan_commit": target_binding["commit"],
                        "plan_sha256": target_binding["sha256"],
                    }
                current_parent = alternate_audit(
                    step["binding"],
                    relative=step["relative"],
                    audit_id=step["audit_id"],
                    scope=step["scope"],
                    target=target,
                    verdict=step["verdict"],
                    findings=step["findings"],
                    final_audit=step.get("final_audit", False),
                )
        alternate_final = alternate["final_implementation"]["commit"]
        assert preparer.git_changed_paths(
            repo, alternate_final
        ) == implementation_paths
        for relative in source_law_sources:
            blobs_are_equal = preparer.git_blob_sha256(
                repo, alternate_final, relative
            ) == preparer.git_blob_sha256(repo, r475, relative)
            assert blobs_are_equal is (relative != mutated_source)
        with pytest.raises(RuntimeError, match="crossed scientific source"):
            preparer.validate_wave60_invalid_preparation_implementation_suffix(
                **alternate
            )
    subprocess.run(
        ["git", "checkout", "-q", "--detach", r493_commit], cwd=repo, check=True
    )
    for (
        binding,
        parent,
        _relative,
        commit_field,
        path_field,
        sha_field,
        _authority,
    ) in audit_specs:
        for field, value in (
            (commit_field, parent),
            (path_field, "wrong-audit-report.md"),
            (sha_field, "0" * 64),
        ):
            case = deepcopy(arguments)
            case[binding][field] = value
            with pytest.raises(
                (RuntimeError, FileNotFoundError, subprocess.CalledProcessError)
            ):
                preparer.validate_wave60_invalid_preparation_implementation_suffix(
                    **case
                )

    for binding, relative, sha_field in (
        ("r485_resolution_plan", r485_resolution_relative, "sha256"),
        ("r487_audit", r487_relative, "sha256"),
    ):
        subprocess.run(
            ["git", "checkout", "-q", "--detach", r493_commit],
            cwd=repo,
            check=True,
        )
        path = repo / relative
        original = path.read_bytes()
        path.write_bytes(original + b"\nphysical drift\n")
        drift_head = commit([relative], f"physical drift for {binding}")
        case = deepcopy(arguments)
        case[binding][sha_field] = file_sha256(path)
        assert preparer.git_blob_sha256(repo, drift_head, relative) == case[
            binding
        ][sha_field]
        assert (
            preparer.git_blob_sha256(repo, case[binding]["commit"], relative)
            != case[binding][sha_field]
        )
        with pytest.raises(RuntimeError, match="blob differs"):
            preparer.validate_wave60_invalid_preparation_implementation_suffix(
                **case
            )
    subprocess.run(
        ["git", "checkout", "-q", "--detach", r493_commit], cwd=repo, check=True
    )

    def commit_tree(tree_source: str, parent: str, message: str) -> str:
        tree = subprocess.check_output(
            ["git", "rev-parse", f"{tree_source}^{{tree}}"],
            cwd=repo,
            text=True,
        ).strip()
        return subprocess.check_output(
            ["git", "commit-tree", tree, "-p", parent],
            cwd=repo,
            input=message + "\n",
            text=True,
        ).strip()

    extra_relative = "unexpected-exclusive-path.txt"
    implementation_paths = set(recovery_sources.values())
    suffix_steps = (
        (r480, rejected_commit, "rejected_implementation", "commit", implementation_paths),
        (rejected_commit, r481_commit, "r481_audit", "commit", {r481_relative}),
        (r481_commit, resolution_commit, "resolution_plan", "commit", {resolution_relative}),
        (resolution_commit, r482_commit, "r482_audit", "commit", {r482_relative}),
        (
            r482_commit,
            intermediate_commit,
            "r481_resolution_implementation",
            "commit",
            implementation_paths,
        ),
        (intermediate_commit, r483_commit, "r483_audit", "commit", {r483_relative}),
        (
            r483_commit,
            r483_resolution_commit,
            "r483_resolution_plan",
            "commit",
            {r483_resolution_relative},
        ),
        (
            r483_resolution_commit,
            r484_commit,
            "r484_audit",
            "commit",
            {r484_relative},
        ),
        (
            r484_commit,
            r483_resolution_implementation_commit,
            "r483_resolution_implementation",
            "commit",
            implementation_paths,
        ),
        (
            r483_resolution_implementation_commit,
            r485_commit,
            "r485_audit",
            "commit",
            {r485_relative},
        ),
        (
            r485_commit,
            r485_resolution_commit,
            "r485_resolution_plan",
            "commit",
            {r485_resolution_relative},
        ),
        (
            r485_resolution_commit,
            r486_commit,
            "r486_audit",
            "commit",
            {r486_relative},
        ),
        (
            r486_commit,
            r486_resolution_commit,
            "r486_resolution_plan",
            "commit",
            {r486_resolution_relative},
        ),
        (
            r486_resolution_commit,
            r487_commit,
            "r487_audit",
            "commit",
            {r487_relative},
        ),
        (
            r487_commit,
            r487_resolution_commit,
            "r487_resolution_plan",
            "commit",
            {r487_resolution_relative},
        ),
        (
            r487_resolution_commit,
            r488_commit,
            "r488_audit",
            "commit",
            {r488_relative},
        ),
        (
            r488_commit,
            r487_resolution_implementation_commit,
            "r487_resolution_implementation",
            "commit",
            implementation_paths,
        ),
        (
            r487_resolution_implementation_commit,
            r489_commit,
            "r489_audit",
            "commit",
            {r489_relative},
        ),
        (
            r489_commit,
            r489_resolution_commit,
            "r489_resolution_plan",
            "commit",
            {r489_resolution_relative},
        ),
        (
            r489_resolution_commit,
            r490_commit,
            "r490_audit",
            "commit",
            {r490_relative},
        ),
        (
            r490_commit,
            r489_resolution_implementation_commit,
            "r489_resolution_implementation",
            "commit",
            implementation_paths,
        ),
        (
            r489_resolution_implementation_commit,
            r491_commit,
            "r491_audit",
            "commit",
            {r491_relative},
        ),
        (
            r491_commit,
            r491_resolution_commit,
            "r491_resolution_plan",
            "commit",
            {r491_resolution_relative},
        ),
        (
            r491_resolution_commit,
            r492_commit,
            "r492_audit",
            "commit",
            {r492_relative},
        ),
        (
            r492_commit,
            final_commit,
            "final_implementation",
            "commit",
            implementation_paths,
        ),
        (
            final_commit,
            r493_commit,
            "final_implementation",
            "audit_commit",
            {r493_relative},
        ),
    )
    assert len(suffix_steps) == 26
    extra_path_cases = []
    for parent, correct, binding, commit_field, expected_paths in suffix_steps:
        subprocess.run(
            ["git", "checkout", "-q", "--detach", parent], cwd=repo, check=True
        )
        subprocess.run(
            ["git", "checkout", correct, "--", *sorted(expected_paths)],
            cwd=repo,
            check=True,
        )
        (repo / extra_relative).write_text("must be rejected\n", encoding="utf-8")
        bad_commit = commit(
            [*sorted(expected_paths), extra_relative],
            f"{binding} with one unexpected path",
        )
        assert preparer.git_changed_paths(repo, bad_commit) == expected_paths | {
            extra_relative
        }
        assert subprocess.check_output(
            ["git", "rev-parse", f"{bad_commit}^"], cwd=repo, text=True
        ).strip() == parent
        case = deepcopy(arguments)
        case[binding][commit_field] = bad_commit
        extra_path_cases.append((bad_commit, case))

    for checkout, drift in extra_path_cases:
        subprocess.run(
            ["git", "checkout", "-q", "--detach", checkout], cwd=repo, check=True
        )
        with pytest.raises(RuntimeError):
            preparer.validate_wave60_invalid_preparation_implementation_suffix(
                **drift
            )

    skipped_cases = []
    for parent, correct, binding, commit_field, expected_paths in suffix_steps:
        hidden = commit_tree(parent, parent, f"hidden before {binding}")
        bad_commit = commit_tree(
            correct, hidden, f"{binding} with skipped parent"
        )
        assert preparer.git_changed_paths(repo, bad_commit) == expected_paths
        case = deepcopy(arguments)
        case[binding][commit_field] = bad_commit
        skipped_cases.append((bad_commit, case))

    for checkout, skipped in skipped_cases:
        subprocess.run(
            ["git", "checkout", "-q", "--detach", checkout], cwd=repo, check=True
        )
        with pytest.raises(RuntimeError, match="directly descend"):
            preparer.validate_wave60_invalid_preparation_implementation_suffix(
                **skipped
            )
    subprocess.run(
        ["git", "checkout", "-q", "--detach", r493_commit], cwd=repo, check=True
    )


def test_invalid_preparation_final_config_partitions_r475_and_r493(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "wave60@test.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Wave 60 Test"], cwd=repo, check=True
    )

    def commit(paths: list[str], message: str) -> str:
        subprocess.run(["git", "add", "-f", "--", *paths], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", message], cwd=repo, check=True)
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()

    source_law_sources = list(preparer.WAVE60_SOURCE_LAW_SOURCES)
    recovery_sources = dict(preparer.WAVE60_RECOVERY_IMPLEMENTATION_SOURCES)
    for relative in [*source_law_sources, *recovery_sources.values()]:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"r475:{relative}\n", encoding="utf-8")
    r475 = commit(
        [*source_law_sources, *recovery_sources.values()], "R475 implementation"
    )
    old_hashes = {
        relative: file_sha256(repo / relative)
        for relative in recovery_sources.values()
    }
    for relative in recovery_sources.values():
        (repo / relative).write_text(f"r493:{relative}\n", encoding="utf-8")
    implementation_commit = commit(
        list(recovery_sources.values()), "invalid-preparation implementation"
    )
    implementation_audit_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "493_wave60_invalid_preparation_recovery_implementation_acceptance_audit.md"
    )
    implementation_audit = repo / implementation_audit_relative
    implementation_audit.parent.mkdir(parents=True, exist_ok=True)
    implementation_authority = {
        "schema_version": "wave60-audit-authority-v1",
        "audit_id": "R493",
        "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        "target": {"implementation_commit": implementation_commit},
        "technical_verdict": "PASS",
        "findings": {"high": 0, "medium": 0, "low": 0},
        "files_modified": False,
        "gpu_used_or_queried": False,
    }
    implementation_audit.write_text(
        "# R493\n\n```json\n"
        + json.dumps(implementation_authority, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    implementation_audit_commit = commit(
        [implementation_audit_relative], "R493 audit"
    )
    recovery_implementation = {
        "commit": implementation_commit,
        "audit_commit": implementation_audit_commit,
        "audit_path": implementation_audit_relative,
        "audit_sha256": file_sha256(implementation_audit),
        "audit_id": "R493",
        "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
        "changed_sources": {
            label: {
                "path": relative,
                "old_sha256": old_hashes[relative],
                "new_sha256": file_sha256(repo / relative),
            }
            for label, relative in recovery_sources.items()
        },
        "unchanged_source_law_sources": source_law_sources,
    }
    amendment_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_INVALID_PREPARATION_RECOVERY_V2_AMENDMENT.json"
    )
    amendment = repo / amendment_relative
    amendment.parent.mkdir(parents=True)
    amendment.write_text(
        json.dumps(
            {
                "schema_version": (
                    preparer.WAVE60_INVALID_PREPARATION_RECOVERY_AMENDMENT_SCHEMA
                ),
                "recovery_implementation": recovery_implementation,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    commit([amendment_relative], "recovery amendment")
    amendment_audit_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "494_wave60_invalid_preparation_recovery_amendment_audit.md"
    )
    amendment_audit = repo / amendment_audit_relative
    amendment_audit.write_text("R494 PASS\n", encoding="utf-8")
    amendment_audit_commit = commit([amendment_audit_relative], "R494 audit")

    config = valid_config()
    config["final_audit"]["audit_id"] = "R495"
    config["implementation_binding"]["commit"] = r475
    config["attempt"]["recovery"] = {
        "amendment_path": amendment_relative,
        "amendment_sha256": file_sha256(amendment),
        "amendment_audit_commit": amendment_audit_commit,
    }
    config["source_sha256"].update(
        {
            relative: file_sha256(repo / relative)
            for relative in [*source_law_sources, *recovery_sources.values()]
        }
    )
    config_relative = (
        "experiments/geometria_proporcional/configs/"
        "wave60_frozen_policy_transport.json"
    )
    config_path = repo / config_relative
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(config_path, config)
    config_commit = commit([config_relative], "v2 config")
    final_audit = repo / config["final_audit"]["audit_path"]
    final_audit.parent.mkdir(parents=True, exist_ok=True)
    final_authority = {
        "schema_version": "wave60-audit-authority-v1",
        "audit_id": config["final_audit"]["audit_id"],
        "scope": "CONFIG",
        "target": {
            "config_commit": config_commit,
            "config_sha256": file_sha256(config_path),
        },
        "technical_verdict": "PASS",
        "findings": {"high": 0, "medium": 0, "low": 0},
        "files_modified": False,
        "gpu_used_or_queried": False,
    }
    final_audit.write_text(
        "# R495\n\n```json\n"
        + json.dumps(final_authority, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    head = commit([config["final_audit"]["audit_path"]], "R495 audit")
    preparer.validate_wave60_final_config_authority(
        repo, config_path, config, head
    )
    crossed = deepcopy(config)
    crossed["source_sha256"][recovery_sources["preparer"]] = old_hashes[
        recovery_sources["preparer"]
    ]
    with pytest.raises(RuntimeError, match="executed blob differs"):
        preparer.validate_wave60_final_config_authority(
            repo, config_path, crossed, head
        )


def test_static_protocol_final_config_partitions_three_changed_sources(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "wave60@test.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Wave 60 Test"], cwd=repo, check=True
    )

    def commit(paths: list[str], message: str) -> str:
        subprocess.run(["git", "add", "-f", "--", *paths], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", message], cwd=repo, check=True)
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()

    changed_sources = dict(preparer.WAVE60_STATIC_PROTOCOL_RECOVERY_SOURCES)
    implementation_sources = sorted(
        {
            *preparer.WAVE60_SOURCE_LAW_SOURCES,
            *preparer.WAVE60_RECOVERY_IMPLEMENTATION_SOURCES.values(),
        }
    )
    for relative in implementation_sources:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"baseline:{relative}\n", encoding="utf-8")
    baseline_commit = commit(implementation_sources, "baseline implementation")
    old_hashes = {
        relative: file_sha256(repo / relative)
        for relative in changed_sources.values()
    }
    config_relative = (
        "experiments/geometria_proporcional/configs/"
        "wave60_frozen_policy_transport.json"
    )
    prior_config = valid_config()
    prior_config["implementation_binding"]["commit"] = baseline_commit
    prior_config["attempt"]["version"] = 2
    prior_config["attempt"]["container"] = "data/prior-v2"
    prior_config["source_sha256"].update(
        {
            relative: file_sha256(repo / relative)
            for relative in implementation_sources
        }
    )
    prior_config_path = repo / config_relative
    prior_config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(prior_config_path, prior_config)
    commit([config_relative], "prior v2 config")
    prior_audit_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "495_wave60_prior_config_audit.md"
    )
    prior_audit = repo / prior_audit_relative
    prior_audit.parent.mkdir(parents=True)
    prior_audit.write_text("R495 PASS\n", encoding="utf-8")
    prior_audit_commit = commit([prior_audit_relative], "R495 prior audit")
    for relative in changed_sources.values():
        (repo / relative).write_text(f"R498:{relative}\n", encoding="utf-8")
    implementation_commit = commit(
        list(changed_sources.values()), "static-protocol implementation"
    )
    implementation_audit_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "498_wave60_static_protocol_identity_guard_implementation_audit.md"
    )
    implementation_audit = repo / implementation_audit_relative
    implementation_audit.parent.mkdir(parents=True, exist_ok=True)
    implementation_authority = {
        "schema_version": "wave60-audit-authority-v1",
        "audit_id": "R498",
        "scope": "STATIC_PROTOCOL_IDENTITY_GUARD_RECOVERY_IMPLEMENTATION",
        "target": {"implementation_commit": implementation_commit},
        "technical_verdict": "PASS",
        "findings": {"high": 0, "medium": 0, "low": 0},
        "files_modified": False,
        "gpu_used_or_queried": False,
    }
    implementation_audit.write_text(
        "# R498\n\n```json\n"
        + json.dumps(implementation_authority, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    implementation_audit_commit = commit(
        [implementation_audit_relative], "R498 audit"
    )
    recovery_implementation = {
        "commit": implementation_commit,
        "audit_commit": implementation_audit_commit,
        "audit_path": implementation_audit_relative,
        "audit_sha256": file_sha256(implementation_audit),
        "audit_id": "R498",
        "scope": "STATIC_PROTOCOL_IDENTITY_GUARD_RECOVERY_IMPLEMENTATION",
        "changed_sources": {
            label: {
                "path": relative,
                "old_sha256": old_hashes[relative],
                "new_sha256": file_sha256(repo / relative),
            }
            for label, relative in changed_sources.items()
        },
        "unchanged_source_law_sources": list(
            preparer.WAVE60_STATIC_PROTOCOL_UNCHANGED_SOURCES
        ),
    }
    amendment_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/"
        "WAVE_60_STATIC_PROTOCOL_IDENTITY_GUARD_RECOVERY_V3_AMENDMENT.json"
    )
    amendment = repo / amendment_relative
    amendment.parent.mkdir(parents=True)
    _write_json(
        amendment,
        {
            "schema_version": preparer.WAVE60_STATIC_PROTOCOL_RECOVERY_AMENDMENT_SCHEMA,
            "recovery_implementation": recovery_implementation,
        },
    )
    commit([amendment_relative], "static-protocol amendment")
    amendment_audit_relative = (
        "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "499_wave60_static_protocol_identity_guard_amendment_audit.md"
    )
    amendment_audit = repo / amendment_audit_relative
    amendment_audit.write_text("R499 PASS\n", encoding="utf-8")
    amendment_audit_commit = commit([amendment_audit_relative], "R499 audit")

    config = valid_config()
    config["final_audit"]["audit_id"] = "R500"
    config["implementation_binding"]["commit"] = baseline_commit
    config["attempt"]["recovery"] = {
        "amendment_path": amendment_relative,
        "amendment_sha256": file_sha256(amendment),
        "amendment_audit_commit": amendment_audit_commit,
        "prior_config_audit_commit": prior_audit_commit,
    }
    config["source_sha256"].update(
        {
            relative: file_sha256(repo / relative)
            for relative in implementation_sources
        }
    )
    config_path = repo / config_relative
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(config_path, config)
    config_commit = commit([config_relative], "v3 config")
    final_audit = repo / config["final_audit"]["audit_path"]
    final_audit.parent.mkdir(parents=True, exist_ok=True)
    final_authority = {
        "schema_version": "wave60-audit-authority-v1",
        "audit_id": "R500",
        "scope": "CONFIG",
        "target": {
            "config_commit": config_commit,
            "config_sha256": file_sha256(config_path),
        },
        "technical_verdict": "PASS",
        "findings": {"high": 0, "medium": 0, "low": 0},
        "files_modified": False,
        "gpu_used_or_queried": False,
    }
    final_audit.write_text(
        "# R500\n\n```json\n"
        + json.dumps(final_authority, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    head = commit([config["final_audit"]["audit_path"]], "R500 audit")
    preparer.validate_wave60_final_config_authority(
        repo, config_path, config, head
    )

    crossed = deepcopy(config)
    runner = changed_sources["runner"]
    crossed["source_sha256"][runner] = old_hashes[runner]
    with pytest.raises(RuntimeError, match="executed blob differs"):
        preparer.validate_wave60_final_config_authority(
            repo, config_path, crossed, head
        )


def test_npz_comparison_is_typed_and_nan_exact(tmp_path: Path) -> None:
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    np.savez(left, value=np.asarray([1.0, np.nan], dtype=np.float64))
    np.savez(right, value=np.asarray([1.0, np.nan], dtype=np.float64))
    assert array_exact(left, right)
    np.savez(right, value=np.asarray([1.0, np.nan], dtype=np.float32))
    assert not array_exact(left, right)


def test_static_protocol_recovery_contract_delta_is_exactly_three_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    amendment_path = tmp_path / "amendment.json"
    source_partition = preparer.WAVE60_STATIC_PROTOCOL_RECOVERY_SOURCES
    old_sources = {
        relative: f"old:{relative}"
        for relative in valid_config()["required_execution_sources"]
    }
    new_sources = dict(old_sources)
    for relative in source_partition.values():
        new_sources[relative] = f"new:{relative}"
    amendment = {
        "schema_version": preparer.WAVE60_STATIC_PROTOCOL_RECOVERY_AMENDMENT_SCHEMA,
        "prior_attempt_container": "data/prior",
        "prior_source_sha256": old_sources,
        "recovery_implementation": {
            "changed_sources": {
                label: {
                    "path": relative,
                    "old_sha256": old_sources[relative],
                    "new_sha256": new_sources[relative],
                }
                for label, relative in source_partition.items()
            }
        },
    }
    _write_json(amendment_path, amendment)
    origin_config = valid_config()
    current_config = deepcopy(origin_config)
    current_config["attempt"] = {
        "version": 2,
        "container": "data/current",
        "primary": "primary",
        "replay": "replay",
        "pair": "pair",
        "recovery": {
            "prior_attempt_container": "data/prior",
            "amendment_path": "amendment.json",
            "amendment_sha256": file_sha256(amendment_path),
        },
    }
    escrow_origin_sources = dict(old_sources)
    for relative in preparer.WAVE60_RECOVERY_IMPLEMENTATION_SOURCES.values():
        escrow_origin_sources[relative] = f"v1:{relative}"
    origin_contract = {
        "prospective_config": origin_config,
        "sources": escrow_origin_sources,
    }
    execution_contract = {
        "prospective_config": current_config,
        "sources": new_sources,
        "git_commit": "current-head",
    }
    monkeypatch.setattr(preparer, "validate_prospective_config", lambda _: None)
    monkeypatch.setattr(preparer, "_git_output", lambda *_: "current-head")
    preparer._validate_wave60_contract_delta(
        origin_contract,
        execution_contract,
        amendment,
        tmp_path,
    )

    crossed = deepcopy(execution_contract)
    unchanged = preparer.WAVE60_STATIC_PROTOCOL_UNCHANGED_SOURCES[0]
    crossed["sources"][unchanged] = "scientific-crossing"
    with pytest.raises(RuntimeError, match="source delta drifted"):
        preparer._validate_wave60_contract_delta(
            origin_contract,
            crossed,
            amendment,
            tmp_path,
        )


def test_static_protocol_recovery_provenance_declares_closed_exception() -> None:
    context = {
        "amendment": {
            "schema_version": preparer.WAVE60_STATIC_PROTOCOL_RECOVERY_AMENDMENT_SCHEMA
        },
        "amendment_path": "amendment.json",
        "amendment_sha256": "a" * 64,
        "implementation_commit": "b" * 40,
        "implementation_audit": {"audit_id": "R498"},
        "final_audit": {"audit_id": "R500"},
        "failed_attempt_basename": "primary",
        "escrow_origin_contract_sha256": "c" * 64,
        "benchmark_manifest_sha256": "d" * 64,
    }
    provenance = preparer.recovery_provenance(context, {"contract": "test"})
    assert provenance["contract_extensions"] == {
        "recovery_kind": "STATIC_PROTOCOL_IDENTITY_GUARD",
        "antecedent_static_byte_exceptions": [
            "benchmark/protocol_config.json"
        ],
    }


def _fake_draw(root: Path, marker: str) -> None:
    (root / "benchmark/attestations").mkdir(parents=True)
    (root / "benchmark/commitments").mkdir(parents=True)
    (root / "benchmark/visible").mkdir(parents=True)
    (root / "prepared").mkdir()
    (root / "generation_escrow.json").write_text(marker, encoding="utf-8")
    (root / "pre_generation_freeze.json").write_text(
        json.dumps({"key_commitments": {"generation_secret.json": marker}}),
        encoding="utf-8",
    )
    visible = root / "benchmark/visible/train.jsonl"
    visible.write_text(marker + "-visible\n", encoding="utf-8")
    protocol = root / "benchmark/protocol_config.json"
    protocol.write_text('{"protocol":"wave60-test"}\n', encoding="utf-8")
    (root / "benchmark/manifest.json").write_text(
        json.dumps(
            {
                "generation_key_commitment": marker,
                "identity_key_commitment": marker + "i",
                "semantic_commitment_key_commitment": marker + "s",
                "files": {
                    "protocol_config.json": {
                        "sha256": file_sha256(protocol),
                        "bytes": protocol.stat().st_size,
                    },
                    "visible/train.jsonl": {
                        "sha256": file_sha256(visible),
                        "bytes": visible.stat().st_size,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "benchmark/attestations/semantic_root.json").write_text(
        marker, encoding="utf-8"
    )
    (root / "benchmark/commitments/semantic.jsonl").write_text(marker, encoding="utf-8")
    for name in (
        "gate_fit_bundle.npz",
        "gate_select_inference_bundle.npz",
        "gate_select_truth_bundle.npz",
        "sealed_monitor_inference_bundle.npz",
        "sealed_monitor_truth_bundle.npz",
    ):
        (root / "prepared" / name).write_text(marker, encoding="utf-8")


def test_new_draw_guard_requires_equal_pair_bytes_distinct_inodes_and_new_antecedent(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    antecedent = tmp_path / "antecedent"
    _fake_draw(primary, "fresh")
    # copytree creates independent regular files, which is the valid pair topology.
    shutil.copytree(primary, replay)
    _fake_draw(antecedent, "old")
    result = validate_new_draw_pair(primary, replay, [antecedent])
    assert result["status"] == "PASS"

    (replay / "prepared/sealed_monitor_truth_bundle.npz").unlink()
    os.link(
        primary / "prepared/sealed_monitor_truth_bundle.npz",
        replay / "prepared/sealed_monitor_truth_bundle.npz",
    )
    with pytest.raises(RuntimeError, match="INVALID_NEW_DRAW_IDENTITY"):
        validate_new_draw_pair(primary, replay, [antecedent])

    # Every manifest member is a physical identity, not one aggregate map.
    replay = tmp_path / "replay-visible-hardlink"
    shutil.copytree(primary, replay)
    (replay / "benchmark/visible/train.jsonl").unlink()
    os.link(
        primary / "benchmark/visible/train.jsonl",
        replay / "benchmark/visible/train.jsonl",
    )
    with pytest.raises(RuntimeError, match="INVALID_NEW_DRAW_IDENTITY"):
        validate_new_draw_pair(primary, replay, [antecedent])


def test_new_draw_guard_keeps_static_protocol_integrity_and_inode_guards(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    antecedent = tmp_path / "antecedent"
    _fake_draw(primary, "fresh")
    shutil.copytree(primary, replay)
    _fake_draw(antecedent, "old")

    # Static protocol bytes may match an antecedent, but physical aliasing may not.
    validate_new_draw_pair(primary, replay, [antecedent])
    (primary / "benchmark/protocol_config.json").unlink()
    os.link(
        antecedent / "benchmark/protocol_config.json",
        primary / "benchmark/protocol_config.json",
    )
    with pytest.raises(RuntimeError, match="INVALID_NEW_DRAW_IDENTITY"):
        validate_new_draw_pair(primary, replay, [antecedent])

    primary = tmp_path / "primary-independent"
    replay = tmp_path / "replay-protocol-hardlink"
    _fake_draw(primary, "fresh-independent")
    shutil.copytree(primary, replay)
    (replay / "benchmark/protocol_config.json").unlink()
    os.link(
        primary / "benchmark/protocol_config.json",
        replay / "benchmark/protocol_config.json",
    )
    with pytest.raises(RuntimeError, match="INVALID_NEW_DRAW_IDENTITY"):
        validate_new_draw_pair(primary, replay, [antecedent])


def test_new_draw_guard_rejects_static_protocol_manifest_or_pair_drift(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    antecedent = tmp_path / "antecedent"
    _fake_draw(primary, "fresh")
    shutil.copytree(primary, replay)
    _fake_draw(antecedent, "old")

    (replay / "benchmark/protocol_config.json").write_text(
        '{"protocol":"changed"}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="INVALID_NEW_DRAW_IDENTITY"):
        validate_new_draw_pair(primary, replay, [antecedent])

    replay = tmp_path / "replay-resigned"
    shutil.copytree(primary, replay)
    protocol = replay / "benchmark/protocol_config.json"
    protocol.write_text('{"protocol":"changed"}\n', encoding="utf-8")
    manifest = load_json(replay / "benchmark/manifest.json")
    manifest["files"]["protocol_config.json"] = {
        "sha256": file_sha256(protocol),
        "bytes": protocol.stat().st_size,
    }
    _write_json(replay / "benchmark/manifest.json", manifest)
    with pytest.raises(RuntimeError, match="INVALID_NEW_DRAW_IDENTITY"):
        validate_new_draw_pair(primary, replay, [antecedent])


def test_new_draw_guard_rejects_nonstatic_antecedent_byte_collision(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    antecedent = tmp_path / "antecedent"
    _fake_draw(primary, "fresh")
    shutil.copytree(primary, replay)
    _fake_draw(antecedent, "old")

    for root in (primary, replay):
        target = root / "prepared/gate_fit_bundle.npz"
        target.write_bytes((antecedent / "prepared/gate_fit_bundle.npz").read_bytes())
    with pytest.raises(RuntimeError, match="INVALID_NEW_DRAW_IDENTITY"):
        validate_new_draw_pair(primary, replay, [antecedent])


def test_worker_module_imports_without_discovering_repository() -> None:
    path = EXPERIMENTS / "_wave60_phase_worker.py"
    spec = importlib.util.spec_from_file_location("wave60_worker_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.PHASE_FILES) == {"verify_source_law", "score_apply", "evaluate"}


def test_source_worker_runs_unprivileged_with_exact_closed_stage(
    tmp_path: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="wave60-worker-test-", dir="/tmp") as raw:
        workspace = Path(raw)
        stage = workspace / "stage"
        stage.mkdir(parents=True)
        request = {
            "schema_version": "wave60-source-law-v1",
            "plan_commit": PLAN_COMMIT,
            "plan_sha256": PLAN_SHA256,
            "implementation_commit": "1" * 40,
            "implementation_audit_commit": "2" * 40,
            "implementation_audit_sha256": "3" * 64,
            "source_paths": {name: str(path) for name, path in SOURCE_ALIASES.items()},
            "source_sha256": dict(SOURCE_HASHES),
            "output_path": "unused-in-test",
            "runtime_budget": {"max_seconds": 900, "max_rss_bytes": 1610612736},
        }
        (stage / "source_law_request.json").write_text(
            json.dumps(request, sort_keys=True) + "\n", encoding="utf-8"
        )
        for alias, source in SOURCE_ALIASES.items():
            copy_regular(source, stage / alias)
        output, receipt, _, peak = run_worker(
            workspace,
            stage,
            "verify_source_law",
            [tmp_path / "forbidden-draw"],
            max_seconds=60,
            max_rss=1610612736,
        )
        assert receipt["status"] == "SOURCE_LAW_VERIFIED"
        assert receipt["uid"] == receipt["gid"] == 65534
        assert peak < 1610612736
        assert {
            "source_law_freeze.json",
            "transport_law_manifest.json",
            "transport_law_arrays.npz",
            "frozen_policy_spec.json",
            "feature_schema.json",
            "verify_source_law_receipt.json",
        } == {path.name for path in output.iterdir()}


def test_source_worker_accepts_typed_recovery_request_without_prior_access(
    tmp_path: Path,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".wave60-recovery-worker-test-", dir=REPO_ROOT
    ) as raw:
        workspace = Path(raw)
        stage = workspace / "stage"
        stage.mkdir(parents=True)
        request = recovery_source_request()
        (stage / "source_law_request.json").write_text(
            json.dumps(request, sort_keys=True) + "\n", encoding="utf-8"
        )
        for alias, source in SOURCE_ALIASES.items():
            copy_regular(source, stage / alias)
        output, receipt, _, peak = run_worker(
            workspace,
            stage,
            "verify_source_law",
            [PRIOR_SOURCE_AUTHORITY, tmp_path / "forbidden-draw"],
            max_seconds=60,
            max_rss=1610612736,
        )
        assert receipt["status"] == "SOURCE_LAW_VERIFIED"
        assert receipt["uid"] == receipt["gid"] == 65534
        assert peak < 1610612736
        assert all(row["denied"] is True for row in receipt["denied_path_probes"])
        assert load_json(output / "source_law_freeze.json")[
            "source_law_request_sha256"
        ] == file_sha256(stage / "source_law_request.json")


def test_score_and_evaluate_workers_have_disjoint_physical_views(
    source_material: dict, verified: dict
) -> None:
    with tempfile.TemporaryDirectory(prefix="wave60-score-test-", dir="/tmp") as raw:
        workspace = Path(raw)
        stage = workspace / "stage"
        stage.mkdir()
        worker_config = {"penalty": 1.25, "bootstrap": {"replicates": 5000}}
        (stage / "config.snapshot.json").write_text(
            json.dumps(worker_config) + "\n", encoding="utf-8"
        )
        (stage / "transport_law_manifest.json").write_text(
            json.dumps(verified["manifest"], sort_keys=True) + "\n", encoding="utf-8"
        )
        np.savez(stage / "transport_law_arrays.npz", **verified["arrays"])
        (stage / "frozen_policy_spec.json").write_text(
            json.dumps(verified["verification"]["spec"], sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (stage / "feature_schema.json").write_text(
            json.dumps(
                {
                    "schema_version": "wave60-source-law-v1",
                    "feature_names": list(FEATURE_NAMES),
                    "feature_count": len(FEATURE_NAMES),
                    "dtype": "float64",
                    "order_authority": "wave56_contextual_gate.FEATURE_NAMES",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        np.savez(
            stage / "sealed_monitor_inference_bundle.npz",
            **source_material["inference"],
        )
        binding_files = {
            "source_law_freeze.json",
            "transport_law_manifest.json",
            "transport_law_arrays.npz",
            "frozen_policy_spec.json",
            "feature_schema.json",
            "verify_source_law_receipt.json",
            "source_law_attestation.json",
        }
        copied = {name: "0" * 64 for name in binding_files}
        for name in binding_files & {path.name for path in stage.iterdir()}:
            copied[name] = file_sha256(stage / name)
        source_binding = {
            "schema_version": "wave60-source-binding-v1",
            "run_role": "primary",
            "config_sha256": file_sha256(stage / "config.snapshot.json"),
            "source_authority_path_sha256": "1" * 64,
            "source_law_freeze_sha256": "2" * 64,
            "source_law_attestation_sha256": "3" * 64,
            "copied_output_hashes": copied,
            "hardlink_checks": {name: False for name in binding_files},
        }
        (stage / "source_bindings.json").write_text(
            json.dumps(source_binding, sort_keys=True) + "\n", encoding="utf-8"
        )
        stage_phase_request(stage, "score_apply", "primary")
        score_output, score_receipt, _, _ = run_worker(
            workspace,
            stage,
            "score_apply",
            [WAVE59 / "prepared/sealed_monitor_truth_bundle.npz"],
            max_seconds=60,
        )
        assert score_receipt["status"] == "LOCKBOX_ACTIONS_FROZEN"
        scored_policies = load_npz(score_output / "monitor_policy_arrays.npz")
        assert set(scored_policies) == set(expected_policy_array_keys())

        with tempfile.TemporaryDirectory(
            prefix="wave60-evaluate-test-", dir="/tmp"
        ) as eval_raw:
            eval_workspace = Path(eval_raw)
            eval_stage = eval_workspace / "stage"
            eval_stage.mkdir()
            (eval_stage / "config.snapshot.json").write_text(
                json.dumps(worker_config) + "\n",
                encoding="utf-8",
            )
            copy_regular(
                stage / "source_bindings.json", eval_stage / "source_bindings.json"
            )
            for name in (
                "evaluation_index.npz",
                "monitor_policy_arrays.npz",
                "monitor_action_freeze.json",
            ):
                copy_regular(score_output / name, eval_stage / name)
            copy_regular(
                WAVE59 / "prepared/sealed_monitor_truth_bundle.npz",
                eval_stage / "sealed_monitor_truth_bundle.npz",
            )
            np.save(eval_stage / "utilities.npy", source_material["utilities"])
            stage_phase_request(eval_stage, "evaluate", "primary")
            eval_output, eval_receipt, _, _ = run_worker(
                eval_workspace,
                eval_stage,
                "evaluate",
                [
                    WAVE59 / "prepared/sealed_monitor_inference_bundle.npz",
                    WAVE59 / "fit/model_state_arrays.npz",
                    WAVE59 / "adjudication/monitor_scores.npz",
                ],
                max_seconds=60,
            )
            assert eval_receipt["status"] == "EVALUATED_IMMUTABLE"
            assert all(
                row["denied"] is True for row in eval_receipt["denied_path_probes"]
            )
            analysis = load_json(eval_output / "analysis.json")
            assert len(analysis["metrics"]) == 14
            assert all(
                len(values) == 7 for values in analysis["core_conditions"].values()
            )


def test_complete_pair_executes_seals_recomputes_and_rejects_postfreeze_tamper(
    tmp_path: Path,
    source_material: dict,
    source_authority: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = valid_config()
    config["implementation_binding"]["commit"] = source_authority["request"][
        "implementation_commit"
    ]
    config["implementation_binding"]["audit_sha256"] = source_authority["request"][
        "implementation_audit_sha256"
    ]
    config["source_law_authority"].update(source_authority["binding"])
    config["source_binding"] = load_json(WAVE59 / "source_bindings.json")
    attempt = tmp_path / "attempt"
    canonical_state = {
        str(path): path.exists()
        for path in (
            wave60_runner.SOURCE_AUTHORITY_DEFAULT,
            wave60_runner.ATTEMPT_DEFAULT,
        )
    }
    antecedent_paths = (
        WAVE59 / "prepared/sealed_monitor_inference_bundle.npz",
        WAVE59 / "prepared/sealed_monitor_truth_bundle.npz",
        WAVE59 / "artifact_manifest.json",
    )
    antecedent_hashes = {str(path): file_sha256(path) for path in antecedent_paths}
    _build_prepared_pair(attempt, config, source_material)
    original_finalize = wave60_runner.finalize_pair
    original_hash = wave60_runner.file_sha256

    def finalize_without_secret_reopen(*args: object, **kwargs: object) -> Path:
        def guarded_hash(path: Path) -> str:
            try:
                relative = str(path.relative_to(attempt))
            except ValueError:
                relative = ""
            if "/benchmark/sealed/" in f"/{relative}" or relative.endswith(
                (
                    "/generation_escrow.json",
                    "/source_law/transport_law_arrays.npz",
                    "/prepared/gate_fit_bundle.npz",
                    "/prepared/gate_select_truth_bundle.npz",
                    "/prepared/sealed_monitor_truth_bundle.npz",
                )
            ):
                raise AssertionError(f"finalize reopened secret bytes: {relative}")
            return original_hash(path)

        monkeypatch.setattr(wave60_runner, "file_sha256", guarded_hash)
        try:
            return original_finalize(*args, **kwargs)
        finally:
            monkeypatch.setattr(wave60_runner, "file_sha256", original_hash)

    monkeypatch.setattr(wave60_runner, "finalize_pair", finalize_without_secret_reopen)
    result = execute_prepared_pair(attempt, config, authority=source_authority["path"])
    assert result == attempt / "pair"
    assert load_json(result / "pair_status.json")["terminal"] == "COMPLETE"
    assert load_json(result / "replay_comparison.json")["status"] == "EXACT"
    runtime_budget = load_json(result / "runtime.json")["budget"]
    assert runtime_budget["budget_enforced"] is True
    assert runtime_budget["finalize_invocations"] == 1
    assert runtime_budget["preparation_seconds"] == 2.0
    assert runtime_budget["observed_total_seconds"] < 900.0
    assert runtime_budget["observed_total_seconds"] == pytest.approx(
        runtime_budget["elapsed_before_finalize_seconds"]
        + runtime_budget["finalize_seconds"]
    )
    assert load_json(result / "replay_finalize_attestation.json")["payload"][
        "runtime_sha256"
    ] == file_sha256(result / "runtime.json")
    for role in ("primary", "replay"):
        assert validate_evaluated_root(attempt / role, role) == file_sha256(
            attempt / role / "artifact_manifest.json"
        )
    recomputed_analysis, recomputed_metrics, recomputed_bootstrap = (
        evaluate_transport_actions(
            load_npz(attempt / "primary/prepared/sealed_monitor_truth_bundle.npz"),
            source_material["utilities"],
            1.25,
            load_npz(attempt / "primary/score/monitor_policy_arrays.npz"),
            replicates=5000,
        )
    )
    assert load_json(attempt / "primary/evaluation/analysis.json") == (
        recomputed_analysis
    )
    for expected, relative in (
        (recomputed_metrics, "evaluation/analysis_arrays.npz"),
        (recomputed_bootstrap, "evaluation/bootstrap_indices.npz"),
    ):
        observed = load_npz(attempt / "primary" / relative)
        assert set(observed) == set(expected)
        for key in expected:
            np.testing.assert_array_equal(observed[key], expected[key])
    assert antecedent_hashes == {
        str(path): file_sha256(path) for path in antecedent_paths
    }
    assert canonical_state == {
        str(path): path.exists()
        for path in (
            wave60_runner.SOURCE_AUTHORITY_DEFAULT,
            wave60_runner.ATTEMPT_DEFAULT,
        )
    }

    primary = attempt / "primary"
    analysis_path = primary / "evaluation/analysis.json"
    manifest_path = primary / "artifact_manifest.json"
    original_analysis = analysis_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    analysis = load_json(analysis_path)
    analysis["status"] = "TAMPERED"
    _write_json(analysis_path, analysis)
    manifest = load_json(manifest_path)
    current = wave60_runner.inventory(primary)
    current.pop("artifact_manifest.json")
    manifest["files"] = current
    manifest["classes"] = {
        relative: wave60_runner.root_artifact_class(relative) for relative in current
    }
    _write_json(manifest_path, manifest)
    with pytest.raises(IntegrityDriftError, match="freeze/output chain"):
        validate_evaluated_root(primary, "primary")
    analysis_path.write_bytes(original_analysis)
    manifest_path.write_bytes(original_manifest)
    assert validate_evaluated_root(primary, "primary")

    tamper_targets = (
        primary / "score/monitor_action_freeze.json",
        primary / "score/evaluation_index.npz",
        primary / "evaluation/evaluate_receipt.json",
        primary / "evaluation/evaluation_attestation.json",
        primary / "artifact_manifest.json",
    )
    for path in tamper_targets:
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with pytest.raises((RuntimeError, json.JSONDecodeError)):
            validate_evaluated_root(primary, "primary")
        path.write_bytes(original)
    assert validate_evaluated_root(primary, "primary")
