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
    SOURCE_ALIASES,
    array_exact,
    copy_regular,
    execute_prepared_pair,
    initialize_attempt_container,
    pair_status,
    publish_pair_failure,
    publish_source_law_authority,
    run_worker,
    seal_root_failure,
    stage_phase_request,
    validate_new_draw_pair,
    validate_pair_failure_package,
    validate_pair_status_against_roots,
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
            "path": "data/geometria_proporcional/wave60_frozen_policy_transport_source_law_v1",
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
        yield {"path": published, "binding": binding, "request": request}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _repacked_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **dict(reversed(list(arrays.items()))))


def _build_prepared_primary(root: Path, config: dict, source_material: dict) -> None:
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
    tmp_path: Path,
) -> None:
    request = tmp_path / "source_law_request.json"
    request.write_text(
        json.dumps({"output_path": "deliberately-invalid"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "source_authority"
    result = publish_source_law_authority(request, output)
    assert result == output
    assert {
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
    } == {
        "source_law_request.json",
        "journals/verify_source_law.json",
        "FAILURE.json",
        "failure_inventory.json",
        "failure_attestation.json",
    }
    assert load_json(output / "FAILURE.json")["terminal"] == "SOURCE_LAW_INVALID"


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
        "WAVE_60_RECOVERY_V3_AMENDMENT.json"
    )
    v3_amendment = {
        "schema_version": "wave60-pretruth-recovery-amendment-v1",
        "status": "APPROVED",
        "prior_attempt_container": v2_container_relative,
        "prior_pair_failure_sha256": file_sha256(v2_pair / "FAILURE.json"),
        "prior_config_audit": {
            "commit": v2_audit_commit,
            "path": v2_audit_relative,
            "sha256": v2_audit_sha256,
        },
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
        "995_wave60_v3_recovery_audit.md"
    )
    write_audit(
        v3_amendment_audit_relative,
        audit_id="R995",
        scope="RECOVERY_AMENDMENT",
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
        "994_wave60_v3_config_audit.md"
    )
    v3["final_audit"] = {
        "audit_id": "R994",
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
        audit_id="R994",
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


def test_npz_comparison_is_typed_and_nan_exact(tmp_path: Path) -> None:
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    np.savez(left, value=np.asarray([1.0, np.nan], dtype=np.float64))
    np.savez(right, value=np.asarray([1.0, np.nan], dtype=np.float64))
    assert array_exact(left, right)
    np.savez(right, value=np.asarray([1.0, np.nan], dtype=np.float32))
    assert not array_exact(left, right)


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
    (root / "benchmark/manifest.json").write_text(
        json.dumps(
            {
                "generation_key_commitment": marker,
                "identity_key_commitment": marker + "i",
                "semantic_commitment_key_commitment": marker + "s",
                "files": {
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
