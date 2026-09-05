from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
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


def test_pair_finalize_resumes_matching_partial_staging_without_rewriting(
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
    result = wave60_runner.finalize_pair(attempt)
    assert result == attempt / "pair"
    assert (result / "pair_status.json").read_bytes() == prior_bytes
    assert not staged.exists()


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
    monkeypatch.setattr(wave60_runner, "seal_evaluated_root", lambda *_: "0" * 64)
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
        authority_binding_sha256="0" * 64,
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
        authority_binding_sha256="0" * 64,
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
            authority_binding_sha256="0" * 64,
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
    config["attempt"]["container"] = recovery["prior_attempt_container"]
    with pytest.raises(RuntimeError, match="namespace"):
        validate_pre_draw_config(config)


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
    result = execute_prepared_pair(attempt, config, authority=source_authority["path"])
    assert result == attempt / "pair"
    assert load_json(result / "pair_status.json")["terminal"] == "COMPLETE"
    assert load_json(result / "replay_comparison.json")["status"] == "EXACT"
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
