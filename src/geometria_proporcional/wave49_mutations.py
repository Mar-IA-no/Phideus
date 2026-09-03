"""Adversarial mutation suite for the independent Wave 49 checker."""

from __future__ import annotations

import copy
import hmac
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from .wave49_checker import (
    ProtocolViolation,
    _validate_factorial_and_translation,
    _validate_visible_sealed_binding,
    _validate_visible_rows,
    freeze_prediction_manifest,
    validate_access_ledger,
    validate_manifest,
    validate_oracle_rows,
    validate_protocol_config,
    validate_semantic_attestation,
    validate_semantic_commitment_rows,
    validate_source_separation,
)
from .wave49_schema import (
    SPLITS,
    ProtocolConfig,
    canonical_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


def _expect_rejection(name: str, action: Callable[[], None], expected: str) -> dict[str, str]:
    try:
        action()
    except (ProtocolViolation, ValueError, KeyError, AssertionError) as exc:
        message = str(exc)
        if expected not in message:
            raise AssertionError(f"{name}: wrong rejection: {message}") from exc
        return {
            "mutation_id": name,
            "design_region": "PROTOCOL_INVALID",
            "status": "REJECTED",
            "expected_signal": expected,
            "observed_signal": message,
        }
    raise AssertionError(f"{name}: checker accepted a protocol-invalid mutation")


def run_mutation_suite(
    output_dir: Path,
    config: ProtocolConfig,
    trusted_public_key_path: Path,
) -> list[dict[str, str]]:
    """Run preregistered protocol-invalid mutations without altering canonical artifacts."""
    output_dir = Path(output_dir)
    visible = read_jsonl(output_dir / "visible" / "train.jsonl")
    sealed = read_jsonl(output_dir / "sealed" / "train.jsonl")
    oracle = read_jsonl(output_dir / "sealed" / "oracle" / "train.jsonl")
    results: list[dict[str, str]] = []

    leaked = copy.deepcopy(visible[0])
    leaked.update({"family_id": "PROP", "latent_x": [0.0], "generator_params": {"k": 1.0}})
    results.append(_expect_rejection(
        "M01_visible_truth_leak",
        lambda: _validate_visible_rows([leaked], "train", config),
        "sealed fields leaked",
    ))

    singular = copy.deepcopy(visible[0])
    singular["covariance"][0] = [[1.0, 1.0], [1.0, 1.0]]
    results.append(_expect_rejection(
        "M02_singular_covariance",
        lambda: _validate_visible_rows([singular], "train", config),
        "non-SPD covariance",
    ))

    changed_correlation = copy.deepcopy(sealed)
    covariance = changed_correlation[0]["true_covariance_canonical"][0]
    altered = 0.5 * (covariance[0][0] * covariance[1][1]) ** 0.5
    covariance[0][1] = altered
    covariance[1][0] = altered
    results.append(_expect_rejection(
        "M03_stale_correlation",
        lambda: _validate_factorial_and_translation(changed_correlation, config, "train"),
        "sealed correlation mismatch",
    ))

    stale_translation = copy.deepcopy(sealed)
    translation = next(row for row in stale_translation if row["representation"] == "origin_translation_break")
    # Keep the translated curve internally coherent while breaking only its
    # declared relation to the originating proportional fixture.
    translation["generator_params"]["b"] += 0.25
    translation["clean_y"] = [
        translation["generator_params"]["a"] + translation["generator_params"]["b"] * x
        for x in translation["latent_x"]
    ]
    results.append(_expect_rejection(
        "M04_stale_translation_truth",
        lambda: _validate_factorial_and_translation(stale_translation, config, "train"),
        "stale translation truth",
    ))

    family_swap = copy.deepcopy(sealed)
    left = next(
        row for row in family_swap
        if row["representation"] == "original" and row["covariance_knowledge"] == "full"
        and row["family_id"] == "AFFINE_OFFSET"
    )
    right = next(
        row for row in family_swap
        if row["representation"] == "original" and row["covariance_knowledge"] == "full"
        and row["family_id"] == "POWER_NONUNIT"
        and len(row["latent_x"]) == len(left["latent_x"])
        and all(row[key] == left[key] for key in (
            "noise_mode", "covariance_mode", "range_mode", "rho", "rival_distance_mode"
        ))
    )
    for key in (
        "family_id",
        "generator_params",
        "clean_y",
        "target_region",
        "target_region_basis",
        "design_separation_index",
    ):
        left[key], right[key] = right[key], left[key]

    # Make the substitution internally coherent and keep each visible observation
    # unchanged. Structural and visible/sealed checks must therefore pass; only the
    # independent pre-execution commitment is allowed to reject this mutation.
    visible_by_id = {row["fixture_id"]: row for row in visible}
    for row in (left, right):
        public = visible_by_id[row["fixture_id"]]
        semantics = public["coordinate_semantics"]
        inverse = np.diag([
            float(semantics["x_scale_to_canonical"]),
            float(semantics["y_scale_to_canonical"]),
        ])
        observed = np.column_stack([public["x"], public["y"]]) @ inverse.T
        clean = np.column_stack([row["latent_x"], row["clean_y"]])
        row["realized_errors_canonical"] = (observed - clean).tolist()

    all_commitment_rows = read_jsonl(output_dir / "commitments" / "semantic.jsonl")
    commitment_key = bytes.fromhex(json.loads(
        (output_dir / "sealed" / "semantic_commitment_secret.json").read_text(encoding="utf-8")
    )["key_hex"])

    mutated_by_id = {row["fixture_id"]: row for row in family_swap}
    mutated_commitments = copy.deepcopy(all_commitment_rows)
    for row in mutated_commitments:
        if row["fixture_id"] in {left["fixture_id"], right["fixture_id"]}:
            row["semantic_hmac_sha256"] = hmac.new(
                commitment_key,
                canonical_json(mutated_by_id[row["fixture_id"]]).encode("utf-8"),
                "sha256",
            ).hexdigest()
    train_mutated_commitments = [
        row for row in mutated_commitments if row["split"] == "train"
    ]

    def reject_coherent_family_swap() -> None:
        _validate_visible_sealed_binding(visible, family_swap, "train")
        _validate_factorial_and_translation(family_swap, config, "train")
        validate_semantic_commitment_rows(
            family_swap, train_mutated_commitments, commitment_key, "train"
        )
        with tempfile.TemporaryDirectory(prefix="wave49-signed-mutation-") as raw:
            temp = Path(raw)
            (temp / "sealed").mkdir(parents=True)
            (temp / "commitments").mkdir(parents=True)
            (temp / "attestations").mkdir(parents=True)
            shutil.copy2(output_dir / "protocol_config.json", temp / "protocol_config.json")
            for filename in (
                "generation_secret.json",
                "identity_secret.json",
                "semantic_commitment_secret.json",
            ):
                shutil.copy2(output_dir / "sealed" / filename, temp / "sealed" / filename)
            for split in (*SPLITS, "calibration_null"):
                rows = family_swap if split == "train" else read_jsonl(
                    output_dir / "sealed" / f"{split}.jsonl"
                )
                write_jsonl(temp / "sealed" / f"{split}.jsonl", rows)
            write_jsonl(temp / "commitments" / "semantic.jsonl", mutated_commitments)
            shutil.copy2(
                output_dir / "attestations" / "semantic_root.json",
                temp / "attestations" / "semantic_root.json",
            )
            validate_semantic_attestation(temp, trusted_public_key_path)

    results.append(_expect_rejection(
        "M05_family_frontier_swap",
        reject_coherent_family_swap,
        "semantic attestation content mismatch",
    ))

    results.append(_expect_rejection(
        "M06_free_latent_per_observation",
        lambda: validate_protocol_config(replace(config, oracle_latent_policy="free_per_observation")),
        "oracle latent policy changed",
    ))

    rival_removed = copy.deepcopy(next(row for row in oracle if not row["is_out_of_catalog"]))
    rival_order = sorted(
        (family for family in rival_removed["family_distances"] if family != rival_removed["family_id"]),
        key=rival_removed["family_distances"].get,
    )
    rival_removed["family_distances"][rival_order[0]] = max(rival_removed["family_distances"].values()) + 1.0
    results.append(_expect_rejection(
        "M07_remove_true_rival_minimum",
        lambda: validate_oracle_rows([rival_removed]),
        "oracle ordering is stale",
    ))

    output_dependent = copy.deepcopy(oracle[0])
    output_dependent["selector_output_dependency"] = True
    results.append(_expect_rejection(
        "M08_output_dependent_oracle",
        lambda: validate_oracle_rows([output_dependent]),
        "oracle depends on selector output",
    ))

    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="wave49-mutation-") as raw:
        altered_oracle = Path(raw) / "wave49_oracle.py"
        original_oracle = repo_root / "src" / "geometria_proporcional" / "wave49_oracle.py"
        altered_oracle.write_text(
            original_oracle.read_text(encoding="utf-8") + "\nfrom .wave49_selector import SELECTORS\n",
            encoding="utf-8",
        )
        sources = {
            "selector": repo_root / "src" / "geometria_proporcional" / "wave49_selector.py",
            "worker": repo_root / "experiments" / "geometria_proporcional" / "_wave49_executor_worker.py",
            "oracle": altered_oracle,
            "checker": repo_root / "src" / "geometria_proporcional" / "wave49_checker.py",
        }
        results.append(_expect_rejection(
            "M09_oracle_imports_selector_output",
            lambda: validate_source_separation(sources),
            "forbidden semantic import",
        ))

    with tempfile.TemporaryDirectory(prefix="wave49-mutation-") as raw:
        temp = Path(raw)
        (temp / "sealed" / "oracle").mkdir(parents=True)
        (temp / "sealed" / "oracle" / "train.jsonl").write_text("{}\n", encoding="utf-8")
        results.append(_expect_rejection(
            "M10_oracle_opened_before_prediction_freeze",
            lambda: freeze_prediction_manifest(temp, {}, {}),
            "oracle must not exist before prediction freeze",
        ))

    with tempfile.TemporaryDirectory(prefix="wave49-mutation-") as raw:
        temp = Path(raw)
        (temp / "predictions").mkdir(parents=True)
        write_json(temp / "predictions" / "access_receipt.json", {
            "effective_uid": 65534,
            "sealed_probe_denied": True,
            "orchestrator_verified": True,
            "boundary_method": "setuid-nobody-over-public-only-staging",
            "input_hashes": {},
            "operations": [{"operation": "fit", "split": "lockbox", "sealed_access": False}],
        })
        results.append(_expect_rejection(
            "M11_lockbox_refit",
            lambda: validate_access_ledger(temp),
            "lockbox used for fit",
        ))

    with tempfile.TemporaryDirectory(prefix="wave49-mutation-") as raw:
        temp = Path(raw)
        target = temp / "payload.json"
        target.write_text('{"value":1}\n', encoding="utf-8")
        write_json(temp / "manifest.json", {
            "files": {"payload.json": {"sha256": sha256_file(target), "bytes": target.stat().st_size}}
        })
        target.write_text('{"value":2}\n', encoding="utf-8")
        results.append(_expect_rejection(
            "M12_manifest_content_mutation",
            lambda: validate_manifest(temp),
            "hash mismatch",
        ))

    split_mutation = copy.deepcopy(visible[0])
    split_mutation["split"] = "lockbox"
    results.append(_expect_rejection(
        "M13_cross_split_fixture",
        lambda: _validate_visible_rows([split_mutation], "train", config),
        "split mismatch",
    ))

    write_jsonl(output_dir / "mutations" / "results.jsonl", results)
    return results
