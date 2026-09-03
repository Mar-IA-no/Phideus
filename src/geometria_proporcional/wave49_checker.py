"""Independent protocol-integrity checker for the Wave 49 benchmark."""

from __future__ import annotations

import ast
import hmac
import json
import platform
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn

from .wave49_attestation import AttestationError, file_record as attested_file_record, verify_attestation
from .wave49_schema import (
    CATALOG_FAMILIES,
    FORBIDDEN_VISIBLE_KEYS,
    OUT_OF_CATALOG_FAMILIES,
    SCHEMA_VERSION,
    SPLITS,
    ProtocolConfig,
    canonical_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    write_json,
)


class ProtocolViolation(RuntimeError):
    """Raised when a benchmark artifact violates the sealed protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolViolation(message)


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _snapshot_sources(
    output_dir: Path,
    phase: str,
    source_paths: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    """Copy source inputs into the artifact tree and record immutable hashes."""
    snapshot_dir = output_dir / "lineage" / f"{phase}_sources"
    _require(not snapshot_dir.exists(), f"{phase} source snapshot already exists")
    snapshot_dir.mkdir(parents=True)
    records: dict[str, dict[str, Any]] = {}
    for index, (label, raw_path) in enumerate(sorted(source_paths.items())):
        source = Path(raw_path)
        _require(source.is_file(), f"source input missing: {label}")
        safe_label = "".join(char if char.isalnum() or char in "-_." else "_" for char in label)
        destination = snapshot_dir / f"{index:02d}_{safe_label}__{source.name}"
        shutil.copy2(source, destination)
        records[label] = {
            "artifact": str(destination.relative_to(output_dir)),
            "origin_name": source.name,
            **_file_record(destination),
        }
    return records


def _validate_source_snapshots(output_dir: Path, records: dict[str, dict[str, Any]], prefix: str) -> None:
    for label, expected in records.items():
        relative = expected.get("artifact")
        _require(isinstance(relative, str), f"{prefix} source artifact missing: {label}")
        artifact = output_dir / relative
        _require(artifact.exists(), f"{prefix} source snapshot missing: {label}")
        _require(artifact.stat().st_size == expected["bytes"], f"{prefix} source size mismatch: {label}")
        _require(sha256_file(artifact) == expected["sha256"], f"{prefix} source hash mismatch: {label}")


def validate_protocol_config(config: ProtocolConfig) -> None:
    _require(config.schema_version == SCHEMA_VERSION, "protocol schema version changed or missing")
    _require(config.oracle_latent_policy == "shared_fixed_design", "oracle latent policy changed")
    _require(0.0 < config.ood_alpha < 1.0, "invalid OOD alpha")
    _require(config.replicates_per_condition > 0, "replicate count must be positive")
    _require(config.calibration_null_per_n > 0, "null calibration must be nonempty")
    _require(set(config.covariance_knowledge_modes) == {"full", "diagonal_only"},
             "covariance-knowledge factorial is incomplete")
    _require(
        config.calibration_sampling_law == "iid_uniform_over_declared_factorial_factors",
        "calibration sampling law changed",
    )
    _require(
        config.calibration_coverage_scope == "marginal_within_n_covariance_knowledge_population",
        "calibration coverage scope changed",
    )


def validate_manifest(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    _require(manifest_path.exists(), "manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = output_dir / relative
        _require(path.exists(), f"manifest file is missing: {relative}")
        _require(path.stat().st_size == expected["bytes"], f"size mismatch: {relative}")
        _require(sha256_file(path) == expected["sha256"], f"hash mismatch: {relative}")
    return manifest


def _validate_visible_rows(rows: list[dict[str, Any]], expected_split: str, config: ProtocolConfig) -> set[str]:
    local_ids: set[str] = set()
    for row in rows:
        fixture_id = row.get("fixture_id")
        _require(row.get("schema_version") == config.schema_version, f"schema mismatch: {fixture_id}")
        leaked = FORBIDDEN_VISIBLE_KEYS.intersection(row)
        _require(not leaked, f"sealed fields leaked in {fixture_id}: {sorted(leaked)}")
        _require(row.get("split") == expected_split, f"split mismatch: {fixture_id}")
        _require(fixture_id not in local_ids, f"duplicate fixture in {expected_split}: {fixture_id}")
        local_ids.add(fixture_id)
        n = int(row["n"])
        x = np.asarray(row["x"], dtype=np.float64)
        y = np.asarray(row["y"], dtype=np.float64)
        covariance = np.asarray(row["covariance"], dtype=np.float64)
        _require(x.shape == (n,) and y.shape == (n,), f"observation shape mismatch: {fixture_id}")
        _require(covariance.shape == (n, 2, 2), f"covariance shape mismatch: {fixture_id}")
        _require(np.all(np.isfinite(x)) and np.all(np.isfinite(y)), f"nonfinite observation: {fixture_id}")
        _require(np.allclose(covariance, covariance.transpose(0, 2, 1)), f"nonsymmetric covariance: {fixture_id}")
        _require(np.all(np.linalg.eigvalsh(covariance) > 0), f"non-SPD covariance: {fixture_id}")
        domain = row["domain"]
        _require(len(domain) == 2 and domain[0] > 0 and domain[0] < domain[1], f"invalid domain: {fixture_id}")
        knowledge = row["coordinate_semantics"].get("covariance_knowledge")
        _require(knowledge in config.covariance_knowledge_modes, f"invalid covariance knowledge: {fixture_id}")
        population = row["coordinate_semantics"].get("calibration_population")
        _require(population in {"canonical_preserving", "origin_translation_break"},
                 f"invalid calibration population: {fixture_id}")
        _require(float(row["coordinate_semantics"].get("x_scale_to_canonical", 0.0)) > 0.0,
                 f"invalid x canonical scale: {fixture_id}")
        _require(float(row["coordinate_semantics"].get("y_scale_to_canonical", 0.0)) > 0.0,
                 f"invalid y canonical scale: {fixture_id}")
        if knowledge == "diagonal_only":
            _require(np.all(covariance[:, 0, 1] == 0.0), f"hidden correlation leaked: {fixture_id}")
    return local_ids


def validate_visible_package(output_dir: Path, config: ProtocolConfig) -> dict[str, int]:
    output_dir = Path(output_dir)
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for split in SPLITS:
        rows = read_jsonl(output_dir / "visible" / f"{split}.jsonl")
        ids = _validate_visible_rows(rows, split, config)
        _require(not (ids & seen), f"fixture appears across splits: {sorted(ids & seen)[:1]}")
        seen.update(ids)
        counts[split] = len(rows)
    calibration = read_jsonl(output_dir / "visible" / "calibration_null.jsonl")
    calibration_ids = _validate_visible_rows(calibration, "calibration_null", config)
    _require(not (calibration_ids & seen), "calibration fixture appears in evaluation split")
    calibration_cells = Counter(
        (
            int(row["n"]),
            row["coordinate_semantics"]["covariance_knowledge"],
            row["coordinate_semantics"]["calibration_population"],
        )
        for row in calibration
    )
    expected_calibration_cells = {
        (n, knowledge, population)
        for n in config.sample_sizes
        for knowledge in config.covariance_knowledge_modes
        for population in ("canonical_preserving", "origin_translation_break")
    }
    _require(set(calibration_cells) == expected_calibration_cells,
             "null-calibration strata missing or extra")
    _require(all(count == config.calibration_null_per_n for count in calibration_cells.values()),
             "null-calibration stratum size mismatch")
    expected_null = (
        len(config.sample_sizes) * len(config.covariance_knowledge_modes)
        * config.calibration_null_per_n * 2
    )
    _require(len(calibration) == expected_null, "null-calibration coverage mismatch")
    return counts


def validate_semantic_commitment_rows(
    sealed_rows: list[dict[str, Any]],
    commitment_rows: list[dict[str, Any]],
    key: bytes,
    expected_split: str,
) -> None:
    """Verify the generator's pre-execution HMAC commitment to each sealed row."""
    sealed = {row["fixture_id"]: row for row in sealed_rows}
    commitments: dict[str, dict[str, Any]] = {}
    for row in commitment_rows:
        fixture_id = row.get("fixture_id")
        _require(row.get("split") == expected_split, f"semantic commitment split mismatch: {fixture_id}")
        _require(fixture_id not in commitments, f"duplicate semantic commitment: {fixture_id}")
        commitments[fixture_id] = row
    _require(set(sealed) == set(commitments), f"semantic commitment coverage mismatch in {expected_split}")
    for fixture_id, row in sealed.items():
        expected = hmac.new(key, canonical_json(row).encode("utf-8"), "sha256").hexdigest()
        observed = commitments[fixture_id].get("semantic_hmac_sha256", "")
        _require(hmac.compare_digest(expected, observed), f"semantic commitment mismatch: {fixture_id}")


def validate_semantic_commitments(output_dir: Path) -> dict[str, int]:
    output_dir = Path(output_dir)
    commitment_path = output_dir / "commitments" / "semantic.jsonl"
    key_path = output_dir / "sealed" / "semantic_commitment_secret.json"
    _require(commitment_path.exists(), "semantic commitment file is missing")
    _require(key_path.exists(), "semantic commitment key is missing")
    commitment_rows = read_jsonl(commitment_path)
    key_payload = json.loads(key_path.read_text(encoding="utf-8"))
    key = bytes.fromhex(key_payload["key_hex"])
    counts: dict[str, int] = {}
    for split in (*SPLITS, "calibration_null"):
        sealed_rows = read_jsonl(output_dir / "sealed" / f"{split}.jsonl")
        split_commitments = [row for row in commitment_rows if row.get("split") == split]
        validate_semantic_commitment_rows(sealed_rows, split_commitments, key, split)
        counts[split] = len(split_commitments)
    expected_ids = {
        row["fixture_id"]
        for split in (*SPLITS, "calibration_null")
        for row in read_jsonl(output_dir / "sealed" / f"{split}.jsonl")
    }
    _require(len(commitment_rows) == len(expected_ids), "semantic commitment has extra rows")
    return counts


def validate_semantic_attestation(
    output_dir: Path,
    trusted_public_key_path: Path,
) -> dict[str, Any]:
    """Verify the detached pre-execution signature against an external trust root."""
    output_dir = Path(output_dir)
    path = output_dir / "attestations" / "semantic_root.json"
    _require(path.exists(), "semantic attestation is missing")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    try:
        verify_attestation(receipt, Path(trusted_public_key_path))
    except AttestationError as exc:
        raise ProtocolViolation(str(exc)) from exc
    key_paths = {
        "generation": output_dir / "sealed" / "generation_secret.json",
        "identity": output_dir / "sealed" / "identity_secret.json",
        "semantic_hmac": output_dir / "sealed" / "semantic_commitment_secret.json",
    }
    key_commitments = {
        name: sha256_bytes(bytes.fromhex(json.loads(key_path.read_text(encoding="utf-8"))["key_hex"]))
        for name, key_path in key_paths.items()
    }
    sealed_paths = {
        split: output_dir / "sealed" / f"{split}.jsonl"
        for split in (*SPLITS, "calibration_null")
    }
    expected = {
        "phase": "sealed-semantics-committed-before-selector",
        "schema_version": json.loads(
            (output_dir / "protocol_config.json").read_text(encoding="utf-8")
        )["schema_version"],
        "protocol_config": attested_file_record(output_dir / "protocol_config.json"),
        "semantic_commitments": attested_file_record(
            output_dir / "commitments" / "semantic.jsonl"
        ),
        "sealed_truth": {
            split: attested_file_record(sealed_path)
            for split, sealed_path in sealed_paths.items()
        },
        "counts": {
            split: len(read_jsonl(sealed_path))
            for split, sealed_path in sealed_paths.items()
        },
        "key_commitments": key_commitments,
    }
    _require(receipt.get("payload") == expected, "semantic attestation content mismatch")
    return receipt


def validate_sealed_alignment(output_dir: Path, config: ProtocolConfig | None = None) -> dict[str, int]:
    output_dir = Path(output_dir)
    counts: dict[str, int] = {}
    for split in SPLITS:
        visible_rows = read_jsonl(output_dir / "visible" / f"{split}.jsonl")
        visible_ids = {row["fixture_id"] for row in visible_rows}
        sealed_rows = read_jsonl(output_dir / "sealed" / f"{split}.jsonl")
        sealed_ids = {row["fixture_id"] for row in sealed_rows}
        _require(visible_ids == sealed_ids, f"visible/sealed fixture mismatch in {split}")
        counts[split] = len(sealed_rows)
        _validate_visible_sealed_binding(visible_rows, sealed_rows, split)
        if config is not None:
            _validate_factorial_and_translation(sealed_rows, config, split)
    calibration_visible = read_jsonl(output_dir / "visible" / "calibration_null.jsonl")
    calibration_sealed = read_jsonl(output_dir / "sealed" / "calibration_null.jsonl")
    _require(
        {row["fixture_id"] for row in calibration_visible}
        == {row["fixture_id"] for row in calibration_sealed},
        "visible/sealed fixture mismatch in calibration_null",
    )
    _validate_visible_sealed_binding(calibration_visible, calibration_sealed, "calibration_null")
    if config is not None:
        for row in calibration_sealed:
            _validate_sealed_row_semantics(row, config)
    counts["calibration_null"] = len(calibration_sealed)
    validate_semantic_commitments(output_dir)
    return counts


def _checker_curve(family: str, x: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Independent structural reconstruction used only by the checker."""
    if family == "PROP":
        return params["k"] * x
    if family == "AFFINE_OFFSET":
        return params["a"] + params["b"] * x
    if family == "POWER_NONUNIT":
        return params["a"] * np.power(x, params["p"])
    if family == "SATURATING":
        return params["L"] * x / (params["K"] + x)
    if family == "PIECEWISE_AFFINE":
        left = params["b1"] * x
        right = params["b1"] * params["knot"] + params["b2"] * (x - params["knot"])
        return np.where(x <= params["knot"], left, right)
    if family == "OFFSET_QUADRATIC":
        return params["a"] + params["b"] * x + params["c"] * x * x
    raise ProtocolViolation(f"unknown sealed family: {family}")


def _expected_target_region(row: dict[str, Any]) -> str:
    if row["family_id"] in OUT_OF_CATALOG_FAMILIES:
        return "OUT_OF_CATALOG"
    basis = row.get("target_region_basis", {})
    if float(row["design_separation_index"]) <= float(basis["compatibility_distance"]):
        return "DELIBERATELY_INDISTINGUISHABLE"
    return "IDENTIFIABLE"


def _expected_calibration_population(representation: str) -> str:
    return (
        "origin_translation_break"
        if representation == "origin_translation_break"
        else "canonical_preserving"
    )


def _validate_sealed_row_semantics(row: dict[str, Any], config: ProtocolConfig) -> None:
    fixture_id = row["fixture_id"]
    latent_x = np.asarray(row["latent_x"], dtype=np.float64)
    clean_y = np.asarray(row["clean_y"], dtype=np.float64)
    reconstructed_y = _checker_curve(row["family_id"], latent_x, row["generator_params"])
    _require(clean_y.shape == latent_x.shape and np.allclose(clean_y, reconstructed_y, atol=1e-12),
             f"sealed curve/parameter mismatch: {fixture_id}")
    covariance = np.asarray(row["true_covariance_canonical"], dtype=np.float64)
    _require(covariance.shape == (len(latent_x), 2, 2), f"sealed covariance shape mismatch: {fixture_id}")
    _require(np.all(np.linalg.eigvalsh(covariance) > 0), f"sealed non-SPD covariance: {fixture_id}")
    errors = np.asarray(row.get("realized_errors_canonical"), dtype=np.float64)
    _require(errors.shape == (len(latent_x), 2) and np.all(np.isfinite(errors)),
             f"sealed realized-error mismatch: {fixture_id}")
    correlation = covariance[:, 0, 1] / np.sqrt(covariance[:, 0, 0] * covariance[:, 1, 1])
    _require(np.allclose(correlation, float(row["rho"]), atol=1e-12),
             f"sealed correlation mismatch: {fixture_id}")
    expected_stratum = (
        "TRANSLATION_RUPTURE" if row["representation"] == "origin_translation_break"
        else "OUT_OF_CATALOG" if row["family_id"] in OUT_OF_CATALOG_FAMILIES
        else "NEAR_RIVAL" if row["rival_distance_mode"] == "near"
        else "FAR_RIVAL"
    )
    _require(row.get("design_stratum") == expected_stratum,
             f"design-stratum mismatch: {fixture_id}")
    _require(row.get("target_region") == _expected_target_region(row),
             f"target-region mismatch: {fixture_id}")
    basis = row.get("target_region_basis", {})
    _require(basis.get("rule") == "generator_preexecution_separation_vs_compatibility_distance",
             f"target-region basis mismatch: {fixture_id}")
    _require(np.isfinite(float(row.get("design_separation_index", np.nan))),
             f"design separation missing: {fixture_id}")
    _require(np.isclose(
        float(basis.get("design_separation_index", np.nan)),
        float(row["design_separation_index"]), rtol=0.0, atol=1e-12,
    ), f"design separation basis mismatch: {fixture_id}")
    _require(np.isclose(
        float(basis.get("compatibility_distance", np.nan)),
        float(config.oracle_compatibility_distance), rtol=0.0, atol=0.0,
    ), f"target compatibility distance mismatch: {fixture_id}")
    _require(row.get("calibration_population") == _expected_calibration_population(row["representation"]),
             f"calibration-population mismatch: {fixture_id}")


def _validate_visible_sealed_binding(
    visible_rows: list[dict[str, Any]],
    sealed_rows: list[dict[str, Any]],
    split: str,
) -> None:
    """Bind visible observations to sealed clean curves, errors, and covariance."""
    visible = {row["fixture_id"]: row for row in visible_rows}
    sealed = {row["fixture_id"]: row for row in sealed_rows}
    _require(set(visible) == set(sealed), f"visible/sealed fixture mismatch in {split}")
    for fixture_id, truth in sealed.items():
        row = visible[fixture_id]
        semantics = row["coordinate_semantics"]
        inverse = np.diag([
            float(semantics["x_scale_to_canonical"]),
            float(semantics["y_scale_to_canonical"]),
        ])
        observed = np.column_stack([row["x"], row["y"]]) @ inverse.T
        expected = np.column_stack([truth["latent_x"], truth["clean_y"]]) + np.asarray(
            truth["realized_errors_canonical"], dtype=np.float64
        )
        _require(np.allclose(observed, expected, rtol=1e-12, atol=1e-12),
                 f"visible/sealed observation mismatch: {fixture_id}")
        visible_covariance = np.asarray(row["covariance"], dtype=np.float64)
        canonical_covariance = np.einsum("ab,nbc,dc->nad", inverse, visible_covariance, inverse)
        expected_covariance = np.asarray(truth["true_covariance_canonical"], dtype=np.float64).copy()
        if truth["covariance_knowledge"] == "diagonal_only":
            expected_covariance[:, 0, 1] = 0.0
            expected_covariance[:, 1, 0] = 0.0
        _require(np.allclose(canonical_covariance, expected_covariance, rtol=1e-12, atol=1e-12),
                 f"visible/sealed covariance mismatch: {fixture_id}")
        canonical_domain = np.asarray(row["domain"], dtype=np.float64) * inverse[0, 0]
        latent_x = np.asarray(truth["latent_x"], dtype=np.float64)
        _require(np.allclose(canonical_domain, [latent_x.min(), latent_x.max()], atol=1e-12),
                 f"visible/sealed domain mismatch: {fixture_id}")
        _require(
            semantics.get("calibration_population") == truth.get("calibration_population"),
            f"visible/sealed calibration population mismatch: {fixture_id}",
        )


def _validate_factorial_and_translation(rows: list[dict[str, Any]], config: ProtocolConfig, split: str) -> None:
    factors = Counter()
    translations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        _validate_sealed_row_semantics(row, config)
        translations[row["pair_token"]].append(row)
        if row["representation"] == "origin_translation_break":
            continue
        factors[(
            row["family_id"],
            len(row["latent_x"]),
            row["range_mode"],
            row["noise_mode"],
            row["covariance_mode"],
            float(row["rho"]),
            row["rival_distance_mode"],
            row["covariance_knowledge"],
            row["representation"],
        )] += 1
    expected_keys = {
        (family, n, range_mode, noise_mode, covariance_mode, float(rho), rival, knowledge, representation)
        for family in CATALOG_FAMILIES + OUT_OF_CATALOG_FAMILIES
        for n in config.sample_sizes
        for range_mode in config.range_modes
        for noise_mode in config.noise_modes
        for covariance_mode in config.covariance_modes
        for rho in config.correlations
        for rival in config.rival_distance_modes
        for knowledge in config.covariance_knowledge_modes
        for representation in config.representation_modes
    }
    _require(set(factors) == expected_keys, f"factorial cells missing or extra in {split}")
    _require(all(count == config.replicates_per_condition for count in factors.values()),
             f"factorial replicate mismatch in {split}")

    translated_count = 0
    for group in translations.values():
        prop_by_knowledge = {
            row["covariance_knowledge"]: row
            for row in group
            if row["family_id"] == "PROP" and row["representation"] == "original"
        }
        for translated in (row for row in group if row["representation"] == "origin_translation_break"):
            translated_count += 1
            source = prop_by_knowledge.get(translated["covariance_knowledge"])
            _require(source is not None, f"translation source missing: {translated['fixture_id']}")
            delta = np.asarray(translated["clean_y"]) - np.asarray(source["clean_y"])
            _require(np.allclose(delta, translated["generator_params"]["a"]),
                     f"stale translation truth: {translated['fixture_id']}")
            _require(translated["family_id"] == "AFFINE_OFFSET" and translated["source_family"] == "PROP",
                     f"translation did not rupture PROP: {translated['fixture_id']}")
    base_conditions = (
        len(config.sample_sizes) * len(config.range_modes) * len(config.noise_modes)
        * len(config.covariance_modes) * len(config.correlations) * len(config.rival_distance_modes)
        * config.replicates_per_condition
    )
    _require(translated_count == base_conditions * len(config.covariance_knowledge_modes),
             f"translation factorial mismatch in {split}")


def validate_predictions(
    output_dir: Path,
    selector_names: set[str],
    config: ProtocolConfig | None = None,
) -> dict[str, int]:
    output_dir = Path(output_dir)
    calibration_path = output_dir / "predictions" / "abstention_calibration.json"
    _require(calibration_path.exists(), "abstention calibration is missing")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    _require(calibration.get("method") == "split-conformal-null-after-family-and-latent-selection",
             "unrecognized abstention calibration")
    if config is not None:
        _require(calibration.get("sampling_law") == config.calibration_sampling_law,
                 "calibration sampling law mismatch")
        _require(calibration.get("coverage_scope") == config.calibration_coverage_scope,
                 "calibration coverage scope mismatch")
        _require(calibration.get("coverage_guarantee") == "marginal_not_conditional_by_nuisance",
                 "calibration guarantee scope mismatch")
    counts: dict[str, int] = {}
    for split in SPLITS:
        visible_ids = {row["fixture_id"] for row in read_jsonl(output_dir / "visible" / f"{split}.jsonl")}
        rows = read_jsonl(output_dir / "predictions" / f"{split}.jsonl")
        keys = {(row["fixture_id"], row["selector"]) for row in rows}
        expected = {(fixture_id, selector) for fixture_id in visible_ids for selector in selector_names}
        _require(keys == expected, f"prediction coverage mismatch in {split}")
        counts[split] = len(rows)
    return counts


def validate_access_ledger(output_dir: Path) -> int:
    output_dir = Path(output_dir)
    path = output_dir / "predictions" / "access_receipt.json"
    _require(path.exists(), "restricted-executor access receipt is missing")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    _require(receipt.get("effective_uid") not in {None, 0}, "executor did not run under a restricted uid")
    _require(receipt.get("sealed_probe_denied") is True, "sealed access was not denied")
    _require(receipt.get("orchestrator_verified") is True, "executor boundary lacks orchestrator verification")
    _require(receipt.get("boundary_method") == "setuid-nobody-over-public-only-staging",
             "executor boundary method changed")
    for filename, digest in receipt.get("input_hashes", {}).items():
        visible_path = output_dir / "visible" / filename
        _require(visible_path.exists(), f"executor input missing from public package: {filename}")
        _require(sha256_file(visible_path) == digest, f"executor input hash mismatch: {filename}")
    rows = receipt.get("operations", [])
    for row in rows:
        operation = row.get("operation")
        split = row.get("split")
        _require(not (split == "lockbox" and operation in {"fit", "tune", "select"}),
                 f"lockbox used for {operation}")
        _require(row.get("sealed_access") is False, f"sealed access recorded for {split}")
    _require({row.get("split") for row in rows} == set(SPLITS) | {"calibration_null"},
             "access receipt lacks prediction or calibration coverage")
    return len(rows)


def validate_source_separation(source_paths: dict[str, Path]) -> None:
    """Reject forbidden semantic dependencies between executor, oracle, and checker."""
    by_name = {Path(path).name: Path(path) for path in source_paths.values()}
    policies = {
        "wave49_selector.py": {"wave49_generator", "wave49_oracle", "wave49_logic_oracle"},
        "_wave49_executor_worker.py": {"wave49_generator", "wave49_oracle", "wave49_checker"},
        "wave49_oracle.py": {"wave49_selector", "wave49_evaluator"},
        "wave49_checker.py": {"wave49_generator", "wave49_oracle", "wave49_selector"},
    }
    for filename, forbidden in policies.items():
        path = by_name.get(filename)
        _require(path is not None, f"source separation input missing: {filename}")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        offenders = sorted(name for name in imports if any(term in name for term in forbidden))
        _require(not offenders, f"forbidden semantic import in {filename}: {offenders}")
    oracle_text = by_name["wave49_oracle.py"].read_text(encoding="utf-8")
    _require('"predictions"' not in oracle_text and "'/predictions'" not in oracle_text,
             "oracle source reads selector predictions")


def freeze_prediction_manifest(
    output_dir: Path,
    invocation: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    """Freeze public inputs and outputs before any oracle artifact exists."""
    output_dir = Path(output_dir)
    oracle_dir = output_dir / "sealed" / "oracle"
    _require(not oracle_dir.exists() or not any(oracle_dir.iterdir()),
             "oracle must not exist before prediction freeze")
    validate_source_separation(source_paths)
    paths = [output_dir / "manifest.json", output_dir / "protocol_config.json",
             output_dir / "attestations" / "semantic_root.json",
             output_dir / "predictions" / "access_receipt.json",
             output_dir / "predictions" / "abstention_calibration.json"]
    paths.extend(output_dir / "visible" / f"{split}.jsonl" for split in SPLITS)
    paths.append(output_dir / "visible" / "calibration_null.jsonl")
    paths.extend(output_dir / "predictions" / f"{split}.jsonl" for split in SPLITS)
    manifest = {
        "phase": "predictions-frozen-before-oracle",
        "invocation": invocation,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "files": {str(path.relative_to(output_dir)): _file_record(path) for path in paths},
        "sources": _snapshot_sources(output_dir, "prediction", source_paths),
    }
    write_json(output_dir / "prediction_manifest.json", manifest)
    return manifest


def _validate_file_map(base: Path, records: dict[str, Any], prefix: str) -> None:
    for relative, expected in records.items():
        artifact = base / relative
        _require(artifact.exists(), f"{prefix} artifact missing: {relative}")
        _require(artifact.stat().st_size == expected["bytes"], f"{prefix} size mismatch: {relative}")
        _require(sha256_file(artifact) == expected["sha256"], f"{prefix} hash mismatch: {relative}")


def validate_prediction_manifest(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    path = output_dir / "prediction_manifest.json"
    _require(path.exists(), "prediction manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("phase") == "predictions-frozen-before-oracle", "prediction freeze phase mismatch")
    _validate_file_map(output_dir, manifest["files"], "prediction")
    _validate_source_snapshots(output_dir, manifest["sources"], "prediction")
    return manifest


def validate_oracle_consistency(output_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in SPLITS:
        rows = read_jsonl(Path(output_dir) / "sealed" / "oracle" / f"{split}.jsonl")
        validate_oracle_rows(rows)
        truth = {
            row["fixture_id"]: row
            for row in read_jsonl(Path(output_dir) / "sealed" / f"{split}.jsonl")
        }
        indexed = {row["fixture_id"]: row for row in rows}
        _require(set(indexed) == set(truth), f"oracle/truth coverage mismatch in {split}")
        linked_fields = (
            "split", "family_id", "is_out_of_catalog", "target_region",
            "target_region_basis", "design_separation_index", "pair_token",
            "representation", "range_mode", "noise_mode", "covariance_mode",
            "covariance_knowledge", "rho", "rival_distance_mode", "design_stratum",
            "calibration_population",
        )
        for fixture_id, oracle_row in indexed.items():
            truth_row = truth[fixture_id]
            _require(int(oracle_row.get("n", -1)) == len(truth_row["latent_x"]),
                     f"oracle/truth sample-size mismatch: {fixture_id}")
            for field in linked_fields:
                _require(oracle_row.get(field) == truth_row.get(field),
                         f"oracle/truth field mismatch ({field}): {fixture_id}")
        counts[split] = len(rows)
    return counts


def validate_oracle_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        _require(row.get("distance_order_match") is True, f"oracle reference-order mismatch: {row['fixture_id']}")
        _require(row.get("oracle_input_scope") == "sealed_truth+public_parameter_catalog",
                 f"oracle input scope changed: {row['fixture_id']}")
        _require(row.get("selector_output_dependency") is False,
                 f"oracle depends on selector output: {row['fixture_id']}")
        target = row.get("target_region")
        observed = row.get("observational_region", "")
        expected_match = (
            observed.startswith("OUT_OF_CATALOG_") if target == "OUT_OF_CATALOG"
            else observed == "OBSERVATIONALLY_INDISTINGUISHABLE"
            if target == "DELIBERATELY_INDISTINGUISHABLE"
            else observed == "OBSERVATIONALLY_IDENTIFIABLE"
            if target == "IDENTIFIABLE"
            else False
        )
        _require(target in {"IDENTIFIABLE", "DELIBERATELY_INDISTINGUISHABLE", "OUT_OF_CATALOG"},
                 f"invalid target region: {row['fixture_id']}")
        _require(row.get("target_region_match") is expected_match,
                 f"target-region adjudication mismatch: {row['fixture_id']}")
        _require(float(row.get("target_design_distance_delta", np.inf)) <= float(row["numeric_atol"]),
                 f"target design distance mismatch: {row['fixture_id']}")
        _require(float(row.get("target_design_reference_delta", np.inf)) <= float(row["numeric_atol"]),
                 f"target design reference mismatch: {row['fixture_id']}")
        atol = float(row.get("numeric_atol", 0.0))
        _require(atol > 0.0, f"oracle numeric tolerance missing: {row['fixture_id']}")
        _require(row.get("max_distance_delta", np.inf) <= atol,
                 f"oracle numerical disagreement: {row['fixture_id']}")
        primary = row["family_distances"]
        reference = row["reference_family_distances"]
        _require(set(primary) == set(reference) == set(CATALOG_FAMILIES),
                 f"oracle catalog mismatch: {row['fixture_id']}")
        for index, left in enumerate(sorted(primary)):
            for right in sorted(primary)[index + 1:]:
                delta_primary = primary[left] - primary[right]
                delta_reference = reference[left] - reference[right]
                if abs(delta_primary) <= atol or abs(delta_reference) <= atol:
                    continue
                _require(np.sign(delta_primary) == np.sign(delta_reference),
                         f"oracle ordering is stale: {row['fixture_id']}")


def freeze_execution_manifest(
    output_dir: Path,
    invocation: dict[str, Any] | None = None,
    source_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    paths = [output_dir / "prediction_manifest.json"]
    paths.extend(output_dir / "sealed" / "oracle" / f"{split}.jsonl" for split in SPLITS)
    for relative in ("evaluation/fixture_scores.jsonl", "evaluation/summary.json",
                     "evaluation/REPORT_WAVE49_CLASSICAL.md", "mutations/results.jsonl"):
        path = output_dir / relative
        if path.exists():
            paths.append(path)
    manifest = {
        "phase": "oracle-opened-after-prediction-freeze",
        "invocation": invocation or {},
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "files": {str(path.relative_to(output_dir)): _file_record(path) for path in paths},
        "sources": _snapshot_sources(output_dir, "execution", source_paths or {}),
    }
    write_json(output_dir / "execution_manifest.json", manifest)
    return manifest


def validate_execution_manifest(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    path = output_dir / "execution_manifest.json"
    _require(path.exists(), "execution manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("phase") == "oracle-opened-after-prediction-freeze", "execution phase mismatch")
    _validate_file_map(output_dir, manifest["files"], "execution")
    _validate_source_snapshots(output_dir, manifest["sources"], "execution")
    return manifest


def validate_all(
    output_dir: Path,
    config: ProtocolConfig,
    selector_names: set[str],
    trusted_public_key_path: Path,
) -> dict[str, Any]:
    validate_protocol_config(config)
    return {
        "manifest": validate_manifest(output_dir),
        "semantic_attestation": validate_semantic_attestation(
            output_dir, trusted_public_key_path
        ),
        "visible_counts": validate_visible_package(output_dir, config),
        "sealed_counts": validate_sealed_alignment(output_dir, config),
        "prediction_counts": validate_predictions(output_dir, selector_names, config),
        "ledger_events": validate_access_ledger(output_dir),
        "prediction_manifest": validate_prediction_manifest(output_dir),
        "oracle_counts": validate_oracle_consistency(output_dir),
        "execution_manifest": validate_execution_manifest(output_dir),
    }
