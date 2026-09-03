"""Deterministic generator for the Wave 49 relational benchmark."""

from __future__ import annotations

import hmac
import json
import platform
import secrets
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .wave49_attestation import file_record, sign_attestation
from .wave49_schema import (
    CATALOG_FAMILIES,
    OUT_OF_CATALOG_FAMILIES,
    SPLITS,
    ProtocolConfig,
    canonical_json,
    public_parameter_catalog,
    sha256_file,
    sha256_bytes,
    write_json,
    write_jsonl,
)


def _family_curve(family: str, x: np.ndarray, params: dict[str, float]) -> np.ndarray:
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
    raise ValueError(f"Unknown generator family: {family}")


def _sample_params(family: str, rival_distance: str, rng: np.random.Generator) -> dict[str, float]:
    if family == "PROP":
        return {"k": float(rng.choice([0.5, 0.75, 1.0, 1.5, 2.0]))}
    if family == "AFFINE_OFFSET":
        return {
            "a": float(rng.choice([-0.25, 0.25] if rival_distance == "near" else [-1.0, -0.75, 0.75, 1.0])),
            "b": float(rng.choice([0.5, 1.0, 1.5, 2.0])),
        }
    if family == "POWER_NONUNIT":
        return {
            "a": float(rng.choice([0.5, 1.0, 1.5, 2.0])),
            "p": float(rng.choice([0.9, 1.1] if rival_distance == "near" else [0.5, 0.75, 1.5, 2.0])),
        }
    if family == "SATURATING":
        return {
            "L": float(rng.choice([1.0, 2.0, 3.0, 4.0])),
            "K": float(rng.choice([3.0, 4.0] if rival_distance == "near" else [0.25, 0.5, 1.0])),
        }
    if family == "PIECEWISE_AFFINE":
        b1 = float(rng.choice([0.5, 1.0, 1.5]))
        return {
            "b1": b1,
            "b2": b1 + (0.25 if rival_distance == "near" else float(rng.choice([1.25, 1.75]))),
            "knot": 1.0,
        }
    if family == "OFFSET_QUADRATIC":
        return {
            "a": float(rng.choice([-0.25, 0.25] if rival_distance == "near" else [-0.75, 0.75])),
            "b": float(rng.choice([0.5, 1.0])),
            "c": float(rng.choice([0.25] if rival_distance == "near" else [0.75, 1.0])),
        }
    raise ValueError(f"Unknown generator family: {family}")


def _covariances(
    latent_x: np.ndarray,
    noise_mode: str,
    covariance_mode: str,
    rho: float,
) -> np.ndarray:
    sx_base, sy_base = {
        "low_balanced": (0.03, 0.03),
        "high_balanced": (0.15, 0.15),
        "x_dominant": (0.15, 0.03),
        "y_dominant": (0.03, 0.15),
    }[noise_mode]
    if covariance_mode == "heteroscedastic":
        position = (latent_x - latent_x.min()) / max(float(np.ptp(latent_x)), 1e-12)
        scale = 0.65 + 0.7 * position
    else:
        scale = np.ones_like(latent_x)
    sx = sx_base * scale
    sy = sy_base * scale
    cov = np.zeros((len(latent_x), 2, 2), dtype=np.float64)
    cov[:, 0, 0] = sx * sx
    cov[:, 1, 1] = sy * sy
    cov[:, 0, 1] = rho * sx * sy
    cov[:, 1, 0] = cov[:, 0, 1]
    return cov


def _opaque_id(secret: bytes, namespace: str, payload: str) -> str:
    digest = hmac.new(secret, f"{namespace}:{payload}".encode("utf-8"), "sha256").hexdigest()
    return f"w49-{digest[:24]}"


def _sealed_key(sealed_dir: Path, filename: str, purpose: str, supplied: bytes | None = None) -> bytes:
    path = sealed_dir / filename
    if supplied is not None:
        secret = supplied
        if len(secret) != 32:
            raise ValueError(f"{purpose} key must contain exactly 32 bytes")
        write_json(path, {"key_hex": secret.hex(), "purpose": purpose})
        return secret
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        secret = bytes.fromhex(payload.get("key_hex", payload.get("hmac_key_hex")))
        if len(secret) != 32:
            raise ValueError(f"{purpose} key must contain exactly 32 bytes")
        return secret
    secret = secrets.token_bytes(32)
    write_json(path, {"key_hex": secret.hex(), "purpose": purpose})
    return secret


def _rng_from_key(key: bytes, namespace: str) -> np.random.Generator:
    digest = hmac.new(key, namespace.encode("utf-8"), "sha256").digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _semantic_commitment(row: dict[str, Any], key: bytes) -> str:
    return hmac.new(key, canonical_json(row).encode("utf-8"), "sha256").hexdigest()


def _write_semantic_commitments(
    output_dir: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    key: bytes,
) -> Path:
    rows = [
        {
            "fixture_id": row["fixture_id"],
            "split": split,
            "semantic_hmac_sha256": _semantic_commitment(row, key),
        }
        for split, sealed_rows in rows_by_split.items()
        for row in sealed_rows
    ]
    path = output_dir / "commitments" / "semantic.jsonl"
    write_jsonl(path, rows)
    return path


def _design_separation_index(
    family: str,
    latent_x: np.ndarray,
    clean_y: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Generator-side design statistic, independently recomputed by the oracle."""
    inv_cov = np.linalg.inv(covariance)
    weight_y = inv_cov[:, 1, 1]
    distances: dict[str, float] = {}
    for candidate_family, candidates in public_parameter_catalog().items():
        best = np.inf
        for params in candidates:
            delta_y = clean_y - _family_curve(candidate_family, latent_x, params)
            best = min(best, 0.5 * float(np.sum(weight_y * delta_y * delta_y)))
        distances[candidate_family] = float(best)
    if family in OUT_OF_CATALOG_FAMILIES:
        return min(distances.values())
    return min(distance for candidate, distance in distances.items() if candidate != family)


def _target_region(family: str, design_separation: float, config: ProtocolConfig) -> str:
    """Assign the preregistered region before selector execution."""
    if family in OUT_OF_CATALOG_FAMILIES:
        return "OUT_OF_CATALOG"
    if design_separation <= config.oracle_compatibility_distance:
        return "DELIBERATELY_INDISTINGUISHABLE"
    return "IDENTIFIABLE"


def _target_region_basis(design_separation: float, config: ProtocolConfig) -> dict[str, Any]:
    return {
        "rule": "generator_preexecution_separation_vs_compatibility_distance",
        "design_separation_index": float(design_separation),
        "compatibility_distance": float(config.oracle_compatibility_distance),
    }


def _calibration_population(representation: str) -> str:
    if representation in {"original", "positive_rescale"}:
        return "canonical_preserving"
    if representation == "origin_translation_break":
        return "origin_translation_break"
    raise ValueError(f"Unknown representation mode: {representation}")


def _generate_one(
    split: str,
    family: str,
    ordinal: int,
    condition: tuple[int, str, str, str, float, str],
    config: ProtocolConfig,
    rng: np.random.Generator,
    identity_secret: bytes,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n, range_mode, noise_mode, covariance_mode, rho, rival_distance = condition
    if range_mode == "wide":
        lo, hi = 0.25, 2.0
    else:
        lo, hi = 0.8, 1.2
    latent_x = np.linspace(lo, hi, n, dtype=np.float64)
    params = _sample_params(family, rival_distance, rng)
    identity_payload = json.dumps(
        {"split": split, "ordinal": ordinal, "family": family, "condition": condition},
        sort_keys=True,
        separators=(",", ":"),
    )
    pair_token = _opaque_id(identity_secret, "pair", identity_payload)
    base_clean_y = _family_curve(family, latent_x, params)
    base_covariance = _covariances(latent_x, noise_mode, covariance_mode, rho)
    base_errors = np.stack([rng.multivariate_normal(np.zeros(2), c) for c in base_covariance])
    base_observed = np.column_stack([latent_x, base_clean_y]) + base_errors
    base_design_separation = _design_separation_index(
        family, latent_x, base_clean_y, base_covariance
    )

    visible_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    design_stratum = (
        "OUT_OF_CATALOG"
        if family in OUT_OF_CATALOG_FAMILIES
        else ("NEAR_RIVAL" if rival_distance == "near" else "FAR_RIVAL")
    )
    for covariance_knowledge in config.covariance_knowledge_modes:
        for representation in config.representation_modes:
            if representation == "original":
                x_scale, y_scale = 1.0, 1.0
            elif representation == "positive_rescale":
                x_scale = float(rng.choice([0.5, 2.0]))
                y_scale = float(rng.choice([0.5, 2.0]))
            else:
                raise ValueError(f"Unknown representation mode: {representation}")

            jacobian = np.diag([x_scale, y_scale])
            true_covariance = np.einsum("ab,nbc,dc->nad", jacobian, base_covariance, jacobian)
            reported_covariance = true_covariance.copy()
            if covariance_knowledge == "diagonal_only":
                reported_covariance[:, 0, 1] = 0.0
                reported_covariance[:, 1, 0] = 0.0
            observed = base_observed @ jacobian.T
            tx = latent_x * x_scale

            fixture_id = _opaque_id(
                identity_secret,
                "fixture",
                f"{identity_payload}:{covariance_knowledge}:{representation}",
            )
            visible_rows.append(
                {
                    "schema_version": config.schema_version,
                    "fixture_id": fixture_id,
                    "split": split,
                    "n": n,
                    "x": observed[:, 0].tolist(),
                    "y": observed[:, 1].tolist(),
                    "covariance": reported_covariance.tolist(),
                    "domain": [float(tx.min()), float(tx.max())],
                    "coordinate_semantics": {
                        "x_origin_declared": True,
                        "y_origin_declared": True,
                        "positive_rescaling_allowed": True,
                        "x_scale_to_canonical": 1.0 / x_scale,
                        "y_scale_to_canonical": 1.0 / y_scale,
                        "covariance_knowledge": covariance_knowledge,
                        "calibration_population": _calibration_population(representation),
                    },
                }
            )
            sealed_rows.append(
                {
                    "schema_version": config.schema_version,
                    "fixture_id": fixture_id,
                    "split": split,
                    "pair_token": pair_token,
                    "representation": representation,
                    "covariance_knowledge": covariance_knowledge,
                    "family_id": family,
                    "generator_params": params,
                    "latent_x": latent_x.tolist(),
                    "clean_y": base_clean_y.tolist(),
                    "true_covariance_canonical": base_covariance.tolist(),
                    "realized_errors_canonical": base_errors.tolist(),
                    "is_out_of_catalog": family in OUT_OF_CATALOG_FAMILIES,
                    "design_stratum": design_stratum,
                    "target_region": _target_region(family, base_design_separation, config),
                    "target_region_basis": _target_region_basis(base_design_separation, config),
                    "design_separation_index": float(base_design_separation),
                    "calibration_population": _calibration_population(representation),
                    "noise_mode": noise_mode,
                    "covariance_mode": covariance_mode,
                    "range_mode": range_mode,
                    "rho": rho,
                    "rival_distance_mode": rival_distance,
                }
            )

    if family == "PROP":
        offset = float(rng.choice([-0.5, 0.5]))
        translated = base_observed.copy()
        translated[:, 1] += offset
        translated_clean_y = base_clean_y + offset
        translated_design_separation = _design_separation_index(
            "AFFINE_OFFSET", latent_x, translated_clean_y, base_covariance
        )
        for covariance_knowledge in config.covariance_knowledge_modes:
            reported_covariance = base_covariance.copy()
            if covariance_knowledge == "diagonal_only":
                reported_covariance[:, 0, 1] = 0.0
                reported_covariance[:, 1, 0] = 0.0
            fixture_id = _opaque_id(
                identity_secret,
                "fixture",
                f"{identity_payload}:{covariance_knowledge}:origin_translation_break",
            )
            visible_rows.append(
                {
                    "schema_version": config.schema_version,
                    "fixture_id": fixture_id,
                    "split": split,
                    "n": n,
                    "x": translated[:, 0].tolist(),
                    "y": translated[:, 1].tolist(),
                    "covariance": reported_covariance.tolist(),
                    "domain": [float(latent_x.min()), float(latent_x.max())],
                    "coordinate_semantics": {
                        "x_origin_declared": True,
                        "y_origin_declared": True,
                        "positive_rescaling_allowed": True,
                        "x_scale_to_canonical": 1.0,
                        "y_scale_to_canonical": 1.0,
                        "covariance_knowledge": covariance_knowledge,
                        "calibration_population": _calibration_population("origin_translation_break"),
                    },
                }
            )
            sealed_rows.append(
                {
                    "schema_version": config.schema_version,
                    "fixture_id": fixture_id,
                    "split": split,
                    "pair_token": pair_token,
                    "representation": "origin_translation_break",
                    "covariance_knowledge": covariance_knowledge,
                    "family_id": "AFFINE_OFFSET",
                    "generator_params": {"a": offset, "b": params["k"]},
                    "latent_x": latent_x.tolist(),
                    "clean_y": translated_clean_y.tolist(),
                    "true_covariance_canonical": base_covariance.tolist(),
                    "realized_errors_canonical": base_errors.tolist(),
                    "is_out_of_catalog": False,
                    "design_stratum": "TRANSLATION_RUPTURE",
                    "target_region": _target_region(
                        "AFFINE_OFFSET", translated_design_separation, config
                    ),
                    "target_region_basis": _target_region_basis(
                        translated_design_separation, config
                    ),
                    "design_separation_index": float(translated_design_separation),
                    "calibration_population": _calibration_population("origin_translation_break"),
                    "noise_mode": noise_mode,
                    "covariance_mode": covariance_mode,
                    "range_mode": range_mode,
                    "rho": rho,
                    "rival_distance_mode": rival_distance,
                    "source_family": "PROP",
                }
            )

    return visible_rows, sealed_rows


def generate_benchmark(
    output_dir: Path,
    config: ProtocolConfig,
    generation_key: bytes | None = None,
    identity_key: bytes | None = None,
    commitment_key: bytes | None = None,
    attestation_private_key_path: Path | None = None,
    trusted_public_key_path: Path | None = None,
) -> dict[str, Any]:
    """Generate visible and sealed packages with separate hashes."""
    output_dir = Path(output_dir)
    visible_dir = output_dir / "visible"
    sealed_dir = output_dir / "sealed"
    visible_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)
    generation_key = _sealed_key(
        sealed_dir, "generation_secret.json", "private deterministic generation", generation_key
    )
    identity_secret = _sealed_key(
        sealed_dir, "identity_secret.json", "opaque fixture identity", identity_key
    )
    commitment_secret = _sealed_key(
        sealed_dir,
        "semantic_commitment_secret.json",
        "pre-execution sealed semantic commitment",
        commitment_key,
    )
    if len({generation_key, identity_secret, commitment_secret}) != 3:
        raise ValueError("generation, identity, and semantic commitment keys must be distinct")

    all_families = CATALOG_FAMILIES + OUT_OF_CATALOG_FAMILIES
    counts: dict[str, int] = {}
    sealed_by_split: dict[str, list[dict[str, Any]]] = {}
    conditions = list(product(
        config.sample_sizes,
        config.range_modes,
        config.noise_modes,
        config.covariance_modes,
        config.correlations,
        config.rival_distance_modes,
    ))
    for split_index, split in enumerate(SPLITS):
        rng = _rng_from_key(generation_key, f"split:{split_index}:{split}")
        visible_rows: list[dict[str, Any]] = []
        sealed_rows: list[dict[str, Any]] = []
        ordinal = 0
        for family in all_families:
            for condition in conditions:
                for _ in range(config.replicates_per_condition):
                    vis, truth = _generate_one(split, family, ordinal, condition, config, rng, identity_secret)
                    visible_rows.extend(vis)
                    sealed_rows.extend(truth)
                    ordinal += 1
        rng.shuffle(visible_rows)
        rng.shuffle(sealed_rows)
        write_jsonl(visible_dir / f"{split}.jsonl", visible_rows)
        write_jsonl(sealed_dir / f"{split}.jsonl", sealed_rows)
        sealed_by_split[split] = sealed_rows
        counts[split] = len(visible_rows)

    # Each target population gets an independent null sample. Original and
    # positive-rescale share the canonical-preserving population because the
    # selector canonicalizes them exactly; translated coordinates do not.
    calibration_visible: list[dict[str, Any]] = []
    calibration_sealed: list[dict[str, Any]] = []
    rng = _rng_from_key(generation_key, "calibration_null")
    nuisance_by_n = {
        n: list(product(
            config.range_modes,
            config.noise_modes,
            config.covariance_modes,
            config.correlations,
            config.rival_distance_modes,
        ))
        for n in config.sample_sizes
    }
    calibration_populations = ("canonical_preserving", "origin_translation_break")
    for n in config.sample_sizes:
        for population_index, population in enumerate(calibration_populations):
            for index in range(config.calibration_null_per_n):
                family = (
                    str(rng.choice(CATALOG_FAMILIES))
                    if population == "canonical_preserving"
                    else "PROP"
                )
                nuisance = nuisance_by_n[n][int(rng.integers(len(nuisance_by_n[n])))]
                condition = (
                    n,
                    str(nuisance[0]),
                    str(nuisance[1]),
                    str(nuisance[2]),
                    float(nuisance[3]),
                    str(nuisance[4]),
                )
                ordinal = 10_000_000 + population_index * 1_000_000 + n * 10_000 + index
                vis, truth = _generate_one(
                    "calibration_null", family, ordinal, condition, config, rng, identity_secret,
                )
                representation = (
                    "original" if population == "canonical_preserving"
                    else "origin_translation_break"
                )
                for knowledge in config.covariance_knowledge_modes:
                    selected = next(
                        row for row in truth
                        if row["representation"] == representation
                        and row["covariance_knowledge"] == knowledge
                    )
                    calibration_visible.append(
                        next(row for row in vis if row["fixture_id"] == selected["fixture_id"])
                    )
                    calibration_sealed.append(selected)
    write_jsonl(visible_dir / "calibration_null.jsonl", calibration_visible)
    write_jsonl(sealed_dir / "calibration_null.jsonl", calibration_sealed)
    sealed_by_split["calibration_null"] = calibration_sealed
    counts["calibration_null"] = len(calibration_visible)

    commitment_path = _write_semantic_commitments(
        output_dir, sealed_by_split, commitment_secret
    )

    config_path = output_dir / "protocol_config.json"
    write_json(config_path, config.to_dict())
    if attestation_private_key_path is None or trusted_public_key_path is None:
        raise ValueError("detached attestation private and trusted public keys are required")
    generation_key_commitment = sha256_bytes(generation_key)
    identity_key_commitment = sha256_bytes(identity_secret)
    semantic_key_commitment = sha256_bytes(commitment_secret)
    sealed_truth_paths = {
        split: sealed_dir / f"{split}.jsonl"
        for split in (*SPLITS, "calibration_null")
    }
    attestation_payload = {
        "phase": "sealed-semantics-committed-before-selector",
        "schema_version": config.schema_version,
        "protocol_config": file_record(config_path),
        "semantic_commitments": file_record(commitment_path),
        "sealed_truth": {
            split: file_record(path) for split, path in sealed_truth_paths.items()
        },
        "counts": counts,
        "key_commitments": {
            "generation": generation_key_commitment,
            "identity": identity_key_commitment,
            "semantic_hmac": semantic_key_commitment,
        },
    }
    attestation_path = output_dir / "attestations" / "semantic_root.json"
    write_json(
        attestation_path,
        sign_attestation(
            attestation_payload,
            Path(attestation_private_key_path),
            Path(trusted_public_key_path),
        ),
    )
    files = [config_path]
    files.extend(visible_dir / f"{split}.jsonl" for split in SPLITS)
    files.extend(sealed_dir / f"{split}.jsonl" for split in SPLITS)
    files.extend([
        visible_dir / "calibration_null.jsonl",
        sealed_dir / "calibration_null.jsonl",
        sealed_dir / "identity_secret.json",
        sealed_dir / "generation_secret.json",
        sealed_dir / "semantic_commitment_secret.json",
        commitment_path,
        attestation_path,
    ])
    manifest = {
        "schema_version": config.schema_version,
        "generator": "wave49_generator",
        "generation_key_commitment": generation_key_commitment,
        "identity_key_commitment": identity_key_commitment,
        "semantic_commitment_key_commitment": semantic_key_commitment,
        "semantic_attestation": {
            "path": str(attestation_path.relative_to(output_dir)),
            "phase": attestation_payload["phase"],
            "trusted_public_key_sha256": json.loads(
                attestation_path.read_text(encoding="utf-8")
            )["trusted_public_key_sha256"],
        },
        "calibration_contract": {
            "sampling_unit": (
                "one independent base draw per (n,population,index); covariance-knowledge "
                "views are paired across strata"
            ),
            "sampling_law": config.calibration_sampling_law,
            "coverage_scope": config.calibration_coverage_scope,
            "target_population_law": {
                "canonical_preserving": (
                    "family uniform over the declared in-catalog families; nuisance cell uniform "
                    "over the declared factorial; original coordinates represent the exactly "
                    "canonicalized original/positive-rescale population"
                ),
                "origin_translation_break": (
                    "PROP source with a sampled nonzero translation; nuisance cell uniform over "
                    "the declared factorial"
                ),
            },
            "guarantee": (
                "marginal finite-sample false-abstention control for exchangeable in-catalog "
                "fixtures within each exact (n,covariance_knowledge,calibration_population) stratum"
            ),
            "factor_diagnostics_are_conditional_guarantees": False,
        },
        "counts": counts,
        "catalog_families": list(CATALOG_FAMILIES),
        "out_of_catalog_families": list(OUT_OF_CATALOG_FAMILIES),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "files": {
            str(path.relative_to(output_dir)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in files
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    sealed_dir.chmod(0o700)
    return manifest
