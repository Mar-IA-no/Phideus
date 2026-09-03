"""Shared structural schema for the Wave 49 relational benchmark.

This module deliberately contains no family-evaluation or adjudication logic.
Generator, oracle, selector, and checker implement their semantic decisions
separately so a shared helper cannot make them agree by construction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CATALOG_FAMILIES = (
    "PROP",
    "AFFINE_OFFSET",
    "POWER_NONUNIT",
    "SATURATING",
)
OUT_OF_CATALOG_FAMILIES = ("PIECEWISE_AFFINE", "OFFSET_QUADRATIC")
SPLITS = ("train", "val", "lockbox")
SCHEMA_VERSION = "wave49-relational-benchmark-v2"

FORBIDDEN_VISIBLE_KEYS = {
    "clean_y",
    "design_stratum",
    "design_separation_index",
    "family_id",
    "generator_family",
    "generator_params",
    "is_out_of_catalog",
    "latent_x",
    "logical_structural_set",
    "oracle_compatible_set",
    "oracle_status",
    "pair_token",
    "property_set",
    "realized_errors_canonical",
    "separation_band",
    "separation_index",
    "target_region",
    "target_region_basis",
    "true_covariance_canonical",
}


@dataclass(frozen=True)
class ProtocolConfig:
    schema_version: str = SCHEMA_VERSION
    replicates_per_condition: int = 1
    calibration_null_per_n: int = 256
    sample_sizes: tuple[int, ...] = (8, 16, 24)
    range_modes: tuple[str, ...] = ("narrow", "wide")
    noise_modes: tuple[str, ...] = ("low_balanced", "high_balanced", "x_dominant", "y_dominant")
    covariance_modes: tuple[str, ...] = ("homoscedastic", "heteroscedastic")
    covariance_knowledge_modes: tuple[str, ...] = ("full", "diagonal_only")
    correlations: tuple[float, ...] = (0.0, 0.4)
    rival_distance_modes: tuple[str, ...] = ("near", "far")
    representation_modes: tuple[str, ...] = ("original", "positive_rescale")
    family_score_delta: float = 4.0
    ood_alpha: float = 0.01
    oracle_compatibility_distance: float = 4.0
    oracle_ood_distance: float = 10.0
    oracle_numeric_atol: float = 1e-10
    separation_bands: tuple[float, ...] = (1e-8, 2.0, 10.0)
    latent_grid_size: int = 96
    oracle_latent_policy: str = "shared_fixed_design"
    calibration_sampling_law: str = "iid_uniform_over_declared_factorial_factors"
    calibration_coverage_scope: str = "marginal_within_n_covariance_knowledge_population"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolConfig":
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"protocol config requires explicit schema_version={SCHEMA_VERSION}")
        tuple_fields = {
            "sample_sizes",
            "range_modes",
            "noise_modes",
            "covariance_modes",
            "covariance_knowledge_modes",
            "correlations",
            "rival_distance_modes",
            "representation_modes",
            "separation_bands",
        }
        normalized = {
            key: tuple(value) if key in tuple_fields else value
            for key, value in data.items()
        }
        return cls(**normalized)


def default_protocol_config(smoke: bool = False) -> ProtocolConfig:
    if not smoke:
        return ProtocolConfig()
    return ProtocolConfig(
        replicates_per_condition=1,
        calibration_null_per_n=16,
        sample_sizes=(12,),
        range_modes=("wide",),
        noise_modes=("low_balanced",),
        covariance_modes=("homoscedastic",),
        covariance_knowledge_modes=("full", "diagonal_only"),
        correlations=(0.0,),
        rival_distance_modes=("far",),
        representation_modes=("original", "positive_rescale"),
        latent_grid_size=64,
    )


def public_parameter_catalog() -> dict[str, list[dict[str, float]]]:
    """Finite public hypothesis catalog used by independent procedures."""
    positive = [0.25 * i for i in range(1, 17)]
    # The affine catalog includes its a=0 boundary so exact PROP/AFFINE
    # observational overlap is represented instead of forced apart.
    offsets = [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    exponents = [0.4, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0]
    saturation_k = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    return {
        "PROP": [{"k": k} for k in positive],
        "AFFINE_OFFSET": [{"a": a, "b": b} for a in offsets for b in positive],
        "POWER_NONUNIT": [{"a": a, "p": p} for a in positive for p in exponents],
        "SATURATING": [{"L": L, "K": K} for L in [0.5 * i for i in range(1, 17)] for K in saturation_k],
    }


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
