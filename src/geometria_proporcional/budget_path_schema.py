"""Structural types for the candidate BudgetPath interface.

This module declares representation only. It contains no Pareto, dominance,
selection, or adjudication logic; the builder and independent checker remain
separate consumers of the schema.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = "proportional-budget-path-artifact-v1"
POLICY_IDS = (
    "identity",
    "budget_0.01",
    "budget_0.02",
    "budget_0.05",
    "budget_0.10",
    "budget_0.20",
    "budget_0.40",
)
BUDGET_FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
PAIR_STATES = (
    "EXPANSION_DOMINATES",
    "BASE_DOMINATES",
    "IID_COST_GROUPED_GAIN",
    "IID_GAIN_GROUPED_COST",
    "EQUIVALENT",
)
TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "lineage",
        "axes",
        "policies",
        "pairs",
        "reader",
        "authority",
        "utility_boundary",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data_sha256": sha256_bytes(array.tobytes(order="C")),
    }
    return sha256_bytes(canonical_json(descriptor).encode("utf-8"))


def artifact_id(lineage: Mapping[str, Any]) -> str:
    digest = sha256_bytes(canonical_json(dict(lineage)).encode("utf-8"))
    return f"budget-path-{digest[:24]}"


@dataclass(frozen=True)
class BudgetPathArtifact:
    schema_version: str
    artifact_id: str
    lineage: dict[str, Any]
    axes: list[dict[str, Any]]
    policies: list[dict[str, Any]]
    pairs: list[dict[str, Any]]
    reader: dict[str, Any]
    authority: dict[str, Any]
    utility_boundary: dict[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BudgetPathArtifact":
        if set(value) != TOP_LEVEL_FIELDS:
            missing = sorted(TOP_LEVEL_FIELDS - set(value))
            extra = sorted(set(value) - TOP_LEVEL_FIELDS)
            raise ValueError(
                f"artifact schema mismatch: missing={missing}, extra={extra}"
            )
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"artifact schema_version must be {SCHEMA_VERSION}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
