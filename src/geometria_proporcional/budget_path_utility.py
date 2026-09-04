"""External synthetic-only utility port for BudgetPath artifacts.

The port consumes a checked path without mutating it. Version 1 deliberately
accepts only synthetic fixtures; empirical or user authority requires a future
declaration supplied by the actual owner of that utility.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .budget_path_schema import POLICY_IDS, BudgetPathArtifact, canonical_json

UTILITY_SCHEMA_VERSION = "proportional-budget-path-external-utility-v1"
DECISION_SCHEMA_VERSION = "proportional-budget-path-external-decision-v1"
AXIS_IDS = ("iid_delta", "grouped_delta")
UTILITY_KINDS = (
    "WEIGHTED_SUM_MINIMIZE",
    "LEXICOGRAPHIC_MINIMIZE",
    "EPSILON_CONSTRAINT_MINIMIZE",
)
DECLARATION_FIELDS = frozenset(
    {
        "schema_version",
        "declaration_id",
        "scope",
        "artifact_id",
        "artifact_sha256",
        "checker_receipt_sha256",
        "utility_kind",
        "parameters",
        "tie_policy",
        "decision_authority",
    }
)
RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "artifact_sha256",
        "checker_status",
        "scope",
        "checker_kind",
    }
)
TOLERANCE = 1e-12
EXPECTED_AXES = [
    {
        "axis_id": "iid_delta",
        "population": "IID",
        "estimator": "mean_quotient_rmse_delta",
        "unit": "quotient_rmse",
        "direction": "minimize",
    },
    {
        "axis_id": "grouped_delta",
        "population": "grouped",
        "estimator": "mean_quotient_rmse_delta",
        "unit": "quotient_rmse",
        "direction": "minimize",
    },
]
EXPECTED_SYNTHETIC_AUTHORITY = {
    "artifact_status": "SYNTHETIC_FIXTURE",
    "formal_claim_status": "STRUCTURE_ONLY",
    "empirical_claim_status": "NOT_APPLICABLE",
    "physical_authority_status": "NOT_CLAIMED",
    "decision_status": "UNRESOLVED",
}


class ExternalUtilityViolation(ValueError):
    """Raised when an external utility declaration or binding is invalid."""


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def declaration_id(payload_without_id: Mapping[str, Any]) -> str:
    return f"utility-{digest(dict(payload_without_id))[:24]}"


def decision_id(payload_without_id: Mapping[str, Any]) -> str:
    return f"decision-{digest(dict(payload_without_id))[:24]}"


@dataclass(frozen=True)
class ExternalUtilityDeclaration:
    schema_version: str
    declaration_id: str
    scope: str
    artifact_id: str
    artifact_sha256: str
    checker_receipt_sha256: str
    utility_kind: str
    parameters: dict[str, Any]
    tie_policy: str
    decision_authority: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExternalUtilityDeclaration":
        if set(value) != DECLARATION_FIELDS:
            raise ExternalUtilityViolation("utility declaration fields mismatch")
        if value.get("schema_version") != UTILITY_SCHEMA_VERSION:
            raise ExternalUtilityViolation("utility schema mismatch")
        declaration = cls(**dict(value))
        payload = declaration.to_dict()
        payload.pop("declaration_id")
        if declaration.declaration_id != declaration_id(payload):
            raise ExternalUtilityViolation("utility declaration id mismatch")
        return declaration

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_declaration(
    *,
    artifact: Mapping[str, Any],
    checker_receipt: Mapping[str, Any],
    utility_kind: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": UTILITY_SCHEMA_VERSION,
        "scope": "SYNTHETIC_FIXTURE",
        "artifact_id": artifact["artifact_id"],
        "artifact_sha256": digest(artifact),
        "checker_receipt_sha256": digest(checker_receipt),
        "utility_kind": utility_kind,
        "parameters": dict(parameters),
        "tie_policy": "RETURN_ALL_OPTIMA",
        "decision_authority": "SYNTHETIC_TEST_ONLY",
    }
    return {"declaration_id": declaration_id(payload), **payload}


def _numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExternalUtilityViolation(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ExternalUtilityViolation(f"{label} must be finite")
    return result


def _validate_binding(
    artifact: BudgetPathArtifact,
    raw_artifact: Mapping[str, Any],
    receipt: Mapping[str, Any],
    declaration: ExternalUtilityDeclaration,
) -> None:
    if set(receipt) != RECEIPT_FIELDS:
        raise ExternalUtilityViolation("checker receipt fields mismatch")
    if receipt["schema_version"] != "budget-path-synthetic-receipt-v1":
        raise ExternalUtilityViolation("checker receipt schema mismatch")
    if receipt["checker_status"] != "VALID":
        raise ExternalUtilityViolation("checker receipt is not valid")
    if receipt["scope"] != "SYNTHETIC_FIXTURE":
        raise ExternalUtilityViolation("checker receipt scope mismatch")
    if receipt["checker_kind"] != "SYNTHETIC_FIXTURE_CHECKER":
        raise ExternalUtilityViolation("checker receipt kind mismatch")
    artifact_hash = digest(raw_artifact)
    if receipt["artifact_id"] != artifact.artifact_id:
        raise ExternalUtilityViolation("checker receipt artifact id mismatch")
    if receipt["artifact_sha256"] != artifact_hash:
        raise ExternalUtilityViolation("checker receipt artifact hash mismatch")
    if declaration.scope != "SYNTHETIC_FIXTURE":
        raise ExternalUtilityViolation("utility scope must be SYNTHETIC_FIXTURE")
    if declaration.decision_authority != "SYNTHETIC_TEST_ONLY":
        raise ExternalUtilityViolation("decision authority must be synthetic only")
    if declaration.tie_policy != "RETURN_ALL_OPTIMA":
        raise ExternalUtilityViolation("tie policy must return all optima")
    if declaration.artifact_id != artifact.artifact_id:
        raise ExternalUtilityViolation("utility artifact id mismatch")
    if declaration.artifact_sha256 != artifact_hash:
        raise ExternalUtilityViolation("utility artifact hash mismatch")
    if declaration.checker_receipt_sha256 != digest(receipt):
        raise ExternalUtilityViolation("utility checker receipt hash mismatch")


def _decision_surface(
    artifact: BudgetPathArtifact,
) -> tuple[list[str], dict[str, dict[str, float]], list[str]]:
    if artifact.lineage.get("scope") != "SYNTHETIC_FIXTURE":
        raise ExternalUtilityViolation("BudgetPath lineage must be synthetic fixture")
    if artifact.authority != EXPECTED_SYNTHETIC_AUTHORITY:
        raise ExternalUtilityViolation("BudgetPath authority must be synthetic only")
    if artifact.axes != EXPECTED_AXES:
        raise ExternalUtilityViolation("BudgetPath axes mismatch")
    if artifact.utility_boundary != {
        "status": "ABSENT_EXTERNAL_REQUIRED",
        "embedded_utility_fields": [],
    }:
        raise ExternalUtilityViolation("BudgetPath utility boundary is not external")
    if artifact.reader.get("kind") != "PARETO_SET" or set(artifact.reader) != {
        "kind",
        "policy_ids",
    }:
        raise ExternalUtilityViolation("BudgetPath reader must be PARETO_SET")
    policy_order = [row.get("policy_id") for row in artifact.policies]
    if policy_order != list(POLICY_IDS):
        raise ExternalUtilityViolation("BudgetPath policy ids or order mismatch")
    candidates = artifact.reader["policy_ids"]
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(candidates) != len(set(candidates))
        or not set(candidates) <= set(policy_order)
    ):
        raise ExternalUtilityViolation(
            "BudgetPath reader has unknown or duplicate candidate"
        )
    coordinates: dict[str, dict[str, float]] = {}
    for row in artifact.policies:
        means = row.get("objective_mean")
        if not isinstance(means, dict) or set(means) != set(AXIS_IDS):
            raise ExternalUtilityViolation("BudgetPath objective coordinates mismatch")
        coordinates[row["policy_id"]] = {
            axis: _numeric(means[axis], f"coordinate {axis}") for axis in AXIS_IDS
        }
    canonical_candidates = [policy for policy in policy_order if policy in candidates]
    return canonical_candidates, coordinates, policy_order


def _weighted(
    candidates: list[str],
    coordinates: dict[str, dict[str, float]],
    parameters: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    if set(parameters) != {"weights"} or not isinstance(parameters["weights"], dict):
        raise ExternalUtilityViolation("weighted utility parameters mismatch")
    weights = parameters["weights"]
    if set(weights) != set(AXIS_IDS):
        raise ExternalUtilityViolation("weighted utility axes mismatch")
    numeric = {axis: _numeric(weights[axis], f"weight {axis}") for axis in AXIS_IDS}
    if any(value <= 0.0 for value in numeric.values()):
        raise ExternalUtilityViolation(
            "weighted utility requires strictly positive weights"
        )
    if abs(sum(numeric.values()) - 1.0) > TOLERANCE:
        raise ExternalUtilityViolation("weighted utility weights must sum to one")
    values = {
        policy: sum(numeric[axis] * coordinates[policy][axis] for axis in AXIS_IDS)
        for policy in candidates
    }
    optimum = min(values.values())
    selected = [
        policy for policy in candidates if abs(values[policy] - optimum) <= TOLERANCE
    ]
    return selected, {"kind": "WEIGHTED_SUM", "values": values, "optimum": optimum}


def _lexicographic(
    candidates: list[str],
    coordinates: dict[str, dict[str, float]],
    parameters: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    if set(parameters) != {"axis_priority"} or not isinstance(
        parameters["axis_priority"], list
    ):
        raise ExternalUtilityViolation("lexicographic utility parameters mismatch")
    priority = parameters["axis_priority"]
    if len(priority) != len(AXIS_IDS) or set(priority) != set(AXIS_IDS):
        raise ExternalUtilityViolation(
            "lexicographic priority must be an axis permutation"
        )
    selected = list(candidates)
    stage_optima = []
    for axis in priority:
        optimum = min(coordinates[policy][axis] for policy in selected)
        stage_optima.append({"axis": axis, "optimum": optimum})
        selected = [
            policy
            for policy in selected
            if abs(coordinates[policy][axis] - optimum) <= TOLERANCE
        ]
    return selected, {"kind": "LEXICOGRAPHIC", "stages": stage_optima}


def _epsilon_constraint(
    candidates: list[str],
    coordinates: dict[str, dict[str, float]],
    parameters: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    if set(parameters) != {"primary_axis", "constraint_axis", "max_value"}:
        raise ExternalUtilityViolation("epsilon utility parameters mismatch")
    primary = parameters["primary_axis"]
    constraint = parameters["constraint_axis"]
    if primary not in AXIS_IDS or constraint not in AXIS_IDS or primary == constraint:
        raise ExternalUtilityViolation(
            "epsilon utility axes must be distinct known axes"
        )
    maximum = _numeric(parameters["max_value"], "epsilon max_value")
    feasible = [
        policy
        for policy in candidates
        if coordinates[policy][constraint] <= maximum + TOLERANCE
    ]
    evidence = {
        "kind": "EPSILON_CONSTRAINT",
        "primary_axis": primary,
        "constraint_axis": constraint,
        "max_value": maximum,
        "feasible_policy_ids": feasible,
    }
    if not feasible:
        return [], evidence
    optimum = min(coordinates[policy][primary] for policy in feasible)
    evidence["optimum"] = optimum
    return [
        policy
        for policy in feasible
        if abs(coordinates[policy][primary] - optimum) <= TOLERANCE
    ], evidence


def evaluate_external_utility(
    raw_artifact: Mapping[str, Any],
    checker_receipt: Mapping[str, Any],
    raw_declaration: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        artifact = BudgetPathArtifact.from_dict(raw_artifact)
    except ValueError as exc:
        raise ExternalUtilityViolation(str(exc)) from exc
    declaration = ExternalUtilityDeclaration.from_dict(raw_declaration)
    _validate_binding(artifact, raw_artifact, checker_receipt, declaration)
    candidates, coordinates, _ = _decision_surface(artifact)
    if declaration.utility_kind not in UTILITY_KINDS:
        raise ExternalUtilityViolation("unsupported utility kind")
    if declaration.utility_kind == "WEIGHTED_SUM_MINIMIZE":
        selected, evidence = _weighted(candidates, coordinates, declaration.parameters)
    elif declaration.utility_kind == "LEXICOGRAPHIC_MINIMIZE":
        selected, evidence = _lexicographic(
            candidates, coordinates, declaration.parameters
        )
    else:
        selected, evidence = _epsilon_constraint(
            candidates, coordinates, declaration.parameters
        )
    status = "SYNTHETIC_FIXTURE_SELECTED" if selected else "ABSTAIN_NO_FEASIBLE_POLICY"
    payload = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "declaration_id": declaration.declaration_id,
        "artifact_id": artifact.artifact_id,
        "scope": "SYNTHETIC_FIXTURE",
        "decision_authority": "SYNTHETIC_TEST_ONLY",
        "decision_status": status,
        "candidate_policy_ids": candidates,
        "selected_policy_ids": selected,
        "evaluation_evidence": evidence,
    }
    return {"decision_id": decision_id(payload), **payload}
