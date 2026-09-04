"""Independent checker for candidate BudgetPath artifacts.

The checker reconstructs actions, objectives, Pareto membership, and pairwise
relations from frozen source arrays. It deliberately does not import the
BudgetPath builder or the R371/R372 analysis implementations.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .budget_path_schema import (
    BUDGET_FRACTIONS,
    PAIR_STATES,
    POLICY_IDS,
    BudgetPathArtifact,
    array_digest,
    artifact_id,
)


class BudgetPathViolation(ValueError):
    """Raised when a materialized BudgetPath violates its declared contract."""


LINEAGE_FIELDS = frozenset(
    {
        "cohort",
        "role",
        "proposal_regime",
        "arm",
        "family",
        "control_replicate",
        "source_attribution_manifest_sha256",
        "source_pareto_manifest_sha256",
        "source_pairwise_manifest_sha256",
        "objective_key",
        "action_keys",
    }
)
POLICY_FIELDS = frozenset(
    {
        "policy_id",
        "budget_fraction",
        "action_count",
        "action_sha256",
        "objective_master_count",
        "objective_sha256",
        "objective_mean",
        "front_member",
        "bootstrap_front_frequency",
    }
)
PAIR_FIELDS = frozenset(
    {
        "pair_id",
        "base_policy",
        "expansion_policy",
        "adjacent",
        "mean_increment",
        "increment_sha256",
        "point_status",
        "bootstrap_status_frequency",
    }
)
AXES = [
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
AUTHORITY = {
    "artifact_status": "CHECKABLE_CANDIDATE",
    "formal_claim_status": "STRUCTURE_ONLY",
    "empirical_claim_status": "OPENED_POSTHOC",
    "physical_authority_status": "NOT_CLAIMED",
    "decision_status": "UNRESOLVED",
}
UTILITY_BOUNDARY = {
    "status": "ABSENT_EXTERNAL_REQUIRED",
    "embedded_utility_fields": [],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _verify_manifest(root: Path, expected: str, label: str) -> None:
    if _sha(root / "manifest.json") != expected:
        raise BudgetPathViolation(f"{label} source manifest mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    for rel, digest in manifest["deterministic_files"].items():
        if _sha(root / rel) != digest:
            raise BudgetPathViolation(f"{label} deterministic source mismatch: {rel}")


def _action_key(regime: str, arm: str, fraction: float, role: str, family: str) -> str:
    middle = "action" if role == "policy_selection" else "ranked|action"
    return f"{regime}|{arm}|budget={fraction:.2f}|{middle}|{family}"


def _objective_key(
    regime: str,
    arm: str,
    role: str,
    family: str,
    replicate: int | None,
) -> str:
    suffix = family if replicate is None else f"{family}={replicate}"
    return f"{regime}|{arm}|{role}|{suffix}|objectives"


def _pareto_mask(means: np.ndarray, tolerance: float) -> np.ndarray:
    points = np.asarray(means, dtype=float)
    keep = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        weak = np.all(points <= point + tolerance, axis=1)
        strict = np.any(points < point - tolerance, axis=1)
        if np.any(weak & strict):
            keep[index] = False
    return keep


def _classify_one(delta: np.ndarray, tolerance: float) -> str:
    iid, grouped = np.asarray(delta, dtype=float)
    if abs(iid) <= tolerance and abs(grouped) <= tolerance:
        return "EQUIVALENT"
    if iid <= tolerance and grouped <= tolerance:
        return "EXPANSION_DOMINATES"
    if iid >= -tolerance and grouped >= -tolerance:
        return "BASE_DOMINATES"
    if iid > tolerance and grouped < -tolerance:
        return "IID_COST_GROUPED_GAIN"
    if iid < -tolerance and grouped > tolerance:
        return "IID_GAIN_GROUPED_COST"
    raise BudgetPathViolation(f"unclassified pairwise increment: {delta}")


def _classify_many(delta: np.ndarray, tolerance: float) -> np.ndarray:
    values = np.asarray(delta, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise BudgetPathViolation(
            "pairwise bootstrap matrix must have shape [replicates, 2]"
        )
    iid, grouped = values[:, 0], values[:, 1]
    result = np.full(len(values), -1, dtype=np.int8)
    zero = (np.abs(iid) <= tolerance) & (np.abs(grouped) <= tolerance)
    result[zero] = PAIR_STATES.index("EQUIVALENT")
    expansion = (~zero) & (iid <= tolerance) & (grouped <= tolerance)
    result[expansion] = PAIR_STATES.index("EXPANSION_DOMINATES")
    base = (~zero) & (iid >= -tolerance) & (grouped >= -tolerance)
    result[base] = PAIR_STATES.index("BASE_DOMINATES")
    result[(iid > tolerance) & (grouped < -tolerance)] = PAIR_STATES.index(
        "IID_COST_GROUPED_GAIN"
    )
    result[(iid < -tolerance) & (grouped > tolerance)] = PAIR_STATES.index(
        "IID_GAIN_GROUPED_COST"
    )
    if np.any(result < 0):
        raise BudgetPathViolation("unclassified pairwise bootstrap increment")
    return result


def _float_close(observed: Any, expected: float, tolerance: float) -> bool:
    return (
        not isinstance(observed, bool)
        and isinstance(observed, (int, float))
        and np.isfinite(observed)
        and abs(float(observed) - expected) <= tolerance
    )


def _require_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        raise BudgetPathViolation(f"{label} fields mismatch")


def validate_nested_actions(actions: list[np.ndarray]) -> None:
    if len(actions) != len(POLICY_IDS):
        raise BudgetPathViolation("action path length mismatch")
    if np.any(actions[0] != 0):
        raise BudgetPathViolation("identity action must be zero")
    shape = actions[0].shape
    for action in actions:
        if action.shape != shape or action.ndim != 1:
            raise BudgetPathViolation("action path shape mismatch")
        if not np.issubdtype(action.dtype, np.integer):
            raise BudgetPathViolation("action path must be integer-coded")
    for smaller, larger in zip(actions, actions[1:]):
        if np.any((smaller != 0) & (larger != smaller)):
            raise BudgetPathViolation("action path is not nested")


class BudgetPathChecker:
    """Cached independent reconstruction of every declared BudgetPath."""

    def __init__(self, repo_root: Path, config: Mapping[str, Any]):
        self.repo_root = Path(repo_root)
        self.config = dict(config)
        self.tolerance = float(config["numerical_zero_tolerance"])
        self.attribution_root = (
            self.repo_root / config["source_mean_ranking_attribution"]
        )
        self.pareto_root = self.repo_root / config["source_pareto_transport"]
        self.pairwise_root = self.repo_root / config["source_pairwise_dominance"]
        _verify_manifest(
            self.attribution_root,
            config["source_attribution_manifest_sha256"],
            "R369",
        )
        _verify_manifest(
            self.pareto_root, config["source_pareto_manifest_sha256"], "R371"
        )
        _verify_manifest(
            self.pairwise_root,
            config["source_pairwise_manifest_sha256"],
            "R372",
        )
        self.objectives = {
            cohort: _read_npz(
                self.pareto_root / f"cohort_{cohort.lower()}_objectives.npz"
            )
            for cohort in ("A", "B")
        }
        self.actions = {
            cohort: {
                "policy_selection": _read_npz(
                    self.attribution_root / f"cohort_{cohort.lower()}_selection.npz"
                ),
                "adjudication": _read_npz(
                    self.attribution_root / f"cohort_{cohort.lower()}_adjudication.npz"
                ),
            }
            for cohort in ("A", "B")
        }

    def _validate_lineage(self, lineage: Mapping[str, Any]) -> None:
        _require_fields(lineage, LINEAGE_FIELDS, "lineage")
        if lineage["cohort"] not in ("A", "B"):
            raise BudgetPathViolation("invalid cohort")
        if lineage["role"] not in self.config["roles"]:
            raise BudgetPathViolation("invalid role")
        if lineage["proposal_regime"] not in self.config["proposal_regimes"]:
            raise BudgetPathViolation("invalid proposal regime")
        if lineage["arm"] not in self.config["arms"]:
            raise BudgetPathViolation("invalid arm")
        family = lineage["family"]
        replicate = lineage["control_replicate"]
        if family in self.config["main_families"]:
            if replicate is not None:
                raise BudgetPathViolation(
                    "main family cannot declare control replicate"
                )
        elif family == self.config["control_family"]:
            if (
                not isinstance(replicate, int)
                or not 0 <= replicate < self.config["control_replicates"]
            ):
                raise BudgetPathViolation("invalid control replicate")
        else:
            raise BudgetPathViolation("invalid family")
        expected_hashes = {
            "source_attribution_manifest_sha256": self.config[
                "source_attribution_manifest_sha256"
            ],
            "source_pareto_manifest_sha256": self.config[
                "source_pareto_manifest_sha256"
            ],
            "source_pairwise_manifest_sha256": self.config[
                "source_pairwise_manifest_sha256"
            ],
        }
        if any(lineage[key] != value for key, value in expected_hashes.items()):
            raise BudgetPathViolation("source manifest lineage mismatch")
        expected_objective = _objective_key(
            lineage["proposal_regime"],
            lineage["arm"],
            lineage["role"],
            family,
            replicate,
        )
        if lineage["objective_key"] != expected_objective:
            raise BudgetPathViolation("objective key mismatch")
        expected_actions = [
            _action_key(
                lineage["proposal_regime"],
                lineage["arm"],
                fraction,
                lineage["role"],
                family,
            )
            for fraction in BUDGET_FRACTIONS[1:]
        ]
        if lineage["action_keys"] != expected_actions:
            raise BudgetPathViolation("action keys mismatch")

    def _source_arrays(
        self, lineage: Mapping[str, Any]
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        cohort, role = lineage["cohort"], lineage["role"]
        family, replicate = lineage["family"], lineage["control_replicate"]
        objective_pack = self.objectives[cohort]
        values = objective_pack[lineage["objective_key"]]
        bootstrap = objective_pack[f"{role}|bootstrap"]
        action_pack = self.actions[cohort][role]
        raw_actions = [action_pack[key] for key in lineage["action_keys"]]
        if replicate is not None:
            raw_actions = [value[replicate] for value in raw_actions]
        identity = np.zeros_like(raw_actions[0])
        actions = [identity, *raw_actions]
        validate_nested_actions(actions)
        if (
            values.shape[:2] != (len(POLICY_IDS), 2)
            or len(actions[0]) != 2 * values.shape[2]
        ):
            raise BudgetPathViolation("objective/action shape mismatch")
        if bootstrap.shape != (
            self.config["bootstrap_replicates"],
            values.shape[2],
        ):
            raise BudgetPathViolation("bootstrap shape mismatch")
        if not np.all(np.isfinite(values)):
            raise BudgetPathViolation("nonfinite source objective")
        return actions, values, bootstrap

    def _front_frequencies(
        self, values: np.ndarray, bootstrap: np.ndarray
    ) -> np.ndarray:
        frequencies = np.zeros(len(POLICY_IDS), dtype=float)
        for indices in bootstrap:
            means = values[:, :, indices].mean(axis=2)
            frequencies += _pareto_mask(means, self.tolerance)
        return frequencies / len(bootstrap)

    def check(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        try:
            artifact = BudgetPathArtifact.from_dict(raw)
        except ValueError as exc:
            raise BudgetPathViolation(str(exc)) from exc
        self._validate_lineage(artifact.lineage)
        if artifact.artifact_id != artifact_id(artifact.lineage):
            raise BudgetPathViolation("artifact id mismatch")
        if artifact.axes != AXES:
            raise BudgetPathViolation("axis contract mismatch")
        if artifact.authority != AUTHORITY:
            raise BudgetPathViolation("authority separation mismatch")
        if artifact.utility_boundary != UTILITY_BOUNDARY:
            raise BudgetPathViolation("utility boundary mismatch")

        actions, values, bootstrap = self._source_arrays(artifact.lineage)
        means = values.mean(axis=2)
        front = _pareto_mask(means, self.tolerance)
        front_frequencies = self._front_frequencies(values, bootstrap)
        if len(artifact.policies) != len(POLICY_IDS):
            raise BudgetPathViolation("policy count mismatch")
        for index, policy in enumerate(artifact.policies):
            _require_fields(policy, POLICY_FIELDS, "policy")
            if policy["policy_id"] != POLICY_IDS[index]:
                raise BudgetPathViolation("policy order mismatch")
            if not _float_close(
                policy["budget_fraction"], BUDGET_FRACTIONS[index], self.tolerance
            ):
                raise BudgetPathViolation("budget fraction mismatch")
            if type(policy["action_count"]) is not int or policy["action_count"] != int(
                np.count_nonzero(actions[index])
            ):
                raise BudgetPathViolation("action count mismatch")
            if policy["action_sha256"] != array_digest(actions[index]):
                raise BudgetPathViolation("action hash mismatch")
            if (
                type(policy["objective_master_count"]) is not int
                or policy["objective_master_count"] != values.shape[2]
            ):
                raise BudgetPathViolation("objective master count mismatch")
            if policy["objective_sha256"] != array_digest(values[index]):
                raise BudgetPathViolation("objective hash mismatch")
            objective_mean = policy["objective_mean"]
            if set(objective_mean) != {"iid_delta", "grouped_delta"} or not all(
                _float_close(objective_mean[key], means[index, axis], self.tolerance)
                for axis, key in enumerate(("iid_delta", "grouped_delta"))
            ):
                raise BudgetPathViolation("objective mean mismatch")
            if type(policy["front_member"]) is not bool or policy[
                "front_member"
            ] != bool(front[index]):
                raise BudgetPathViolation("front membership mismatch")
            if not _float_close(
                policy["bootstrap_front_frequency"],
                front_frequencies[index],
                self.tolerance,
            ):
                raise BudgetPathViolation("front bootstrap frequency mismatch")

        expected_front = [POLICY_IDS[index] for index in np.flatnonzero(front)]
        if artifact.reader != {"kind": "PARETO_SET", "policy_ids": expected_front}:
            raise BudgetPathViolation("reader output mismatch")

        if len(artifact.pairs) != 21:
            raise BudgetPathViolation("pair count mismatch")
        for pair, (base, expansion) in zip(
            artifact.pairs, combinations(range(len(POLICY_IDS)), 2), strict=True
        ):
            _require_fields(pair, PAIR_FIELDS, "pair")
            pair_id = f"{POLICY_IDS[base]}->{POLICY_IDS[expansion]}"
            expected_identity = {
                "pair_id": pair_id,
                "base_policy": POLICY_IDS[base],
                "expansion_policy": POLICY_IDS[expansion],
                "adjacent": expansion == base + 1,
            }
            if type(pair["adjacent"]) is not bool:
                raise BudgetPathViolation("pair adjacency type mismatch")
            if any(pair[key] != value for key, value in expected_identity.items()):
                raise BudgetPathViolation("pair identity mismatch")
            increment = values[expansion] - values[base]
            increment_mean = increment.mean(axis=1)
            if pair["increment_sha256"] != array_digest(increment):
                raise BudgetPathViolation("pair increment hash mismatch")
            if set(pair["mean_increment"]) != {"iid", "grouped"} or not all(
                _float_close(
                    pair["mean_increment"][key], increment_mean[axis], self.tolerance
                )
                for axis, key in enumerate(("iid", "grouped"))
            ):
                raise BudgetPathViolation("pair increment mean mismatch")
            expected_status = _classify_one(increment_mean, self.tolerance)
            if pair["point_status"] != expected_status:
                raise BudgetPathViolation("pair status mismatch")
            bootstrap_means = increment[:, bootstrap].mean(axis=2).T
            state_codes = _classify_many(bootstrap_means, self.tolerance)
            counts = np.bincount(state_codes, minlength=len(PAIR_STATES))
            frequencies = pair["bootstrap_status_frequency"]
            if set(frequencies) != set(PAIR_STATES) or not all(
                _float_close(
                    frequencies[state], counts[index] / len(bootstrap), self.tolerance
                )
                for index, state in enumerate(PAIR_STATES)
            ):
                raise BudgetPathViolation("pair bootstrap frequency mismatch")

        return {
            "artifact_id": artifact.artifact_id,
            "checker_status": "VALID",
            "structural_claim_status": "CHECKED",
            "decision_status": "UNRESOLVED",
            "policy_count": len(artifact.policies),
            "pair_count": len(artifact.pairs),
            "front_size": int(front.sum()),
        }
