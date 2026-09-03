"""Classical selectors for the Wave 49 relational benchmark.

This module only consumes the public observation package. It deliberately does
not import generator or oracle code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .wave49_schema import (
    CATALOG_FAMILIES,
    SPLITS,
    ProtocolConfig,
    public_parameter_catalog,
    read_jsonl,
    write_jsonl,
)


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    use_eiv: bool
    allow_set: bool
    allow_abstention: bool
    force_prop: bool = False


SELECTORS = (
    SelectorSpec("single_prop_forced", use_eiv=False, allow_set=False, allow_abstention=False, force_prop=True),
    SelectorSpec("catalog_no_eiv", use_eiv=False, allow_set=True, allow_abstention=False),
    SelectorSpec("catalog_eiv", use_eiv=True, allow_set=True, allow_abstention=False),
    SelectorSpec("catalog_eiv_abstain", use_eiv=True, allow_set=True, allow_abstention=True),
)


def _selector_curve(family: str, x: np.ndarray, params: dict[str, float]) -> np.ndarray:
    if family == "PROP":
        return params["k"] * x
    if family == "AFFINE_OFFSET":
        return params["a"] + params["b"] * x
    if family == "POWER_NONUNIT":
        return params["a"] * np.power(np.maximum(x, 1e-9), params["p"])
    if family == "SATURATING":
        safe_x = np.maximum(x, 1e-9)
        return params["L"] * safe_x / (params["K"] + safe_x)
    raise ValueError(f"Unknown selector family: {family}")


def _score_no_eiv(
    family: str,
    candidates: list[dict[str, float]],
    x: np.ndarray,
    y: np.ndarray,
    covariance: np.ndarray,
) -> tuple[float, dict[str, float]]:
    variance_y = np.maximum(covariance[:, 1, 1], 1e-12)
    curves = np.stack([_selector_curve(family, x, params) for params in candidates])
    scores = np.sum((y[None, :] - curves) ** 2 / variance_y[None, :], axis=1)
    best = int(np.argmin(scores))
    return float(scores[best]), candidates[best]


def _score_eiv(
    family: str,
    candidates: list[dict[str, float]],
    x: np.ndarray,
    y: np.ndarray,
    covariance: np.ndarray,
    domain: tuple[float, float],
    latent_grid_size: int,
) -> tuple[float, dict[str, float]]:
    latent = np.linspace(domain[0], domain[1], latent_grid_size, dtype=np.float64)
    inv_cov = np.linalg.inv(covariance)
    curves = np.stack([_selector_curve(family, latent, params) for params in candidates])
    dx = x[:, None, None] - latent[None, None, :]
    dy = y[:, None, None] - curves[None, :, :]
    quadratic = (
        inv_cov[:, 0, 0, None, None] * dx * dx
        + 2.0 * inv_cov[:, 0, 1, None, None] * dx * dy
        + inv_cov[:, 1, 1, None, None] * dy * dy
    )
    scores = np.sum(np.min(quadratic, axis=2), axis=0)
    best = int(np.argmin(scores))
    return float(scores[best]), candidates[best]


def _family_properties(family: str, params: dict[str, float]) -> set[str]:
    props = {"smooth", "monotone_increasing"}
    if family == "PROP":
        props.update({"origin_passing", "affine", "homogeneous_degree_1"})
    elif family == "AFFINE_OFFSET":
        props.add("affine")
    elif family == "POWER_NONUNIT":
        props.update({"origin_passing", f"homogeneous_degree_{params['p']:g}"})
    elif family == "SATURATING":
        props.update({"origin_passing", "saturating"})
    return props


def _empirical_properties(x: np.ndarray, y: np.ndarray) -> list[str]:
    order = np.argsort(x)
    differences = np.diff(y[order])
    if len(differences) == 0:
        return []
    positive_fraction = float(np.mean(differences >= 0))
    negative_fraction = float(np.mean(differences <= 0))
    if positive_fraction >= 0.9:
        return ["monotone_increasing"]
    if negative_fraction >= 0.9:
        return ["monotone_decreasing"]
    return []


def predict_fixture(
    row: dict,
    spec: SelectorSpec,
    config: ProtocolConfig,
    ood_cutoffs: dict[tuple[int, str, str], float] | None = None,
) -> dict:
    x = np.asarray(row["x"], dtype=np.float64)
    y = np.asarray(row["y"], dtype=np.float64)
    covariance = np.asarray(row["covariance"], dtype=np.float64)
    semantics = row["coordinate_semantics"]
    inverse_jacobian = np.diag([
        semantics["x_scale_to_canonical"],
        semantics["y_scale_to_canonical"],
    ])
    observed = np.column_stack([x, y]) @ inverse_jacobian.T
    x, y = observed[:, 0], observed[:, 1]
    covariance = np.einsum("ab,nbc,dc->nad", inverse_jacobian, covariance, inverse_jacobian)
    domain = (
        row["domain"][0] * semantics["x_scale_to_canonical"],
        row["domain"][1] * semantics["x_scale_to_canonical"],
    )
    catalog = public_parameter_catalog()

    if spec.force_prop:
        score, params = _score_no_eiv("PROP", catalog["PROP"], x, y, covariance)
        family_scores = {"PROP": score}
        best_params = {"PROP": params}
        compatible = ["PROP"]
    else:
        family_scores: dict[str, float] = {}
        best_params: dict[str, dict[str, float]] = {}
        for family in CATALOG_FAMILIES:
            if spec.use_eiv:
                score, params = _score_eiv(
                    family,
                    catalog[family],
                    x,
                    y,
                    covariance,
                    domain,
                    config.latent_grid_size,
                )
            else:
                score, params = _score_no_eiv(family, catalog[family], x, y, covariance)
            family_scores[family] = score
            best_params[family] = params
        best_score = min(family_scores.values())
        compatible = sorted(
            family for family, score in family_scores.items()
            if score <= best_score + (config.family_score_delta if spec.allow_set else 0.0)
        )

    best_score = min(family_scores.values())
    covariance_knowledge = str(semantics["covariance_knowledge"])
    calibration_population = str(semantics["calibration_population"])
    cutoff_key = (len(x), covariance_knowledge, calibration_population)
    ood_cutoff = float("inf") if ood_cutoffs is None else float(ood_cutoffs[cutoff_key])
    if spec.allow_abstention and best_score > ood_cutoff:
        status = "ABSTAIN_OUT_OF_CATALOG"
        compatible = []
        properties = _empirical_properties(x, y)
        property_basis = "empirical_observation_independent_of_structural_abstention"
    else:
        status = "INDISTINGUISHABLE" if len(compatible) > 1 else "COMPATIBLE_SET"
        property_sets = [_family_properties(family, best_params[family]) for family in compatible]
        properties = sorted(set.intersection(*property_sets)) if property_sets else []
        property_basis = "intersection_of_selected_family_properties"
    return {
        "schema_version": config.schema_version,
        "fixture_id": row["fixture_id"],
        "split": row["split"],
        "selector": spec.name,
        "structural_compatible_set": compatible,
        "property_set": properties,
        "property_basis": property_basis,
        "status": status,
        "best_score_per_point": float(best_score / max(len(x), 1)),
        "ood_cutoff_per_point": float(ood_cutoff / max(len(x), 1)),
        "family_scores": family_scores,
        "best_params": best_params,
    }


def _conformal_cutoff(scores: list[float], alpha: float) -> tuple[float, dict[str, float | int]]:
    """Finite-sample split-conformal upper cutoff for a nonconformity score."""
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    if len(ordered) == 0:
        raise ValueError("abstention calibration requires at least one null score")
    rank = min(int(np.ceil((len(ordered) + 1) * (1.0 - alpha))), len(ordered))
    return float(ordered[rank - 1]), {
        "n": int(len(ordered)),
        "rank_1based": rank,
        "resolution": float(1.0 / (len(ordered) + 1)),
        "finite_sample_tail_bound": float((len(ordered) - rank + 1) / (len(ordered) + 1)),
    }


def execute_selectors(
    visible_dir: Path,
    predictions_dir: Path,
    config: ProtocolConfig,
) -> dict[str, int]:
    """Run selectors from a public directory without receiving the sealed root."""
    visible_dir = Path(visible_dir)
    predictions_dir = Path(predictions_dir)
    calibration_rows = read_jsonl(visible_dir / "calibration_null.jsonl")
    calibration_scores: dict[tuple[int, str, str], list[float]] = {}
    calibration_spec = next(spec for spec in SELECTORS if spec.name == "catalog_eiv")
    for row in calibration_rows:
        score = predict_fixture(row, calibration_spec, config)["best_score_per_point"] * row["n"]
        knowledge = str(row["coordinate_semantics"]["covariance_knowledge"])
        population = str(row["coordinate_semantics"]["calibration_population"])
        calibration_scores.setdefault((int(row["n"]), knowledge, population), []).append(float(score))
    cutoff_rows = {n: _conformal_cutoff(scores, config.ood_alpha) for n, scores in calibration_scores.items()}
    cutoffs = {n: row[0] for n, row in cutoff_rows.items()}
    from .wave49_schema import write_json
    write_json(predictions_dir / "abstention_calibration.json", {
        "method": "split-conformal-null-after-family-and-latent-selection",
        "alpha": config.ood_alpha,
        "target_populations": {
            "canonical_preserving": ["original", "positive_rescale"],
            "origin_translation_break": ["origin_translation_break"],
        },
        "sampling_unit": (
            "one independent base draw per (n,population,index); covariance-knowledge "
            "views are paired across strata"
        ),
        "sampling_law": config.calibration_sampling_law,
        "coverage_scope": config.calibration_coverage_scope,
        "coverage_guarantee": "marginal_not_conditional_by_nuisance",
        "target_population_law": {
            "canonical_preserving": (
                "uniform in-catalog family and uniform declared nuisance cell; original view "
                "stands for original and exactly canonicalized positive-rescale fixtures"
            ),
            "origin_translation_break": (
                "PROP source with sampled nonzero translation and uniform declared nuisance cell"
            ),
        },
        "coverage_statement": (
            "finite-sample marginal false-abstention control only for exchangeable in-catalog "
            "fixtures within each named stratum; OOD detection and factor-specific tables are "
            "empirical diagnostics, not conditional guarantees"
        ),
        "stratification": ["n", "covariance_knowledge", "calibration_population"],
        "scores_by_stratum": {
            f"{n}|{knowledge}|{population}": scores
            for (n, knowledge, population), scores in calibration_scores.items()
        },
        "cutoff_by_stratum": {
            f"{n}|{knowledge}|{population}": cutoff
            for (n, knowledge, population), cutoff in cutoffs.items()
        },
        "finite_sample": {
            f"{n}|{knowledge}|{population}": row[1]
            for (n, knowledge, population), row in cutoff_rows.items()
        },
        "source": "visible/calibration_null.jsonl",
    })
    counts: dict[str, int] = {}
    for split in SPLITS:
        rows = read_jsonl(visible_dir / f"{split}.jsonl")
        predictions = [
            predict_fixture(row, spec, config, cutoffs)
            for row in rows
            for spec in SELECTORS
        ]
        write_jsonl(predictions_dir / f"{split}.jsonl", predictions)
        counts[split] = len(predictions)
    return counts
