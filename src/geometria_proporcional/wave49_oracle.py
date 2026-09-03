"""Independent observational-separation oracle for Wave 49."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .wave49_schema import (
    CATALOG_FAMILIES,
    SPLITS,
    ProtocolConfig,
    public_parameter_catalog,
    read_jsonl,
    write_jsonl,
)
from .wave49_logic_oracle import logical_properties, logical_structural_set
from .wave49_oracle_reference import reference_family_distances


def _oracle_curve(family: str, x: np.ndarray, params: dict[str, float]) -> np.ndarray:
    if family == "PROP":
        return params["k"] * x
    if family == "AFFINE_OFFSET":
        return params["a"] + params["b"] * x
    if family == "POWER_NONUNIT":
        return params["a"] * np.power(x, params["p"])
    if family == "SATURATING":
        return params["L"] * x / (params["K"] + x)
    raise ValueError(f"Unknown oracle family: {family}")


def _family_distance(
    family: str,
    latent_x: np.ndarray,
    clean_y: np.ndarray,
    covariance: np.ndarray,
    candidates: list[dict[str, float]],
) -> float:
    inv_cov = np.linalg.inv(covariance)
    weight_y = inv_cov[:, 1, 1]
    best = np.inf
    for params in candidates:
        rival_y = _oracle_curve(family, latent_x, params)
        delta_y = clean_y - rival_y
        distance = 0.5 * float(np.sum(weight_y * delta_y * delta_y))
        best = min(best, distance)
    return float(best)


def separation_band(value: float, cuts: tuple[float, ...]) -> str:
    if value <= cuts[0]:
        return "S0"
    if value < cuts[1]:
        return "S1"
    if value < cuts[2]:
        return "S2"
    return "S3"


def _orders_agree_up_to_ties(
    primary: dict[str, float],
    reference: dict[str, float],
    atol: float,
) -> tuple[bool, list[list[str]]]:
    families = sorted(primary)
    ties: list[list[str]] = []
    for index, left in enumerate(families):
        for right in families[index + 1:]:
            delta_primary = primary[left] - primary[right]
            delta_reference = reference[left] - reference[right]
            if abs(delta_primary) <= atol or abs(delta_reference) <= atol:
                ties.append([left, right])
                continue
            if np.sign(delta_primary) != np.sign(delta_reference):
                return False, ties
    return True, ties


def _target_region_matches(target_region: str, observational_region: str) -> bool:
    if target_region == "OUT_OF_CATALOG":
        return observational_region.startswith("OUT_OF_CATALOG_")
    if target_region == "DELIBERATELY_INDISTINGUISHABLE":
        return observational_region == "OBSERVATIONALLY_INDISTINGUISHABLE"
    if target_region == "IDENTIFIABLE":
        return observational_region == "OBSERVATIONALLY_IDENTIFIABLE"
    raise ValueError(f"Unknown target region: {target_region}")


def oracle_rows_from_truth(
    truth_rows: list[dict[str, object]],
    config: ProtocolConfig,
) -> list[dict[str, object]]:
    """Compute oracle rows for an explicitly authorized truth-row collection."""
    catalog = public_parameter_catalog()
    oracle_rows = []
    for truth in truth_rows:
        split = str(truth["split"])
        fixture_id = truth["fixture_id"]
        latent_x = np.asarray(truth["latent_x"], dtype=np.float64)
        clean_y = np.asarray(truth["clean_y"], dtype=np.float64)
        covariance = np.asarray(truth["true_covariance_canonical"], dtype=np.float64)
        distances = {
            family: _family_distance(family, latent_x, clean_y, covariance, catalog[family])
            for family in CATALOG_FAMILIES
        }
        reference_distances = reference_family_distances(
            latent_x, clean_y, covariance, catalog
        )
        order_match, numerical_ties = _orders_agree_up_to_ties(
            distances, reference_distances, config.oracle_numeric_atol
        )
        if not order_match:
            raise RuntimeError(f"distance-order mismatch for {fixture_id}")
        max_distance_delta = max(
            abs(distances[family] - reference_distances[family])
            for family in CATALOG_FAMILIES
        )
        true_family = truth["family_id"]
        if truth["is_out_of_catalog"]:
            separation = min(distances.values())
            reference_separation = min(reference_distances.values())
            compatible = [
                family for family, distance in distances.items()
                if distance <= config.oracle_compatibility_distance
            ]
            status = (
                "ABSTAIN_OUT_OF_CATALOG"
                if separation >= config.oracle_ood_distance
                else "INDISTINGUISHABLE"
            )
        else:
            rival_distances = {
                family: distance for family, distance in distances.items()
                if family != true_family
            }
            reference_rival_distances = {
                family: distance for family, distance in reference_distances.items()
                if family != true_family
            }
            separation = min(rival_distances.values())
            reference_separation = min(reference_rival_distances.values())
            compatible = [true_family]
            compatible.extend(
                family for family, distance in rival_distances.items()
                if distance <= config.oracle_compatibility_distance
            )
            compatible = sorted(set(compatible))
            status = "INDISTINGUISHABLE" if len(compatible) > 1 else "COMPATIBLE_SET"
        if truth["is_out_of_catalog"]:
            observational_region = (
                "OUT_OF_CATALOG_SEPARATED"
                if status == "ABSTAIN_OUT_OF_CATALOG"
                else "OUT_OF_CATALOG_OVERLAP"
            )
        else:
            observational_region = (
                "OBSERVATIONALLY_INDISTINGUISHABLE"
                if status == "INDISTINGUISHABLE"
                else "OBSERVATIONALLY_IDENTIFIABLE"
            )
        oracle_rows.append(
            {
                "schema_version": config.schema_version,
                "fixture_id": fixture_id,
                "split": split,
                "n": len(latent_x),
                "family_id": true_family,
                "logical_structural_set": logical_structural_set(true_family),
                "oracle_compatible_set": compatible,
                "oracle_status": status,
                "property_set": logical_properties(
                    true_family,
                    truth["generator_params"],
                    (float(latent_x.min()), float(latent_x.max())),
                ),
                "separation_index": float(separation),
                "separation_band": separation_band(float(separation), config.separation_bands),
                "family_distances": distances,
                "reference_family_distances": reference_distances,
                "distance_order_match": order_match,
                "numerical_ties": numerical_ties,
                "numeric_atol": config.oracle_numeric_atol,
                "max_distance_delta": float(max_distance_delta),
                "reference_dtype": str(np.dtype(np.longdouble)),
                "oracle_input_scope": "sealed_truth+public_parameter_catalog",
                "selector_output_dependency": False,
                "is_out_of_catalog": truth["is_out_of_catalog"],
                "target_region": truth["target_region"],
                "target_region_basis": truth["target_region_basis"],
                "design_separation_index": truth["design_separation_index"],
                "target_design_distance_delta": float(
                    abs(float(separation) - float(truth["design_separation_index"]))
                ),
                "target_design_reference_delta": float(
                    abs(float(reference_separation) - float(truth["design_separation_index"]))
                ),
                "observational_region": observational_region,
                "target_region_match": _target_region_matches(
                    truth["target_region"], observational_region
                ) and abs(
                    float(separation) - float(truth["design_separation_index"])
                ) <= config.oracle_numeric_atol and abs(
                    float(reference_separation) - float(truth["design_separation_index"])
                ) <= config.oracle_numeric_atol,
                "pair_token": truth["pair_token"],
                "representation": truth["representation"],
                "range_mode": truth["range_mode"],
                "noise_mode": truth["noise_mode"],
                "covariance_mode": truth["covariance_mode"],
                "covariance_knowledge": truth["covariance_knowledge"],
                "rho": truth["rho"],
                "rival_distance_mode": truth["rival_distance_mode"],
                "design_stratum": truth["design_stratum"],
                "calibration_population": truth["calibration_population"],
            }
        )
    return oracle_rows


def compute_oracle_splits(
    output_dir: Path,
    config: ProtocolConfig,
    splits: tuple[str, ...],
    destination_dir: Path | None = None,
) -> dict[str, int]:
    """Materialize only the explicitly named splits."""
    invalid = set(splits) - set(SPLITS)
    if invalid:
        raise ValueError(f"unknown oracle splits: {sorted(invalid)}")
    output_dir = Path(output_dir)
    destination = Path(destination_dir) if destination_dir else output_dir / "sealed" / "oracle"
    counts: dict[str, int] = {}
    for split in splits:
        truth_rows = read_jsonl(output_dir / "sealed" / f"{split}.jsonl")
        rows = oracle_rows_from_truth(truth_rows, config)
        write_jsonl(destination / f"{split}.jsonl", rows)
        counts[split] = len(rows)
    return counts


def compute_oracle(output_dir: Path, config: ProtocolConfig) -> dict[str, int]:
    """Compute oracle rows from sealed truth and visible covariance.

    The selector never imports this module. Distances use the sealed latent
    design and generator mean; they are unavailable during prediction.
    """
    return compute_oracle_splits(output_dir, config, SPLITS)
