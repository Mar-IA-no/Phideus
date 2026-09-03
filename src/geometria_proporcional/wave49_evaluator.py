"""Post-freeze evaluator for the Wave 49 relational benchmark."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .wave49_schema import SPLITS, ProtocolConfig, read_jsonl, write_json, write_jsonl


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _score_row(prediction: dict, oracle: dict) -> dict[str, Any]:
    predicted = set(prediction["structural_compatible_set"])
    authorized = set(oracle["oracle_compatible_set"])
    true_family = oracle["family_id"]
    truth_covered = true_family in predicted if not oracle["is_out_of_catalog"] else None
    singleton_correct = predicted == {true_family} if not oracle["is_out_of_catalog"] else None
    incompatible = bool(predicted - authorized) if authorized else bool(predicted)
    abstain_expected = oracle["oracle_status"] == "ABSTAIN_OUT_OF_CATALOG"
    abstain_correct = prediction["status"] == "ABSTAIN_OUT_OF_CATALOG" if abstain_expected else None
    true_properties = set(oracle["property_set"])
    predicted_properties = set(prediction["property_set"])
    return {
        "fixture_id": prediction["fixture_id"],
        "split": prediction["split"],
        "selector": prediction["selector"],
        "n": int(oracle["n"]),
        "family_id": true_family,
        "oracle_status": oracle["oracle_status"],
        "predicted_status": prediction["status"],
        "separation_band": oracle["separation_band"],
        "separation_index": oracle["separation_index"],
        "range_mode": oracle["range_mode"],
        "noise_mode": oracle["noise_mode"],
        "covariance_mode": oracle["covariance_mode"],
        "covariance_knowledge": oracle["covariance_knowledge"],
        "rho": oracle["rho"],
        "rival_distance_mode": oracle["rival_distance_mode"],
        "design_stratum": oracle["design_stratum"],
        "target_region": oracle["target_region"],
        "observational_region": oracle["observational_region"],
        "target_region_match": oracle["target_region_match"],
        "pair_token": oracle["pair_token"],
        "representation": oracle["representation"],
        "calibration_population": oracle["calibration_population"],
        "is_out_of_catalog": oracle["is_out_of_catalog"],
        "truth_covered": truth_covered,
        "singleton_correct": singleton_correct,
        "selection_incompatible": incompatible,
        "abstain_correct": abstain_correct,
        "status_match": prediction["status"] == oracle["oracle_status"],
        "predicted_set_width": len(predicted),
        "property_tp": len(predicted_properties & true_properties),
        "property_fp": len(predicted_properties - true_properties),
        "property_fn": len(true_properties - predicted_properties),
        "best_score_per_point": prediction["best_score_per_point"],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    in_catalog = [row for row in rows if not row["is_out_of_catalog"]]
    out_of_catalog = [row for row in rows if row["is_out_of_catalog"]]
    answered = [row for row in in_catalog if row["predicted_status"] != "ABSTAIN_OUT_OF_CATALOG"]
    abstain_cases = [row for row in rows if row["abstain_correct"] is not None]
    abstained = [row for row in rows if row["predicted_status"] == "ABSTAIN_OUT_OF_CATALOG"]
    false_abstentions = [row for row in in_catalog if row["predicted_status"] == "ABSTAIN_OUT_OF_CATALOG"]
    correct_ood_abstentions = [row for row in abstain_cases if row["abstain_correct"]]
    tp = sum(row["property_tp"] for row in rows)
    fp = sum(row["property_fp"] for row in rows)
    fn = sum(row["property_fn"] for row in rows)
    return {
        "n": len(rows),
        "n_in_catalog": len(in_catalog),
        "n_out_of_catalog": len(out_of_catalog),
        "n_abstained": len(abstained),
        "n_false_abstentions": len(false_abstentions),
        "n_ood_expected_abstentions": len(abstain_cases),
        "n_correct_ood_abstentions": len(correct_ood_abstentions),
        "truth_coverage": _safe_div(sum(bool(row["truth_covered"]) for row in in_catalog), len(in_catalog)),
        "structural_answer_coverage": _safe_div(len(answered), len(in_catalog)),
        "selective_truth_coverage": _safe_div(sum(bool(row["truth_covered"]) for row in answered), len(answered)),
        "singleton_correct_rate": _safe_div(sum(bool(row["singleton_correct"]) for row in in_catalog), len(in_catalog)),
        "incompatible_selection_rate": _safe_div(sum(row["selection_incompatible"] for row in rows), len(rows)),
        "correct_ood_abstention_rate": _safe_div(len(correct_ood_abstentions), len(abstain_cases)),
        "status_match_rate": _safe_div(sum(row["status_match"] for row in rows), len(rows)),
        "target_region_match_rate": _safe_div(sum(row["target_region_match"] for row in rows), len(rows)),
        "abstention_rate": _safe_div(
            len(abstained), len(rows)
        ),
        "in_catalog_false_abstention_rate": _safe_div(
            len(false_abstentions), len(in_catalog),
        ),
        "mean_set_width": float(np.mean([row["predicted_set_width"] for row in rows])) if rows else float("nan"),
        "property_precision": _safe_div(tp, tp + fp),
        "property_recall": _safe_div(tp, tp + fn),
    }


def _invariance(rows: list[dict[str, Any]], predictions: dict[tuple[str, str], dict]) -> dict[str, float | int]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["selector"], row["pair_token"], row["covariance_knowledge"])].append(row)
    comparable = 0
    preserved = 0
    for (selector, _, _), group in groups.items():
        if len(group) < 2:
            continue
        original = next((row for row in group if row["representation"] == "original"), None)
        scaled = next((row for row in group if row["representation"] == "positive_rescale"), None)
        if not original or not scaled:
            continue
        pa = predictions[(original["fixture_id"], selector)]
        pb = predictions[(scaled["fixture_id"], selector)]
        comparable += 1
        if pa["status"] == pb["status"] and pa["structural_compatible_set"] == pb["structural_compatible_set"]:
            preserved += 1
    return {"n_pairs": comparable, "preservation_rate": _safe_div(preserved, comparable)}


def _translation_rupture(rows: list[dict[str, Any]], predictions: dict[tuple[str, str], dict]) -> dict[str, float | int]:
    translated = [row for row in rows if row["representation"] == "origin_translation_break"]
    correct = 0
    for row in translated:
        prediction = predictions[(row["fixture_id"], row["selector"])]
        selected = set(prediction["structural_compatible_set"])
        correct += "AFFINE_OFFSET" in selected and "PROP" not in selected
    return {"n_cases": len(translated), "strict_rupture_rate": _safe_div(correct, len(translated))}


def evaluate_benchmark(output_dir: Path, config: ProtocolConfig) -> dict[str, Any]:
    """Open sealed truth only after prediction artifacts have been frozen."""
    output_dir = Path(output_dir)
    all_scores: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": config.schema_version,
        "calibration_contract": {
            "sampling_law": config.calibration_sampling_law,
            "coverage_scope": config.calibration_coverage_scope,
            "guarantee": "marginal_not_conditional_by_nuisance",
        },
        "splits": {},
    }
    for split in SPLITS:
        oracle = {
            row["fixture_id"]: row
            for row in read_jsonl(output_dir / "sealed" / "oracle" / f"{split}.jsonl")
        }
        predictions = read_jsonl(output_dir / "predictions" / f"{split}.jsonl")
        prediction_index = {(row["fixture_id"], row["selector"]): row for row in predictions}
        scored = [_score_row(row, oracle[row["fixture_id"]]) for row in predictions]
        all_scores.extend(scored)
        by_selector: dict[str, Any] = {}
        for selector in sorted({row["selector"] for row in scored}):
            selected = [row for row in scored if row["selector"] == selector]
            by_band = {
                band: _aggregate([row for row in selected if row["separation_band"] == band])
                for band in sorted({row["separation_band"] for row in selected})
            }
            by_design_stratum = {
                region: _aggregate([row for row in selected if row["design_stratum"] == region])
                for region in sorted({row["design_stratum"] for row in selected})
            }
            by_observational_region = {
                region: _aggregate([row for row in selected if row["observational_region"] == region])
                for region in sorted({row["observational_region"] for row in selected})
            }
            by_target_region = {
                region: _aggregate([row for row in selected if row["target_region"] == region])
                for region in sorted({row["target_region"] for row in selected})
            }
            by_covariance_knowledge = {
                mode: _aggregate([row for row in selected if row["covariance_knowledge"] == mode])
                for mode in sorted({row["covariance_knowledge"] for row in selected})
            }
            by_calibration_stratum = {
                f"{n}|{knowledge}|{population}": _aggregate([
                    row for row in selected
                    if row["n"] == n
                    and row["covariance_knowledge"] == knowledge
                    and row["calibration_population"] == population
                ])
                for n, knowledge, population in sorted({
                    (row["n"], row["covariance_knowledge"], row["calibration_population"])
                    for row in selected
                })
            }
            factor_fields = (
                "n", "calibration_population", "representation", "range_mode", "noise_mode",
                "covariance_mode", "rho", "rival_distance_mode",
            )
            by_factor = {
                field: {
                    str(value): _aggregate([row for row in selected if row[field] == value])
                    for value in sorted({row[field] for row in selected}, key=str)
                }
                for field in factor_fields
            }
            by_selector[selector] = {
                "overall": _aggregate(selected),
                "by_separation_band": by_band,
                "by_design_stratum": by_design_stratum,
                "by_observational_region": by_observational_region,
                "by_target_region": by_target_region,
                "by_covariance_knowledge": by_covariance_knowledge,
                "by_calibration_stratum": by_calibration_stratum,
                "by_factor": by_factor,
                "positive_rescale_invariance": _invariance(selected, prediction_index),
                "origin_translation_rupture": _translation_rupture(selected, prediction_index),
            }
        summary["splits"][split] = by_selector

    write_jsonl(output_dir / "evaluation" / "fixture_scores.jsonl", all_scores)
    write_json(output_dir / "evaluation" / "summary.json", summary)
    _write_report(output_dir / "evaluation" / "REPORT_WAVE49_CLASSICAL.md", summary)
    return summary


def _fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if value != value:
        return "NA"
    return f"{value:.3f}"


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Wave 49 classical benchmark report",
        "",
        "> Synthetic protocol audit. These observations do not constitute architectural promotion or GO/NO-GO.",
        "> The conformal claim is marginal false-abstention control for exchangeable in-catalog fixtures within each exact `(n, covariance knowledge, population)` stratum. OOD detection and factor tables are empirical diagnostics, not conditional guarantees.",
        "",
    ]
    for split, selectors in summary["splits"].items():
        lines.extend([
            f"## {split}",
            "",
            "| selector | coverage | selective coverage | singleton | incompatible | OOD abstain | false abstain | status | target match | width | scale invariance | translation rupture |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for selector, result in selectors.items():
            overall = result["overall"]
            invariance = result["positive_rescale_invariance"]
            rupture = result["origin_translation_rupture"]
            lines.append(
                "| " + " | ".join([
                    selector,
                    _fmt(overall["truth_coverage"]),
                    _fmt(overall["selective_truth_coverage"]),
                    _fmt(overall["singleton_correct_rate"]),
                    _fmt(overall["incompatible_selection_rate"]),
                    _fmt(overall["correct_ood_abstention_rate"]),
                    _fmt(overall["in_catalog_false_abstention_rate"]),
                    _fmt(overall["status_match_rate"]),
                    _fmt(overall["target_region_match_rate"]),
                    _fmt(overall["mean_set_width"]),
                    _fmt(invariance["preservation_rate"]),
                    _fmt(rupture["strict_rupture_rate"]),
                ]) + " |"
            )
        lines.append("")
        abstention = selectors.get("catalog_eiv_abstain")
        if abstention is not None:
            lines.extend([
                "### Empirical abstention by calibration stratum",
                "",
                "| n | covariance | population | N | abstained | false abstain (in-catalog) | rate | correct OOD abstain | rate |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|",
            ])
            for key, metrics in abstention["by_calibration_stratum"].items():
                n, knowledge, population = key.split("|", 2)
                false_fraction = f"{metrics['n_false_abstentions']}/{metrics['n_in_catalog']}"
                ood_fraction = f"{metrics['n_correct_ood_abstentions']}/{metrics['n_ood_expected_abstentions']}"
                lines.append(
                    "| " + " | ".join([
                        n,
                        knowledge,
                        population,
                        str(metrics["n"]),
                        str(metrics["n_abstained"]),
                        false_fraction,
                        _fmt(metrics["in_catalog_false_abstention_rate"]),
                        ood_fraction,
                        _fmt(metrics["correct_ood_abstention_rate"]),
                    ]) + " |"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
