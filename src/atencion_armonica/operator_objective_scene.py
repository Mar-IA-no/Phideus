"""One-scene diagnostic and within-scene reductions; no file/resource access."""
from __future__ import annotations

import numpy as np

from . import operator_objective_core as core

ARMS = ("local", "generative", "decoupled")
CHECKPOINTS = (2026090721, 2026090722, 2026090723)
READERS = (2026090991, 2026090992, 2026090993)
CELLS = tuple(f"{cp}:{reader}" for cp in CHECKPOINTS for reader in READERS)
CLASSICAL = ("extended_ub", "base_ub", "extended_lb", "base_lb")
DECISION_METRICS = ("target64_regret", "target32_regret", "same_target64_choice", "target64_optimal",
                    "ari", "ari_gap", "k_absolute_error", "exact")
REGRESSION_METRICS = ("split_bias", "merge_bias", "split_mse", "merge_mse", "loss_like", "sum_mse")


def mean_record(values):
    """Defined values only, with explicit numerator support and total count."""
    values = list(values)
    if any(v is not None and (not isinstance(v, (int, float, np.integer, np.floating))
                              or isinstance(v, (bool, np.bool_))) for v in values):
        raise ValueError("metric values must be numeric or explicitly undefined")
    valid = [float(x) for x in values if x is not None]
    if not np.isfinite(valid).all():
        raise ValueError("nonfinite metric is not an explicit undefined value")
    return {"mean": float(np.mean(valid)) if valid else None,
            "defined": len(valid), "total": len(values)}


def _metric_values(description, errors, ids):
    decision = description["decision"]
    metrics = {"tau": description["tau"]["value"],
               **{k: float(decision[k]) if decision is not None else None for k in DECISION_METRICS}}
    if errors is not None:
        ids = np.asarray(ids, np.int64)
        bias = errors["component_bias"][ids]
        squared = errors["component_squared"][ids]
        values = ([*bias.mean(axis=0), *squared.mean(axis=0), squared.mean(axis=1).mean(),
                   errors["sum_squared"][ids].mean()] if len(ids) else [None]*6)
        metrics.update({k: float(v) if v is not None else None for k, v in zip(REGRESSION_METRICS, values)})
    return metrics


def _method(score, targets, schemes, errors=None):
    by_scheme, summaries = {}, {}
    names = ("tau", *DECISION_METRICS, *(REGRESSION_METRICS if errors is not None else ()))
    for scheme, strata in schemes.items():
        rows = {}
        for key, ids in strata.items():
            description = core.describe_score(score, targets, ids)
            rows[key] = {**description, "metrics": _metric_values(description, errors, ids)}
        summaries[scheme] = {metric: mean_record(row["metrics"][metric] for row in rows.values())
                             for metric in names}
        statuses = {}
        for row in rows.values():
            status = row["tau"]["status"]
            statuses[status] = statuses.get(status, 0)+1
        by_scheme[scheme] = {"strata": rows, "tau_status_counts": statuses}
    return {"schemes": by_scheme, "summary": summaries}


def _arm_summary(methods):
    exemplar = next(iter(methods.values()))
    return {scheme: {metric: mean_record(methods[cell]["summary"][scheme][metric]["mean"] for cell in CELLS)
                     for metric in exemplar["summary"][scheme]}
            for scheme in exemplar["summary"]}


def _paired(left, right):
    exemplar = left[CELLS[0]]
    result = {}
    for scheme in exemplar["summary"]:
        common_metrics = sorted(set(exemplar["summary"][scheme]) & set(right[CELLS[0]]["summary"][scheme]))
        result[scheme] = {}
        for metric in common_metrics:
            sides = [{cell: {stratum: row["metrics"][metric]
                             for stratum, row in source[cell]["schemes"][scheme]["strata"].items()}
                      for cell in CELLS} for source in (left, right)]
            result[scheme][metric] = core.paired_scene_difference(*sides)
    return result


def diagnose_scene(compact, canonical_labels, predictions):
    """Compute the fixed 4 classical/27 learned methods and matching contrasts.

    The caller must authenticate compact sources, canonical truth and output
    offsets before calling, and compare decisions to those already archived.
    It must also enforce resource budget and preserve returned arrays losslessly.
    """
    if (set(predictions) != set(ARMS)
            or any(set(predictions[arm]) != set(CELLS) for arm in ARMS)
            or set(compact["scores"]) != set(CLASSICAL)):
        raise ValueError("diagnostic requires the complete original 27-cell and four-score roster")
    targets = core.partition_targets(compact["partitions"], canonical_labels)
    if len(canonical_labels) != compact["n"]:
        raise ValueError("canonical labels differ from the compact event extent")
    schemes = core.observable_strata(compact["partitions"], compact["available"],
                                     compact["winning_branches"]["extended_ub"])
    oracles = {scheme: {key: core.oracle_report(targets["t64"], targets["t32"], targets["ari"], ids)
                       for key, ids in strata.items()} for scheme, strata in schemes.items()}
    classical = {name: _method(compact["scores"][name], targets, schemes) for name in CLASSICAL}
    learned, errors = {}, {}
    for arm in ARMS:
        learned[arm], errors[arm] = {}, {}
        for cell in CELLS:
            error = core.regression_errors(predictions[arm][cell], targets["u32"])
            errors[arm][cell] = {key: error[key] for key in
                                 ("s32", "component_bias", "component_squared", "sum_squared")}
            learned[arm][cell] = _method(error["s32"], targets, schemes, error)
    paired = {}
    reference = {cell: classical["extended_ub"] for cell in CELLS}
    for arm in ARMS:
        paired[f"{arm}-extended_ub"] = _paired(learned[arm], reference)
    for other in ("local", "decoupled"):
        paired[f"generative-{other}"] = _paired(learned["generative"], learned[other])
    return {"schema": "operator-objective-scene-v1", "candidate_count": len(compact["partitions"]),
            "strata": schemes, "oracles": oracles, "classical": classical,
            "learned": learned, "arm_summary": {arm: _arm_summary(learned[arm]) for arm in ARMS},
            "paired": paired, "arrays": {"targets": targets, "errors": errors}}
