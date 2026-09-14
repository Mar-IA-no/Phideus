"""Byte-equivalent cache prototype; no file, budget, runtime or GPU authority.

The frozen operator_objective_* modules remain the reference implementation.
Caches live for one scene only and key on complete canonical candidate tuples,
never on a target value, metric, regime or selected outcome.
"""
from __future__ import annotations

import numpy as np

from . import operator_objective_core as core
from . import operator_objective_scene as reference


def _description(score, targets, ids, oracle):
    choice = core.order_record(score, ids)
    tau = core.kendall_tau_b(score[ids], targets["t64"][ids])
    out = {"candidate_ids": ids.tolist(), "choice": choice, "tau": tau}
    if not len(ids):
        return {**out, "decision": None}
    i = choice["chosen"]
    t, u, a = oracle["target64"], oracle["target32"], oracle["ari"]
    return {**out, "decision": {
        "target64_regret": float(targets["t64"][i]-t["minimum"]),
        "target32_regret": float(np.float64(targets["t32"][i])-u["minimum"]),
        "same_target64_choice": i == t["chosen"], "target64_optimal": i in t["optima"],
        "ari": float(targets["ari"][i]),
        "ari_gap": float(targets["ari"][a["chosen"]]-targets["ari"][i]),
        "k_absolute_error": int(abs(targets["k_error"][i])), "exact": bool(targets["exact"][i])}}


def _method(score, targets, schemes, shared, errors=None):
    score = core._vector(score, size=len(targets["t64"]))
    descriptions, by_scheme, summaries = {}, {}, {}
    names = ("tau", *reference.DECISION_METRICS, *(reference.REGRESSION_METRICS if errors is not None else ()))
    for scheme, strata in schemes.items():
        rows = {}
        for key, indices in strata.items():
            identity = tuple(indices)
            if identity not in descriptions:
                ids, oracle = shared[identity]
                description = _description(score, targets, ids, oracle)
                descriptions[identity] = {**description,
                                         "metrics": reference._metric_values(description, errors, ids)}
            rows[key] = descriptions[identity]
        summaries[scheme] = {metric: reference.mean_record(row["metrics"][metric] for row in rows.values())
                             for metric in names}
        statuses = {}
        for row in rows.values():
            status = row["tau"]["status"]
            statuses[status] = statuses.get(status, 0)+1
        by_scheme[scheme] = {"strata": rows, "tau_status_counts": statuses}
    return {"schemes": by_scheme, "summary": summaries}


def diagnose_scene(compact, canonical_labels, predictions):
    """Same output schema/arithmetic as v1, reusing only identical subproblems."""
    if (set(predictions) != set(reference.ARMS)
            or any(set(predictions[arm]) != set(reference.CELLS) for arm in reference.ARMS)
            or set(compact["scores"]) != set(reference.CLASSICAL)):
        raise ValueError("diagnostic requires the complete original 27-cell and four-score roster")
    targets = core.partition_targets(compact["partitions"], canonical_labels)
    if len(canonical_labels) != compact["n"]:
        raise ValueError("canonical labels differ from the compact event extent")
    schemes = core.observable_strata(compact["partitions"], compact["available"],
                                     compact["winning_branches"]["extended_ub"])
    shared, oracles = {}, {}
    for scheme, strata in schemes.items():
        oracles[scheme] = {}
        for key, indices in strata.items():
            identity = tuple(indices)
            if identity not in shared:
                ids = core._indices(indices, len(targets["t64"]))
                shared[identity] = (ids, core.oracle_report(targets["t64"], targets["t32"], targets["ari"], ids))
            oracles[scheme][key] = shared[identity][1]
    classical = {name: _method(compact["scores"][name], targets, schemes, shared) for name in reference.CLASSICAL}
    learned, errors = {}, {}
    for arm in reference.ARMS:
        learned[arm], errors[arm] = {}, {}
        for cell in reference.CELLS:
            error = core.regression_errors(predictions[arm][cell], targets["u32"])
            errors[arm][cell] = {key: error[key] for key in
                                 ("s32", "component_bias", "component_squared", "sum_squared")}
            learned[arm][cell] = _method(error["s32"], targets, schemes, shared, error)
    paired = {}
    comparison = {cell: classical["extended_ub"] for cell in reference.CELLS}
    for arm in reference.ARMS:
        paired[f"{arm}-extended_ub"] = reference._paired(learned[arm], comparison)
    for other in ("local", "decoupled"):
        paired[f"generative-{other}"] = reference._paired(learned["generative"], learned[other])
    return {"schema": "operator-objective-scene-v1", "candidate_count": len(compact["partitions"]),
            "strata": schemes, "oracles": oracles, "classical": classical,
            "learned": learned, "arm_summary": {arm: reference._arm_summary(learned[arm]) for arm in reference.ARMS},
            "paired": paired, "arrays": {"targets": targets, "errors": errors}}
