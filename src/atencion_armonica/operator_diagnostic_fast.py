"""Byte-equivalent pure diagnostic optimizations; no runtime or budget authority."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import struct

import numpy as np

from . import operator_objective_core as core
from . import operator_objective_scene as reference
from . import operator_objective_artifacts as artifacts

CELLS = reference.CELLS


class MeanCache:
    """One-scene exact sequence cache, including signed zero and missing values."""
    def __init__(self):
        self.records = {}

    def __call__(self, values):
        values = list(values)
        key = []
        for value in values:
            if value is None:
                key.append(None)
            else:
                if (not isinstance(value, (int, float, np.integer, np.floating))
                        or isinstance(value, (bool, np.bool_))):
                    raise ValueError("metric values must be numeric or explicitly undefined")
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError("nonfinite metric is not an explicit undefined value")
                key.append(struct.pack(">d", numeric))
        identity = tuple(key)
        if identity not in self.records:
            self.records[identity] = reference.mean_record(values)
        return dict(self.records[identity])


def bundle_bytes(value):
    """Same v1 wire bytes, with shared-object validation and native JSON traversal."""
    converted, done, active = {}, set(), set()
    fallback = False

    def validate(item):
        nonlocal fallback
        kind = type(item)
        if item is None or kind in (str, bool, int):
            return
        if kind is float:
            if not math.isfinite(item):
                raise ValueError("nonfinite bundle scalar")
            return
        # Custom container views may disagree with the JSON encoder's traversal.
        # Delegate the entire operation, without invoking those views twice.
        if isinstance(item, (dict, list, tuple)) and kind not in (dict, list, tuple):
            fallback = True
            return
        identity = id(item)
        if identity in active:
            raise ValueError("cyclic scientific bundle")
        if identity in done:
            return
        active.add(identity)
        if isinstance(item, (np.ndarray, np.generic)):
            packed = artifacts._pack(item)
            if isinstance(packed, np.generic):
                raise ValueError("unsupported NumPy scalar conversion")
            # Reserved array tags are produced only by the trusted v1 array port.
            if isinstance(item, np.generic):
                validate(packed)
            converted[identity] = packed
        elif isinstance(item, dict):
            if any(type(k) is not str for k in item) or "__ndarray__" in item:
                raise ValueError("bundle mappings require string keys without reserved tags")
            for child in item.values():
                validate(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                validate(child)
        else:
            raise ValueError("unsupported scientific bundle value")
        active.remove(identity)
        done.add(identity)

    validate(value)
    if fallback:
        return artifacts.bundle_bytes(value)
    def default(item):
        if id(item) not in converted:
            raise ValueError("unvalidated value reached JSON encoder")
        return converted[id(item)]
    decoded = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False, default=default)+"\n").encode()
    raw = gzip.compress(decoded, compresslevel=3, mtime=0)
    return raw, {"codec": "diagnostic-json-arrays-gzip3-v1", "bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest(), "decoded_bytes": len(decoded),
                 "decoded_sha256": hashlib.sha256(decoded).hexdigest()}


def _paired_scene_difference(left_cells, right_cells, *, means):
    """Pair strata within each cell, then cells within scene; complete-nine sensitivity.

    Inputs map all nine cell IDs to {stratum_id: scalar or None}. For a classic
    reference the caller supplies the same map under each matched cell ID.
    """
    if set(left_cells) != set(right_cells) or len(left_cells) != 9:
        raise ValueError("paired scene requires the same nine canonical cell IDs")
    cell_results, common = {}, None
    for cell in sorted(left_cells):
        left, right = left_cells[cell], right_cells[cell]
        if set(left) != set(right):
            raise ValueError("paired comparators must describe identical strata")
        valid = []
        for key in sorted(left):
            for value in (left[key], right[key]):
                if value is not None and (not isinstance(value, (int, float, np.integer, np.floating))
                                         or isinstance(value, (bool, np.bool_))
                                         or not math.isfinite(float(value))):
                    raise ValueError("paired metric must be finite or explicitly undefined")
            if left[key] is not None and right[key] is not None:
                valid.append(key)
        delta = [float(left[key])-float(right[key]) for key in valid]
        cell_results[cell] = {"strata": valid, "count": len(valid),
                              "mean": means(delta)["mean"] if delta else None}
        common = set(valid) if common is None else common & set(valid)
    values = [row["mean"] for row in cell_results.values() if row["mean"] is not None]
    complete = [means([float(left_cells[cell][key])-float(right_cells[cell][key])
                               for key in sorted(common)])["mean"] for cell in sorted(left_cells)] if common else []
    return {"cells": cell_results, "valid_cell_count": len(values),
            "mean": means(values)["mean"] if values else None,
            "complete_nine": {"strata": sorted(common), "count": len(common),
                              "mean": means(complete)["mean"] if complete else None}}


def _arm_summary(methods, *, means):
    exemplar = next(iter(methods.values()))
    return {scheme: {metric: means(methods[cell]["summary"][scheme][metric]["mean"] for cell in CELLS)
                     for metric in exemplar["summary"][scheme]}
            for scheme in exemplar["summary"]}


def _paired(left, right, *, means):
    exemplar = left[CELLS[0]]
    result = {}
    for scheme in exemplar["summary"]:
        common_metrics = sorted(set(exemplar["summary"][scheme]) & set(right[CELLS[0]]["summary"][scheme]))
        result[scheme] = {}
        for metric in common_metrics:
            sides = [{cell: {stratum: row["metrics"][metric]
                             for stratum, row in source[cell]["schemes"][scheme]["strata"].items()}
                      for cell in CELLS} for source in (left, right)]
            result[scheme][metric] = _paired_scene_difference(*sides, means=means)
    return result




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


def _method(score, targets, schemes, shared, errors=None, *, means):
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
        summaries[scheme] = {metric: means(row["metrics"][metric] for row in rows.values())
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
    means = MeanCache()
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
    classical = {name: _method(compact["scores"][name], targets, schemes, shared, means=means) for name in reference.CLASSICAL}
    learned, errors = {}, {}
    for arm in reference.ARMS:
        learned[arm], errors[arm] = {}, {}
        for cell in reference.CELLS:
            error = core.regression_errors(predictions[arm][cell], targets["u32"])
            errors[arm][cell] = {key: error[key] for key in
                                 ("s32", "component_bias", "component_squared", "sum_squared")}
            learned[arm][cell] = _method(error["s32"], targets, schemes, shared, error, means=means)
    paired = {}
    comparison = {cell: classical["extended_ub"] for cell in reference.CELLS}
    for arm in reference.ARMS:
        paired[f"{arm}-extended_ub"] = _paired(learned[arm], comparison, means=means)
    for other in ("local", "decoupled"):
        paired[f"generative-{other}"] = _paired(learned["generative"], learned[other], means=means)
    return {"schema": "operator-objective-scene-v1", "candidate_count": len(compact["partitions"]),
            "strata": schemes, "oracles": oracles, "classical": classical,
            "learned": learned, "arm_summary": {arm: _arm_summary(learned[arm], means=means) for arm in reference.ARMS},
            "paired": paired, "arrays": {"targets": targets, "errors": errors}}
