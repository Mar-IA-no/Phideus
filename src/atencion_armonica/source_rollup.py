"""Fixed scene-first descriptive estimands; no inferential intervals."""
from collections import Counter

import numpy as np

from .source_artifacts import ARMS, SEEDS, reader_key

GROUP_METRICS = ("fine_rms_cents", "coarse_rms_cents", "coarse_minus_fine_cents",
                 "internal_endpoint_median_cents", "internal_endpoint_max_cents")


def mean_coverage(values):
    present = [float(v) for v in values if v is not None]
    if any(not np.isfinite(v) for v in present):
        raise ValueError("nonfinite metric")
    return {"mean": float(np.mean(present)) if present else None,
            "eligible": len(present), "total": len(values)}


def scene_group_strata(groups):
    strata = {}
    for group in groups:
        key = (group["size"], group["category"])
        strata.setdefault(key, []).append(group)
    result = []
    for (size, category), rows in sorted(strata.items()):
        metrics = {}
        for name in GROUP_METRICS:
            values = []
            for group in rows:
                fit = group["fit"]
                if name.startswith("internal_endpoint_"):
                    values.append(group.get(name))
                    continue
                if fit["status"] != "GRID_WITNESS_APPROXIMATE":
                    values.append(None)
                    continue
                value = (fit["fine"]["minimum_cents"] if name == "fine_rms_cents" else
                         fit["coarse"]["minimum_cents"] if name == "coarse_rms_cents" else
                         fit["coarse_minus_fine_cents"] if name == "coarse_minus_fine_cents" else group[name])
                values.append(value)
            # Denominator includes non-evaluable groups, not just successful fits.
            metrics[name] = mean_coverage(values)
        result.append({"size": size, "category": category, "groups": len(rows),
                       "members": sum(g["size"] for g in rows),
                       "statuses": dict(Counter(g["fit"]["status"] for g in rows)), "metrics": metrics})
    return result


def pressure_metrics(summary):
    metrics = {}
    for category, row in summary["classes"].items():
        for name in ("edge_count", "B", "P", "S", "P_over_B", "S_over_B"):
            enabled = (row["edge_count"] > 0 or name == "edge_count")
            if name in ("S", "S_over_B"):
                enabled = enabled and summary["sham_evaluable"]
            metrics[f"class_{category}.{name}"] = row[name] if enabled else None
    for penalty, row in summary["penalties"].items():
        for name in ("L", "F_true", "L_truth", "mean_weight_true", "mean_weight_mixed",
                     "true_triple_count", "mixed_triple_count"):
            metrics[f"{penalty}.{name}"] = row[name] if row["status"] == "EVALUABLE" else None
    return metrics


def aggregate_split(scene_rows, readers, *, scene_count=32):
    """Group means → reader/scene → complete three-seed scene mean → split."""
    names = [reader_key(r) for r in readers]+["privileged_truth_groups"]
    indexed = {(row["scene_id"], row["reader"]): row for row in scene_rows}
    if len(indexed) != len(scene_rows) or set(indexed) != {(i, r) for i in range(scene_count) for r in names}:
        raise ValueError("roll-up requires full unique fixed scene/reader roster")
    strata = sorted({(s["size"], s["category"]) for row in scene_rows for s in row["strata"]})
    cells = {(i, r): {(s["size"], s["category"]): s for s in indexed[i, r]["strata"]}
             for i in range(scene_count) for r in names}
    result = {"readers": {}, "arms": {}, "scene_count": scene_count,
              "authority": "DESCRIPTIVE_FIXED_OPEN_SAMPLE_NO_INFERENTIAL_INTERVALS"}
    def group_value(i, r, stratum, metric):
        cell = cells[i, r].get(stratum)
        return None if cell is None else cell["metrics"][metric]["mean"]
    def metric_rollup(member_readers, value, metric_names):
        out = {}
        for metric in metric_names:
            scene_values, eligible_seed_counts = [], []
            for i in range(scene_count):
                vals = [value(i, r, metric) for r in member_readers]
                eligible_seed_counts.append(sum(v is not None for v in vals))
                scene_values.append(float(np.mean(vals)) if all(v is not None for v in vals) else None)
            out[metric] = mean_coverage(scene_values)|{"eligible_readers_by_scene": eligible_seed_counts,
                                                     "required_readers_per_scene": len(member_readers)}
        return out
    for name in names+list(ARMS):
        members = ([f"{name}__seed_{seed}" for seed in SEEDS] if name in ARMS else [name])
        target = {"group_strata": [], "pressure": {}}
        for stratum in strata:
            values = [cells[i, r].get(stratum) for i in range(scene_count) for r in members]
            counts = Counter()
            for cell in values:
                if cell:
                    counts.update(cell["statuses"])
            metrics = metric_rollup(members, lambda i, r, m: group_value(i, r, stratum, m), GROUP_METRICS)
            target["group_strata"].append({"size": stratum[0], "category": stratum[1], "metrics": metrics,
                "counts_over_reader_scene_cells": {"groups": sum(c["groups"] for c in values if c),
                    "members": sum(c["members"] for c in values if c), "statuses": dict(counts),
                    "cells_with_groups": sum(c is not None for c in values), "total_cells": len(values)}})
        pressure_rows = [indexed[i, r]["pressure_metrics"] for i in range(scene_count) for r in members]
        keys = sorted({key for row in pressure_rows if row is not None for key in row})
        if keys:
            target["pressure"] = metric_rollup(members, lambda i, r, m: indexed[i, r]["pressure_metrics"][m], keys)
        result["arms" if name in ARMS else "readers"][name] = target
    return result
