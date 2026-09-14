"""Scene-weighted descriptive summaries, kept separate for each scenario.

No confidence intervals, significance decisions, pooled four-test score or
candidate-weighted aggregation. Per-scene artifacts remain the primary record.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .operator_objective_scene import ARMS, CELLS, CLASSICAL, mean_record

PRESENCE = ("pool", "neighbor", "absent")
SCENARIOS = ("iid", "ood_beta", "ood_polyphony", "deformed_family")


def distribution(values):
    values = list(values)
    record = mean_record(values)
    valid = [float(v) for v in values if v is not None]
    quantiles = np.quantile(valid, [.1, .5, .9], method="linear").tolist() if valid else [None]*3
    return {**record, "undefined": record["total"]-record["defined"],
            "q10": quantiles[0], "q50": quantiles[1], "q90": quantiles[2]}


def _summaries(result):
    for name in CLASSICAL:
        yield ("classical", name), result["classical"][name]["summary"], "strata"
    for arm in ARMS:
        yield ("arm", arm), result["arm_summary"][arm], "cells"
        for cell in CELLS:
            yield ("cell", arm, cell), result["learned"][arm][cell]["summary"], "strata"


def scene_scalars(result):
    """Yield stable paths, scalar scene estimands, and descriptive support totals."""
    for prefix, summary, unit in _summaries(result):
        for scheme, metrics in summary.items():
            for metric, record in metrics.items():
                yield (*prefix, scheme, metric), record["mean"], {
                    f"{unit}_defined": record["defined"], f"{unit}_total": record["total"]}
    for contrast, schemes in result["paired"].items():
        for scheme, metrics in schemes.items():
            for metric, record in metrics.items():
                yield ("paired", contrast, scheme, metric), record["mean"], {
                    "cells_defined": record["valid_cell_count"], "cells_total": 9,
                    "paired_strata_across_cells": sum(c["count"] for c in record["cells"].values())}
                complete = record["complete_nine"]
                yield ("paired_complete_nine", contrast, scheme, metric), complete["mean"], {
                    "common_strata": complete["count"], "cells_total": 9}
    for scheme, strata in result["oracles"].items():
        comparisons = [r["comparisons"] for r in strata.values()]
        names = ("target64_target32_same_choice", "target32_in_target64_optima", "target32_target64_regret",
                 "target64_target32_regret", "target32_in_near_target64_optima", "target64_ari_same_choice",
                 "target64_ari_optima_intersect", "ari_gap_of_target64_choice", "target64_regret_of_ari_choice",
                 "near_target64_optima_count")
        for name in names:
            values = [None if c is None else float(len(c["near_target64_optima"]))
                      if name == "near_target64_optima_count" else float(c[name]) for c in comparisons]
            record = mean_record(values)
            yield ("oracle", scheme, name), record["mean"], {
                "strata_defined": record["defined"], "strata_total": record["total"]}


def _statuses(result):
    for name in CLASSICAL:
        for scheme, record in result["classical"][name]["schemes"].items():
            yield ("classical", name, scheme), record["tau_status_counts"]
    for arm in ARMS:
        for cell in CELLS:
            for scheme, record in result["learned"][arm][cell]["schemes"].items():
                yield ("cell", arm, cell, scheme), record["tau_status_counts"]


class ScenarioAccumulator:
    """Ordered streaming input; never present a partial roster as complete.

    `expected_scenes` is only smaller than 512 for explicit mathematical tests.
    The campaign runner must instantiate the fixed production extent of 512.
    """

    def __init__(self, scenario, *, expected_scenes=512):
        if scenario not in SCENARIOS or type(expected_scenes) is not int or not 1 <= expected_scenes <= 512:
            raise ValueError("unknown scenario or invalid declared extent")
        self.scenario, self.expected_scenes = scenario, expected_scenes
        self.count = 0
        self.schema = None
        self.slices = {p: {"scenes": [], "candidate_counts": [], "metrics": {}, "tau_status_counts": {}}
                       for p in ("all", *PRESENCE)}

    def add(self, scene_id, presence, result):
        if (type(scene_id) is not int or scene_id != self.count or scene_id >= self.expected_scenes
                or presence not in PRESENCE or result.get("schema") != "operator-objective-scene-v1"
                or type(result.get("candidate_count")) is not int or not 0 <= result["candidate_count"] <= 82):
            raise ValueError("scene extent, order, presence or diagnostic schema differs")
        scalars = list(scene_scalars(result))
        schema = {path for path, _, _ in scalars}
        if len(schema) != len(scalars) or (self.schema is not None and schema != self.schema):
            raise ValueError("scalar metric roster changed between scenes")
        # Validate the whole row before mutating accumulator state.
        for _, value, support in scalars:
            mean_record([value])
            if any(type(n) is not int or n < 0 for n in support.values()):
                raise ValueError("invalid metric support count")
        statuses = list(_statuses(result))
        for _, counts in statuses:
            if any(type(n) is not int or n < 0 for n in counts.values()):
                raise ValueError("invalid tau status count")
        self.schema = schema
        for presence_slice in ("all", presence):
            target = self.slices[presence_slice]
            target["scenes"].append(scene_id)
            target["candidate_counts"].append(result["candidate_count"])
            for path, value, support in scalars:
                key = "/".join(path)
                entry = target["metrics"].setdefault(key, {"values": [], "support_totals": Counter()})
                entry["values"].append(value)
                entry["support_totals"].update(support)
            for path, counts in statuses:
                key = "/".join(path)
                target["tau_status_counts"].setdefault(key, Counter()).update(counts)
        self.count += 1

    def finalize(self):
        if self.count != self.expected_scenes:
            raise ValueError("cannot summarize an incomplete scenario roster")
        slices = {}
        for name, value in self.slices.items():
            metrics = {}
            # Empty presence slices keep the same metric schema, all undefined.
            for path in sorted(self.schema):
                key = "/".join(path)
                record = value["metrics"].get(key, {"values": [], "support_totals": {}})
                metrics[key] = {**distribution(record["values"]),
                                "support_totals": dict(sorted(record["support_totals"].items()))}
            slices[name] = {"scene_ids": value["scenes"].copy(), "scene_count": len(value["scenes"]),
                            "candidate_count_distribution": distribution(value["candidate_counts"]),
                            "metrics": metrics,
                            "tau_status_counts": {k: dict(sorted(v.items())) for k, v in
                                                  sorted(value["tau_status_counts"].items())}}
        return {"schema": "operator-objective-scenario-summary-v1", "scenario": self.scenario,
                "scene_count": self.count, "unit": "scene", "quantile_method": "linear",
                "authority": "RETROSPECTIVE_DESCRIPTIVE_NOT_PROMOTED", "slices": slices}
