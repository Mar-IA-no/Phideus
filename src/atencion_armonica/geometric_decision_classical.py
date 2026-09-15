"""Observable classical references in raw fit precision, no truth or refitting."""
from __future__ import annotations

import numpy as np

from . import generative_evidence as ge
from .geometric_decision_core import choose_energy
from .geometric_decision_scene_store import load_fit, load_source, fixed_arrays


def scores(partitions, fits, *, n, delivered):
    ps = ge.partitions_checked(partitions, n)
    ge.candidate_channel(ps, fits, n)
    if delivered.dtype != np.float32 or delivered.shape != (len(ps), 8) or not np.isfinite(delivered).all():
        raise ValueError("classical reference requires exact delivered eight-channel rows")
    result = {"z": delivered[:, 6].astype(np.float64), "d": delivered[:, 7].astype(np.float64)}
    branches = {}
    for family, allowed in ge.law.FAMILIES.items():
        selected = [min((b for b in allowed if b in fit["branches"]),
                        key=lambda b: (fit["branches"][b]["UB"], b)) for fit in fits]
        result[family] = np.asarray([fit["branches"][b]["UB"] for fit, b in zip(fits, selected)], np.float64)
        branches[family] = selected
    choices = {name: choose_energy(values, ps) for name, values in result.items()}
    return result, {"choices": choices, "upper_branches": branches,
        "extended_vs_z_same_choice": choices["extended"] == choices["z"],
        "authority": "raw UB is not reconstructed from delivered/log channels"}


def preserve_classical(store, folder, cache, *, check):
    batch = store.json(cache["ref"])
    cp = ge.CHECKPOINTS[0]
    if batch["binding"] != store.binding or batch["scene_ids"] != cache["scene_ids"]:
        raise ValueError("classical reference input identity differs")
    records = []
    for i, input_ref in enumerate(batch["records"]):
        check()
        data = store.json(input_ref)
        scene = load_source(store, data["source"])
        fit = load_fit(store, data["fit"], scene=scene)
        arrays, diagnostic = scores(scene["partitions"], fit["fits"], n=len(scene["q32"]),
            delivered=cache["rows"][cp][i]["evidence"])
        prefix = f"{folder}/{cache['scene_ids'][i]:05d}"
        ref = fixed_arrays(store, prefix+".npz", arrays)
        records.append(store.publish_json(prefix+".json", {"schema": "geometric-decision-classical-v1",
            "binding": store.binding, "inputs": input_ref, "source": data["source"], "fit": data["fit"],
            "scene_id": cache["scene_ids"][i], "arrays": ref, **diagnostic, "truth_access": False}))
    return store.publish_json(folder+"/index.json", {"schema": "geometric-decision-classical-index-v1",
        "binding": store.binding, "inputs": cache["ref"], "scene_ids": cache["scene_ids"],
        "records": records, "truth_access": False})
