"""Archived-head readouts and coordinate probes; caller owns fresh admission.

Load and authenticate scene inputs once per batch, not once per head. No
sampler, labels, new fit, calibration selection or execution authority here.
All model parameters are replaced from the archived numeric checkpoint.
"""
from __future__ import annotations

import numpy as np
import torch

from . import generative_evidence as ge
from .geometric_decision_core import choose_energy, route_inputs
from .geometric_decision_head_archive import read_head_arrays
from .geometric_decision_inference import predict, transport_prediction, validate_rows
from .geometric_decision_model import GeometricDecisionHead
from .geometric_decision_scene_store import fixed_arrays, load_inputs
from .partial_compatibility_cache import encoded


def input_batch(store, ref, *, normalization_ref, scale, check):
    batch = store.json(ref)
    if (set(batch) != {"schema", "binding", "sources", "scene_ids", "normalization", "scale", "records"}
            or batch["schema"] != "geometric-decision-input-batch-v1" or batch["binding"] != store.binding
            or batch["normalization"] != normalization_ref or batch["scale"] != scale
            or not 1 <= len(batch["records"]) <= 512
            or len(batch["records"]) != len(batch["scene_ids"])
            or batch["scene_ids"] != sorted(set(batch["scene_ids"]))):
        raise ValueError("prediction input batch differs from the authenticated interface")
    sources = store.json(batch["sources"])
    if (set(sources) != {"schema", "binding", "features", "logits", "split", "split_seed", "scene_ids", "sources"}
            or sources["schema"] != "geometric-decision-source-batch-v1" or sources["binding"] != store.binding
            or sources["scene_ids"] != batch["scene_ids"] or len(sources["sources"]) != len(batch["records"])):
        raise ValueError("prediction source roster differs")
    rows, raw = {cp: [] for cp in ge.CHECKPOINTS}, {cp: [] for cp in ge.CHECKPOINTS}
    for ref_input, ref_source in zip(batch["records"], sources["sources"]):
        check()
        value = load_inputs(store, ref_input, normalization_ref=normalization_ref)
        if value["record"]["source"] != ref_source or value["record"]["scale"] != scale:
            raise ValueError("prediction input belongs to another source or scale")
        source = store.json(ref_source)
        i = len(rows[ge.CHECKPOINTS[0]])
        if (source["scene"]["observation"]["scene_id"] != batch["scene_ids"][i]
                or source["scene"]["split"] != sources["split"]
                or source["expected_seed"] != sources["split_seed"]):
            raise ValueError("prediction scene identity differs from source batch")
        for cp in ge.CHECKPOINTS:
            rows[cp].append(value["inputs"][cp])
            raw[cp].append(value["raw"][cp])
    counts = validate_rows(rows[ge.CHECKPOINTS[0]])
    if any(validate_rows(rows[cp]) != counts for cp in ge.CHECKPOINTS):
        raise ValueError("backbones do not share the candidate universe")
    return {"ref": ref, "binding": store.binding, "scene_ids": batch["scene_ids"],
        "split": sources["split"], "split_seed": sources["split_seed"], "rows": rows, "raw": raw,
        "probe_indices": [i for i, count in enumerate(counts) if count][:4]}


def archived_model(head_store, head_ref, *, device):
    if device not in ("cpu", "cuda:0"):
        raise ValueError("unsupported readout device")
    record, arrays = read_head_arrays(head_store, head_ref)
    model = GeometricDecisionHead(record["reader_seed"], record["arm"].rsplit("_", 1)[0])
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in arrays.items()}, strict=True)
    model.to(device).eval()
    return record, model


def checked_prediction(arrays, rows, raw, *, route, initial):
    counts = validate_rows(rows)
    offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    if (set(arrays) != {"components", "energy", "offsets"} or arrays["offsets"].dtype != np.int64
            or not np.array_equal(arrays["offsets"], offsets) or arrays["components"].dtype != np.float64
            or arrays["components"].shape != (int(offsets[-1]), 2) or arrays["energy"].dtype != np.float64
            or arrays["energy"].shape != (int(offsets[-1]),) or not np.isfinite(arrays["components"]).all()
            or not np.array_equal(arrays["energy"], arrays["components"].sum(-1, dtype=np.float64))
            or len(raw) != len(rows)):
        raise ValueError("invalid preserved signed prediction")
    choices = []
    for i, (row, source) in enumerate(zip(rows, raw)):
        energy = arrays["energy"][offsets[i]:offsets[i+1]]
        components = arrays["components"][offsets[i]:offsets[i+1]]
        if len(source["partitions"]) != len(energy):
            raise ValueError("prediction and candidate universe differ")
        if initial:
            bypass = row["evidence"][:, 6 if route == "geometric" else 7].astype(np.float64) if route in ("geometric", "decoupled") else np.zeros(len(energy), np.float64)
            if not np.array_equal(energy, bypass) or not np.array_equal(components, np.repeat((bypass/2)[:, None], 2, axis=1)):
                raise ValueError("initial prediction differs from imposed bypass")
        choices.append(None if not len(energy) else choose_energy(energy, source["partitions"]))
    return choices


def preserve_prediction(store, folder, cache, head_store, head_ref, *, device, runtime, check):
    """Reuse exact outputs; an orphan payload stops before a new forward."""
    if cache["binding"] != store.binding or not isinstance(runtime, dict) or not runtime:
        raise ValueError("prediction cache/runtime lacks explicit provenance")
    head, _ = read_head_arrays(head_store, head_ref)
    cp, arm, seed, stage = head["checkpoint_seed"], head["arm"], head["reader_seed"], head["stage"]
    rows, raw = cache["rows"][cp], cache["raw"][cp]
    route = arm.rsplit("_", 1)[0]
    prefix = f"{folder}/cp_{cp}/{arm}/seed_{seed}/{stage}"
    metadata = {"schema": "geometric-decision-prediction-v1", "binding": store.binding,
        "inputs": cache["ref"], "head_root": str(head_store.root), "head": head_ref,
        "head_binding": head_store.binding, "checkpoint_seed": cp, "arm": arm,
        "reader_seed": seed, "stage": stage, "epoch": head["epoch"],
        "split": cache["split"], "split_seed": cache["split_seed"], "scene_ids": cache["scene_ids"],
        "runtime": runtime, "device": device, "truth_access": False}
    check()
    if store.path(prefix+".json").exists():
        ref = store.reference(store.path(prefix+".json"))
        record = store.json(ref)
        if set(record) != set(metadata) | {"arrays", "choices"} or encoded({k: record[k] for k in metadata}) != encoded(metadata):
            raise ValueError("saved prediction has another head, input or runtime")
        arrays = store.arrays(record["arrays"])
        choices = checked_prediction(arrays, rows, raw, route=route, initial=stage == "initial")
        if choices != record["choices"]:
            raise ValueError("saved choice differs from exact energy and canonical tie rule")
        return ref
    if store.path(prefix+".npz").exists():
        raise RuntimeError("unreceipted prediction; reconcile before any re-forward")
    _, model = archived_model(head_store, head_ref, device=device)
    arrays = predict(model, rows, device=device, check=check)
    choices = checked_prediction(arrays, rows, raw, route=route, initial=stage == "initial")
    array_ref = fixed_arrays(store, prefix+".npz", arrays)
    return store.publish_json(prefix+".json", {**metadata, "arrays": array_ref, "choices": choices})


def checked_transport(arrays, row, raw, batched, *, route, initial):
    """Recompute transport relationships/diagnostics without model execution.

    This validates preserved arithmetic, not that a forward actually ran;
    execution provenance remains the enclosing operator's responsibility.
    """
    count = validate_rows([row])[0]
    if not count:
        raise ValueError("transport requires an eligible scene")
    ps = ge.partitions_checked(raw["partitions"], sum(map(len, raw["partitions"][0])))
    groups = sorted({g for p in ps for g in p})
    n = sum(map(len, ps[0]))
    incidence = np.array([[len(g)/n if g in p else 0 for g in groups] for p in ps], np.float32)
    if len(ps) != count or not np.array_equal(row["incidence"], incidence):
        raise ValueError("transport incidence differs from canonical partitions")
    orders = {"candidate_order": np.arange(count-1, -1, -1, dtype=np.int64),
        "group_order": np.arange(len(groups)-1, -1, -1, dtype=np.int64),
        "channel_order": np.arange(7, -1, -1, dtype=np.int64)}
    c, g, h = (orders[k] for k in ("candidate_order", "group_order", "channel_order"))
    orders["weight_column_order"] = np.r_[np.arange(33, dtype=np.int64), 33+h]
    routed = route_inputs(row, route)
    expected = {**orders, "transported_inputs/groups": routed["groups"][g],
        "transported_inputs/globals": routed["globals"][c],
        "transported_inputs/incidence": routed["incidence"][np.ix_(c, g)],
        "transported_inputs/evidence": routed["evidence"][np.ix_(c, h)],
        "batched_components": batched}
    numeric = {"baseline/components": (count, 2), "baseline/energy": (count,),
        "transported_components": (count, 2), "restored_components": (count, 2),
        "transported_energy": (count,), "restored_energy": (count,)}
    if set(arrays) != set(expected) | set(numeric) | {"baseline/offsets"}:
        raise ValueError("transport array schema differs")
    for k, v in expected.items():
        if arrays[k].dtype != v.dtype or arrays[k].shape != v.shape or not np.array_equal(arrays[k], v):
            raise ValueError("transport coordinates or batched baseline differ: "+k)
    for k, shape in numeric.items():
        if arrays[k].dtype != np.float64 or arrays[k].shape != shape or not np.isfinite(arrays[k]).all():
            raise ValueError("invalid transport signed array: "+k)
    baseline = {k: arrays["baseline/"+k] for k in ("components", "energy", "offsets")}
    before = checked_prediction(baseline, [row], [raw], route=route, initial=initial)[0]
    moved, restored = arrays["transported_components"], arrays["restored_components"]
    moved_energy, restored_energy = arrays["transported_energy"], arrays["restored_energy"]
    if (not np.array_equal(restored, moved[np.argsort(c)])
            or not np.array_equal(moved_energy, moved.sum(-1, dtype=np.float64))
            or not np.array_equal(restored_energy, restored.sum(-1, dtype=np.float64))):
        raise ValueError("transport energy or inverse permutation differs")
    checked_prediction({"components": restored, "energy": restored_energy, "offsets": baseline["offsets"]},
        [row], [raw], route=route, initial=initial)
    moved_ps = [ps[i] for i in c]
    after = choose_energy(moved_energy, moved_ps)
    def margin(energy):
        ordered = np.sort(energy)
        return None if len(ordered) < 2 else float(ordered[1]-ordered[0])
    bypass = 6 if route == "geometric" else 7 if route == "decoupled" else None
    return {"atol": 1e-6, "rtol": 1e-5,
        "within_numeric_tolerance": bool(np.allclose(restored, baseline["components"], atol=1e-6, rtol=1e-5)),
        "max_component_error": float(np.max(np.abs(restored-baseline["components"]))),
        "max_energy_error": float(np.max(np.abs(restored_energy-baseline["energy"]))),
        "same_exact_choice": ps[before] == moved_ps[after],
        "original_margin": margin(baseline["energy"]), "transported_margin": margin(moved_energy),
        "original_choice": ps[before], "transported_choice": moved_ps[after],
        "scope": "coordinate transport and float32 reduction stability, not learned physical invariance",
        "bypass_channel": None if bypass is None else int(np.flatnonzero(h == bypass)[0]),
        "singleton_vs_batch_max_error": float(np.max(np.abs(baseline["components"]-batched))),
        "singleton_vs_batch_same_exact_choice": before == choose_energy(batched.sum(-1, dtype=np.float64), ps)}


def preserve_transport(store, folder, cache, prediction_ref, head_store, *, check):
    """Save transported coordinates and distinguish singleton/batched numerics.

    Must be called for all initial/selected heads before the global seal.
    This receipt is not a seal and does not authorize opening any labels.
    """
    prediction = store.json(prediction_ref)
    if (prediction["binding"] != store.binding or prediction["inputs"] != cache["ref"]
            or prediction["head_binding"] != head_store.binding or prediction["head_root"] != str(head_store.root)
            or prediction["truth_access"] is not False):
        raise ValueError("transport is not bound to the preserved prediction")
    cp = prediction["checkpoint_seed"]
    full = store.arrays(prediction["arrays"])
    route = prediction["arm"].rsplit("_", 1)[0]
    if checked_prediction(full, cache["rows"][cp], cache["raw"][cp], route=route,
                          initial=prediction["stage"] == "initial") != prediction["choices"]:
        raise ValueError("transport baseline choices differ")
    indices = [i for i, count in enumerate(validate_rows(cache["rows"][cp])) if count][:4]
    if cache["probe_indices"] != indices:
        raise ValueError("transport probe roster must contain the first four eligible scenes")
    records, model = [], None
    for i in indices:
        check()
        prefix = f"{folder}/cp_{cp}/{prediction['arm']}/seed_{prediction['reader_seed']}/{prediction['stage']}/{cache['scene_ids'][i]:05d}"
        metadata = {"schema": "geometric-decision-transport-v1", "binding": store.binding,
            "prediction": prediction_ref, "scene_id": cache["scene_ids"][i], "scene_index": i,
            "truth_access": False}
        a, b = full["offsets"][i:i+2]
        def validate(arrays):
            return checked_transport(arrays, cache["rows"][cp][i], cache["raw"][cp][i],
                full["components"][a:b], route=route, initial=prediction["stage"] == "initial")
        if store.path(prefix+".json").exists():
            ref = store.reference(store.path(prefix+".json"))
            record = store.json(ref)
            if (set(record) != set(metadata) | {"arrays", "diagnostic"}
                    or encoded({k: record[k] for k in metadata}) != encoded(metadata)):
                raise ValueError("saved transport belongs to another prediction")
            diagnostic = validate(store.arrays(record["arrays"]))
            if encoded(diagnostic) != encoded(record["diagnostic"]):
                raise ValueError("saved transport diagnostic differs from preserved arrays")
            records.append(ref)
            continue
        if store.path(prefix+".npz").exists():
            raise RuntimeError("unreceipted transport; no silent re-forward")
        if model is None:
            _, model = archived_model(head_store, prediction["head"], device=prediction["device"])
        result = transport_prediction(model, cache["rows"][cp][i], cache["raw"][cp][i]["partitions"],
            device=prediction["device"], check=check)
        arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        arrays.update({"baseline/"+k: v for k, v in result["baseline"].items()})
        arrays.update({"transported_inputs/"+k: v for k, v in result["transported_inputs"].items()})
        arrays["batched_components"] = full["components"][a:b].copy()
        diagnostic = {**result["diagnostic"], "bypass_channel": result["bypass_channel"],
            "singleton_vs_batch_max_error": float(np.max(np.abs(result["baseline"]["components"]-arrays["batched_components"]))),
            "singleton_vs_batch_same_exact_choice": choose_energy(result["baseline"]["energy"], cache["raw"][cp][i]["partitions"]) == prediction["choices"][i]}
        if encoded(validate(arrays)) != encoded(diagnostic):
            raise ValueError("generated transport diagnostic differs from preserved arrays")
        arrays_ref = fixed_arrays(store, prefix+".npz", arrays)
        records.append(store.publish_json(prefix+".json", {**metadata, "arrays": arrays_ref, "diagnostic": diagnostic}))
    return store.publish_json(f"{folder}/cp_{cp}/{prediction['arm']}/seed_{prediction['reader_seed']}/{prediction['stage']}/index.json",
        {"schema": "geometric-decision-transport-index-v1", "binding": store.binding,
         "prediction": prediction_ref, "scene_ids": [cache["scene_ids"][i] for i in indices],
         "records": records, "truth_access": False})
