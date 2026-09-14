"""Recoverable scene artifacts with explicit source, fit and input boundaries.

Caller authenticates execution authority and resources. No draw, forward,
fitter or supervision port. Historical split tables and stores stay untouched.
"""
from __future__ import annotations

from math import comb
from itertools import combinations
import math

import numpy as np

from . import generative_evidence as ge
from . import generative_evidence_storage as storage
from .geometric_decision_observables import scene_from_sources, inputs_from_fits, roundtrip_observation
from .geometric_decision_core import delivered_interface
from .generative_evidence_cache import ROW_KEYS, FIELDS
from .geometric_decision_profile import assert_same_arrays
from .partial_compatibility_cache import encoded


def fixed_arrays(store, path, arrays):
    member = store.path(path)
    if member.exists():
        ref = store.reference(member)
        assert_same_arrays(store.arrays(ref), arrays)
        return ref
    ref = store.publish_arrays(path, arrays)
    assert_same_arrays(store.arrays(ref), arrays)
    return ref


def source_metadata(scene):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in scene.items()
            if k not in ("features", "logits")}


def source_derivation(store, observation, derivation, *, expected_seed, split):
    if not isinstance(derivation, dict):
        raise ValueError("source requires typed original/roundtrip derivation")
    if derivation.get("kind") == "original" and set(derivation) == {"kind", "observation"}:
        if encoded(store.json(derivation["observation"])) != encoded(observation):
            raise ValueError("original source differs from authenticated observation")
        return derivation.copy(), None
    if derivation.get("kind") != "roundtrip" or set(derivation) != {"kind", "parent"}:
        raise ValueError("roundtrip source requires its authenticated original parent")
    parent_record = store.json(derivation["parent"])
    if parent_record["derivation"]["kind"] != "original":
        raise ValueError("roundtrip parent must be an original, not another probe")
    parent = load_source(store, derivation["parent"])
    if parent["split"] != split:
        raise ValueError("roundtrip parent belongs to another role")
    result = roundtrip_observation(parent["observation"], expected_seed=expected_seed)
    if encoded(result["observation"]) != encoded(observation):
        raise ValueError("roundtrip source differs from original parent transformation")
    return {**derivation, "lineage": result["lineage"]}, result["coordinates"]


def save_source(store, folder, scene, *, origin, expected_seed, derivation):
    # Rebuild observable geometry before publication, never trusting an extra
    # field or a separately supplied pool to stand in for the source operation.
    checked = scene_from_sources(scene["split"], scene["observation"], scene["features"], scene["logits"],
                                 expected_seed=expected_seed)
    if set(scene) != set(checked) or encoded(source_metadata(scene)) != encoded(source_metadata(checked)):
        raise ValueError("source scene differs from its observable reconstruction")
    if not isinstance(origin, dict) or not origin:
        raise ValueError("source requires caller-authenticated provenance")
    derived, coordinates = source_derivation(store, checked["observation"], derivation, expected_seed=expected_seed, split=scene["split"])
    coordinate_ref = None if coordinates is None else fixed_arrays(store, folder+"/coordinates.npz", coordinates)
    arrays = {**{"feature/"+k: v for k, v in checked["features"].items()},
              **{f"logits/{cp}": v for cp, v in checked["logits"].items()}}
    ref = fixed_arrays(store, folder+"/source.npz", arrays)
    return store.publish_json(folder+"/source.json", {"schema": "geometric-decision-source-v1",
        "binding": store.binding, "origin": origin, "expected_seed": expected_seed,
        "scene": source_metadata(checked), "arrays": ref, "derivation": derived, "coordinates": coordinate_ref})


def load_source(store, ref):
    record = store.json(ref)
    if (set(record) != {"schema", "binding", "origin", "expected_seed", "scene", "arrays", "derivation", "coordinates"}
            or record["schema"] != "geometric-decision-source-v1" or record["binding"] != store.binding):
        raise ValueError("source receipt schema or binding differs")
    arrays = store.arrays(record["arrays"])
    features = {k.removeprefix("feature/"): v for k, v in arrays.items() if k.startswith("feature/")}
    logits = {cp: arrays[f"logits/{cp}"] for cp in ge.CHECKPOINTS}
    if set(arrays) != {*("feature/"+k for k in features), *(f"logits/{cp}" for cp in ge.CHECKPOINTS)}:
        raise ValueError("source arrays contain unexpected fields")
    meta = record["scene"]
    scene = scene_from_sources(meta["split"], meta["observation"], features, logits, expected_seed=record["expected_seed"])
    if encoded(source_metadata(scene)) != encoded(meta):
        raise ValueError("saved source metadata differs from observable reconstruction")
    supplied = {k: v for k, v in record["derivation"].items() if k != "lineage"}
    derived, coordinates = source_derivation(store, scene["observation"], supplied, expected_seed=record["expected_seed"], split=scene["split"])
    if encoded(derived) != encoded(record["derivation"]):
        raise ValueError("saved roundtrip lineage differs from authenticated parent")
    if coordinates is None:
        if record["coordinates"] is not None:
            raise ValueError("original source cannot carry undeclared probe coordinates")
    else:
        assert_same_arrays(store.arrays(record["coordinates"]), coordinates)
    return scene


def validate_fits(scene, fitted):
    if not isinstance(fitted, dict) or set(fitted) != {"fits", "group_factors"}:
        raise ValueError("fit archive requires candidates and complete group factors")
    ps, n = scene["partitions"], len(scene["q32"])
    ge.candidate_channel(ps, fitted["fits"], n)
    expected = {(b, g) for p in ps for b in ge.BRANCHES if len(p) in ge.law.BRANCHES[b][2] for g in p}
    seen = set()
    for row in fitted["group_factors"]:
        if not isinstance(row, dict) or set(row) != {"branch", "group", "factor"}:
            raise ValueError("fit factor schema differs")
        key, factor = (row["branch"], tuple(row["group"])), row["factor"]
        if not isinstance(factor, dict) or set(factor) != {"branch", "size", "mean_observed", "fine", "coarse"}:
            raise ValueError("fit factor contains unexpected fields")
        if (key not in expected or key in seen or factor["branch"] != key[0] or factor["size"] != len(key[1])
                or any(len(factor[r]) != comb(8, len(key[1])) for r in ("coarse", "fine"))):
            raise ValueError("fit factors lost branches, groups or assignments")
        m = len(key[1])
        scalar = lambda x: type(x) in (int, float) and math.isfinite(x)
        if not scalar(factor["mean_observed"]):
            raise ValueError("factor mean is not a finite scalar")
        assignment_keys = {"indices", "beta_index", "gamma_index", "beta", "gamma", "template", "template_mean", "sse"}
        for resolution in ("fine", "coarse"):
            assignments = factor[resolution]
            if not isinstance(assignments, list):
                raise ValueError("factor assignments require an explicit list")
            for assignment in assignments:
                if (not isinstance(assignment, dict) or set(assignment) != assignment_keys
                        or any(not scalar(assignment[k]) for k in ("beta", "gamma", "template_mean", "sse"))
                        or assignment["sse"] < 0
                        or any(type(assignment[k]) is not int or assignment[k] < 0 for k in ("beta_index", "gamma_index"))
                        or not isinstance(assignment["indices"], list) or len(assignment["indices"]) != m
                        or any(type(i) is not int for i in assignment["indices"])
                        or not isinstance(assignment["template"], list) or len(assignment["template"]) != m
                        or any(not scalar(x) for x in assignment["template"])):
                    raise ValueError("factor assignment contains unexpected fields or values")
            if [tuple(a["indices"]) for a in assignments] != list(combinations(range(1, 9), m)):
                raise ValueError("factor assignments do not retain the complete ordered roster")
        seen.add(key)
    if seen != expected:
        raise ValueError("fit factor roster is incomplete")
    replayed = ge.law.replay_fits(scene["q32"], fitted["group_factors"], ps)
    if encoded(replayed) != encoded(fitted["fits"]):
        raise ValueError("candidate fits differ from their factors or contain unexpected fields")


def save_fit(store, folder, source_ref, fitted, *, origin):
    scene = load_source(store, source_ref)
    validate_fits(scene, fitted)
    if not isinstance(origin, dict) or not origin:
        raise ValueError("fit requires caller-authenticated runtime/grid provenance")
    path = store.path(folder+"/fit.json")
    expected = {"schema": "geometric-decision-fit-v1", "binding": store.binding,
                "source": source_ref, "origin": origin, "identity": scene["identity"]}
    if path.exists():
        ref = store.reference(path)
        record = store.json(ref)
        if {k: v for k, v in record.items() if k != "artifact"} != expected:
            raise ValueError("cannot replace fit provenance")
        if encoded(load_fit(store, ref, scene=scene)) != encoded(fitted):
            raise ValueError("cannot replace completed fit")
        return ref
    payload = store.path(folder+"/factors.json.gz")
    if payload.exists():
        raise RuntimeError("unreceipted factor bytes require reconciliation, never refitting")
    payload.parent.mkdir(parents=True, exist_ok=True)
    receipt = storage.write_scene(payload, fitted)
    storage.read_scene(payload, receipt)
    return store.publish_json(folder+"/fit.json", {**expected,
        "artifact": {"path": payload.relative_to(store.root).as_posix(), **receipt}})


def load_fit(store, ref, *, scene=None):
    record = store.json(ref)
    if (set(record) != {"schema", "binding", "source", "origin", "identity", "artifact"}
            or record["schema"] != "geometric-decision-fit-v1" or record["binding"] != store.binding):
        raise ValueError("fit receipt schema or binding differs")
    scene = load_source(store, record["source"]) if scene is None else scene
    if record["identity"] != scene["identity"]:
        raise ValueError("fit identity differs from its source")
    artifact = record["artifact"]
    fitted = storage.read_scene(store.path(artifact["path"]), {k: v for k, v in artifact.items() if k != "path"})
    validate_fits(scene, fitted)
    return fitted


def save_inputs(store, folder, source_ref, fit_ref, normalizers, *, scale, normalization_ref):
    source = store.json(source_ref)
    scene = load_source(store, source_ref)
    if store.json(fit_ref)["source"] != source_ref:
        raise ValueError("inputs require fit and source from the same scene")
    fitted = load_fit(store, fit_ref, scene=scene)
    delivered = inputs_from_fits(scene, fitted["fits"], normalizers, scale=scale, expected_seed=source["expected_seed"])
    arrays, metadata = {}, {}
    for cp in ge.CHECKPOINTS:
        row = delivered["raw"][cp]
        arrays.update({f"raw/{cp}/{k}": v for k, v in row.items() if isinstance(v, np.ndarray)})
        arrays.update({f"inputs/{cp}/{k}": v for k, v in delivered["inputs"][cp].items()})
        metadata[str(cp)] = {k: v for k, v in row.items() if not isinstance(v, np.ndarray)}
    ref = fixed_arrays(store, folder+"/inputs.npz", arrays)
    return store.publish_json(folder+"/inputs.json", {"schema": "geometric-decision-inputs-v1",
        "binding": store.binding, "source": source_ref, "fit": fit_ref, "identity": scene["identity"],
        "normalization": normalization_ref, "scale": scale, "metadata": metadata,
        "diagnostics": {str(cp): delivered["diagnostics"][cp] for cp in ge.CHECKPOINTS}, "arrays": ref})


def load_inputs(store, ref, *, normalization_ref):
    """Reopen preserved rows and authenticate the complete source/fit chain.

    Rebuilds the inexpensive observable pool, never a forward or fit. TRAIN
    normalization authority remains the caller's authenticated reference;
    full raw-to-normalized replay additionally requires its normalizer values.
    """
    record = store.json(ref)
    if (set(record) != {"schema", "binding", "source", "fit", "identity", "normalization", "scale", "metadata", "diagnostics", "arrays"}
            or record["schema"] != "geometric-decision-inputs-v1" or record["binding"] != store.binding
            or record["normalization"] != normalization_ref
            or set(record["metadata"]) != {str(cp) for cp in ge.CHECKPOINTS}
            or set(record["diagnostics"]) != {str(cp) for cp in ge.CHECKPOINTS}):
        raise ValueError("delivered receipt schema or normalization differs")
    source, fit = store.json(record["source"]), store.json(record["fit"])
    if (source["scene"]["identity"] != record["identity"] or fit["identity"] != record["identity"]
            or fit["source"] != record["source"]):
        raise ValueError("delivered source/fit identity differs")
    scene = load_source(store, record["source"])
    load_fit(store, record["fit"], scene=scene)
    arrays = store.arrays(record["arrays"])
    rows, raw, consumed = {}, {}, set()
    for cp in ge.CHECKPOINTS:
        rows[cp], raw[cp] = {}, dict(record["metadata"][str(cp)])
        for name, value in arrays.items():
            for prefix, target in ((f"inputs/{cp}/", rows[cp]), (f"raw/{cp}/", raw[cp])):
                if name.startswith(prefix):
                    key = name.removeprefix(prefix)
                    if key in target:
                        raise ValueError("array/metadata field collision")
                    target[key] = value
                    consumed.add(name)
    if consumed != set(arrays):
        raise ValueError("unknown delivered array fields")
    from .geometric_decision_inference import validate_rows
    validate_rows([rows[cp] for cp in ge.CHECKPOINTS])
    for cp in ge.CHECKPOINTS:
        if set(raw[cp]) != ROW_KEYS or type(raw[cp]["n"]) is not int or raw[cp]["n"] != len(scene["q32"]):
            raise ValueError("raw row schema or observed extent differs")
        ps = ge.partitions_checked(raw[cp]["partitions"], raw[cp]["n"])
        groups = sorted({g for p in ps for g in p})
        if (ps != scene["partitions"] or [tuple(g) for g in raw[cp]["group_ids"]] != groups
                or not np.array_equal(raw[cp]["canonical_to_observed"], scene["canonical_to_observed"])
                or type(raw[cp]["q_tie_count"]) is not int
                or raw[cp]["q_tie_count"] != len(scene["q32"])-len(np.unique(scene["q32"]))):
            raise ValueError("raw row candidate, group or event identity differs")
        for name, (width, dtype) in FIELDS.items():
            value, count = raw[cp][name], len(groups) if name == "groups" else len(ps)
            if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != (count, width) or not np.isfinite(value).all():
                raise ValueError("raw row numeric field differs")
        expected = np.array([[len(g)/raw[cp]["n"] if g in p else 0 for g in groups] for p in ps], np.float32).reshape(len(ps), len(groups))
        if (len(ps) != len(rows[cp]["globals"]) or not np.array_equal(rows[cp]["incidence"], expected)
                or raw[cp]["incidence"].dtype != np.float32 or not np.array_equal(raw[cp]["incidence"], expected)):
            raise ValueError("delivered incidence differs from its candidate membership")
        available = np.array([[len(p) in ge.law.BRANCHES[b][2] for b in ge.BRANCHES] for p in ps], bool).reshape(-1, 3)
        if not np.array_equal(raw[cp]["available"], available):
            raise ValueError("raw branch availability differs")
        six = {**rows[cp], "evidence": rows[cp]["evidence"][:, :6]}
        reconstructed, diagnostics = delivered_interface(six, raw[cp], scale=record["scale"],
            split_seed=source["expected_seed"], scene_id=scene["observation"]["scene_id"])
        assert_same_arrays(rows[cp], reconstructed)
        if encoded(diagnostics) != encoded(record["diagnostics"][str(cp)]):
            raise ValueError("delivered donor diagnostics differ")
    return {"record": record, "inputs": rows, "raw": raw}
