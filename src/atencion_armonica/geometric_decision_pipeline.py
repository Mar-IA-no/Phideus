"""Recoverable observable stages, under caller-owned freeze and resource lock.

The caller supplies admitted forward/fitting callbacks. No sampler, truth,
checkpoint discovery or GPU lease is obtained here. New observation seeds
are explicit, never borrowed from historical wrapper tables.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np

from . import generative_evidence as ge
from .geometric_decision_observables import validate_observation, canonical_observation, scene_from_sources
from .geometric_decision_scene_store import fixed_arrays, save_source, load_source, save_fit, load_fit, save_inputs, load_inputs
from .geometric_decision_profile import assert_same_arrays
from .partial_compatibility_cache import encoded, feature_record
from .structured_source_data import validate_record


def pack_logits(matrices, observations):
    if len(matrices) != len(observations):
        raise ValueError("forward must cover every supplied observation")
    sizes = np.asarray([len(o["log_f"]) for o in observations], np.int64)
    for z, n in zip(matrices, sizes):
        if (not isinstance(z, np.ndarray) or z.dtype != np.float32 or z.shape != (n, n)
                or not np.isfinite(z).all() or not np.array_equal(z, z.T)):
            raise ValueError("forward requires finite symmetric float32 logits")
    return {"sizes": sizes, "offsets": np.r_[np.int64(0), np.cumsum(sizes*sizes, dtype=np.int64)],
            "logits": np.concatenate([z.ravel() for z in matrices])}


def unpack_logits(arrays, observations):
    sizes = np.asarray([len(o["log_f"]) for o in observations], np.int64)
    offsets = np.r_[np.int64(0), np.cumsum(sizes*sizes, dtype=np.int64)]
    if (set(arrays) != {"sizes", "offsets", "logits"} or arrays["sizes"].dtype != np.int64
            or arrays["offsets"].dtype != np.int64 or not np.array_equal(arrays["sizes"], sizes)
            or not np.array_equal(arrays["offsets"], offsets) or arrays["logits"].shape != (int(offsets[-1]),)):
        raise ValueError("saved logits differ from the observation roster")
    matrices = [arrays["logits"][offsets[i]:offsets[i+1]].reshape(n, n).copy() for i, n in enumerate(sizes)]
    pack_logits(matrices, observations)
    return matrices


def prepare_sources(store, folder, split, observations, *, expected_seed, observation_origin,
                    checkpoints, runtime, forward, check):
    """One source batch (original or derived); authenticate inputs in caller.

    Already preserved features and all three logit arrays are reused. A
    receiptless logit payload stops before any new forward. Forward callbacks
    must enforce the frozen checkpoint/runtime and run resource checks.
    """
    if (not isinstance(observations, list) or not 1 <= len(observations) <= 512
            or not isinstance(observation_origin, dict) or not observation_origin
            or not isinstance(runtime, dict) or not runtime or not callable(forward) or not callable(check)
            or not isinstance(checkpoints, list) or [c.get("seed") for c in checkpoints] != list(ge.CHECKPOINTS)):
        raise ValueError("source preparation requires explicit roster, provenance, runtime and callbacks")
    # Freeze receipt identity independently of objects exposed to callbacks.
    observations, observation_origin, checkpoints, runtime = deepcopy(
        (observations, observation_origin, checkpoints, runtime))
    ids = [o["scene_id"] for o in observations]
    if ids != sorted(set(ids)):
        raise ValueError("source observation IDs must be unique and ordered")
    for o in observations:
        q = validate_observation(o, scene_id=o["scene_id"], split_seed=expected_seed)
        if encoded(o) != encoded(canonical_observation(o, q)):
            raise ValueError("original observation must use canonical JSON before any forward")
    if (observation_origin.get("kind") == "original" and set(observation_origin) == {"kind", "records"}
            and len(observation_origin["records"]) == len(observations)):
        derivations = [{"kind": "original", "observation": ref} for ref in observation_origin["records"]]
    elif (observation_origin.get("kind") == "roundtrip" and set(observation_origin) == {"kind", "parents"}
            and len(observation_origin["parents"]) == len(observations)):
        derivations = [{"kind": "roundtrip", "parent": ref} for ref in observation_origin["parents"]]
    else:
        raise ValueError("source batch requires original observation receipts or roundtrip parent sources")
    # Validate parent/observation lineage BEFORE any expensive forward.
    from .geometric_decision_scene_store import source_derivation
    for o, derivation in zip(observations, derivations):
        source_derivation(store, o, derivation, expected_seed=expected_seed, split=split)
    observation_digest = hashlib.sha256(encoded(observations)).hexdigest()
    identity = {"schema": "geometric-decision-feature-batch-v1", "binding": store.binding,
        "split": split, "split_seed": expected_seed, "scene_ids": ids,
        "observation_origin": observation_origin, "observation_sha256": observation_digest}
    features, feature_refs = [], []
    for o in observations:
        check()
        q = np.asarray(o["log_f"], np.float32)
        values = feature_record(o)
        validate_record(values, q)
        ref = fixed_arrays(store, f"{folder}/features/{o['scene_id']:05d}.npz", values)
        restored = store.arrays(ref)
        validate_record(restored, q)
        features.append(restored)
        feature_refs.append(ref)
    feature_ref = store.publish_json(folder+"/features.json", {**identity, "records": feature_refs})
    logits, logit_refs = {}, {}
    for checkpoint in checkpoints:
        check()
        cp = checkpoint["seed"]
        metadata = {"schema": "geometric-decision-logits-v1", "binding": store.binding,
            "features": feature_ref, "checkpoint": checkpoint, "runtime": runtime,
            "observation_sha256": observation_digest}
        receipt_path, payload_path = f"{folder}/logits-{cp}.json", f"{folder}/logits-{cp}.npz"
        if store.path(receipt_path).exists():
            ref = store.reference(store.path(receipt_path))
            record = store.json(ref)
            if set(record) != set(metadata) | {"arrays"} or encoded({k: record[k] for k in metadata}) != encoded(metadata):
                raise ValueError("saved logits have another checkpoint, observation or runtime")
            matrices = unpack_logits(store.arrays(record["arrays"]), observations)
        else:
            if store.path(payload_path).exists():
                raise RuntimeError("unreceipted logits require reconciliation; no re-forward")
            callback_checkpoint = deepcopy(checkpoint)
            matrices = forward(callback_checkpoint, features)
            check()
            if encoded(callback_checkpoint) != encoded(checkpoint):
                raise ValueError("forward callback mutated admitted checkpoint identity")
            for values, feature in zip(features, feature_refs):
                assert_same_arrays(values, store.arrays(feature))
            arrays_ref = fixed_arrays(store, payload_path, pack_logits(matrices, observations))
            matrices = unpack_logits(store.arrays(arrays_ref), observations)
            ref = store.publish_json(receipt_path, {**metadata, "arrays": arrays_ref})
        logits[cp], logit_refs[str(cp)] = matrices, ref
    sources = []
    for i, o in enumerate(observations):
        check()
        scene = scene_from_sources(split, o, features[i], {cp: logits[cp][i] for cp in ge.CHECKPOINTS}, expected_seed=expected_seed)
        sources.append(save_source(store, f"{folder}/scenes/{o['scene_id']:05d}", scene, expected_seed=expected_seed,
            derivation=derivations[i],
            origin={"features": feature_ref, "logits": logit_refs, "observation_origin": observation_origin}))
    return store.publish_json(folder+"/sources.json", {"schema": "geometric-decision-source-batch-v1",
        "binding": store.binding, "features": feature_ref, "logits": logit_refs,
        "split": split, "split_seed": expected_seed, "scene_ids": ids, "sources": sources})


def prepare_inputs(store, folder, sources_ref, *, normalizers, normalization_ref, scale,
                   fit_origin, fit_candidates, check):
    """Fit each scene once for all backbones; preserve complete factors.

    A completed input record takes the validation/reuse branch. An orphan
    factor payload stops before fitting; caller must reconcile its provenance.
    """
    source_batch = store.json(sources_ref)
    if (set(source_batch) != {"schema", "binding", "features", "logits", "split", "split_seed", "scene_ids", "sources"}
            or source_batch["schema"] != "geometric-decision-source-batch-v1" or source_batch["binding"] != store.binding
            or not 1 <= len(source_batch["sources"]) <= 512
            or len(source_batch["sources"]) != len(source_batch["scene_ids"])
            or source_batch["scene_ids"] != sorted(set(source_batch["scene_ids"]))
            or not callable(fit_candidates) or not callable(check)):
        raise ValueError("input preparation requires the preserved source batch")
    outputs = []
    for sid, source_ref in zip(source_batch["scene_ids"], source_batch["sources"]):
        check()
        scene = load_source(store, source_ref)
        if (scene["observation"]["scene_id"] != sid or scene["split"] != source_batch["split"]
                or scene["observation"]["split_seed"] != source_batch["split_seed"]):
            raise ValueError("source batch scene identity differs")
        location = f"{folder}/scenes/{sid:05d}"
        input_path, fit_path = store.path(location+"/inputs.json"), store.path(location+"/fit.json")
        if input_path.exists():
            ref = store.reference(input_path)
            saved = load_inputs(store, ref, normalization_ref=normalization_ref)
            if saved["record"]["source"] != source_ref or saved["record"]["scale"] != scale:
                raise ValueError("saved input source or scale differs")
            if encoded(store.json(saved["record"]["fit"])["origin"]) != encoded(fit_origin):
                raise ValueError("saved input fit runtime differs")
        else:
            if fit_path.exists():
                fit_ref = store.reference(fit_path)
                record = store.json(fit_ref)
                if record["source"] != source_ref or encoded(record["origin"]) != encoded(fit_origin):
                    raise ValueError("saved fit belongs to another source or runtime")
                load_fit(store, fit_ref, scene=scene)
            else:
                if store.path(location+"/factors.json.gz").exists():
                    raise RuntimeError("unreceipted factors require reconciliation; no refit")
                fitted = fit_candidates(scene["q32"], scene["partitions"])
                check()
                fit_ref = save_fit(store, location, source_ref, fitted, origin=fit_origin)
            ref = save_inputs(store, location, source_ref, fit_ref, normalizers,
                              scale=scale, normalization_ref=normalization_ref)
            load_inputs(store, ref, normalization_ref=normalization_ref)
        outputs.append(ref)
    return store.publish_json(folder+"/inputs.json", {"schema": "geometric-decision-input-batch-v1",
        "binding": store.binding, "sources": sources_ref, "scene_ids": source_batch["scene_ids"],
        "normalization": normalization_ref, "scale": scale, "records": outputs})
