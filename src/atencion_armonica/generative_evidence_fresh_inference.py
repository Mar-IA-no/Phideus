"""Recoverable observable preparation and sealed fresh-test predictions.

The caller owns resources and execution authority.  This module authenticates
the frozen interface, never parses sidecars, and publishes a separate seal only
after all 45 learned/intervention outputs have been reopened byte-exactly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import generative_evidence as ge
from . import generative_evidence_cache as cache
from . import generative_evidence_inputs as inputs
from . import generative_evidence_inference as inference
from . import generative_evidence_references as references
from . import generative_evidence_storage as storage
from .generative_evidence_cell import CellArtifacts
from .generative_evidence_fresh_store import CANONICAL, TESTS, FreshObservableStore, validate_observation
from .generative_evidence_model import EvidenceHead
from .generative_evidence_reuse import ROOT, OpenReuse, VerifiedBytes
from .partial_compatibility_cache import feature_record
from .source_artifacts import load_ordered_logits
from .structured_source_data import validate_record
from .structured_source_reader import build_pool


STATUS = "FRESH_PREDICTIONS_COMPLETE_NO_TRUTH_ACCESS"
SEAL_STATUS = "FRESH_PREDICTIONS_SEALED_NO_TRUTH_ACCESS"
GRID = {"beta_count": 257, "gamma_count": 65, "stride": 4}
GPU_RUNTIME_KEYS = {"torch", "numpy", "cuda", "cudnn", "device"}


def _freeze_authority(ref, check):
    if not callable(check):
        raise TypeError("fresh inference requires a resource callback")
    check()
    from .generative_evidence_test_freeze import verify_freeze
    return verify_freeze(ref, check=check)


def _observations(split, freeze_ref, check):
    from .generative_evidence_fresh_data import FreshObservations
    return FreshObservations(split, freeze_ref=freeze_ref, check=check)


def _gpu_runtime(package_runtime, device):
    """Open CUDA identity only on the caller-authorized unfinished path."""
    if device != "cuda:0":
        raise ValueError("the frozen backbone forward requires explicit cuda:0")
    from .structured_source_runner import gpu_runtime
    runtime = gpu_runtime()
    _validated_gpu_runtime(runtime, package_runtime)
    return runtime


def _validated_gpu_runtime(runtime, package_runtime):
    if (not isinstance(runtime, dict) or set(runtime) != GPU_RUNTIME_KEYS
            or runtime.get("device") != "NVIDIA GeForce RTX 3090"
            or not isinstance(runtime.get("cuda"), str) or not runtime["cuda"]
            or type(runtime.get("cudnn")) is not int
            or any(runtime.get(k) != package_runtime.get(k) for k in ("torch", "numpy"))):
        raise ValueError("backbone CUDA runtime differs from its frozen package identity")
    return runtime


def _checkpoint_forward(checkpoint, records, runtime, device):
    if device != "cuda:0":
        raise ValueError("the frozen backbone forward requires explicit cuda:0")
    _validated_gpu_runtime(runtime, runtime)
    from .structured_source_runner import checkpoint_forward
    return checkpoint_forward(checkpoint, records, runtime)


def _root_ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def _is_path(ref, path):
    return isinstance(ref, dict) and ref.get("path") == Path(path).resolve().relative_to(ROOT).as_posix()


def _root_artifact(store, local):
    if not isinstance(local, dict) or set(local) != {"path", "sha256", "bytes"}:
        raise ValueError("invalid local artifact reference")
    path = store.path(local["path"])
    actual = store.reference(path)
    if actual != local:
        raise ValueError("local artifact bytes differ")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": local["sha256"], "bytes": local["bytes"]}


def _local(store, root_ref):
    if (not isinstance(root_ref, dict) or set(root_ref) != {"path", "sha256", "bytes"}
            or type(root_ref["bytes"]) is not int or root_ref["bytes"] < 0):
        raise ValueError("invalid root artifact reference")
    path = (ROOT/root_ref["path"]).resolve()
    if not path.is_relative_to(store.root):
        raise ValueError("root artifact is outside the fresh store")
    local = {"path": path.relative_to(store.root).as_posix(),
             "sha256": root_ref["sha256"], "bytes": root_ref["bytes"]}
    if store.reference(path) != local:
        raise ValueError("root/local artifact reference differs")
    return local


def _same_arrays(first, second):
    return (set(first) == set(second)
            and all(first[k].dtype == second[k].dtype and first[k].shape == second[k].shape
                    and (np.array_equal(first[k], second[k], equal_nan=True)
                         if first[k].dtype.kind in "fc" else np.array_equal(first[k], second[k]))
                    for k in first))


def _fixed_arrays(store, relative, arrays):
    path = store.path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ref = store.reference(path)
        if not _same_arrays(store.arrays(ref), arrays):
            raise ValueError(f"cannot replace completed arrays: {relative}")
        return ref
    storage.write_arrays(path, arrays)
    ref = store.reference(path)
    if not _same_arrays(store.arrays(ref), arrays):
        raise ValueError("stored arrays changed after publication")
    return ref


def _fixed_json(store, relative, value):
    path = store.path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ref = store.reference(path)
        if store.json(ref) != value:
            raise ValueError(f"cannot replace completed JSON: {relative}")
        return ref
    storage.write_json(path, value)
    ref = store.reference(path)
    if store.json(ref) != value:
        raise ValueError("stored JSON changed after publication")
    return ref


def _logit_arrays(matrices, observations):
    from .partial_compatibility_inference import observation_identity
    ids, seeds, fingerprints = observation_identity(observations)
    sizes = np.asarray([len(m) for m in matrices], np.int64)
    expected_sizes = np.asarray([len(o["log_f"]) for o in observations], np.int64)
    if (len(matrices) != len(observations)
            or not np.array_equal(sizes, expected_sizes)
            or any(not isinstance(m, np.ndarray) or m.dtype != np.float32
                   or m.shape != (int(n), int(n)) or not np.isfinite(m).all()
                   or not np.array_equal(m, m.T) for m, n in zip(matrices, sizes))):
        raise ValueError("backbone logits differ from the complete observable roster")
    return {"logits": np.concatenate([m.ravel() for m in matrices]), "sizes": sizes,
        "offsets": np.r_[np.int64(0), np.cumsum(sizes*sizes)], "scene_ids": ids,
        "split_seeds": seeds, "observation_fingerprints": fingerprints}


def _feature_archive(store, split, observations, check):
    refs, records = [], []
    for scene_id, observation in enumerate(observations):
        check()
        q = validate_observation(observation, scene_id, split)
        expected = feature_record(observation)
        validate_record(expected, q)
        local = _fixed_arrays(store, f"{split}/source/features/{scene_id:05d}.npz", expected)
        loaded = store.arrays(local)
        validate_record(loaded, q)
        if not _same_arrays(loaded, expected):
            raise ValueError("preserved feature record differs from current observable features")
        refs.append(_root_artifact(store, local))
        records.append(loaded)
    value = {"schema": "generative-evidence-fresh-features-v1", "binding": store.binding,
        "split": split, "scene_ids": list(range(512)), "records": refs,
        "status": "OBSERVABLE_FEATURES_COMPLETE_NO_TRUTH_ACCESS"}
    return records, _root_ref(store.path(_fixed_json(store, f"{split}/source/features.json", value)["path"]))


def _logit_archives(store, split, observations, records, common, features_ref, device, check):
    checkpoints = common.get("checkpoints") if isinstance(common, dict) else None
    checkpoint_keys = {"arm", "seed", "checkpoint", "checkpoint_sha256", "threshold"}
    if (not isinstance(checkpoints, list) or [c.get("seed") for c in checkpoints] != list(ge.CHECKPOINTS)
            or any(set(c) != checkpoint_keys or c["arm"] != "pairs_descriptors"
                   or c["checkpoint_sha256"] != c["checkpoint"].get("sha256")
                   or c["threshold"] != references.THRESHOLDS.get(c["seed"])
                   for c in checkpoints)):
        raise ValueError("frozen backbone checkpoint roster differs")
    result, refs, preserved_runtime, active_runtime = {}, {}, None, None
    checkpoint_reader = VerifiedBytes(ROOT)
    for checkpoint in checkpoints:
        check()
        cp = checkpoint["seed"]
        checkpoint_reader.read(checkpoint["checkpoint"])
        check()
        payload = store.path(f"{split}/source/logits-{cp}.npz")
        receipt_path = store.path(f"{split}/source/logits-{cp}.json")
        base = {"schema": "generative-evidence-fresh-logits-v1", "binding": store.binding,
            "split": split, "checkpoint_seed": cp, "checkpoint": checkpoint["checkpoint"],
            "features": features_ref,
            "status": "OBSERVABLE_LOGITS_COMPLETE_NO_TRUTH_ACCESS"}
        if receipt_path.exists():
            receipt = VerifiedBytes(ROOT).json(_root_ref(receipt_path))
            runtime = _validated_gpu_runtime(receipt.get("runtime"), common["runtime"])
            if preserved_runtime is not None and runtime != preserved_runtime:
                raise ValueError("fresh checkpoint receipts disagree on CUDA runtime")
            preserved_runtime = runtime
            identity = {**base, "runtime": runtime}
            if (set(receipt) != set(identity) | {"logits"}
                    or any(receipt[k] != v for k, v in identity.items())
                    or not _is_path(receipt["logits"], payload)):
                raise ValueError("preserved logit receipt has another source identity")
            local = _local(store, receipt["logits"])
            matrices = load_ordered_logits(store.path(local["path"]), observations)
        else:
            if payload.exists():
                raise RuntimeError("unreceipted backbone logits are ambiguous; no re-forward")
            if device is None:
                raise RuntimeError("missing backbone logit receipt; verifier never forwards")
            if active_runtime is None:
                active_runtime = _gpu_runtime(common["runtime"], device)
            if preserved_runtime is not None and active_runtime != preserved_runtime:
                raise ValueError("active CUDA runtime differs from recovered checkpoint receipts")
            preserved_runtime = active_runtime
            identity = {**base, "runtime": active_runtime}
            matrices = _checkpoint_forward(checkpoint, records, active_runtime, device)
            check()
            local = _fixed_arrays(store, f"{split}/source/logits-{cp}.npz",
                                  _logit_arrays(matrices, observations))
            receipt = {**identity, "logits": _root_artifact(store, local)}
            _fixed_json(store, f"{split}/source/logits-{cp}.json", receipt)
            matrices = load_ordered_logits(store.path(local["path"]), observations)
        result[cp], refs[str(cp)] = matrices, _root_ref(receipt_path)
    return result, refs


def _scene_from_sources(split, scene_id, observation, feature, matrices):
    q = validate_observation(observation, scene_id, split)
    pools = {str(cp): build_pool(q, matrices[cp][scene_id], feature["pair_support"])
             for cp in ge.CHECKPOINTS}
    inventory = ge.law.candidate_inventory(
        {cp: pool["partitions"] for cp, pool in pools.items()}, len(q))
    partitions = ge.partitions_checked(sorted(ge.law.signature(row["partition"])
        for row in inventory["candidates"] if row["status"] == "SUPPORTED"), len(q))
    order = np.argsort(q, kind="stable")
    return {"observation": observation, "features": feature,
        "logits": {cp: matrices[cp][scene_id] for cp in ge.CHECKPOINTS}, "pools": pools,
        "inventory": inventory, "partitions": partitions, "canonical_to_observed": order,
        "q32": q[order], "status": "ELIGIBLE" if partitions else "NO_OBSERVABLE_CANDIDATE"}


def _source_record_value(store, split, scene_id, scene, features_ref, logit_refs):
    return {"schema": "generative-evidence-fresh-source-scene-v1", "binding": store.binding,
        "split": split, "scene_id": scene_id, "feature": features_ref,
        "logits": logit_refs, "pools": scene["pools"], "inventory": scene["inventory"],
        "partitions": scene["partitions"], "canonical_to_observed": scene["canonical_to_observed"].tolist(),
        "q32": scene["q32"].astype(np.float64).tolist(), "status": scene["status"]}


def _source_record(store, split, scene_id, scene, features_ref, logit_refs):
    value = _source_record_value(store, split, scene_id, scene, features_ref, logit_refs)
    local = _fixed_json(store, f"{split}/source/scenes/{scene_id:05d}.json", value)
    return _root_ref(store.path(local["path"]))


def _normalizers(frozen):
    value = VerifiedBytes(ROOT).json(frozen["manifest"]["normalizers"])
    if set(value) != {"prepared_train", "normalizers"}:
        raise ValueError("freeze normalizer wrapper differs")
    return value["normalizers"]


def _delivered_inputs(store, split, observable_ref, frozen, check):
    check()
    index, _ = store._index(split)
    norms = _normalizers(frozen)
    result = {}
    for cp in ge.CHECKPOINTS:
        check()
        raw_local = index["raw"][str(cp)]
        raw_root = _root_artifact(store, raw_local)
        raw = store.arrays(raw_local)
        unpacked = cache.unpack_rows(raw, binding=store.binding, split=split, checkpoint_seed=cp)
        packed = inputs.pack_inputs(raw, norms["common"][str(cp)], norms["evidence"],
            binding=store.binding, split=split, checkpoint_seed=cp, raw_ref=raw_root,
            normalizer_ref=frozen["manifest"]["normalizers"])
        local = _fixed_arrays(store, f"{split}/inputs/cp_{cp}.npz", packed)
        delivered = inputs.unpack_inputs(store.arrays(local), binding=store.binding, split=split,
            checkpoint_seed=cp, raw_ref=raw_root, normalizer_ref=frozen["manifest"]["normalizers"],
            identities=unpacked["identities"])
        value = {"schema": "generative-evidence-fresh-delivered-v1", "binding": store.binding,
            "split": split, "checkpoint_seed": cp, "scene_ids": list(range(512)),
            "identities": unpacked["identities"], "raw": raw_root,
            "normalizers": frozen["manifest"]["normalizers"], "inputs": _root_artifact(store, local),
            "sham": delivered["sham"], "observables": observable_ref,
            "status": "TRAIN_NORMALIZED_INPUTS_NO_TRUTH_ACCESS"}
        receipt = _fixed_json(store, f"{split}/inputs/cp_{cp}.json", value)
        result[str(cp)] = _root_ref(store.path(receipt["path"]))
        del raw, packed, delivered
    return result


def _selected(frozen):
    rows = frozen["manifest"]["selected_states"]
    if not isinstance(rows, list) or len(rows) != 27:
        raise ValueError("freeze lacks its 27 selected states")
    result = {}
    for row in rows:
        cell = row.get("cell") if isinstance(row, dict) else None
        if (not isinstance(cell, dict) or set(cell) != {"arm", "checkpoint_seed", "reader_seed", "cell_id"}
                or set(row) != {"cell", "epoch", "complete", "state"}):
            raise ValueError("selected state roster schema differs")
        key = cell["arm"], cell["checkpoint_seed"], cell["reader_seed"]
        if key in result:
            raise ValueError("duplicate selected state identity")
        result[key] = row
    return result


def _roster(frozen):
    selected = _selected(frozen)
    raw = frozen["manifest"]["prediction_roster"]
    freeze_rows = []
    for i, row in enumerate(raw):
        intervention = "original" if row.get("kind") == "original" else row.get("intervention")
        key = row.get("arm"), row.get("checkpoint_seed"), row.get("reader_seed"), intervention
        freeze_rows.append((key, i, row))
    if len(freeze_rows) != 45 or len({k for k, _, _ in freeze_rows}) != 45:
        raise ValueError("freeze prediction roster is not 45 unique outputs")
    lookup = {k: (i, row) for k, i, row in freeze_rows}
    result = []
    for inference_index, row in enumerate(inference.inference_roster()):
        key = row["arm"], row["checkpoint_seed"], row["reader_seed"], row["intervention"]
        if key not in lookup or key[:3] not in selected:
            raise ValueError("kernel and freeze prediction rosters differ")
        freeze_index, _ = lookup[key]
        state = selected[key[:3]]
        result.append({**row, "cell_id": state["cell"]["cell_id"], "state": state["state"],
                       "freeze_index": freeze_index, "inference_index": inference_index})
    if {tuple(r[k] for k in ("arm", "checkpoint_seed", "reader_seed", "intervention")) for r in result} != set(lookup):
        raise ValueError("kernel roster does not cover the complete frozen roster")
    return result, selected


def _model(selected, device):
    complete = VerifiedBytes(ROOT).json(selected["complete"])
    root = ROOT/selected["complete"]["path"]
    store = CellArtifacts(root.parent, binding=complete["binding"])
    local = complete["snapshots"][selected["epoch"]]
    if _root_ref(store.path(local["path"])) != selected["state"]:
        raise ValueError("selected state no longer names its frozen snapshot")
    state = store.load_state(local)
    cell = selected["cell"]
    if (state["epoch"] != selected["epoch"] or state["next_batch"] != 0
            or any(state[k] != cell[k] for k in ("arm", "checkpoint_seed", "reader_seed"))):
        raise ValueError("selected state payload identity differs")
    model = EvidenceHead(cell["reader_seed"])
    model.load_state_dict(state["model"], strict=True)
    return model.to(device)


def _prediction_arrays(store, ref, rows):
    arrays = store.arrays(_local(store, ref))
    counts = [len(row["globals"]) for row in rows]
    offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    if (set(arrays) != {"components", "offsets"} or arrays["components"].dtype != np.float32
            or arrays["components"].shape != (int(offsets[-1]), 2)
            or not np.isfinite(arrays["components"]).all() or np.any(arrays["components"] < 0)
            or arrays["offsets"].dtype != np.int64 or not np.array_equal(arrays["offsets"], offsets)):
        raise ValueError("preserved prediction payload differs from its 512-scene candidate roster")
    return arrays


def _intervention_rows(delivered, row):
    if row["intervention"] == "original":
        if row["arm"] not in ge.ARMS:
            raise ValueError("original prediction has an unknown arm")
        return delivered["inputs"][row["arm"]]
    if row["arm"] != "generative":
        raise ValueError("only the Generativa head has intervention outputs")
    return inference.intervention_inputs(delivered["inputs"], row["intervention"])


def _prediction_paths(store, split, row):
    slug = (f"cp_{row['checkpoint_seed']}/seed_{row['reader_seed']}/"
            f"{row['arm']}-{row['intervention']}")
    return (store.path(f"{split}/predictions/{slug}.npz"),
            store.path(f"{split}/predictions/{slug}.json"), slug)


def _prediction_record(store, split, row, delivered_ref, delivered, model, check):
    payload_path, record_path, slug = _prediction_paths(store, split, row)
    base = {"schema": "generative-evidence-fresh-prediction-v1", "binding": store.binding,
        "status": "PREDICTION_COMPLETE_NO_TRUTH_ACCESS", "split": split,
        "split_seed": cache.SPLITS[split][1], "scene_ids": list(range(512)),
        **{k: row[k] for k in ("arm", "checkpoint_seed", "reader_seed", "cell_id", "intervention",
                               "freeze_index", "inference_index", "state")},
        "delivered": delivered_ref, "truth_access": False}
    arm_rows = _intervention_rows(delivered, row)
    if record_path.exists():
        record = VerifiedBytes(ROOT).json(_root_ref(record_path))
        if set(record) != set(base) | {"prediction"} or any(record[k] != v for k, v in base.items()):
            raise ValueError("prediction receipt identity differs")
        _prediction_arrays(store, record["prediction"], arm_rows)
    else:
        if payload_path.exists():
            raise RuntimeError("unreceipted prediction is ambiguous; no re-forward")
        if model is None:
            raise RuntimeError("missing selected head for an unfinished prediction")
        check()
        predicted = inference.predict(model, arm_rows, device=str(next(model.parameters()).device), check=check)
        local = _fixed_arrays(store, f"{split}/predictions/{slug}.npz", predicted)
        record = {**base, "prediction": _root_artifact(store, local)}
        _fixed_json(store, f"{split}/predictions/{slug}.json", record)
        _prediction_arrays(store, record["prediction"], arm_rows)
    return _root_ref(record_path), record


def _choices(store, split, records):
    value = {"schema": "generative-evidence-fresh-observable-choices-v1", "binding": store.binding,
        "status": "OBSERVABLE_CHOICES_COMPLETE_NO_TRUTH_ACCESS", "split": split,
        "split_seed": cache.SPLITS[split][1], "scene_ids": list(range(512)), "records": records,
        "truth_access": False}
    local = _fixed_json(store, f"{split}/choices.json", value)
    return _root_ref(store.path(local["path"])), value


def _seal_verified_index(store, split, index_value, frozen, check, root):
    """Publish the distinct seal only after the canonical index fully reopens."""
    index_ref = _root_ref(store.path(f"{split}/predictions.json"))
    preseal = _verify(split, freeze_ref=frozen["freeze"], check=check, root=root,
                      require_seal=False)
    if preseal["seal"] is not None or preseal["index"] != index_value:
        raise ValueError("fresh prediction index changed before sealing")
    seal_value = {"schema": "generative-evidence-fresh-prediction-seal-v1", "binding": store.binding,
        "status": SEAL_STATUS, "split": split, "prediction_index": index_ref,
        "prediction_count": 45, "truth_access": False}
    seal_local = _fixed_json(store, f"{split}/prediction_seal.json", seal_value)
    result = _verify(split, freeze_ref=frozen["freeze"], check=check, root=root)
    if result["seal"] != _root_ref(store.path(seal_local["path"])):
        raise ValueError("fresh prediction seal changed after publication")
    return result["seal"]


def _run(split, *, freeze_ref, device, check, progress, root):
    if split not in TESTS or device not in ("cpu", "cuda:0") or not callable(progress):
        raise ValueError("fresh inference requires a fixed test split, explicit backend and progress callback")
    frozen = _freeze_authority(freeze_ref, check)
    source = _observations(split, frozen["freeze"], check)
    observations = []
    for scene_id in range(512):
        check()
        observations.append(source.observation(scene_id))
    store = FreshObservableStore(root, binding={"test_freeze": frozen["freeze"]})
    records, features_ref = _feature_archive(store, split, observations, check)
    reuse = OpenReuse()
    for key in ("python", "numpy", "torch"):
        if reuse.common["runtime"][key] != frozen["manifest"]["runtime"][key]:
            raise ValueError("backbone and selected-head package runtimes differ")
    matrices, logit_refs = _logit_archives(store, split, observations, records, reuse.common,
                                           features_ref, device, check)
    source_refs, choice_records, fitter = [], [], None
    for scene_id, (observation, feature) in enumerate(zip(observations, records)):
        check()
        scene = _scene_from_sources(split, scene_id, observation, feature, matrices)
        source_ref = _source_record(store, split, scene_id, scene, features_ref, logit_refs)
        fit_path = store.path(f"{split}/{scene_id:05d}/fit.json")
        scene_folder = fit_path.parent
        if (not fit_path.exists()
                and (list(scene_folder.glob("factors-*.json.gz")) if scene_folder.exists() else [])):
            raise RuntimeError("unreceipted fresh fit is ambiguous; no refit")
        observable_path = scene_folder/"observable.json"
        if (not observable_path.exists() and scene_folder.exists()
                and list(scene_folder.glob("raw-*.npz"))):
            raise RuntimeError("unreceipted fresh observable arrays are ambiguous")
        origin = {"kind": "NEW_FRESH_FIT", "device": device, "test_freeze": frozen["freeze"],
                  "source": source_ref, "grid": GRID}
        reused_fit = fit_path.exists()
        if reused_fit:
            fit, fit_ref = store.load_fit(split, scene_id)
            if store.json(fit_ref)["origin"] != origin or fit["inventory"] != scene["inventory"]:
                raise ValueError("completed fresh fit has another source or backend")
            fitted = {k: fit[k] for k in ("fits", "group_factors")}
        else:
            if fitter is None:
                fitter = ge.law.GroupFitter(ge.law.Grid(257, 65, 4),
                                            device="cuda" if device == "cuda:0" else "cpu")
            fitted = ge.law.fit_candidates(scene["q32"], scene["partitions"], fitter)
        store.save_fit(split, scene_id, scene, fitted, origin=origin)
        check()
        store.save_observables(split, scene_id, scene)
        source_refs.append(source_ref)
        fit, fit_ref = store.load_fit(split, scene_id)
        choice = references.observable_choices(scene["partitions"], fit["fits"], scene["logits"],
                                                scene["canonical_to_observed"])
        identity = store.load_row(split, scene_id, ge.CHECKPOINTS[0])["identities"][0]
        choice_records.append({"scene_id": scene_id, "identity": identity, "source": source_ref,
            "fit": _root_ref(store.path(fit_ref["path"])), "choice": choice})
        progress(json.dumps({"stage": "fresh_observable", "split": split, "scene_id": scene_id,
                             "candidates": len(scene["partitions"]), "recovered_fit": reused_fit}))
    source_value = {"schema": "generative-evidence-fresh-source-v1", "binding": store.binding,
        "status": "OBSERVABLE_SOURCE_COMPLETE_NO_TRUTH_ACCESS", "split": split,
        "split_seed": cache.SPLITS[split][1], "scene_ids": list(range(512)),
        "features": features_ref, "logits": logit_refs, "records": source_refs}
    source_local = _fixed_json(store, f"{split}/source/index.json", source_value)
    source_root = _root_ref(store.path(source_local["path"]))
    observable_local = store.seal_observables(split, check=check)
    observable_root = _root_ref(store.path(observable_local["path"]))
    delivered_refs = _delivered_inputs(store, split, observable_root, frozen, check)
    choices_ref, _ = _choices(store, split, choice_records)
    roster, selected = _roster(frozen)
    prediction_refs = []
    current, model, delivered = None, None, None
    for row in roster:
        key = row["arm"], row["checkpoint_seed"], row["reader_seed"]
        if key != current:
            if model is not None:
                del model
            receipt = VerifiedBytes(ROOT).json(delivered_refs[str(row["checkpoint_seed"])])
            arrays = store.arrays(_local(store, receipt["inputs"]))
            delivered = inputs.unpack_inputs(arrays, binding=store.binding, split=split,
                checkpoint_seed=row["checkpoint_seed"], raw_ref=receipt["raw"],
                normalizer_ref=receipt["normalizers"], identities=receipt["identities"])
            model = None
            current = key
        if not _prediction_paths(store, split, row)[1].exists() and model is None:
            check()
            model = _model(selected[key], device)
            check()
        ref, _ = _prediction_record(store, split, row,
            delivered_refs[str(row["checkpoint_seed"])], delivered, model, check)
        prediction_refs.append(ref)
    if len(prediction_refs) != 45:
        raise ValueError("fresh inference did not preserve all 45 outputs")
    index_value = {"schema": "generative-evidence-fresh-prediction-index-v1", "binding": store.binding,
        "status": STATUS, "split": split, "split_seed": cache.SPLITS[split][1],
        "scene_ids": list(range(512)), "source": source_root, "observables": observable_root,
        "delivered": delivered_refs, "choices": choices_ref, "roster": roster,
        "predictions": prediction_refs, "prediction_count": 45, "truth_access": False}
    _fixed_json(store, f"{split}/predictions.json", index_value)
    # Do not overlap the producer's large observations/logits/input objects with
    # the independent reopening pass under the worker's fixed memory cap.
    del model, delivered, matrices, records, observations, source_refs, choice_records, fitter
    del source, reuse
    del scene, fit, choice, feature, observation, fitted, receipt, arrays
    # A killed or invalid run cannot expose a premature truth gate.
    return _seal_verified_index(store, split, index_value, frozen, check, root)


def _verify_sources_and_choices(store, split, index, choices, frozen, check):
    source = VerifiedBytes(ROOT).json(index["source"])
    source_keys = {"schema", "binding", "status", "split", "split_seed", "scene_ids",
                   "features", "logits", "records"}
    if (set(source) != source_keys or source["schema"] != "generative-evidence-fresh-source-v1"
            or source["binding"] != store.binding
            or source["status"] != "OBSERVABLE_SOURCE_COMPLETE_NO_TRUTH_ACCESS"
            or source["split"] != split or source["split_seed"] != cache.SPLITS[split][1]
            or source["scene_ids"] != list(range(512)) or len(source["records"]) != 512
            or set(source["logits"]) != {str(cp) for cp in ge.CHECKPOINTS}
            or not _is_path(index["source"], store.path(f"{split}/source/index.json"))
            or not _is_path(source["features"], store.path(f"{split}/source/features.json"))
            or any(not _is_path(source["logits"][str(cp)],
                                store.path(f"{split}/source/logits-{cp}.json"))
                   for cp in ge.CHECKPOINTS)
            or any(not _is_path(ref, store.path(f"{split}/source/scenes/{i:05d}.json"))
                   for i, ref in enumerate(source["records"]))):
        raise ValueError("fresh observable source index differs")
    observed = _observations(split, frozen["freeze"], check)
    observations = []
    for scene_id in range(512):
        check()
        observations.append(observed.observation(scene_id))
    feature_index = VerifiedBytes(ROOT).json(source["features"])
    feature_keys = {"schema", "binding", "split", "scene_ids", "records", "status"}
    if (set(feature_index) != feature_keys
            or feature_index["schema"] != "generative-evidence-fresh-features-v1"
            or feature_index["binding"] != store.binding or feature_index["split"] != split
            or feature_index["scene_ids"] != list(range(512))
            or feature_index["status"] != "OBSERVABLE_FEATURES_COMPLETE_NO_TRUTH_ACCESS"
            or not isinstance(feature_index["records"], list) or len(feature_index["records"]) != 512
            or any(not _is_path(ref, store.path(f"{split}/source/features/{i:05d}.npz"))
                   for i, ref in enumerate(feature_index["records"]))):
        raise ValueError("fresh feature index differs")
    records = []
    for scene_id, (observation, ref) in enumerate(zip(observations, feature_index["records"])):
        check()
        q = validate_observation(observation, scene_id, split)
        record = store.arrays(_local(store, ref))
        expected = feature_record(observation)
        validate_record(record, q)
        if not _same_arrays(record, expected):
            raise ValueError("preserved feature differs from current observable feature")
        records.append(record)
    reuse = OpenReuse()
    for key in ("python", "numpy", "torch"):
        if reuse.common["runtime"][key] != frozen["manifest"]["runtime"][key]:
            raise ValueError("backbone and selected-head package runtimes differ")
    matrices, logit_refs = _logit_archives(store, split, observations, records, reuse.common,
                                           source["features"], None, check)
    if logit_refs != source["logits"]:
        raise ValueError("fresh source index lost a backbone logit receipt")
    for scene_id, (observation, feature, source_ref, choice_record) in enumerate(
            zip(observations, records, source["records"], choices["records"])):
        check()
        scene = _scene_from_sources(split, scene_id, observation, feature, matrices)
        if VerifiedBytes(ROOT).json(source_ref) != _source_record_value(
                store, split, scene_id, scene, source["features"], source["logits"]):
            raise ValueError("fresh source scene or candidate pool differs")
        fit, fit_ref = store.load_fit(split, scene_id)
        if fit["observation"] != observation or fit["inventory"] != scene["inventory"]:
            raise ValueError("fresh fit differs from its observable source")
        fit_receipt = store.json(fit_ref)
        origin = fit_receipt.get("origin")
        if (not isinstance(origin, dict)
                or set(origin) != {"kind", "device", "test_freeze", "source", "grid"}
                or origin["kind"] != "NEW_FRESH_FIT"
                or origin["device"] not in ("cpu", "cuda:0")
                or origin["test_freeze"] != frozen["freeze"]
                or origin["source"] != source_ref or origin["grid"] != GRID):
            raise ValueError("fresh fit lost its frozen source or fixed grid provenance")
        choice = references.observable_choices(scene["partitions"], fit["fits"], scene["logits"],
                                                scene["canonical_to_observed"])
        identity = store.load_row(split, scene_id, ge.CHECKPOINTS[0])["identities"][0]
        wanted = {"scene_id": scene_id, "identity": identity, "source": source_ref,
                  "fit": _root_ref(store.path(fit_ref["path"])), "choice": choice}
        if choice_record != wanted:
            raise ValueError("observable choice differs from its preserved source and fit")


def _verify(split, *, freeze_ref, check, root, require_seal=True):
    if split not in TESTS:
        raise ValueError("unknown fixed fresh split")
    frozen = _freeze_authority(freeze_ref, check)
    root = Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError("fresh prediction store does not exist for verification")
    store = FreshObservableStore(root, binding={"test_freeze": frozen["freeze"]})
    if type(require_seal) is not bool:
        raise TypeError("seal verification mode must be explicit")
    canonical_index = store.path(f"{split}/predictions.json")
    if require_seal:
        seal_ref = _root_ref(store.path(f"{split}/prediction_seal.json"))
        seal = VerifiedBytes(ROOT).json(seal_ref)
        if (set(seal) != {"schema", "binding", "status", "split", "prediction_index",
                          "prediction_count", "truth_access"}
                or seal["schema"] != "generative-evidence-fresh-prediction-seal-v1"
                or seal["binding"] != store.binding or seal["status"] != SEAL_STATUS
                or seal["split"] != split or seal["prediction_count"] != 45
                or seal["truth_access"] is not False
                or not _is_path(seal["prediction_index"], canonical_index)):
            raise ValueError("fresh prediction seal schema or authority differs")
        index_ref = seal["prediction_index"]
    else:
        seal_ref, seal = None, None
        index_ref = _root_ref(canonical_index)
    index = VerifiedBytes(ROOT).json(index_ref)
    keys = {"schema", "binding", "status", "split", "split_seed", "scene_ids", "source",
            "observables", "delivered", "choices", "roster", "predictions",
            "prediction_count", "truth_access"}
    roster, _ = _roster(frozen)
    if (set(index) != keys or index["schema"] != "generative-evidence-fresh-prediction-index-v1"
            or index["binding"] != store.binding or index["status"] != STATUS
            or index["split"] != split or index["split_seed"] != cache.SPLITS[split][1]
            or index["scene_ids"] != list(range(512)) or index["roster"] != roster
            or set(index["delivered"]) != {str(cp) for cp in ge.CHECKPOINTS}
            or not isinstance(index["predictions"], list) or len(index["predictions"]) != 45
            or index["prediction_count"] != 45 or index["truth_access"] is not False
            or not _is_path(index_ref, canonical_index)
            or not _is_path(index["observables"], store.path(f"{split}/observables.json"))
            or not _is_path(index["choices"], store.path(f"{split}/choices.json"))
            or any(not _is_path(index["delivered"][str(cp)],
                                store.path(f"{split}/inputs/cp_{cp}.json"))
                   for cp in ge.CHECKPOINTS)
            or any(not _is_path(ref, _prediction_paths(store, split, row)[1])
                   for row, ref in zip(roster, index["predictions"]))):
        raise ValueError("fresh prediction index differs from its frozen roster")
    choices = VerifiedBytes(ROOT).json(index["choices"])
    choice_keys = {"schema", "binding", "status", "split", "split_seed", "scene_ids",
                   "records", "truth_access"}
    if (set(choices) != choice_keys
            or choices["schema"] != "generative-evidence-fresh-observable-choices-v1"
            or choices["binding"] != store.binding
            or choices["status"] != "OBSERVABLE_CHOICES_COMPLETE_NO_TRUTH_ACCESS"
            or choices["split"] != split or choices["split_seed"] != cache.SPLITS[split][1]
            or choices["scene_ids"] != list(range(512))
            or not isinstance(choices["records"], list) or len(choices["records"]) != 512
            or choices["truth_access"] is not False):
        raise ValueError("observable choice roster differs")
    _verify_sources_and_choices(store, split, index, choices, frozen, check)
    observable_value, observable_local = store._index(split)
    if _root_ref(store.path(observable_local["path"])) != index["observables"]:
        raise ValueError("prediction index lost its complete observable aggregate")
    delivered_receipts, contents = {}, []
    delivered_keys = {"schema", "binding", "split", "checkpoint_seed", "scene_ids",
                      "identities", "raw", "normalizers", "inputs", "sham",
                      "observables", "status"}
    for cp in ge.CHECKPOINTS:
        value = VerifiedBytes(ROOT).json(index["delivered"][str(cp)])
        if (set(value) != delivered_keys
                or value["schema"] != "generative-evidence-fresh-delivered-v1"
                or value["binding"] != store.binding or value["split"] != split
                or value["checkpoint_seed"] != cp or value["scene_ids"] != list(range(512))
                or not isinstance(value["identities"], list) or len(value["identities"]) != 512
                or not isinstance(value["sham"], list) or len(value["sham"]) != 512
                or value["normalizers"] != frozen["manifest"]["normalizers"]
                or value["status"] != "TRAIN_NORMALIZED_INPUTS_NO_TRUTH_ACCESS"
                or not _is_path(value["inputs"], store.path(f"{split}/inputs/cp_{cp}.npz"))
                or value["raw"] != _root_artifact(store, observable_value["raw"][str(cp)])):
            raise ValueError("fresh delivered input receipt differs")
        if value.get("observables") != index["observables"]:
            raise ValueError("fresh delivered input lost its observable aggregate")
        delivered_receipts[cp] = value
    current_cp, delivered = None, None
    prediction_keys = {"schema", "binding", "status", "split", "split_seed", "scene_ids",
                       "arm", "checkpoint_seed", "reader_seed", "cell_id", "intervention",
                       "freeze_index", "inference_index", "state", "delivered", "truth_access",
                       "prediction"}
    for row, ref in zip(roster, index["predictions"]):
        check()
        value = VerifiedBytes(ROOT).json(ref)
        identity = {k: row[k] for k in ("arm", "checkpoint_seed", "reader_seed", "cell_id",
                                         "intervention", "freeze_index", "inference_index", "state")}
        if (set(value) != prediction_keys
                or value["schema"] != "generative-evidence-fresh-prediction-v1"
                or value["binding"] != store.binding
                or value["status"] != "PREDICTION_COMPLETE_NO_TRUTH_ACCESS"
                or value["split"] != split or value["split_seed"] != cache.SPLITS[split][1]
                or value["scene_ids"] != list(range(512)) or value["truth_access"] is not False
                or any(value[k] != v for k, v in identity.items())
                or value["delivered"] != index["delivered"][str(row["checkpoint_seed"])]
                or not _is_path(value["prediction"], _prediction_paths(store, split, row)[0])):
            raise ValueError("fresh prediction record identity differs")
        if row["checkpoint_seed"] != current_cp:
            receipt = delivered_receipts[row["checkpoint_seed"]]
            arrays = store.arrays(_local(store, receipt["inputs"]))
            delivered = inputs.unpack_inputs(arrays, binding=store.binding, split=split,
                checkpoint_seed=row["checkpoint_seed"], raw_ref=receipt["raw"],
                normalizer_ref=receipt["normalizers"], identities=receipt["identities"])
            if delivered["sham"] != receipt["sham"]:
                raise ValueError("preserved sham attribution differs")
            current_cp = row["checkpoint_seed"]
        rows = _intervention_rows(delivered, row)
        _prediction_arrays(store, value["prediction"], rows)
        contents.append(value)
    if require_seal and seal["prediction_index"] != _root_ref(canonical_index):
        raise ValueError("prediction seal does not bind the canonical index bytes")
    return {"seal": seal_ref, "index": index, "choices": choices, "records": contents}


def prepare_and_predict(split, *, freeze_ref, device, check, progress=print):
    return _run(split, freeze_ref=freeze_ref, device=device, check=check,
                progress=progress, root=CANONICAL)


def verify_predictions(split, *, freeze_ref, check):
    return _verify(split, freeze_ref=freeze_ref, check=check, root=CANONICAL)


def read_prediction(verified, record):
    """Read one NPZ only after membership in a complete verified seal result."""
    if (not isinstance(verified, dict) or set(verified) != {"seal", "index", "choices", "records"}
            or not isinstance(verified["records"], list) or record not in verified["records"]):
        raise ValueError("prediction read requires membership in a verified complete seal")
    try:
        seal = VerifiedBytes(ROOT).json(verified["seal"])
        index = VerifiedBytes(ROOT).json(seal["prediction_index"])
    except (KeyError, TypeError, ValueError, FileNotFoundError):
        raise ValueError("prediction read requires an authenticated complete seal") from None
    if (seal.get("status") != SEAL_STATUS or index != verified["index"]
            or index.get("status") != STATUS):
        raise ValueError("prediction read requires an authenticated complete seal")
    ref = record["prediction"]
    return storage.read_arrays(ROOT/ref["path"], {"sha256": ref["sha256"], "bytes": ref["bytes"]})


__all__ = ["prepare_and_predict", "verify_predictions", "read_prediction", "STATUS", "SEAL_STATUS"]
