"""Bounded mechanical workloads; no sampler, new tests or campaign admission."""
from __future__ import annotations

import time

import numpy as np
import torch

from .generative_evidence import CHECKPOINTS
from .generative_evidence_cell import state_digest
from .generative_evidence_normalization import arrays_digest
from .geometric_decision_core import READER_SEEDS, route_inputs
from .geometric_decision_model import collate
from .geometric_decision_training import TrainingKernel, collate_targets, epoch_batches
from . import observable_source_rivals as law


def assert_same_arrays(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("array names differ")
    for key in actual:
        a, b = actual[key], expected[key]
        if (a.dtype.str != b.dtype.str or a.shape != b.shape
                or a.tobytes(order="C") != b.tobytes(order="C")):
            raise ValueError(f"array dtype, shape or bytes differ: {key}")
    return arrays_digest(actual)


def extract_first_batch(preparation, complete_ref, *, check):
    """Read authenticated TRAIN only, without recomputing its scale or fits."""
    complete = preparation.completion(complete_ref)
    source = preparation.source
    scale = preparation.store.json(complete["scale"])
    eligible = source.corpus.norm["eligible_scene_ids"]
    ids = epoch_batches(READER_SEEDS[0], 0, eligible)[0].tolist()
    cp = CHECKPOINTS[0]
    entries = {(e["split"], e["checkpoint_seed"], e["shard"]): e["index"] for e in complete["entries"]}
    found, refs = {}, []
    for shard in sorted({i//512 for i in ids}):
        check()
        obs = source.observable_shard("train", shard, check=check)
        interface = source.interface_shard(obs, scale)[cp]
        entry = entries["train", cp, shard]
        preparation._verify_entry(entry, preparation._identity(obs, cp, complete["scale"], interface),
                                  preparation._arrays(interface))
        # This explicit port is OPEN TRAIN supervision, never fitter input.
        targets = source.supervision_shard(obs)
        refs.append({"shard": shard, "index": entry, "targets": obs["targets_ref"]})
        for i in ids:
            if i//512 != shard:
                continue
            offset = i-shard*512
            if obs["scene_ids"][offset] != i:
                raise ValueError("source shard scene order differs")
            q = np.asarray(obs["observations"][offset]["log_f"], np.float32)
            q = q[np.argsort(q, kind="stable")]
            law.observable_q32(q)
            found[i] = {"scene_id": i, "identity": obs["identities"][offset],
                "inputs": {k: v.copy() for k, v in interface[offset]["inputs"].items()},
                "targets": targets[offset]["targets"].copy(), "q32": q,
                "partitions": obs["raw"][cp][offset]["partitions"]}
    check()
    rows = [found[i] for i in ids]
    batch = collate([r["inputs"] for r in rows])
    collate_targets([r["targets"] for r in rows], batch["candidate_mask"])
    return rows, {"purpose": "FIRST_TRAIN_BATCH_PROFILE_NOT_CAMPAIGN", "split": "train",
        "checkpoint_seed": cp, "reader_seed": READER_SEEDS[0], "epoch": 0,
        "scene_ids": ids, "eligible_train_count": len(eligible),
        "prepared_binding": preparation.store.binding, "complete": complete_ref,
        "scale": complete["scale"], "shards": refs}


def save_batch(store, rows, provenance):
    if [r["scene_id"] for r in rows] != provenance["scene_ids"] or len(rows) != 32:
        raise ValueError("profile must retain the exact first batch")
    entries = []
    for i, row in enumerate(rows):
        arrays = {**row["inputs"], "targets": row["targets"], "q32": row["q32"]}
        ref = store.publish_arrays(f"rows/{i:02d}.npz", arrays)
        entries.append({"scene_id": row["scene_id"], "identity": row["identity"],
                        "partitions": row["partitions"], "arrays": ref})
    return store.publish_json("batch.json", {"schema": "geometric-decision-profile-batch-v1",
        "binding": store.binding, "provenance": provenance, "rows": entries})


def read_batch(store, ref):
    record = store.json(ref)
    if (record["schema"] != "geometric-decision-profile-batch-v1" or record["binding"] != store.binding
            or len(record["rows"]) != 32
            or [r["scene_id"] for r in record["rows"]] != record["provenance"]["scene_ids"]):
        raise ValueError("profile batch binding or order differs")
    rows = []
    for row in record["rows"]:
        arrays = store.arrays(row["arrays"])
        if set(arrays) != {"groups", "globals", "evidence", "incidence", "targets", "q32"}:
            raise ValueError("profile array schema differs")
        q = arrays.pop("q32")
        if q.dtype != np.float32:
            raise ValueError("profile observation must retain q32")
        law.observable_q32(q)
        ps = [law.validate_partition(p, len(q)) for p in row["partitions"]]
        if ps != sorted(set(ps)) or len(ps) != len(arrays["globals"]):
            raise ValueError("profile candidate order or extent differs")
        targets = arrays.pop("targets")
        rows.append({"scene_id": row["scene_id"], "identity": row["identity"], "q32": q,
                     "partitions": ps, "inputs": arrays, "targets": targets})
    batch = collate([r["inputs"] for r in rows])
    collate_targets([r["targets"] for r in rows], batch["candidate_mask"])
    return rows, record["provenance"]


def envelope_rows():
    """Dense shape envelope, explicitly not a realizable scene or partition."""
    e = np.sin(np.arange(82*8, dtype=np.float32)).reshape(82, 8)
    e[:, 6:] = np.abs(e[:, 6:])
    inputs = {"groups": np.sin(np.arange(328*9, dtype=np.float32)).reshape(328, 9),
              "globals": np.cos(np.arange(82*17, dtype=np.float32)).reshape(82, 17),
              "evidence": e, "incidence": np.full((82, 328), 1/328, np.float32)}
    target = np.linspace(0, 1, 164, dtype=np.float32).reshape(82, 2)
    return [{"inputs": {k: v.copy() for k, v in inputs.items()}, "targets": target.copy()} for _ in range(32)]


def head_case(store, rows, objective, device, *, check, clock=time.monotonic):
    """25 real-kernel mechanical updates plus exact snapshot recovery replay."""
    if objective not in ("mse", "decision") or len(rows) != 32:
        raise ValueError("fixed objective and 32 profile rows required")
    def sync():
        if device == "cuda:0":
            torch.cuda.synchronize(0)
    started = clock()
    kernel = TrainingKernel(f"geometric_{objective}", CHECKPOINTS[0], READER_SEEDS[0],
        binding=store.binding, scene_ids=list(range(32)), device=device)
    initial = store.save_state(kernel, None)
    setup = clock()-started
    updates, snapshots, previous, middle = [], [], initial, None
    def step():
        ids = kernel.expected_scene_ids()
        batch = collate([route_inputs(rows[i]["inputs"], "geometric") for i in ids])
        target = collate_targets([rows[i]["targets"] for i in ids], batch["candidate_mask"])
        kernel.step(batch, target, ids)
        sync()
    for i in range(25):
        check()
        before = clock()
        step()
        updates.append(clock()-before)
        if i in (9, 24):
            before = clock()
            previous = store.save_state(kernel, previous)
            snapshots.append(clock()-before)
            if i == 9:
                middle = previous
    expected_digest = state_digest(kernel.state())
    before = clock()
    kernel.restore(store.load_state(middle))
    for _ in range(15):
        check()
        step()
    actual_digest = state_digest(kernel.state())
    if actual_digest != expected_digest:
        raise ValueError("profile exact snapshot recovery failed")
    replay_seconds = clock()-before
    evaluation, outputs = [], []
    for i in range(3):
        check()
        before = clock()
        kernel.model.eval()
        batch = {k: v.to(device) for k, v in collate([route_inputs(r["inputs"], "geometric") for r in rows]).items()}
        with torch.no_grad():
            pred = kernel.model(batch).cpu().numpy()
        mask = batch["candidate_mask"].cpu().numpy()
        ref = store.publish_arrays(f"evaluation/{i}.npz", {"components": pred, "candidate_mask": mask})
        restored = store.arrays(ref)
        np.testing.assert_array_equal(restored["components"], pred)
        np.testing.assert_array_equal(restored["candidate_mask"], mask)
        sync()
        evaluation.append(clock()-before)
        outputs.append(ref)
    check()
    result = {"status": "MECHANICAL_PROFILE_NOT_TRAINED_CELL", "binding": store.binding,
        "objective": objective, "device": device, "mechanical_scene_ids": list(range(32)),
        "candidate_counts": [len(r["inputs"]["globals"]) for r in rows],
        "group_counts": [len(r["inputs"]["groups"]) for r in rows],
        "setup_seconds": setup, "all_update_seconds": updates,
        "snapshot_io_seconds": snapshots, "evaluation_batch_io_seconds": evaluation,
        "recovery_seconds": replay_seconds, "exact_recovery_digest": actual_digest,
        "initial": initial, "middle": middle, "last": previous, "outputs": outputs}
    return store.publish_json("result.json", result)


def fitter_workloads(rows):
    """Observable only: targets/inputs are deliberately not inspected here."""
    result = []
    for kind in ("arithmetic", "first_train_batch"):
        for branch, (_, _, ks) in law.BRANCHES.items():
            for m in range(4, 9):
                values, origins = [], []
                if kind == "arithmetic":
                    for i in range(8):
                        q = np.log(np.arange(1, m+1, dtype=np.float64)) + .0001*i*np.arange(m)
                        values.append((q-q.mean()).tolist())
                        origins.append({"fixture": i})
                else:
                    for row in rows:
                        q = law.centered(law.observable_q32(row["q32"]))
                        groups = sorted({g for p in row["partitions"] if law.supported(p) and len(p) in ks
                                         for g in law.signature(p) if len(g) == m})
                        for g in groups:
                            if len(values) == 8:
                                break
                            values.append(q[list(g)].tolist())
                            origins.append({"scene_id": row["scene_id"], "identity": row["identity"], "group": list(g)})
                        if len(values) == 8:
                            break
                result.append({"kind": kind, "branch": branch, "size": m,
                               "values": values, "origins": origins})
    return result


def fit_profile(store, workloads, device, *, check, clock=time.monotonic):
    fitter = law.GroupFitter(law.Grid(257, 65, 4), device="cuda" if device == "cuda:0" else "cpu", assignment_batch=8)
    results = []
    for i, item in enumerate(workloads):
        check()
        if not item["values"]:
            results.append({"index": i, "status": "NO_GROUPS_IN_BATCH", "seconds": 0.})
            continue
        before = clock()
        factors = fitter.fit(item["values"], item["branch"])
        if device == "cuda:0":
            torch.cuda.synchronize(0)
        fit_seconds = clock()-before
        before = clock()
        ref = store.publish_json(f"factors/{i:02d}.json", {"workload": item, "factors": factors})
        if store.json(ref)["workload"] != item:
            raise ValueError("fitter workload serialization changed")
        results.append({"index": i, "status": "MEASURED", "fit_seconds": fit_seconds,
                        "io_seconds": clock()-before, "groups": len(factors), "factors": ref})
        store.publish_json(f"timings/{i:02d}.json", results[-1])
        check()
    return store.publish_json("result.json", {"status": "GROUP_PRIMITIVE_PROFILE_NOT_FULL_PIPELINE",
        "binding": store.binding, "device": device, "grid": [257, 65, 4], "assignment_batch": 8,
        "results": results})
