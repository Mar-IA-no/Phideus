"""Full-roster cell runner, durable signed calibration, and resumable training.

The caller authenticates delivered data and admits a backend/resource budget.
This module neither discovers datasets nor selects an epoch or reads tests.
"""
from __future__ import annotations

import uuid

import numpy as np
import torch

from .generative_evidence_cell import CellData as SixChannelData
from .geometric_decision_core import ARMS, READER_SEEDS, route_inputs
from .geometric_decision_model import collate
from .geometric_decision_training import TrainingKernel, collate_targets
from .generative_evidence import CHECKPOINTS

EPOCHS = (0, *range(5, 51, 5))


class CellData:
    def __init__(self, train, calibration, *, binding):
        # Reuse the frozen validator for exact rosters, canonical candidate/group
        # order, incidence and supervision; only its six-column view is passed.
        # Actual tensors retained and delivered below always have eight columns.
        original = {"train": train, "calibration": calibration}
        views = {}
        for split, rows in original.items():
            if not isinstance(rows, list):
                raise ValueError("complete canonical cell rows required")
            views[split] = []
            for row in rows:
                ev = row["inputs"]["evidence"]
                if (not isinstance(ev, np.ndarray) or ev.dtype != np.float32 or ev.ndim != 2
                        or ev.shape != (len(row["partitions"]), 8) or not np.isfinite(ev).all()
                        or np.any(ev[:, 6:] < 0)):
                    raise ValueError("common cell evidence requires eight finite channels")
                views[split].append({**row, "inputs": {**row["inputs"], "evidence": ev[:, :6]}})
        validated = SixChannelData(views["train"], views["calibration"], binding=binding)
        self.rows, self.binding, self.eligible = original, binding, validated.eligible

    def batch(self, split, ids, route):
        if split not in self.rows or not isinstance(ids, list) or not 1 <= len(ids) <= 32:
            raise ValueError("OPEN batch requires one to 32 eligible scene IDs")
        if len(set(ids)) != len(ids) or any(type(i) is not int or i not in self.eligible[split] for i in ids):
            raise ValueError("OPEN batch ID outside eligible roster")
        return collate([route_inputs(self.rows[split][i]["inputs"], route) for i in ids])


def _calibration_identity(data, store, state_ref, epoch):
    if epoch not in EPOCHS:
        raise ValueError("calibration epoch outside initial/every-five roster")
    state = store.json(state_ref)
    store._state_record(state)
    if state["epoch"] != epoch or state["next_batch"] != 0:
        raise ValueError("calibration requires a completed epoch snapshot")
    rows = data.rows["calibration"]
    offsets = np.r_[np.int64(0), np.cumsum([len(r["partitions"]) for r in rows], dtype=np.int64)]
    identity = {"schema": "geometric-decision-calibration-v1", "binding": store.binding,
                "epoch": epoch, "state": state_ref, "identities": [r["identity"] for r in rows],
                "eligible_scene_ids": data.eligible["calibration"]}
    return identity, offsets


def read_calibration(data, store, ref, state_ref, epoch):
    identity, offsets = _calibration_identity(data, store, state_ref, epoch)
    record = store.json(ref)
    if (set(record) != set(identity) | {"predictions"}
            or any(record[k] != v for k, v in identity.items())):
        raise ValueError("calibration binding or full scene roster differs")
    arrays = store.arrays(record["predictions"])
    if set(arrays) != {"components", "energy", "offsets"}:
        raise ValueError("calibration array schema differs")
    components, energy, saved_offsets = (arrays[k] for k in ("components", "energy", "offsets"))
    if (components.dtype != np.float64 or components.shape != (offsets[-1], 2)
            or not np.isfinite(components).all() or energy.dtype != np.float64
            or not np.array_equal(energy, components.sum(-1, dtype=np.float64))
            or saved_offsets.dtype != np.int64 or not np.array_equal(saved_offsets, offsets)):
        raise ValueError("signed64 calibration extent or energy differs")
    return arrays


def calibration_outputs(kernel, data, store, state_ref, check):
    if kernel.binding != store.binding or kernel.next_batch != 0:
        raise ValueError("calibration kernel binding or epoch boundary differs")
    epoch = kernel.epoch
    identity, offsets = _calibration_identity(data, store, state_ref, epoch)
    # Do not label another live model with a valid but unrelated state receipt.
    from .generative_evidence_cell import state_digest
    if state_digest(kernel.state()) != store.json(state_ref)["state_digest"]:
        raise ValueError("calibration live state differs from snapshot")
    path = store.path(f"calibration/epoch_{epoch:02d}/index.json")
    if path.exists():
        ref = store.reference(path)
        read_calibration(data, store, ref, state_ref, epoch)
        return ref
    components = np.empty((offsets[-1], 2), np.float64)
    ids = data.eligible["calibration"]
    kernel.model.eval()
    for start in range(0, len(ids), 32):
        check()
        batch_ids = ids[start:start+32]
        batch = {k: v.to(kernel.device) for k, v in data.batch("calibration", batch_ids, kernel.model.route).items()}
        with torch.no_grad():
            values = kernel.model(batch).cpu().numpy()
        for j, scene_id in enumerate(batch_ids):
            first, last = offsets[scene_id:scene_id+2]
            components[first:last] = values[j, :last-first]
    check()
    arrays = {"components": components, "energy": components.sum(-1, dtype=np.float64), "offsets": offsets}
    blob = store.publish_arrays(f"calibration/epoch_{epoch:02d}/outputs-{uuid.uuid4().hex}.npz", arrays)
    ref = store.publish_json(path.relative_to(store.root).as_posix(), {**identity, "predictions": blob})
    read_calibration(data, store, ref, state_ref, epoch)
    return ref


def run_cell(data, store, *, arm, checkpoint_seed, reader_seed, device, check, progress=print):
    """Run all 50 epochs or raise with partial durable state, never partial success."""
    if arm not in ARMS or checkpoint_seed not in CHECKPOINTS or reader_seed not in READER_SEEDS:
        raise ValueError("unknown fixed cell")
    expected = {"schema": "geometric-decision-cell-binding-v1", "data": data.binding,
                "arm": arm, "checkpoint_seed": checkpoint_seed, "reader_seed": reader_seed,
                "device": device}
    if store.binding != expected:
        raise ValueError("cell store does not bind exact data, arm, backbone, seed and backend")
    kernel = TrainingKernel(arm, checkpoint_seed, reader_seed, binding=store.binding,
                            scene_ids=data.eligible["train"], device=device)
    check()
    previous = store.latest(kernel)
    if previous is None:
        previous = store.save_state(kernel, None)
    batches = len(kernel.schedule[0])
    # Calibrations at completed boundaries must exist. If interrupted after a
    # boundary snapshot but before calibration, only that current boundary may
    # be reconstructed; older missing outputs are not silently re-forwarded.
    calibration = []
    for epoch in EPOCHS:
        if epoch > kernel.epoch:
            break
        state_path = store.path(f"snapshots/step_{epoch*batches:06d}.json")
        if not state_path.exists():
            raise ValueError("missing retained calibration boundary snapshot")
        state_ref = store.reference(state_path)
        path = store.path(f"calibration/epoch_{epoch:02d}/index.json")
        if path.exists():
            ref = store.reference(path)
            read_calibration(data, store, ref, state_ref, epoch)
        elif epoch == kernel.epoch and kernel.next_batch == 0:
            ref = calibration_outputs(kernel, data, store, state_ref, check)
        else:
            raise ValueError("missing earlier calibration output; explicit recovery required")
        calibration.append(ref)
    try:
        while kernel.epoch < 50:
            check()
            ids = kernel.expected_scene_ids().tolist()
            batch = data.batch("train", ids, kernel.model.route)
            target = collate_targets([data.rows["train"][i]["targets"] for i in ids], batch["candidate_mask"])
            kernel.step(batch, target, np.asarray(ids, np.int64))
            if kernel.steps % 32 == 0 or kernel.next_batch == 0:
                previous = store.save_state(kernel, previous)
            if kernel.next_batch == 0:
                progress({"stage": "training", "arm": arm, "checkpoint_seed": checkpoint_seed,
                          "reader_seed": reader_seed, "epoch": kernel.epoch, "steps": kernel.steps})
                if kernel.epoch in EPOCHS:
                    calibration.append(calibration_outputs(kernel, data, store, previous, check))
    except BaseException:
        # A failed optimizer step is unsafe; retain only the preceding snapshot.
        if kernel.safe_boundary and store.json(previous)["steps"] < kernel.steps:
            store.save_state(kernel, previous)
        raise
    check()
    if len(calibration) != len(EPOCHS) or kernel.next_batch != 0:
        raise ValueError("cell cannot complete without all retained calibrations")
    record = {"schema": "geometric-decision-cell-complete-v1", "binding": store.binding,
              "last_epoch": 50, "steps": kernel.steps, "last_state": previous,
              "calibration": calibration, "history": kernel.history}
    return store.publish_json("complete.json", record)
