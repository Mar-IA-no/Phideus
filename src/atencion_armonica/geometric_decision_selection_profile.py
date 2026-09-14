"""CPU arithmetic/IO profile for selection; no scene producer or model calls."""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import torch

from .geometric_decision_open import ReadOnlyStore
from .geometric_decision_selection import ROSTER, select_epochs
from .geometric_decision_profile import assert_same_arrays


def profile_selection(store, *, check):
    """Full 512 x 82 envelope, 720 distinct arrays, eleven full-size IO reads.

    Fixture scores/targets are arithmetic and have no relationship to source
    identities, train/calibration observations or held-out test generators.
    """
    check()
    count = 512*82
    components = np.sin(np.arange(count*2, dtype=np.float64)).reshape(count, 2)
    arrays = {"components": components, "energy": components.sum(-1, dtype=np.float64),
              "offsets": np.arange(513, dtype=np.int64)*82}
    targets = [np.tile(np.array([[.2, .3]], dtype=np.float32), (82, 1)) for _ in range(512)]
    energies = {key: arrays["energy"].copy() for key in ROSTER}
    io = []
    for epoch in range(0, 51, 5):
        check()
        blob = store.publish_arrays(f"fixture-epoch-{epoch}.npz", arrays)
        ref = store.publish_json(f"fixture-epoch-{epoch}.json", {"fixture": "arithmetic-not-observed",
            "epoch": epoch, "identities": [f"fixture-{i:04d}" for i in range(512)], "predictions": blob})
        started = time.monotonic()
        value = store.json(ref)
        restored = store.arrays(value["predictions"])
        assert_same_arrays(restored, arrays)
        io.append({"record": ref, "read_verify_seconds": time.monotonic()-started})
    check()
    started = time.monotonic()
    selected = select_epochs(targets, energies)
    selection_seconds = time.monotonic()-started
    check()
    result = store.publish_json("fixture-selection.json", selected)
    # Dense arithmetic snapshot payload, not a TrainingKernel and not an
    # optimizer run. Include 24 tensors (weights and two moments), RNG-sized
    # arrays, and a deliberately large 50-row scalar history. Measure trusted
    # CPU unpickle/digest plus eleven independently authenticated blobs.
    snapshots, previous = [], None
    shapes = ((32, 9), (32,), (16, 32), (16,), (32, 41), (32,), (2, 32), (2,))
    for epoch in range(0, 51, 5):
        check()
        state = {"binding": store.binding, "epoch": epoch, "next_batch": 0, "steps": epoch*127,
            "fixture": "dense-arithmetic-state-not-trained",
            "tensors": {f"{kind}-{i}": torch.zeros(shape, dtype=torch.float32)
                        for kind in ("weights", "moment1", "moment2") for i, shape in enumerate(shapes)},
            "rng_sized": torch.zeros(3*5056, dtype=torch.uint8),
            "history": [{"epoch": i, "arithmetic_payload": [j/1000 for j in range(512)]} for i in range(50)]}
        previous = store.save_state(SimpleNamespace(state=lambda: state), previous)
        snapshots.append(previous)
    started = time.monotonic()
    final = store.load_state(snapshots[-1])
    if final["history"] != state["history"]:
        raise ValueError("arithmetic state serialization differs")
    for ref in snapshots:
        check()
        record = store.json(ref)
        store.read(record["state"])
    state_io_seconds = time.monotonic()-started
    check()
    return store.publish_json("profile.json", {"schema": "geometric-decision-selection-profile-v1",
        "binding": store.binding, "fixture": "arithmetic-not-observed", "scenes": 512,
        "candidates_per_scene": 82, "energy_arrays": 720, "io": io,
        "selection_seconds": selection_seconds, "fixture_result": result,
        "snapshots": snapshots, "cell_state_io_seconds": state_io_seconds})


def selection_forecast(report, *, corpus_load_seconds):
    if (report["schema"] != "geometric-decision-selection-profile-v1"
            or report["fixture"] != "arithmetic-not-observed" or report["scenes"] != 512
            or report["candidates_per_scene"] != 82 or report["energy_arrays"] != 720
            or len(report["io"]) != 11 or len(report["snapshots"]) != 11):
        raise ValueError("selection profile differs from the full arithmetic envelope")
    values = [corpus_load_seconds, report["selection_seconds"], report["cell_state_io_seconds"],
              *[r["read_verify_seconds"] for r in report["io"]]]
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("selection profile timings must be positive finite values")
    # Same authenticated full-CP loader as training. No additional forwards.
    # Max of the eleven full-array read/verify timings, not their minimum.
    load = 3*corpus_load_seconds
    # validate_complete reads every calibration; selection reopens its energy.
    io = 2*72*11*max(r["read_verify_seconds"] for r in report["io"])
    states = 72*report["cell_state_io_seconds"]
    subtotal = load+io+states+report["selection_seconds"]
    return {"schema": "geometric-decision-selection-forecast-v1", "cpu_only": True,
        "corpus_loads": 3, "calibration_reads": 1584, "selected_arrays": 720,
        "final_state_loads": 72, "state_blob_reads": 792, "state_io_seconds": states,
        "load_seconds": load, "read_verify_seconds": io,
        "selection_seconds": report["selection_seconds"], "margin": 1.25,
        "projected_seconds": 1.25*subtotal,
        "scope": "selection only; measured envelope estimate, not a runtime guarantee"}


def admitted_selection_profile(control, *, root, corpus_load_seconds):
    candidates = []
    for path in sorted(control.path("attempts").glob("*/finish.json")):
        ref = control.reference(path)
        finish = control.json(ref)
        start = control.json(finish["start"])
        manifest = control.json(start["manifest"])
        if manifest.get("operation") != "profile-selection" or finish["status"] != "COMPLETE":
            continue
        output = control.json(finish["completion"])
        if (start["stage"] != "profile" or start["binding"] != control.binding
                or manifest["root"] != str(root) or output["root"] != str(root)
                or output["manifest"] != start["manifest"]):
            raise ValueError("selection profile completion provenance differs")
        view = ReadOnlyStore(root, binding_ref=output["binding"])
        report = view.json(output["profile"])
        if view.binding != manifest["binding"] or report["binding"] != view.binding:
            raise ValueError("selection profile binding differs")
        for row in report["io"]:
            value = view.json(row["record"])
            view.arrays(value["predictions"])
        view.json(report["fixture_result"])
        from .geometric_decision_admission import ProfileReadOnlyStore
        state_view = ProfileReadOnlyStore(root, binding_ref=output["binding"])
        for state_ref in report["snapshots"]:
            state_view.read(state_view.json(state_ref)["state"])
        candidates.append({"finish": ref, "output": output, "binding": view.binding,
            "forecast": selection_forecast(report, corpus_load_seconds=corpus_load_seconds)})
    if len(candidates) != 1:
        raise ValueError("exactly one COMPLETE selection CPU profile required")
    return candidates[0]
