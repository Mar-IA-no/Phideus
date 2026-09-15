"""Producer-side persisted emissions and bounded CPU sensor profile.

Called only inside PhaseController after source/admission/resource verification.
Test roles are intentionally unavailable in this OPEN producer. A later test
adapter must require the frozen contract, not reinterpret these calls as test IO.
"""
from __future__ import annotations

import time

import numpy as np

from .measurement_contract import identity, SCENARIOS
from .measurement_contract import digest
from .measurement_emission import emission_from_draw, audio_from_emission
from .measurement_metrics import correspondence, detection_cost
from .measurement_sensor import CONDITIONS, detect_peaks


def open_scene(store, role, scenario, scene_id, *, source_snapshot, check, draw):
    if role not in ("development", "calibration"):
        raise PermissionError("OPEN producer cannot draw prospective test scenes")
    unit = identity(role, scenario, "canonical", scene_id)
    stem = f"open/{role}/{scenario}/{scene_id:04d}"
    base_identity = {"unit": unit, "source_snapshot": source_snapshot}
    check()
    raw_identity = {**base_identity, "operation": "raw-draw"}
    raw = store.completed(stem+"/raw", raw_identity)
    if raw is None:
        observation, sidecar = draw(scenario, scene_id, unit["split_seed"])
        check()
        # Validate conversion before preserving the raw draw. No observed peak or
        # model output can affect the source draw, amplitude, phase or noise.
        emission_from_draw(unit, observation, sidecar)
        store.publish_stage(stem+"/raw", raw_identity,
                            {"observation": observation, "sidecar": sidecar}, {})
        raw = store.completed(stem+"/raw", raw_identity)
    emission_identity = {**base_identity, "operation": "emission", "raw": raw[0]}
    emission = store.completed(stem+"/emission", emission_identity)
    if emission is None:
        observed, _ = emission_from_draw(unit, raw[1]["observation"], raw[1]["sidecar"])
        store.publish_stage(stem+"/emission", emission_identity, observed["metadata"], observed["arrays"])
        emission = store.completed(stem+"/emission", emission_identity)
    waveform_identity = {**base_identity, "operation": "waveform", "emission": emission[0]}
    waveform = store.completed(stem+"/waveform", waveform_identity)
    if waveform is None:
        metadata, arrays = audio_from_emission({"metadata": emission[1], "arrays": emission[2]})
        check()
        store.publish_stage(stem+"/waveform", waveform_identity, metadata, arrays)
        waveform = store.completed(stem+"/waveform", waveform_identity)
    check()
    return {"unit": unit, "raw": raw, "emission": emission, "waveform": waveform, "stem": stem}


def observed_audio(store, scene, condition, *, height, prominence, check):
    if condition not in ("nominal", "short", "noisy"):
        raise ValueError("unknown audio condition")
    unit = {**scene["unit"], "condition": condition}
    operation = {"unit": unit, "operation": "detect", "waveform": scene["waveform"][0],
                 "height": height, "prominence": prominence}
    stem = scene["stem"]+f"/detect/{condition}/h{height:g}-p{prominence:g}"
    check()
    restored = store.completed(stem, operation)
    if restored is None:
        detected = detect_peaks(scene["waveform"][2][condition], height=height, prominence=prominence)
        arrays = {k: v for k, v in detected.items() if isinstance(v, np.ndarray)}
        arrays.update({"property/"+k: v for k, v in detected["properties"].items()})
        metadata = {"unit": unit, "nfft": detected["nfft"], "status": detected["status"],
                    "property_names": sorted(detected["properties"]), "height": height, "prominence": prominence}
        check()
        store.publish_stage(stem, operation, metadata, arrays)
        restored = store.completed(stem, operation)
    check()
    return restored


def cpu_profile(store, source_snapshot, check, *, draw, clock=time.monotonic):
    """Four fixed development scene0, canonical plus three audio conditions.

Reports actual measured cardinalities, not a worst-case coverage assertion.
No model features, checkpoints, fitter or learned-reader forward occurs here.
"""
    rows, phases = [], []
    for scenario in SCENARIOS:
        check()
        started = clock()
        scene = open_scene(store, "development", scenario, 0, source_snapshot=source_snapshot,
                           check=check, draw=draw)
        phases.append({"scenario": scenario, "render_and_persistence_seconds": clock()-started})
        emitted = scene["emission"][2]["frequencies"]
        rows.append({"unit": scene["unit"], "n": len(emitted), "detector_seconds": 0.,
                     "evaluation_seconds": 0., "emission": scene["emission"][0], "detected": None})
        for condition, _, _ in CONDITIONS:
            started = clock()
            detected = observed_audio(store, scene, condition, height=-30., prominence=6., check=check)
            detector_seconds = clock()-started
            started = clock()
            frequencies = detected[2]["frequencies"]
            matches = correspondence(emitted, frequencies)
            cost = detection_cost(emitted, frequencies)
            check()
            rows.append({"unit": {**scene["unit"], "condition": condition}, "n": len(frequencies),
                "detector_seconds": detector_seconds, "evaluation_seconds": clock()-started,
                "detection_cost": cost["cost"], "unique_matches": int(np.sum(matches["detected_to_emitted"] >= 0)),
                "emission": scene["emission"][0], "detected": detected[0]})
    return {"schema": "measurement-cpu-profile-v1", "source_snapshot": source_snapshot,
            "rows": rows, "render": phases, "thresholds": {"height": -30., "prominence": 6.},
            "coverage": "four development scenes; not a measured worst-case bound"}, {}


def validate_cpu_profile(store, source_snapshot, result, arrays):
    expected = [identity("development", scenario, condition, 0)
                for scenario in SCENARIOS for condition in ("canonical", "nominal", "short", "noisy")]
    if (set(result) != {"schema", "source_snapshot", "rows", "render", "thresholds", "coverage"}
            or result["schema"] != "measurement-cpu-profile-v1" or arrays
            or digest(result["source_snapshot"]) != digest(source_snapshot)
            or digest([row["unit"] for row in result["rows"]]) != digest(expected)
            or result["thresholds"] != {"height": -30., "prominence": 6.}
            or [r["scenario"] for r in result["render"]] != list(SCENARIOS)):
        raise ValueError("CPU profile does not cover its fixed OPEN roster")
    durations = [r["render_and_persistence_seconds"] for r in result["render"]]
    for row in result["rows"]:
        durations += [row["detector_seconds"], row["evaluation_seconds"]]
        if type(row["n"]) is not int or row["n"] < 0:
            raise ValueError("invalid measured cardinality")
        store.read(row["emission"])
        if row["unit"]["condition"] == "canonical":
            if row["detected"] is not None or not 8 <= row["n"] <= 32:
                raise ValueError("canonical profile observation differs")
        else:
            complete = store.json(row["detected"])
            metadata, detected_arrays = store._payload(complete["payload"], complete["identity"])
            if digest(metadata["unit"]) != digest(row["unit"]) or len(detected_arrays["frequencies"]) != row["n"]:
                raise ValueError("profile detector receipt differs")
    if any(type(x) not in (int, float) or not np.isfinite(x) or x < 0 for x in durations):
        raise ValueError("invalid measured CPU profile duration")
