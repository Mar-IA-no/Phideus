"""Nine fixed detectors on OPEN calibration, independent of every reader score."""
from __future__ import annotations

import numpy as np

from .measurement_contract import SCENARIOS, ROLES, identity, unit_roster, digest, calibrate_detector
from .measurement_open import open_scene, observed_audio
from .measurement_payload import stage
from .measurement_metrics import correspondence, detection_cost
from .measurement_sensor import DETECTOR_GRID
from .partial_compatibility_cache import encoded


def cost_record(store, scene, condition, *, height, prominence, check):
    detected = observed_audio(store, scene, condition, height=height, prominence=prominence, check=check)
    unit = {**scene["unit"], "condition": condition}
    operation = {"unit": unit, "emission": scene["emission"][0], "detected": detected[0],
                 "height": height, "prominence": prominence}
    def calculate():
        emitted, observed = scene["emission"][2]["frequencies"], detected[2]["frequencies"]
        matching = correspondence(emitted, observed)
        # String status vectors are JSON labels, not tensors for the readout.
        matching = {k: v.tolist() if isinstance(v, np.ndarray) and v.dtype.kind == "U" else v
                    for k, v in matching.items()}
        return {"unit": unit, "matching": matching,
                "cost": detection_cost(emitted, observed)}
    def validate(value):
        from .measurement_payload import pack
        expected = calculate()
        left, la = pack(value)
        right, ra = pack(expected)
        if encoded(left) != encoded(right) or set(la) != set(ra) or any(
                la[k].dtype != ra[k].dtype or not np.array_equal(la[k], ra[k]) for k in la):
            raise ValueError("calibration cost/matching does not replay observed frequencies")
    folder = scene["stem"]+f"/cost/{condition}/h{height:g}-p{prominence:g}"
    return stage(store, folder, operation, produce=calculate, validate=validate, check=check)


def calibrate(store, source_snapshot, check, *, draw):
    """Only OPEN role; retain 1728 costs and their individual sensor/matching refs."""
    units = unit_roster("calibration", audio_only=True)
    lookup = {digest(unit): i for i, unit in enumerate(units)}
    costs = np.full((9, len(units)), np.nan, np.float64)
    refs = [[None]*len(units) for _ in DETECTOR_GRID]
    for scenario in SCENARIOS:
        for sid in range(ROLES["calibration"][0]):
            check()
            scene = open_scene(store, "calibration", scenario, sid, source_snapshot=source_snapshot,
                               check=check, draw=draw)
            for condition in ("nominal", "short", "noisy"):
                j = lookup[digest(identity("calibration", scenario, condition, sid))]
                for i, (height, prominence) in enumerate(DETECTOR_GRID):
                    ref, result = cost_record(store, scene, condition, height=height,
                                               prominence=prominence, check=check)
                    costs[i, j], refs[i][j] = result["cost"]["cost"], ref
    selected = calibrate_detector(costs, units)
    store.publish_json("open/calibration/cost-index.json", {
        "schema": "measurement-calibration-cost-index-v1", "source_snapshot": source_snapshot,
        "units": units, "grid": [list(v) for v in DETECTOR_GRID], "records": refs,
        "calibration_sha256": selected["calibration_sha256"]})
    check()
    return selected, {}


def validate_calibration(store, source_snapshot, result, arrays, check):
    from .measurement_payload import unpack
    expected = calibrate_detector(np.asarray(result["costs"], np.float64), result["units"])
    if arrays or encoded(result) != encoded(expected):
        raise ValueError("calibration selection does not replay")
    index = store.json(store.reference("open/calibration/cost-index.json"))
    if (set(index) != {"schema", "source_snapshot", "units", "grid", "records", "calibration_sha256"}
            or index["schema"] != "measurement-calibration-cost-index-v1"
            or digest(index["source_snapshot"]) != digest(source_snapshot)
            or digest(index["units"]) != digest(result["units"])
            or encoded(index["grid"]) != encoded(result["grid"])
            or index["calibration_sha256"] != result["calibration_sha256"]
            or len(index["records"]) != 9 or any(len(row) != 192 for row in index["records"])):
        raise ValueError("calibration cost index differs")
    for i, row in enumerate(index["records"]):
        for j, ref in enumerate(row):
            check()
            receipt = store.json(ref)
            operation = receipt["identity"]
            if (digest(operation["unit"]) != digest(result["units"][j])
                    or [operation["height"], operation["prominence"]] != result["grid"][i]):
                raise ValueError("calibration record unit or detector differs")
            metadata, payload = store._payload(receipt["payload"], operation)
            value = unpack(metadata, payload)
            if digest(value["unit"]) != digest(result["units"][j]) or value["cost"]["cost"] != result["costs"][i][j]:
                raise ValueError("calibration cost differs from preserved individual state")
