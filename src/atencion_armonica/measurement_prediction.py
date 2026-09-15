"""Persisted observable-only inference using the unchanged geometric kernels.

The caller supplies a prepared observation, its authenticated receipt and an
admitted backend. No emission, matching, ground truth, sampler or GPU discovery
is accepted here. Unit stages run inside the campaign's measured phase.
"""
from __future__ import annotations

import time

import numpy as np

from .measurement_operator import validate_prepared, features_for_input, assemble_observable, candidate_origins
from .measurement_payload import stage
from .measurement_reuse import CHECKPOINTS, READERS, EPOCHS
from .partial_compatibility_cache import encoded


def predict_unit(store, folder, prepared, *, observation_ref, source_snapshot, reused, backend, check,
                 clock=time.monotonic):
    q = validate_prepared(prepared)
    base = {"unit": prepared["unit"], "observation": observation_ref, "source_snapshot": source_snapshot}
    refs, timings = {}, {}
    def saved(name, inputs, fn, validate):
        started = clock()
        ref, result = stage(store, folder+"/"+name, {**base, "operation": name, "inputs": inputs},
                            produce=fn, validate=validate, check=check)
        refs[name] = ref
        timings[name] = clock()-started
        return result
    def gpu_saved(name, inputs, fn, validate):
        def produce():
            execution = backend.execution(check)
            return {"execution": execution, "value": fn()}
        def checked(value):
            if not isinstance(value, dict) or set(value) != {"execution", "value"}:
                raise ValueError("GPU payload must preserve its actual execution receipt")
            backend.validate_execution(value["execution"])
            validate(value["value"])
        return saved(name, inputs, produce, checked)["value"]
    if q is None:
        return {"schema": "measurement-inference-unit-v1", "unit": prepared["unit"],
                "status": "OUTSIDE_OPERATOR_DOMAIN", "stages": refs, "seconds": timings,
                "candidate_count": 0, "choices": {}}
    from .structured_source_data import validate_record
    features = saved("features", observation_ref, lambda: features_for_input(prepared),
                     lambda value: validate_record(value, q))
    logits = {}
    for checkpoint in reused["checkpoints"]:
        cp = checkpoint["seed"]
        def validate_logits(value):
            if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                    or value.shape != (len(q), len(q)) or not np.isfinite(value).all()
                    or not np.array_equal(value, value.T)):
                raise ValueError("invalid preserved observable logits")
        logits[cp] = gpu_saved(f"logits-{cp}", {"features": refs["features"], "checkpoint": checkpoint},
                           lambda cp=cp: backend.forward(cp, features, check), validate_logits)
    if tuple(logits) != CHECKPOINTS:
        raise ValueError("inference requires the exact three-backbone roster")
    def validate_scene(value):
        # CPU replay of the proposer, not a second forward or fit.
        expected = assemble_observable(prepared, features, logits)
        from .measurement_payload import pack
        left, la = pack(value)
        right, ra = pack(expected)
        if encoded(left) != encoded(right) or set(la) != set(ra) or any(
                la[k].dtype != ra[k].dtype or not np.array_equal(la[k], ra[k]) for k in la):
            raise ValueError("candidate source does not replay from saved observables")
    assembled = saved("scene", {k: v for k, v in refs.items()},
                      lambda: assemble_observable(prepared, features, logits), validate_scene)
    scene = assembled["scene"]
    if assembled["status"] == "NO_CANDIDATE":
        return {"schema": "measurement-inference-unit-v1", "unit": prepared["unit"],
                "status": "NO_CANDIDATE", "stages": refs, "seconds": timings,
                "candidate_count": 0, "choices": {}}
    from .geometric_decision_scene_store import validate_fits
    fitted = gpu_saved("fits", refs["scene"], lambda: backend.fit(scene["q32"], scene["partitions"], check),
                   lambda value: validate_fits(scene, value))
    from .geometric_decision_observables import inputs_from_fits
    normalizers = {key: reused["normalizers"][key] for key in ("common", "evidence")}
    def build_inputs():
        return inputs_from_fits(scene, fitted["fits"], normalizers, scale=reused["scale"],
                                expected_seed=prepared["unit"]["split_seed"])
    def validate_inputs(value):
        from .measurement_payload import pack
        expected = build_inputs()
        left, la = pack(value)
        right, ra = pack(expected)
        if encoded(left) != encoded(right) or set(la) != set(ra) or any(
                la[k].dtype != ra[k].dtype or not np.array_equal(la[k], ra[k]) for k in la):
            raise ValueError("delivered inputs do not replay from frozen normalizers and fits")
    inputs = saved("inputs", {"scene": refs["scene"], "fits": refs["fits"],
                              "reuse": reused["references"]}, build_inputs, validate_inputs)
    from .geometric_decision_classical import scores
    def classical():
        arrays, metadata = scores(scene["partitions"], fitted["fits"], n=len(q),
                                  delivered=inputs["inputs"][CHECKPOINTS[0]]["evidence"])
        return {"arrays": arrays, "metadata": metadata, "origins": candidate_origins(scene)}
    def validate_classical(value):
        expected = classical()
        if encoded(value["metadata"]) != encoded(expected["metadata"]) or encoded(value["origins"]) != encoded(expected["origins"]):
            raise ValueError("classical decisions or candidate provenance differ")
        if set(value["arrays"]) != set(expected["arrays"]) or any(
                value["arrays"][k].dtype != np.float64 or not np.array_equal(value["arrays"][k], a)
                for k, a in expected["arrays"].items()):
            raise ValueError("classical raw scores differ")
    classic = saved("classical", refs["inputs"], classical, validate_classical)
    choices = {"classical/"+k: v for k, v in classic["metadata"]["choices"].items()}
    from .geometric_decision_predictions import checked_prediction
    roster = [(arm, cp, seed) for arm in EPOCHS for cp in CHECKPOINTS for seed in READERS]
    if [(h["arm"], h["checkpoint_seed"], h["reader_seed"]) for h in reused["heads"]] != roster:
        raise ValueError("inference requires all 36 preselected heads in declared order")
    for head in reused["heads"]:
        cp, arm, seed = head["checkpoint_seed"], head["arm"], head["reader_seed"]
        name = f"head-{arm}-{cp}-{seed}"
        row, raw = inputs["inputs"][cp], inputs["raw"][cp]
        def selected(value, row=row, raw=raw, arm=arm):
            return checked_prediction(value, [row], [raw], route=arm.rsplit("_", 1)[0], initial=False)[0]
        prediction = gpu_saved(name, {"inputs": refs["inputs"], "head": head["record"]},
                           lambda head=head, row=row: backend.predict(head, row, check), selected)
        choices[name] = selected(prediction)
    return {"schema": "measurement-inference-unit-v1", "unit": prepared["unit"],
            "status": "ELIGIBLE", "stages": refs, "seconds": timings,
            "candidate_count": len(scene["partitions"]), "choices": choices}
