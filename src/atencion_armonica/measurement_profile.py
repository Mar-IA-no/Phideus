"""OPEN GPU profile over the exact sixteen observations preserved by CPU profile."""
from __future__ import annotations

from .measurement_contract import digest
from .measurement_open import validate_cpu_profile
from .measurement_operator import prepare_input, validate_prepared
from .measurement_payload import stage, unpack
from .measurement_prediction import predict_unit
from .partial_compatibility_cache import encoded


def _artifact(store, ref):
    complete = store.json(ref)
    if (complete.get("schema") != "measurement-stage-complete-v1"
            or encoded(complete.get("binding")) != encoded(store.binding)):
        raise ValueError("profile input is not a complete artifact of this campaign")
    result, arrays = store._payload(complete["payload"], complete["identity"])
    return complete, result, arrays


def execution_refs(store, state):
    result = {}
    for name, ref in state["stages"].items():
        if name == "fits" or name.startswith(("logits-", "head-")):
            _, metadata, arrays = _artifact(store, ref)
            value = unpack(metadata, arrays)
            if set(value) != {"execution", "value"}:
                raise ValueError("GPU profile artifact lacks its producing execution")
            execution = value["execution"]
            store.read(execution)
            result[digest(execution)] = execution
    return result


def gpu_profile(store, source_snapshot, cpu_profile_ref, reused, backend, check):
    receipt, cpu, arrays = _artifact(store, cpu_profile_ref)
    if encoded(receipt["identity"]) != encoded({"phase": "profile_cpu", "source_snapshot": source_snapshot}):
        raise ValueError("GPU profile requires the exact admitted CPU profile")
    validate_cpu_profile(store, source_snapshot, cpu, arrays)
    records, executions = [], {}
    for row in cpu["rows"]:
        check()
        unit = row["unit"]
        origin = row["emission"] if unit["condition"] == "canonical" else row["detected"]
        _, metadata, values = _artifact(store, origin)
        coordinates = values["canonical_q32"] if unit["condition"] == "canonical" else values["frequencies"]
        prepared = prepare_input(unit, coordinates)
        folder = f"open/gpu-profile/{unit['scenario']}/{unit['scene_id']:04d}/{unit['condition']}"
        observation_identity = {"unit": unit, "source_snapshot": source_snapshot, "origin": origin}
        observation_ref = store.publish_stage(folder+"/observation", observation_identity, prepared, {})
        def produce():
            return predict_unit(store, folder+"/inference", prepared, observation_ref=observation_ref,
                source_snapshot=source_snapshot, reused=reused, backend=backend, check=check)
        def validate(value):
            if (value.get("schema") != "measurement-inference-unit-v1"
                    or digest(value.get("unit")) != digest(unit)
                    or value.get("status") not in ("ELIGIBLE", "NO_CANDIDATE", "OUTSIDE_OPERATOR_DOMAIN")
                    or type(value.get("candidate_count")) is not int or not 0 <= value["candidate_count"] <= 82):
                raise ValueError("GPU profile unit output differs")
            validate_prepared(prepared)
            if (prepared["status"] == "OUTSIDE_OPERATOR_DOMAIN") != (value["status"] == "OUTSIDE_OPERATOR_DOMAIN"):
                raise ValueError("GPU profile discarded or admitted the wrong domain")
            if value["status"] == "ELIGIBLE" and (value["candidate_count"] == 0 or len(value["choices"]) != 40):
                raise ValueError("GPU profile lost eligible reader/classical choices")
            for ref in value["stages"].values():
                _artifact(store, ref)
        ref, result = stage(store, folder+"/result", {"unit": unit, "source_snapshot": source_snapshot,
            "observation": observation_ref}, produce=produce, validate=validate, check=check)
        used = execution_refs(store, result)
        for execution in used.values():
            backend.validate_execution(execution)
        executions.update(used)
        records.append({"unit": unit, "observation": observation_ref, "result": ref,
                        "status": result["status"], "candidate_count": result["candidate_count"]})
    return {"schema": "measurement-gpu-profile-v1", "source_snapshot": source_snapshot,
            "cpu_profile": cpu_profile_ref, "records": records,
            "executions": [executions[k] for k in sorted(executions)],
            "projection_basis": "accumulated ledger cost of all GPU-profile attempts; not cache-hit unit timings"}, {}


def validate_gpu_profile(store, source_snapshot, cpu_profile_ref, result, arrays, check):
    _, cpu, cpu_arrays = _artifact(store, cpu_profile_ref)
    validate_cpu_profile(store, source_snapshot, cpu, cpu_arrays)
    if (set(result) != {"schema", "source_snapshot", "cpu_profile", "records", "executions", "projection_basis"}
            or result["schema"] != "measurement-gpu-profile-v1" or arrays
            or encoded(result["source_snapshot"]) != encoded(source_snapshot)
            or encoded(result["cpu_profile"]) != encoded(cpu_profile_ref)
            or digest([r["unit"] for r in result["records"]]) != digest([r["unit"] for r in cpu["rows"]])):
        raise ValueError("GPU profile does not cover the fixed paired OPEN observations")
    executions = {}
    for row in result["records"]:
        check()
        _, metadata, values = _artifact(store, row["result"])
        state = unpack(metadata, values)
        executions.update(execution_refs(store, state))
        if (digest(state["unit"]) != digest(row["unit"]) or state["status"] != row["status"]
                or state["candidate_count"] != row["candidate_count"]):
            raise ValueError("GPU profile index differs from its persisted unit")
    if encoded(result["executions"]) != encoded([executions[k] for k in sorted(executions)]):
        raise ValueError("GPU profile must retain all original producing execution receipts")
