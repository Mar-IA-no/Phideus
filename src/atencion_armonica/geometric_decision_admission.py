"""Authenticate completed head profiles and project the unchanged full campaign."""
from __future__ import annotations

from pathlib import Path
import hashlib

from .geometric_decision_open import ReadOnlyStore
from .geometric_decision_store import ArtifactStore
from .geometric_decision_work import head_forecast
from .partial_compatibility_cache import encoded


class ProfileReadOnlyStore(ReadOnlyStore):
    """Authenticate local .pt bytes without unpickling or any write methods."""
    path = ArtifactStore.path

HEAD_FINISHES = {
    "cpu": {"path": "attempts/0002/finish.json", "bytes": 502,
            "sha256": "e79e8497fe69e4c5d52abbff5c94da091d7619bca22d0d8b324166d2a70249b8"},
    "cuda:0": {"path": "attempts/0004/finish.json", "bytes": 504,
            "sha256": "daa4a3d741f9224a519facf4c72046cc3b22bb447d3d479f7920cfa9efc13d37"},
}


def head_admission(control, input_provenance, *, profile_root, finishes=HEAD_FINISHES):
    if set(finishes) != {"cpu", "cuda:0"}:
        raise ValueError("both backend profiles are required")
    expected_cases = {(kind, objective) for kind in ("envelope", "first_train_batch") for objective in ("mse", "decision")}
    forecasts, evidence, runtimes = {}, {}, {}
    load = input_provenance["checkpoint_load"]
    for device, finish_ref in finishes.items():
        finish = control.json(finish_ref)
        if finish["status"] != "COMPLETE" or finish["completion"] is None:
            raise ValueError("head profile did not finish COMPLETE")
        start = control.json(finish["start"])
        output = control.json(finish["completion"])
        manifest = control.json(start["manifest"])
        root = Path(profile_root)/f"head-{device.replace(':', '-')}"
        if (start["stage"] != "profile" or start["binding"] != control.binding
                or output["manifest"] != start["manifest"] or output["root"] != str(root)
                or manifest["operation"] != "profile-head" or manifest["root"] != str(root)):
            raise ValueError("head profile start/output/manifest differs")
        view = ReadOnlyStore(root, binding_ref=output["binding"])
        if (view.binding != manifest["binding"] or view.binding["runtime"]["device"] != device
                or view.binding["input_provenance"] != input_provenance):
            raise ValueError("head profile does not bind the common prepared batch/runtime")
        report = view.json(output["result"])
        if (report["binding"] != view.binding or len(report["cases"]) != 4
                or {(c["case"], c["objective"]) for c in report["cases"]} != expected_cases):
            raise ValueError("head profile must include all four cases once")
        cases, refs = {}, []
        for case in report["cases"]:
            relative = f"{case['case']}-{case['objective']}"
            if case["root"] != str(root/relative):
                raise ValueError("profile case root outside the declared roster")
            # Case result is pinned by its parent report; read through the
            # parent's authenticated port before adopting any child identity.
            def nested(ref):
                return {**ref, "path": relative+"/"+ref["path"]}
            value = view.json(nested(case["result"]))
            expected_binding = {**view.binding, "input_provenance": input_provenance,
                                "case": case["case"], "objective": case["objective"]}
            if value["binding"] != expected_binding or value["device"] != device:
                raise ValueError("profile case binding changed")
            binding_bytes = encoded(expected_binding)
            child = ProfileReadOnlyStore(root/relative, binding_ref={"path": "binding.json",
                "bytes": len(binding_bytes), "sha256": hashlib.sha256(binding_bytes).hexdigest()})
            previous = None
            for key, steps in (("initial", 0), ("middle", 10), ("last", 25)):
                ref = value[key]
                record = view.json(nested(ref))
                if (record["binding"] != expected_binding or record["steps"] != steps or record["previous"] != previous):
                    raise ValueError("profile recovery snapshot chain differs")
                child.read(record["state"])
                previous = ref
            if record["state_digest"] != value["exact_recovery_digest"]:
                raise ValueError("head profile recovery digest differs from final state")
            for ref in value["outputs"]:
                view.arrays(nested(ref))
            cases[case["case"], case["objective"]] = value
            refs.append(case)
        forecasts[device] = head_forecast(cases, eligible_train=load["eligible_train"],
            eligible_calibration=load["eligible_calibration"], corpus_load_seconds=load["seconds"])
        evidence[device] = {"finish": finish_ref, "output": output, "cases": refs}
        runtimes[device] = view.binding["runtime"]
    chosen = min(forecasts, key=lambda d: (forecasts[d]["projected_seconds_before_margin"], d))
    return {"schema": "geometric-decision-head-admission-v1", "device": chosen,
        "forecasts": forecasts, "profiles": evidence, "input_provenance": input_provenance,
        "profile_runtime": runtimes[chosen], "margin": 1.25,
        "projected_seconds": 1.25*forecasts[chosen]["projected_seconds_before_margin"],
        "scope": "head training only; no fresh pipeline admission or scientific promotion"}
