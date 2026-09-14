"""Pinned arithmetic profile receipts, without GPU or real experimental inputs."""
from copy import deepcopy

import numpy as np
import pytest

from src.atencion_armonica.geometric_decision_store import ArtifactStore
from src.atencion_armonica.geometric_decision_admission import head_admission


def fixtures(tmp_path):
    control = ArtifactStore(tmp_path/"control", binding={"fixture": "control"})
    provenance = {"fixture": "common-input", "checkpoint_load": {
        "eligible_train": 4036, "eligible_calibration": 503, "seconds": 29.}}
    finishes = {}
    for index, device in enumerate(("cpu", "cuda:0")):
        root = tmp_path/"profiles"/f"head-{device.replace(':', '-')}"
        binding = {"runtime": {"device": device}, "input_provenance": provenance}
        head = ArtifactStore(root, binding=binding)
        cases = []
        for kind in ("envelope", "first_train_batch"):
            for objective in ("mse", "decision"):
                child_binding = {**binding, "case": kind, "objective": objective}
                child = ArtifactStore(root/f"{kind}-{objective}", binding=child_binding)
                previous, snapshots = None, {}
                for key, step in (("initial", 0), ("middle", 10), ("last", 25)):
                    # An authenticated JSON blob suffices for this reader test:
                    # the admission port must never unpickle a profile state.
                    blob = child.publish_json(f"blob-{step}.json", {"fixture": "not-a-real-checkpoint", "step": step})
                    previous = child.publish_json(f"state-{step}.json", {"binding": child_binding,
                        "steps": step, "previous": previous, "state": blob, "state_digest": "fixture-digest"})
                    snapshots[key] = previous
                arrays = child.publish_arrays("eval.npz", {"components": np.zeros((32, 1, 2), np.float64)})
                rate = .02 if device == "cpu" else .01
                report = {"status": "MECHANICAL_PROFILE_NOT_TRAINED_CELL", "binding": child_binding,
                    "objective": objective, "device": device, "all_update_seconds": [rate]*25,
                    "snapshot_io_seconds": [.01, .02], "evaluation_batch_io_seconds": [.01]*3,
                    "setup_seconds": .02, "exact_recovery_digest": "fixture-digest", "outputs": [arrays], **snapshots}
                result = child.publish_json("result.json", report)
                cases.append({"case": kind, "objective": objective, "root": str(child.root), "result": result})
        result = head.publish_json("result.json", {"binding": binding, "cases": cases})
        manifest = control.publish_json(f"manifest-{index}.json", {"operation": "profile-head", "binding": binding, "root": str(root)})
        start = control.publish_json(f"start-{index}.json", {"stage": "profile", "binding": control.binding, "manifest": manifest})
        output = control.publish_json(f"output-{index}.json", {"manifest": manifest, "root": str(root),
            "binding": head.reference(head.path("binding.json")), "result": result})
        finishes[device] = control.publish_json(f"finish-{index}.json", {"status": "COMPLETE", "start": start, "completion": output})
    return control, provenance, finishes


def test_both_authenticated_profiles_select_lowest_full_cost(tmp_path):
    control, provenance, finishes = fixtures(tmp_path)
    result = head_admission(control, provenance, profile_root=tmp_path/"profiles", finishes=finishes)
    assert result["device"] == "cuda:0"
    assert result["projected_seconds"] == 1.25*result["forecasts"]["cuda:0"]["projected_seconds_before_margin"]
    assert result["forecasts"]["cpu"]["work"]["updates"] == 457200
    assert result["profiles"]["cpu"]["finish"] == finishes["cpu"]


def test_partial_finish_or_different_common_batch_rejected(tmp_path):
    control, provenance, finishes = fixtures(tmp_path)
    bad = deepcopy(provenance)
    bad["fixture"] = "another-input"
    with pytest.raises(ValueError, match="common prepared"):
        head_admission(control, bad, profile_root=tmp_path/"profiles", finishes=finishes)
    incomplete = control.json(finishes["cuda:0"])
    incomplete["status"] = "PAUSED"
    finishes["cuda:0"] = control.publish_json("paused-fixture.json", incomplete)
    with pytest.raises(ValueError, match="COMPLETE"):
        head_admission(control, provenance, profile_root=tmp_path/"profiles", finishes=finishes)


def test_tampered_case_blob_rejected_before_forecast(tmp_path):
    control, provenance, finishes = fixtures(tmp_path)
    head_root = tmp_path/"profiles/head-cpu/envelope-mse"
    blob = head_root/"blob-25.json"
    blob.rename(head_root/"preserved-fixture-blob.json")
    with pytest.raises(FileNotFoundError):
        head_admission(control, provenance, profile_root=tmp_path/"profiles", finishes=finishes)
