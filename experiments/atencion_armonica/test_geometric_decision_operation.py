"""Finite OPEN operation over arithmetic ports, with real locks and receipts."""
import fcntl
from pathlib import Path
import time

import pytest

from experiments.atencion_armonica.prepare_geometric_decision_open import execute
from experiments.atencion_armonica.test_geometric_decision_corpus import preparation
from src.atencion_armonica.geometric_decision_budget import LIMITS
from src.atencion_armonica.geometric_decision_corpus import source_identity
from src.atencion_armonica.geometric_decision_store import ArtifactStore


def context(tmp_path):
    prep = preparation(tmp_path)
    control_root = tmp_path/"control"
    control = ArtifactStore(control_root, binding={"limits": LIMITS, "prior_charges": [],
        "output_roots": [str(prep.store.root), str(control_root.resolve())]})
    manifest = control.publish_json("manifest.json", {"operation": "open", "source": source_identity(prep.source),
        "preparation_binding": prep.store.binding})
    return prep, control, manifest


def run(prep, control, manifest, progress=lambda row: None):
    return execute(prep, control, manifest, reservation_seconds=60., started_at=time.monotonic(), progress=progress,
                   resource_overrides={"rss": lambda: 1, "disk_free": lambda: 100*1024**3,
                                       "bytes_used": lambda: 1, "vram": lambda: 0})


def test_operator_retains_pause_and_completes_only_full_preparation(tmp_path):
    prep, control, manifest = context(tmp_path)
    def stop(row):
        raise InterruptedError("explicit pause after first published shard")
    with pytest.raises(InterruptedError):
        run(prep, control, manifest, stop)
    assert control.json(control.reference(control.path("attempts/0000/finish.json")))["status"] == "PAUSED"
    assert not control.path("outputs/open.json").exists()
    result = run(prep, control, manifest)
    finish = control.json(result["finish"])
    assert finish["status"] == "COMPLETE" and finish["seconds"] > 0
    assert control.json(finish["start"])["charged_before"]["open"] > 0
    output = control.json(result["output"])
    assert output["manifest"] == manifest and Path(output["root"]) == prep.store.root
    assert len(prep.completion(output["complete"])["entries"]) == 27
    assert prep.source.target_calls == 0


def test_concurrent_operator_cannot_create_attempt(tmp_path):
    prep, control, manifest = context(tmp_path)
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run(prep, control, manifest)
    assert not control.path("attempts").exists()


def test_wrong_manifest_prevents_execution(tmp_path):
    prep, control, manifest = context(tmp_path)
    wrong = {**control.json(manifest), "operation": "training"}
    bad_ref = control.publish_json("wrong-fixture.json", wrong)
    with pytest.raises(ValueError, match="does not bind"):
        run(prep, control, bad_ref)
    assert prep.source.scale_calls == prep.source.target_calls == 0
