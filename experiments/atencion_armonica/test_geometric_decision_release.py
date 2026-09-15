"""Four-test structural receipts; no draws, labels, checkpoints or GPU.

Recovery and draw-index semantic validation are explicit stubs here. Their
real implementations have separate full pipeline/once-only tests.
"""
import pytest

from src.atencion_armonica import geometric_decision_release as module
from src.atencion_armonica import geometric_decision_evaluation as evaluation
from src.atencion_armonica.geometric_decision_store import ArtifactStore
from src.atencion_armonica.generative_evidence_exclusions import SCHEMA


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    control = ArtifactStore(tmp_path/"control", binding={"protocol": {"fixture": "protocol"}})
    exclusion = control.publish_json("exclusions.json", {"schema": SCHEMA,
        "status": "EXCLUSIONS_ONLY_NOT_TEST_AUTHORIZED", "test_access": False, "fingerprints": [], "unique_count": 0})
    heads = [{"path": f"heads/{i}.json", "sha256": "0"*64, "bytes": 0} for i in range(144)]
    freeze = control.publish_json("freeze.json", {"schema": "geometric-decision-prospective-freeze-v1",
        "protocol": control.binding["protocol"], "test_roster": module.TEST_ROSTER,
        "head_roster": heads, "fresh_root": str(tmp_path/"fresh"), "exclusions": exclusion})
    store = ArtifactStore(tmp_path/"fresh", binding={"test_freeze": freeze})
    source = store.publish_json("known-empty-source.json", {"scene": {"partitions": []}})
    batches = []
    for split, seed in module.TESTS:
        sources = store.publish_json(f"{split}/sources.json", {"binding": store.binding,
            "split": split, "split_seed": seed, "scene_ids": list(range(512)), "sources": [source]*512})
        observed = store.publish_json(f"{split}/complete.json", {"schema": "geometric-decision-observed-run-v1",
            "binding": store.binding, "split": split, "split_seed": seed, "scene_ids": list(range(512)),
            "truth_access": False, "global_seal": False, "sources": sources,
            "records": [{"head": h} for h in heads], "roundtrip_scene_ids": [], "roundtrip": None})
        draws = store.publish_json(f"{split}/draws.json", {"fixture": "complete draw index"})
        batches.append({"split": split, "draws": draws, "observed": observed})
    def verify_index(self, split, produced, *, check):
        row = next(r for r in batches if r["split"] == split)
        return row["draws"], {}, produced
    monkeypatch.setattr(module.DrawBatch, "verify_index", verify_index)
    def recover(split):
        return next(r["observed"] for r in batches if r["split"] == split)
    seal = module.seal_outputs(control, store, freeze, batches, recover=recover, check=lambda: None)
    manifest = control.publish_json("manifest.json", {"operation": "prospective-observables",
        "test_freeze": freeze, "root": str(store.root)})
    start = control.publish_json("start.json", {"binding": control.binding, "stage": "fresh", "manifest": manifest})
    output = control.publish_json("output.json", {"manifest": manifest, "test_freeze": freeze, "seal": seal})
    finish = control.publish_json("finish.json", {"status": "COMPLETE", "start": start, "completion": output})
    return control, store, freeze, batches, recover, seal, finish, start


def test_complete_seal_verified_and_requires_all_batches(sealed):
    control, store, freeze, batches, recover, seal, finish, _ = sealed
    view, record, ref = module.admitted_seal(control, finish, freeze, check=lambda: None)
    assert ref == seal and record["original_scenes"] == 2048
    assert not hasattr(view, "publish_json")
    for bad in (batches[:-1], batches[::-1]):
        with pytest.raises(ValueError, match="four ordered"):
            module.seal_outputs(control, store, freeze, bad, recover=recover, check=lambda: None)


def test_recovery_must_match_before_any_seal_publication(sealed):
    control, store, freeze, batches, _, _, _, _ = sealed
    with pytest.raises(ValueError, match="recovery changed"):
        module.seal_outputs(control, store, freeze, batches, recover=lambda split: {}, check=lambda: None)


def test_extra_file_after_seal_blocks_label_port(sealed):
    control, store, freeze, _, _, seal, finish, _ = sealed
    store.publish_json("extra.json", {"fixture": "not in seal"})
    output = ArtifactStore(store.root.parent/"evaluation", binding={"test_freeze": freeze, "prediction_seal": seal})
    with pytest.raises(PermissionError, match="tree changed"):
        evaluation.evaluate_fresh(control, finish, freeze, output, check=lambda: None)
    assert not output.path("complete.json").exists()


def test_partial_finish_blocks_label_port_even_with_seal(sealed):
    control, store, freeze, _, _, seal, _, start = sealed
    finish = control.publish_json("paused.json", {"status": "PAUSED", "start": start, "completion": None})
    output = ArtifactStore(store.root.parent/"evaluation", binding={"test_freeze": freeze, "prediction_seal": seal})
    with pytest.raises(PermissionError, match="COMPLETE observable"):
        evaluation.evaluate_fresh(control, finish, freeze, output, check=lambda: None)
    assert not output.path("complete.json").exists()


def test_inventory_hashes_compressed_payload_without_parsing(tmp_path):
    store = ArtifactStore(tmp_path/"opaque", binding={"fixture": "no JSON parse"})
    # A normal immutable writer creates JSON bytes at a .gz-suffixed fixture
    # path; the inventory must neither decompress nor parse its content.
    store.publish_json("opaque.gz", {"fixture": "opaque byte payload"})
    before = module.file_inventory(store, check=lambda: None)
    view = module.ReadOnlyStore(store.root, binding_ref=store.reference(store.path("binding.json")))
    assert module.file_inventory(view, check=lambda: None) == before
