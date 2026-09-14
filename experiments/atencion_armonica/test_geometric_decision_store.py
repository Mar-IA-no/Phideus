"""Owned-store recovery tests; explicit fixtures only, never a sampled campaign."""
import numpy as np
import pytest

from experiments.atencion_armonica.test_geometric_decision_training import kernel, runtime, step
from src.atencion_armonica.generative_evidence_cell import state_digest
from src.atencion_armonica.geometric_decision_store import ArtifactStore


def test_atomic_checkpoint_chain_and_exact_resume(tmp_path):
    cell = kernel()
    store = ArtifactStore(tmp_path/"cell", binding=cell.binding)
    initial = store.save_state(cell, None)
    assert store.save_state(cell, None) == initial
    step(cell)
    middle = store.save_state(cell, initial)
    assert store.save_state(cell, initial) == middle
    with pytest.raises(ValueError, match="parent"):
        store.save_state(cell, None)
    step(cell)
    end = store.save_state(cell, middle)
    restored = kernel()
    assert store.latest(restored) == end
    assert state_digest(restored.state()) == state_digest(cell.state())
    step(restored)
    step(cell)
    assert state_digest(restored.state()) == state_digest(cell.state())


def test_snapshot_missing_parent_is_not_silently_skipped(tmp_path):
    cell = kernel()
    store = ArtifactStore(tmp_path/"cell", binding=cell.binding)
    initial = store.save_state(cell, None)
    step(cell)
    store.save_state(cell, initial)
    store.path(initial["path"]).rename(store.path("saved-initial.json"))
    with pytest.raises(ValueError, match="chain"):
        store.latest(kernel())


def test_changed_state_blob_rejected_before_restore(tmp_path):
    cell = kernel()
    store = ArtifactStore(tmp_path/"cell", binding=cell.binding)
    initial = store.save_state(cell, None)
    state_ref = store.json(initial)["state"]
    store.path(state_ref["path"]).write_bytes(b"explicit corrupted fixture")
    with pytest.raises(ValueError, match="pinned"):
        store.latest(kernel())


def test_owned_store_rejects_adoption_traversal_symlinks(tmp_path):
    stray = tmp_path/"stray"
    stray.mkdir()
    (stray/"unowned").touch()
    with pytest.raises(ValueError, match="unbound"):
        ArtifactStore(stray, binding={"fixture": True})
    with pytest.raises(ValueError, match="owned"):
        ArtifactStore(tmp_path/".."/"other", binding={"fixture": True})
    store = ArtifactStore(tmp_path/"owned", binding={"fixture": True})
    with pytest.raises(ValueError, match="binding"):
        ArtifactStore(store.root, binding={"fixture": False})
    for name in ("../escape", "/absolute", "a//b", "."):
        with pytest.raises(ValueError):
            store.path(name)
    store.path("linked").symlink_to(stray, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        store.path("linked/child")


def test_immutable_json_arrays_and_receipts(tmp_path):
    store = ArtifactStore(tmp_path/"owned", binding={"fixture": True})
    ref = store.publish_json("scale/index.json", {"scale": 1.})
    assert store.publish_json("scale/index.json", {"scale": 1.}) == ref
    with pytest.raises(ValueError, match="replace"):
        store.publish_json("scale/index.json", {"scale": 2.})
    arrays = {"components": np.array([[-1., 2.]], np.float64)}
    blob = store.publish_arrays("outputs/predictions.npz", arrays)
    np.testing.assert_array_equal(store.arrays(blob)["components"], arrays["components"])
    with pytest.raises(FileExistsError):
        store.publish_arrays("outputs/predictions.npz", arrays)
    with pytest.raises(ValueError):
        store.read({**ref, "bytes": True})
