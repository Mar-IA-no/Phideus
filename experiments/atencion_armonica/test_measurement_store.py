import numpy as np
import pytest

from src.atencion_armonica import measurement_store as module


@pytest.fixture
def store(tmp_path):
    # pytest basetemp MUST live under the protocol's project-owned fixture root.
    return module.MeasurementStore(tmp_path/"store", binding={"schema": "mechanical", "fixture": 1})


def test_publication_replay_and_identity(store):
    identity = {"stage": "fixture", "inputs": ["pinned-source"]}
    arrays = {"value": np.arange(8, dtype=np.float32)}
    with store.exclusive():
        assert store.completed("case", identity) is None
        ref = store.publish_stage("case", identity, {"status": "OK"}, arrays)
        same, result, saved = store.completed("case", identity)
        assert ref == same and result == {"status": "OK"}
        np.testing.assert_array_equal(saved["value"], arrays["value"])
        assert store.publish_stage("case", identity, result, arrays) == ref
        with pytest.raises(ValueError):
            store.completed("case", {"stage": "changed"})
        with pytest.raises(ValueError):
            store.publish_stage("case", identity, {"status": "other"}, arrays)


def test_orphan_payload_recovers_without_computation(store, monkeypatch):
    original = store.publish_json
    def interrupted(relative, value):
        if relative.endswith("complete.json"):
            raise InterruptedError("fixture interrupt after payload publication")
        return original(relative, value)
    with store.exclusive():
        monkeypatch.setattr(store, "publish_json", interrupted)
        with pytest.raises(InterruptedError):
            store.publish_stage("case", {"stage": "forward"}, {"computed_once": True},
                                {"logits": np.eye(2, dtype=np.float32)})
        assert store.path("case/payload.npz").exists()
        assert not store.path("case/complete.json").exists()
        monkeypatch.setattr(store, "publish_json", original)
        ref, result, arrays = store.completed("case", {"stage": "forward"})
        assert result["computed_once"] and ref["path"] == "case/complete.json"
        np.testing.assert_array_equal(arrays["logits"], np.eye(2, dtype=np.float32))


def test_lock_and_nonadoption(store):
    with pytest.raises(RuntimeError):
        store.publish_stage("case", {"stage": "x"}, {}, {})
    rival = module.MeasurementStore(store.root, binding=store.binding)
    with store.exclusive():
        with pytest.raises(BlockingIOError):
            with rival.exclusive():
                raise AssertionError("rival acquired the same store")
        with pytest.raises(RuntimeError):
            with store.exclusive():
                pass
    with pytest.raises(ValueError):
        module.MeasurementStore(store.root, binding={"fixture": 2})


def test_path_and_hash_guards(store):
    for path in ("../x", "/tmp/x", ".", "a/../b", "a\\b"):
        with pytest.raises(ValueError):
            store.path(path)
    with store.exclusive():
        ref = store.publish_json("record.json", {"a": 1})
        with pytest.raises(ValueError):
            store.read({**ref, "sha256": "0"*64})
        with pytest.raises(ValueError):
            store.publish_json("record.json", {"a": 2})


def test_partial_bytes_do_not_count_as_complete(store):
    with store.exclusive():
        store.publish_json("case/.payload.fixture.partial", {"not": "a payload"})
        assert store.completed("case", {"stage": "x"}) is None
        assert store.path("case/.payload.fixture.partial").exists()
