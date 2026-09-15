"""Small manually authored CPU fixtures; no historical model or sampler call."""
import hashlib
from types import SimpleNamespace

import pytest

from src.atencion_armonica import measurement_resources as resources
from src.atencion_armonica import measurement_snapshot as snapshots
from src.atencion_armonica import measurement_reuse as reuse
from src.atencion_armonica.measurement_store import MeasurementStore


def test_resource_guard_rate_limit_and_forced_boundaries(tmp_path, monkeypatch):
    root = tmp_path/"artifacts"
    root.mkdir()
    (root/"tiny.bin").write_bytes(b"123")
    monkeypatch.setattr(resources.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=1))
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    guard = resources.ResourceGuard(root, ram_bytes=2048, artifact_bytes=5, minimum_free_bytes=10)
    assert guard()["artifact_bytes"] == 3
    assert guard()["vram_peak_bytes"] is None
    (root/"tiny.bin").write_bytes(b"123456")
    assert guard()["artifact_bytes"] == 3  # Explicitly cached, not continuous enforcement.
    with pytest.raises(OSError, match="artifact budget"):
        guard(force=True)


def test_resource_ram_disk_gpu_boundaries_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(resources.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=1))
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    with pytest.raises(MemoryError, match="RSS"):
        resources.ResourceGuard(tmp_path, ram_bytes=1)()
    with pytest.raises(OSError, match="disk safety"):
        resources.ResourceGuard(tmp_path, minimum_free_bytes=101)()
    calls = []
    def gpu_fixture():
        calls.append(1)
        return 7
    guard = resources.ResourceGuard(tmp_path, minimum_free_bytes=10, vram_bytes=6, gpu_peak=gpu_fixture)
    with pytest.raises(MemoryError, match="GPU"):
        guard()
    assert calls == [1]  # Manual integer callback, no CUDA library imported.


def test_source_snapshot_pins_bytes_paths_runtime_and_reuse(tmp_path, monkeypatch):
    root = tmp_path/"fake-source"
    root.mkdir()
    source = root/"fixture.py"
    source.write_bytes(b"fixture = 1\n")
    monkeypatch.setattr(snapshots, "ROOT", root)
    monkeypatch.setattr(reuse, "ROOT", root)
    monkeypatch.setattr(snapshots, "source_paths", lambda: ["fixture.py"])
    monkeypatch.setattr(snapshots, "load_reuse", lambda: {"references": [{"fixture": "no-model"}]})
    monkeypatch.setattr(snapshots, "runtime_versions", lambda: {"fixture": "1"})
    store = MeasurementStore(tmp_path/"store", binding={"fixture": "snapshot"})
    with store.exclusive():
        captured = snapshots.capture()
        assert captured["files"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        ref = store.publish_json("snapshot.json", captured)
        assert snapshots.verify(store, ref) == captured
        source.write_bytes(b"fixture = 2\n")
        with pytest.raises(ValueError, match="artifact changed"):
            snapshots.verify(store, ref)
        source.write_bytes(b"fixture = 1\n")
        monkeypatch.setattr(snapshots, "runtime_versions", lambda: {"fixture": "2"})
        with pytest.raises(ValueError, match="runtime changed"):
            snapshots.verify(store, ref)
        monkeypatch.setattr(snapshots, "runtime_versions", lambda: {"fixture": "1"})
        monkeypatch.setattr(snapshots, "load_reuse", lambda: {"references": []})
        with pytest.raises(ValueError, match="reuse inventory"):
            snapshots.verify(store, ref)
        monkeypatch.setattr(snapshots, "source_paths", lambda: ["fixture.py", "new.py"])
        with pytest.raises(ValueError, match="source closure"):
            snapshots.verify(store, ref)
