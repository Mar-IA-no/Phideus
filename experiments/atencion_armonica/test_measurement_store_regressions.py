import numpy as np
import pytest

from src.atencion_armonica.measurement_store import MeasurementStore
from src.atencion_armonica.measurement_stage import run_stage


def test_binding_accessor_cannot_mutate_store(tmp_path):
    store = MeasurementStore(tmp_path/"store", binding={"run": 1, "nested": {"x": 2}})
    borrowed = store.binding
    borrowed["run"] = 4
    borrowed["nested"]["x"] = 5
    assert store.binding == {"run": 1, "nested": {"x": 2}}


@pytest.mark.parametrize("collision", [True, 1.0])
def test_binding_identity_uses_canonical_bytes(tmp_path, collision):
    store = MeasurementStore(tmp_path/"store", binding={"run": 1})
    with pytest.raises(ValueError, match="binding"):
        MeasurementStore(store.root, binding={"run": collision})


@pytest.mark.parametrize("collision", [True, 1.0])
def test_stage_identity_uses_canonical_bytes(tmp_path, collision):
    store = MeasurementStore(tmp_path/"store", binding={"fixture": 1})
    with store.exclusive():
        run_stage(store, "stage", {"unit": 1}, authorize=lambda _: None,
                  produce=lambda: ({}, {"x": np.arange(2)}), check=lambda: None)
        with pytest.raises(ValueError):
            store.completed("stage", {"unit": collision})
        with pytest.raises(ValueError):
            run_stage(store, "stage", {"unit": collision}, authorize=lambda _: None,
                      produce=lambda: (_ for _ in ()).throw(AssertionError("no repeat")), check=lambda: None)


@pytest.mark.parametrize("name", ["file", "allow_pickle"])
def test_logical_array_names_do_not_collide_with_numpy_keywords(tmp_path, name):
    store = MeasurementStore(tmp_path/"store", binding={"fixture": 1})
    with store.exclusive():
        store.publish_stage("stage", {"unit": 1}, {}, {name: np.arange(3)})
        _, _, arrays = store.completed("stage", {"unit": 1})
        assert set(arrays) == {name}
        np.testing.assert_array_equal(arrays[name], np.arange(3))


def test_root_cannot_be_redirected_under_another_lock(tmp_path):
    a = MeasurementStore(tmp_path/"a", binding={"fixture": 1})
    b = MeasurementStore(tmp_path/"b", binding={"fixture": 2})
    with a.exclusive():
        with pytest.raises(AttributeError):
            a.root = b.root
        a.publish_json("owned.json", {"a": True})
    assert a.path("owned.json").exists() and not b.path("owned.json").exists()


def test_pinned_binding_file_checked_on_lock_entry(tmp_path):
    store = MeasurementStore(tmp_path/"store", binding={"fixture": 1})
    # Deliberately corrupt only a pytest-owned fixture, not a campaign artifact.
    store.path("binding.json").write_bytes(b'{"fixture":2}\n')
    with pytest.raises(ValueError, match="binding|pinned"):
        with store.exclusive():
            raise AssertionError("changed binding was admitted")


def test_exclusive_lock_in_an_independent_process(tmp_path):
    import subprocess
    import sys
    store = MeasurementStore(tmp_path/"store", binding={"fixture": 1})
    script = '''
import sys
from src.atencion_armonica.measurement_store import MeasurementStore
s = MeasurementStore(sys.argv[1], binding={"fixture": 1})
try:
    with s.exclusive():
        acquired = True
except BlockingIOError:
    acquired = False
assert acquired == (sys.argv[2] == "free")
'''
    with store.exclusive():
        subprocess.run([sys.executable, "-c", script, str(store.root), "held"], check=True, timeout=10)
    subprocess.run([sys.executable, "-c", script, str(store.root), "free"], check=True, timeout=10)
