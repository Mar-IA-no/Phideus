"""Read-only source-boundary fixtures; no actual corpus, sampler or GPU."""
import hashlib
from io import BytesIO

import numpy as np
import pytest

from src.atencion_armonica.geometric_decision_open import OpenSource, ReadOnlyStore
from src.atencion_armonica.partial_compatibility_cache import encoded


def write_fixture(root, name, raw):
    # Test fixture, deliberately outside all historical stores.
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_reader_authenticates_same_bytes_without_writes(tmp_path):
    binding = write_fixture(tmp_path, "binding.json", encoded({"fixture": "read-only"}))
    ref = write_fixture(tmp_path, "example.json", encoded({"observation": "arithmetic fixture"}))
    stream = BytesIO()
    np.savez_compressed(stream, value=np.arange(8, dtype=np.float32))
    arrays = write_fixture(tmp_path, "example.npz", stream.getvalue())
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    reader = ReadOnlyStore(tmp_path, binding_ref=binding)
    assert reader.json(ref) == {"observation": "arithmetic fixture"}
    np.testing.assert_array_equal(reader.arrays(arrays)["value"], np.arange(8, dtype=np.float32))
    assert reader.consumed == {r["path"]: r for r in (binding, ref, arrays)}
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert not hasattr(reader, "materialize") and not hasattr(reader, "save_fit")


def test_source_requires_existing_directory_and_pinned_binding(tmp_path):
    with pytest.raises(ValueError, match="already exist"):
        ReadOnlyStore(tmp_path/"missing", binding_ref={})
    assert not (tmp_path/"missing").exists()
    ref = write_fixture(tmp_path, "binding.json", encoded({"fixture": True}))
    wrong = {**ref, "sha256": "0"*64}
    with pytest.raises(ValueError, match="bytes differ"):
        ReadOnlyStore(tmp_path, binding_ref=wrong)


def test_source_rejects_traversal_symlinks_and_unsupported_files(tmp_path):
    binding = write_fixture(tmp_path, "binding.json", encoded({"fixture": True}))
    reader = ReadOnlyStore(tmp_path, binding_ref=binding)
    for name in ("../outside.json", "/outside.json", "foo/../binding.json", "binding.json/", "sidecar.json.gz"):
        with pytest.raises(ValueError):
            reader.path(name)
    (tmp_path/"alias.json").symlink_to(tmp_path/"binding.json")
    with pytest.raises(ValueError, match="symlink"):
        reader.path("alias.json")
    (tmp_path/"directory-link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        reader.path("directory-link/binding.json")


def test_source_rejects_noncanonical_json_and_pickle_arrays(tmp_path):
    binding = write_fixture(tmp_path, "binding.json", encoded({"fixture": True}))
    reader = ReadOnlyStore(tmp_path, binding_ref=binding)
    noncanonical = write_fixture(tmp_path, "noncanonical.json", b'{"x": 1}')
    with pytest.raises(ValueError, match="canonical"):
        reader.json(noncanonical)
    stream = BytesIO()
    np.savez(stream, bad=np.array([{"arbitrary": "object"}], dtype=object))
    ref = write_fixture(tmp_path, "object.npz", stream.getvalue())
    with pytest.raises(ValueError):
        reader.arrays(ref)


@pytest.mark.parametrize("split", ["iid", "ood_beta", "ood_polyphony", "deformed_family", "unknown"])
def test_no_fresh_test_role(split):
    with pytest.raises(PermissionError):
        OpenSource._role(split, 0)


def test_fixed_open_extent():
    assert OpenSource._role("train", 7) is None
    assert OpenSource._role("calibration", 0) is None
    for split, shard in (("train", 8), ("calibration", 1), ("train", True), ("train", -1)):
        with pytest.raises(ValueError):
            OpenSource._role(split, shard)
