"""Lossless archive checks; no campaign scenes or model execution."""
from copy import deepcopy
import gzip
import hashlib

import numpy as np
import pytest

from src.atencion_armonica import generative_evidence_storage as storage
from src.atencion_armonica.partial_compatibility_cache import encoded


def test_atomic_publication_never_overwrites(tmp_path):
    path = tmp_path/"receipt.json"
    storage.write_json(path, {"status": "COMPLETE", "parent": None})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        storage.write_json(path, {"status": "CHANGED"})
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["receipt.json"]


def test_canonical_gzip_preserves_nested_factors_and_float64(tmp_path):
    value = {"factors": [{"branch": "base-low", "grid": list(range(257)),
                          "witness": [1., 1.+2**-52, -0., float(np.nextafter(0., 1.))]}],
             "inventory": [{"partition": [[0, 1], [2, 3]], "status": "EXCLUDED"}]}
    a, b = tmp_path/"a.json.gz", tmp_path/"b.json.gz"
    ra, rb = storage.write_scene(a, value), storage.write_scene(b, value)
    assert a.read_bytes() == b.read_bytes() and ra == rb
    assert gzip.decompress(a.read_bytes()) == encoded(value)
    assert encoded(storage.read_scene(a, ra)) == encoded(value)
    for key in ("sha256", "decoded_sha256", "bytes", "decoded_bytes"):
        invalid = deepcopy(ra)
        invalid[key] = "0"*64 if "sha256" in key else invalid[key]+1
        with pytest.raises(ValueError, match="identity differs"):
            storage.read_scene(a, invalid)


def test_numeric_archive_keeps_dtype_shape_and_bits(tmp_path):
    arrays = {"raw": np.array([[1.+2**-52, -0.]], np.float64),
              "delivered": np.array([[1., np.nextafter(np.float32(1), np.float32(2))]], np.float32),
              "empty": np.zeros((0, 6), np.float32), "mask": np.array([False, True]),
              "ids": np.arange(4, dtype=np.int64)}
    path = tmp_path/"arrays.npz"
    ref = storage.write_arrays(path, arrays)
    saved = storage.read_arrays(path, ref)
    assert set(saved) == set(arrays)
    for k in arrays:
        assert saved[k].dtype == arrays[k].dtype and saved[k].shape == arrays[k].shape
        assert saved[k].tobytes() == arrays[k].tobytes()
    with pytest.raises(ValueError, match="without pickle"):
        storage.write_arrays(tmp_path/"unsafe.npz", {"obj": np.array([{}], object)})


def test_noncanonical_gzip_and_negative_size_fail_closed(tmp_path):
    raw = encoded({"x": list(range(200))})
    compressed = gzip.compress(raw, compresslevel=1, mtime=123456)
    path = tmp_path/"different-codec.gz"
    receipt = {**storage.atomic_bytes(path, compressed), "decoded_sha256": hashlib.sha256(raw).hexdigest(),
               "decoded_bytes": len(raw), "codec": "canonical-json-gzip3-mtime0"}
    with pytest.raises(ValueError, match="declared canonical codec"):
        storage.read_scene(path, receipt)
    for size in (-2, True, 0.5):
        with pytest.raises(ValueError, match="size schema"):
            storage.read_scene(path, {**receipt, "decoded_bytes": size})
