"""Recipe checks with a fake receipt store; no historical data or sampler."""
from copy import deepcopy

import numpy as np
import pytest

from src.atencion_armonica import geometric_decision_mechanical_catalog as module
from src.atencion_armonica.generative_evidence_exclusions import SCHEMA, fingerprint


class Reader:
    def __init__(self):
        self.consumed = {}
        self.q = (np.arange(8, dtype=np.float32)/8).tolist()

    def read(self, ref):
        self.consumed[ref["path"]] = ref["sha256"]
        return b"explicit-fake-receipt-not-real-source"

    def json(self, ref):
        self.read(ref)
        if ref == module.PRIOR:
            h = fingerprint(self.q)
            return {"schema": SCHEMA, "status": "EXCLUSIONS_ONLY_NOT_TEST_AUTHORIZED", "test_access": False,
                "groups": [{"name": "fixture", "source": {"path": "fixture.py", "sha256": "0"*64},
                    "count": 1, "unique_count": 1, "additional_unique": 1, "declared_alias": False, "fingerprints": [h]}],
                "fingerprints": [h], "unique_count": 1, "consumed_sha256": {"fixture.py": "0"*64}}
        if ref == module.INCIDENT:
            return {"status": "DETERMINISTIC_RECONSTRUCTION_NOT_ORIGINAL_BYTE_PRESERVATION",
                "reconstructed": {"observation": {"path": "observation.json", "sha256": "1"*64},
                    "observation_q32_sorted_sha256": fingerprint(self.q)}}
        assert ref["path"].endswith("/observation.json")
        return {"scene_id": 0, "split_seed": module.RETIRED, "log_f": self.q}


def test_recipe_covers_prior_alias_integer16_and_declared_roundtrips():
    reader = Reader()
    refs = [{"path": p, "sha256": "0"*64} for p in module.SOURCE_PATHS]
    actual = module.build_catalog(reader, refs, check=lambda: None)
    records = {r["name"]: r for r in actual["records"]}
    assert records["shared_arithmetic8"]["alias"] is True
    assert records["store_roundtrip_alias"]["alias"] is True
    assert records["retired_iid0_reconstruction"]["alias"] is True
    np.testing.assert_array_equal(records["operator_integer8_16"]["q32"][1], np.arange(16, dtype=np.float32))
    for name in ("core_roundtrip", "tied_roundtrip", "pipeline_roundtrip"):
        q, shifted, center, probe = [np.asarray(v, np.float32) for v in records[name]["q32"]]
        np.testing.assert_array_equal(shifted, (q.astype(np.float64)+np.log(2.)).astype(np.float32))
        np.testing.assert_array_equal(center, (q.astype(np.float64)-q.astype(np.float64).mean()).astype(np.float32))
        np.testing.assert_array_equal(probe, (shifted.astype(np.float64)-shifted.astype(np.float64).mean()).astype(np.float32))
    assert all(p in reader.consumed for p in module.SOURCE_PATHS)
    assert "Byte exclusions" in actual["scope"]


def test_recipe_requires_exact_source_roster():
    refs = [{"path": p, "sha256": "0"*64} for p in module.SOURCE_PATHS]
    with pytest.raises(ValueError, match="source roster"):
        module.build_catalog(Reader(), list(reversed(refs)), check=lambda: None)
