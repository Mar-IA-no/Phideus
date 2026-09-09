"""CPU exclusion mechanics; no fresh observations, models, or sidecar access."""
import hashlib
import json

import numpy as np
import pytest

from src.atencion_armonica import generative_evidence_exclusions as e
from src.atencion_armonica.generative_evidence_reuse import VerifiedBytes
from src.atencion_armonica.generative_evidence_storage import write_json


def ref(path, root):
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_fingerprint_is_exact_q32_permutation_not_gauge():
    q = np.arange(8, dtype=np.float32)/8
    assert e.fingerprint(q) == hashlib.sha256(q.astype("<f4").tobytes()).hexdigest()
    assert e.fingerprint(q[::-1]) == e.fingerprint(q.astype(">f4"))
    assert e.fingerprint(q+2) != e.fingerprint(q)
    for bad in ([], [[0, 1]], [np.nan], [np.inf], [.1], ["0", "1"]):
        with pytest.raises(ValueError):
            e.fingerprint(bad)


def test_aliases_keep_origin_without_adding_samples(tmp_path):
    path = tmp_path/"source.json"
    write_json(path, {"fixture": "arithmetic"})
    source = ref(path, tmp_path)
    inv = e.Inventory(VerifiedBytes(tmp_path))
    q = np.arange(8, dtype=np.float32)/8
    inv.add("first", [q], source=source)
    inv.add("permuted", [q[::-1]], source=source, alias=True)
    result = inv.result()
    assert result["unique_count"] == 1 and len(result["groups"]) == 2
    assert result["groups"][1]["additional_unique"] == 0
    assert result["consumed_sha256"] == {source["path"]: source["sha256"]}
    assert result["test_access"] is False
    with pytest.raises(ValueError, match="alias"):
        inv.add("false_alias", [q+2], source=source, alias=True)
    with pytest.raises(ValueError, match="unique"):
        inv.add("first", [q], source=source)
    with pytest.raises(ValueError, match="hash"):
        inv.add("wrong_source", [q], source={**source, "sha256": "0"*64})


def test_duplicate_check_includes_already_produced_without_mutation(tmp_path):
    path = tmp_path/"source.json"
    write_json(path, {})
    q = np.arange(8, dtype=np.float32)/8
    inv = e.Inventory(VerifiedBytes(tmp_path))
    inv.add("first", [q], source=ref(path, tmp_path))
    frozen = inv.result()
    assert e.duplicate_matches(q[::-1], frozen)["matches_frozen"]
    shifted = e.fingerprint(q+2)
    assert not e.duplicate_matches(q+2, frozen)["duplicate"]
    assert e.duplicate_matches(q+2, frozen, [shifted])["matches_produced"]
    assert frozen["unique_count"] == 1
    with pytest.raises(ValueError):
        e.duplicate_matches(q, {**frozen, "unique_count": 2})


def test_real_inventory_reads_no_privileged_payloads(tmp_path, monkeypatch):
    # Intentionally incomplete fixture catalog: tests the real inherited corpus
    # adapter, NOT authority/completeness of a campaign freeze.
    catalog_path = tmp_path/"catalog.json"
    source_path = e.ROOT/"experiments/atencion_armonica/test_generative_evidence.py"
    write_json(catalog_path, {"schema": "generative-evidence-mechanical-exclusions-v1",
        "scope": "INCOMPLETE TEST FIXTURE; NOT A CAMPAIGN FREEZE", "records": [{
            "name": "arithmetic8", "source": ref(source_path, e.ROOT),
            "q32": [(np.arange(8, dtype=np.float32)/8).tolist()], "alias": False}]})
    original = VerifiedBytes.read
    consumed = []
    def guarded(self, source):
        assert not any(s in source["path"] for s in ("sidecars", "targets.npz", "metrics.json", "_pool.json"))
        consumed.append(source["path"])
        return original(self, source)
    monkeypatch.setattr(VerifiedBytes, "read", guarded)
    result = e.build_exclusions(ref(catalog_path, e.ROOT))
    groups = {r["name"]: r for r in result["groups"]}
    assert sum(g["count"] for n, g in groups.items() if n.startswith("historical_role_")) == 15680
    assert sum(g["count"] for n, g in groups.items() if n.startswith("open_")) == 4608
    assert sum(g["count"] for n, g in groups.items() if n.startswith("released_")) == 2048
    assert groups["rivals_96_alias"]["count"] == 96
    assert groups["rivals_96_alias"]["additional_unique"] == 0
    assert len(result["fingerprints"]) == result["unique_count"]
    assert catalog_path.relative_to(e.ROOT).as_posix() in consumed
