"""Once-only observable draw mechanics under externally admitted freeze/lock.

No default sampler is exposed. The real operator must verify its complete
freeze and resource profile before supplying the historical producer. A
receipt is not sampling authority. Sidecars are saved and hashed, never read
as JSON by this observable port; incomplete pairs stop without a new draw.
"""
from __future__ import annotations

from .generative_evidence_exclusions import SCHEMA, fingerprint
from .geometric_decision_observables import validate_observation, canonical_observation
from .partial_compatibility_cache import encoded

TESTS = (("iid", 2026091582), ("ood_beta", 2026091483),
         ("ood_polyphony", 2026091484), ("deformed_family", 2026091485))


def exclusion_set(value):
    hashes = value["fingerprints"]
    if (value["schema"] != SCHEMA or value["status"] != "EXCLUSIONS_ONLY_NOT_TEST_AUTHORIZED"
            or value["test_access"] is not False or not isinstance(hashes, list)
            or hashes != sorted(set(hashes)) or len(hashes) != value["unique_count"]
            or any(type(h) is not str or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes)):
        raise ValueError("invalid frozen exclusion fingerprint roster")
    return set(hashes)


class DrawBatch:
    """Store mechanics, deliberately not a freeze-verification capability."""
    def __init__(self, store, freeze_ref, *, exclusions):
        if store.binding.get("test_freeze") != freeze_ref:
            raise ValueError("draw store belongs to another freeze")
        self.store, self.freeze = store, dict(freeze_ref)
        self.excluded = exclusion_set(exclusions)

    def intent(self, split, scene_id):
        if split not in dict(TESTS) or type(scene_id) is not int or not 0 <= scene_id < 512:
            raise ValueError("draw outside the fixed prospective roster")
        return {"schema": "geometric-decision-draw-v1", "binding": self.store.binding,
            "test_freeze": self.freeze, "split": split, "split_seed": dict(TESTS)[split], "scene_id": scene_id}

    def reopen(self, split, scene_id, produced):
        prefix = f"draws/{split}/{scene_id:05d}"
        intent = self.intent(split, scene_id)
        s = self.store
        if s.json(s.reference(s.path(prefix+"/intent.json"))) != intent:
            raise ValueError("draw intent differs from frozen roster")
        obs_ref = s.reference(s.path(prefix+"/observation.json"))
        sidecar_ref = s.reference(s.path(prefix+"/sidecar.json"))
        obs = s.json(obs_ref)
        q = validate_observation(obs, scene_id=scene_id, split_seed=intent["split_seed"])
        if encoded(obs) != encoded(canonical_observation(obs, q)):
            raise ValueError("draw must preserve canonical observation JSON")
        s.read(sidecar_ref)  # Integrity bytes only, not parsed supervision.
        fp = fingerprint(q)
        match = {"fingerprint": fp, "matches_frozen": fp in self.excluded, "matches_produced": fp in produced,
                 "duplicate": fp in self.excluded or fp in produced}
        record = {**intent, "observation": obs_ref, "sidecar": sidecar_ref, "duplicate_check": match,
            "status": "DUPLICATE_PRESERVED" if match["duplicate"] else "DRAW_PRESERVED"}
        receipt_path = s.path(prefix+"/draw.json")
        if receipt_path.exists() and s.json(s.reference(receipt_path)) != record:
            raise ValueError("draw payload/receipt or duplicate history differs")
        return obs, record

    def verify_index(self, split, produced, *, check):
        self.intent(split, 0)
        s = self.store
        ref = s.reference(s.path(f"draws/{split}/index.json"))
        records, fingerprints = [], set(produced)
        for sid in range(512):
            check()
            _, expected = self.reopen(split, sid, fingerprints)
            if expected["status"] != "DRAW_PRESERVED":
                raise RuntimeError("duplicate preserved; no complete observable batch")
            records.append(s.reference(s.path(f"draws/{split}/{sid:05d}/draw.json")))
            fingerprints.add(expected["duplicate_check"]["fingerprint"])
        expected = {"schema": "geometric-decision-draw-index-v1", "binding": s.binding,
            "test_freeze": self.freeze, "split": split, "split_seed": dict(TESTS)[split],
            "scene_ids": list(range(512)), "records": records, "status": "DRAWN_NO_TRUTH_ACCESS"}
        if s.json(ref) != expected:
            raise ValueError("draw index differs from its complete ordered records")
        return ref, expected, fingerprints

    def previous(self, split, *, check):
        self.intent(split, 0)
        produced = set()
        for old, _ in TESTS:
            if old == split:
                return produced
            _, _, produced = self.verify_index(old, produced, check=check)
        raise AssertionError("validated split missing")

    def produce(self, split, *, draw, check):
        """Callback admission is external; each attempted tuple is durable first."""
        if not callable(draw) or not callable(check):
            raise ValueError("admitted producer and resource check are required")
        check()
        s, produced = self.store, self.previous(split, check=check)
        index = f"draws/{split}/index.json"
        if s.path(index).exists():
            return self.verify_index(split, produced, check=check)[0]
        records = []
        for sid in range(512):
            check()
            prefix = f"draws/{split}/{sid:05d}"
            folder, intent_path = s.path(prefix), s.path(prefix+"/intent.json")
            if not intent_path.exists():
                if folder.exists() and any(folder.iterdir()):
                    raise RuntimeError("unbound draw files require reconciliation; no draw")
                if any(s.path(f"draws/{split}/{i:05d}").exists() for i in range(sid+1, 512)):
                    raise RuntimeError("noncontiguous draw prefix; no hole filling")
                s.publish_json(prefix+"/intent.json", self.intent(split, sid))
                observation, sidecar = draw(split, sid, dict(TESTS)[split])
                s.publish_json(prefix+"/observation.json", observation)
                s.publish_json(prefix+"/sidecar.json", sidecar)
            if not all(s.path(prefix+"/"+name).is_file() for name in ("observation.json", "sidecar.json")):
                raise RuntimeError("attempted draw has incomplete pair; never redraw")
            _, record = self.reopen(split, sid, produced)
            ref = s.publish_json(prefix+"/draw.json", record)
            if record["status"] == "DUPLICATE_PRESERVED":
                raise RuntimeError(f"duplicate preserved at {split}/{sid}; no replacement")
            records.append(ref)
            produced.add(record["duplicate_check"]["fingerprint"])
        s.publish_json(index, {"schema": "geometric-decision-draw-index-v1", "binding": s.binding,
            "test_freeze": self.freeze, "split": split, "split_seed": dict(TESTS)[split],
            "scene_ids": list(range(512)), "records": records, "status": "DRAWN_NO_TRUTH_ACCESS"})
        return self.verify_index(split, self.previous(split, check=check), check=check)[0]

    def observations(self, split, *, check):
        ref, index, _ = self.verify_index(split, self.previous(split, check=check), check=check)
        rows, refs = [], []
        for sid, draw_ref in enumerate(index["records"]):
            check()
            receipt = self.store.json(draw_ref)
            obs_ref = receipt["observation"]
            obs = self.store.json(obs_ref)
            validate_observation(obs, scene_id=sid, split_seed=dict(TESTS)[split])
            rows.append(obs)
            refs.append(obs_ref)
        return {"index": ref, "observations": rows, "origin": {"kind": "original", "records": refs}}
