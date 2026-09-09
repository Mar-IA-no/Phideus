"""Observable-only exclusion inventory, not permission to draw fresh tests.

Reopen the exact bytes of prior observations and declared mechanical fixtures.
Aliases retain their provenance but do not increase the unique sample count.
The caller must freeze this inventory together with selection and execution
authority; absence of byte-identical q32 never proves semantic independence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .generative_evidence_reuse import AUTHORIZATION, ROOT, OpenReuse, VerifiedBytes
from .observable_rival_campaign import RELEASE, RELEASE_SHA, SPLITS, selected_ids

RIVALS = {"path": "data/atencion_armonica/observable_source_rivals_v1/campaign.json",
          "sha256": "a60213c6fb6fa3911bdc6c32733ab779b4b3895db5aaba519fe01ddae00c485c"}
SCHEMA = "generative-evidence-observed-exclusions-v1"


def fingerprint(q):
    """Absolute sorted little-endian float32 bytes, with no gauge recentering."""
    original = np.asarray(q)
    if (original.ndim != 1 or not len(original) or original.dtype.kind not in "fiu"
            or not np.isfinite(original).all()):
        raise ValueError("exclusion requires a finite numeric observation vector")
    delivered = original.astype("<f4")
    if not np.array_equal(original, delivered.astype(np.float64)):
        raise ValueError("exclusion observation must be exactly q32")
    return hashlib.sha256(np.sort(delivered, kind="stable").tobytes()).hexdigest()


class Inventory:
    def __init__(self, reader):
        self.reader, self.groups, self.seen = reader, [], set()

    def add(self, name, vectors, *, source, alias=False):
        if not isinstance(name, str) or not name or any(g["name"] == name for g in self.groups):
            raise ValueError("exclusion group name must be unique and nonempty")
        self.reader.read(source)
        hashes = [fingerprint(q) for q in vectors]
        unique = set(hashes)
        if alias and not unique.issubset(self.seen):
            raise ValueError("declared alias includes previously unrecorded observations")
        self.groups.append({"name": name, "source": source.copy(), "fingerprints": hashes,
                            "count": len(hashes), "unique_count": len(unique),
                            "additional_unique": len(unique-self.seen), "declared_alias": alias})
        self.seen.update(unique)

    def result(self):
        return {"schema": SCHEMA, "status": "EXCLUSIONS_ONLY_NOT_TEST_AUTHORIZED",
                "fingerprint": "SHA256 sorted absolute little-endian float32 log_f bytes",
                "groups": self.groups, "fingerprints": sorted(self.seen),
                "unique_count": len(self.seen),
                "consumed_sha256": dict(sorted(self.reader.consumed.items())),
                "test_access": False}


def _member(reader, ref, name, *, role=None):
    manifest = reader.json(ref)
    if (manifest.get("schema") != "structured-source-bundle-v1"
            or manifest.get("status") != "COMPLETE"
            or (role is not None and manifest.get("role") != role)):
        raise ValueError("incomplete or wrong historical exclusion bundle")
    member = {"path": (Path(ref["path"]).parent/name).as_posix(),
              "sha256": manifest["artifacts_sha256"][name]}
    return manifest, member, reader.read(member)


def _observations(reader, ref, *, count=None, ids=None, seed=None):
    rows = [json.loads(line) for line in reader.read(ref).splitlines()]
    if count is not None and len(rows) != count:
        raise ValueError("observation exclusion count differs")
    if ids is not None and [r.get("scene_id") for r in rows] != ids:
        raise ValueError("observation exclusion IDs differ")
    for row in rows:
        if (set(row) != {"scene_id", "split_seed", "log_f"}
                or type(row["scene_id"]) is not int or type(row["split_seed"]) is not int
                or (seed is not None and row["split_seed"] != seed)):
            raise ValueError("observation exclusion schema or seed differs")
        fingerprint(row["log_f"])
    return rows


def build_exclusions(mechanical_ref):
    """Assemble real prior/open/released cohorts plus an audited fixture catalog.

    mechanical_ref names a JSON catalog of q32 vectors and exact source refs.
    Its completeness is a review obligation, not inferred from file names.
    No fresh observation generator, backbone, fitter or truth reader is used.
    """
    reuse = OpenReuse()
    reader = reuse.reader
    inventory = Inventory(reader)
    auth = reader.json(AUTHORIZATION)
    prior = auth["prior_corpus"]
    refs = prior["observation_references"]
    if len(refs) != 13 or len({r["path"] for r in refs}) != 13:
        raise ValueError("closed historical corpus requires thirteen roles")
    for i, ref in enumerate(refs):
        rows = _observations(reader, ref)
        inventory.add(f"historical_role_{i:02d}", [r["log_f"] for r in rows], source=ref)
    fixtures = reader.json(prior["mechanical_roster"])
    mechanical_seen = set()
    for i, entry in enumerate(fixtures["files"]):
        ref = {k: entry[k] for k in ("path", "sha256")}
        value, vectors = reader.json(ref), []
        for keys in entry["observation_keys"]:
            q = value
            for key in keys:
                q = q[key]
            if entry["value_kind"] == "observation_object":
                q = q["log_f"]
            elif entry["value_kind"] != "log_f32_vector":
                raise ValueError("unknown historical fixture extraction")
            vectors.append(q)
        mechanical_seen.update(map(fingerprint, vectors))
        inventory.add(f"historical_fixture_{i:02d}", vectors, source=ref)
    if sorted(mechanical_seen) != fixtures["unique_fingerprints"]:
        raise ValueError("historical fixture fingerprint roster differs")
    alias = fixtures["development_alias"]
    if alias["additional_observations"] != 0:
        raise ValueError("historical alias claims additional observations")
    canonical = {"path": alias["canonical_path"], "sha256": alias["sha256"]}
    reader.read(canonical)
    ref = {"path": alias["path"], "sha256": alias["sha256"]}
    inventory.add("historical_development_alias", [r["log_f"] for r in _observations(reader, ref)],
                  source=ref, alias=True)
    if sorted(inventory.seen) != prior["fingerprints"]:
        raise ValueError("reconstructed historical exclusions differ from pinned authorization")
    for kind, count in (("geometry", 2), ("cpu", 0), ("gpu", 1)):
        role = "learned_geometry_profile" if kind == "geometry" else f"learned_training_{kind}_profile"
        _, ref, raw = _member(reader, auth["profiles"][kind], "report.json", role=role)
        rows = json.loads(raw)["observations"]
        if len(rows) != count:
            raise ValueError("historical profile observation roster differs")
        inventory.add(f"learned_profile_{kind}", [r["log_f"] for r in rows], source=ref)
    for split, entries in reuse.shards.items():
        for shard, entry in enumerate(entries):
            _, ref, _ = _member(reader, entry["data"], "observations.jsonl", role="learned_observation_shard")
            rows = _observations(reader, ref, count=512, ids=list(range(shard*512, (shard+1)*512)),
                                 seed=2026090880 if split == "train" else 2026090881)
            inventory.add(f"open_{split}_{shard:02d}", [r["log_f"] for r in rows], source=ref)
    release_ref = {"path": RELEASE.as_posix(), "sha256": RELEASE_SHA}
    release = reader.json(release_ref)
    if release.get("schema") != "partition-evaluation-release-mixed-roster-v1" or set(release["splits"]) != set(SPLITS):
        raise ValueError("old released test roster differs")
    released = {}
    for split in SPLITS:
        manifest, ref, _ = _member(reader, release["splits"][split]["data"], "observations.jsonl",
                                   role="learned_observation_shard")
        if manifest["binding"]["split"] != split:
            raise ValueError("old test observation split differs")
        rows = _observations(reader, ref, count=512, ids=list(range(512)), seed=manifest["binding"]["split_seed"])
        inventory.add(f"released_{split}", [r["log_f"] for r in rows], source=ref)
        released[split] = rows
    rivals = reader.json(RIVALS)["scenes"]
    expected = [(split, sid) for split, ids in selected_ids().items() for sid in ids]
    if [(r["split"], r["scene_id"]) for r in rivals] != expected:
        raise ValueError("rival alias roster differs")
    for row in rivals:
        if fingerprint(row["q32"]) != fingerprint(released[row["split"]][row["scene_id"]]["log_f"]):
            raise ValueError("rival alias differs from its released observation")
    inventory.add("rivals_96_alias", [r["q32"] for r in rivals], source=RIVALS, alias=True)
    catalog = reader.json(mechanical_ref)
    if (set(catalog) != {"schema", "records", "scope"}
            or catalog["schema"] != "generative-evidence-mechanical-exclusions-v1"
            or not isinstance(catalog["records"], list) or not catalog["records"]):
        raise ValueError("a nonempty reviewed mechanical observation catalog is required")
    for entry in catalog["records"]:
        if set(entry) != {"name", "source", "q32", "alias"} or type(entry["alias"]) is not bool:
            raise ValueError("mechanical catalog entry schema differs")
        inventory.add("mechanical_"+entry["name"], entry["q32"], source=entry["source"], alias=entry["alias"])
    return inventory.result()


def duplicate_matches(q, frozen, produced=()):
    """Pure check after persisting the draw; never redraw or silently replace it."""
    if (frozen.get("schema") != SCHEMA or frozen.get("status") != "EXCLUSIONS_ONLY_NOT_TEST_AUTHORIZED"
            or frozen.get("test_access") is not False):
        raise ValueError("wrong exclusion inventory schema or status")
    hashes = frozen["fingerprints"]
    if (hashes != sorted(set(hashes)) or len(hashes) != frozen["unique_count"]
            or any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
                   for h in [*hashes, *produced])):
        raise ValueError("invalid exclusion fingerprint roster")
    fp = fingerprint(q)
    return {"fingerprint": fp, "matches_frozen": fp in hashes, "matches_produced": fp in produced,
            "duplicate": fp in hashes or fp in produced}
