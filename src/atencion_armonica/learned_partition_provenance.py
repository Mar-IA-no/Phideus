"""Pinned source and previously observed corpus identities; no Torch or draws."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from . import structured_source_gate as historical
from .partial_compatibility_cache import observation_fingerprint, sha_file
from .structured_source_data import validate_observation as validate_historical_observation

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {"path": "experiments/atencion_armonica/PROTOCOL_LEARNED_PARTITION_READER.md",
            "sha256": "939afc335fb43f9223bb0c58a331c56835199589efd012f4a301ee96c8ed35e9"}
PLAN = {"path": "experiments/atencion_armonica/PLAN_LEARNED_PARTITION_READER.md",
        "sha256": "508b19b43390d8ddc913393a16767eaf79977cff882f7e4ffa306b65ccce2b44"}
FIXTURES = {"path": "experiments/atencion_armonica/learned_partition_prior_fixtures.json",
            "sha256": "4a4f6c0270b674527a6b8f0814f90e2afc1b9272f714df2e3c8c16c0b5d9fe74"}
STRUCTURED_LAST = {"path": "data/atencion_armonica/structured_source_reader_v1/deformed_family_data/manifest.json",
                   "sha256": "dff89ca24ae3bc69d23b5e587460da6f99022a0730427a3ffaa1e3d0a90da039"}

# Deliberately fail closed while the complete campaign integration is absent.
MODULES = ("core", "model", "readout", "training", "state", "snapshots", "cache", "inference", "metrics",
           "provenance", "data", "gate", "resources", "profile", "runner", "supervisor")
SOURCES = (*[f"src/atencion_armonica/learned_partition_{name}.py" for name in MODULES],
           "experiments/atencion_armonica/run_learned_partition.py")

reference = historical.reference
verify_reference = historical.verify_reference
read_reference = historical.read_reference


def current_sources():
    sources = historical.current_sources()
    for ref in (PLAN, PROTOCOL, FIXTURES):
        verify_reference(ref)
        sources[ref["path"]] = ref["sha256"]
    for name in SOURCES:
        sources[name] = sha_file(ROOT/name)
    return dict(sorted(sources.items()))


def runtime_versions():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "torch")}}


def prior_observation_references():
    """Eight original roles plus five structured-reader roles, never sidecars."""
    result = list(historical.historical_observations())
    last = read_reference(STRUCTURED_LAST)
    previous = last["binding"]["previous"]
    roles = ("calibration", "iid", "ood_beta", "ood_polyphony", "deformed_family")
    if set(previous) != set(roles[:-1]):
        raise ValueError("structured-reader historical roster differs")
    refs = {**previous, "deformed_family": STRUCTURED_LAST}
    for role in roles:
        ref = refs[role]
        manifest = read_reference(ref)
        binding = manifest["binding"]
        if (manifest["status"] != "COMPLETE" or manifest["role"] != f"{role}_data"
                or binding["split"] != role or binding["count"] != 256
                or binding["previous"] != {r: refs[r] for r in roles[:roles.index(role)]}):
            raise ValueError("historical structured split or prefix differs")
        path = (Path(ref["path"]).parent/"observations.jsonl").as_posix()
        observation_ref = {"path": path, "sha256": manifest["artifacts_sha256"]["observations.jsonl"]}
        rows = [json.loads(line) for line in verify_reference(observation_ref).read_bytes().splitlines()]
        if len(rows) != 256:
            raise ValueError("historical structured observation roster is incomplete")
        for i, obs in enumerate(rows):
            validate_historical_observation(obs, i, role)
        result.append(observation_ref)
    if len(result) != 13 or len({r["path"] for r in result}) != 13:
        raise ValueError("expected thirteen distinct historical roles")
    return result


def mechanical_fingerprints():
    """Verify all seven typed sources and the alias before using their roster."""
    roster = read_reference(FIXTURES)
    seen = set()
    for record in roster["files"]:
        document = read_reference({k: record[k] for k in ("path", "sha256")})
        for keys in record["observation_keys"]:
            value = document
            for key in keys:
                value = value[key]
            if record["value_kind"] == "observation_object":
                if set(value) != {"scene_id", "split_seed", "log_f"}:
                    raise ValueError("historical mechanical observation schema differs")
                q = np.asarray(value["log_f"])
            elif record["value_kind"] == "log_f32_vector":
                q = np.asarray(value)
            else:
                raise ValueError("unknown mechanical extraction kind")
            if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or not np.array_equal(q, q.astype(np.float32).astype(np.float64)):
                raise ValueError("mechanical fixture is not exact finite q32")
            seen.add(observation_fingerprint({"log_f": q}))
    alias = roster["development_alias"]
    for key in ("path", "canonical_path"):
        verify_reference({"path": alias[key], "sha256": alias["sha256"]})
    if sorted(seen) != roster["unique_fingerprints"] or alias["additional_observations"] != 0:
        raise ValueError("mechanical fingerprint roster or alias differs")
    return seen


def prior_corpus():
    """Closed prior corpus only. New profile and new role rosters are added by the gate."""
    references = prior_observation_references()
    seen = set()
    for ref in references:
        for line in verify_reference(ref).read_bytes().splitlines():
            fp = observation_fingerprint(json.loads(line))
            if fp in seen:
                raise ValueError("closed prior corpus contains a duplicated observation")
            seen.add(fp)
    seen.update(mechanical_fingerprints())
    return {"observation_references": references, "mechanical_roster": FIXTURES,
            "fingerprints": sorted(seen)}
