"""Extend, never overwrite, the archive inventory with observed OPEN probes.

The caller authenticates the COMPLETE profile operator/report before entry.
No sampler, forward, fitting, targets or prospective test capability here.
"""
from __future__ import annotations

from copy import deepcopy

from .generative_evidence_exclusions import Inventory, fingerprint
from .geometric_decision_exclusions import checked_prior
from .geometric_decision_scene_store import load_source


def extend_profile_exclusions(reader, prior_ref, profile, report_ref, *, check):
    prior = reader.json(prior_ref)
    seen = checked_prior(prior)
    report = profile.json(report_ref)
    if (report["schema"] != "geometric-decision-observed-profile-v1"
            or report["binding"] != profile.binding or report["new_test_observations"] != 0
            or report["recovery"] != report["result"] or report["exclusion_extension_required"] is not True):
        raise ValueError("profile report is not a complete OPEN observable recovery")
    result = profile.json(report["result"])
    if (result["schema"] != "geometric-decision-observed-run-v1" or result["binding"] != profile.binding
            or result["split"] != "train" or result["split_seed"] != 2026090880
            or result["scene_ids"] != list(range(16)) or result["truth_access"] is not False
            or result["global_seal"] is not False or result["roundtrip"] is None):
        raise ValueError("profile exclusions require the fixed first16 TRAIN batch")
    inventory = Inventory(reader)
    inventory.groups, inventory.seen = deepcopy(prior["groups"]), set(seen)
    def root_ref(ref):
        profile.read(ref)
        return {"path": profile.path(ref["path"]).relative_to(reader.root).as_posix(), "sha256": ref["sha256"]}
    def batch(ref, ids):
        value = profile.json(ref)
        if (set(value) != {"schema", "binding", "features", "logits", "split", "split_seed", "scene_ids", "sources"}
                or value["schema"] != "geometric-decision-source-batch-v1" or value["binding"] != profile.binding
                or value["split"] != "train" or value["split_seed"] != 2026090880
                or value["scene_ids"] != ids or len(value["sources"]) != len(ids)):
            raise ValueError("profile source roster differs")
        return value["sources"]
    originals = batch(result["sources"], list(range(16)))
    eligible = []
    for sid, ref in enumerate(originals):
        check()
        scene = load_source(profile, ref)
        if scene["observation"]["scene_id"] != sid or fingerprint(scene["q32"]) not in seen:
            raise ValueError("profile original is not an already-excluded TRAIN observation")
        if scene["partitions"]:
            eligible.append(sid)
        inventory.add(f"observed_profile_train_{sid:05d}", [scene["q32"]], source=root_ref(ref), alias=True)
    ids = eligible[:4]
    if len(ids) != 4 or result["roundtrip_scene_ids"] != ids or report["roundtrip_scene_ids"] != ids:
        raise ValueError("profile probe roster is not first-four eligible")
    probes = batch(result["roundtrip"]["sources"], ids)
    for sid, ref in zip(ids, probes):
        check()
        scene = load_source(profile, ref)
        record = profile.json(ref)
        if (scene["observation"]["scene_id"] != sid or record["derivation"]["kind"] != "roundtrip"
                or record["derivation"]["parent"] != originals[sid]):
            raise ValueError("profile roundtrip lineage differs")
        coordinates = profile.arrays(record["coordinates"])
        # shifted64 is not a delivered q32 observation; the four vectors below are.
        for key in ("original", "shifted32", "q_center", "q_probe"):
            q = coordinates[key]
            inventory.add(f"observed_profile_probe_{sid:05d}_{key}", [q],
                source=root_ref(record["coordinates"]), alias=fingerprint(q) in inventory.seen)
    complete = inventory.result()
    inherited, consumed = prior["consumed_sha256"], complete["consumed_sha256"]
    if any(p in inherited and inherited[p] != h for p, h in consumed.items()):
        raise ValueError("profile exclusion source conflicts with inherited bytes")
    complete["consumed_sha256"] = dict(sorted({**inherited, **consumed}.items()))
    complete["extension"] = {"parent": prior_ref, "inherited_extension": prior.get("extension"),
        "profile_root": str(profile.root), "profile_report": report_ref,
        "original_count": 16, "probe_count": 4, "delivered_coordinate_vectors": 16,
        "authority": "exact observable fingerprints only; caller admits COMPLETE profile"}
    checked_prior(complete)
    return complete
