"""Global observable seal and COMPLETE-only entrance to privileged evaluation.

No sampler, label parser, model or fitter. Recovery callback belongs to the
audited enclosing operator; it must run the frozen recovery-only pipeline.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .geometric_decision_draws import DrawBatch, TESTS
from .geometric_decision_open import ReadOnlyStore

TEST_ROSTER = [{"split": split, "split_seed": seed, "count": 512} for split, seed in TESTS]


def frozen_scope(control, freeze_ref):
    freeze = control.json(freeze_ref)
    if (freeze.get("schema") != "geometric-decision-prospective-freeze-v1"
            or freeze.get("protocol") != control.binding["protocol"]
            or freeze.get("test_roster") != TEST_ROSTER
            or len(freeze.get("head_roster", [])) != 144
            or len({(r["path"], r["sha256"]) for r in freeze["head_roster"]}) != 144):
        raise ValueError("freeze does not bind the full prospective scope")
    return freeze


def file_inventory(store, *, check):
    """Exact owned file tree; sidecars are opaque bytes and never parsed here."""
    result = []
    for path in sorted(store.root.rglob("*")):
        check()
        if path.is_symlink():
            raise ValueError("observable seal cannot include a symlink")
        if path.is_dir():
            continue
        if not path.is_file() or path.suffix not in (".json", ".npz", ".gz"):
            raise ValueError("unrecognized/orphan artifact cannot enter the seal")
        relative = path.relative_to(store.root).as_posix()
        digest, size = hashlib.sha256(), 0
        # ReadOnlyStore deliberately refuses parsing .gz; a seal hashes it as
        # opaque bytes instead, after checking its entire in-root ancestry.
        if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(store.root)):
            raise ValueError("observable seal cannot traverse a symlink")
        with path.open("rb") as stream:
            while chunk := stream.read(1024*1024):
                check()
                digest.update(chunk)
                size += len(chunk)
        result.append({"path": relative, "bytes": size, "sha256": digest.hexdigest()})
    if not result:
        raise ValueError("cannot seal an empty tree")
    return result


def _observed_scope(store, ref, split, seed, heads):
    value = store.json(ref)
    if (value["schema"] != "geometric-decision-observed-run-v1" or value["binding"] != store.binding
            or value["split"] != split or value["split_seed"] != seed
            or value["scene_ids"] != list(range(512)) or value["truth_access"] is not False
            or value["global_seal"] is not False or len(value["records"]) != 144
            or [r["head"] for r in value["records"]] != heads):
        raise ValueError("observable completion does not cover the frozen roster")
    sources = store.json(value["sources"])
    if (sources["scene_ids"] != list(range(512)) or len(sources["sources"]) != 512
            or sources["split"] != split or sources["split_seed"] != seed or sources["binding"] != store.binding):
        raise ValueError("observable source roster differs")
    eligible = [i for i, source in enumerate(sources["sources"])
                if store.json(source)["scene"]["partitions"]][:4]
    if value["roundtrip_scene_ids"] != eligible:
        raise ValueError("roundtrip roster is not first-four eligible")
    if eligible:
        derived = value["roundtrip"]
        if derived is None or len(derived["records"]) != 144 or [r["head"] for r in derived["records"]] != heads:
            raise ValueError("roundtrip is missing frozen heads")
    elif value["roundtrip"] is not None:
        raise ValueError("no eligible scene permits a roundtrip")
    return value


def seal_outputs(control, store, freeze_ref, batches, *, recover, check):
    """Enclosing operator holds its lock and admits resources before entry."""
    freeze = frozen_scope(control, freeze_ref)
    if (str(store.root) != freeze["fresh_root"] or store.binding != {"test_freeze": freeze_ref}
            or len(batches) != 4 or [r["split"] for r in batches] != [s for s, _ in TESTS]
            or any(set(r) != {"split", "draws", "observed"} for r in batches)):
        raise ValueError("seal requires the four ordered frozen batches")
    draws = DrawBatch(store, freeze_ref, exclusions=control.json(freeze["exclusions"]))
    produced = set()
    for row, (split, seed) in zip(batches, TESTS):
        check()
        draw_ref, _, produced = draws.verify_index(split, produced, check=check)
        if draw_ref != row["draws"]:
            raise ValueError("seal draw index differs")
        _observed_scope(store, row["observed"], split, seed, freeze["head_roster"])
        if recover(split) != row["observed"]:
            raise ValueError("full observable recovery changed its completion")
    files = file_inventory(store, check=check)
    # Rechecking metadata after recovery rejects callbacks substituting completions.
    for row, (split, seed) in zip(batches, TESTS):
        _observed_scope(store, row["observed"], split, seed, freeze["head_roster"])
    if control.json(freeze_ref) != freeze:
        raise ValueError("freeze changed during seal")
    return control.publish_json("seals/fresh-observables.json", {
        "schema": "geometric-decision-global-seal-v1", "status": "ALL_OBSERVABLES_SEALED",
        "test_freeze": freeze_ref, "fresh_root": str(store.root),
        "fresh_binding": store.reference(store.path("binding.json")),
        "batches": batches, "files": files, "original_scenes": 2048,
        "all_observable_recoveries_exact": True, "truth_parsed": False})


def admitted_seal(control, finish_ref, freeze_ref, *, check):
    """No caller-supplied seal alone can unlock labels: require operator COMPLETE."""
    check()
    freeze = frozen_scope(control, freeze_ref)
    finish = control.json(finish_ref)
    if finish["status"] != "COMPLETE" or finish["completion"] is None:
        raise PermissionError("privileged evaluation requires COMPLETE observable operator")
    start = control.json(finish["start"])
    manifest = control.json(start["manifest"])
    output = control.json(finish["completion"])
    if (start["stage"] != "fresh" or start["binding"] != control.binding
            or manifest != {"operation": "prospective-observables", "test_freeze": freeze_ref,
                            "root": freeze["fresh_root"]}
            or output["manifest"] != start["manifest"] or output["test_freeze"] != freeze_ref):
        raise PermissionError("observable finish belongs to another frozen operation")
    seal = control.json(output["seal"])
    if (set(seal) != {"schema", "status", "test_freeze", "fresh_root", "fresh_binding", "batches", "files",
                     "original_scenes", "all_observable_recoveries_exact", "truth_parsed"}
            or seal["schema"] != "geometric-decision-global-seal-v1"
            or seal["status"] != "ALL_OBSERVABLES_SEALED" or seal["test_freeze"] != freeze_ref
            or seal["fresh_root"] != freeze["fresh_root"] or seal["original_scenes"] != 2048
            or seal["all_observable_recoveries_exact"] is not True or seal["truth_parsed"] is not False
            or len(seal["batches"]) != 4 or [r["split"] for r in seal["batches"]] != [s for s, _ in TESTS]):
        raise PermissionError("seal does not cover every test and probe")
    store = ReadOnlyStore(Path(freeze["fresh_root"]), binding_ref=seal["fresh_binding"])
    if store.binding != {"test_freeze": freeze_ref} or file_inventory(store, check=check) != seal["files"]:
        raise PermissionError("observable tree changed after the global seal")
    for row, (split, seed) in zip(seal["batches"], TESTS):
        _observed_scope(store, row["observed"], split, seed, freeze["head_roster"])
    return store, seal, output["seal"]
