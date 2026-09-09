"""Fail-closed freeze for the four fresh tests; no producer or truth port.

The caller owns the four-hour resource supervisor and supplies its check.
This module authenticates selection, selected CPU states, TRAIN normalizers,
future execution sources and an observable-only exclusion inventory before
publishing one immutable manifest.  It never draws, forwards or reads sidecars.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .generative_evidence_reuse import ROOT, VerifiedBytes
from .generative_evidence_storage import write_json
from .partial_compatibility_cache import encoded


FREEZE = ROOT/"data/atencion_armonica/generative_evidence_reader_v1/test_freeze.json"
PROTOCOL = ROOT/"experiments/atencion_armonica/PROTOCOL_GENERATIVE_EVIDENCE_READER.md"
TESTS = (("iid", 512, 2026090982), ("ood_beta", 512, 2026090983),
         ("ood_polyphony", 512, 2026090984), ("deformed_family", 512, 2026090985))
INTERVENTIONS = ("zero", "decoupled")
REQUIRED_SOURCES = (
    "src/atencion_armonica/generative_evidence_test_freeze.py",
    "src/atencion_armonica/generative_evidence_fresh_data.py",
    "src/atencion_armonica/generative_evidence_fresh_store.py",
    "src/atencion_armonica/generative_evidence_fresh_inference.py",
    "src/atencion_armonica/generative_evidence_inference.py",
    "src/atencion_armonica/generative_evidence_fresh_evaluation.py",
    "src/atencion_armonica/generative_evidence_exclusions.py",
    "src/atencion_armonica/generative_evidence_references.py",
    "src/atencion_armonica/learned_partition_data.py",
    "experiments/atencion_armonica/build_generative_exclusions.py",
    "experiments/atencion_armonica/run_generative_tests.py",
)
STATUS = "TEST_INTERFACE_FROZEN_NOT_PRODUCED"


def _selection_runner():
    from experiments.atencion_armonica import run_generative_selection
    return run_generative_selection


def _training_runner():
    from experiments.atencion_armonica import run_generative_training
    return run_generative_training


def _exclusion_builder():
    from experiments.atencion_armonica import build_generative_exclusions
    return build_generative_exclusions


def _exclusion_kernel():
    from . import generative_evidence_exclusions
    return generative_evidence_exclusions


def _valid_ref(ref):
    return (isinstance(ref, dict) and set(ref) == {"path", "sha256"}
            and isinstance(ref["path"], str) and bool(ref["path"])
            and isinstance(ref["sha256"], str) and len(ref["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in ref["sha256"]))


def _reference(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def _local_to_root(store, ref):
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256", "bytes"}
            or type(ref["bytes"]) is not int or ref["bytes"] < 0):
        raise ValueError("invalid local artifact reference")
    path = store.path(ref["path"])
    local = store.reference(path)
    root = _reference(path)
    if local != ref or local["sha256"] != root["sha256"]:
        raise ValueError("local/root artifact reference differs")
    return root


def _source_hashes(check):
    values, missing = {}, []
    for name in REQUIRED_SOURCES:
        check()
        path = ROOT/name
        if not path.is_file() or path.is_symlink():
            missing.append(name)
            continue
        values[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if missing:
        raise FileNotFoundError("test freeze requires all execution sources: "+", ".join(missing))
    return values


def _verify_exclusions(ref, check):
    if not _valid_ref(ref):
        raise ValueError("test freeze requires an immutable exclusion reference")
    check()
    reader = VerifiedBytes(ROOT)
    raw = reader.read(ref)
    inventory = json.loads(raw)
    if raw != encoded(inventory):
        raise ValueError("exclusion inventory must use canonical JSON bytes")
    keys = {"schema", "status", "fingerprint", "groups", "fingerprints", "unique_count",
            "consumed_sha256", "test_access"}
    if (set(inventory) != keys
            or inventory["schema"] != "generative-evidence-observed-exclusions-v1"
            or inventory["status"] != "EXCLUSIONS_ONLY_NOT_TEST_AUTHORIZED"
            or inventory["fingerprint"] != "SHA256 sorted absolute little-endian float32 log_f bytes"
            or inventory["test_access"] is not False):
        raise ValueError("wrong exclusion inventory schema or authority")
    fingerprints = inventory["fingerprints"]
    if (not isinstance(fingerprints, list) or fingerprints != sorted(set(fingerprints))
            or len(fingerprints) != inventory["unique_count"]
            or any(not isinstance(v, str) or len(v) != 64
                   or any(c not in "0123456789abcdef" for c in v) for v in fingerprints)):
        raise ValueError("invalid frozen exclusion fingerprint roster")
    groups = inventory["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("exclusion inventory requires nonempty groups")
    seen, names = set(), set()
    for group in groups:
        check()
        group_keys = {"name", "source", "fingerprints", "count", "unique_count",
                      "additional_unique", "declared_alias"}
        hashes = group.get("fingerprints") if isinstance(group, dict) else None
        if (not isinstance(group, dict) or set(group) != group_keys or not isinstance(group["name"], str)
                or not group["name"] or group["name"] in names or not _valid_ref(group["source"])
                or not isinstance(hashes, list) or group["count"] != len(hashes)
                or group["unique_count"] != len(set(hashes))
                or group["additional_unique"] != len(set(hashes)-seen)
                or type(group["declared_alias"]) is not bool
                or (group["declared_alias"] and not set(hashes).issubset(seen))
                or any(h not in fingerprints for h in hashes)):
            raise ValueError("exclusion group roster or alias accounting differs")
        names.add(group["name"])
        seen.update(hashes)
    if seen != set(fingerprints):
        raise ValueError("exclusion groups do not reconstruct the unique inventory")
    consumed = inventory["consumed_sha256"]
    if (not isinstance(consumed, dict) or not consumed
            or any(not isinstance(path, str) or not path or not isinstance(sha, str) or len(sha) != 64
                   or any(c not in "0123456789abcdef" for c in sha)
                   or path.startswith("data/atencion_armonica/generative_evidence_reader_v1/fresh/")
                   for path, sha in consumed.items())
            or any(group["source"]["sha256"] != consumed.get(group["source"]["path"])
                   for group in inventory["groups"])):
        raise ValueError("exclusion consumed-source ledger differs or crosses the fresh port")
    for path, sha in sorted(consumed.items()):
        check()
        reader.read({"path": path, "sha256": sha})
    # `ref` is also consumed above, but the producer's ledger covers inputs only.
    if {path: sha for path, sha in reader.consumed.items() if path != ref["path"]} != consumed:
        raise ValueError("exclusion source payloads do not match their consumed ledger")
    catalogs = [path for path in consumed if Path(path).name == "mechanical_catalog.json"]
    if len(catalogs) != 1:
        raise ValueError("exclusion inventory requires one exact mechanical catalog")
    catalog_ref = {"path": catalogs[0], "sha256": consumed[catalogs[0]]}
    catalog_raw = reader.read(catalog_ref)
    catalog = json.loads(catalog_raw)
    if catalog_raw != encoded(catalog):
        raise ValueError("mechanical exclusion catalog must use canonical JSON bytes")
    check()
    if catalog != _exclusion_builder().mechanical_catalog():
        raise ValueError("mechanical exclusion catalog is not the complete current builder output")
    check()
    rebuilt = _exclusion_kernel().build_exclusions(catalog_ref)
    if inventory != rebuilt:
        raise ValueError("exclusion inventory differs from its complete exact reconstruction")
    check()
    return inventory


def _selection_context(check):
    """Reopen selected state payloads and the common TRAIN normalizer; no forward."""
    selection_runner, training = _selection_runner(), _training_runner()
    check()
    selection_ref = selection_runner.verified_selection()
    selection = VerifiedBytes(ROOT).json(selection_ref)
    keys = {"schema", "binding", "status", "training_accumulated_seconds", "cell_count",
            "calibration_record_count", "calibration_records", "initial_models", "selection",
            "selected_states", "test_access"}
    if (set(selection) != keys or selection["schema"] != "generative-evidence-calibration-selection-v1"
            or selection["status"] != "CALIBRATION_SELECTED_NOT_TEST_AUTHORIZED"
            or selection["cell_count"] != 27 or selection["calibration_record_count"] != 270
            or not isinstance(selection["calibration_records"], list)
            or len(selection["calibration_records"]) != 270
            or selection["test_access"] is not False):
        raise ValueError("typed selection content is not the exact pre-test closure")
    binding = selection["binding"]
    if (not isinstance(binding, dict)
            or set(binding) != {"selection_manifest", "training_manifest", "training_complete"}
            or any(not _valid_ref(binding[name]) for name in binding)):
        raise ValueError("selection lacks its exact stage/training binding")
    selection_manifest = selection_runner.read_manifest(binding["selection_manifest"])
    if (selection_manifest["training"]["manifest"] != binding["training_manifest"]
            or selection_manifest["training"]["index"] != binding["training_complete"]
            or selection_manifest["output"]["path"] != selection_ref["path"]
            or selection_manifest["output"]["status"] != selection["status"]):
        raise ValueError("selection output and authenticated stage manifest differ")
    training_manifest, delivery = training.read_training_manifest(binding["training_manifest"])
    expected = training.cell_roster()
    training_index = VerifiedBytes(ROOT).json(binding["training_complete"])
    index_keys = {"schema", "status", "manifest", "cells", "cell_count",
                  "accumulated_seconds", "test_access"}
    cells = training_index.get("cells") if isinstance(training_index, dict) else None
    if (set(training_index) != index_keys
            or training_index["schema"] != "generative-evidence-training-complete-v1"
            or training_index["status"] != "TRAINED_NOT_SELECTED"
            or training_index["manifest"] != binding["training_manifest"]
            or training_index["cell_count"] != 27 or training_index["test_access"] is not False
            or not isinstance(cells, list) or len(cells) != 27
            or [row.get("cell") for row in cells] != expected
            or any(not isinstance(row, dict) or set(row) != {"cell", "complete"}
                   or not _valid_ref(row["complete"]) for row in cells)):
        raise ValueError("training completion roster changed before test freeze")
    completes = {row["cell"]["cell_id"]: row["complete"] for row in cells}
    arms = list(dict.fromkeys(cell["arm"] for cell in expected))
    chosen = (selection["selection"].get("selected")
              if isinstance(selection["selection"], dict) else None)
    if (not isinstance(chosen, dict) or set(chosen) != set(arms)
            or set(selection["selected_states"]) != set(arms)):
        raise ValueError("selection arm roster differs from training")

    from .generative_evidence_cell import CellArtifacts
    normalizer_local = delivery["corpus"].normalizer_ref
    check()
    delivery["store"].json(normalizer_local)
    normalizer = _local_to_root(delivery["store"], normalizer_local)
    states, seen = [], set()
    for arm in arms:
        arm_cells = [cell for cell in expected if cell["arm"] == arm]
        selected = selection["selected_states"][arm]
        epoch = chosen[arm].get("epoch") if isinstance(chosen[arm], dict) else None
        if (not isinstance(selected, dict) or set(selected) != {"epoch", "cells"}
                or selected["epoch"] != epoch or epoch not in range(5, 51, 5)
                or [row.get("cell") for row in selected["cells"]] != arm_cells):
            raise ValueError("selected epoch or nine-cell arm roster differs")
        for row in selected["cells"]:
            check()
            row_keys = {"cell", "cell_complete", "epoch", "state", "calibration",
                        "predictions", "calibration_readout"}
            cell = row["cell"]
            if (set(row) != row_keys or row["epoch"] != epoch or cell["cell_id"] in seen
                    or row["cell_complete"] != completes[cell["cell_id"]]
                    or any(not _valid_ref(row[name]) for name in
                           ("cell_complete", "state", "calibration", "predictions"))):
                raise ValueError("selected cell reference roster differs")
            seen.add(cell["cell_id"])
            complete_path = ROOT/row["cell_complete"]["path"]
            if not complete_path.is_file() or not (complete_path.parent/"binding.json").is_file():
                raise ValueError("selected cell boundary is missing")
            complete = VerifiedBytes(ROOT).json(row["cell_complete"])
            data_binding = complete.get("binding", {}).get("data", {})
            if (complete.get("status") != "TRAINED_NOT_SELECTED"
                    or complete.get("binding", {}).get("training_manifest") != binding["training_manifest"]
                    or data_binding.get("normalizers") != normalizer_local
                    or any(complete.get(name) != cell[name] for name in
                           ("arm", "checkpoint_seed", "reader_seed"))):
                raise ValueError("selected cell binding or TRAIN normalizer differs")
            store = CellArtifacts(complete_path.parent, binding=complete["binding"])
            local_complete = store.reference(complete_path)
            if _local_to_root(store, local_complete) != row["cell_complete"]:
                raise ValueError("selected complete root reference differs")
            local_state = complete["snapshots"][epoch]
            if _local_to_root(store, local_state) != row["state"]:
                raise ValueError("selected state is not its declared epoch snapshot")
            state = store.load_state(local_state)
            if (state["binding"] != complete["binding"] or state["epoch"] != epoch
                    or state["next_batch"] != 0 or any(state[name] != cell[name] for name in
                       ("arm", "checkpoint_seed", "reader_seed"))):
                raise ValueError("selected state payload identity differs")
            states.append({"cell": cell, "epoch": epoch, "complete": row["cell_complete"],
                           "state": row["state"]})
    if seen != {cell["cell_id"] for cell in expected} or len(states) != 27:
        raise ValueError("test freeze requires all 27 selected states")
    return {"selection": selection_ref, "selection_manifest": binding["selection_manifest"],
            "training_manifest": binding["training_manifest"],
            "training_complete": binding["training_complete"], "normalizers": normalizer,
            "selected_states": states, "runtime": selection_manifest["runtime"], "arms": arms}


def _prediction_roster(states, arms):
    originals = [{"kind": "original", **row["cell"]} for row in states]
    generative = [row["cell"] for row in states if row["cell"]["arm"] == "generative"]
    if len(originals) != 27 or len(generative) != 9 or set(arms) != {"local", "generative", "decoupled"}:
        raise ValueError("prediction roster requires the canonical 27 learned cells")
    interventions = [{"kind": "intervention", **cell, "intervention": intervention}
                     for cell in generative for intervention in INTERVENTIONS]
    result = originals+interventions
    if len(result) != 45:
        raise ValueError("test prediction roster must contain 45 outputs")
    return result


def _expected(exclusions_ref, check):
    if not callable(check):
        raise TypeError("test freeze requires a resource check callback")
    # Missing future execution code fails before selection is treated as eligible.
    sources = _source_hashes(check)
    context = _selection_context(check)
    exclusions = _verify_exclusions(exclusions_ref, check)
    check()
    protocol = _reference(PROTOCOL)
    roster = _prediction_roster(context["selected_states"], context["arms"])
    value = {"schema": "generative-evidence-test-freeze-v1", "status": STATUS,
        "protocol": protocol, "sources": sources, "runtime": context["runtime"],
        "inherited_sources": {"selection_manifest": context["selection_manifest"],
                              "training_manifest": context["training_manifest"]},
        "selection": context["selection"], "training_complete": context["training_complete"],
        "normalizers": context["normalizers"], "exclusions": exclusions_ref,
        "tests": [{"split": split, "scene_count": count, "scene_ids": list(range(count)),
                   "split_seed": seed} for split, count, seed in TESTS],
        "selected_states": context["selected_states"], "prediction_roster": roster,
        "prediction_count_per_test": 45, "original_prediction_count_per_test": 27,
        "intervention_prediction_count_per_test": 18,
        "draw_count": 0, "test_truth_access": False, "sidecar_access": False}
    return value, exclusions


def verify_freeze(ref, *, check):
    if (not _valid_ref(ref) or ref["path"] != FREEZE.relative_to(ROOT).as_posix()):
        raise ValueError("unexpected test freeze reference")
    frozen = VerifiedBytes(ROOT).json(ref)
    if not isinstance(frozen, dict) or not _valid_ref(frozen.get("exclusions")):
        raise ValueError("test freeze lacks its exclusion reference")
    expected, exclusions = _expected(frozen["exclusions"], check)
    check()
    raw = FREEZE.read_bytes()
    value = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != ref["sha256"] or raw != encoded(value) or value != expected:
        raise ValueError("test freeze content, hash or current dependencies differ")
    return {"freeze": ref, "manifest": value, "exclusions": exclusions}


def freeze_tests(*, exclusions_ref, check):
    if not callable(check):
        raise TypeError("test freeze requires a resource check callback")
    value, _ = _expected(exclusions_ref, check)
    if not FREEZE.parent.is_dir() or FREEZE.parent.is_symlink():
        raise ValueError("test freeze requires the existing canonical campaign root")
    if FREEZE.exists():
        result = verify_freeze(_reference(FREEZE), check=check)
        if result["manifest"] != value:
            raise ValueError("cannot replace an existing test freeze")
        return result
    check()
    write_json(FREEZE, value)
    return verify_freeze(_reference(FREEZE), check=check)


__all__ = ["freeze_tests", "verify_freeze", "FREEZE", "STATUS", "TESTS"]
