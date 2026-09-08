"""Prospective closed-test gate. Pure NumPy/JSON; never initializes a model.

All code and sixteen validation-selected readers must be frozen before this
module can authorize any held-out draw. A freeze is provenance, not encryption.
"""
from __future__ import annotations

import json
import math
import importlib.metadata
import platform
from pathlib import Path

from .partial_compatibility_cache import encoded, sha_file
from .partial_compatibility_registry import ARM_NAMES, TRAINING_SEEDS, training_registry
from .shared_partial_data import OPEN_SPLITS, SPLITS

ROOT = Path(__file__).resolve().parents[2]
TEST_SPLITS = tuple(s for s in SPLITS if s not in OPEN_SPLITS)
ADDITIONAL_SOURCES = (
    "experiments/atencion_armonica/collect_partial_validation_logits.py",
    "experiments/atencion_armonica/analyze_partial_validation.py",
    "src/atencion_armonica/partial_compatibility_registry.py",
    "src/atencion_armonica/partial_compatibility_inference.py",
    "src/atencion_armonica/partial_compatibility_metrics.py",
    "src/atencion_armonica/partial_compatibility_evaluation.py",
    "src/atencion_armonica/partial_compatibility_test_gate.py",
    "src/atencion_armonica/partial_compatibility_test_cache.py",
    "experiments/atencion_armonica/freeze_partial_evaluation.py",
    "experiments/atencion_armonica/prepare_partial_test_cache.py",
    "experiments/atencion_armonica/collect_partial_test_logits.py",
    "experiments/atencion_armonica/analyze_partial_test.py",
)


def evaluation_sources():
    from experiments.atencion_armonica.train_partial_compatibility_campaign import SOURCE_NAMES
    return {name: sha_file(ROOT/name) for name in (*SOURCE_NAMES, *ADDITIONAL_SOURCES)}


def numerical_runtime():
    return {"python": platform.python_version(), "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"), "scikit_learn": importlib.metadata.version("scikit-learn")}


def verified_readers(selection, registry):
    manifest = json.loads((selection/"manifest.json").read_text())
    if (manifest["status"] != "VALIDATION_SELECTION_COMPLETE" or manifest["test_status"] != "CLOSED"
            or (selection/"FAILURE.json").exists()
            or sha_file(selection/"readers.json") != manifest["readers_sha256"]):
        raise ValueError("validation reader selection incomplete")
    readers = json.loads((selection/"readers.json").read_text())
    if (readers["status"] != "VALIDATION_SELECTION_COMPLETE" or readers["test_status"] != "CLOSED"
            or readers["source_sha256"] != manifest["source_sha256"]
            or readers["runtime"] != manifest["runtime"] or readers["runtime"] != numerical_runtime()
            or readers["training_manifest_sha256"] != registry["manifest_sha256"]):
        raise ValueError("reader selection belongs to another training campaign")
    for name, digest in readers["source_sha256"].items():
        if sha_file(ROOT/name) != digest:
            raise ValueError("selection implementation changed before test freeze")
    expected = {(arm, seed) for arm in ARM_NAMES for seed in TRAINING_SEEDS} | {("analytic_support", None)}
    seen = set()
    grid = [i/20 for i in range(1, 20)]
    for row in readers["readers"]:
        key = (row["arm"], row["seed"])
        name = "analytic_support" if key == ("analytic_support", None) else f"{key[0]}__seed_{key[1]}"
        if key not in expected or key in seen or row["threshold"] not in grid:
            raise ValueError("reader roster or threshold mismatch")
        seen.add(key)
        for prefix, suffix in (("selection", "selection.json"), ("metrics", "scene_metrics.jsonl"), ("summary", "summary.json")):
            if row[prefix+"_path"] != f"{name}/{suffix}" or sha_file(selection/row[prefix+"_path"]) != row[prefix+"_sha256"]:
                raise ValueError("reader artifact identity mismatch")
        chosen = json.loads((selection/row["selection_path"]).read_text())
        scores = chosen["mean_ari_grid"]
        if (chosen["selection_split"] != "validation" or chosen["scene_count"] != 1024
                or chosen["threshold_grid"] != grid or len(scores) != len(grid) or not all(math.isfinite(s) for s in scores)
                or chosen["threshold"] != row["threshold"]
                or grid[max(range(len(grid)), key=lambda i: scores[i])] != row["threshold"]):
            raise ValueError("reader was not chosen by the declared validation rule")
        if key in registry["cells"] and row["checkpoint_sha256"] != registry["cells"][key]["checkpoint_sha256"]:
            raise ValueError("selected reader/checkpoint pairing changed")
        if key == ("analytic_support", None) and row["checkpoint_sha256"] is not None:
            raise ValueError("analytic reference cannot have a neural checkpoint")
    if seen != expected:
        raise ValueError("all sixteen readers must be selected before test")
    return readers


def create_freeze(output, training, selection, raw_validation, audit):
    registry = training_registry(training)
    readers = verified_readers(selection, registry)
    if sha_file(raw_validation/"manifest.json") != readers["raw_manifest_sha256"]:
        raise ValueError("validation raw lineage differs")
    if not audit.is_file() or not audit.read_text().strip():
        raise ValueError("independent evaluation-implementation audit receipt required")
    bindings = {"training": training/"manifest.json", "selection": selection/"manifest.json",
                "readers": selection/"readers.json", "raw_validation": raw_validation/"manifest.json",
                "implementation_audit": audit}
    record = {"schema_version": 1, "status": "FROZEN_BEFORE_TEST", "test_splits": list(TEST_SPLITS),
              "source_sha256": evaluation_sources(),
              "bindings": {key: {"path": str(path.resolve()), "sha256": sha_file(path)} for key, path in bindings.items()},
              "readers": [{"arm": r["arm"], "seed": r["seed"], "threshold": r["threshold"],
                           "checkpoint_sha256": r["checkpoint_sha256"]} for r in readers["readers"]],
              "validation_manifest_sha256": readers["validation_manifest_sha256"]}
    record["runtime"] = readers["runtime"]
    with output.open("xb") as handle:
        handle.write(encoded(record))


def verify_freeze(path):
    record = json.loads(path.read_text())
    if (record["schema_version"] != 1 or record["status"] != "FROZEN_BEFORE_TEST"
            or record["test_splits"] != list(TEST_SPLITS) or record["source_sha256"] != evaluation_sources()
            or record["runtime"] != numerical_runtime()):
        raise ValueError("missing or changed prospective test freeze")
    for binding in record["bindings"].values():
        if sha_file(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("prospective evaluation binding changed")
    expected_bindings = {"training", "selection", "readers", "raw_validation", "implementation_audit"}
    expected_readers = {(arm, seed) for arm in ARM_NAMES for seed in TRAINING_SEEDS} | {("analytic_support", None)}
    if (set(record["bindings"]) != expected_bindings or len(record["readers"]) != 16
            or {(r["arm"], r["seed"]) for r in record["readers"]} != expected_readers
            or any(r["threshold"] not in [i/20 for i in range(1, 20)] for r in record["readers"])):
        raise ValueError("incomplete frozen readers")
    return record
