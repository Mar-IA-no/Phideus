"""Synthetic JSON fixtures exercise freeze/selection gates; no test scene is drawn."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.atencion_armonica import partial_compatibility_test_gate as gate
from src.atencion_armonica.partial_compatibility_cache import encoded, sha_file


def fixture(root):
    selection = root/"selection"
    selection.mkdir()
    training, raw = root/"training", root/"raw"
    training.mkdir()
    raw.mkdir()
    (training/"manifest.json").write_bytes(b"training fixture")
    (raw/"manifest.json").write_bytes(b"raw fixture")
    audit = root/"audit"
    audit.write_bytes(b"fixture review receipt, no operational authority")
    registry = {"manifest_sha256": sha_file(training/"manifest.json"), "cells": {}}
    rows = []
    for arm, seed in [(a, s) for a in gate.ARM_NAMES for s in gate.TRAINING_SEEDS]+[("analytic_support", None)]:
        name = "analytic_support" if seed is None else f"{arm}__seed_{seed}"
        cell = selection/name
        cell.mkdir()
        chosen = {"threshold": .05, "selection_split": "validation", "scene_count": 1024,
                  "threshold_grid": [i/20 for i in range(1, 20)], "mean_ari_grid": [1.]*19}
        (cell/"selection.json").write_bytes(encoded(chosen))
        (cell/"scene_metrics.jsonl").write_bytes(b"fixture")
        (cell/"summary.json").write_bytes(b"fixture")
        row = {"arm": arm, "seed": seed, "threshold": .05,
               "checkpoint_sha256": None if seed is None else "checkpoint fixture"}
        for prefix, filename in (("selection", "selection.json"), ("metrics", "scene_metrics.jsonl"), ("summary", "summary.json")):
            row[prefix+"_path"] = f"{name}/{filename}"
            row[prefix+"_sha256"] = sha_file(cell/filename)
        rows.append(row)
        if seed is not None:
            registry["cells"][(arm, seed)] = {"checkpoint_sha256": "checkpoint fixture"}
    readers = {"status": "VALIDATION_SELECTION_COMPLETE", "test_status": "CLOSED", "source_sha256": {},
               "runtime": gate.numerical_runtime(), "training_manifest_sha256": registry["manifest_sha256"],
               "raw_manifest_sha256": sha_file(raw/"manifest.json"), "readers": rows,
               "validation_manifest_sha256": "validation fixture"}
    (selection/"readers.json").write_bytes(encoded(readers))
    (selection/"manifest.json").write_bytes(encoded({"status": "VALIDATION_SELECTION_COMPLETE", "test_status": "CLOSED",
         "source_sha256": {}, "runtime": gate.numerical_runtime(), "readers_sha256": sha_file(selection/"readers.json")}))
    return training, selection, raw, audit, registry


class EvaluationFreezeTests(unittest.TestCase):
    def test_complete_fixture_freeze_and_drift_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training, selection, raw, audit, registry = fixture(root)
            with patch.object(gate, "training_registry", return_value=registry), \
                 patch.object(gate, "evaluation_sources", return_value={"fixture": "source"}):
                gate.create_freeze(root/"freeze.json", training, selection, raw, audit)
                record = gate.verify_freeze(root/"freeze.json")
                self.assertEqual(len(record["readers"]), 16)
                with self.assertRaises(FileExistsError):
                    gate.create_freeze(root/"freeze.json", training, selection, raw, audit)
                audit.write_bytes(b"changed receipt")
                with self.assertRaisesRegex(ValueError, "binding changed"):
                    gate.verify_freeze(root/"freeze.json")

    def test_unfinished_selection_or_source_drift_cannot_authorize_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training, selection, raw, audit, registry = fixture(root)
            with patch.object(gate, "training_registry", return_value=registry), \
                 patch.object(gate, "evaluation_sources", return_value={}):
                gate.create_freeze(root/"freeze.json", training, selection, raw, audit)
            with patch.object(gate, "evaluation_sources", return_value={"changed": "source"}):
                with self.assertRaisesRegex(ValueError, "changed prospective"):
                    gate.verify_freeze(root/"freeze.json")
            (selection/"FAILURE.json").write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "selection incomplete"):
                gate.verified_readers(selection, registry)


if __name__ == "__main__":
    unittest.main()
