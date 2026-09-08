"""Selection/test gate denials and fixed candidate serialization; no campaign."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.atencion_armonica.test_learned_partition_data import BASE
from experiments.atencion_armonica.test_learned_partition_cache import fixture
from src.atencion_armonica import learned_partition_selection as s
from src.atencion_armonica import learned_partition_test as t


class SelectionTests(unittest.TestCase):
    def test_incomplete_freeze_and_draft_denied_before_corpus_reads(self):
        with patch.object(s, "selection_context") as load:
            with self.assertRaises(ValueError):
                s.validate_freeze({"status": "SELECTION_FROZEN", "cells": []}, {})
            load.assert_not_called()
        with patch.object(s.p, "read_reference", return_value={"selection": None}), patch.object(s, "validate_freeze") as validate:
            with self.assertRaises(ValueError):
                s.verify_selection_chain({}, {})
            validate.assert_not_called()

    def test_test_authorization_requires_independent_target_audit_before_write(self):
        BASE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            output = Path(folder)/"not_written.json"
            with patch.object(s.gate, "common_binding", return_value={}), \
                 patch.object(s.gate, "verify_audit", side_effect=PermissionError), patch.object(s, "verify_selection_chain") as validate:
                with self.assertRaises(PermissionError):
                    s.create_test_authorization(output, freeze={}, freeze_audit={})
                validate.assert_not_called()
                self.assertFalse(output.exists())

    def test_prediction_roster_and_exact_roundtrip(self):
        roster = t.inference_roster()
        self.assertEqual(len(roster), 99)
        self.assertEqual(len({t.prefix(r) for r in roster}), 99)
        self.assertEqual(sum(r["intervention"] == "original" for r in roster), 36)
        self.assertEqual(sum(r["intervention"] == "original_sham" for r in roster), 9)
        rows = [fixture()]*512
        values = [np.array([[.1, .2], [.3, .4]], np.float32)]*512
        with tempfile.TemporaryDirectory(dir=BASE) as folder:
            path = Path(folder)/"predictions.npz"
            t.write_predictions(path, values)
            restored = t.read_predictions(path, rows)
            for before, after in zip(values, restored):
                np.testing.assert_array_equal(before, after)
            with self.assertRaises(ValueError):
                t.read_predictions(path, rows[:-1])

    def test_test_denial_precedes_truth_and_forward(self):
        with patch.object(t.runner, "stage_inputs", side_effect=PermissionError), patch.object(t, "load_supervision") as truth:
            with self.assertRaises(PermissionError):
                t.evaluate_test("unused", "iid", authorization={}, data={}, logits={}, scored={}, normalized={}, predictions={})
            with self.assertRaises(PermissionError):
                t.test_inputs("train", authorization={}, data={}, logits={}, scored={}, normalized={})
            truth.assert_not_called()

    def test_support_requires_all_cases_and_preserves_scene_denominators(self):
        scene = {"input": {"status": "INPUT_CHANGED"}, "prediction": {
            "maximum_absolute_component_delta": .1, "decision_changed": False, "constant_sum_delta": False}}
        records = [{**r, "scenes": [scene]*512} for r in t.inference_roster() if r["intervention"] != "original"]
        result = t.summarize_support(records)["shared_source/original_sham"]
        self.assertEqual(result["input_changed_cases"], 1536)
        self.assertEqual(result["component_changed_cases"], 4608)
        self.assertEqual(result["excluded_from_primary_metrics"], 0)
        self.assertTrue(all(r["input_supported"] == 3 and r["trained_supported"] == 9 for r in result["coverage_by_scene"]))
        with self.assertRaises(ValueError):
            t.summarize_support(records[:-1])

    def test_support_replay_uses_inputs_and_predictions_without_model_forward(self):
        row = fixture()
        norm = {"mean": np.zeros(5), "scale": np.ones(5)}
        original = np.array([[.1, .2], [.3, .4]], np.float32)
        changed = np.array([[.5, .6], [.1, .2]], np.float32)
        with patch("src.atencion_armonica.learned_partition_inference.predict_inputs",
                   side_effect=AssertionError("replay must not forward")):
            result = t.replay_support([row], norm, "shared_source", "zero", [original], [changed])
        self.assertEqual(result[0]["input"]["status"], "INPUT_CHANGED")
        self.assertTrue(result[0]["prediction"]["decision_changed"])
        self.assertGreater(result[0]["prediction"]["maximum_absolute_component_delta"], 0)
        with self.assertRaises(ValueError):
            t.replay_support([row], norm, "shared_source", "zero", [], [changed])

    def test_interventions_reuse_candidate_metrics_with_paired_descriptive_intervals(self):
        row = fixture()
        changed = np.array([[1., 1.], [0., 0.]], np.float32)
        predictions = {(r["arm"], r["checkpoint_seed"], r["reader_seed"], r["intervention"]): [changed]
                       for r in t.inference_roster() if r["intervention"] != "original"}
        baseline = dict.fromkeys(t.METRICS, 0.)
        candidate = dict.fromkeys(t.METRICS, 1.)
        evaluated = {"candidate_metrics": [baseline, candidate], "learned": {
            (arm, reader): {"metrics": baseline, "choice": {"candidate_index": 0}}
            for arm in t.ARMS for reader in t.READER_SEEDS}}
        initial = []
        for seed in t.SEEDS:
            initial.extend(t.intervention_metrics_for_scene(0, seed, row, predictions, evaluated))
        self.assertEqual(len(initial), 63)
        self.assertTrue(all(r["candidate_index"] == 1 and r["delta_vs_original"]["k_inferred"] == 1 for r in initial))
        records = [{**r, "scene_id": i} for r in initial for i in range(512)]
        indices = t.bootstrap_indices()
        result = t.summarize_interventions(records, split="ood_polyphony", indices=indices)
        case = result["interventions"]["shared_source/zero"]
        self.assertEqual(case["cases"], 4608)
        for key in ("ari", "k_inferred", "sub3_member_fraction"):
            self.assertEqual(case["delta_vs_original"][key], {
                "delta": 1., "interval": [1., 1.], "nominal_coverage": .95, "family": "descriptive"})
        with self.assertRaises(ValueError):
            t.summarize_interventions(records[:-1], split="ood_polyphony", indices=indices)


if __name__ == "__main__":
    unittest.main()
