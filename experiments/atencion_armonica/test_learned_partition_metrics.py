"""Synthetic metric vectors only; no campaign data, predictions or GPU."""
import unittest

import numpy as np

from src.atencion_armonica.learned_partition_metrics import (ARMS, READER_SEEDS, SEEDS, EPOCHS, SPLITS,
    METRICS, REFERENCES, bootstrap_indices, candidate_targets, evaluate_preserved_predictions, select_epochs, summarize_test)
from src.atencion_armonica.structured_source_reader import partition_energies, signature


def calibration_records():
    return [{"arm": a, "checkpoint_seed": c, "reader_seed": s, "epoch": e,
             "split": "calibration", "split_seed": SPLITS["calibration"][1],
             "scene_ids": list(range(512)), "ari": [(.7 if e in (10, 15) else .6)]*512}
            for a in ARMS for c in SEEDS for s in READER_SEEDS for e in EPOCHS]


def test_records():
    rows = []
    for i in range(512):
        for ci, c in enumerate(SEEDS):
            for si, s in enumerate(READER_SEEDS):
                metrics = {}
                for name in (*ARMS, *REFERENCES):
                    m = dict.fromkeys(METRICS, 0.)
                    m["ari"] = ((.8 if name == "shared_source" else .7)+ci*.01+si*.002
                                if name in ARMS else .65+ci*.01)
                    m["k_inferred"] = 2.
                    metrics[name] = m
                rows.append({"scene_id": i, "checkpoint_seed": c, "reader_seed": s,
                             "split": "ood_polyphony", "split_seed": SPLITS["ood_polyphony"][1],
                             "metrics": metrics})
    return rows


class MetricTests(unittest.TestCase):
    def test_full_preserved_prediction_evaluation_keeps_oracles_separate(self):
        z = np.zeros((4, 4), np.float32)
        candidates = sorted([signature([[0], [1], [2], [3]]), signature([[0, 1], [2, 3]])])
        groups = sorted({g for p in candidates for g in p})
        scored = {"pool": {"canonical_to_observed": [1, 3, 0, 2]},
                  "candidates": partition_energies(z.astype(np.float64), candidates, groups,
                      {a: np.zeros(len(groups)) for a in ARMS[1:]})}
        predictions = {(a, s): np.array([[.4, 0.], [0., 0.]], np.float32) for a in ARMS for s in READER_SEEDS}
        result = evaluate_preserved_predictions(scored, np.array([1, 0, 1, 0]), z, SEEDS[0], predictions)
        self.assertEqual(result["oracle"]["maximum_ari"], 1.)
        self.assertEqual(result["oracle"]["minimum_vi"], 0.)
        self.assertEqual(result["neural_brier"], .25)
        self.assertEqual(set(result["references"]), set(REFERENCES))
        for r in result["learned"].values():
            self.assertEqual(set(r["metrics"]), set(METRICS))
            self.assertEqual(r["metrics"]["ari"], 1.)
            self.assertEqual(r["metrics"]["vi"], 0.)

    def test_targets_use_canonical_to_observed_mapping(self):
        scored = {"pool": {"canonical_to_observed": [1, 3, 0, 2]},
                  "candidates": [{"signature": [[0, 1], [2, 3]]},
                                 {"signature": [[0], [1], [2], [3]]}]}
        r = candidate_targets(scored, np.array([1, 0, 1, 0]))
        np.testing.assert_array_equal(r["targets"], [[0., 0.], [.5, 0.]])
        self.assertEqual(r["raw"].dtype, np.float64)
        self.assertEqual(r["targets"].dtype, np.float32)

    def test_one_epoch_per_arm_lowest_tie(self):
        result = select_epochs(calibration_records())
        self.assertEqual({r["epoch"] for r in result["selected"].values()}, {10})
        self.assertEqual(result["count"], 512)
        self.assertEqual(len(result["selected"]), 4)

    def test_selection_rejects_test_role_and_missing_cell(self):
        rows = calibration_records()
        with self.assertRaises(ValueError):
            select_epochs(rows[:-1])
        rows[-1]["split"] = "ood_polyphony"
        with self.assertRaises(ValueError):
            select_epochs(rows)
        rows = calibration_records()
        rows[0]["epoch"] = float(rows[0]["epoch"])
        with self.assertRaises(ValueError):
            select_epochs(rows)

    def test_primary_scene_paired_family_and_reference_repetition(self):
        rows = test_records()
        indices = bootstrap_indices()
        result = summarize_test(rows, split="ood_polyphony", indices=indices)
        self.assertEqual(result["scene_count"], 512)
        self.assertEqual(indices.shape, (2000, 512))
        for name in ARMS:
            if name == "shared_source":
                continue
            contrast = result["contrasts_shared_minus_control"][name]["ari"]
            self.assertAlmostEqual(contrast["delta"], .1, places=14)
            np.testing.assert_allclose(contrast["interval"], [.1, .1], rtol=0, atol=1e-14)
            self.assertAlmostEqual(contrast["nominal_coverage"], 1-.05/3)
        self.assertEqual(result["contrasts_shared_minus_control"]["historical"]["ari"]["nominal_coverage"], .95)
        rows[1]["metrics"]["fixed_pairs"]["ari"] += .1
        with self.assertRaises(ValueError):
            summarize_test(rows, split="ood_polyphony", indices=indices)

    def test_wrong_bootstrap_or_missing_replica_rejected(self):
        rows = test_records()
        with self.assertRaises(ValueError):
            summarize_test(rows[:-1], split="ood_polyphony", indices=bootstrap_indices())
        wrong = bootstrap_indices()
        wrong[0, 0] = (wrong[0, 0]+1) % 512
        with self.assertRaises(ValueError):
            summarize_test(rows, split="ood_polyphony", indices=wrong)


if __name__ == "__main__":
    unittest.main()
