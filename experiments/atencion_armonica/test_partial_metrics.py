"""Mechanical scene fixtures only: no access to prospective test splits."""
import unittest

import numpy as np
from scipy.special import expit

from src.atencion_armonica.partial_compatibility_metrics import (
    paired_bootstrap, pair_metrics, partition_labels, scene_metrics, seed_disagreement, select_reader,
)


class MetricsTests(unittest.TestCase):
    def test_perfect_partition_and_near_denominators(self):
        truth = [0, 0, 1, 1]
        p = np.equal.outer(truth, truth).astype(float)*.8+.1
        result = scene_metrics(p, truth, [0, .001, .4, .8], .5)
        self.assertEqual(result["ari"], 1)
        self.assertTrue(result["exact_partition"])
        self.assertEqual(result["pairs"]["n_pairs"], 12)
        self.assertEqual(result["near_collision_pairs"]["n_pairs"], 2)
        self.assertAlmostEqual(result["pairs"]["brier"], .01)
        permutation = [2, 0, 3, 1]
        permuted = scene_metrics(p[np.ix_(permutation, permutation)],
                                 [8, 3, 8, 3], np.array([0, .001, .4, .8])[permutation], .5)
        for key in ("pairs", "near_collision_pairs", "ari", "exact_partition", "k_error"):
            self.assertEqual(result[key], permuted[key])

    def test_extreme_logits_single_class_empty_and_constant(self):
        z = np.array([-1000., 1000.])
        result = pair_metrics(expit(z), [1, 0], z)
        self.assertEqual(result["bce"], 1000)
        self.assertEqual(result["binary_entropy_nats"], 0)
        self.assertEqual(pair_metrics([], [])["status"], "NO_PAIRS")
        one = pair_metrics([.1, .1], [0, 0])
        self.assertIsNone(one["positive_recall_at_half"])
        self.assertIsNone(one["auc"])
        self.assertTrue(one["constant_hard_prediction_at_half"])
        self.assertTrue(one["constant_probability"])
        with self.assertRaises(ValueError):
            pair_metrics([.5], [1], [1])

    def test_float32_sigmoid_is_not_rejected_as_corruption(self):
        z = np.array([-.3, .7, -8., 12.], dtype=np.float32)
        p = expit(z)
        result = pair_metrics(p, [0, 1, 0, 1], z)
        expected = np.mean(np.logaddexp(0, z.astype(np.float64))-np.array([0, 1, 0, 1])*z)
        self.assertEqual(result["bce"], expected)
        p[0] += .001
        with self.assertRaises(ValueError):
            pair_metrics(p, [0, 1, 0, 1], z)

    def test_selection_only_validation_lowest_tie_no_true_k(self):
        p = np.eye(3)
        chosen = select_reader([p], [[2, 3, 4]], split="validation")
        self.assertEqual(chosen["threshold"], .05)
        self.assertEqual(chosen["mean_ari_grid"], [1.]*19)
        with self.assertRaises(PermissionError):
            select_reader([p], [[2, 3, 4]], split="iid")
        with self.assertRaises(ValueError):
            partition_labels(((0,), (0, 1)), 2)

    def test_bootstrap_scene_units_sign_and_reproducibility(self):
        a, b = np.full(1024, .1), np.full(1024, .2)
        result = paired_bootstrap(a, b, control_index=0, seed=2026090721)
        self.assertAlmostEqual(result["mean_delta"], -.1)
        np.testing.assert_allclose(result["percentile_95"], [-.1, -.1])
        self.assertEqual(result, paired_bootstrap(a, b, control_index=0, seed=2026090721))
        with self.assertRaises(ValueError):
            paired_bootstrap(a[:2], b[:2], control_index=0, seed=0)

    def test_seed_disagreement_retains_scene_unit(self):
        p = np.full((3, 2, 2), .3)
        zero = seed_disagreement(p)
        self.assertEqual(zero["mean_probability_sd_across_three_seeds"], 0)
        p[1] = .8
        some = seed_disagreement(p)
        self.assertEqual(some["fraction_pairs_with_hard_seed_disagreement"], 1)
        self.assertEqual(some["n_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
