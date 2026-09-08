"""Mechanical CPU fixtures only; no corpus, Torch, accelerator or forwards."""
import inspect
from itertools import combinations
import sys
import unittest

import numpy as np

from src.atencion_armonica.source_coherence import CENTS, GroupFitCache, SourceFitter
from src.atencion_armonica.source_loss_pressure import edge_pressure, evaluate_pressure


class SourceFitterTests(unittest.TestCase):
    def setUp(self):
        self.fitter = SourceFitter()

    def test_known_family_witness_and_nested_grids(self):
        indices = np.array([1, 2, 3, 5, 7, 8])
        beta = self.fitter.grid[512]
        q = (np.log(indices)+.5*np.log1p(beta*indices**2)-1.7).astype(np.float32)
        r = self.fitter.fit(q, np.arange(len(q)))
        self.assertEqual(r["status"], "GRID_WITNESS_APPROXIMATE")
        self.assertLess(r["fine"]["minimum_cents"], .001)
        self.assertEqual(r["fine"]["witness_indices"], indices.tolist())
        self.assertEqual(r["fine"]["witness_grid_index"], 512)
        self.assertGreaterEqual(r["coarse_minus_fine_cents"], -1e-9)
        witness = r["fine"]
        residual = (q.astype(float)-np.log(witness["witness_indices"])
                    -.5*np.log1p(witness["witness_beta"]*np.square(witness["witness_indices"]))
                    -witness["witness_offset"])
        self.assertAlmostEqual(CENTS*np.sqrt(np.mean(residual**2)), witness["witness_residual_cents"], places=8)

    def test_permutation_and_offset_are_not_source_identification(self):
        q = np.log(np.array([1., 2., 4., 6., 8.])).astype(np.float32)
        a = self.fitter.fit(q, np.arange(5))
        b = self.fitter.fit(q[::-1].copy(), np.arange(5))
        self.assertEqual(a["fine"], b["fine"])
        c = self.fitter.fit((q+np.float32(1.5)).astype(np.float32), np.arange(5))
        self.assertAlmostEqual(a["fine"]["minimum_cents"], c["fine"]["minimum_cents"], delta=.001)
        self.assertEqual(c["fine"]["index_authority"], "WITNESS_NOT_IDENTIFIED")
        self.assertEqual(c["fine"]["offset_authority"], "CENTERED_LOG_GAUGE")

    def test_tie_count_gap_and_canonical_witness(self):
        residual = np.array([[2., 1.+.5e-9, 4.], [1., 3., 5.]])
        r = self.fitter._witness(residual, np.array([[1, 2, 3], [1, 2, 4]]), np.zeros((2, 3)), np.arange(3))
        self.assertEqual(r["minimum_cents"], 1.)
        self.assertEqual(r["co_minimum_count"], 2)
        self.assertEqual(r["witness_indices"], [1, 2, 3])
        self.assertEqual(r["witness_grid_index"], 1)
        self.assertEqual(r["next_level_gap_cents"], 1.)

    def test_cardinality_and_no_privileged_inputs(self):
        q = np.arange(12, dtype=np.float32)
        for members, status in (([0], "UNDERCONSTRAINED"), ([0, 1], "UNDERCONSTRAINED"),
                                (list(range(9)), "OUTSIDE_DECLARED_CARDINALITY")):
            self.assertEqual(self.fitter.fit(q, members)["status"], status)
        self.assertEqual(list(inspect.signature(self.fitter.fit).parameters), ["observed_q", "members"])
        for members in ([], [0, 0], [-1], [12], [0., 1.]):
            with self.assertRaises(ValueError):
                self.fitter.fit(q, members)
        with self.assertRaises(ValueError):
            self.fitter.fit(q.astype(np.float64), [0, 1, 2])

    def test_cache_identity_and_permuted_members(self):
        cache = GroupFitCache(self.fitter)
        q = np.arange(8, dtype=np.float32)/4
        a = cache.fit("mechanical", 0, q, [0, 1, 3])
        b = cache.fit("mechanical", 0, q, [3, 0, 1])
        self.assertEqual(a, b)
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        with self.assertRaises(ValueError):
            cache.fit("mechanical", 0, q+1, [0, 1, 3])
        with self.assertRaises(ValueError):
            cache.fit("mechanical", 0, q, [0., 1., 3.])
        cache.fit("another_split", 0, q, [0, 1, 3])
        self.assertEqual(cache.misses, 2)


class PressureTests(unittest.TestCase):
    def fixture(self):
        z = np.array([[0., .2, -1., .7], [.2, 0., .5, -.1], [-1., .5, 0., 1.1], [.7, -.1, 1.1, 0.]])
        triples = np.array(list(combinations(range(4), 3)))
        weights = np.array([.1, .3, .8, .5], dtype=np.float32)
        return z, triples, weights

    def test_derivatives_against_symmetric_finite_differences(self):
        z, t, w = self.fixture()
        raw = edge_pressure(z, t, w, w[::-1], sham_evaluable=True)
        summary, evaluation = evaluate_pressure(raw, [0, 0, 0, 1])
        step = 1e-5
        for edge, (i, j) in enumerate(raw["edges"]):
            values = []
            for sign in (-1, 1):
                changed = z.copy()
                changed[i, j] += sign*step
                changed[j, i] += sign*step
                other = edge_pressure(changed, t, w, w[::-1], sham_evaluable=True)
                labels = np.array([0, 0, 0, 1])
                target = labels[:, None] == labels[None, :]
                mask = ~np.eye(4, dtype=bool)
                bce = np.mean((np.logaddexp(0., changed)-target*changed)[mask])
                values.append([bce, other["physical_contributions"].mean(), other["sham_contributions"].mean()])
            measured = (np.array(values[1])-values[0])/(2*step)
            expected = [evaluation["bce_derivative"][edge], raw["physical_derivative"][edge], raw["sham_derivative"][edge]]
            np.testing.assert_allclose(measured, expected, atol=1e-10, rtol=1e-7)
        self.assertEqual(summary["penalties"]["physical"]["true_triple_count"], 1)
        self.assertAlmostEqual(summary["penalties"]["physical"]["L_truth"], float(w[0])/4)
        self.assertEqual(summary["classes"]["1"]["edge_count"], 3)

    def test_missing_classes_zero_denominators_and_disabled_sham(self):
        z, t, w = self.fixture()
        z.fill(1000.)
        raw = edge_pressure(z, t, w, np.zeros(4), sham_evaluable=False)
        result, _ = evaluate_pressure(raw, [0, 0, 0, 0])
        self.assertEqual(result["classes"]["0"]["P_ratio_status"], "NO_CLASS")
        self.assertEqual(result["classes"]["1"]["P_ratio_status"], "ZERO_BCE_DENOMINATOR")
        self.assertIsNone(result["classes"]["1"]["P_over_B"])
        self.assertIsNone(result["penalties"]["sham"]["F_true"])
        self.assertEqual(result["penalties"]["sham"]["status"], "SHAM_NOT_EVALUABLE")
        self.assertEqual(result["penalties"]["sham"]["F_true_status"], "SHAM_NOT_EVALUABLE")
        self.assertFalse(result["sham_evaluable"])
        result, _ = evaluate_pressure(raw, [0, 1, 2, 3])
        self.assertIsNone(result["penalties"]["physical"]["mean_weight_true"])
        self.assertGreater(result["classes"]["0"]["B"], 0.)
        self.assertEqual(result["classes"]["0"]["P_ratio_status"], "EVALUABLE")
        self.assertEqual(result["classes"]["0"]["S_ratio_status"], "SHAM_NOT_EVALUABLE")
        self.assertIsNone(result["classes"]["0"]["S_over_B"])
        self.assertEqual(result["classes"]["0"]["S"], 0.)

    def test_reject_asymmetry_incomplete_triples_and_nonfinite(self):
        z, t, w = self.fixture()
        bad = z.copy()
        bad[0, 1] += .001
        for logits, triples, weights, enabled in ((bad, t, w, True), (z, t[:-1], w, True),
                                                 (z, t, w*np.nan, True), (z, t, w, False)):
            with self.assertRaises(ValueError):
                edge_pressure(logits, triples, weights, w, sham_evaluable=enabled)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
