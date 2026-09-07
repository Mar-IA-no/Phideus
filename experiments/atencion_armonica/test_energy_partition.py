"""CPU/NumPy tests; run with unittest, avoiding unrelated pytest conftest."""

import inspect
import itertools
import math
import sys
import unittest

import numpy as np

from atencion_armonica.energy_partition import (
    accessible_amplitudes, canonical_partition, solve_energy_partition, valid_partition,
)
from atencion_armonica.peak_tokens import compute_tokens


def unique_fixture():
    return np.sqrt(np.array([1/11, 2/11, 3/11, 5/11, 1/38, 7/38, 13/38, 17/38]))


class EnergyPartitionTests(unittest.TestCase):
    def test_unique_and_permutation(self):
        a = unique_fixture()
        truth = ((0, 1, 2, 3), (4, 5, 6, 7))
        for seed in (2026090701, 2026090702):
            perm = np.random.default_rng(np.random.SeedSequence([seed, 123])).permutation(8)
            result = solve_energy_partition(a[perm], tolerance=1e-10)
            self.assertEqual(result["status"], "UNIQUE")
            self.assertTrue(result["search_exhaustive"])
            restored = canonical_partition(perm[block] for block in result["solutions"][0])
            self.assertEqual(restored, truth)

    def test_float32_channel(self):
        a = unique_fixture()
        decoded = accessible_amplitudes(a)
        actual = np.exp(compute_tokens(np.arange(1, 9), a)[:, 1].astype(np.float64))
        np.testing.assert_array_equal(decoded, actual)
        result = solve_energy_partition(decoded, tolerance=1e-6)
        self.assertEqual(result["status"], "UNIQUE")

    def test_multiple_quotients_not_relabelings(self):
        a = np.full(8, 0.5)
        result = solve_energy_partition(a, tolerance=1e-10)
        self.assertEqual(result["status"], "MULTIPLE")
        self.assertEqual(len(result["solutions"]), 2)
        self.assertNotEqual(*map(canonical_partition, result["solutions"]))
        for blocks in result["solutions"]:
            self.assertTrue(valid_partition(a, blocks, 2, 1e-10))

    def test_no_partition_with_integer_total(self):
        result = solve_energy_partition(np.sqrt([1.2] + [0.8/7]*7), tolerance=1e-10)
        self.assertEqual(result["status"], "NO_PARTITION")
        self.assertEqual(result["k"], 2)
        self.assertTrue(result["search_exhaustive"])

    def test_caps_never_certify_unique(self):
        a = unique_fixture()
        self.assertEqual(solve_energy_partition(a, tolerance=1e-10, max_candidates=1)["status"],
                         "LIMIT_CANDIDATES")
        self.assertEqual(solve_energy_partition(a, tolerance=1e-10, max_nodes=1)["status"],
                         "LIMIT_NODES")
        self.assertEqual(solve_energy_partition(np.ones(25), tolerance=1e-10)["status"],
                         "LIMIT_INPUT_SIZE")

    def test_intervention_is_prior_sanity(self):
        a = unique_fixture() * np.repeat([0.7, 1.1], 4)
        result = solve_energy_partition(a, tolerance=1e-10)
        self.assertEqual(result["status"], "PRIOR_VIOLATION")
        self.assertEqual(result["nodes"], 0)
        self.assertAlmostEqual(result["total_energy"], 1.70)

    def test_ambiguous_k_not_arbitrarily_selected(self):
        result = solve_energy_partition(unique_fixture(), tolerance=2)
        self.assertEqual(result["status"], "AMBIGUOUS_K")
        self.assertIsNone(result["k"])

    def test_invalid(self):
        for values in ([], [[1]], [0], [-1], [float("nan")], [float("inf")], [1j]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                solve_energy_partition(values, tolerance=1e-10)
        for tolerance in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                solve_energy_partition([1], tolerance=tolerance)

    def test_candidates_against_independent_enumeration(self):
        a = unique_fixture()
        expected = set()
        for size in range(4, 9):
            for block in itertools.combinations(range(8), size):
                if abs(math.fsum(float(a[i])*float(a[i]) for i in block)-1) <= 1e-10:
                    expected.add(sum(1 << i for i in block))
        result = solve_energy_partition(a, tolerance=1e-10)
        self.assertEqual(set(result["candidate_masks"]), expected)

    def test_interface_and_no_torch(self):
        self.assertEqual(list(inspect.signature(solve_energy_partition).parameters),
                         ["amplitudes", "tolerance", "max_candidates", "max_nodes"])
        self.assertNotIn("torch", sys.modules)
        self.assertEqual(canonical_partition([[3, 2], [1, 0]]), ((0, 1), (2, 3)))


if __name__ == "__main__":
    unittest.main()
