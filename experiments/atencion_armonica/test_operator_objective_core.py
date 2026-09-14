"""Small mathematical fixtures, never new scenes for the scientific campaign."""
import math
import unittest

import numpy as np
from scipy.stats import kendalltau

from src.atencion_armonica import operator_objective_core as core
from src.atencion_armonica.learned_partition_core import partition_errors


class TargetTests(unittest.TestCase):
    def test_entropy_components_and_float32_match_inherited_contract(self):
        ps = [((0, 1), (2, 3)), ((0, 2), (1, 3))]
        y = np.array([0, 1, 0, 1])
        result = core.partition_targets(ps, y)
        np.testing.assert_allclose(result['raw'][0], [math.log(2), math.log(2)], atol=1e-15)
        np.testing.assert_array_equal(result['u64'], [[.5, .5], [0., 0.]])
        np.testing.assert_array_equal(result['ari'], [-.5, 1.])
        np.testing.assert_array_equal(result['exact'], [False, True])
        for i, p in enumerate(ps):
            inherited = partition_errors(p, y)
            np.testing.assert_array_equal(result['raw'][i], inherited['raw'])
            np.testing.assert_array_equal(result['u32'][i], inherited['normalized'].astype(np.float32))

    def test_empty_roster_is_not_a_fake_candidate(self):
        target = core.partition_targets([], np.array([0, 0, 1, 1]))
        self.assertEqual(target['u32'].shape, (0, 2))
        self.assertEqual(target['t64'].shape, (0,))
        row = core.describe_score(np.array([]), target, [])
        self.assertIsNone(row['decision'])
        self.assertEqual(row['tau']['status'], 'NO_CANDIDATE')

    def test_invalid_partition_order_overlap_and_float_labels_rejected(self):
        for ps, labels in [([((1, 0), (2, 3))], np.array([0, 0, 1, 1])),
                           ([((0, 1), (1, 2, 3))], np.array([0, 0, 1, 1])),
                           ([((0, 1), (2, 3))], np.array([0., 0., 1., 1.])),
                           ([((0, 1), (2, 3)), ((0, 1), (2, 3))], np.array([0, 0, 1, 1]))]:
            with self.assertRaises(ValueError):
                core.partition_targets(ps, labels)

    def test_single_group_and_singletons_degenerate_ari(self):
        for p, y in [([((0, 1, 2, 3),)], np.zeros(4, dtype=int)),
                     ([((0,), (1,), (2,), (3,))], np.arange(4))]:
            result = core.partition_targets(p, y)
            self.assertEqual(result['ari'][0], 1.)
            self.assertEqual(result['t64'][0], 0.)


class OrderTests(unittest.TestCase):
    def test_order_retains_global_indices_and_cooptima(self):
        value = core.order_record(np.array([9., 2., 2., 1.]), [1, 2, 3])
        self.assertEqual(value['order'], [3, 1, 2])
        self.assertEqual(value['tie_blocks'], [[3], [1, 2]])
        self.assertEqual(value['next_gap'], 1.)
        value = core.order_record(np.array([9., 2., 2., 1.]), [1, 2])
        self.assertEqual(value['chosen'], 1)
        self.assertEqual(value['optima'], [1, 2])
        self.assertIsNone(value['next_gap'])

    def test_order_empty_and_invalid(self):
        self.assertEqual(core.order_record([])['status'], 'NO_CANDIDATE')
        for values, ids in [([0., float('nan')], None), ([1., 2.], [1, 0]),
                            ([1., 2.], [1, 1]), ([1., 2.], [0.5])]:
            with self.assertRaises(ValueError):
                core.order_record(values, ids)

    def test_tau_known_tie_counts_and_reference(self):
        x, y = np.array([0., 0., 1., 1.]), np.array([0., 1., 1., 2.])
        result = core.kendall_tau_b(x, y)
        self.assertEqual(result['concordant'], 3)
        self.assertEqual(result['score_only_ties'], 2)
        self.assertEqual(result['target_only_ties'], 1)
        self.assertAlmostEqual(result['value'], 3/math.sqrt(20))
        for a, b in [(x, y), (x, -y), (np.arange(82), -np.arange(82)),
                     (np.arange(82) % 5, np.arange(82) % 7)]:
            self.assertAlmostEqual(core.kendall_tau_b(a, b)['value'], kendalltau(a, b).statistic)

    def test_tau_undefined_is_not_zero(self):
        for a, b, status in [([], [], 'NO_CANDIDATE'), ([1], [2], 'INSUFFICIENT_PAIRS'),
                              ([1, 1], [2, 2], 'BOTH_CONSTANT'),
                              ([1, 1], [1, 2], 'CONSTANT_SCORE'),
                              ([1, 2], [1, 1], 'CONSTANT_TARGET')]:
            row = core.kendall_tau_b(a, b)
            self.assertEqual(row['status'], status)
            self.assertIsNone(row['value'])

    def test_tau_finite_extremes_do_not_overflow(self):
        x = np.array([-np.finfo(float).max, 0, np.finfo(float).max])
        with np.errstate(over='raise'):
            self.assertEqual(core.kendall_tau_b(x, x)['value'], 1.)


class RegressionOracleTests(unittest.TestCase):
    def test_sum_float32_before_error_not_recomposed_float64(self):
        h = np.array([[.5, 2**-25]], np.float32)
        u = np.zeros((1, 2), np.float32)
        result = core.regression_errors(h, u)
        self.assertEqual(result['s32'][0], .5)
        self.assertEqual(result['sum_squared'][0], .25)
        self.assertNotEqual(result['sum_squared'][0], h.astype(float).sum(axis=1)[0]**2)
        np.testing.assert_array_equal(result['component_bias'], h.astype(float))
        self.assertEqual(result['loss_like'], result['component_squared'].mean())

    def test_regression_no_rows_and_wrong_dtype(self):
        empty = np.empty((0, 2), np.float32)
        self.assertIsNone(core.regression_errors(empty, empty)['loss_like'])
        with self.assertRaises(ValueError):
            core.regression_errors(np.zeros((1, 2)), np.zeros((1, 2), np.float32))

    def test_different_canonical_oracles_can_have_overlapping_optima(self):
        result = core.oracle_report(np.array([0., 0., 1.]), np.array([0., 0., 1.], np.float32),
                                    np.array([0., 1., 1.]))['comparisons']
        self.assertFalse(result['target64_ari_same_choice'])
        self.assertTrue(result['target64_ari_optima_intersect'])
        self.assertEqual(result['ari_gap_of_target64_choice'], 1.)
        self.assertEqual(result['target64_regret_of_ari_choice'], 0.)

    def test_target32_rounding_changes_choice_and_records_regret(self):
        t = np.array([1., 1.-2e-8])
        result = core.oracle_report(t, t.astype(np.float32), np.array([.1, .2]))
        self.assertEqual(result['target64']['chosen'], 1)
        self.assertEqual(result['target32']['chosen'], 0)
        self.assertGreater(result['comparisons']['target32_target64_regret'], 0)
        self.assertFalse(result['comparisons']['target32_in_near_target64_optima'])

    def test_score_uses_its_own_oracles_inside_stratum(self):
        targets = core.partition_targets([((0, 1), (2, 3)), ((0, 2), (1, 3))], np.array([0, 1, 0, 1]))
        full = core.describe_score(np.array([0., 1.]), targets)
        subset = core.describe_score(np.array([0., 1.]), targets, [0])
        self.assertEqual(full['decision']['target64_regret'], 1.)
        self.assertEqual(subset['decision']['target64_regret'], 0.)
        self.assertEqual(subset['tau']['status'], 'INSUFFICIENT_PAIRS')
        self.assertFalse(subset['decision']['exact'])


class AggregationTests(unittest.TestCase):
    def test_strata_keep_full_distinct_and_groups_complete(self):
        ps = [((0, 1), (2, 3)), ((0, 1, 2), (3,))]
        result = core.observable_strata(ps, np.ones((2, 3), bool), ['base-low', 'deformed-low'])
        self.assertEqual(result['full'], {'all': [0, 1]})
        self.assertEqual(len(result['k']), 1)
        self.assertEqual(len(result['sizes_available']), 2)
        for strata in result.values():
            self.assertEqual(sorted(i for ids in strata.values() for i in ids), [0, 1])

    def test_paired_cell_first_and_complete_nine_are_distinct(self):
        left = {str(i): {'a': 1., 'b': 100.} for i in range(9)}
        right = {str(i): {'a': 0., 'b': 0.} for i in range(9)}
        right['0']['b'] = None
        result = core.paired_scene_difference(left, right)
        self.assertEqual(result['valid_cell_count'], 9)
        self.assertEqual(result['mean'], (1+8*50.5)/9)
        self.assertEqual(result['complete_nine']['mean'], 1.)
        self.assertEqual(result['complete_nine']['strata'], ['a'])

    def test_no_paired_support_is_null(self):
        left = {str(i): {'a': 1., 'b': None} for i in range(9)}
        right = {str(i): {'a': None, 'b': 1.} for i in range(9)}
        result = core.paired_scene_difference(left, right)
        self.assertIsNone(result['mean'])
        self.assertEqual(result['valid_cell_count'], 0)
        self.assertIsNone(result['complete_nine']['mean'])

    def test_pairing_rejects_roster_and_non_numeric_metrics(self):
        left = {str(i): {'a': 1.} for i in range(9)}
        with self.assertRaises(ValueError):
            core.paired_scene_difference(left, {})
        for invalid in ('1', float('nan'), True):
            right = {str(i): {'a': invalid} for i in range(9)}
            with self.assertRaises(ValueError):
                core.paired_scene_difference(left, right)


if __name__ == '__main__':
    unittest.main()
