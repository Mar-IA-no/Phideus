import copy
import unittest

import numpy as np

from src.atencion_armonica.partial_compatibility import triple_geometry
from src.atencion_armonica.structured_source_reader import FACTORS, GAMMAS, score_scene
from src.atencion_armonica.structured_source_metrics import (
    METRICS, READERS, SEEDS, SPLIT_SEEDS, bootstrap_indices, calibration_grid, contrast_flags,
    evaluate_scene, partition_metrics, read_selected, select_gammas, summarize_test)


def calibration_fixture():
    rows = []
    for scene in range(256):
        for seed in SEEDS:
            rows.append({"scene_id": scene, "seed": seed,
                         "split_role": "calibration", "split_seed": SPLIT_SEEDS["calibration"],
                         "ari_grid": {factor: [0., .5, .5, .4, .3, .1] for factor in FACTORS[1:]}})
    return rows


def summary_fixture(split="ood_polyphony"):
    rows = []
    for scene in range(256):
        for j, seed in enumerate(SEEDS):
            readers = {}
            for reader in READERS:
                gain = .1 if reader == "shared_source" and scene < 128 else 0
                readers[reader] = {"metrics": {"ari": .1*j+gain, "exact_partition": False,
                    "pair_disagreement": .5-gain, "k_inferred": 4+(reader == "shared_source"),
                    "k_absolute_error": int(reader == "shared_source"),
                    "sub3_member_fraction": .25 if reader == "shared_source" else 0.},
                    "group_statistics": {"4": {"group_count": 1, "rms_cents_mean": 2.,
                        "cost_means": {factor: .5 for factor in FACTORS[1:]}}}}
            rows.append({"scene_id": scene, "seed": seed, "readers": readers,
                         "split_role": split, "split_seed": SPLIT_SEEDS[split],
                         "pool_oracle_ari": .9, "neural_brier": .2,
                         "sham_support_status": "RANKING_CONTRAST_AVAILABLE"})
    return rows


class MetricsTests(unittest.TestCase):
    def test_partition_metrics_permutation_and_rejections(self):
        p = ((0, 1), (2, 3))
        m = partition_metrics(p, [8, 8, 2, 2])
        self.assertEqual(m["ari"], 1.)
        self.assertTrue(m["exact_partition"])
        self.assertEqual(m["pair_disagreement"], 0.)
        self.assertEqual(m["sub3_member_fraction"], 1.)
        self.assertEqual(m, partition_metrics(((2, 3), (0, 1)), [3, 3, 9, 9]))
        for bad in (((0, 1), (2,)), ((0, 1), (1, 2, 3)), ((), (0, 1, 2, 3))):
            with self.assertRaises(ValueError):
                partition_metrics(bad, [0, 0, 1, 1])
        with self.assertRaises(ValueError):
            partition_metrics(p, [0., 0., 1., 1.])

    def test_evaluate_pool_oracle_and_unchanged_logits(self):
        q = np.array([0., .125, .25, .375], dtype=np.float32)
        z = np.zeros((4, 4), dtype=np.float32)
        g = triple_geometry(q)
        scored = score_scene(q, z, g["pair_support"], g["triples"], g["residual_cents"], split_seed=123, scene_id=0)
        evaluated = evaluate_scene(scored, [0, 0, 1, 1], z, .55)
        self.assertEqual(evaluated["neural_brier"], .25)
        self.assertEqual(evaluated["pool_oracle_ari"], max(m["ari"] for m in evaluated["candidate_metrics"]))
        selected = read_selected(scored, evaluated, {factor: 0. for factor in FACTORS[1:]})
        for factor in FACTORS:
            self.assertEqual(selected["readers"][factor]["metrics"], selected["readers"]["pairs"]["metrics"])
        obs = {"scene_id": 0, "split_seed": SPLIT_SEEDS["calibration"], "log_f": q.tolist()}
        grid = calibration_grid(scored, evaluated, observation=obs, seed=SEEDS[0])
        self.assertEqual(set(grid["ari_grid"]), set(FACTORS[1:]))
        self.assertTrue(all(len(v) == len(GAMMAS) for v in grid["ari_grid"].values()))
        np.testing.assert_array_equal(z, np.zeros((4, 4)))
        with self.assertRaises(PermissionError):
            calibration_grid(scored, evaluated, observation={**obs, "split_seed": SPLIT_SEEDS["ood_beta"]}, seed=SEEDS[0])
        with self.assertRaises(ValueError):
            calibration_grid(scored, evaluated, observation={**obs, "log_f": (q+.125).tolist()}, seed=SEEDS[0])

    def test_selection_exact_roster_calibration_only_and_lowest_tie(self):
        rows = calibration_fixture()
        result = select_gammas(rows[::-1], split="calibration")
        for factor in FACTORS[1:]:
            self.assertEqual(result["factors"][factor]["gamma"], .01)
        with self.assertRaises(PermissionError):
            select_gammas(rows, split="ood_polyphony")
        with self.assertRaises(ValueError):
            select_gammas(rows[:-1], split="calibration")
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaises(ValueError):
            select_gammas(rows, split="calibration")

    def test_test_rows_cannot_be_relabelled_as_calibration(self):
        rows = calibration_fixture()
        for row in rows:
            row.update(split_role="ood_beta", split_seed=SPLIT_SEEDS["ood_beta"])
        with self.assertRaises(PermissionError):
            select_gammas(rows, split="calibration")
        for row in rows:
            row["split_role"] = "calibration"
        with self.assertRaises(PermissionError):
            select_gammas(rows, split="calibration")

    def test_paired_scene_bootstrap_nominal_family_and_no_substitution(self):
        rows = summary_fixture()
        indices = bootstrap_indices()
        result = summarize_test(rows, split="ood_polyphony", indices=indices)
        self.assertEqual(result["sham_support"]["eligible_scene_seed"], 768)
        expected_delta = np.r_[np.full(128, .1), np.zeros(128)]
        expected_interval = np.percentile(expected_delta[indices].mean(axis=1), [.05/3/2*100, (1-.05/3/2)*100])
        for control in ("pairs", "decoupled_source", "local_compatibility"):
            c = result["comparisons"][control]
            self.assertAlmostEqual(c["mean_deltas"]["ari"], .05)
            np.testing.assert_allclose(c["intervals"]["ari"]["bounds"], expected_interval, atol=1e-15)
            self.assertEqual(c["intervals"]["ari"]["status"], "PRIMARY_BONFERRONI_NOMINAL")
            self.assertIn("FRAGMENTATION_ATTRIBUTION_UNRESOLVED", c["flags"])
        self.assertEqual(result["comparisons"]["historical"]["intervals"]["ari"]["nominal_coverage"], .95)
        self.assertEqual(result["comparisons"]["pairs"]["intervals"]["k_inferred"]["nominal_coverage"], .95)
        iid = summarize_test(summary_fixture("iid"), split="iid", indices=indices)
        self.assertEqual(iid["comparisons"]["pairs"]["intervals"]["ari"]["nominal_coverage"], .95)
        indices[0, 0] = (indices[0, 0]+1) % 256
        with self.assertRaises(ValueError):
            summarize_test(rows, split="ood_polyphony", indices=indices)

    def test_missing_sham_support_does_not_filter_scene_estimand(self):
        rows = summary_fixture("ood_beta")
        indices = bootstrap_indices()
        before = summarize_test(rows, split="ood_beta", indices=indices)
        rows[0]["sham_support_status"] = "NOT_EVALUABLE_SHAM_SUPPORT"
        after = summarize_test(rows, split="ood_beta", indices=indices)
        self.assertEqual(after["comparisons"], before["comparisons"])
        self.assertEqual(after["sham_support"]["status"], "PARTIAL_SHAM_SUPPORT")
        self.assertEqual(after["sham_support"]["three_seed_eligible_scenes"], 255)
        for row in rows:
            row["sham_support_status"] = "NOT_EVALUABLE_SHAM_SUPPORT"
        absent = summarize_test(rows, split="ood_beta", indices=indices)
        self.assertEqual(absent["comparisons"], before["comparisons"])
        self.assertEqual(absent["sham_support"]["status"], "NOT_EVALUABLE_SHAM_SUPPORT")

    def test_fragmentation_flags_are_descriptive_signs(self):
        means = dict.fromkeys(METRICS, 0.)
        self.assertEqual(contrast_flags(means), [])
        means.update(ari=.1, k_inferred=1.)
        self.assertEqual(contrast_flags(means), ["GAIN_WITH_MORE_GROUPS"])
        means["sub3_member_fraction"] = .1
        self.assertEqual(len(contrast_flags(means)), 2)
        means["ari"] = -.1
        self.assertEqual(contrast_flags(means), [])

    def test_group_rollup_requires_three_seeds_and_keeps_absent_counts(self):
        rows = summary_fixture("iid")
        rows[0]["readers"]["shared_source"]["group_statistics"] = {}
        result = summarize_test(rows, split="iid", indices=bootstrap_indices())
        size4 = result["group_statistics"]["shared_source"]["4"]
        self.assertAlmostEqual(size4["mean_group_count"], 767/768)
        self.assertEqual(size4["conditional_metrics"]["rms_cents_mean"]["three_seed_eligible_scenes"], 255)
        self.assertEqual(size4["conditional_metrics"]["rms_cents_mean"]["mean"], 2.)
        self.assertIsNone(result["group_statistics"]["shared_source"]["3"]["conditional_metrics"]["rms_cents_mean"]["mean"])


if __name__ == "__main__":
    unittest.main()
