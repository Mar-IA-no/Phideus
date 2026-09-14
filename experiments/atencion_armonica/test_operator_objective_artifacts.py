"""Own-root publication, lossless replay codec and cumulative budget fixtures."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.atencion_armonica import operator_objective_artifacts as artifacts
from src.atencion_armonica import operator_objective_budget as budget

FIXTURES = Path(__file__).resolve().parents[2]/".agent-work/phideus-operator-objective-20260914/artifact-tests"


class CodecTests(unittest.TestCase):
    def test_lossless_float32_bits_and_deterministic_bytes(self):
        x = np.array([.5, np.nextafter(np.float32(.5), np.float32(1)), -0.], np.float32)
        value = {"x": x, "mask": np.array([True, False]), "nested": [("a", np.zeros((0, 2), np.float64))]}
        raw, receipt = artifacts.bundle_bytes(value)
        other, _ = artifacts.bundle_bytes({"nested": value["nested"], "mask": value["mask"], "x": x})
        self.assertEqual(raw, other)
        decoded = artifacts.load_bundle(raw, receipt)
        self.assertEqual(decoded["x"].dtype, np.float32)
        self.assertEqual(decoded["x"].tobytes(), x.tobytes())
        self.assertEqual(decoded["nested"][0][1].shape, (0, 2))
        self.assertEqual(artifacts.bundle_bytes(decoded)[0], raw)

    def test_codec_rejects_corruption_limits_reserved_tags_and_objects(self):
        raw, receipt = artifacts.bundle_bytes({"a": np.arange(3)})
        for bad in ({**receipt, "sha256": "0"*64}, {**receipt, "decoded_bytes": 1},
                    {**receipt, "decoded_sha256": "0"*64}):
            with self.assertRaises(ValueError):
                artifacts.load_bundle(raw, bad)
        with self.assertRaises(ValueError):
            artifacts.load_bundle(raw, receipt, maximum_decoded_bytes=1)
        for value in ({"__ndarray__": {}}, np.array([{}], object), np.array([np.nan]), {1: 0}):
            with self.assertRaises(ValueError):
                artifacts.bundle_bytes(value)


class StoreBudgetTests(unittest.TestCase):
    def setUp(self):
        FIXTURES.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="case-", dir=FIXTURES)
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.root = self.project/"data/atencion_armonica/operator_objective_alignment_v1"
        self.store = artifacts.DiagnosticStore(self.root, project_root=self.project)
        self.now = 0.
        self.limits = {"seconds": 20., "profile_seconds": 4., "audit_reserve_seconds": 5.,
                       "rss_bytes": 10**12, "new_bytes": 10**7, "free_bytes": 1}
        self.store.initialize({"status": "PREPARED", "source": "mathematical_fixture", "limits": self.limits})

    def attempt(self, operation="run"):
        return budget.AttemptBudget(self.store, operation, limits=self.limits, clock=lambda: self.now)

    def test_immutable_publication_and_no_adoption_or_path_escape(self):
        ref = self.store.publish_json("scene/item.json", {"x": 1})
        self.assertEqual(self.store.json(ref), {"x": 1})
        with self.assertRaises(FileExistsError):
            self.store.publish_json("scene/item.json", {"x": 2})
        self.assertEqual(self.store.json(ref), {"x": 1})
        with self.assertRaises(FileExistsError):
            self.store.initialize({"status": "PREPARED"})
        for path in ("../escape", "/tmp/escape"):
            with self.assertRaises(ValueError):
                self.store.publish_json(path, {})
        (self.root/"alias").symlink_to(self.root/"scene", target_is_directory=True)
        with self.assertRaises(ValueError):
            self.store.publish_json("alias/new.json", {})

    def test_success_and_failure_both_consume_same_cumulative_budget(self):
        first = self.attempt()
        self.now = 3.
        self.assertEqual(first.finish("FAILED")["charged_total"], 3.)
        second = self.attempt("profile")
        self.assertEqual(second.charged_before, 3.)
        self.assertEqual(second.allocation, 4.)
        self.now = 5.
        second.finish("COMPLETE")
        third = self.attempt("profile")
        self.assertEqual(third.profile_before, 2.)
        self.assertEqual(third.allocation, 2.)
        self.now = 6.
        third.finish("PAUSED")
        fourth = self.attempt()
        self.assertEqual(fourth.allocation, 9.)
        fourth.finish("COMPLETE")

    def test_unclean_attempt_consumes_entire_reservation(self):
        first = self.attempt("profile")
        self.assertEqual(first.allocation, 4.)
        # Simulates terminal process with no finish receipt; caller must verify
        # the process lock before resuming. No actual process is launched here.
        second = self.attempt("run")
        self.assertEqual(second.charged_before, 4.)
        second.finish("COMPLETE")
        with self.assertRaises(budget.BudgetExceeded):
            self.attempt("profile")

    def test_exceeded_time_is_not_clamped_or_automatically_retried(self):
        run = self.attempt("profile")
        self.now = 5.
        with self.assertRaises(budget.BudgetExceeded):
            run.check()
        end = run.finish("BUDGET_EXHAUSTED")
        self.assertEqual(end["seconds"], 5.)
        with self.assertRaises(budget.BudgetExceeded):
            self.attempt("run")

    def test_changed_manifest_and_output_envelope_stop_attempt(self):
        run = self.attempt()
        with self.assertRaises(budget.BudgetExceeded):
            run.check(additional_bytes=10**7)
        path = self.root/"manifest.json"
        # Explicitly owned fixture mutation, never a real manifest edit.
        path.write_bytes(artifacts.encoded({"status": "PREPARED", "source": "changed"}))
        with self.assertRaisesRegex(ValueError, "changed"):
            run.check()

    def test_attempt_cannot_silently_expand_manifest_budget(self):
        with self.assertRaisesRegex(ValueError, "limits differ"):
            budget.AttemptBudget(self.store, "run", limits={**self.limits, "seconds": 200.})


class ForecastTests(unittest.TestCase):
    def profile(self):
        return [{"split": split, "scene_id": 0, "candidate_count": 82, "decoded_bytes": 1000,
                 "bundle_bytes": 100, "extraction_seconds": .001, "diagnostic_seconds": .001}
                for split in ("iid", "ood_beta", "ood_polyphony", "deformed_family")]

    def test_forecast_includes_decode_replay_margin_reserve_and_whole_roster(self):
        result = budget.forecast(self.profile(), {"scene_count": 2048, "decoded_bytes": 2048000}, charged_seconds=10.)
        self.assertAlmostEqual(result["projected_extraction_seconds"], 2.048)
        self.assertAlmostEqual(result["projected_diagnostic_seconds"], 2.048)
        self.assertAlmostEqual(result["projected_total_seconds"], 622.288)
        self.assertTrue(result["time_fits"])
        self.assertEqual(result["projected_scene_bundle_bytes"], 409600.)

    def test_no_pairs_or_missing_split_cannot_authorize_roster(self):
        for profile in (self.profile()[:3], [{**r, "candidate_count": 1} for r in self.profile()]):
            with self.assertRaises(ValueError):
                budget.forecast(profile, {"scene_count": 2048, "decoded_bytes": 2048000}, charged_seconds=1.)
        profile = [{**r, "diagnostic_seconds": 10.} for r in self.profile()]
        self.assertFalse(budget.forecast(profile, {"scene_count": 2048, "decoded_bytes": 2048000}, charged_seconds=1.)["time_fits"])

    def test_invalid_profile_costs_or_sizes_fail_before_projection(self):
        for key, bad in (("setup_seconds", float("nan")), ("bundle_bytes", -1), ("decoded_bytes", True)):
            profile = self.profile()
            profile[0][key] = bad
            with self.assertRaises(ValueError):
                budget.forecast(profile, {"scene_count": 2048, "decoded_bytes": 2048000}, charged_seconds=1.)


if __name__ == "__main__":
    unittest.main()
