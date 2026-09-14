"""Byte-level differential tests on mathematical fixtures, not campaign scenes."""
from copy import deepcopy
import itertools
import hashlib
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.atencion_armonica import operator_diagnostic_cached as cached
from src.atencion_armonica import operator_objective_core as core
from src.atencion_armonica import operator_objective_scene as reference
from src.atencion_armonica.operator_objective_artifacts import bundle_bytes
from experiments.atencion_armonica.test_operator_objective_scene import example
from experiments.atencion_armonica.test_operator_objective_corpus import extracted_fixture
from experiments.atencion_armonica import profile_cached_diagnostic as profile


class CachedTests(unittest.TestCase):
    def equivalent(self, arguments):
        before = bundle_bytes(arguments)[0]
        expected = bundle_bytes(reference.diagnose_scene(*arguments))[0]
        actual = bundle_bytes(cached.diagnose_scene(*arguments))[0]
        self.assertEqual(actual, expected)
        self.assertEqual(bundle_bytes(arguments)[0], before)

    def test_empty_singleton_and_identical_schemes(self):
        for partitions in ([], [((0, 1, 2, 3), (4, 5, 6, 7))], None):
            with self.subTest(partitions=partitions):
                self.equivalent(example(partitions))

    def test_ties_constant_cells_float32_neighbors_and_distinct_predictions(self):
        arguments = example()
        compact, labels, predictions = arguments
        for ai, arm in enumerate(reference.ARMS):
            for ci, cell in enumerate(reference.CELLS):
                value = np.float32((ai*9+ci)/32)
                predictions[arm][cell] = np.array([[value, value],
                    [np.nextafter(value, np.float32(1)), np.float32(value+.125)]], np.float32)
        predictions["generative"][reference.CELLS[0]][:] = .125
        predictions["local"][reference.CELLS[1]][:] = 0.
        for score in compact["scores"].values():
            score[:] = 1.
        self.equivalent(arguments)

    def test_different_k_sizes_and_branch_strata(self):
        partitions = sorted([((0, 1, 2, 3), (4, 5, 6, 7), tuple(range(8, 16))),
                             (tuple(range(8)), tuple(range(8, 16))),
                             tuple(tuple(range(i, i+4)) for i in range(0, 16, 4))])
        arguments = example(partitions, 16)
        arguments[0]["winning_branches"]["extended_ub"] = ["base-low", "base-high", "deformed-low"]
        self.equivalent(arguments)

    def test_complete_two_group_fixture_roster_and_deterministic_predictions(self):
        # All 35 canonical 4+4 partitions of eight indices; no scene sampler/RNG.
        all_events = set(range(8))
        partitions = sorted((tuple(group), tuple(sorted(all_events-set(group))))
                            for tail in itertools.combinations(range(1, 8), 3)
                            for group in [(0, *tail)])
        arguments = example(partitions)
        count = len(partitions)
        for ai, arm in enumerate(reference.ARMS):
            for ci, cell in enumerate(reference.CELLS):
                values = np.arange(count*2, dtype=np.int64).reshape(count, 2)
                arguments[2][arm][cell] = ((values*(ci+1)+ai*13) % 37).astype(np.float32)/np.float32(37)
        self.equivalent(arguments)

    def test_oracle_cache_is_per_candidate_tuple_and_does_not_leak_between_calls(self):
        arguments = example()
        with patch.object(core, "oracle_report", wraps=core.oracle_report) as oracle:
            cached.diagnose_scene(*arguments)
            self.assertEqual(oracle.call_count, 1)  # Four schemes describe the same two indices.
            arguments[1][:] = 0
            cached.diagnose_scene(*arguments)
            self.assertEqual(oracle.call_count, 2)  # New scene call cannot reuse the previous truth.
        self.equivalent(arguments)

    def test_invalid_roster_dtype_negative_and_nonfinite_remain_rejected(self):
        mutations = [lambda a: a[2]["local"].pop(reference.CELLS[0]),
                     lambda a: a[2]["local"].__setitem__(reference.CELLS[0], np.zeros((2, 2), np.float64)),
                     lambda a: a[2]["local"][reference.CELLS[0]].fill(-1),
                     lambda a: a[0]["scores"]["base_ub"].fill(np.nan)]
        for mutate in mutations:
            arguments = example()
            mutate(arguments)
            for implementation in (reference.diagnose_scene, cached.diagnose_scene):
                with self.assertRaises(ValueError):
                    implementation(*deepcopy(arguments))


class ProfileOperatorTests(unittest.TestCase):
    def setUp(self):
        root = profile.ROOT/".agent-work/phideus-operator-objective-20260914/cache-profile-tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="case-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.store = profile.base.DiagnosticStore(self.project/profile.base.OUTPUT, project_root=self.project)
        self.store.initialize({"status": "PREPARED", "limits": profile.base.LIMITS})

    def fixture_ports(self):
        extracted, result = extracted_fixture()
        payload = {"extracted": extracted, "result": result,
                   "coverage": profile.base.verify_scene_result(extracted, result)}
        raw = bundle_bytes(payload)[0]
        baseline = {"status": "PROFILE_COMPLETE", "profiles": [
            {"split": split, "scene_id": 0, "candidate_count": 2, "decoded_bytes": 1000,
             "bundle_bytes": len(raw), "extraction_seconds": .001, "diagnostic_seconds": .1,
             "setup_seconds": .001} for split in profile.base.TESTS]}
        (self.project/"fixture-report.md").write_bytes(b"Independent fixture report.\n")
        review = {"status": "CACHE_PROFILE_REVIEW_COMPLETE", "snapshot": {"fixture": "fixed"},
                  "reports": [profile.reference(self.project, "fixture-report.md")]}
        (self.project/"fixture-review.json").write_bytes(profile.base.encoded(review))
        self.review_ref = profile.reference(self.project, "fixture-review.json")
        patches = [patch.object(profile, "accept_review", return_value=review),
                   patch.object(profile.base.DiagnosticRunner, "record", return_value=baseline),
                   patch.object(profile.base.DiagnosticRunner, "inventory", return_value={"scene_count": 2048, "decoded_bytes": 2048000}),
                   patch.object(profile.base.DiagnosticRunner, "load_unit", return_value=(payload, {"scene_id": 0}, raw)),
                   patch("builtins.print")]
        for replacement in patches:
            replacement.start()
            self.addCleanup(replacement.stop)
        return raw

    def test_fixed_four_profile_seals_revision_and_report_without_roster_authority(self):
        raw = self.fixture_ports()
        report = profile.execute(self.review_ref, project=self.project)
        self.assertEqual(report["status"], "CACHE_PROFILE_CANDIDATE")
        self.assertEqual([r["split"] for r in report["profiles"]], list(profile.base.TESTS))
        for row in report["profiles"]:
            self.assertTrue(row["scientific_bytes_equal"])
            self.assertEqual(self.store.read({k: row["bundle"][k] for k in ("path", "sha256", "bytes")}), raw)
        finish = self.store.json(profile.reference(self.store.root, "attempts/0000/finish.json"))
        self.assertEqual(finish["status"], "COMPLETE")
        self.assertEqual(self.store.json(finish["completion"]["report"]), report)
        revision = self.store.json(finish["completion"]["runtime_revision"])
        self.assertEqual(self.store.json(revision["start"]), finish["start"])
        self.assertFalse(self.store.path("complete.json").exists())
        self.assertFalse(self.store.path("replayed.json").exists())
        # Another review path with the same snapshot cannot trigger fresh timing.
        alias = self.project/"review-alias.json"
        alias.write_bytes((self.project/"fixture-review.json").read_bytes())
        with patch.object(profile.base, "AttemptBudget", side_effect=AssertionError("no new profile attempt")):
            repeated = profile.execute(profile.reference(self.project, "review-alias.json"), project=self.project)
        self.assertEqual(repeated, report)
        self.assertFalse(self.store.path("attempts/0001").exists())

    def test_paused_profile_keeps_charged_terminal_and_restores_handlers(self):
        self.fixture_ports()
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
        with patch.object(profile.cached, "diagnose_scene", side_effect=profile.base.Paused("fixture")):
            with self.assertRaises(profile.base.Paused):
                profile.execute(self.review_ref, project=self.project)
        finish = self.store.json(profile.reference(self.store.root, "attempts/0000/finish.json"))
        self.assertEqual(finish["status"], "PAUSED")
        self.assertGreater(finish["charged_total"], 0.)
        self.assertNotIn("completion", finish)
        self.assertEqual(previous, {sig: signal.getsignal(sig) for sig in previous})
        self.assertFalse(self.store.path("attempts/0000/cache_profile.json").exists())

    def test_profile_review_rejects_missing_reports_and_changed_snapshot(self):
        report = self.project/"report.md"
        report.write_bytes(b"Independent fixture report.\n")
        fixed = {"manifest": "fixture", "files": ["five-file-fixture"]}
        review = {"status": "CACHE_PROFILE_REVIEW_COMPLETE", "snapshot": fixed,
                  "reports": [profile.reference(self.project, "report.md")]}
        def review_ref(value):
            raw = profile.base.encoded(value)
            (self.project/"review.json").write_bytes(raw)
            return {"path": "review.json", "sha256": hashlib.sha256(raw).hexdigest()}
        with patch.object(profile, "snapshot", return_value=fixed):
            self.assertEqual(profile.accept_review(self.store, review_ref(review)), review)
            with self.assertRaises(ValueError):
                profile.accept_review(self.store, review_ref({**review, "reports": []}))
            with self.assertRaises(ValueError):
                profile.accept_review(self.store, review_ref({**review, "snapshot": {"changed": True}}))
            report.write_bytes(b"corrupted owned fixture")
            with self.assertRaises(ValueError):
                profile.accept_review(self.store, review_ref(review))


if __name__ == "__main__":
    unittest.main()
