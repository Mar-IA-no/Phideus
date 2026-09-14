"""Fixture-only lifecycle tests for the separate fast timing operator."""
import hashlib
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from experiments.atencion_armonica import profile_fast_diagnostic as profile
from experiments.atencion_armonica.test_operator_objective_corpus import extracted_fixture
from src.atencion_armonica.operator_objective_artifacts import bundle_bytes

class ProfileOperatorTests(unittest.TestCase):
    def setUp(self):
        root = profile.ROOT/".agent-work/phideus-operator-objective-20260914/fast-profile-tests"
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
        review = {"status": "FAST_PROFILE_REVIEW_COMPLETE", "snapshot": {"fixture": "fixed"},
                  "reports": [profile.reference(self.project, "fixture-report.md")]}
        (self.project/"fixture-review.json").write_bytes(profile.base.encoded(review))
        self.review_ref = profile.reference(self.project, "fixture-review.json")
        patches = [patch.object(profile, "accept_review", return_value=review),
                   patch.object(profile.base.DiagnosticRunner, "record", return_value=baseline),
                   patch.object(profile.base.DiagnosticRunner, "inventory", return_value={"scene_count": 2048, "decoded_bytes": 2048000}),
                   patch.object(profile.base.DiagnosticRunner, "load_unit", return_value=(payload, {"scene_id": 0, "bundle": bundle_bytes(payload)[1]}, raw)),
                   patch("builtins.print")]
        for replacement in patches:
            replacement.start()
            self.addCleanup(replacement.stop)
        return raw

    def test_fixed_four_profile_seals_revision_and_report_without_roster_authority(self):
        raw = self.fixture_ports()
        report = profile.execute(self.review_ref, project=self.project)
        self.assertEqual(report["status"], "FAST_PROFILE_CANDIDATE")
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
        with patch.object(profile.fast, "diagnose_scene", side_effect=profile.base.Paused("fixture")):
            with self.assertRaises(profile.base.Paused):
                profile.execute(self.review_ref, project=self.project)
        finish = self.store.json(profile.reference(self.store.root, "attempts/0000/finish.json"))
        self.assertEqual(finish["status"], "PAUSED")
        self.assertGreater(finish["charged_total"], 0.)
        self.assertNotIn("completion", finish)
        self.assertEqual(previous, {sig: signal.getsignal(sig) for sig in previous})
        self.assertFalse(self.store.path("attempts/0000/fast_profile.json").exists())

    def test_profile_review_rejects_missing_reports_and_changed_snapshot(self):
        report = self.project/"report.md"
        report.write_bytes(b"Independent fixture report.\n")
        fixed = {"manifest": "fixture", "files": ["five-file-fixture"]}
        review = {"status": "FAST_PROFILE_REVIEW_COMPLETE", "snapshot": fixed,
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

    def test_codec_receipt_mismatch_is_failed_even_with_identical_bytes(self):
        raw = self.fixture_ports()
        extracted, result = extracted_fixture()
        payload = {"extracted": extracted, "result": result,
                   "coverage": profile.base.verify_scene_result(extracted, result)}
        codec = {**bundle_bytes(payload)[1], "decoded_sha256": "0"*64}
        with patch.object(profile.fast, "bundle_bytes", return_value=(raw, codec)):
            with self.assertRaisesRegex(ValueError, "codec receipt differs"):
                profile.execute(self.review_ref, project=self.project)
        finish = self.store.json(profile.reference(self.store.root, "attempts/0000/finish.json"))
        self.assertEqual(finish["status"], "FAILED")
        self.assertNotIn("completion", finish)
        self.assertFalse(self.store.path("attempts/0000/fast_profile.json").exists())


if __name__ == "__main__":
    unittest.main()
