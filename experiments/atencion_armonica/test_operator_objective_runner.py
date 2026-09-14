"""Operational lifecycle fixtures; no production profile or factor decoding."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from src.atencion_armonica import operator_objective_runner as runner
from src.atencion_armonica.operator_objective_artifacts import DiagnosticStore
from src.atencion_armonica import operator_objective_artifacts as artifacts
from src.atencion_armonica.operator_objective_budget import LIMITS
from experiments.atencion_armonica.test_operator_objective_corpus import extracted_fixture

FIXTURES = Path(__file__).resolve().parents[2]/(runner.WORK+"/runner-tests")


class CheckOnly:
    charged_before = 0.
    def __init__(self):
        self.started = time.monotonic()
    def check(self, **kwargs):
        return None


class FixtureCorpus:
    extracted_ids = []
    def __init__(self, reader):
        self.receipts = None
    def inventory(self):
        return {"completion": runner.COMPLETION, "scene_count": 8, "decoded_bytes": 8000,
                "records": {s: [{"scene_id": i} for i in (0, 1)] for s in runner.TESTS}}
    def load_split(self, split):
        return {"split": split}
    def extract_scene(self, loaded, scene_id):
        split = loaded["split"]
        self.extracted_ids.append((split, scene_id))
        extracted, _ = extracted_fixture()
        compact = extracted["compact"]
        gen = compact["log_channel"].astype(np.float32)
        from src.atencion_armonica.generative_evidence import decouple
        dec, sham = decouple(compact["partitions"], gen, split_seed=3, scene_id=0)
        extracted.update({"split": split, "scene_id": scene_id,
                          "provenance": {"artifact": {"decoded_bytes": 1000}},
                          "normalizer": {"mean": [0.]*6, "scale": [1.]*6},
                          "channels": {str(cp): {"local": np.zeros_like(gen), "generative": gen.copy(),
                                                 "decoupled": dec.copy(), "sham": deepcopy(sham)}
                                       for cp in (2026090721, 2026090722, 2026090723)}})
        return extracted


class RunnerTests(unittest.TestCase):
    def setUp(self):
        FIXTURES.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="case-", dir=FIXTURES)
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.store = DiagnosticStore(self.project/runner.OUTPUT, project_root=self.project)
        self.store.initialize({"status": "PREPARED", "limits": LIMITS})
        self.run = runner.DiagnosticRunner(self.store, CheckOnly())
        FixtureCorpus.extracted_ids = []

    def small(self):
        patches = [patch.object(runner, "SCENE_IDS", (0, 1)),
                   patch.object(runner, "ClosedCorpus", FixtureCorpus),
                   patch.object(runner, "forecast", return_value={"time_fits": True, "outputs_fit": True}),
                   patch("builtins.print")]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def commit(self, operation):
        if self.run.pending is not None:
            self.run.stage_completion(f"fixture-{operation}")
            terminal = {"status": "COMPLETE", "start": {"manifest": self.run.manifest_ref, "operation": operation},
                        "completion": self.run.pending_reference()}
            ref = self.store.publish_json(f"fixture-{operation}-finish.json", terminal)
            self.run.publish_completion(ref)

    def test_complete_cycle_reuses_profiles_and_replays_exact_scientific_bytes(self):
        self.small()
        self.assertEqual(self.run.prepare()["scene_count"], 8)
        self.assertEqual(self.run.profile()["status"], "PROFILE_COMPLETE")
        self.assertEqual(len(FixtureCorpus.extracted_ids), 4)
        self.assertEqual(self.run.run(), {"status": "ROSTER_READY_FOR_COMMIT", "scene_count": 8})
        self.assertFalse(self.store.path("complete.json").exists())
        self.commit("run")
        self.assertEqual(len(FixtureCorpus.extracted_ids), 8)  # Four profiles reused.
        before = {p.relative_to(self.store.root): p.read_bytes() for p in self.store.root.rglob("*") if p.is_file()}
        self.assertEqual(self.run.replay(), {"status": "REPLAY_READY_FOR_COMMIT", "scene_count": 8})
        self.assertFalse(self.store.path("replayed.json").exists())
        self.commit("replay")
        self.assertEqual(len(FixtureCorpus.extracted_ids), 8)  # Replay never extracts source.
        for path, raw in before.items():
            self.assertEqual((self.store.root/path).read_bytes(), raw)
        self.assertEqual(self.run.run()["status"], "COMPLETE_REUSED")

    def test_prepare_does_not_adopt_orphaned_inventory(self):
        self.small()
        self.store.publish_json("inventory/inventory.json", {"orphan": True})
        with self.assertRaisesRegex(ValueError, "orphaned"):
            self.run.prepare()
        self.assertFalse(self.store.path("inventory/complete.json").exists())

    def test_partial_profile_resume_reuses_only_complete_scene(self):
        self.small()
        self.run.prepare()
        corpus = FixtureCorpus(None)
        first = next(iter(runner.TESTS))
        payload, record = self.run.unit(corpus, corpus.load_split(first), 0, phase="profile")
        self.assertEqual(len(FixtureCorpus.extracted_ids), 1)
        self.run.profile()
        self.assertEqual(len(FixtureCorpus.extracted_ids), 4)
        self.assertEqual(self.run.load_unit(first, 0)[1], record)

    def test_orphaned_scene_cannot_be_adopted_and_corruption_is_not_repaired(self):
        self.small()
        split = next(iter(runner.TESTS))
        self.store.publish(f"scenes/{split}/00000/bundle.gz", b"orphan")
        with self.assertRaisesRegex(ValueError, "orphaned"):
            self.run.unit(FixtureCorpus(None), {"split": split}, 0, phase="profile")
        self.assertFalse(FixtureCorpus.extracted_ids)
        self.run.unit(FixtureCorpus(None), {"split": split}, 1, phase="run")
        path = self.store.path(f"scenes/{split}/00001/bundle.gz")
        path.write_bytes(b"changed owned fixture")
        with self.assertRaises(ValueError):
            self.run.load_unit(split, 1)
        self.assertEqual(path.read_bytes(), b"changed owned fixture")

    def test_forecast_rejection_blocks_run_before_any_extraction(self):
        self.small()
        self.run.prepare()
        self.run.profile()
        count = len(FixtureCorpus.extracted_ids)
        with patch.object(runner, "forecast", return_value={"time_fits": False, "outputs_fit": True}):
            with self.assertRaises(runner.Paused):
                self.run.run()
        self.assertEqual(len(FixtureCorpus.extracted_ids), count)
        self.assertFalse(self.store.path("complete.json").exists())

    def test_replay_needs_complete_record_not_merely_some_bundles(self):
        self.small()
        self.run.prepare()
        self.run.profile()
        with self.assertRaises(FileNotFoundError):
            self.run.replay()
        self.assertFalse(self.store.path("replayed.json").exists())

    def test_scene_interruption_before_and_after_commit_has_no_half_published_unit(self):
        self.small()
        split = next(iter(runner.TESTS))
        original = artifacts.StagedPublication.publish
        def interrupt_after_staged_bundle(publication, relative, raw):
            ref = original(publication, relative, raw)
            if relative == "bundle.gz":
                raise runner.Paused("fixture after staged bundle")
            return ref
        with patch.object(artifacts.StagedPublication, "publish", interrupt_after_staged_bundle):
            with self.assertRaises(runner.Paused):
                self.run.unit(FixtureCorpus(None), {"split": split}, 0, phase="profile")
        self.assertFalse(self.store.path(f"scenes/{split}/00000").exists())
        self.assertTrue(list(self.store.path(f"scenes/{split}").glob("*.partial")))
        original_commit = artifacts._commit_directory
        def interrupt_after_commit(staging, target):
            original_commit(staging, target)
            raise runner.Paused("fixture after whole unit commit")
        with patch.object(artifacts, "_commit_directory", interrupt_after_commit):
            with self.assertRaises(runner.Paused):
                self.run.unit(FixtureCorpus(None), {"split": split}, 0, phase="profile")
        extracted_count = len(FixtureCorpus.extracted_ids)
        self.run.unit(FixtureCorpus(None), {"split": split}, 0, phase="profile")
        self.assertEqual(len(FixtureCorpus.extracted_ids), extracted_count)

    def test_replay_rejects_changed_terminal_status_and_failed_finish(self):
        self.small()
        self.run.prepare()
        self.run.profile()
        self.run.run()
        # A computed roster is not globally complete before successful finish.
        with self.assertRaises(FileNotFoundError):
            self.run.replay()
        bad = self.store.publish_json("fixture-failed-finish.json", {"status": "PAUSED"})
        with self.assertRaisesRegex(ValueError, "failed operator"):
            self.run.publish_completion(bad)
        self.commit("run")
        path = self.store.path("complete.json")
        complete = runner.json.loads(path.read_bytes())
        path.write_bytes(runner.encoded({**complete, "status": "FAILED"}))
        with self.assertRaises(ValueError):
            self.run.replay()
        self.assertFalse(self.store.path("replayed.json").exists())

    def test_successful_durable_candidate_recovers_without_new_operator_or_budget(self):
        self.small()
        self.run.prepare()
        self.run.profile()
        self.run.run()
        # Simulate complete consumption of non-audit allocation just before
        # the finish/marker gap. Recovery must not launch another operator.
        start = {"manifest": self.run.manifest_ref, "operation": "run",
                 "charged_before": 0., "allocated_seconds": 1200.}
        self.store.publish_json("attempts/0000/start.json", start)
        self.run.stage_completion("attempts/0000")
        terminal = {"status": "COMPLETE", "start": start, "seconds": 1200.,
                    "completion": self.run.pending_reference()}
        self.store.publish_json("attempts/0000/finish.json", terminal)
        self.assertFalse(self.store.path("complete.json").exists())
        with patch.object(runner, "AttemptBudget", side_effect=AssertionError("no new operator")) as constructor:
            # Keep the static parser real; constructing an operator is forbidden.
            from src.atencion_armonica.operator_objective_budget import AttemptBudget
            constructor._local_json = AttemptBudget._local_json
            recovered = runner.recover_completion(self.store, "run")
        self.assertEqual(recovered["status"], "COMPLETE_RECOVERED")
        self.assertFalse(self.store.path("attempts/0001").exists())
        self.assertEqual(self.run.record("complete.json")["status"], "COMPLETE")

    def test_paused_candidate_never_receives_global_authority(self):
        self.small()
        start = {"manifest": self.run.manifest_ref, "operation": "run"}
        self.store.publish_json("attempts/0000/start.json", start)
        self.store.publish_json("attempts/0000/completion_candidate.json", {"status": "COMPLETE"})
        self.store.publish_json("attempts/0000/finish.json", {"status": "PAUSED", "start": start})
        self.assertIsNone(runner.recover_completion(self.store, "run"))
        self.assertFalse(self.store.path("complete.json").exists())

    def test_lock_rejects_concurrent_handle_and_releases_on_exception(self):
        with runner.exclusive_lock(self.project):
            with self.assertRaises(BlockingIOError):
                with runner.exclusive_lock(self.project):
                    pass
        with self.assertRaises(RuntimeError):
            with runner.exclusive_lock(self.project):
                raise RuntimeError("fixture interruption")
        with runner.exclusive_lock(self.project):
            pass

    def test_production_entrypoint_rejects_reduced_fixture_roster(self):
        with patch.object(runner, "SCENE_IDS", (0, 1)):
            with self.assertRaisesRegex(ValueError, "reduced fixture"):
                runner.execute("prepare", project=self.project)

    def test_review_gate_binds_snapshot_and_authenticates_independent_reports(self):
        report = b"Independent operational fixture report.\n"
        (self.project/"report.md").write_bytes(report)
        report_ref = {"path": "report.md", "sha256": hashlib.sha256(report).hexdigest()}
        snapshot = {"files": {"fixture.py": "a"*64}, "python": "fixture", "numpy": "fixture"}
        review = {"status": "IMPLEMENTATION_REVIEW_COMPLETE", "snapshot": snapshot,
                  "reports": [report_ref]}
        def ref_for(value):
            raw = runner.encoded(value)
            (self.project/"review.json").write_bytes(raw)
            return {"path": "review.json", "sha256": hashlib.sha256(raw).hexdigest()}
        with patch.object(runner, "source_snapshot", return_value=snapshot):
            self.assertEqual(runner.reviewed_manifest(self.project, ref_for(review))["snapshot"], snapshot)
            pending = {**review, "status": "PENDING"}
            with self.assertRaisesRegex(ValueError, "not complete"):
                runner.reviewed_manifest(self.project, ref_for(pending))
            with self.assertRaisesRegex(ValueError, "no independent report"):
                runner.reviewed_manifest(self.project, ref_for({**review, "reports": []}))
            stale = {**review, "snapshot": {**snapshot, "files": {"fixture.py": "b"*64}}}
            with self.assertRaisesRegex(ValueError, "current code/runtime"):
                runner.reviewed_manifest(self.project, ref_for(stale))
            (self.project/"report.md").write_bytes(b"corrupted owned fixture")
            with self.assertRaises(ValueError):
                runner.reviewed_manifest(self.project, ref_for(review))

    def test_execute_pause_records_terminal_and_restores_lock_and_handlers(self):
        manifest, _ = self.store.manifest()
        # The fixture manifest gets a review identity before the operation; no
        # production source or review is edited or claimed audited by this test.
        manifest["review"] = {"path": "fixture.json", "sha256": "0"*64}
        self.store.path("manifest.json").write_bytes(runner.encoded(manifest))
        import signal
        before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        with patch.object(runner, "reviewed_manifest", return_value=manifest), \
             patch.object(runner.DiagnosticRunner, "prepare", side_effect=runner.Paused("fixture")), \
             patch("builtins.print"):
            with self.assertRaises(runner.Paused):
                runner.execute("prepare", project=self.project)
        terminal = runner.AttemptBudget._local_json(self.store.path("attempts/0000/finish.json"))
        self.assertEqual(terminal["status"], "PAUSED")
        self.assertEqual(before, {s: signal.getsignal(s) for s in before})
        with runner.exclusive_lock(self.project):
            pass


if __name__ == "__main__":
    unittest.main()
