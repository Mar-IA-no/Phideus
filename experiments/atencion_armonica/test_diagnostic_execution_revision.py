"""Revision fixtures only; no production extraction, timing profile, or CUDA."""
from copy import deepcopy
import math
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from src.atencion_armonica import diagnostic_execution_revision as revision
from src.atencion_armonica import operator_objective_budget as old_budget
from experiments.atencion_armonica.test_operator_objective_runner import CheckOnly, FixtureCorpus
from experiments.atencion_armonica.test_operator_objective_corpus import extracted_fixture

base = revision.base
FIXTURES = base.ROOT/base.WORK/"execution-revision-tests"


class FixtureBase(unittest.TestCase):
    def setUp(self):
        FIXTURES.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="case-", dir=FIXTURES)
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.store = base.DiagnosticStore(self.project/base.OUTPUT, project_root=self.project)
        self.store.initialize({"status": "PREPARED", "limits": old_budget.LIMITS, "review": {"fixture": True}})
        self.now = 0.
        for i, duration in enumerate((1., 2., 3., 4., 5., 8.820174567052163)):
            attempt = old_budget.AttemptBudget(self.store, "prepare" if i == 0 else "profile",
                                               clock=lambda: self.now)
            self.now += duration
            attempt.finish("COMPLETE")
        self.historical_total = self.now

    def make_review(self, prefixes=()):
        self.store.publish_json("fixture-independent-report.json", {"fixture": "review evidence"})
        report = revision.reference(self.project, base.OUTPUT+"/fixture-independent-report.json")
        self.review = {"status": "EXECUTION_REVISION_REVIEW_COMPLETE",
                       "supersedes_operational_limits": old_budget.LIMITS,
                       "effective_limits": revision.EFFECTIVE_LIMITS,
                       "snapshot": {"historical_attempts": [
                           {k: revision.reference(self.store.root, f"attempts/{i:04d}/{k}.json")
                            for k in ("start", "finish")} for i in range(6)],
                           "legacy_prefixes": list(prefixes)}, "reports": [report]}
        self.store.publish_json("fixture-review.json", self.review)
        self.ref = revision.reference(self.project, base.OUTPUT+"/fixture-review.json")

    def attempt(self, operation="run"):
        return revision.RevisedBudget(self.store, operation, self.ref, self.review, clock=lambda: self.now)


class BudgetTests(FixtureBase):
    def setUp(self):
        super().setUp()
        self.make_review()

    def test_carryover_failure_pause_and_effective_reserve(self):
        run = self.attempt()
        self.assertEqual(run.charged_before, 23.820174567052163)
        self.assertEqual(run.start_record["operation"], "revision_run")
        self.assertEqual(run.start_record["execution_revision"], self.ref)
        self.assertEqual(run.allocation, 6600.-self.historical_total)
        self.now += 4.
        run.finish("FAILED")
        replay = self.attempt("replay")
        self.assertEqual(replay.charged_before, self.historical_total+4.)
        self.now += 2.
        replay.finish("PAUSED")
        audit = self.attempt("audit")
        self.assertEqual(audit.allocation, 7200.-self.historical_total-6.)
        audit.finish("COMPLETE")
        self.assertEqual(self.store.manifest()[0]["limits"]["seconds"], 1800.)

    def test_unclean_attempt_charges_entire_remaining_allocation(self):
        self.attempt()  # Simulate process gone after immutable start, under caller's lock.
        with self.assertRaises(base.BudgetExceeded):
            self.attempt()
        audit = self.attempt("audit")
        self.assertEqual(audit.charged_before, 6600.)
        self.assertEqual(audit.allocation, 600.)
        audit.finish("COMPLETE")

    def test_terminal_exhaustion_is_not_retryable(self):
        attempt = self.attempt()
        self.now += attempt.allocation+1.
        with self.assertRaises(base.BudgetExceeded):
            attempt.check()
        attempt.finish("BUDGET_EXHAUSTED")
        with self.assertRaises(base.BudgetExceeded):
            self.attempt("audit")

    def test_history_mutation_or_revision_alias_cannot_reset_time(self):
        run = self.attempt()
        run.finish("COMPLETE")
        alias = self.store.publish_json("fixture-review-alias.json", self.review)
        alias_ref = revision.reference(self.project, base.OUTPUT+"/"+alias["path"])
        with self.assertRaisesRegex(ValueError, "revision differs"):
            revision.RevisedBudget(self.store, "run", alias_ref, self.review, clock=lambda: self.now)
        path = self.store.path("attempts/0000/finish.json")
        path.write_bytes(b"{}\n")  # Owned mutation fixture only.
        with self.assertRaises(ValueError):
            self.attempt()

    def test_review_changed_mid_attempt_and_output_guards(self):
        run = self.attempt()
        with self.assertRaises(base.BudgetExceeded):
            run.check(additional_bytes=4*1024**3)
        self.store.path("fixture-review.json").write_bytes(b"{}\n")
        with self.assertRaises(ValueError):
            run.check()
        run.finish("FAILED")

    def test_initial_resource_failure_is_terminal_and_profile_forbidden(self):
        with self.assertRaises(ValueError):
            self.attempt("profile")
        with patch.object(revision.RevisedBudget, "check", side_effect=base.BudgetExceeded("fixture")):
            with self.assertRaises(base.BudgetExceeded):
                self.attempt()
        end = old_budget.AttemptBudget._local_json(self.store.path("attempts/0006/finish.json"))
        self.assertEqual(end["status"], "BUDGET_EXHAUSTED")

    def test_execute_failed_paused_exhausted_restores_handlers_without_completion(self):
        # Consecutive failed/paused/exhausted attempts; no production ports run.
        for error, expected in ((ValueError("fixture"), "FAILED"), (base.Paused("fixture"), "PAUSED"),
                                (base.BudgetExceeded("fixture"), "BUDGET_EXHAUSTED")):
            before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
            with patch.object(revision, "accept_review", return_value=self.review), \
                    patch.object(revision.RevisedRunner, "run", side_effect=error), patch("builtins.print"):
                with self.assertRaises(type(error)):
                    revision.execute("run", self.ref, project=self.project)
            self.assertEqual(before, {s: signal.getsignal(s) for s in before})
            ends = sorted((self.store.root/"attempts").glob("*/finish.json"))
            end = old_budget.AttemptBudget._local_json(ends[-1])
            self.assertEqual(end["status"], expected)
            self.assertNotIn("completion", end)
            self.assertFalse(self.store.path(revision.MARKERS["run"]).exists())


def fake_forecast_inputs():
    profiles = [{"split": s, "scene_id": 0, "candidate_count": 31,
                 "decoded_bytes": 1000, "bundle_bytes": 100, "extraction_seconds": .001,
                 "diagnostic_seconds": .001, "setup_seconds": .01} for s in base.TESTS]
    observed = {"scene_count": 2048, "candidate_floor": 31,
                "splits": {s: {"rows": [{"scene_id": i, "candidate_count": (0, 1, 31, 70)[i % 4]}
                                        for i in range(512)]} for s in base.TESTS}}
    observed["pairs_with_floor31"] = sum(math.comb(max(r["candidate_count"], 31), 2)
                                         for v in observed["splits"].values() for r in v["rows"])
    return profiles, {"scene_count": 2048, "decoded_bytes": 2048000}, observed


class ForecastTests(unittest.TestCase):
    def test_formula_complete_roster_floor_allowance_three_legs_and_margin(self):
        profiles, inventory, observed = fake_forecast_inputs()
        result = revision.revised_forecast(profiles, inventory, observed, charged_seconds=23.)
        leg = .001/math.comb(31, 2)*observed["pairs_with_floor31"]
        self.assertEqual(result["projected_total_seconds"], 23.+2*(2.048+3*leg+.08+900)+600)
        self.assertTrue(result["time_fits"])
        self.assertEqual(result["unmeasured_bookkeeping_allowance_seconds"], 900.)
        old = base.forecast(profiles, inventory, charged_seconds=23.)
        self.assertEqual(result["projected_new_bytes"], old["projected_new_bytes"])
        self.assertFalse(revision.revised_forecast(profiles, inventory, observed,
                                                  charged_seconds=7200.)["time_fits"])
        for value in (True, -1, 83):
            bad = deepcopy(observed)
            bad["splits"]["iid"]["rows"][0]["candidate_count"] = value
            with self.assertRaises(ValueError):
                revision.revised_forecast(profiles, inventory, bad, charged_seconds=23.)


class AdmissionReviewTests(FixtureBase):
    def setUp(self):
        super().setUp()
        self.make_review()

    def test_review_must_cover_snapshot_limits_status_and_independent_reports(self):
        report = self.store.publish_json("fixture-audit.json", {"status": "PASS", "fixture": True})
        report_ref = revision.reference(self.project, base.OUTPUT+"/"+report["path"])
        good = {**self.review, "reports": [report_ref]}
        candidates = [good, {**good, "effective_limits": {**revision.EFFECTIVE_LIMITS, "seconds": 9000.}},
                      {**good, "status": "PENDING"}, {**good, "reports": []},
                      {**good, "snapshot": {"changed": True}}]
        for i, value in enumerate(candidates):
            self.store.publish_json(f"review-{i}.json", value)
            ref = revision.reference(self.project, base.OUTPUT+f"/review-{i}.json")
            with patch.object(revision, "snapshot", return_value=self.review["snapshot"]):
                if i == 0:
                    self.assertEqual(revision.accept_review(self.store, ref), good)
                else:
                    with self.assertRaises(ValueError):
                        revision.accept_review(self.store, ref)
        self.assertFalse(self.store.path("attempts/0006/start.json").exists())

    def admission_context(self, runner, *, expensive=False):
        profiles, inventory, observed = fake_forecast_inputs()
        if expensive:
            for row in profiles:
                row["diagnostic_seconds"] = 10.
        for p in (patch.object(revision, "forecast_evidence", return_value=(
                    {"profiles": profiles, "runtime_revision": {"fixture": "fast"}},
                    {"workload": observed, "runtime_revision": {"fixture": "workload"}})),
                  patch.object(runner, "inventory", return_value=inventory),
                  patch.object(revision.time, "monotonic", side_effect=lambda: self.now)):
            p.start()
            self.addCleanup(p.stop)

    def test_first_admission_then_resume_uses_original_forecast_without_refund(self):
        attempt = self.attempt()
        runner = revision.RevisedRunner(self.store, attempt, self.ref, self.review)
        self.admission_context(runner)
        first = runner.admit()
        self.now += 3.
        attempt.finish("PAUSED")
        self.assertEqual(first, runner.admit(require_existing=True))
        resumed = self.attempt()
        self.assertEqual(resumed.charged_before, self.historical_total+3.)
        resumed.finish("COMPLETE")
        path = self.store.path("execution/admission.json")
        changed = deepcopy(first)
        changed["forecast"]["unmeasured_bookkeeping_allowance_seconds"] = 0.
        path.write_bytes(revision.encoded(changed))
        with self.assertRaisesRegex(ValueError, "saved admission formula differs"):
            runner.admit(require_existing=True)

    def test_rejected_forecast_or_missing_admission_cannot_start_roster(self):
        attempt = self.attempt()
        runner = revision.RevisedRunner(self.store, attempt, self.ref, self.review)
        self.admission_context(runner, expensive=True)
        with self.assertRaisesRegex(ValueError, "requires the original"):
            runner.admit(require_existing=True)
        with self.assertRaises(base.Paused):
            runner.admit()
        self.assertFalse(self.store.path("execution/admission.json").exists())
        self.assertFalse(self.store.path("scenes").exists())
        attempt.finish("PAUSED")


class MixedCorpus(FixtureCorpus):
    def extract_scene(self, loaded, scene_id):
        extracted = super().extract_scene(loaded, scene_id)
        if scene_id == 1:
            from src.atencion_armonica.generative_evidence import decouple
            empty, _ = extracted_fixture(empty=True)
            extracted.update(empty)
            gen = empty["compact"]["log_channel"].astype(base.np.float32)
            dec, sham = decouple(empty["compact"]["partitions"], gen, split_seed=3, scene_id=0)
            extracted["channels"] = {str(cp): {"local": base.np.zeros_like(gen), "generative": gen,
                                               "decoupled": dec, "sham": sham}
                                     for cp in (2026090721, 2026090722, 2026090723)}
        return extracted


class RunnerTests(FixtureBase):
    def setUp(self):
        super().setUp()
        for p in (patch.object(base, "SCENE_IDS", (0, 1)), patch.object(revision, "SCENE_IDS", (0, 1)),
                  patch.object(base, "ClosedCorpus", MixedCorpus), patch.object(revision, "ClosedCorpus", MixedCorpus),
                  patch.object(base, "forecast", return_value={"time_fits": True, "outputs_fit": True}),
                  patch("builtins.print")):
            p.start()
            self.addCleanup(p.stop)
        FixtureCorpus.extracted_ids = []
        old = base.DiagnosticRunner(self.store, CheckOnly())
        old.prepare()
        old.profile()
        refs = [revision.reference(self.store.root, f"scenes/{s}/00000/receipt.json") for s in base.TESTS]
        self.make_review(refs)
        self.snapshot_patch = patch.object(revision, "snapshot", return_value=self.review["snapshot"])
        self.snapshot_patch.start()
        self.addCleanup(self.snapshot_patch.stop)
        self.old_prefix_bytes = {r["path"]: self.store.read(r) for r in refs}
        self.admission_patch = patch.object(revision.RevisedRunner, "admit", return_value={"fixture": True})
        self.admission_patch.start()
        self.addCleanup(self.admission_patch.stop)

    def start(self, operation):
        attempt = self.attempt(operation)
        return revision.RevisedRunner(self.store, attempt, self.ref, self.review), attempt

    def seal(self, runner, attempt, *, publish=True):
        runner.stage_completion(attempt.relative)
        attempt.check()
        self.now += .1
        attempt.finish("COMPLETE", completion=runner.pending_reference())
        ref = revision.reference(self.store.root, attempt.relative+"/finish.json")
        if publish:
            runner.publish_completion(ref)
        return ref

    def test_full_fixture_run_and_replay_preserve_prefixes_and_empty_scenes(self):
        runner, attempt = self.start("run")
        self.assertEqual(runner.run()["scene_count"], 8)
        self.assertFalse(self.store.path(revision.MARKERS["run"]).exists())
        self.seal(runner, attempt)
        self.assertEqual(len(FixtureCorpus.extracted_ids), 8)
        for path, raw in self.old_prefix_bytes.items():
            self.assertEqual(self.store.path(path).read_bytes(), raw)
        for split in base.TESTS:
            payload, _, original = runner.load_unit(split, 1)
            self.assertEqual(payload["result"]["candidate_count"], 0)
            extracted = payload["extracted"]
            result = base.diagnose_scene(extracted["compact"], extracted["canonical_labels"], extracted["predictions"])
            self.assertEqual(base.bundle_bytes({**payload, "result": result})[0], original)
        replay, second = self.start("replay")
        self.assertEqual(replay.replay()["scene_count"], 8)
        self.seal(replay, second)
        self.assertEqual(len(FixtureCorpus.extracted_ids), 8)
        self.assertFalse(self.store.path("complete.json").exists())
        self.assertFalse(self.store.path("replayed.json").exists())
        self.assertIsNone(revision.recover_completion(self.store, "replay", self.ref, self.review))
        repeated, third = self.start("replay")
        with patch.object(repeated, "load_unit", wraps=repeated.load_unit) as loads:
            self.assertEqual(repeated.replay()["status"], "REPLAYED_REUSED")
        self.assertEqual(loads.call_count, 8)
        third.finish("COMPLETE")

    def test_sealed_recovery_is_revision_only_and_does_not_consume_another_attempt(self):
        runner, attempt = self.start("run")
        runner.run()
        self.seal(runner, attempt, publish=False)
        self.assertIsNone(base.recover_completion(self.store, "run"))
        with self.assertRaises(FileNotFoundError):
            base.DiagnosticRunner(self.store, CheckOnly()).replay()
        self.assertEqual(revision.recover_completion(self.store, "run", self.ref, self.review)["status"],
                         "COMPLETE_RECOVERED")
        self.assertFalse(self.store.path("attempts/0007/start.json").exists())
        self.assertIsNone(base.recover_completion(self.store, "run"))
        with self.assertRaises(FileNotFoundError):
            base.DiagnosticRunner(self.store, CheckOnly()).replay()

    def test_paused_candidate_is_not_promoted_and_prefix_resume_reuses(self):
        runner, attempt = self.start("run")
        runner.run()
        runner.stage_completion(attempt.relative)
        attempt.finish("PAUSED")
        self.assertIsNone(revision.recover_completion(self.store, "run", self.ref, self.review))
        self.assertFalse(self.store.path(revision.MARKERS["run"]).exists())
        count = len(FixtureCorpus.extracted_ids)
        resumed, second = self.start("run")
        resumed.run()
        self.seal(resumed, second)
        self.assertEqual(len(FixtureCorpus.extracted_ids), count)

    def test_wrong_candidate_revision_is_not_recovered(self):
        runner, attempt = self.start("run")
        runner.run()
        runner.pending[1]["execution_revision"] = {"path": "wrong", "sha256": "0"*64}
        self.seal(runner, attempt, publish=False)
        with self.assertRaisesRegex(ValueError, "candidate revision differs"):
            revision.recover_completion(self.store, "run", self.ref, self.review)
        self.assertFalse(self.store.path(revision.MARKERS["run"]).exists())

    def test_production_entrypoint_cannot_accept_fixture_roster(self):
        with self.assertRaisesRegex(ValueError, "complete fixed roster"):
            revision.execute("run", self.ref, project=self.project)

    def test_direct_recovery_authenticates_review_before_mutation(self):
        runner, attempt = self.start("run")
        runner.run()
        self.seal(runner, attempt, publish=False)
        self.store.path("fixture-review.json").write_bytes(b"{}\n")
        with self.assertRaises(ValueError):
            revision.recover_completion(self.store, "run", self.ref, self.review)
        self.assertFalse(self.store.path(revision.MARKERS["run"]).exists())

    def test_legacy_execute_cannot_adopt_revised_complete_marker(self):
        runner, attempt = self.start("run")
        runner.run()
        self.seal(runner, attempt)
        manifest = self.store.manifest()[0]
        # Pass the legacy CLI's fixed-roster and manifest prechecks; the failure
        # must be missing LEGACY complete.json, not unrelated fixture setup.
        with patch.object(base, "SCENE_IDS", tuple(range(512))), \
                patch.object(base, "reviewed_manifest", return_value=manifest):
            with self.assertRaisesRegex(FileNotFoundError, "complete.json"):
                base.execute("replay", project=self.project)
        self.assertFalse(self.store.path("complete.json").exists())
        self.assertFalse(self.store.path("replayed.json").exists())
        end = old_budget.AttemptBudget._local_json(self.store.path("attempts/0007/finish.json"))
        self.assertEqual(end["status"], "FAILED")

    def test_execute_success_returns_committed_not_staged_status(self):
        original = revision.RevisedRunner.run
        def run_small(instance):
            with patch.object(revision, "SCENE_IDS", (0, 1)):
                return original(instance)
        # Exercise execute's real commit/return path with the same eight-unit
        # computational fixture; reduction is patched only inside this test.
        with patch.object(revision, "SCENE_IDS", tuple(range(512))), \
                patch.object(revision.RevisedRunner, "run", autospec=True, side_effect=run_small):
            result = revision.execute("run", self.ref, project=self.project)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["scene_count"], 8)
        self.assertTrue(self.store.path(revision.MARKERS["run"]).is_file())

    def test_existing_marker_does_not_hide_corrupted_or_missing_bundle(self):
        runner, attempt = self.start("run")
        runner.run()
        self.seal(runner, attempt)
        self.assertIsNone(revision.recover_completion(self.store, "run", self.ref, self.review))
        self.store.path("scenes/iid/00001/bundle.gz").write_bytes(b"corrupted owned fixture")
        resumed, second = self.start("run")
        with self.assertRaises(ValueError):
            resumed.run()
        second.finish("FAILED")
        self.store.path("scenes/iid/00001/bundle.gz").unlink()  # Owned missing-blob fixture.
        repeated, third = self.start("replay")
        with self.assertRaises(FileNotFoundError):
            repeated.replay()
        third.finish("FAILED")


if __name__ == "__main__":
    unittest.main()
