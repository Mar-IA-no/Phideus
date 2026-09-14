"""Explicit operational amendment; immutable scientific v1 lineage is preserved.

This is not a new profile, experiment, budget reset, or CUDA path. New operation
names and completion markers deliberately cannot be recovered by the v1 CLI.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import signal
import time

from . import operator_objective_runner as base  # CPU environment before NumPy ports.
from . import operator_diagnostic_fast as fast
from .operator_objective_budget import AttemptBudget, LIMITS
from experiments.atencion_armonica import inventory_diagnostic_workload as workload

ROOT, WORK, OUTPUT = base.ROOT, base.WORK, base.OUTPUT
TESTS, SCENE_IDS = base.TESTS, base.SCENE_IDS
EFFECTIVE_LIMITS = {**LIMITS, "seconds": 7200.}
MARKERS = {"run": "execution/complete.json", "replay": "execution/replayed.json"}
WORKLOAD_REVIEW = {"path": WORK+"/workload_review.v1.json",
                   "sha256": "4d6158d0769daf8b6456bdfe982452f1ea9327f44e2c837ece76fb31d8d0a537"}
FILES = ("src/atencion_armonica/diagnostic_execution_revision.py",
         "experiments/atencion_armonica/execute_diagnostic_revision.py",
         "experiments/atencion_armonica/test_diagnostic_execution_revision.py",
         "experiments/atencion_armonica/AMENDMENT_OPERATOR_OBJECTIVE_EXECUTION.md")
reference = workload.prior.reference


def snapshot(store):
    workload.accept_review(store, WORKLOAD_REVIEW)
    return {"manifest": store.manifest()[1], "workload_review": WORKLOAD_REVIEW,
            "historical_attempts": [
                {k: reference(store.root, f"attempts/{i:04d}/{k}.json")
                 for k in ("start", "finish")} for i in range(6)],
            "legacy_prefixes": [reference(store.root, f"scenes/{s}/00000/receipt.json") for s in TESTS],
            "files": [reference(store.project, name) for name in FILES]}


def accept_review(store, ref):
    reader = base.AuthenticatedReader(store.project)
    review = reader.json(ref)
    if (review.get("status") != "EXECUTION_REVISION_REVIEW_COMPLETE" or not review.get("reports")
            or review.get("supersedes_operational_limits") != LIMITS
            or review.get("effective_limits") != EFFECTIVE_LIMITS):
        raise ValueError("execution revision authority or effective limits differ")
    base.equal(review["snapshot"], snapshot(store), "execution review does not cover current sources/history")
    for report in review["reports"]:
        reader.bytes(report)
    return review


def ledger_state(store, review_ref, review):
    """Authenticate the fixed legacy prefix and retain every subsequent charge."""
    manifest_ref = store.manifest()[1]
    history = review["snapshot"]["historical_attempts"]
    if len(history) != 6:
        raise ValueError("historical attempt extent differs")
    for i, row in enumerate(history):
        for kind in ("start", "finish"):
            if row[kind]["path"] != f"attempts/{i:04d}/{kind}.json":
                raise ValueError("historical attempt identity differs")
            store.json(row[kind])
    paths = sorted((store.root/"attempts").glob("*/start.json"))
    if len(paths) < len(history):
        raise ValueError("historical ledger is incomplete")
    charged, profile_charged = 0., 0.
    for i, path in enumerate(paths):
        if path.parent.name != f"{i:04d}":
            raise ValueError("attempt ledger is not contiguous")
        start = AttemptBudget._local_json(path)
        base.equal(start["manifest"], manifest_ref, "attempt manifest differs")
        if (type(start["allocated_seconds"]) not in (float, int)
                or not math.isfinite(start["allocated_seconds"]) or start["allocated_seconds"] <= 0
                or start["charged_before"] != charged):
            raise ValueError("attempt allocation or cumulative charge differs")
        if i >= len(history):
            base.equal(start.get("execution_revision"), review_ref, "attempt execution revision differs")
            if start["operation"] not in ("revision_run", "revision_replay", "revision_audit"):
                raise ValueError("legacy/unrecognized operation after revision")
            reserve = 0 if start["operation"] == "revision_audit" else EFFECTIVE_LIMITS["audit_reserve_seconds"]
            if start["allocated_seconds"] != EFFECTIVE_LIMITS["seconds"]-charged-reserve:
                raise ValueError("revised reservation differs from effective residual")
        finish_path = path.parent/"finish.json"
        if finish_path.exists():
            finish = AttemptBudget._local_json(finish_path)
            base.equal(finish["start"], start, "attempt finish differs from start")
            if (finish["status"] not in ("COMPLETE", "FAILED", "PAUSED", "BUDGET_EXHAUSTED")
                    or type(finish["seconds"]) not in (float, int)
                    or not math.isfinite(finish["seconds"]) or finish["seconds"] < 0):
                raise ValueError("attempt terminal state/charge invalid")
            if finish["status"] == "BUDGET_EXHAUSTED":
                raise base.BudgetExceeded("terminal exhausted budget requires explicit redesign")
            used = finish["seconds"]
            if finish["charged_total"] != charged+used:
                raise ValueError("finish cumulative charge differs")
        else:
            used = start["allocated_seconds"]
        charged += used
        if start["operation"] == "profile":
            profile_charged += used
    return {"charged_seconds": charged, "profile_seconds": profile_charged, "attempt_count": len(paths)}


class RevisedBudget(AttemptBudget):
    """Same guards/finish as v1; explicit reviewed limits, no synthetic carryover."""

    def __init__(self, store, operation, review_ref, review, *, clock=time.monotonic):
        if operation not in ("run", "replay", "audit"):
            raise ValueError("revision forbids another profile or preparation")
        self.store, self.operation, self.clock = store, operation, clock
        self.execution_ref = review_ref
        self.limits = dict(EFFECTIVE_LIMITS)
        manifest, self.manifest_ref = store.manifest()
        if manifest.get("limits") != LIMITS:
            raise ValueError("historical manifest limits changed")
        self.started = clock()
        ledger = ledger_state(store, review_ref, review)
        self.charged_before, self.profile_before = ledger["charged_seconds"], ledger["profile_seconds"]
        reserve = 0 if operation == "audit" else self.limits["audit_reserve_seconds"]
        self.allocation = self.limits["seconds"]-self.charged_before-reserve
        if self.allocation <= 0:
            raise base.BudgetExceeded("cumulative revised budget exhausted before launch")
        self.relative = f"attempts/{ledger['attempt_count']:04d}"
        self.start_record = {"manifest": self.manifest_ref, "operation": "revision_"+operation,
                             "execution_revision": review_ref, "charged_before": self.charged_before,
                             "allocated_seconds": self.allocation}
        self.closed = False
        store.publish_json(self.relative+"/start.json", self.start_record)
        try:
            self.check()
        except base.BudgetExceeded:
            self.finish("BUDGET_EXHAUSTED")
            raise
        except Exception:
            self.finish("FAILED")
            raise

    def check(self, *, additional_bytes=0):
        base.AuthenticatedReader(self.store.project).bytes(self.execution_ref)
        return super().check(additional_bytes=additional_bytes)


def revised_forecast(profiles, inventory, observed, *, charged_seconds):
    old = base.forecast(profiles, inventory, charged_seconds=charged_seconds)
    if (observed["scene_count"] != 2048 or observed["candidate_floor"] != 31
            or set(observed["splits"]) != set(TESTS)):
        raise ValueError("forecast workload extent differs")
    pair_count = 0
    for split in TESTS:
        rows = observed["splits"][split]["rows"]
        if [r["scene_id"] for r in rows] != list(range(512)):
            raise ValueError("forecast scene roster differs")
        for row in rows:
            c = row["candidate_count"]
            if type(c) is not int or not 0 <= c <= 82:
                raise ValueError("forecast candidate extent differs")
            pair_count += math.comb(max(c, 31), 2)
    if pair_count != observed["pairs_with_floor31"]:
        raise ValueError("forecast workload aggregate differs")
    rate = max(r["diagnostic_seconds"]/math.comb(r["candidate_count"], 2) for r in profiles)
    leg = rate*pair_count
    total = charged_seconds+2*(old["projected_extraction_seconds"]+3*leg
                              +old["projected_remaining_setup_seconds"]+900.)+600.
    return {"schema": "operator-objective-revised-forecast-v1", "charged_seconds": charged_seconds,
            "projected_extraction_seconds": old["projected_extraction_seconds"],
            "diagnostic_seconds_per_pair": rate, "observed_pairs_floor31": pair_count,
            "projected_diagnostic_leg_seconds": leg, "diagnostic_equivalent_legs": 3,
            "projected_setup_seconds": old["projected_remaining_setup_seconds"],
            "unmeasured_bookkeeping_allowance_seconds": 900., "remaining_safety_factor": 2,
            "audit_reserve_seconds": 600., "projected_total_seconds": total,
            "effective_limit_seconds": EFFECTIVE_LIMITS["seconds"],
            "projected_new_bytes": old["projected_new_bytes"],
            "time_fits": total <= EFFECTIVE_LIMITS["seconds"], "outputs_fit": old["outputs_fit"]}


def forecast_evidence(store):
    old_review = workload.accept_review(store, WORKLOAD_REVIEW)
    observed = workload.previous(store, old_review)
    fast_review = workload.prior.accept_review(store, workload.FAST_REVIEW)
    profile = workload.prior.previous_profile(store, fast_review)
    if observed is None or profile is None:
        raise ValueError("completed workload/first fast profile is required")
    return profile, observed

# Local names used by the three narrowly copied v1 orchestration methods.
# Numerical reductions and publication primitives are inherited unchanged.
ClosedCorpus = base.ClosedCorpus
AuthenticatedReader = base.AuthenticatedReader
ScenarioAccumulator = base.ScenarioAccumulator
verify_scene_result = base.verify_scene_result
verify_delivered_channel = base.verify_delivered_channel
equal, encoded = base.equal, base.encoded


class RevisedRunner(base.DiagnosticRunner):
    def __init__(self, store, attempt, review_ref, review):
        super().__init__(store, attempt)
        self.execution_ref, self.review = review_ref, review
        self.legacy_prefixes = {r["path"]: r for r in review["snapshot"]["legacy_prefixes"]}

    def record(self, relative):
        value = super().record(relative)
        scoped = (relative.startswith(("scenes/", "summaries/", "execution/")))
        if scoped:
            if relative in self.legacy_prefixes:
                equal(self.store.json(self.legacy_prefixes[relative]), value, "legacy prefix changed")
            else:
                equal(value.get("execution_revision"), self.execution_ref, "record revision differs")
        if relative in MARKERS.values():
            finish = self.store.json(value["attempt_finish"])
            operation = "revision_run" if relative == MARKERS["run"] else "revision_replay"
            if finish["status"] != "COMPLETE" or finish["start"]["operation"] != operation:
                raise ValueError("revised completion has no matching successful operator")
            equal(finish["start"]["manifest"], self.manifest_ref, "completion manifest differs")
            equal(finish["start"]["execution_revision"], self.execution_ref, "finish revision differs")
            scientific = {k: v for k, v in value.items() if k != "attempt_finish"}
            seal = finish["completion"]
            if seal["path"] != relative or seal["sha256"] != hashlib.sha256(encoded(scientific)).hexdigest():
                raise ValueError("completion differs from successful revised finish")
            equal(self.store.json(seal["candidate"]), scientific, "revised completion candidate differs")
        return value

    def publish_record(self, relative, value, *, publisher=None):
        return super().publish_record(relative, {"execution_revision": self.execution_ref, **value},
                                      publisher=publisher)

    def publish_completion(self, finish_ref):
        if self.pending is None:
            return
        finish = self.store.json(finish_ref)
        relative, value = self.pending
        operation = next((k for k, v in MARKERS.items() if v == relative), None)
        if (operation is None or finish["status"] != "COMPLETE"
                or finish["start"]["operation"] != "revision_"+operation):
            raise ValueError("cannot publish without matching successful revised operator")
        equal(finish["start"]["manifest"], self.manifest_ref, "finish manifest differs")
        equal(finish["start"]["execution_revision"], self.execution_ref, "finish revision differs")
        equal(value["execution_revision"], self.execution_ref, "candidate revision differs")
        equal(finish["completion"], self.pending_reference(), "finish does not seal pending result")
        equal(self.store.json(self.pending_candidate), value, "staged completion differs")
        self.store.publish_json(relative, {**value, "attempt_finish": finish_ref})
        self.pending = self.pending_candidate = None

    def admit(self, *, require_existing=False):
        relative = "execution/admission.json"
        if self.store.path(relative).exists():
            admitted = self.record(relative)
            if (admitted["status"] != "ROSTER_ADMITTED"
                    or not admitted["forecast"]["time_fits"] or not admitted["forecast"]["outputs_fit"]):
                raise ValueError("existing admission does not authorize complete roster")
            # The immutable first admission is attached to its actual reservation.
            start = self.store.json(admitted["start"])
            equal(start["execution_revision"], self.execution_ref, "admission revision differs")
            if start["operation"] != "revision_run":
                raise ValueError("admission has wrong operator")
            profile, observed = forecast_evidence(self.store)
            equal(admitted["fast_profile_revision"], profile["runtime_revision"], "admission profile differs")
            equal(admitted["workload_revision"], observed["runtime_revision"], "admission workload differs")
            charged = admitted["forecast"]["charged_seconds"]
            if not start["charged_before"] <= charged < start["charged_before"]+start["allocated_seconds"]:
                raise ValueError("admission charge outside its original reservation")
            equal(admitted["forecast"], revised_forecast(profile["profiles"], self.inventory(),
                  observed["workload"], charged_seconds=charged), "saved admission formula differs")
            return admitted
        if require_existing:
            raise ValueError("replay requires the original revised admission")
        if self.store.path(MARKERS["run"]).exists():
            raise ValueError("completed roster has no admission")
        profile, observed = forecast_evidence(self.store)
        charged = self.attempt.charged_before+(time.monotonic()-self.attempt.started)
        projection = revised_forecast(profile["profiles"], self.inventory(), observed["workload"],
                                      charged_seconds=charged)
        if not projection["time_fits"] or not projection["outputs_fit"]:
            raise base.Paused("revised forecast does not fit; explicit review required")
        self.publish_record(relative, {"status": "ROSTER_ADMITTED", "forecast": projection,
            "start": reference(self.store.root, self.attempt.relative+"/start.json"),
            "fast_profile_revision": profile["runtime_revision"],
            "workload_revision": observed["runtime_revision"]})
        return self.record(relative)

    def unit(self, corpus, loaded, scene_id, *, phase, setup_seconds=0.):
        split = loaded["split"]
        relative = f"scenes/{split}/{scene_id:05d}"
        self.attempt.check()
        if self.store.path(relative+"/receipt.json").exists():
            return self.load_unit(split, scene_id)[:2]
        if self.store.path(relative).exists():
            raise ValueError("orphaned scene cannot be adopted or overwritten")
        started = time.monotonic()
        extracted = corpus.extract_scene(loaded, scene_id)
        extraction_seconds = time.monotonic()-started
        self.attempt.check()
        started = time.monotonic()
        result = fast.diagnose_scene(extracted["compact"], extracted["canonical_labels"], extracted["predictions"])
        coverage = verify_scene_result(extracted, result)
        payload = {"extracted": extracted, "result": result, "coverage": coverage}
        raw, receipt = fast.bundle_bytes(payload)
        self.attempt.check(additional_bytes=len(raw))
        with self.store.transaction(relative) as publication:
            ref = publication.publish("bundle.gz", raw)
            ref.update(receipt)
            # Codec and fsynced bundle IO are included in the diagnostic leg.
            diagnostic_seconds = time.monotonic()-started
            timing = {"split": split, "scene_id": scene_id, "candidate_count": result["candidate_count"],
                      "decoded_bytes": extracted["provenance"]["artifact"]["decoded_bytes"],
                      "bundle_bytes": len(raw), "setup_seconds": setup_seconds,
                      "extraction_seconds": extraction_seconds, "diagnostic_seconds": diagnostic_seconds}
            record = {"execution_revision": self.execution_ref, "status": "SCENE_COMPLETE", "split": split, "scene_id": scene_id,
                      "phase": phase, "bundle": ref, "timing": timing}
            self.publish_record("receipt.json", record, publisher=publication)
        return payload, {"manifest": self.manifest_ref, **record}

    def run(self):
        self.admit()
        inventory = self.inventory()
        if self.store.path(MARKERS["run"]).exists():
            record = self.record(MARKERS["run"])
            self.validate_complete(record)
            return {"status": "COMPLETE_REUSED"}
        corpus = ClosedCorpus(AuthenticatedReader(self.store.project))
        corpus.receipts = inventory["records"]
        summaries, counts, units = {}, {}, {}
        for split in TESTS:
            loaded = corpus.load_split(split)
            aggregate = ScenarioAccumulator(split, expected_scenes=len(SCENE_IDS))
            units[split] = []
            for scene_id in SCENE_IDS:
                payload, unit_record = self.unit(corpus, loaded, scene_id, phase="run")
                receipt_raw = encoded(unit_record)
                units[split].append({"path": f"scenes/{split}/{scene_id:05d}/receipt.json",
                                     "sha256": hashlib.sha256(receipt_raw).hexdigest(), "bytes": len(receipt_raw)})
                aggregate.add(scene_id, payload["coverage"]["planted"], payload["result"])
                if scene_id % 32 == 0:
                    print(json.dumps({"split": split, "completed_scene": scene_id}), flush=True)
            summary = aggregate.finalize()
            self.attempt.check(additional_bytes=len(encoded(summary)))
            path = f"summaries/{split}"
            receipt_path = path+"/receipt.json"
            if self.store.path(receipt_path).exists():
                receipt = self.record(receipt_path)
                ref = receipt["summary"]
                if self.store.read(ref) != encoded(summary):
                    raise ValueError("existing summary differs; cannot repair or overwrite")
            else:
                if self.store.path(path).exists():
                    raise ValueError("orphaned summary cannot be adopted")
                with self.store.transaction(path) as publication:
                    ref = publication.publish_json("summary.json", summary)
                    self.publish_record("receipt.json", {"status": "SUMMARY_COMPLETE", "summary": ref},
                                        publisher=publication)
            summaries[split], counts[split] = ref, len(SCENE_IDS)
            del loaded, aggregate
        record = {"status": "COMPLETE", "scene_counts": counts, "summaries": summaries, "scenes": units,
                  "authority": "RETROSPECTIVE_NOT_PROMOTED"}
        self.validate_complete({"manifest": self.manifest_ref, "execution_revision": self.execution_ref, **record})
        self.pending = (MARKERS["run"], {"manifest": self.manifest_ref, "execution_revision": self.execution_ref, **record})
        return {"status": "ROSTER_READY_FOR_COMMIT", "scene_count": sum(counts.values())}

    def replay(self):
        self.admit(require_existing=True)
        complete = self.record(MARKERS["run"])
        self.validate_complete_header(complete)
        count = 0
        for split in TESTS:
            aggregate = ScenarioAccumulator(split, expected_scenes=len(SCENE_IDS))
            for scene_id in SCENE_IDS:
                self.attempt.check()
                authoritative = self.store.json(complete["scenes"][split][scene_id])
                payload, local, original = self.load_unit(split, scene_id)
                equal(authoritative, local, "replay scene receipt differs from complete index")
                extracted = payload["extracted"]
                for channel in extracted["channels"].values():
                    verify_delivered_channel(extracted["compact"], extracted["normalizer"],
                                             {a: channel[a] for a in ("local", "generative", "decoupled")}, channel["sham"])
                result = fast.diagnose_scene(extracted["compact"], extracted["canonical_labels"], extracted["predictions"])
                coverage = verify_scene_result(extracted, result)
                repeated, _ = fast.bundle_bytes({"extracted": extracted, "result": result, "coverage": coverage})
                if repeated != original:
                    raise ValueError(f"scientific replay differs: {split}/{scene_id}")
                aggregate.add(scene_id, coverage["planted"], result)
                count += 1
                if scene_id % 32 == 0:
                    print(json.dumps({"replay_split": split, "completed_scene": scene_id}), flush=True)
            if encoded(aggregate.finalize()) != self.store.read(complete["summaries"][split]):
                raise ValueError(f"scientific summary replay differs: {split}")
        record = {"status": "REPLAYED", "scene_count": count, "summaries": complete["summaries"],
                  "authority": "RETROSPECTIVE_NOT_PROMOTED"}
        if self.store.path(MARKERS["replay"]).exists():
            previous = self.record(MARKERS["replay"])
            equal({k: v for k, v in previous.items() if k != "attempt_finish"},
                  {"manifest": self.manifest_ref, "execution_revision": self.execution_ref, **record}, "old replay record differs")
            return {"status": "REPLAYED_REUSED", "scene_count": count}
        else:
            self.pending = (MARKERS["replay"], {"manifest": self.manifest_ref, "execution_revision": self.execution_ref, **record})
        return {"status": "REPLAY_READY_FOR_COMMIT", "scene_count": count}


def recover_completion(store, operation, review_ref, review):
    """Only revised sealed candidates; never delegate recovery to the v1 CLI."""
    if operation not in MARKERS:
        return None
    authenticated = accept_review(store, review_ref)
    equal(review, authenticated, "recovery supplied review differs from authenticated review")
    ledger_state(store, review_ref, authenticated)
    review = authenticated
    runner = RevisedRunner(store, None, review_ref, review)
    marker = MARKERS[operation]
    if store.path(marker).exists():
        # An existing marker is not an interrupted commit to recover. The CLI
        # must budget and execute full validation/replay, including blob reads.
        return None
    for path in sorted((store.root/"attempts").glob("*/finish.json"), reverse=True):
        ref = reference(store.root, path.relative_to(store.root).as_posix())
        finish = store.json(ref)
        if (finish["status"] != "COMPLETE" or finish["start"]["operation"] != "revision_"+operation
                or "completion" not in finish):
            continue
        equal(finish["start"]["execution_revision"], review_ref, "recovery revision differs")
        equal(finish["start"]["manifest"], runner.manifest_ref, "recovery manifest differs")
        equal(AttemptBudget._local_json(path.parent/"start.json"), finish["start"], "recovery start differs")
        seal = finish["completion"]
        expected = path.parent.relative_to(store.root).as_posix()+"/completion_candidate.json"
        if seal["path"] != marker or seal["candidate"]["path"] != expected:
            raise ValueError("recovery candidate identity differs")
        value = store.json(seal["candidate"])
        if hashlib.sha256(encoded(value)).hexdigest() != seal["sha256"]:
            raise ValueError("recovery candidate differs from successful seal")
        equal(value["manifest"], runner.manifest_ref, "candidate manifest differs")
        equal(value["execution_revision"], review_ref, "candidate revision differs")
        if operation == "run":
            runner.validate_complete_header(value)
        else:
            parent = runner.record(MARKERS["run"])
            runner.validate_complete_header(parent)
            if value["status"] != "REPLAYED" or value["scene_count"] != len(TESTS)*len(SCENE_IDS):
                raise ValueError("recovery replay extent differs")
            equal(value["summaries"], parent["summaries"], "recovery replay summaries differ")
        runner.pending, runner.pending_candidate = (marker, value), seal["candidate"]
        runner.publish_completion(ref)
        return {"status": "COMPLETE_RECOVERED" if operation == "run" else "REPLAYED_RECOVERED"}
    return None


def execute(operation, review_ref, *, project=ROOT):
    if operation not in MARKERS or SCENE_IDS != tuple(range(512)) or len(TESTS) != 4:
        raise ValueError("production requires complete fixed roster and run/replay operation")
    project = Path(project)
    with base.exclusive_lock(project):
        store = base.DiagnosticStore(project/OUTPUT, project_root=project)
        review = accept_review(store, review_ref)
        ledger_state(store, review_ref, review)
        recovered = recover_completion(store, operation, review_ref, review)
        if recovered is not None:
            print(json.dumps(recovered), flush=True)
            return recovered
        attempt = RevisedBudget(store, operation, review_ref, review)
        status, value, runner = "FAILED", None, None
        previous = {}
        try:
            runner = RevisedRunner(store, attempt, review_ref, review)
            previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
            def paused(signum, frame):
                raise base.Paused(f"revised operator interrupted by signal {signum}")
            def timeout(signum, frame):
                raise base.BudgetExceeded("revised cumulative time guard reached")
            for s in (signal.SIGINT, signal.SIGTERM):
                signal.signal(s, paused)
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, max(.001, attempt.allocation-(time.monotonic()-attempt.started)))
            value = getattr(runner, operation)()
            runner.stage_completion(attempt.relative)
            attempt.check()
            status = "COMPLETE"
        except base.BudgetExceeded:
            status = "BUDGET_EXHAUSTED"
            raise
        except base.Paused:
            status = "PAUSED"
            raise
        finally:
            if previous:
                signal.setitimer(signal.ITIMER_REAL, 0)
                for s, handler in previous.items():
                    signal.signal(s, handler)
            terminal = attempt.finish(status, completion=runner.pending_reference()
                                      if status == "COMPLETE" and runner is not None else None)
            print(json.dumps({"operation": "revision_"+operation, "terminal": terminal, "result": value}), flush=True)
        if runner.pending is not None:
            runner.publish_completion(reference(store.root, attempt.relative+"/finish.json"))
            value = {**value, "status": "COMPLETE" if operation == "run" else "REPLAYED"}
            print(json.dumps({"operation": "revision_"+operation, "committed": value}), flush=True)
        return value
