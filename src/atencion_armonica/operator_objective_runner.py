"""CPU-only finite operator diagnostic: prepare, profile, run and exact replay.

The CLI requires a separately reviewed source snapshot before initialization.
It never trains, fits, forwards, draws a scene, or enters a CUDA code path.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import time

# Set before importing any numerical code, including the inherited truth port.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from .operator_objective_aggregate import ScenarioAccumulator
from .operator_objective_artifacts import DiagnosticStore, bundle_bytes, load_bundle
from .operator_objective_budget import AttemptBudget, BudgetExceeded, LIMITS, forecast
from .operator_objective_corpus import ClosedCorpus, COMPLETION, TESTS, equal, verify_scene_result
from .operator_objective_scene import diagnose_scene
from .operator_objective_sources import AuthenticatedReader, encoded, verify_delivered_channel

ROOT = Path(__file__).resolve().parents[2]
WORK = ".agent-work/phideus-operator-objective-20260914"
OUTPUT = "data/atencion_armonica/operator_objective_alignment_v1"
PROTOCOL = {"path": "experiments/atencion_armonica/PROTOCOL_OPERATOR_OBJECTIVE_ALIGNMENT.md",
            "sha256": "bbdc52f010523059f95d87664670f82dd1579a7e4d43492b44393907fc50dc0b"}
SCENE_IDS = tuple(range(512))


class Paused(RuntimeError):
    pass


@contextmanager
def exclusive_lock(project):
    folder = Path(project)/WORK
    if any(p.is_symlink() for p in [folder, *folder.parents]):
        raise ValueError("lock path cannot traverse symlinks")
    folder.mkdir(parents=True, exist_ok=True)
    fd = os.open(folder/"operator.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def source_snapshot(project):
    project = Path(project)
    names = sorted(p.relative_to(project).as_posix()
                   for p in (project/"src/atencion_armonica").glob("operator_objective_*.py"))
    if len(names) != 8:
        raise ValueError("diagnostic computational module roster differs")
    # Pin the actual inherited pure ports used here as well as the original
    # campaign's frozen sources (which do not enumerate all these helpers).
    names += ["src/atencion_armonica/"+name+".py" for name in
              ("generative_evidence_supervision", "observable_rival_evaluation", "observable_source_rivals")]
    reader = AuthenticatedReader(project)
    completion = reader.json(COMPLETION)
    freeze = reader.json(completion["stages"][0]["output"])
    for path, digest in freeze["sources"].items():
        reader.bytes({"path": path, "sha256": digest})
    result = {}
    for relative in names:
        path = project/relative
        if path.is_symlink():
            raise ValueError("diagnostic source cannot be a symlink")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    reader.bytes(PROTOCOL)
    return {"files": result, "inherited_sources": freeze["sources"], "protocol": PROTOCOL,
            "python": platform.python_version(), "numpy": np.__version__}


def reviewed_manifest(project, review_ref):
    reader = AuthenticatedReader(project)
    review = reader.json(review_ref)
    snapshot = source_snapshot(project)
    if review.get("status") != "IMPLEMENTATION_REVIEW_COMPLETE":
        raise ValueError("operational implementation review is not complete")
    equal(review["snapshot"], snapshot, "review does not cover current code/runtime")
    if not review.get("reports"):
        raise ValueError("review has no independent report evidence")
    for ref in review["reports"]:
        reader.bytes(ref)
    return {"schema": "operator-objective-run-manifest-v1", "status": "PREPARED",
            "completion": COMPLETION, "snapshot": snapshot, "review": review_ref,
            "limits": LIMITS, "tests": TESTS, "scene_count_per_test": 512,
            "diagnostic_cell_count": 27, "authority": "RETROSPECTIVE_NOT_PROMOTED"}


class DiagnosticRunner:
    def __init__(self, store, attempt):
        self.store, self.attempt = store, attempt
        self.manifest, self.manifest_ref = store.manifest()
        self.pending = None
        self.pending_candidate = None

    def record(self, relative):
        """Read a completed local receipt, authenticate payloads via its refs."""
        path = self.store.path(relative)
        if not path.is_file():
            raise FileNotFoundError(relative)
        raw = path.read_bytes()
        value = json.loads(raw)
        if encoded(value) != raw:
            raise ValueError("noncanonical completion receipt")
        equal(value["manifest"], self.manifest_ref, "receipt belongs to another manifest")
        if relative in ("complete.json", "replayed.json"):
            finish = self.store.json(value["attempt_finish"])
            operation = "run" if relative == "complete.json" else "replay"
            if finish["status"] != "COMPLETE" or finish["start"]["operation"] != operation:
                raise ValueError("completion has no successful matching operator")
            equal(finish["start"]["manifest"], self.manifest_ref, "operator manifest differs")
            scientific = {k: v for k, v in value.items() if k != "attempt_finish"}
            seal = finish["completion"]
            if seal["path"] != relative or seal["sha256"] != hashlib.sha256(encoded(scientific)).hexdigest():
                raise ValueError("completion differs from successful finish")
            equal(self.store.json(seal["candidate"]), scientific, "completion candidate differs")
        return value

    def pending_reference(self):
        if self.pending is None:
            return None
        relative, value = self.pending
        if self.pending_candidate is None:
            raise ValueError("completion must be staged before successful finish")
        return {"path": relative, "sha256": hashlib.sha256(encoded(value)).hexdigest(),
                "candidate": self.pending_candidate}

    def stage_completion(self, attempt_path):
        if self.pending is None:
            return
        relative, value = self.pending
        raw = encoded(value)
        self.attempt.check(additional_bytes=len(raw))
        self.pending_candidate = self.store.publish(attempt_path+"/completion_candidate.json", raw)

    def publish_completion(self, finish_ref):
        """Final commit point, only after post-check and successful finish."""
        if self.pending is None:
            return
        finish = self.store.json(finish_ref)
        if finish["status"] != "COMPLETE":
            raise ValueError("cannot publish completion after a failed operator")
        equal(finish["completion"], self.pending_reference(), "finish does not seal the pending result")
        equal(self.store.json(self.pending_candidate), self.pending[1], "staged completion differs")
        relative, value = self.pending
        expected_operation = "run" if relative == "complete.json" else "replay"
        equal(finish["start"]["manifest"], self.manifest_ref, "finish belongs to another manifest")
        if finish["start"]["operation"] != expected_operation:
            raise ValueError("finish belongs to another operation")
        self.store.publish_json(relative, {**value, "attempt_finish": finish_ref})
        self.pending = None
        self.pending_candidate = None

    def publish_record(self, relative, value, *, publisher=None):
        data = {"manifest": self.manifest_ref, **value}
        self.attempt.check(additional_bytes=len(encoded(data)))
        return (self.store if publisher is None else publisher).publish_json(relative, data)

    def inventory(self):
        receipt = self.record("inventory/complete.json")
        if receipt["status"] != "INVENTORY_AUTHENTICATED":
            raise ValueError("inventory receipt is not complete")
        inventory = self.store.json(receipt["inventory"])
        equal(inventory["completion"], COMPLETION, "inventory belongs to another campaign")
        if inventory["scene_count"] != len(TESTS)*len(SCENE_IDS) or set(inventory["records"]) != set(TESTS):
            raise ValueError("inventory does not contain the fixed roster")
        for split, rows in inventory["records"].items():
            if [r["scene_id"] for r in rows] != list(SCENE_IDS):
                raise ValueError(f"inventory scene roster differs: {split}")
        return inventory

    def prepare(self):
        if self.store.path("inventory/complete.json").exists():
            return {"status": "INVENTORY_REUSED", "scene_count": self.inventory()["scene_count"]}
        if self.store.path("inventory").exists():
            raise ValueError("orphaned inventory cannot be adopted")
        corpus = ClosedCorpus(AuthenticatedReader(self.store.project))
        self.attempt.check()
        inventory = corpus.inventory()
        self.attempt.check(additional_bytes=len(encoded(inventory)))
        with self.store.transaction("inventory") as publication:
            ref = publication.publish_json("inventory.json", inventory)
            self.publish_record("complete.json", {"status": "INVENTORY_AUTHENTICATED", "inventory": ref},
                                publisher=publication)
        return {"status": "INVENTORY_AUTHENTICATED", "scene_count": inventory["scene_count"]}

    def load_unit(self, split, scene_id):
        relative = f"scenes/{split}/{scene_id:05d}"
        record = self.record(relative+"/receipt.json")
        if record["status"] != "SCENE_COMPLETE" or record["split"] != split or record["scene_id"] != scene_id:
            raise ValueError("scene receipt identity differs")
        ref = record["bundle"]
        raw = self.store.read({k: ref[k] for k in ("path", "sha256", "bytes")})
        payload = load_bundle(raw, ref)
        extracted = payload["extracted"]
        if extracted["split"] != split or extracted["scene_id"] != scene_id:
            raise ValueError("scientific payload scene identity differs")
        return payload, record, raw

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
        result = diagnose_scene(extracted["compact"], extracted["canonical_labels"], extracted["predictions"])
        coverage = verify_scene_result(extracted, result)
        payload = {"extracted": extracted, "result": result, "coverage": coverage}
        raw, receipt = bundle_bytes(payload)
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
            record = {"status": "SCENE_COMPLETE", "split": split, "scene_id": scene_id,
                      "phase": phase, "bundle": ref, "timing": timing}
            self.publish_record("receipt.json", record, publisher=publication)
        return payload, {"manifest": self.manifest_ref, **record}

    def profile(self):
        if self.store.path("profile.complete.json").exists():
            record = self.record("profile.complete.json")
            for split in TESTS:
                self.load_unit(split, 0)
            return {"status": "PROFILE_REUSED", "forecast": record["forecast"]}
        inventory = self.inventory()
        corpus = ClosedCorpus(AuthenticatedReader(self.store.project))
        corpus.receipts = inventory["records"]
        profiles = []
        for split in TESTS:
            started = time.monotonic()
            loaded = corpus.load_split(split)
            setup = time.monotonic()-started
            _, record = self.unit(corpus, loaded, 0, phase="profile", setup_seconds=setup)
            profiles.append(record["timing"])
            del loaded
        charged = self.attempt.charged_before+(time.monotonic()-self.attempt.started)
        projection = forecast(profiles, inventory, charged_seconds=charged)
        self.publish_record("profile.complete.json", {"status": "PROFILE_COMPLETE", "profiles": profiles,
                            "forecast": projection, "roster_authorized": projection["time_fits"] and projection["outputs_fit"]})
        return {"status": "PROFILE_COMPLETE", "forecast": projection}

    def run(self):
        profile = self.record("profile.complete.json")
        inventory = self.inventory()
        # Re-check remaining total after every intervening attempt, not only the
        # original forecast's old charged time. No roster reduction on failure.
        projection = forecast(profile["profiles"], inventory, charged_seconds=self.attempt.charged_before)
        if not projection["time_fits"] or not projection["outputs_fit"]:
            raise Paused("profile forecast cannot fit the fixed budget; explicit cost review required")
        if self.store.path("complete.json").exists():
            record = self.record("complete.json")
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
        self.validate_complete({"manifest": self.manifest_ref, **record})
        self.pending = ("complete.json", {"manifest": self.manifest_ref, **record})
        return {"status": "ROSTER_READY_FOR_COMMIT", "scene_count": sum(counts.values())}

    def validate_complete_header(self, record):
        equal(record["manifest"], self.manifest_ref, "complete record manifest differs")
        equal(record["scene_counts"], {split: len(SCENE_IDS) for split in TESTS}, "complete scene roster differs")
        if (record["status"] != "COMPLETE" or set(record["summaries"]) != set(TESTS)
                or set(record["scenes"]) != set(TESTS)):
            raise ValueError("complete record schema differs")
        for split in TESTS:
            if len(record["scenes"][split]) != len(SCENE_IDS):
                raise ValueError("complete scene receipt extent differs")
            for scene_id, ref in zip(SCENE_IDS, record["scenes"][split]):
                if ref["path"] != f"scenes/{split}/{scene_id:05d}/receipt.json":
                    raise ValueError("complete scene receipt identity differs")

    def validate_complete(self, record):
        self.validate_complete_header(record)
        for split in TESTS:
            self.store.json(record["summaries"][split])
            for scene_id, ref in zip(SCENE_IDS, record["scenes"][split]):
                self.attempt.check()
                authoritative = self.store.json(ref)
                _, local, _ = self.load_unit(split, scene_id)
                equal(authoritative, local, "complete scene receipt differs from recorded prefix")

    def replay(self):
        complete = self.record("complete.json")
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
                result = diagnose_scene(extracted["compact"], extracted["canonical_labels"], extracted["predictions"])
                coverage = verify_scene_result(extracted, result)
                repeated, _ = bundle_bytes({"extracted": extracted, "result": result, "coverage": coverage})
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
        if self.store.path("replayed.json").exists():
            previous = self.record("replayed.json")
            equal({k: v for k, v in previous.items() if k != "attempt_finish"},
                  {"manifest": self.manifest_ref, **record}, "old replay record differs")
            return {"status": "REPLAYED_REUSED", "scene_count": count}
        else:
            self.pending = ("replayed.json", {"manifest": self.manifest_ref, **record})
        return {"status": "REPLAY_READY_FOR_COMMIT", "scene_count": count}


def recover_completion(store, operation):
    """Finish a sealed commit, not a new operator or a refund of its budget.

    The previous successful operator already authenticated the full roster.
    Only its exact durable candidate may be published; no scientific work is
    repeated and no PAUSED/FAILED/unobserved attempt is promoted.
    """
    if operation not in ("run", "replay"):
        return None
    marker = "complete.json" if operation == "run" else "replayed.json"
    if store.path(marker).exists():
        return None
    validator = DiagnosticRunner(store, None)
    for path in sorted((store.root/"attempts").glob("*/finish.json"), reverse=True):
        relative = path.relative_to(store.root).as_posix()
        raw = store.path(relative).read_bytes()
        ref = {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        finish = store.json(ref)
        if finish["status"] != "COMPLETE" or finish["start"]["operation"] != operation or "completion" not in finish:
            continue
        equal(finish["start"]["manifest"], validator.manifest_ref, "recovery manifest differs")
        equal(AttemptBudget._local_json(path.parent/"start.json"), finish["start"], "recovery start differs")
        seal = finish["completion"]
        if seal["path"] != marker or seal["candidate"]["path"] != path.parent.relative_to(store.root).as_posix()+"/completion_candidate.json":
            raise ValueError("recovery candidate identity differs")
        value = store.json(seal["candidate"])
        if hashlib.sha256(encoded(value)).hexdigest() != seal["sha256"]:
            raise ValueError("recovery candidate differs from successful seal")
        equal(value["manifest"], validator.manifest_ref, "candidate manifest differs")
        if operation == "run":
            validator.validate_complete_header(value)
        else:
            validator.validate_complete_header(validator.record("complete.json"))
            if value["status"] != "REPLAYED" or value["scene_count"] != len(TESTS)*len(SCENE_IDS):
                raise ValueError("recovery replay extent or status differs")
        validator.pending, validator.pending_candidate = (marker, value), seal["candidate"]
        validator.publish_completion(ref)
        return {"status": "COMPLETE_RECOVERED" if operation == "run" else "REPLAYED_RECOVERED",
                "finish": ref, "scene_count": len(TESTS)*len(SCENE_IDS)}
    return None


def execute(operation, *, project=ROOT, review_ref=None):
    """Production path: fixed roster/resources; no test-size or budget override."""
    project = Path(project)
    if SCENE_IDS != tuple(range(512)) or len(TESTS) != 4:
        raise ValueError("production execution cannot use a reduced fixture roster")
    with exclusive_lock(project):
        store = DiagnosticStore(project/OUTPUT, project_root=project)
        if not store.root.exists():
            if operation != "prepare" or review_ref is None:
                raise ValueError("initialize with prepare and an independently reviewed snapshot")
            store.initialize(reviewed_manifest(project, review_ref))
        manifest, _ = store.manifest()
        expected = reviewed_manifest(project, manifest["review"])
        equal(manifest, expected, "operational manifest or source snapshot differs")
        recovered = recover_completion(store, operation)
        if recovered is not None:
            print(json.dumps({"operation": operation, "committed": recovered}), flush=True)
            return recovered
        attempt = AttemptBudget(store, operation)
        runner = DiagnosticRunner(store, attempt)
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        def pause(signum, frame):
            raise Paused(f"operator interrupted by signal {signum}")
        def timeout(signum, frame):
            raise BudgetExceeded("operator hard time guard reached")
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, pause)
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, max(.001, attempt.allocation-(time.monotonic()-attempt.started)))
        status, value = "FAILED", None
        try:
            value = getattr(runner, operation)()
            runner.stage_completion(attempt.relative)
            attempt.check()
            status = "COMPLETE"
        except BudgetExceeded:
            status = "BUDGET_EXHAUSTED"
            raise
        except Paused:
            status = "PAUSED"
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            terminal = attempt.finish(status, completion=runner.pending_reference() if status == "COMPLETE" else None)
            print(json.dumps({"operation": operation, "terminal": terminal, "result": value}), flush=True)
        if runner.pending is not None:
            finish_raw = encoded(terminal)
            runner.publish_completion({"path": attempt.relative+"/finish.json", "bytes": len(finish_raw),
                                       "sha256": hashlib.sha256(finish_raw).hexdigest()})
            value = {**value, "status": "COMPLETE" if operation == "run" else "REPLAYED"}
            print(json.dumps({"operation": operation, "committed": value}), flush=True)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("sources", "prepare", "profile", "run", "replay"))
    parser.add_argument("--review-path")
    parser.add_argument("--review-sha256")
    args = parser.parse_args()
    if args.operation == "sources":
        print(encoded(source_snapshot(ROOT)).decode(), end="")
        return
    if (args.review_path is None) != (args.review_sha256 is None):
        parser.error("review path and SHA256 must be supplied together")
    ref = None if args.review_path is None else {"path": args.review_path, "sha256": args.review_sha256}
    try:
        execute(args.operation, review_ref=ref)
    except Paused as exc:
        print(str(exc), flush=True)
        raise SystemExit(75)
    except BudgetExceeded as exc:
        print(str(exc), flush=True)
        raise SystemExit(76)


if __name__ == "__main__":
    main()
