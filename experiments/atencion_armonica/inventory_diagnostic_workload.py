"""Authenticated observed workload only; no factors, truth reconstruction or scores."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.atencion_armonica import profile_fast_diagnostic as prior
from src.atencion_armonica.operator_objective_corpus import CHECKPOINTS, offsets

base = prior.base
FILES = ("experiments/atencion_armonica/inventory_diagnostic_workload.py",
         "experiments/atencion_armonica/test_inventory_diagnostic_workload.py",
         "experiments/atencion_armonica/PLAN_OPERATOR_OBJECTIVE_WORKLOAD.md")
FAST_REVIEW = {"path": base.WORK+"/fast_profile_review.v1.json",
               "sha256": "467ed2a7a18e593779b6cb9b23eafb863643b9b95b2e86a558b2d16e8d776299"}


def snapshot(store):
    prior.accept_review(store, FAST_REVIEW)
    return {"manifest": store.manifest()[1], "fast_review": FAST_REVIEW,
            "files": [prior.reference(store.project, name) for name in FILES]}


def accept_review(store, ref):
    reader = base.AuthenticatedReader(store.project)
    review = reader.json(ref)
    if review.get("status") != "WORKLOAD_REVIEW_COMPLETE" or not review.get("reports"):
        raise ValueError("workload implementation review incomplete")
    base.equal(review["snapshot"], snapshot(store), "workload sources differ from review")
    for report in review["reports"]:
        reader.bytes(report)
    return review


def scene_workload(split, scene_id, scene, candidate_count):
    if type(candidate_count) is not int or not 0 <= candidate_count <= 82:
        raise ValueError("candidate count outside fixed support")
    if set(scene) != {"observation", "partitions", "identity_sha256"}:
        raise ValueError("observed scene schema differs")
    observation, partitions = scene["observation"], scene["partitions"]
    if (split not in base.TESTS or type(scene_id) is not int or not 0 <= scene_id < 512
            or set(observation) != {"scene_id", "split_seed", "log_f"}
            or type(observation["scene_id"]) is not int or observation["scene_id"] != scene_id
            or type(observation["split_seed"]) is not int or observation["split_seed"] != base.TESTS[split]):
        raise ValueError("observed scene/seed identity differs")
    q = base.np.asarray(observation["log_f"], base.np.float64)
    if (q.ndim != 1 or not 8 <= len(q) <= 32 or not base.np.isfinite(q).all()
            or not base.np.array_equal(q, q.astype(base.np.float32).astype(base.np.float64))):
        raise ValueError("observed event extent or q32 precision differs")
    n = len(q)
    identity = hashlib.sha256(base.encoded({"split": split, "observation": observation,
                                          "partitions": partitions})).hexdigest()
    if identity != scene["identity_sha256"] or len(partitions) != candidate_count:
        raise ValueError("observed candidate identity/count differs")
    canonical, sizes = [], Counter()
    for partition in partitions:
        p = tuple(tuple(g) for g in partition)
        if (not 2 <= len(p) <= 4 or any(not 4 <= len(g) <= 8 for g in p)
                or any(type(i) is not int for g in p for i in g)
                or p != tuple(sorted(tuple(sorted(g)) for g in p))
                or sorted(i for g in p for i in g) != list(range(n))):
            raise ValueError("observed partition is not supported, canonical and complete")
        canonical.append(p)
        sizes[",".join(map(str, sorted(map(len, p))))] += 1
    if canonical != sorted(set(canonical)):
        raise ValueError("observed candidate roster is not sorted and unique")
    return {"scene_id": scene_id, "identity_sha256": identity, "event_count": n,
            "candidate_count": candidate_count, "group_count": sum(map(len, canonical)),
            "sizes_histogram": dict(sorted(sizes.items())),
            "partitions_sha256": hashlib.sha256(base.encoded(partitions)).hexdigest()}


def split_workload(loaded, check):
    split = loaded["split"]
    if set(loaded["inputs"]) != set(CHECKPOINTS):
        raise ValueError("workload requires the three original checkpoints")
    anchor = loaded["inputs"][CHECKPOINTS[0]]
    off = offsets(anchor["arrays"]["candidate_offsets"])
    scenes = anchor["shard"]["scenes"]
    if len(scenes) != 512:
        raise ValueError("workload scene roster differs")
    for cp in CHECKPOINTS:
        item = loaded["inputs"][cp]
        other = offsets(item["arrays"]["candidate_offsets"])
        if not base.np.array_equal(off, other):
            raise ValueError("checkpoint candidate offsets differ")
        base.equal(item["shard"]["scenes"], scenes, "checkpoint observed partitions differ")
    rows = []
    for i in range(512):
        if i % 32 == 0:
            check()
        rows.append(scene_workload(split, i, scenes[i], int(off[i+1]-off[i])))
    counts = [r["candidate_count"] for r in rows]
    return {"split": split, "scene_count": 512, "rows": rows,
            "sources": {str(cp): loaded["inputs"][cp]["meta"]["inputs"] for cp in CHECKPOINTS},
            "offsets_sha256": hashlib.sha256(off.tobytes()).hexdigest(),
            "candidate_histogram": {str(k): v for k, v in sorted(Counter(counts).items())},
            "candidate_count": sum(counts), "candidate_pairs": sum(math.comb(c, 2) for c in counts),
            "pairs_with_floor31": sum(math.comb(max(c, 31), 2) for c in counts),
            "maximum_candidate_count": max(counts),
            "group_count": sum(r["group_count"] for r in rows)}


def collect(store, attempt):
    inventory = base.DiagnosticRunner(store, attempt).inventory()
    corpus = base.ClosedCorpus(base.AuthenticatedReader(store.project))
    corpus.receipts = inventory["records"]
    results = {}
    for split in base.TESTS:
        attempt.check()
        loaded = corpus.load_split(split)
        results[split] = split_workload(loaded, attempt.check)
        results[split]["prediction_seal"] = corpus.completion["tests"][split]["prediction_seal"]
        del loaded
    return {"splits": results, "scene_count": 2048, "candidate_floor": 31,
            "candidate_count": sum(r["candidate_count"] for r in results.values()),
            "candidate_pairs": sum(r["candidate_pairs"] for r in results.values()),
            "pairs_with_floor31": sum(r["pairs_with_floor31"] for r in results.values()),
            "all_scenes_at82_pairs": 2048*math.comb(82, 2),
            "authority": "OBSERVED_WORKLOAD_NOT_ROSTER_AUTHORIZATION"}


def previous(store, review):
    reader = base.AuthenticatedReader(store.project)
    for path in sorted((store.root/"attempts").glob("*/finish.json")):
        finish = store.json(prior.reference(store.root, path.relative_to(store.root).as_posix()))
        seal = finish.get("completion", {})
        if finish["status"] != "COMPLETE" or "workload" not in seal:
            continue
        revision = store.json(seal["runtime_revision"])
        old = reader.json(revision["review"])
        if old["snapshot"] != review["snapshot"]:
            continue
        if old.get("status") != "WORKLOAD_REVIEW_COMPLETE" or not old.get("reports"):
            raise ValueError("previous workload review incomplete")
        for report in old["reports"]:
            reader.bytes(report)
        manifest = store.manifest()[1]
        base.equal(revision["manifest"], manifest, "previous workload manifest differs")
        base.equal(store.json(revision["start"]), finish["start"], "previous workload start differs")
        base.equal(finish["start"]["manifest"], manifest, "previous operator manifest differs")
        report = store.json(seal["workload"])
        base.equal(report["runtime_revision"], seal["runtime_revision"], "workload revision differs")
        base.equal(report["manifest"], manifest, "workload manifest differs")
        if (finish["start"]["operation"] != "profile" or report["status"] != "WORKLOAD_COMPLETE"
                or report["workload"]["scene_count"] != 2048
                or set(report["workload"]["splits"]) != set(base.TESTS)):
            raise ValueError("previous workload extent or status differs")
        return report
    return None


def execute(review_ref, *, project=ROOT):
    with base.exclusive_lock(project):
        store = base.DiagnosticStore(Path(project)/base.OUTPUT, project_root=project)
        review = accept_review(store, review_ref)
        reused = previous(store, review)
        if reused is not None:
            print("FIRST_SUCCESS_REUSED", flush=True)
            return reused
        handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
        attempt = base.AttemptBudget(store, "profile")
        status, seal = "FAILED", None
        try:
            def pause(signum, frame):
                raise base.Paused("workload interrupted")
            def timeout(signum, frame):
                raise base.BudgetExceeded("workload cumulative profile guard reached")
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, pause)
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, max(.001, attempt.allocation-(time.monotonic()-attempt.started)))
            def publish(name, value):
                raw = base.encoded(value)
                attempt.check(additional_bytes=len(raw))
                return store.publish(attempt.relative+"/"+name, raw)
            manifest = store.manifest()[1]
            revision = publish("runtime_revision.json", {"manifest": manifest, "review": review_ref,
                "start": prior.reference(store.root, attempt.relative+"/start.json")})
            workload = collect(store, attempt)
            report = {"status": "WORKLOAD_COMPLETE", "manifest": manifest,
                      "runtime_revision": revision, "workload": workload}
            output = publish("workload.json", report)
            attempt.check()
            seal = {"workload": output, "runtime_revision": revision}
            status = "COMPLETE"
        except base.Paused:
            status = "PAUSED"
            raise
        except base.BudgetExceeded:
            status = "BUDGET_EXHAUSTED"
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            print(json.dumps(attempt.finish(status, completion=seal if status == "COMPLETE" else None)), flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--review-path")
    parser.add_argument("--review-sha256")
    args = parser.parse_args()
    if args.snapshot:
        print(base.encoded(snapshot(base.DiagnosticStore(ROOT/base.OUTPUT, project_root=ROOT))).decode(), end="")
        return
    if not args.review_path or not args.review_sha256:
        parser.error("a reviewed workload snapshot is required")
    try:
        report = execute({"path": args.review_path, "sha256": args.review_sha256})
        print(json.dumps({k: v for k, v in report["workload"].items() if k != "splits"}), flush=True)
    except base.Paused as exc:
        print(str(exc), flush=True)
        raise SystemExit(75)
    except base.BudgetExceeded as exc:
        print(str(exc), flush=True)
        raise SystemExit(76)


if __name__ == "__main__":
    main()
