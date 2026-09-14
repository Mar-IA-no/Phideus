"""Reviewed four-compact timing operator; no production roster or source decode."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.atencion_armonica import operator_objective_runner as base
from src.atencion_armonica import operator_diagnostic_cached as cached

FILES = (
    "src/atencion_armonica/operator_diagnostic_cached.py",
    "experiments/atencion_armonica/test_operator_diagnostic_cached.py",
    "experiments/atencion_armonica/profile_cached_diagnostic.py",
    "experiments/atencion_armonica/PLAN_OPERATOR_OBJECTIVE_COST_REVISION.md",
    "experiments/atencion_armonica/AMENDMENT_OPERATOR_OBJECTIVE_CACHE_PROFILE.md",
)


def reference(root, relative):
    path = Path(root).joinpath(*base.AuthenticatedReader.parts(relative))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("profile reference cannot traverse symlinks")
    raw = path.read_bytes()
    return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def snapshot(store):
    manifest, manifest_ref = store.manifest()
    base.equal(manifest, base.reviewed_manifest(store.project, manifest["review"]), "v1 review changed")
    return {"manifest": manifest_ref,
            "base_profile": reference(store.project, base.OUTPUT+"/profile.complete.json"),
            "files": [reference(store.project, relative) for relative in FILES]}


def accept_review(store, review_ref):
    reader = base.AuthenticatedReader(store.project)
    review = reader.json(review_ref)
    if review.get("status") != "CACHE_PROFILE_REVIEW_COMPLETE" or not review.get("reports"):
        raise ValueError("cache profile implementation review is not complete")
    base.equal(review["snapshot"], snapshot(store), "cache profile review does not cover current sources")
    for ref in review["reports"]:
        reader.bytes(ref)
    return review


def previous_profile(store, current_review):
    """First successful fixed-snapshot profile only; never select faster repeats."""
    reader = base.AuthenticatedReader(store.project)
    _, manifest_ref = store.manifest()
    for path in sorted((store.root/"attempts").glob("*/finish.json")):
        finish = store.json(reference(store.root, path.relative_to(store.root).as_posix()))
        completion = finish.get("completion", {})
        if (finish["status"] != "COMPLETE" or finish["start"]["operation"] != "profile"
                or not {"report", "runtime_revision"} <= set(completion)):
            continue
        revision = store.json(completion["runtime_revision"])
        old_review = reader.json(revision["review"])
        if old_review["snapshot"] != current_review["snapshot"]:
            continue
        if old_review.get("status") != "CACHE_PROFILE_REVIEW_COMPLETE" or not old_review.get("reports"):
            raise ValueError("previous profile review was not complete")
        for ref in old_review["reports"]:
            reader.bytes(ref)
        base.equal(revision["manifest"], manifest_ref, "previous profile manifest differs")
        base.equal(store.json(revision["start"]), finish["start"], "previous profile start differs")
        base.equal(finish["start"]["manifest"], manifest_ref, "previous operator manifest differs")
        report = store.json(completion["report"])
        base.equal(report["runtime_revision"], completion["runtime_revision"], "previous runtime revision differs")
        base.equal(report["manifest"], manifest_ref, "previous report manifest differs")
        if (report["status"] != "CACHE_PROFILE_CANDIDATE"
                or [r["split"] for r in report["profiles"]] != list(base.TESTS)
                or any(r["scene_id"] != 0 or r["scientific_bytes_equal"] is not True for r in report["profiles"])):
            raise ValueError("previous successful profile extent differs")
        for row in report["profiles"]:
            store.read({k: row["bundle"][k] for k in ("path", "sha256", "bytes")})
        return report
    return None


def execute(review_ref, *, project=ROOT):
    project = Path(project)
    with base.exclusive_lock(project):
        store = base.DiagnosticStore(project/base.OUTPUT, project_root=project)
        review = accept_review(store, review_ref)
        previous_report = previous_profile(store, review)
        if previous_report is not None:
            print(json.dumps({"operation": "cache_profile", "status": "FIRST_SUCCESS_REUSED"}), flush=True)
            return previous_report
        attempt = base.AttemptBudget(store, "profile")
        diagnostic = base.DiagnosticRunner(store, attempt)
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
        def interrupted(signum, frame):
            raise base.Paused(f"cache profile interrupted by signal {signum}")
        def timeout(signum, frame):
            raise base.BudgetExceeded("cache profile cumulative time guard reached")
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, interrupted)
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, max(.001, attempt.allocation-(time.monotonic()-attempt.started)))
        status, sealed, report = "FAILED", None, None
        def publish(relative, value):
            raw = base.encoded(value)
            attempt.check(additional_bytes=len(raw))
            return store.publish(attempt.relative+"/"+relative, raw)
        try:
            revision = publish("runtime_revision.json", {"manifest": diagnostic.manifest_ref,
                "start": reference(store.root, attempt.relative+"/start.json"), "review": review_ref})
            baseline = diagnostic.record("profile.complete.json")
            if baseline["status"] != "PROFILE_COMPLETE":
                raise ValueError("cache timing requires the completed original profile")
            old = {row["split"]: row for row in baseline["profiles"]}
            if set(old) != set(base.TESTS) or len(baseline["profiles"]) != 4:
                raise ValueError("baseline profile extent differs")
            inventory = diagnostic.inventory()
            rows = []
            for split in base.TESTS:
                attempt.check()
                payload, unit, original = diagnostic.load_unit(split, 0)
                if old[split]["scene_id"] != 0 or unit["scene_id"] != 0:
                    raise ValueError("profile scene selection differs")
                extracted = payload["extracted"]
                started = time.monotonic()
                result = cached.diagnose_scene(extracted["compact"], extracted["canonical_labels"], extracted["predictions"])
                calculated = time.monotonic()
                coverage = base.verify_scene_result(extracted, result)
                verified = time.monotonic()
                repeated, codec = base.bundle_bytes({"extracted": extracted, "result": result, "coverage": coverage})
                encoded_at = time.monotonic()
                if repeated != original:
                    raise ValueError("cache profile differs from frozen scientific bytes")
                attempt.check(additional_bytes=len(repeated))
                bundle = store.publish(attempt.relative+f"/bundles/{split}.gz", repeated)
                measured = time.monotonic()-started
                if result["candidate_count"] != old[split]["candidate_count"]:
                    raise ValueError("cache profile candidate extent differs")
                rows.append({**old[split], "diagnostic_seconds": measured, "bundle_bytes": len(repeated),
                             "calculation_seconds": calculated-started, "cotejo_seconds": verified-calculated,
                             "codec_seconds": encoded_at-verified,
                             "publication_and_guards_seconds": started+measured-encoded_at,
                             "bundle": {**bundle, **codec}, "scientific_bytes_equal": True})
            charged = attempt.charged_before+(time.monotonic()-attempt.started)
            projection = base.forecast(rows, inventory, charged_seconds=charged)
            report = {"status": "CACHE_PROFILE_CANDIDATE", "manifest": diagnostic.manifest_ref,
                      "runtime_revision": revision, "profiles": rows, "forecast": projection,
                      "authority": "TIMING_ONLY_NOT_ROSTER_AUTHORIZATION"}
            report_ref = publish("cache_profile.json", report)
            attempt.check()
            sealed = {"report": report_ref, "runtime_revision": revision}
            status = "COMPLETE"
        except base.BudgetExceeded:
            status = "BUDGET_EXHAUSTED"
            raise
        except base.Paused:
            status = "PAUSED"
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            terminal = attempt.finish(status, completion=sealed if status == "COMPLETE" else None)
            print(json.dumps({"operation": "cache_profile", "terminal": terminal}), flush=True)
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
        parser.error("a reviewed snapshot path and SHA256 are required")
    try:
        report = execute({"path": args.review_path, "sha256": args.review_sha256})
        print(json.dumps({"forecast": report["forecast"], "scene_count": len(report["profiles"])}), flush=True)
    except base.Paused as exc:
        print(str(exc), flush=True)
        raise SystemExit(75)
    except base.BudgetExceeded as exc:
        print(str(exc), flush=True)
        raise SystemExit(76)


if __name__ == "__main__":
    main()
