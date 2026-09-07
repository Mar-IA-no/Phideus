"""Frozen, opened-pool CPU diagnostic. No training, torch, CUDA or new draw."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from atencion_armonica.energy_partition import (
    accessible_amplitudes, canonical_partition, solve_energy_partition, valid_partition,
)

SEEDS = (2026090701, 2026090702)
PRECISIONS = {"pool_float64": 1e-10, "observed_logamp_float32": 1e-6}
GAINS = (0.70, 1.10, 1.40)
POOL = ROOT / "data/atencion_armonica/final_pool"
OUTPUT = ROOT / "data/atencion_armonica/geometry_energy_audit_v1"
REPLAY = ROOT / "data/atencion_armonica/geometry_energy_audit_v1_replay"
SOURCE_PATHS = (
    "experiments/atencion_armonica/PLAN_GEOMETRIC_RESEARCH_ACTION.md",
    "experiments/atencion_armonica/run_energy_partition_audit.py",
    "src/atencion_armonica/energy_partition.py",
    "src/atencion_armonica/peak_tokens.py",
    "src/atencion_armonica/harmonic_synth.py",
)


def encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_new(path: Path, raw: bytes):
    with path.open("xb") as handle:
        handle.write(raw)


def select_records(pool: Path):
    meta_raw = (pool / "pool_meta.json").read_bytes()
    meta = json.loads(meta_raw)
    expected_cells = {f"poly{k}_{regime}" for k in (1, 2, 3) for regime in ("easy", "hard")}
    if set(meta["cell_ranges"]) != expected_cells:
        raise ValueError("unexpected metadata cells")
    wanted = {}
    for cell, (start, stop) in sorted(meta["cell_ranges"].items()):
        if type(start) is not int or type(stop) is not int or stop-start < 4:
            raise ValueError("invalid cell range")
        for mixture_id in range(start, start+4):
            if mixture_id in wanted:
                raise ValueError("overlapping selected IDs")
            wanted[mixture_id] = cell
    selected = {}
    file_hash = hashlib.sha256()
    with (pool / "mixtures.jsonl").open("rb") as handle:
        for line in handle:
            file_hash.update(line)
            record = json.loads(line)
            mid = record["mixture_id"]
            if mid not in wanted:
                continue
            if type(mid) is not int or mid in selected:
                raise ValueError("duplicate or invalid selected ID")
            cell = f"poly{record['polyphony']}_{record['regime']}"
            if cell != wanted[mid]:
                raise ValueError("selected ID disagrees with declared cell")
            peaks = record["peaks"]
            if not 4 <= len(peaks) <= 24:
                raise ValueError("selected record size outside protocol")
            source_ids = [p["source_id"] for p in peaks]
            if any(type(s) is not int for s in source_ids):
                raise ValueError("noninteger source ID")
            counts = Counter(source_ids)
            if set(counts) != set(range(record["polyphony"])) or any(not 4 <= n <= 8 for n in counts.values()):
                raise ValueError("selected truth outside generator contract")
            a = np.array([p["amp"] for p in peaks], dtype=np.float64)
            if not np.all(np.isfinite(a)) or np.any(a <= 0):
                raise ValueError("invalid selected amplitudes")
            selected[mid] = (record, line)
    if set(selected) != set(wanted):
        raise ValueError("missing selected IDs")
    return selected, {"metadata_sha256": digest(meta_raw),
                      "pool_sha256": file_hash.hexdigest(), "selected_cells": wanted}


def infer(amplitudes, mixture_id: int, precision: str, seed: int):
    """Producer passes only decoded/permuted amplitudes to the pure solver."""
    observed = (accessible_amplitudes(amplitudes)
                if precision == "observed_logamp_float32" else np.array(amplitudes, copy=True))
    perm = np.random.default_rng(np.random.SeedSequence([seed, mixture_id])).permutation(len(observed))
    delivered = observed[perm]
    result = solve_energy_partition(delivered, tolerance=PRECISIONS[precision])
    restored = [canonical_partition(perm[block] for block in blocks)
                for blocks in result["solutions"]]
    valid = [valid_partition(delivered, blocks, result["k"], PRECISIONS[precision])
             for blocks in result["solutions"]]
    if not all(valid) or len(set(restored)) != len(restored):
        raise AssertionError("invalid or duplicate witnesses")
    if result["status"] == "MULTIPLE" and len(restored) != 2:
        raise AssertionError("MULTIPLE without two witnesses")
    return {"seed": seed, "permutation": perm.tolist(),
            "delivered_amplitudes": delivered.tolist(), "solver": result,
            "restored_partitions": restored, "witness_valid": valid}


def evaluate(prediction: dict, source_ids):
    # Separate post-prediction supervision; never select a witness with truth.
    truth = canonical_partition(np.flatnonzero(np.asarray(source_ids) == s)
                                for s in sorted(set(source_ids)))
    matches = [canonical_partition(p) == truth for p in prediction["restored_partitions"]]
    status = prediction["solver"]["status"]
    return {"truth_partition": truth, "witness_exact_match": matches,
            "truth_in_returned_witnesses": any(matches),
            "unique_exact_match": matches[0] if status == "UNIQUE" else None}


def equivariance(first: dict, second: dict) -> str:
    a, b = first["solver"], second["solver"]
    if a["status"].startswith("LIMIT") or b["status"].startswith("LIMIT"):
        return "NOT_EVALUABLE_LIMIT"
    if a["status"] != b["status"] or a["k"] != b["k"]:
        return "FAIL"
    if a["status"] == "UNIQUE":
        return "PASS" if first["restored_partitions"] == second["restored_partitions"] else "FAIL"
    # MULTIPLE witnesses already validated; their truncated subsets can differ.
    return "PASS"


def summarize(rows):
    summary = []
    for view in ("original", "source_gain"):
        for precision in PRECISIONS:
            for polyphony in (1, 2, 3):
                group = [r for r in rows if (r["view"], r["precision"], r["polyphony"])
                         == (view, precision, polyphony)]
                if len(group) != 8:
                    raise AssertionError("wrong denominator")
                statuses = Counter(r["predictions"][0]["solver"]["status"] for r in group)
                exact = sum(r["evaluations"][0]["unique_exact_match"] is True for r in group)
                unique = statuses["UNIQUE"]
                decided = sum(statuses[s] for s in ("UNIQUE", "MULTIPLE", "NO_PARTITION", "PRIOR_VIOLATION"))
                summary.append({"view": view, "precision": precision, "polyphony": polyphony,
                                "n": len(group), "status_counts": dict(statuses),
                                "unique_count": unique, "unique_correct": exact,
                                "unique_wrong": unique-exact, "unique_coverage": unique/len(group),
                                "unique_correct_over_all": exact/len(group),
                                "decided_coverage": decided/len(group),
                                "equivariance_counts": dict(Counter(r["equivariance"] for r in group))})
    return summary


def run(output: Path, pool: Path = POOL):
    start = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    try:
        source_hashes = {p: digest((ROOT/p).read_bytes()) for p in SOURCE_PATHS}
        selected, provenance = select_records(pool)
        raw_selected = b"".join(selected[mid][1] for mid in sorted(selected))
        write_new(output / "selected_records.jsonl", raw_selected)
        rows = []
        for mid, (record, raw_line) in sorted(selected.items()):
            labels = [p["source_id"] for p in record["peaks"]]
            original = np.array([p["amp"] for p in record["peaks"]])
            for view in ("original", "source_gain"):
                gains = np.array([1.0 if view == "original" else GAINS[s] for s in labels])
                amplitudes = original*gains
                for precision in PRECISIONS:
                    predictions = [infer(amplitudes, mid, precision, seed) for seed in SEEDS]
                    # Both predictions closed before consulting labels for scoring.
                    evaluations = [evaluate(p, labels) for p in predictions]
                    rows.append({"mixture_id": mid, "polyphony": record["polyphony"],
                                 "regime": record["regime"], "selected_line_sha256": digest(raw_line),
                                 "view": view, "gain_factors_per_peak": gains.tolist(),
                                 "precision": precision, "tolerance": PRECISIONS[precision],
                                 "predictions": predictions, "evaluations": evaluations,
                                 "equivariance": equivariance(*predictions)})
        write_new(output / "predictions.json", encoded(rows))
        report = {"scope": "opened_historical_nonrandom_24_records_not_neural_attribution",
                  "primary_precision": "observed_logamp_float32", "primary_seed": SEEDS[0],
                  "summary": summarize(rows), "gpu_used_or_queried": False}
        write_new(output / "summary.json", encoded(report))
        if source_hashes != {p: digest((ROOT/p).read_bytes()) for p in SOURCE_PATHS}:
            raise RuntimeError("source changed during execution")
        manifest = {"schema_version": 1, "provenance": provenance,
                    "source_sha256": source_hashes,
                    "config": {"seeds": SEEDS, "precisions": PRECISIONS, "gains": GAINS,
                               "max_candidates": 20000, "max_nodes": 50000},
                    "python_version": sys.version, "numpy_version": np.__version__,
                    "artifacts_sha256": {p.name: digest(p.read_bytes()) for p in sorted(output.iterdir())}}
        write_new(output / "manifest.json", encoded(manifest))
        runtime = {"seconds": time.monotonic()-start,
                   "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   "output_path": str(output), "torch_imported": "torch" in sys.modules}
        runtime["scientific_bytes"] = sum(p.stat().st_size for p in output.iterdir())
        runtime["status"] = ("COMPLETE" if runtime["peak_rss_bytes"] < 256*1024**2
                             and runtime["scientific_bytes"]+4096 < 10*1024**2
                             and not runtime["torch_imported"] else "INCOMPLETE_RESOURCE")
        write_new(output / "runtime.json", encoded(runtime))
        if runtime["status"] != "COMPLETE":
            raise RuntimeError("resource contract not met")
        print(json.dumps(runtime, sort_keys=True), flush=True)
    except BaseException as exc:
        write_new(output / "FAILURE.json", encoded({"status": "INCOMPLETE", "error_type": type(exc).__name__,
                                                   "message": str(exc)}))
        raise


def campaign(primary: Path, replay: Path):
    if primary.exists() or replay.exists() or primary.resolve() == replay.resolve():
        raise FileExistsError("campaign outputs must be new and distinct")
    start = time.monotonic()
    for path in (primary, replay):
        subprocess.run([sys.executable, __file__, "--output", str(path)], check=True,
                       timeout=max(0.001, 180-(time.monotonic()-start)))
    names = ("selected_records.jsonl", "predictions.json", "summary.json", "manifest.json")
    comparison = {name: digest((primary/name).read_bytes()) == digest((replay/name).read_bytes())
                  for name in names}
    receipt = {"status": "PASS" if all(comparison.values()) else "FAIL",
               "files": comparison, "total_seconds": time.monotonic()-start,
               "primary": str(primary), "replay": str(replay)}
    write_new(primary / "replay_comparison.json", encoded(receipt))
    if not all(comparison.values()) or receipt["total_seconds"] > 180:
        raise AssertionError("campaign parity or time budget failed")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--campaign", action="store_true")
    args = parser.parse_args()
    if args.campaign:
        if args.output is not None:
            parser.error("--output and --campaign are exclusive")
        campaign(OUTPUT, REPLAY)
    elif args.output is not None:
        run(args.output)
    else:
        parser.error("choose --campaign or --output")
