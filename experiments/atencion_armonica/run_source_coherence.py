"""Bounded post-hoc CPU diagnostic using only already closed artifacts."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.atencion_armonica.preflight_source_coherence import source_hashes as preflight_sources
from src.atencion_armonica.partial_compatibility_cache import encoded, sha_file
from src.atencion_armonica.source_artifacts import ClosedArtifacts, SPLITS, reader_key
from src.atencion_armonica.source_coherence import GroupFitCache, SourceFitter
from src.atencion_armonica.source_loss_pressure import edge_pressure, evaluate_pressure
from src.atencion_armonica.source_rollup import aggregate_split, pressure_metrics, scene_group_strata

SOURCES = ("src/atencion_armonica/source_artifacts.py", "src/atencion_armonica/source_rollup.py",
           "experiments/atencion_armonica/run_source_coherence.py")
PREFLIGHT = ROOT/"data/atencion_armonica/source_coherence_preflight_v1"


def sources():
    return preflight_sources()|{name: sha_file(ROOT/name) for name in SOURCES}


def write_json(path, obj):
    with path.open("xb") as handle:
        handle.write(encoded(obj))


def guard(started):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if time.monotonic()-started > 300 or rss >= 1024**3 or "torch" in sys.modules:
        raise RuntimeError("diagnostic exceeded CPU 300s/1GiB or imported Torch")
    return rss


def verify_preflight():
    manifest = json.loads((PREFLIGHT/"manifest.json").read_bytes())
    if manifest["status"] != "READY_FOR_AUDITED_RUN" or (PREFLIGHT/"FAILURE.json").exists():
        raise ValueError("preflight did not enable a real run")
    if manifest["source_sha256"] != preflight_sources():
        raise ValueError("preflight source drift")
    for name, digest in manifest["artifacts_sha256"].items():
        if sha_file(PREFLIGHT/name) != digest:
            raise ValueError("preflight artifact drift")
    report = json.loads((PREFLIGHT/"report.json").read_bytes())
    if (report["status"] != "READY_FOR_AUDITED_RUN" or report["projection"]["per_run_seconds"] > 300
            or report["peak_rss_bytes"] >= 1024**3):
        raise ValueError("preflight cost outside contract")
    return sha_file(PREFLIGHT/"manifest.json")


def group_record(cache, split, scene_id, q, members, source_ids, endpoint):
    fit = cache.fit(split, scene_id, q, members)  # Observable-only call finishes before truth evaluation.
    members = sorted(members)
    labels, counts = np.unique(np.asarray(source_ids)[members], return_counts=True)
    triples = list(combinations(members, 3))
    local = [float(endpoint[triple]) for triple in triples]
    return {"members": members, "size": len(members), "purity": int(counts.max())/len(members),
            "category": "pure" if len(labels) == 1 else "mixed", "fit": fit,
            "internal_triples": [list(t) for t in triples], "internal_endpoint_residual_cents": local,
            "internal_endpoint_median_cents": float(np.median(local)) if local else None,
            "internal_endpoint_max_cents": max(local) if local else None,
            "comparison_authority": "ENDPOINT_AND_PROFILED_RMS_ARE_DIFFERENT_ESTIMANDS"}


def run(output):
    started = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError("diagnostic 300s walltime expired")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 300.)
    try:
        if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
            raise ValueError("single-thread CPU environment required")
        source_hashes = sources()
        preflight_hash = verify_preflight()
        artifacts = ClosedArtifacts()
        fitter = SourceFitter()
        cache = GroupFitCache(fitter)
        for size in range(3, 9):
            fitter.prepare(size)
        with (output/"beta_grid.npy").open("xb") as handle:
            np.save(handle, fitter.grid, allow_pickle=False)
        summaries, total_groups, total_pressure = {}, 0, 0
        with (output/"groups.jsonl").open("xb") as groups_file, (output/"scenes.jsonl").open("xb") as scenes_file:
            for split in SPLITS:
                observations, records, truths, partitions, logits = artifacts.load_split(split)
                guard(started)
                split_rows = []
                for scene_id, (obs, record, truth) in enumerate(zip(observations, records, truths)):
                    folder = output/split/f"{scene_id:05d}"
                    folder.mkdir(parents=True)
                    q = np.asarray(obs["log_f"], dtype=np.float32)
                    # Save observed inputs apart from explicitly privileged evaluation labels.
                    with (folder/"observed.npz").open("xb") as handle:
                        np.savez_compressed(handle, observed_q32=q, triples=record["triples"],
                            residual_cents=record["residual_cents"], physical_weights=record["weights"],
                            sham_weights=record["sham_weights"], sham_evaluable=record["sham_evaluable"])
                    write_json(folder/"evaluation_truth.json", {"source_ids": truth["source_ids"],
                        "authority": "PRIVILEGED_EVALUATION_ONLY", "scene_id": scene_id, "split_seed": obs["split_seed"]})
                    endpoint = {tuple(t): r for t, r in zip(record["triples"].tolist(), record["residual_cents"])}
                    reader_groups = [(reader_key(r), partitions[reader_key(r)][scene_id]) for r in artifacts.readers]
                    reader_groups.append(("privileged_truth_groups", [[i for i, s in enumerate(truth["source_ids"]) if s == label]
                                         for label in sorted(set(truth["source_ids"]))]))
                    for key, partition in reader_groups:
                        groups = [group_record(cache, split, scene_id, q, members, truth["source_ids"], endpoint)
                                  for members in partition]
                        authority = "PRIVILEGED_TRUE_MEMBERSHIP_REFERENCE" if key == "privileged_truth_groups" else "FROZEN_PREDICTED_PARTITION"
                        for group_id, group in enumerate(groups):
                            groups_file.write(encoded({"split": split, "scene_id": scene_id, "reader": key,
                                "group_id": group_id, "membership_authority": authority, **group}))
                        total_groups += len(groups)
                        pressure = None
                        if key in logits:
                            raw = edge_pressure(logits[key][scene_id], record["triples"], record["weights"],
                                                record["sham_weights"], sham_evaluable=bool(record["sham_evaluable"]))
                            pressure, evaluation = evaluate_pressure(raw, truth["source_ids"])
                            with (folder/f"{key}.npz").open("xb") as handle:
                                np.savez_compressed(handle, raw_logits=logits[key][scene_id], **raw, **evaluation)
                            total_pressure += 1
                        row = {"split": split, "scene_id": scene_id, "split_seed": obs["split_seed"], "reader": key,
                            "membership_authority": authority, "partition": partition, "n_events": len(q),
                            "strata": scene_group_strata(groups), "pressure": pressure,
                            "pressure_metrics": pressure_metrics(pressure) if pressure is not None else None}
                        scenes_file.write(encoded(row))
                        split_rows.append(row)
                        guard(started)
                summaries[split] = aggregate_split(split_rows, artifacts.readers)
                summaries[split]["sample_authority"] = "READER_SELECTION_IN_SAMPLE" if split == "validation" else "OPEN_TEST_POSTHOC"
        if total_pressure != 1152:
            raise ValueError("incomplete pressure roster")
        write_json(output/"summary.json", {"splits": summaries, "total_groups": total_groups,
            "total_pressure_records": total_pressure, "cache_hits": cache.hits, "cache_misses": cache.misses,
            "index_authority": "WITNESS_NOT_IDENTIFIED", "continuous_minimum": "NOT_COMPUTED",
            "global_ambiguity_status": "UNADJUDICATED", "causal_training_claim": "NOT_IDENTIFIED"})
        write_json(output/"config.json", {"source_sha256": source_hashes, "input_sha256": artifacts.inputs,
            "preflight_manifest_sha256": preflight_hash, "readers": artifacts.readers, "scene_ids": list(range(32)),
            "splits": list(SPLITS), "grid_sha256": fitter.grid_hash, "grid_size": 1025, "coarse_stride": 4,
            "lambda": .1, "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "compute": "SINGLE_THREAD_CPU_NUMPY_ONLY", "new_draws_forwards_training": False})
        for name, digest in artifacts.inputs.items():
            if sha_file(ROOT/name) != digest:
                raise ValueError("input changed during diagnostic")
        if sources() != source_hashes:
            raise ValueError("diagnostic source drift")
        scientific = {str(p.relative_to(output)): sha_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
        write_json(output/"resources.json", {"seconds": time.monotonic()-started, "peak_rss_bytes": guard(started)})
        write_json(output/"manifest.json", {"status": "COMPLETE_POSTHOC_NOT_ADJUDICATED", "scientific_sha256": scientific,
            "resources_sha256": sha_file(output/"resources.json"), "source_sha256": source_hashes})
        guard(started)
        print(json.dumps({"status": "COMPLETE_POSTHOC_NOT_ADJUDICATED", "groups": total_groups,
                          "pressure": total_pressure, "seconds": time.monotonic()-started}), flush=True)
    except BaseException as exc:
        if not (output/"FAILURE.json").exists():
            write_json(output/"FAILURE.json", {"status": "INCOMPLETE", "error": repr(exc)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous)


def supervised_run(output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        subprocess.run([sys.executable, __file__, "--output", str(output), "--worker"], check=True, timeout=305)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        if not (output/"FAILURE.json").exists():
            write_json(output/"FAILURE.json", {"status": "INCOMPLETE", "error": repr(exc), "reported_by": "SUPERVISOR"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run(args.output) if args.worker else supervised_run(args.output)
