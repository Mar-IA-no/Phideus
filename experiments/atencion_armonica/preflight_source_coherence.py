"""Mechanical CPU resource gate; no experimental corpus or source truth."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.atencion_armonica.partial_compatibility_cache import encoded, sha_file
from src.atencion_armonica.source_coherence import GroupFitCache, SourceFitter
from src.atencion_armonica.source_loss_pressure import edge_pressure

SOURCES = ("experiments/atencion_armonica/PLAN_SHARED_SOURCE_COHERENCE.md",
           "src/atencion_armonica/source_coherence.py", "src/atencion_armonica/source_loss_pressure.py",
           "experiments/atencion_armonica/preflight_source_coherence.py",
           "src/atencion_armonica/partial_compatibility_cache.py")
MAX_FITS = 13440
MAX_PRESSURE = 96*12


def source_hashes():
    return {name: sha_file(ROOT/name) for name in SOURCES}


def resource_projection(fit_seconds, pressure_seconds, io_seconds, preparation_seconds):
    components = {"fits": MAX_FITS*max(fit_seconds.values()),
                  "pressure": MAX_PRESSURE*pressure_seconds,
                  "io": MAX_FITS*io_seconds, "preparation": preparation_seconds}
    return {"components_seconds": components, "per_run_seconds": sum(components.values()),
            "primary_plus_replay_seconds": 2*sum(components.values()),
            "projected_fit_count": MAX_FITS, "projected_pressure_count": MAX_PRESSURE,
            "status": "ESTIMATE_NOT_GUARANTEED_BOUND"}


def guard(started):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if time.monotonic()-started > 120 or rss >= 1024**3:
        raise RuntimeError("mechanical preflight exceeded 120s or 1GiB")
    return rss


def run(output, *, prepared=False):
    sources = source_hashes()
    if not prepared:
        output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        q = (np.linspace(-1.8, 1.8, 32)+.01*np.sin(np.arange(32))).astype(np.float32)
        fitter = SourceFitter()
        preparation = time.monotonic()
        for size in range(3, 9):
            fitter.prepare(size)
        preparation = time.monotonic()-preparation
        durations, witnesses = {}, {}
        for size in range(3, 9):
            values = []
            for repeat in range(3):
                tick = time.monotonic()
                witness = fitter.fit(q, np.arange(repeat, repeat+size))
                values.append(time.monotonic()-tick)
                witnesses[f"size_{size}_repeat_{repeat}"] = witness
                guard(started)
            durations[str(size)] = max(values)
        partitions = []
        cache = GroupFitCache(fitter)
        group_results = []
        for reader in range(13):
            size = 3+reader % 6
            permutation = np.roll(np.arange(32), reader)
            groups = [sorted(permutation[i:i+size].tolist()) for i in range(0, 32, size)]
            partitions.append(groups)
            group_results.append([cache.fit("mechanical", 0, q, group) for group in groups])
            guard(started)
        if len({tuple(sorted(tuple(g) for g in partition)) for partition in partitions}) != 13:
            raise ValueError("mechanical partitions must be distinct")
        for groups in partitions:  # Deliberate second access verifies the cache path.
            for group in groups:
                cache.fit("mechanical", 0, q, group)
        triples = np.array(list(combinations(range(32), 3)), dtype=np.int64)
        weights = np.linspace(0., 1., len(triples), dtype=np.float32)
        logits = np.cos(q.astype(float)[:, None]-q.astype(float)[None, :])
        pressure_times = []
        for _ in range(3):
            tick = time.monotonic()
            pressure = edge_pressure(logits, triples, weights, weights[::-1], sham_evaluable=True)
            pressure_times.append(time.monotonic()-tick)
            guard(started)
        # Explicit mechanical stand-ins for read + serialization + write overhead.
        payload = {"observation": q.tolist(), "groups": group_results[0],
                   "pressure": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in pressure.items()}}
        io_times = []
        for repeat in range(3):
            tick = time.monotonic()
            path = output/f"io_fixture_{repeat}.json"
            with path.open("xb") as handle:
                handle.write(encoded(payload))
            loaded = json.loads(path.read_text())
            if loaded != payload:
                raise ValueError("mechanical IO roundtrip changed content")
            io_times.append(time.monotonic()-tick)
            guard(started)
        projected = resource_projection(durations, max(pressure_times), max(io_times), preparation)
        report = {"status": projection_status(projected),
                  "fit_max_seconds_by_size": durations, "pressure_max_seconds": max(pressure_times),
                  "io_fixture_max_seconds": max(io_times), "projection": projected,
                  "peak_rss_bytes": guard(started), "seconds": time.monotonic()-started,
                  "cache_hits": cache.hits, "cache_misses": cache.misses,
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                  "grid_version": witnesses["size_3_repeat_0"]["grid_version"], "grid_sha256": fitter.grid_hash,
                  "observed_q32": q.tolist(), "partitions": partitions, "witnesses": witnesses,
                  "source_sha256": sources, "data_authority": "MECHANICAL_FIXTURES_ONLY_NO_TRUTH"}
        with (output/"report.json").open("xb") as handle:
            handle.write(encoded(report))
        with (output/"beta_grid.npy").open("xb") as handle:
            np.save(handle, fitter.grid, allow_pickle=False)
        if sources != source_hashes() or "torch" in sys.modules:
            raise ValueError("sources changed or forbidden Torch import")
        guard(started)
        artifacts = {p.name: sha_file(p) for p in output.iterdir() if p.is_file()}
        with (output/"manifest.json").open("xb") as handle:
            handle.write(encoded({"status": report["status"], "source_sha256": sources, "artifacts_sha256": artifacts}))
        print(json.dumps({k: report[k] for k in ("status", "projection", "peak_rss_bytes", "seconds")}), flush=True)
    except BaseException as exc:
        with (output/"FAILURE.json").open("xb") as handle:
            handle.write(encoded({"status": "INCOMPLETE", "error": repr(exc)}))
        raise


def projection_status(projected):
    return "READY_FOR_AUDITED_RUN" if projected["per_run_seconds"] <= 300 else "INCOMPLETE"


def supervised_run(output):
    # Reserve this exact output before launching; never mark another run's directory.
    output.mkdir(parents=True, exist_ok=False)
    try:
        subprocess.run([sys.executable, __file__, "--output", str(output), "--worker"],
                       check=True, timeout=125)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        # subprocess.run has killed and waited for the timed-out child before raising.
        try:
            with (output/"FAILURE.json").open("xb") as handle:
                handle.write(encoded({"status": "INCOMPLETE", "error": repr(exc),
                                      "reported_by": "SUPERVISOR"}))
        except FileExistsError:
            pass  # Preserve the child's immutable failure evidence.
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run(args.output, prepared=True)
    else:
        supervised_run(args.output)
