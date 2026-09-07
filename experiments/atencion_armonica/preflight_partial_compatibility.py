"""Bounded development-only preflight; no training, test draw or GPU query."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.atencion_armonica.shared_partial_data import generate_scene, mechanical_fixture
from src.atencion_armonica.partial_compatibility import frequency_features, sham_geometry
from src.atencion_armonica.partial_compatibility_evaluation import read_partition

SOURCES = (
    "experiments/atencion_armonica/PLAN_SHARED_PARTIAL_COMPATIBILITY.md",
    "experiments/atencion_armonica/preflight_partial_compatibility.py",
    "src/atencion_armonica/shared_partial_data.py",
    "src/atencion_armonica/partial_compatibility.py",
    "src/atencion_armonica/partial_compatibility_learning.py",
    "src/atencion_armonica/partial_compatibility_evaluation.py",
    "src/atencion_armonica/pairformer.py", "src/atencion_armonica/peak_tokens.py",
)


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))+"\n").encode()


def write_new(path, data):
    with path.open("xb") as handle:
        handle.write(data)


def source_hashes():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("empty or nonfinite diagnostic distribution")
    return {"count": len(x), "mean": float(x.mean()), "variance": float(x.var()),
            "quantile_levels": [0, .01, .1, .5, .9, .99, 1],
            "quantiles": np.quantile(x, [0, .01, .1, .5, .9, .99, 1]).tolist()}


def geometry_stage(output):
    started = time.monotonic()
    arrays, rows, observations, truths = {}, [], [], []
    all_r, all_w, same_r, cross_r, same_w, cross_w = [], [], [], [], [], []

    def process(name, obs, truth, *, is_development):
        begin = time.monotonic()
        q = np.asarray(obs["log_f"], dtype=np.float32)
        feat = frequency_features(q)
        g = feat["geometry"]
        sham = sham_geometry(q, g, split_seed=obs["split_seed"], scene_id=obs["scene_id"])
        for key in ("tokens", "pair_cont", "ratio_class_id"):
            arrays[f"{name}/{key}"] = feat[key]
        for key, value in g.items():
            arrays[f"{name}/{key}"] = value
        arrays[f"{name}/sham_weights"] = sham["weights"]
        if sham["canonical_to_delivered"] is not None:
            arrays[f"{name}/canonical_to_delivered"] = sham["canonical_to_delivered"]
        # Frozen diagnostic cut, not validation selection and never true k.
        partition = read_partition(g["pair_support"], .5)
        weights, shuffled = g["weights"], sham["weights"]
        correlation = (float(np.corrcoef(weights, shuffled)[0, 1])
                       if weights.var() > 0 and shuffled.var() > 0 else None)
        row = {"name": name, "n": len(q), "triples": len(weights),
               "sham_evaluable": sham["evaluable"], "sham_shift": sham["shift"],
               "changed_fraction": float(np.mean(weights != shuffled)),
               "physical_sham_correlation": correlation, "reader_at_distance_0_5": partition,
               "pair_mask_count": len(q)*(len(q)-1), "seconds": time.monotonic()-begin}
        rows.append(row)
        if is_development:
            labels = np.asarray(truth["source_ids"])[g["triples"]]
            same = (labels[:, 0] == labels[:, 1]) & (labels[:, 0] == labels[:, 2])
            all_r.extend(g["residual_cents"].tolist())
            all_w.extend(weights.tolist())
            same_r.extend(g["residual_cents"][same].tolist())
            cross_r.extend(g["residual_cents"][~same].tolist())
            same_w.extend(weights[same].tolist())
            cross_w.extend(weights[~same].tolist())

    for scene_id in range(64):
        obs, truth = generate_scene("development", scene_id)
        observations.append(obs)
        truths.append(truth)
        process(f"development_{scene_id:03d}", obs, truth, is_development=True)
    fixtures = {}
    for name, k, deformed in (("max_train", 3, False), ("max_eval", 4, False),
                               ("deformed_mechanical", 4, True)):
        obs, truth = mechanical_fixture(k, deformed=deformed)
        fixtures[name] = {"observation": obs, "sidecar": truth}
        process(name, obs, truth, is_development=False)
    write_new(output/"observations.jsonl", b"".join(encode(o) for o in observations))
    write_new(output/"sidecars.jsonl", b"".join(encode(t) for t in truths))
    write_new(output/"fixtures.json", encode(fixtures))
    with (output/"features.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    report = {"scope": "development_64_and_3_mechanical_fixtures_not_training_or_test",
              "residual_cents": distribution(all_r), "weights": distribution(all_w),
              "support": distribution(1-np.asarray(all_w)), "same_source_triple_R": distribution(same_r),
              "cross_source_triple_R": distribution(cross_r),
              "same_source_triple_w": distribution(same_w),
              "cross_source_triple_w": distribution(cross_w),
              "same_source_triple_support": distribution(1-np.asarray(same_w)),
              "cross_source_triple_support": distribution(1-np.asarray(cross_w)), "rows": rows,
              "seconds": time.monotonic()-started,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              "torch_imported": "torch" in sys.modules}
    report["status"] = ("PASS" if report["weights"]["variance"] > 0
                        and report["seconds"] < 120 and report["peak_rss_bytes"] < 1024**3
                        and not report["torch_imported"] else "REVIEW_REQUIRED")
    write_new(output/"geometry_report.json", encode(report))
    if report["status"] != "PASS":
        raise RuntimeError("geometry preflight requires review")


def gradients_stage(output):
    from unittest.mock import patch
    import torch
    from src.atencion_armonica.partial_compatibility_learning import (
        collate_observations, collate_targets, loss_components, seed_cpu,
    )
    from src.atencion_armonica.pairformer import build_model
    started = time.monotonic()
    torch.set_num_threads(1)
    observations = [json.loads(line) for line in (output/"observations.jsonl").read_bytes().splitlines()][:2]
    truths = [json.loads(line) for line in (output/"sidecars.jsonl").read_bytes().splitlines()][:2]
    rows = []
    with patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CUDA seeding forbidden")), \
         patch("torch.cuda.is_available", side_effect=AssertionError("CUDA query forbidden")):
        batch = collate_observations(observations)
        targets = collate_targets([t["source_ids"] for t in truths], batch)
        for name in ("B-local", "A-rich"):
            seed_cpu(2026090721)
            model = build_model(name).cpu()
            logits = model(batch)
            components = loss_components(logits, batch, targets)
            norms = {}
            for key, component in components.items():
                grads = torch.autograd.grad(component.mean(), tuple(model.parameters()),
                                            retain_graph=True, allow_unused=True)
                norms[key] = float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None)))
            row = {"model": name, "seed": 2026090721,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "per_scene_components": {k: v.detach().tolist() for k, v in components.items()},
                   "component_gradient_norms": norms,
                   "weighted_penalty_to_bce_norm": {k: .1*norms[k]/norms["bce"]
                                                     for k in ("physical", "sham", "transitivity")}}
            rows.append(row)
            # Persist initial state and raw outputs; this is not a trained checkpoint.
            with (output/f"initial_{name}.pt").open("xb") as handle:
                torch.save({"model": model.state_dict(), "cpu_rng": torch.get_rng_state(),
                            "logits": logits.detach(), "scope": "initialization_no_optimizer_step"}, handle)
    paired = next(r for r in rows if r["model"] == "B-local")
    review = (any(paired["component_gradient_norms"][k] <= 0 for k in ("physical", "sham"))
              or any(v > 1 for v in paired["weighted_penalty_to_bce_norm"].values()))
    report = {"scope": "two_development_scenes_initialization_only_no_training",
              "scene_ids": [o["scene_id"] for o in observations], "rows": rows,
              "seconds": time.monotonic()-started, "torch_version": torch.__version__,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              "status": "REVIEW_REQUIRED" if review else "PASS"}
    write_new(output/"gradient_report.json", encode(report))
    if review:
        raise RuntimeError("gradient preflight requires protocol review")


def campaign(output):
    output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    write_new(output/"source_freeze.json", encode(hashes))
    started = time.monotonic()
    try:
        for stage, seconds in (("geometry", 120), ("gradients", 30)):
            subprocess.run([sys.executable, __file__, "--stage", stage, "--output", str(output)],
                           check=True, timeout=seconds)
        if source_hashes() != hashes:
            raise RuntimeError("source changed during preflight")
        receipt = {"status": "PASS", "scope": "CPU_preflight_not_full_experiment",
                   "source_sha256": hashes, "numpy_version": np.__version__,
                   "python_version": sys.version, "total_seconds": time.monotonic()-started,
                   "artifacts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(output.iterdir())}}
        write_new(output/"manifest.json", encode(receipt))
        print(json.dumps(receipt, sort_keys=True), flush=True)
    except BaseException as exc:
        write_new(output/"FAILURE.json", encode({"status": "INCOMPLETE", "error": repr(exc)}))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("geometry", "gradients"))
    args = parser.parse_args()
    if args.stage == "geometry":
        geometry_stage(args.output)
    elif args.stage == "gradients":
        gradients_stage(args.output)
    else:
        campaign(args.output)
