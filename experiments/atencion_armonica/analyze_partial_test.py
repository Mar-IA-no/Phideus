"""Frozen scene-level held-out analysis, without fitting or selecting anything."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.atencion_armonica.analyze_partial_validation import cpu_runtime, summarize
from experiments.atencion_armonica.train_partial_compatibility_campaign import CACHE_ROOT, write_new
from experiments.atencion_armonica.collect_partial_test_logits import TEST_CACHE_ROOT
from src.atencion_armonica.partial_compatibility_cache import encoded, load_supervision, sha_file
from src.atencion_armonica.partial_compatibility_inference import load_logits
from src.atencion_armonica.partial_compatibility_metrics import paired_bootstrap, scene_metrics, seed_disagreement
from src.atencion_armonica.partial_compatibility_registry import ARM_NAMES, TRAINING_SEEDS, training_registry
from src.atencion_armonica.partial_compatibility_test_cache import FrozenTestCache, verify_disjoint_manifests
from src.atencion_armonica.partial_compatibility_test_gate import TEST_SPLITS, verify_freeze

CONTROLS = ("pairs_descriptors", "pairs_sham", "pairs_transitivity")


def summarize_test(rows):
    """Keep the closed validation summary and add eligible-scene near metrics."""
    result = summarize(rows)
    result["near_collision_pairs"] = {}
    for key in result["pairs"]:
        values = [row["near_collision_pairs"].get(key) for row in rows]
        usable = [value for value in values if value is not None]
        result["near_collision_pairs"][key] = {
            "mean": float(np.mean(usable)) if usable else None,
            "eligible_scenes": len(usable), "total_scenes": len(rows)}
    return result


def primary_comparisons(scores):
    report, deltas = {}, {}
    for control_index, control in enumerate(CONTROLS):
        per_seed, vectors = [], []
        for seed in TRAINING_SEEDS:
            candidate, baseline = scores["pairs_compatibility"][seed], scores[control][seed]
            per_seed.append(paired_bootstrap(candidate, baseline, control_index=control_index, seed=seed))
            delta = np.asarray(candidate, np.float64)-np.asarray(baseline, np.float64)
            vectors.append(delta)
            deltas[f"{control}__seed_{seed}"] = delta
        averaged = np.stack(vectors).mean(axis=0)  # Average three paired deltas WITHIN each scene first.
        deltas[f"{control}__mean_three_seeds"] = averaged
        means = [row["mean_delta"] for row in per_seed]
        report[control] = {"per_seed": per_seed, "mean_training_seed_delta": float(np.mean(means)),
                           "training_seed_delta_range": [min(means), max(means)],
                           "within_scene_mean_three_seeds": paired_bootstrap(averaged, np.zeros(1024),
                                                                            control_index=control_index, seed=0),
                           "role": "mandatory_attribution" if control_index == 2 else "co_primary"}
    return report, deltas


def validated_raw(raw_root, freeze_path, freeze, registry):
    manifest = json.loads((raw_root/"manifest.json").read_text())
    if (manifest["status"] != "RAW_TEST_COMPLETE" or manifest["freeze_sha256"] != sha_file(freeze_path)
            or manifest["source_sha256"] != freeze["source_sha256"] or (raw_root/"FAILURE.json").exists()):
        raise ValueError("raw test stage incomplete or foreign freeze")
    for name, key in (("request.json", "request_sha256"), ("forward.json", "forward_sha256"), ("worker.log", "worker_log_sha256")):
        if sha_file(raw_root/name) != manifest[key]:
            raise ValueError("raw test artifact hash mismatch")
    request = json.loads((raw_root/"request.json").read_text())
    forward = json.loads((raw_root/"forward.json").read_text())
    if (request["freeze_sha256"] != sha_file(freeze_path) or request["source_sha256"] != freeze["source_sha256"]
            or forward["status"] != "RAW_TEST_COMPLETE" or forward["freeze_sha256"] != sha_file(freeze_path)):
        raise ValueError("raw test bindings disagree")
    for split in TEST_SPLITS:
        if sha_file(TEST_CACHE_ROOT/split/"manifest.json") != request["test_manifest_sha256"][split]:
            raise ValueError("test manifest changed since forward")
    expected = {(split, arm, seed) for split in TEST_SPLITS for arm in ARM_NAMES for seed in TRAINING_SEEDS}
    rows = {}
    for row in forward["rows"]:
        key = (row["split"], row["arm"], row["seed"])
        if (key not in expected or key in rows or row["scene_count"] != 1024
                or row["path"] != f"{key[0]}/{key[1]}__seed_{key[2]}.npz"
                or row["checkpoint_sha256"] != registry["cells"][(key[1], key[2])]["checkpoint_sha256"]
                or sha_file(raw_root/row["path"]) != row["sha256"]):
            raise ValueError("raw test cell identity mismatch")
        rows[key] = row
    if set(rows) != expected:
        raise ValueError("all 75 neural test cells must be present")
    return rows


def analyze(output, raw_root, freeze_path):
    freeze = verify_freeze(freeze_path)
    registry = training_registry(Path(freeze["bindings"]["training"]["path"]).parent)
    rows = validated_raw(raw_root, freeze_path, freeze, registry)
    for split, digest in registry["request"]["cache_manifests"].items():
        if sha_file(CACHE_ROOT/split/"manifest.json") != digest:
            raise ValueError("open split manifest changed before cross-split identity check")
    verify_disjoint_manifests([CACHE_ROOT/s for s in ("development", "train", "validation")]
                             + [TEST_CACHE_ROOT/s for s in TEST_SPLITS])
    output.mkdir(parents=True, exist_ok=False)
    started, results, artifacts = time.monotonic(), {}, {}
    primary_scores = {}
    try:
        for split in TEST_SPLITS:
            cache = FrozenTestCache(TEST_CACHE_ROOT/split, split, freeze_path)
            truths = load_supervision(cache)
            split_root = output/split
            split_root.mkdir()
            results[split], all_probabilities = {}, {}
            for reader in freeze["readers"]:
                arm, seed, threshold = reader["arm"], reader["seed"], reader["threshold"]
                name = "analytic_support" if seed is None else f"{arm}__seed_{seed}"
                if seed is None:
                    logits = None
                    probabilities = [r["pair_support"].astype(np.float64) for r in cache.records]
                else:
                    row = rows[(split, arm, seed)]
                    logits = load_logits(raw_root/row["path"], cache.observations)
                    probabilities = [expit(z.astype(np.float64)) for z in logits]
                    all_probabilities.setdefault(arm, {})[seed] = probabilities
                metrics = []
                path = split_root/f"{name}.jsonl"
                with path.open("xb") as handle:
                    for i, (p, truth, obs) in enumerate(zip(probabilities, truths, cache.observations)):
                        metric = {"scene_id": obs["scene_id"], **scene_metrics(p, truth["source_ids"], obs["log_f"],
                                  threshold, logits=None if logits is None else logits[i])}
                        metrics.append(metric)
                        handle.write(encoded(metric))
                artifacts[str(path.relative_to(output))] = sha_file(path)
                results[split][name] = {"threshold_from_validation": threshold, **summarize_test(metrics)}
                if split == "ood_beta" and seed is not None:
                    primary_scores.setdefault(arm, {})[seed] = np.array([m["pairs"]["brier"] for m in metrics])
            disagreement = {}
            for arm in ARM_NAMES:
                values = [seed_disagreement([all_probabilities[arm][seed][i] for seed in TRAINING_SEEDS]) for i in range(len(cache))]
                path = split_root/f"{arm}__seed_disagreement.jsonl"
                with path.open("xb") as handle:
                    for i, value in enumerate(values):
                        handle.write(encoded({"scene_id": i, **value}))
                artifacts[str(path.relative_to(output))] = sha_file(path)
                disagreement[arm] = {"scene_count": len(values),
                                     "mean_scene_probability_sd": float(np.mean([v["mean_probability_sd_across_three_seeds"] for v in values])),
                                     "mean_scene_hard_disagreement_fraction": float(np.mean([v["fraction_pairs_with_hard_seed_disagreement"] for v in values]))}
            results[split]["seed_disagreement"] = disagreement
            print(json.dumps({"split": split, "status": "SCENE_METRICS_COMPLETE", "seconds": time.monotonic()-started}), flush=True)
        primary, deltas = primary_comparisons(primary_scores)
        with (output/"primary_deltas.npz").open("xb") as handle:
            np.savez_compressed(handle, **deltas)
        artifacts["primary_deltas.npz"] = sha_file(output/"primary_deltas.npz")
        verify_freeze(freeze_path)
        validated_raw(raw_root, freeze_path, freeze, registry)
        write_new(output/"report.json", {"status": "ANALYSIS_COMPLETE_NOT_ADJUDICATED", "primary_slice": "ood_beta",
                  "primary_comparisons": primary, "secondary_by_split": results, "runtime": cpu_runtime(),
                  "limits": ["one_train_corpus", "three_training_seeds", "descriptive_intervals_no_multiplicity_adjustment",
                             "synthetic_partial_events_not_detected_audio_peaks", "global_ambiguity_unadjudicated"],
                  "freeze_sha256": sha_file(freeze_path), "seconds": time.monotonic()-started})
        artifacts["report.json"] = sha_file(output/"report.json")
        write_new(output/"manifest.json", {"status": "ANALYSIS_COMPLETE_NOT_ADJUDICATED", "artifacts_sha256": artifacts,
                  "source_sha256": freeze["source_sha256"], "freeze_sha256": sha_file(freeze_path),
                  "raw_manifest_sha256": sha_file(raw_root/"manifest.json"), "runtime": cpu_runtime()})
    except BaseException as exc:
        write_new(output/"FAILURE.json", {"status": "INCOMPLETE", "error": repr(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.output, args.raw, args.freeze)
