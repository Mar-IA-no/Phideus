"""CPU-only validation reader selection, after GPU raw inference has closed.

This stage freezes sixteen reader choices, not an architecture or a test result.
It cannot generate a test split. Neural probabilities are expit(float64(raw32)).
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy
import sklearn
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.atencion_armonica.collect_partial_validation_logits import source_hashes as forward_sources
from experiments.atencion_armonica.train_partial_compatibility_campaign import CACHE_ROOT, write_new
from src.atencion_armonica.partial_compatibility_cache import ObservationCache, encoded, load_supervision, sha_file
from src.atencion_armonica.partial_compatibility_inference import load_logits
from src.atencion_armonica.partial_compatibility_metrics import scene_metrics, select_reader
from src.atencion_armonica.partial_compatibility_registry import training_registry


def sources():
    names = ("experiments/atencion_armonica/analyze_partial_validation.py",
             "src/atencion_armonica/partial_compatibility_metrics.py",
             "src/atencion_armonica/partial_compatibility_evaluation.py")
    return {**forward_sources(), **{name: sha_file(ROOT/name) for name in names}}


def cpu_runtime():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}


def validated_raw_forward(raw_root, registry):
    manifest = json.loads((raw_root/"manifest.json").read_text())
    if (manifest["status"] != "RAW_VALIDATION_COMPLETE" or manifest["test_status"] != "CLOSED"
            or (raw_root/"FAILURE.json").exists()):
        raise ValueError("raw validation inference incomplete")
    for name, key in (("request.json", "request_sha256"), ("forward.json", "forward_sha256"),
                      ("worker.log", "worker_log_sha256")):
        if sha_file(raw_root/name) != manifest[key]:
            raise ValueError("raw inference manifest mismatch")
    request = json.loads((raw_root/"request.json").read_text())
    forward = json.loads((raw_root/"forward.json").read_text())
    if (request["training_manifest_sha256"] != registry["manifest_sha256"]
            or request["source_sha256"] != forward_sources()
            or manifest["source_sha256"] != request["source_sha256"]
            or request["validation_manifest_sha256"] != sha_file(CACHE_ROOT/"validation"/"manifest.json")
            or forward["status"] != "RAW_VALIDATION_COMPLETE" or forward["test_status"] != "CLOSED"):
        raise ValueError("raw validation binding differs")
    rows = {}
    for row in forward["rows"]:
        key = (row["arm"], row["seed"])
        if (key not in registry["cells"] or key in rows or row["scene_count"] != 1024
                or row["path"] != f"{row['arm']}__seed_{row['seed']}.npz"
                or row["checkpoint_sha256"] != registry["cells"][key]["checkpoint_sha256"]
                or sha_file(raw_root/row["path"]) != row["sha256"]):
            raise ValueError("raw validation cell mismatch")
        rows[key] = row
    if set(rows) != set(registry["cells"]):
        raise ValueError("raw validation roster incomplete")
    return rows


def summarize(rows):
    """Unweighted scene means with explicit eligible-scene denominators."""
    result = {"scene_count": len(rows), "aggregation": "mean_of_scene_metrics"}
    groups = {"partition": ("ari", "exact_partition", "k_inferred", "k_error", "k_absolute_error"),
              "pairs": ("brier", "bce", "ap", "auc", "target_prevalence", "mean_probability",
                        "predicted_prevalence_at_half", "positive_recall_at_half",
                        "constant_hard_prediction_at_half", "constant_probability", "binary_entropy_nats")}
    for group, keys in groups.items():
        result[group] = {}
        for key in keys:
            values = [(r if group == "partition" else r[group]).get(key) for r in rows]
            usable = [v for v in values if v is not None]
            result[group][key] = {"mean": float(np.mean(usable)) if usable else None,
                                  "eligible_scenes": len(usable), "total_scenes": len(rows)}
    result["near_collision_scene_count"] = sum(r["near_collision_pairs"]["n_pairs"] > 0 for r in rows)
    result["near_collision_directed_pair_count"] = sum(r["near_collision_pairs"]["n_pairs"] for r in rows)
    result["global_ambiguity_status"] = "UNADJUDICATED"
    return result


def analyze(output, raw_root, training):
    registry = training_registry(training)
    raw_rows = validated_raw_forward(raw_root, registry)
    cache = ObservationCache(CACHE_ROOT/"validation", "validation")
    truths = load_supervision(cache)
    labels = [t["source_ids"] for t in truths]
    output.mkdir(parents=True, exist_ok=False)
    started, frozen, readers = time.monotonic(), sources(), []
    runtime = cpu_runtime()
    try:
        arms = [(arm, seed, row) for (arm, seed), row in raw_rows.items()]
        arms.append(("analytic_support", None, None))
        for arm, seed, row in arms:
            if row is None:
                logits = None
                probabilities = [r["pair_support"].astype(np.float64) for r in cache.records]
            else:
                logits = load_logits(raw_root/row["path"], cache.observations)
                probabilities = [expit(z.astype(np.float64)) for z in logits]
            selection = select_reader(probabilities, labels, split="validation")
            name = f"{arm}__seed_{seed}" if seed is not None else arm
            cell = output/name
            cell.mkdir()
            metrics = []
            with (cell/"scene_metrics.jsonl").open("xb") as handle:
                for i, (p, truth, obs) in enumerate(zip(probabilities, truths, cache.observations)):
                    result = {"scene_id": i, **scene_metrics(p, truth["source_ids"], obs["log_f"],
                              selection["threshold"], logits=None if logits is None else logits[i])}
                    handle.write(encoded(result))
                    metrics.append(result)
            write_new(cell/"summary.json", summarize(metrics))
            write_new(cell/"selection.json", selection)
            readers.append({"arm": arm, "seed": seed, "threshold": selection["threshold"],
                            "selection_path": f"{name}/selection.json", "selection_sha256": sha_file(cell/"selection.json"),
                            "metrics_path": f"{name}/scene_metrics.jsonl", "metrics_sha256": sha_file(cell/"scene_metrics.jsonl"),
                            "summary_path": f"{name}/summary.json", "summary_sha256": sha_file(cell/"summary.json"),
                            "checkpoint_sha256": None if row is None else row["checkpoint_sha256"]})
            print(json.dumps({"arm": arm, "seed": seed, "status": "READER_SELECTED",
                              "seconds": time.monotonic()-started}), flush=True)
        if sources() != frozen:
            raise ValueError("validation analysis sources changed")
        validated_raw_forward(raw_root, registry)
        write_new(output/"readers.json", {"status": "VALIDATION_SELECTION_COMPLETE", "readers": readers,
                  "training_manifest_sha256": registry["manifest_sha256"],
                  "raw_manifest_sha256": sha_file(raw_root/"manifest.json"),
                  "validation_manifest_sha256": sha_file(CACHE_ROOT/"validation"/"manifest.json"),
                  "source_sha256": frozen, "runtime": runtime, "test_status": "CLOSED"})
        write_new(output/"manifest.json", {"status": "VALIDATION_SELECTION_COMPLETE", "test_status": "CLOSED",
                  "readers_sha256": sha_file(output/"readers.json"), "source_sha256": frozen, "runtime": runtime,
                  "seconds": time.monotonic()-started})
    except BaseException as exc:
        write_new(output/"FAILURE.json", {"status": "INCOMPLETE", "error": repr(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.output, args.raw, args.training)
