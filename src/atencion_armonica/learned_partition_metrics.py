"""Explicit supervision, calibration selection and scene-paired estimands.

No forward/model loading and no dataset generation. Stage gates must bind
these records to the exact sealed data and prediction manifests.
"""
from __future__ import annotations

import numpy as np

from .learned_partition_core import ARMS, READER_SEEDS, partition_errors
from .learned_partition_readout import choose_costs
from .structured_source_metrics import SEEDS, evaluate_scene
from .structured_source_reader import choose_partition

SPLITS = {"train": (4096, 2026090880), "calibration": (512, 2026090881),
          "iid": (512, 2026090882), "ood_beta": (512, 2026090883),
          "ood_polyphony": (512, 2026090884), "deformed_family": (512, 2026090885)}
TESTS = ("iid", "ood_beta", "ood_polyphony", "deformed_family")
EPOCHS = tuple(range(5, 51, 5))
REFERENCES = ("fixed_pairs", "fixed_shared", "historical")
METRICS = ("ari", "exact_partition", "pair_disagreement", "k_inferred", "k_error",
           "k_absolute_error", "sub3_member_fraction", "vi", "vi_normalized",
           "split_entropy", "merge_entropy", "split_normalized", "merge_normalized")


def candidate_targets(scored, source_ids):
    """Truth-only port: return raw64 errors and delivered float32 targets."""
    y = np.asarray(source_ids)
    order = np.asarray(scored["pool"]["canonical_to_observed"])
    if (y.ndim != 1 or y.dtype.kind not in "iu" or order.dtype.kind not in "iu"
            or not np.array_equal(np.sort(order), np.arange(len(y)))):
        raise ValueError("truth and candidate event identities differ")
    values = [partition_errors(p["signature"], y[order]) for p in scored["candidates"]]
    if not values:
        raise ValueError("empty candidate pool")
    raw = np.stack([v["raw"] for v in values])
    target = np.stack([v["normalized"] for v in values]).astype(np.float32)
    return {"raw": raw, "targets": target}


def _augment(metrics, error, n, true_k):
    split, merge = map(float, error)
    return {**metrics, "k_error": metrics["k_inferred"]-true_k,
            "vi": split+merge, "vi_normalized": (split+merge)/np.log(n),
            "split_entropy": split, "merge_entropy": merge,
            "split_normalized": split/np.log(n), "merge_normalized": merge/np.log(n)}


def evaluate_preserved_predictions(scored, source_ids, logits, checkpoint_seed, predictions):
    """Evaluate already produced predictions for one scene/checkpoint, no model."""
    if checkpoint_seed not in SEEDS or set(predictions) != {(a, s) for a in ARMS for s in READER_SEEDS}:
        raise ValueError("wrong checkpoint or incomplete learned reader roster")
    threshold = {SEEDS[0]: .55, SEEDS[1]: .65, SEEDS[2]: .6}[checkpoint_seed]
    base = evaluate_scene(scored, source_ids, logits, threshold)
    errors = candidate_targets(scored, source_ids)["raw"]
    n, true_k = len(source_ids), len(np.unique(source_ids))
    candidates = [p["signature"] for p in scored["candidates"]]
    candidate_metrics = [_augment(m, error, n, true_k) for m, error in zip(base["candidate_metrics"], errors)]
    references = {}
    for name, factor, gamma in (("fixed_pairs", "pairs", 0.), ("fixed_shared", "shared_source", 1.)):
        choice = choose_partition(scored, factor, gamma)
        references[name] = {"choice": choice, "metrics": candidate_metrics[choice["candidate_index"]]}
    historical = base["historical"]
    references["historical"] = {**historical, "metrics": _augment(historical["metrics"],
        partition_errors(historical["partition"], source_ids)["raw"], n, true_k)}
    learned = {}
    for key, values in predictions.items():
        choice = choose_costs(values, candidates)
        learned[key] = {"choice": choice, "metrics": candidate_metrics[choice["candidate_index"]]}
    return {"learned": learned, "references": references, "candidate_metrics": candidate_metrics,
            "oracle": {"maximum_ari": max(m["ari"] for m in candidate_metrics),
                       "minimum_vi": min(m["vi"] for m in candidate_metrics),
                       "authority": "PRIVILEGED_EVALUATION_NOT_DEPLOYABLE"},
            "neural_brier": base["neural_brier"]}


def select_epochs(records):
    """360 complete per-cell calibration vectors; one shared epoch per arm."""
    expected = {(a, c, s, e) for a in ARMS for c in SEEDS for s in READER_SEEDS for e in EPOCHS}
    indexed = {}
    for r in records:
        if (set(r) != {"arm", "checkpoint_seed", "reader_seed", "epoch", "split", "split_seed", "scene_ids", "ari"}
                or r["split"] != "calibration" or r["split_seed"] != SPLITS["calibration"][1]
                or any(type(r[k]) is not int for k in ("checkpoint_seed", "reader_seed", "epoch", "split_seed"))):
            raise ValueError("selection requires exact calibration record schema and role")
        key = (r["arm"], r["checkpoint_seed"], r["reader_seed"], r["epoch"])
        a = np.asarray(r["ari"], np.float64)
        if (key not in expected or key in indexed or r["scene_ids"] != list(range(512))
                or any(type(i) is not int for i in r["scene_ids"])
                or a.shape != (512,) or not np.isfinite(a).all() or np.any(a < -1) or np.any(a > 1)):
            raise ValueError("invalid, duplicated or misordered calibration vector")
        indexed[key] = a
    if set(indexed) != expected:
        raise ValueError("incomplete 36-cell by 10-epoch calibration roster")
    selected = {}
    for arm in ARMS:
        means, matrices = [], []
        for epoch in EPOCHS:
            v = np.stack([indexed[arm, c, s, epoch] for c in SEEDS for s in READER_SEEDS], axis=1)
            means.append(float(v.mean(axis=1).mean()))
            matrices.append(v.mean(axis=0).reshape(3, 3).tolist())
        selected[arm] = {"epoch": EPOCHS[int(np.argmax(means))], "mean_ari_by_epoch": means,
                         "cell_mean_ari_by_epoch": matrices}
    return {"split": "calibration", "split_seed": SPLITS["calibration"][1], "count": 512,
            "epochs": list(EPOCHS), "selected": selected,
            "checkpoint_seeds": list(SEEDS), "reader_seeds": list(READER_SEEDS)}


def bootstrap_indices():
    return np.random.default_rng(2026090894).integers(0, 512, size=(2000, 512), dtype=np.int64)


def summarize_test(records, *, split, indices):
    """One metric record per scene/checkpoint/reader-seed, never pooled logits."""
    if split not in TESTS or np.asarray(indices).dtype != np.int64 or not np.array_equal(indices, bootstrap_indices()):
        raise ValueError("test role or paired bootstrap indices differ")
    expected = {(i, c, s) for i in range(512) for c in SEEDS for s in READER_SEEDS}
    names = (*ARMS, *REFERENCES)
    indexed = {}
    for r in records:
        if (set(r) != {"scene_id", "checkpoint_seed", "reader_seed", "split", "split_seed", "metrics"}
                or r["split"] != split or r["split_seed"] != SPLITS[split][1]
                or any(type(r[k]) is not int for k in ("scene_id", "checkpoint_seed", "reader_seed", "split_seed"))):
            raise ValueError("wrong test metric role or schema")
        key = (r["scene_id"], r["checkpoint_seed"], r["reader_seed"])
        if key not in expected or key in indexed or set(r["metrics"]) != set(names):
            raise ValueError("duplicate or wrong test metric reader roster")
        for metric in r["metrics"].values():
            if set(metric) != set(METRICS) or not np.isfinite([metric[k] for k in METRICS]).all():
                raise ValueError("incomplete or nonfinite metric vector")
        indexed[key] = r
    if set(indexed) != expected:
        raise ValueError("incomplete 512-scene by 3x3 test roster")
    values = {name: np.asarray([[[[indexed[i, c, s]["metrics"][name][m] for m in METRICS]
                                 for s in READER_SEEDS] for c in SEEDS] for i in range(512)], np.float64)
              for name in names}
    for name in REFERENCES:
        if not np.array_equal(values[name], np.repeat(values[name][:, :, :1], 3, axis=2)):
            raise ValueError("a fixed reference changed across learned-reader initialization")
    scene = {name: v.mean(axis=(1, 2)) for name, v in values.items()}
    means = {name: dict(zip(METRICS, v.mean(axis=0).tolist())) for name, v in scene.items()}
    contrasts = {}
    for control in names:
        if control == "shared_source":
            continue
        delta = scene["shared_source"]-scene[control]
        samples = delta[indices].mean(axis=1)
        result = {}
        for j, metric in enumerate(METRICS):
            primary = split == "ood_polyphony" and control in ARMS and metric == "ari"
            coverage = 1-.05/3 if primary else .95
            lo, hi = np.percentile(samples[:, j], [(1-coverage)*50, (1+coverage)*50])
            result[metric] = {"delta": float(delta[:, j].mean()), "interval": [float(lo), float(hi)],
                              "nominal_coverage": coverage, "family": "primary_three_ari" if primary else "descriptive"}
        contrasts[control] = result
    return {"split": split, "split_seed": SPLITS[split][1], "scene_count": 512,
            "conditional_on_checkpoint_and_reader_seeds": True, "metric_order": list(METRICS),
            "means": means, "contrasts_shared_minus_control": contrasts,
            "per_cell_means": {name: v.mean(axis=0).tolist() for name, v in values.items()},
            "bootstrap": {"unit": "scene", "resamples": 2000, "seed": 2026090894,
                          "method": "percentile", "primary_adjustment": "nominal_bonferroni_three"}}
