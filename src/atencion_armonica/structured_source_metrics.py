"""Explicit evaluation authority and calibration-only structured-reader choice."""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.metrics import adjusted_rand_score

from .partial_compatibility_evaluation import read_partition
from .partial_compatibility_metrics import partition_labels
from .structured_source_reader import FACTORS, GAMMAS, choose_partition

SEEDS = (2026090721, 2026090722, 2026090723)
TEST_SPLITS = ("iid", "ood_beta", "ood_polyphony", "deformed_family")
SPLIT_SEEDS = {"calibration": 2026090780, "iid": 2026090781, "ood_beta": 2026090782,
               "ood_polyphony": 2026090783, "deformed_family": 2026090784}
READERS = (*FACTORS, "historical")
METRICS = ("ari", "exact_partition", "pair_disagreement", "k_inferred", "k_absolute_error", "sub3_member_fraction")


def partition_metrics(partition, source_ids):
    truth = np.asarray(source_ids)
    if truth.ndim != 1 or not 3 <= len(truth) <= 32 or not np.issubdtype(truth.dtype, np.integer):
        raise ValueError("expected integer evaluation-only source labels")
    if (not partition or any(not group for group in partition)
            or any(type(i) not in (int, np.int64) for group in partition for i in group)):
        raise ValueError("invalid partition")
    labels = partition_labels(partition, len(truth))
    upper = np.triu_indices(len(truth), 1)
    y = (truth[:, None] == truth[None, :])[upper]
    predicted = (labels[:, None] == labels[None, :])[upper]
    return {"ari": float(adjusted_rand_score(truth, labels)),
            "exact_partition": bool(np.array_equal(y, predicted)),
            "pair_disagreement": float(np.mean(y != predicted)),
            "k_inferred": len(partition), "k_absolute_error": abs(len(partition)-len(np.unique(truth))),
            "sub3_member_fraction": sum(len(g) for g in partition if len(g) < 3)/len(truth)}


def evaluate_scene(scored, source_ids, logits, historical_threshold):
    n = len(source_ids)
    z = np.asarray(logits)
    if (z.dtype != np.float32 or z.shape != (n, n) or not np.isfinite(z).all()
            or not np.array_equal(z, z.T)):
        raise ValueError("invalid preserved neural logits")
    mapping = scored["pool"]["canonical_to_observed"]
    if sorted(mapping) != list(range(n)):
        raise ValueError("pool and truth have different event rosters")
    metrics = []
    for candidate in scored["candidates"]:
        delivered = [[mapping[i] for i in group] for group in candidate["signature"]]
        metrics.append(partition_metrics(delivered, source_ids))
    probabilities = expit(z.astype(np.float64))
    historical = read_partition(probabilities, historical_threshold)
    upper = np.triu_indices(n, 1)
    truth = np.asarray(source_ids)
    target = (truth[:, None] == truth[None, :])[upper]
    return {"candidate_metrics": metrics,
            "pool_oracle_ari": max(m["ari"] for m in metrics),
            "oracle_authority": "PRIVILEGED_EVALUATION_NOT_DEPLOYABLE",
            "neural_brier": float(np.mean((probabilities[upper]-target)**2)),
            "historical": {"partition": historical, "threshold": historical_threshold,
                           "metrics": partition_metrics(historical, truth)}}


def group_statistics(scored, candidate_index):
    groups = [scored["groups"][i] for i in scored["candidates"][candidate_index]["group_ids"]]
    result = {}
    for size in sorted({g["size"] for g in groups}):
        selected = [g for g in groups if g["size"] == size]
        rms = [g["witness"]["fine"]["minimum_cents"] for g in selected if size >= 3]
        result[str(size)] = {"group_count": len(selected),
                             "rms_cents_mean": float(np.mean(rms)) if rms else None,
                             "rms_status": "GRID_WITNESS_APPROXIMATE" if rms else "UNDERCONSTRAINED",
                             "cost_means": {factor: float(np.mean([g["costs"][factor] for g in selected]))
                                            for factor in FACTORS[1:]}}
    return result


def calibration_grid(scored, evaluation, *, observation, seed):
    if (set(observation) != {"scene_id", "split_seed", "log_f"}
            or type(observation["split_seed"]) is not int
            or observation["split_seed"] != SPLIT_SEEDS["calibration"]):
        raise PermissionError("gamma grid requires the actual calibration observation")
    scene_id = observation["scene_id"]
    if type(scene_id) is not int or not 0 <= scene_id < 256 or type(seed) is not int or seed not in SEEDS:
        raise ValueError("invalid calibration identity")
    q = np.asarray(observation["log_f"], dtype=np.float32)
    order = np.asarray(scored["pool"]["canonical_to_observed"])
    if (q.ndim != 1 or not 3 <= len(q) <= 32 or not np.isfinite(q).all()
            or not np.array_equal(order, np.argsort(q, kind="stable"))
            or not np.array_equal(q[order], np.asarray(scored["pool"]["canonical_q32"], dtype=np.float32))):
        raise ValueError("calibration pool does not belong to this ordered observation")
    return {"scene_id": scene_id, "seed": seed, "split_role": "calibration",
            "split_seed": observation["split_seed"],
            "ari_grid": {factor: [evaluation["candidate_metrics"][choose_partition(scored, factor, gamma)["candidate_index"]]["ari"]
                                   for gamma in GAMMAS] for factor in FACTORS[1:]}}


def _ordered_rows(rows, split):
    if len(rows) != 256*3:
        raise ValueError("requires 256 scenes by exactly three historical seeds")
    ordered = {}
    for row in rows:
        scene, seed = row["scene_id"], row["seed"]
        if (row.get("split_role") != split or type(row.get("split_seed")) is not int
                or row["split_seed"] != SPLIT_SEEDS[split]):
            raise PermissionError("row role/split seed differs from the declared observed corpus")
        if (type(scene) is not int or not 0 <= scene < 256 or type(seed) is not int
                or seed not in SEEDS or (scene, seed) in ordered):
            raise ValueError("duplicate or invalid scene/seed identity")
        ordered[scene, seed] = row
    if set(ordered) != {(s, seed) for s in range(256) for seed in SEEDS}:
        raise ValueError("incomplete scene/seed roster")
    return [[ordered[s, seed] for seed in SEEDS] for s in range(256)]


def select_gammas(rows, *, split):
    if split != "calibration":
        raise PermissionError("gamma selection is calibration-only")
    ordered = _ordered_rows(rows, split)
    selection = {}
    for factor in FACTORS[1:]:
        values = np.asarray([[r["ari_grid"][factor] for r in scene] for scene in ordered], dtype=np.float64)
        if values.shape != (256, 3, 6) or not np.isfinite(values).all() or np.any((values < -1)|(values > 1)):
            raise ValueError("invalid ARI grid")
        means = values.mean(axis=1).mean(axis=0)
        chosen = int(means.argmax())
        selection[factor] = {"gamma": GAMMAS[chosen], "mean_ari_grid": means.tolist(),
                             "per_seed_mean_ari_grid": values.mean(axis=0).tolist()}
    return {"selection_split": split, "scene_count": 256, "seeds": list(SEEDS),
            "gamma_grid": list(GAMMAS), "factors": selection}


def read_selected(scored, evaluation, gammas):
    if set(gammas) != set(FACTORS[1:]):
        raise ValueError("incomplete selected reader roster")
    readers = {}
    for factor in FACTORS:
        gamma = 0. if factor == "pairs" else gammas[factor]
        choice = choose_partition(scored, factor, gamma)
        index = choice["candidate_index"]
        readers[factor] = {"gamma": gamma, "choice": choice,
                           "metrics": evaluation["candidate_metrics"][index],
                           "group_statistics": group_statistics(scored, index)}
    readers["historical"] = evaluation["historical"]
    return {"readers": readers, "neural_brier": evaluation["neural_brier"],
            "pool_oracle_ari": evaluation["pool_oracle_ari"],
            "sham_support_status": scored["sham_support"]["status"]}


def bootstrap_indices():
    return np.random.default_rng(2026090786).integers(0, 256, size=(2000, 256), dtype=np.int64)


def contrast_flags(deltas):
    flags = []
    if deltas["ari"] > 0 and deltas["k_inferred"] > 0:
        flags.append("GAIN_WITH_MORE_GROUPS")
    if deltas["ari"] > 0 and (deltas["sub3_member_fraction"] > 0 or deltas["k_absolute_error"] > 0):
        flags.append("FRAGMENTATION_ATTRIBUTION_UNRESOLVED")
    return flags


def group_rollup(ordered):
    """Group→scene means already computed; require all seeds for conditional RMS.

    Counts include absent-size zeros. Conditional cost/RMS means never fill
    missing seeds with zero; coverage is part of each reported metric.
    """
    output = {}
    for reader in FACTORS:
        output[reader] = {}
        for size in range(1, 9):
            stats = [[r["readers"][reader]["group_statistics"].get(str(size)) for r in scene] for scene in ordered]
            counts = np.array([[0 if g is None else g["group_count"] for g in scene] for scene in stats])
            if np.any(counts < 0) or not np.issubdtype(counts.dtype, np.integer):
                raise ValueError("invalid group-count summaries")
            row = {"mean_group_count": float(counts.mean(axis=1).mean()), "conditional_metrics": {}}
            for metric in ("rms_cents_mean", *FACTORS[1:]):
                values = []
                for scene in stats:
                    seed_values = [None if g is None else g["rms_cents_mean"] if metric == "rms_cents_mean"
                                   else g["cost_means"][metric] for g in scene]
                    if any(v is None for v in seed_values):
                        continue
                    if not np.isfinite(seed_values).all() or np.any(np.asarray(seed_values) < 0):
                        raise ValueError("invalid group metric")
                    values.append(float(np.mean(seed_values)))
                row["conditional_metrics"][metric] = {"mean": float(np.mean(values)) if values else None,
                                                       "three_seed_eligible_scenes": len(values), "total_scenes": 256}
            output[reader][str(size)] = row
    return output


def summarize_test(rows, *, split, indices):
    if split not in TEST_SPLITS:
        raise PermissionError("test summary requires a declared test slice")
    indices = np.asarray(indices)
    if indices.dtype != np.int64 or not np.array_equal(indices, bootstrap_indices()):
        raise ValueError("bootstrap indices differ from the fixed shared schedule")
    ordered = _ordered_rows(rows, split)
    values = np.asarray([[[[r["readers"][reader]["metrics"][metric] for metric in METRICS]
                           for reader in READERS] for r in scene] for scene in ordered], dtype=np.float64)
    if values.shape != (256, 3, 5, 6) or not np.isfinite(values).all():
        raise ValueError("invalid complete metric tensor")
    per_scene = values.mean(axis=1)
    summaries = {reader: {"means": dict(zip(METRICS, values[:, :, i].mean(axis=1).mean(axis=0).tolist())),
                          "per_seed_means": [dict(zip(METRICS, v)) for v in values[:, :, i].mean(axis=0).tolist()]}
                 for i, reader in enumerate(READERS)}
    comparisons = {}
    for control in ("pairs", "decoupled_source", "local_compatibility", "historical"):
        delta = per_scene[:, READERS.index("shared_source")]-per_scene[:, READERS.index(control)]
        samples = delta[indices].mean(axis=1)
        means = dict(zip(METRICS, delta.mean(axis=0).tolist()))
        intervals = {}
        for i, metric in enumerate(METRICS):
            primary = split == "ood_polyphony" and control != "historical" and metric == "ari"
            alpha = .05/3 if primary else .05
            bounds = np.percentile(samples[:, i], [100*alpha/2, 100*(1-alpha/2)])
            intervals[metric] = {"bounds": bounds.tolist(), "nominal_coverage": 1-alpha,
                                 "status": "PRIMARY_BONFERRONI_NOMINAL" if primary else "SECONDARY_DESCRIPTIVE"}
        comparisons[control] = {"mean_deltas": means, "intervals": intervals, "flags": contrast_flags(means),
                                "direction": "shared_source_minus_control"}
    allowed = {"RANKING_CONTRAST_AVAILABLE", "NOT_EVALUABLE_SHAM_SUPPORT"}
    if any(r["sham_support_status"] not in allowed for scene in ordered for r in scene):
        raise ValueError("unknown sham support state")
    support = np.array([[r["sham_support_status"] == "RANKING_CONTRAST_AVAILABLE" for r in scene] for scene in ordered])
    state = ("NOT_EVALUABLE_SHAM_SUPPORT" if not support.any() else
             "FULL_SHAM_SUPPORT" if support.all() else "PARTIAL_SHAM_SUPPORT")
    oracle = np.array([[r["pool_oracle_ari"] for r in scene] for scene in ordered], dtype=np.float64)
    brier = np.array([[r["neural_brier"] for r in scene] for scene in ordered], dtype=np.float64)
    if not np.isfinite(oracle).all() or not np.isfinite(brier).all():
        raise ValueError("invalid reference metrics")
    return {"split": split, "scene_count": 256, "seeds": list(SEEDS), "readers": summaries,
            "comparisons": comparisons, "sham_support": {"status": state,
                "eligible_scene_seed": int(support.sum()), "total_scene_seed": 768,
                "three_seed_eligible_scenes": int(support.all(axis=1).sum()), "total_scenes": 256},
            "pool_oracle_ari": float(oracle.mean(axis=1).mean()),
            "neural_brier": float(brier.mean(axis=1).mean()),
            "group_statistics": group_rollup(ordered),
            "reference_authority": "POOL_ORACLE_PRIVILEGED_LOGITS_UNCHANGED"}
