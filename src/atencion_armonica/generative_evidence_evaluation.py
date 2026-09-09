"""Pure preserved-output calibration and scene-paired test estimands.

No models, files, draws or gates. Callers authenticate complete prediction and
metric archives, and must seal all test predictions before opening truth.
These kernels do not confer test access or select an architectural winner.
"""
from __future__ import annotations

import numpy as np

from . import generative_evidence as ge
from .generative_evidence_cache import SPLITS
from .learned_partition_metrics import METRICS
from .learned_partition_readout import choose_costs

EPOCHS = tuple(range(5, 51, 5))
TESTS = ("iid", "ood_beta", "ood_polyphony", "deformed_family")


def chosen_metrics(partitions, candidate_metrics, components, offsets):
    """All 512 scenes, with null decision / NaN metric row for absent output.

    NaN is an internal masked-array sentinel, never an imputed metric. Public
    JSON summaries turn undefined quantities into null. The caller preserves
    all candidate metrics and prediction components separately.
    """
    if len(partitions) != 512 or len(candidate_metrics) != 512:
        raise ValueError("evaluation requires the full 512-scene roster")
    expected = np.r_[np.int64(0), np.cumsum([len(ps) for ps in partitions], dtype=np.int64)]
    if (not isinstance(offsets, np.ndarray) or offsets.dtype != np.int64 or not np.array_equal(offsets, expected)
            or not isinstance(components, np.ndarray) or components.dtype != np.float32
            or components.shape != (expected[-1], 2) or not np.isfinite(components).all()
            or np.any(components < 0)):
        raise ValueError("prediction dtype, extent, finiteness or offsets differ")
    result = np.full((512, len(METRICS)), np.nan, np.float64)
    decisions, eligible = [], np.zeros(512, bool)
    for i, (ps, metrics) in enumerate(zip(partitions, candidate_metrics)):
        canonical = [ge.law.signature(p) for p in ps]
        if canonical != sorted(set(canonical)) or len(metrics) != len(ps):
            raise ValueError("candidate order or metric extent differs")
        if not ps:
            decisions.append(None)
            continue
        ge.partitions_checked(ps, sum(map(len, canonical[0])))
        if any(set(m) != set(METRICS) or not np.isfinite([m[k] for k in METRICS]).all() for m in metrics):
            raise ValueError("incomplete or nonfinite candidate metrics")
        choice = choose_costs(components[offsets[i]:offsets[i+1]], ps)
        result[i] = [metrics[choice["candidate_index"]][m] for m in METRICS]
        decisions.append(choice)
        eligible[i] = True
    return {"decisions": decisions, "metrics": result, "eligible": eligible}


def select_epochs(records):
    """270 complete cell×epoch vectors; shared observable support, one epoch/arm."""
    expected = {(a, cp, seed, e) for a in ge.ARMS for cp in ge.CHECKPOINTS for seed in ge.READER_SEEDS for e in EPOCHS}
    indexed, identities, support = {}, None, None
    for r in records:
        if (set(r) != {"arm", "checkpoint_seed", "reader_seed", "epoch", "split", "split_seed", "identities", "ari"}
                or r["split"] != "calibration" or r["split_seed"] != SPLITS["calibration"][1]
                or any(type(r[k]) is not int for k in ("checkpoint_seed", "reader_seed", "epoch", "split_seed"))):
            raise ValueError("selection requires the exact OPEN calibration role")
        key = r["arm"], r["checkpoint_seed"], r["reader_seed"], r["epoch"]
        ids, ari = r["identities"], r["ari"]
        if (key not in expected or key in indexed or not isinstance(ids, list) or len(ids) != 512
                or any(not isinstance(i, str) or len(i) != 64 for i in ids) or len(set(ids)) != 512
                or not isinstance(ari, list) or len(ari) != 512):
            raise ValueError("invalid or duplicated calibration cell/identity roster")
        mask = np.array([a is not None for a in ari], bool)
        values = np.asarray([a for a in ari if a is not None], np.float64)
        if (not np.isfinite(values).all() or np.any(values < -1) or np.any(values > 1)
                or (identities is not None and identities != ids)
                or (support is not None and not np.array_equal(support, mask))):
            raise ValueError("calibration identity, ARI or common output support differs")
        identities, support = ids.copy(), mask
        indexed[key] = values
    if set(indexed) != expected:
        raise ValueError("selection requires all 27 cells and ten epochs")
    if not support.any():
        raise ValueError("epoch selection is undefined without calibration output")
    selected = {}
    for arm in ge.ARMS:
        means, cells = [], []
        for epoch in EPOCHS:
            values = np.stack([indexed[arm, cp, seed, epoch] for cp in ge.CHECKPOINTS for seed in ge.READER_SEEDS], axis=1)
            means.append(float(values.mean(axis=1).mean()))
            cells.append(values.mean(axis=0).reshape(3, 3).tolist())
        selected[arm] = {"epoch": EPOCHS[int(np.argmax(means))], "mean_ari_by_epoch": means,
                         "cell_mean_ari_by_epoch": cells}
    return {"split": "calibration", "split_seed": SPLITS["calibration"][1], "count": 512,
            "identities": identities, "eligible_scene_ids": np.flatnonzero(support).tolist(),
            "excluded_scene_ids": np.flatnonzero(~support).tolist(), "epochs": list(EPOCHS),
            "checkpoint_seeds": list(ge.CHECKPOINTS), "reader_seeds": list(ge.READER_SEEDS), "selected": selected}


def bootstrap_indices(split, eligible):
    if split not in TESTS or not isinstance(eligible, np.ndarray) or eligible.dtype != bool or eligible.shape != (512,):
        raise ValueError("bootstrap requires a fresh test and its full boolean output mask")
    count = int(eligible.sum())
    if not count:
        return np.empty((2000, 0), np.int64)
    return np.random.default_rng(np.random.SeedSequence([2026090994, SPLITS[split][1]])).integers(
        0, count, size=(2000, count), dtype=np.int64)


def summarize_learned(values, *, split, eligible):
    """Three (512, checkpoint3, reader3, metric13) arrays, never an ensemble.

    Empty outputs must be explicit NaNs in every cell, with one common mask.
    References and inference interventions are separate system diagnostics.
    """
    indices = bootstrap_indices(split, eligible)
    if set(values) != set(ge.ARMS):
        raise ValueError("test summary requires all three learned arms")
    for v in values.values():
        if (not isinstance(v, np.ndarray) or v.dtype != np.float64 or v.shape != (512, 3, 3, len(METRICS))
                or not np.isfinite(v[eligible]).all() or not np.isnan(v[~eligible]).all()):
            raise ValueError("test metric extent or common empty-output mask differs")
        if np.any(v[eligible, :, :, 0] < -1) or np.any(v[eligible, :, :, 0] > 1):
            raise ValueError("invalid observed ARI")
    count = int(eligible.sum())
    scene = {a: v[eligible].mean(axis=(1, 2)) for a, v in values.items()}

    def intervals(v, primary=False):
        if not count:
            return {m: {"mean": None, "interval": None, "nominal_percent": 97.5 if primary and m == "ari" else 95}
                    for m in METRICS}
        # One shared draw matrix, scene as the unit; cells already averaged.
        samples = v[indices].mean(axis=1)
        result = {}
        for j, metric in enumerate(METRICS):
            corrected = primary and metric == "ari"
            endpoints = [1.25, 98.75] if corrected else [2.5, 97.5]
            result[metric] = {"mean": float(v[:, j].mean()),
                "interval": np.percentile(samples[:, j], endpoints, method="linear").tolist(),
                "nominal_percent": 97.5 if corrected else 95}
        return result

    return {"split": split, "split_seed": SPLITS[split][1], "count": 512, "output_count": count,
            "coverage": count/512, "eligible_scene_ids": np.flatnonzero(eligible).tolist(),
            "excluded_scene_ids": np.flatnonzero(~eligible).tolist(),
            "uncertainty": "SCENE_PAIRED_CONDITIONAL_ON_TRAINED_CHECKPOINTS_AND_READER_SEEDS",
            "interval_status": "UNDEFINED_NO_OUTPUT" if not count else "DEGENERATE_SINGLE_SCENE" if count == 1 else "BOOTSTRAP",
            "arms": {a: intervals(v) for a, v in scene.items()},
            "contrasts": {f"generative-minus-{a}": intervals(scene["generative"]-scene[a], split == "deformed_family")
                          for a in ("decoupled", "local")},
            "cell_means": {a: v[eligible].mean(axis=0).tolist() if count else None for a, v in values.items()},
            "metric_order": list(METRICS), "checkpoint_seeds": list(ge.CHECKPOINTS),
            "reader_seeds": list(ge.READER_SEEDS), "bootstrap_indices": indices}
