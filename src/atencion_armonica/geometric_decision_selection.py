"""Single epoch per arm from complete OPEN calibration; no file or test access.

Candidate rows must use the canonical order authenticated by CellData. The
caller pins every prediction receipt and the shared calibration target bytes.
"""
from __future__ import annotations

import numpy as np

from .generative_evidence import CHECKPOINTS, MAX_CANDIDATES
from .geometric_decision_core import ARMS, READER_SEEDS

EPOCHS = tuple(range(5, 51, 5))
ROSTER = tuple((arm, cp, seed, epoch) for arm in ARMS for cp in CHECKPOINTS
               for seed in READER_SEEDS for epoch in EPOCHS)


def select_epochs(targets, energies):
    if not isinstance(targets, list) or len(targets) != 512:
        raise ValueError("selection requires all 512 calibration scene rows")
    if not isinstance(energies, dict) or set(energies) != set(ROSTER):
        raise ValueError("selection requires all 720 declared calibration arrays, excluding initial")
    for t in targets:
        if (not isinstance(t, np.ndarray) or t.dtype != np.float32 or t.ndim != 2
                or t.shape[1] != 2 or len(t) > MAX_CANDIDATES or not np.isfinite(t).all()
                or np.any(t < 0) or np.any(t > 1)):
            raise ValueError("selection target is not delivered float32 components")
    offsets = np.r_[np.int64(0), np.cumsum([len(t) for t in targets], dtype=np.int64)]
    eligible = [i for i, t in enumerate(targets) if len(t)]
    if not eligible:
        raise ValueError("selection has no eligible calibration scene")
    for energy in energies.values():
        if (not isinstance(energy, np.ndarray) or energy.dtype != np.float64
                or energy.shape != (offsets[-1],) or not np.isfinite(energy).all()):
            raise ValueError("selection energy precision, complete extent or finiteness differs")
    cost = [t.astype(np.float64).sum(-1, dtype=np.float64) for t in targets]
    results = {}
    for arm in ARMS:
        epochs = []
        for epoch in EPOCHS:
            regrets = np.empty((9, len(eligible)), np.float64)
            for cell, (cp, seed) in enumerate((cp, seed) for cp in CHECKPOINTS for seed in READER_SEEDS):
                energy = energies[arm, cp, seed, epoch]
                for j, scene_id in enumerate(eligible):
                    first, last = offsets[scene_id:scene_id+2]
                    choice = int(np.argmin(energy[first:last]))
                    regrets[cell, j] = cost[scene_id][choice]-np.min(cost[scene_id])
            scene_regret = regrets.mean(axis=0, dtype=np.float64)
            epochs.append({"epoch": epoch, "mean_regret_tD": float(scene_regret.mean(dtype=np.float64)),
                           "scene_mean_regret_tD": scene_regret.tolist()})
        chosen = min(epochs, key=lambda row: (row["mean_regret_tD"], row["epoch"]))
        results[arm] = {"selected_epoch": chosen["epoch"], "epochs": epochs}
    return {"schema": "geometric-decision-calibration-selection-v1",
            "criterion": "mean nine cells within scene then mean eligible scenes; tD; earliest exact tie",
            "eligible_scene_ids": eligible, "empty_scene_ids": [i for i in range(512) if i not in eligible],
            "arms": results}
