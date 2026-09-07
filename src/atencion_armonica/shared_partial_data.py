"""Frequency-only observations and separate supervision for the frozen loss study.

No historical pool is modified. Test splits require an explicit caller opt-in;
the campaign runner must additionally verify its frozen implementation receipt.
"""
from __future__ import annotations

import numpy as np

SPLITS = {
    "development": (64, 2026090710),
    "train": (8192, 2026090711),
    "validation": (1024, 2026090712),
    "iid": (1024, 2026090713),
    "ood_beta": (1024, 2026090714),
    "ood_polyphony": (1024, 2026090715),
    "ood_noise": (1024, 2026090716),
    "deformed_family": (1024, 2026090717),
}
OPEN_SPLITS = frozenset(("development", "train", "validation"))
CENTS_TO_LOG = np.log(2.0) / 1200.0


def _observe(sources, rng, sigma_cents, scene_id, split_seed):
    ideal, labels, indices = [], [], []
    for sid, source in enumerate(sources):
        ns = np.asarray(source["indices"], dtype=np.float64)
        log_f = (np.log(source["f0"]) + np.log(ns)
                 + 0.5*np.log1p(source["beta"]*ns**2 + source["gamma"]*ns**4))
        ideal.extend(log_f.tolist())
        labels.extend([sid]*len(ns))
        indices.extend(ns.astype(int).tolist())
    ideal = np.asarray(ideal, dtype=np.float64)
    noise = rng.normal(0.0, sigma_cents*CENTS_TO_LOG, len(ideal))
    measured = ideal + noise
    center = float(measured.mean(dtype=np.float64))
    delivered = (measured-center).astype(np.float32)
    perm = rng.permutation(len(ideal))
    observation = {"scene_id": scene_id, "split_seed": split_seed,
                   "log_f": delivered[perm].tolist()}
    sidecar = {"scene_id": scene_id, "split_seed": split_seed, "sources": sources,
               "sigma_cents": sigma_cents, "mean_log_f_observed": center,
               "log_f_ideal": ideal[perm].tolist(), "sensor_log_noise": noise[perm].tolist(),
               "source_ids": np.asarray(labels)[perm].tolist(),
               "partial_indices": np.asarray(indices)[perm].tolist(),
               "permutation": perm.tolist()}
    if not np.all(np.isfinite(delivered)):
        raise FloatingPointError("nonfinite observation")
    return observation, sidecar


def generate_scene(split: str, scene_id: int, *, allow_test: bool = False):
    """Pure generation, default closed-test boundary, no file writes."""
    if split not in SPLITS:
        raise ValueError("unknown split")
    if split not in OPEN_SPLITS and not allow_test:
        raise PermissionError("test generation requires the frozen campaign stage")
    count, seed = SPLITS[split]
    if type(scene_id) is not int or not 0 <= scene_id < count:
        raise ValueError("scene ID outside declared split")
    rng = np.random.default_rng(np.random.SeedSequence([seed, scene_id]))
    k = 4 if split == "ood_polyphony" else int(rng.integers(2, 4))
    beta_lo, beta_hi = (3e-3, 1e-2) if split == "ood_beta" else (1e-4, 1e-3)
    sources = []
    for _ in range(k):
        f0 = float(np.exp(rng.uniform(np.log(100.0), np.log(500.0))))
        beta = float(np.exp(rng.uniform(np.log(beta_lo), np.log(beta_hi))))
        size = int(rng.integers(4, 9))
        ns = sorted(rng.choice(np.arange(1, 9), size, replace=False).tolist())
        gamma = (float(np.exp(rng.uniform(np.log(5e-6), np.log(5e-5))))
                 if split == "deformed_family" else 0.0)
        sources.append({"f0": f0, "beta": beta, "gamma": gamma, "indices": ns})
    sigma = 8.0 if split == "ood_noise" else 2.0
    return _observe(sources, rng, sigma, scene_id, seed)


def mechanical_fixture(k: int, *, deformed: bool = False):
    """Predeclared maximum-size fixtures; never draws an evaluation split."""
    if type(k) is not int or k not in (3, 4):
        raise ValueError("fixture k must be 3 or 4")
    seed = 2026090732
    rng = np.random.default_rng(seed)
    sources = [{"f0": f0, "beta": 5e-4, "gamma": 1e-5 if deformed else 0.0,
                "indices": list(range(1, 9))} for f0 in (110., 173., 281., 419.)[:k]]
    return _observe(sources, rng, 2.0, k, seed)
