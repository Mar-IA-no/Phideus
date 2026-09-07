"""Tensor adapter and losses for the shared-partial study, no training launcher.

Observation collation never receives truth. Supervision enters loss_components
separately. Importing this module does not select or query an accelerator.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .partial_compatibility import frequency_features, sham_geometry

ARMS = {
    "pairs_descriptors": ("B-local", "bce"),
    "pairs_compatibility": ("B-local", "physical"),
    "pairs_sham": ("B-local", "sham"),
    "pairs_transitivity": ("B-local", "transitivity"),
    "tokens_descriptors": ("A-rich", "bce"),
}
PENALTY_COEFFICIENT = 0.1


def seed_cpu(seed: int):
    # torch.manual_seed also seeds CUDA/MPS/XPU; never use it in this CPU path.
    torch.random.default_generator.manual_seed(seed)


def collate_observations(observations):
    """CPU tensors only; moving to an authorized device belongs to the runner."""
    if not observations:
        raise ValueError("empty batch")
    features, shams = [], []
    for obs in observations:
        if set(obs) != {"scene_id", "split_seed", "log_f"}:
            raise ValueError("observation schema contains missing or privileged fields")
        q = np.asarray(obs["log_f"], dtype=np.float32)
        feat = frequency_features(q)
        features.append(feat)
        shams.append(sham_geometry(q, feat["geometry"], split_seed=obs["split_seed"],
                                   scene_id=obs["scene_id"]))
    bsize = len(features)
    max_n = max(len(f["tokens"]) for f in features)
    max_t = max(len(f["geometry"]["triples"]) for f in features)
    arrays = {
        "tokens": np.zeros((bsize, max_n, 2), np.float32),
        "pair_cont": np.zeros((bsize, max_n, max_n, 4), np.float32),
        "ratio_class_id": np.zeros((bsize, max_n, max_n), np.int64),
        "token_mask": np.zeros((bsize, max_n), bool),
        "pair_valid": np.zeros((bsize, max_n, max_n), bool),
        "triples": np.zeros((bsize, max_t, 3), np.int64),
        "triple_valid": np.zeros((bsize, max_t), bool),
        "physical_weights": np.zeros((bsize, max_t), np.float32),
        "sham_weights": np.zeros((bsize, max_t), np.float32),
        "sham_evaluable": np.asarray([s["evaluable"] for s in shams]),
    }
    for row, (f, s) in enumerate(zip(features, shams)):
        n, t = len(f["tokens"]), len(f["geometry"]["triples"])
        arrays["tokens"][row, :n] = f["tokens"]
        arrays["pair_cont"][row, :n, :n] = f["pair_cont"]
        arrays["ratio_class_id"][row, :n, :n] = f["ratio_class_id"]
        arrays["token_mask"][row, :n] = True
        arrays["pair_valid"][row, :n, :n] = ~np.eye(n, dtype=bool)
        arrays["triples"][row, :t] = f["geometry"]["triples"]
        arrays["triple_valid"][row, :t] = True
        arrays["physical_weights"][row, :t] = f["geometry"]["weights"]
        arrays["sham_weights"][row, :t] = s["weights"]
    return {key: torch.from_numpy(value) for key, value in arrays.items()}


def collate_targets(source_id_lists, batch):
    """Supervision adapter, deliberately not part of observation construction."""
    if len(source_id_lists) != len(batch["tokens"]):
        raise ValueError("target batch size mismatch")
    targets = torch.zeros(batch["pair_valid"].shape, dtype=torch.float32, device="cpu")
    for row, labels in enumerate(source_id_lists):
        n = int(batch["token_mask"][row].sum())
        if len(labels) != n or any(type(s) is not int for s in labels):
            raise ValueError("target length or label type mismatch")
        ids = torch.tensor(labels, dtype=torch.int64, device="cpu")
        targets[row, :n, :n] = (ids[:, None] == ids[None, :]).float()
    return targets


def loss_components(logits, batch, targets):
    """Per-scene BCE, physical/sham clique cost, and generic transitivity."""
    if logits.shape != targets.shape or logits.shape != batch["pair_valid"].shape:
        raise ValueError("logit, target and mask shapes must agree")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("nonfinite logits or targets")
    valid = batch["pair_valid"]
    counts = valid.sum(dim=(1, 2))
    triple_valid = batch["triple_valid"]
    triple_counts = triple_valid.sum(dim=1)
    if torch.any(counts == 0) or torch.any(triple_counts == 0):
        raise ValueError("scene has no valid pairs or triples")
    bce = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
           * valid).sum(dim=(1, 2))/counts
    p = torch.sigmoid(logits)
    i, j, k = batch["triples"].unbind(dim=-1)
    row = torch.arange(len(logits), device=logits.device)[:, None]
    pij, pik, pjk = p[row, i, j], p[row, i, k], p[row, j, k]
    clique = pij*pik*pjk
    physical = (clique*batch["physical_weights"]*triple_valid).sum(dim=1)/triple_counts
    sham = (clique*batch["sham_weights"]*triple_valid).sum(dim=1)/triple_counts
    trans = (F.relu(pij*pjk-pik).square() + F.relu(pij*pik-pjk).square()
             + F.relu(pik*pjk-pij).square())/3
    trans = (trans*triple_valid).sum(dim=1)/triple_counts
    return {"bce": bce, "physical": physical, "sham": sham, "transitivity": trans}


def objective(components, arm: str):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    penalty = ARMS[arm][1]
    loss = components["bce"]
    if penalty != "bce":
        loss = loss+PENALTY_COEFFICIENT*components[penalty]
    return loss.mean()
