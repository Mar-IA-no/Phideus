"""Bounded inference over already authenticated, observable-only test inputs.

No state loading, test draws, sidecars, device selection or execution authority.
The caller verifies the frozen selected model and delivered archives, guards
resources, and seals every one of the 45 outputs before opening test truth.
"""
from __future__ import annotations

import numpy as np
import torch

from . import generative_evidence as ge
from .generative_evidence_model import EvidenceHead, collate
from .learned_partition_readout import prediction_dependence


def inference_roster():
    return [{"arm": arm, "checkpoint_seed": cp, "reader_seed": seed, "intervention": intervention}
            for cp in ge.CHECKPOINTS for seed in ge.READER_SEEDS for arm in ge.ARMS
            for intervention in (("original", "zero", "decoupled") if arm == "generative" else ("original",))]


def predict(model, rows, *, device, check):
    """Return only real candidate components and 513 offsets, no padded output."""
    if (not isinstance(model, EvidenceHead) or device not in ("cpu", "cuda:0")
            or any(p.device != torch.device(device) for p in model.parameters())
            or not isinstance(rows, list) or len(rows) != 512):
        raise ValueError("inference requires the selected head and complete 512-scene inputs on its fixed device")
    eligible, counts = [], []
    for i, r in enumerate(rows):
        if (set(r) != {"groups", "globals", "evidence", "incidence"}
                or any(not isinstance(a, np.ndarray) or a.dtype != np.float32 or not np.isfinite(a).all() for a in r.values())
                or r["groups"].ndim != 2 or r["groups"].shape[1] != 9
                or not 0 <= len(r["groups"]) <= ge.MAX_GROUPS
                or r["globals"].ndim != 2 or r["globals"].shape[1] != 17
                or not 0 <= len(r["globals"]) <= ge.MAX_CANDIDATES
                or r["evidence"].shape != (len(r["globals"]), 6)
                or r["incidence"].shape != (len(r["globals"]), len(r["groups"]))):
            raise ValueError("inference row contains supervision or invalid observable arrays")
        count = len(r["globals"])
        if bool(count) != bool(len(r["groups"])):
            raise ValueError("empty candidate scene must retain no model groups")
        if count:
            eligible.append(i)
        counts.append(count)
    offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    values = np.empty((offsets[-1], 2), np.float32)
    check()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(eligible), 32):
            check()
            ids = eligible[start:start+32]
            batch = {k: v.to(device) for k, v in collate([rows[i] for i in ids]).items()}
            predicted = model(batch).cpu().numpy()
            for j, i in enumerate(ids):
                values[offsets[i]:offsets[i+1]] = predicted[j, :counts[i]]
    check()
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("nonfinite or negative selected-head output")
    return {"components": values, "offsets": offsets}


def intervention_inputs(delivered, intervention):
    """Use the exact stored Local/Desacoplada channels for a Generativa head.

    The delivered codec already binds shared geometry, TRAIN normalization,
    candidate order and deterministic sham; it must have been verified first.
    """
    if set(delivered) != set(ge.ARMS) or intervention not in ("original", "zero", "decoupled"):
        raise ValueError("unknown inference intervention or missing delivered arms")
    source = {"original": "generative", "zero": "local", "decoupled": "decoupled"}[intervention]
    original, changed = delivered["generative"], delivered[source]
    if len(original) != 512 or len(changed) != 512:
        raise ValueError("intervention requires all 512 observable scenes")
    for a, b in zip(original, changed):
        if (set(a) != {"groups", "globals", "evidence", "incidence"} or set(b) != set(a)
                or any(not np.array_equal(a[k], b[k]) for k in ("groups", "globals", "incidence"))
                or a["evidence"].shape != b["evidence"].shape
                or (intervention == "zero" and np.any(b["evidence"] != 0))):
            raise ValueError("intervention changed common inputs or zero-channel semantics")
    return changed


def dependence(original, changed, partitions):
    """Preserved-output diagnostic; no re-forward or label-dependent support."""
    if not partitions:
        if original.shape != (0, 2) or changed.shape != (0, 2):
            raise ValueError("absent candidates cannot have predictions")
        return {"status": "NO_OBSERVABLE_CANDIDATE", "dependence": None}
    return {"status": "OUTPUT_PRESENT", "dependence": prediction_dependence(original, changed, partitions)}
