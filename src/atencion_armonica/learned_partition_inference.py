"""Observable-only inference and fixed post-hoc physical-channel interventions.

The caller validates dataset/checkpoint authorization before calling this
module and seals its predictions before opening any evaluation sidecars.
"""
from __future__ import annotations

import numpy as np
import torch

from .learned_partition_cache import validate_rows
from .learned_partition_core import ARMS, model_inputs
from .learned_partition_readout import input_support, prediction_dependence
from .learned_partition_training import collate_inputs


def intervention_inputs(row, normalizer, arm, intervention):
    validate_rows(row)
    if arm not in ARMS[1:] or intervention not in ("zero", "rotate_one", "original_sham"):
        raise ValueError("intervention requires a declared physical arm and channel")
    if intervention == "original_sham":
        if arm != "shared_source":
            raise ValueError("original-sham replay is defined for the same shared head")
        return model_inputs(row, normalizer, "decoupled_source")
    result = model_inputs(row, normalizer, arm)
    if intervention == "zero":
        result["groups"][:, 8] = 0
    else:
        sizes = np.asarray([len(g) for g in row.groups])
        for size in np.unique(sizes):
            ids = np.flatnonzero(sizes == size)
            result["groups"][ids, 8] = np.roll(result["groups"][ids, 8], 1)
    return result


def predict_inputs(model, inputs):
    """Ordered batch32 inference; preserve all candidate components as float32."""
    if not inputs:
        raise ValueError("inference requires a nonempty ordered scene roster")
    was_training = model.training
    device = next(model.parameters()).device
    predictions = []
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(inputs), 32):
                rows = inputs[start:start+32]
                batch = collate_inputs(rows)
                values = model({k: v.to(device) for k, v in batch.items()}).detach().cpu().numpy()
                for i, row in enumerate(rows):
                    a = values[i, :len(row["globals"])].copy()
                    if a.dtype != np.float32 or not np.isfinite(a).all() or np.any(a < 0):
                        raise ValueError("invalid preserved learned cost")
                    predictions.append(a)
    finally:
        model.train(was_training)
    return predictions


def support_by_size(row, original, changed):
    support = input_support(original, changed)
    mask = np.asarray(support["changed_group_mask"])
    sizes = np.asarray([len(g) for g in row.groups])
    support["by_group_size"] = {str(size): {"group_count": int(np.sum(sizes == size)),
        "changed_group_count": int(mask[sizes == size].sum()),
        "changed_group_fraction": float(mask[sizes == size].mean())} for size in np.unique(sizes)}
    return support


def intervention_report(row, original_input, changed_input, original_prediction, changed_prediction):
    """No truth: input support and empirical dependence, not semantic validity."""
    return {"input": support_by_size(row, original_input, changed_input),
            "prediction": prediction_dependence(original_prediction, changed_prediction, row.candidates)}
