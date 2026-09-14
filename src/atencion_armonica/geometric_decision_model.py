"""Common signed head and two objectives; no corpus, fitter or CUDA selection."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .generative_evidence import MAX_CANDIDATES, MAX_GROUPS
from .geometric_decision_core import INPUT_KEYS, READER_SEEDS, ROUTES


def collate(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
        raise ValueError("expected one to 32 eligible scenes")
    for row in rows:
        if set(row) != INPUT_KEYS:
            raise ValueError("observable schema differs or includes supervision")
        g, c, w, e = (row[k] for k in ("groups", "globals", "incidence", "evidence"))
        if (any(not isinstance(a, np.ndarray) or a.dtype != np.float32
                or not np.isfinite(a).all() for a in row.values())
                or g.ndim != 2 or not 1 <= len(g) <= MAX_GROUPS or g.shape[1] != 9
                or c.ndim != 2 or not 1 <= len(c) <= MAX_CANDIDATES or c.shape[1] != 17
                or w.shape != (len(c), len(g)) or e.shape != (len(c), 8)):
            raise ValueError("invalid eligible ragged input")
    b, g, c = len(rows), max(len(r["groups"]) for r in rows), max(len(r["globals"]) for r in rows)
    result = {"groups": torch.zeros((b, g, 9), dtype=torch.float32, device="cpu"),
              "globals": torch.zeros((b, c, 17), dtype=torch.float32, device="cpu"),
              "incidence": torch.zeros((b, c, g), dtype=torch.float32, device="cpu"),
              "evidence": torch.zeros((b, c, 8), dtype=torch.float32, device="cpu"),
              "group_mask": torch.zeros((b, g), dtype=torch.bool, device="cpu"),
              "candidate_mask": torch.zeros((b, c), dtype=torch.bool, device="cpu")}
    for i, row in enumerate(rows):
        ng, nc = len(row["groups"]), len(row["globals"])
        result["groups"][i, :ng] = torch.from_numpy(row["groups"])
        result["globals"][i, :nc] = torch.from_numpy(row["globals"])
        result["evidence"][i, :nc] = torch.from_numpy(row["evidence"])
        result["incidence"][i, :nc, :ng] = torch.from_numpy(row["incidence"])
        result["group_mask"][i, :ng] = True
        result["candidate_mask"][i, :nc] = True
    return result


class GeometricDecisionHead(nn.Module):
    """2258 parameters in every route; route determines only the fixed bypass."""
    def __init__(self, seed, route):
        super().__init__()
        if type(seed) is not int or seed not in READER_SEEDS or route not in ROUTES:
            raise ValueError("seed or route outside the protocol")
        self.seed, self.route = seed, route
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.group1 = nn.Linear(9, 32, device="cpu", dtype=torch.float32)
            self.group2 = nn.Linear(32, 16, device="cpu", dtype=torch.float32)
            self.partition1 = nn.Linear(41, 32, device="cpu", dtype=torch.float32)
            self.partition2 = nn.Linear(32, 2, device="cpu", dtype=torch.float32)
        with torch.no_grad():
            for block, layer in zip((10, 20, 30), (self.group1, self.group2, self.partition1)):
                key = int(np.random.SeedSequence([seed, block]).generate_state(1, dtype=np.uint64)[0])
                rng = torch.Generator(device="cpu").manual_seed(key)
                limit = 1/math.sqrt(layer.in_features)
                layer.weight.uniform_(-limit, limit, generator=rng)
                layer.bias.uniform_(-limit, limit, generator=rng)
            self.partition2.weight.zero_()
            self.partition2.bias.zero_()

    def forward(self, batch, *, return_activations=False):
        if set(batch) != INPUT_KEYS | {"group_mask", "candidate_mask"}:
            raise ValueError("observable batch schema differs or includes supervision")
        x, glob, inc, ev = (batch[k] for k in ("groups", "globals", "incidence", "evidence"))
        gm, cm = batch["group_mask"], batch["candidate_mask"]
        if (x.ndim != 3 or not 1 <= x.shape[0] <= 32 or not 1 <= x.shape[1] <= MAX_GROUPS
                or x.shape[2] != 9 or glob.ndim != 3 or glob.shape[0] != x.shape[0]
                or not 1 <= glob.shape[1] <= MAX_CANDIDATES or glob.shape[2] != 17
                or inc.shape != (*glob.shape[:2], x.shape[1]) or ev.shape != (*glob.shape[:2], 8)
                or gm.shape != x.shape[:2] or cm.shape != glob.shape[:2]
                or gm.dtype != torch.bool or cm.dtype != torch.bool
                or any(v.dtype != torch.float32 for v in (x, glob, inc, ev))
                or any(v.device != self.group1.weight.device for v in batch.values())):
            raise ValueError("invalid batch shapes, dtype, device or masks")
        if (not gm.any(1).all() or not cm.any(1).all()
                or any(not torch.isfinite(v).all() for v in (x, glob, inc, ev))
                or (inc < 0).any() or (inc > 1).any()
                or (x[~gm] != 0).any() or (glob[~cm] != 0).any() or (ev[~cm] != 0).any()
                or (inc[~cm] != 0).any()
                or (inc.masked_select(~gm[:, None, :].expand_as(inc)) != 0).any()
                or not torch.allclose(inc.double().sum(-1)[cm],
                                      torch.ones_like(inc.double().sum(-1)[cm]), rtol=0, atol=2e-7)
                or (ev[..., 6:] < 0).any()
                or (self.route == "local" and (ev != 0).any())):
            raise ValueError("invalid finite padding, scalar, Local or incidence support")
        first = torch.relu(self.group1(x))
        second = torch.relu(self.group2(first))
        aggregate = torch.bmm(inc, second)
        hidden = torch.relu(self.partition1(torch.cat((aggregate, glob, ev), dim=-1)))
        residual = self.partition2(hidden)
        bypass = ev[..., 6] if self.route == "geometric" else ev[..., 7] if self.route == "decoupled" else torch.zeros_like(ev[..., 6])
        output = residual.double()+bypass.double().unsqueeze(-1)/2
        output = torch.where(cm.unsqueeze(-1), output, torch.zeros_like(output))
        if not torch.isfinite(output).all():
            raise ValueError("nonfinite signed partition output")
        if return_activations:
            return output, {"group1": first, "group2": second, "partition1": hidden}
        return output


def _loss_inputs(prediction, target, mask):
    if (prediction.ndim != 3 or prediction.shape[-1] != 2 or prediction.dtype != torch.float64
            or target.shape != prediction.shape or target.dtype != torch.float32
            or target.requires_grad or mask.shape != prediction.shape[:2] or mask.dtype != torch.bool
            or not 1 <= prediction.shape[0] <= 32 or not 1 <= prediction.shape[1] <= MAX_CANDIDATES
            or target.device != prediction.device or mask.device != prediction.device
            or not mask.any(1).all() or not torch.isfinite(prediction).all()
            or not torch.isfinite(target).all() or (target < 0).any() or (target > 1).any()
            or (target[~mask] != 0).any()):
        raise ValueError("invalid signed64 prediction, delivered target32 or candidate mask")


def component_mse(prediction, target, mask):
    _loss_inputs(prediction, target, mask)
    residual = torch.where(mask[..., None], prediction-target.double(), torch.zeros_like(prediction))
    return (residual.square().mean(-1).sum(-1)/mask.sum(-1)).mean()


def decision_losses(prediction, target, mask):
    """One cooptimal-aware structured regret surrogate per scene, float64."""
    _loss_inputs(prediction, target, mask)
    energy, cost = prediction.sum(-1), target.double().sum(-1)
    minimum = cost.masked_fill(~mask, torch.inf).amin(-1, keepdim=True)
    optimal = (cost == minimum) & mask
    reference = torch.where(optimal, energy, torch.zeros_like(energy)).sum(-1)/optimal.sum(-1)
    violations = cost-minimum+reference[:, None]-energy
    return violations.masked_fill(~mask, -torch.inf).amax(-1)


def decision_loss(prediction, target, mask):
    return decision_losses(prediction, target, mask).mean()
