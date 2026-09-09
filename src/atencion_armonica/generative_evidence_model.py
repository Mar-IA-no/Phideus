"""One paired head for all evidence interventions; no campaign or data producer."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .generative_evidence import MAX_CANDIDATES, MAX_GROUPS, READER_SEEDS
from .learned_partition_model import partition_cost_loss

INPUT_KEYS = {"groups", "globals", "incidence", "evidence"}


def collate(rows):
    """No-candidate scenes are represented outside the model, never padded into one."""
    if not rows or len(rows) > 32:
        raise ValueError("expected one to 32 eligible scenes")
    for row in rows:
        if set(row) != INPUT_KEYS:
            raise ValueError("observable input schema differs or contains supervision")
        g, c, w, e = (row[k] for k in ("groups", "globals", "incidence", "evidence"))
        if (g.ndim != 2 or not 1 <= len(g) <= MAX_GROUPS or g.shape[1] != 9
                or c.ndim != 2 or not 1 <= len(c) <= MAX_CANDIDATES or c.shape[1] != 17
                or w.shape != (len(c), len(g)) or e.shape != (len(c), 6)
                or any(a.dtype != np.float32 or not np.isfinite(a).all() for a in row.values())):
            raise ValueError("invalid ragged observable row")
    b, g, c = len(rows), max(len(r["groups"]) for r in rows), max(len(r["globals"]) for r in rows)
    result = {"groups": torch.zeros((b, g, 9), dtype=torch.float32),
              "globals": torch.zeros((b, c, 17), dtype=torch.float32),
              "incidence": torch.zeros((b, c, g), dtype=torch.float32),
              "evidence": torch.zeros((b, c, 6), dtype=torch.float32),
              "group_mask": torch.zeros((b, g), dtype=torch.bool),
              "candidate_mask": torch.zeros((b, c), dtype=torch.bool)}
    for i, row in enumerate(rows):
        ng, nc = len(row["groups"]), len(row["globals"])
        result["groups"][i, :ng] = torch.from_numpy(row["groups"])
        result["globals"][i, :nc] = torch.from_numpy(row["globals"])
        result["evidence"][i, :nc] = torch.from_numpy(row["evidence"])
        result["incidence"][i, :nc, :ng] = torch.from_numpy(row["incidence"])
        result["group_mask"][i, :ng] = True
        result["candidate_mask"][i, :nc] = True
    return result


class EvidenceHead(nn.Module):
    """2194 parameters; intervention is input data, never an architecture flag."""
    def __init__(self, seed):
        super().__init__()
        if seed not in READER_SEEDS or type(seed) is not int:
            raise ValueError("reader seed outside the protocol")
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.group1 = nn.Linear(9, 32, device="cpu", dtype=torch.float32)
            self.group2 = nn.Linear(32, 16, device="cpu", dtype=torch.float32)
            self.partition1 = nn.Linear(39, 32, device="cpu", dtype=torch.float32)
            self.partition2 = nn.Linear(32, 2, device="cpu", dtype=torch.float32)
        with torch.no_grad():
            for block, layer in zip((10, 20, 30, 40),
                                    (self.group1, self.group2, self.partition1, self.partition2)):
                key = int(np.random.SeedSequence([seed, block]).generate_state(1, dtype=np.uint64)[0])
                rng = torch.Generator(device="cpu").manual_seed(key)
                limit = 1/math.sqrt(layer.in_features)
                layer.weight.uniform_(-limit, limit, generator=rng)
                layer.bias.uniform_(-limit, limit, generator=rng)

    def forward(self, batch, *, return_activations=False):
        if set(batch) != INPUT_KEYS | {"group_mask", "candidate_mask"}:
            raise ValueError("observable batch schema differs or contains supervision")
        x, globals_, incidence, evidence = (batch[k] for k in ("groups", "globals", "incidence", "evidence"))
        gm, cm = batch["group_mask"], batch["candidate_mask"]
        if (x.ndim != 3 or not 1 <= x.shape[0] <= 32 or not 1 <= x.shape[1] <= MAX_GROUPS
                or x.shape[2] != 9 or globals_.ndim != 3 or globals_.shape[0] != x.shape[0]
                or not 1 <= globals_.shape[1] <= MAX_CANDIDATES or globals_.shape[2] != 17
                or incidence.shape != (*globals_.shape[:2], x.shape[1])
                or evidence.shape != (*globals_.shape[:2], 6)
                or gm.shape != x.shape[:2] or cm.shape != globals_.shape[:2]
                or gm.dtype != torch.bool or cm.dtype != torch.bool
                or any(a.dtype != torch.float32 for a in (x, globals_, incidence, evidence))
                or any(a.device != self.group1.weight.device for a in batch.values())):
            raise ValueError("invalid batch shapes, masks, dtype or device")
        if (not gm.any(dim=1).all() or not cm.any(dim=1).all()
                or any(not torch.isfinite(a).all() for a in (x, globals_, incidence, evidence))
                or (incidence < 0).any() or (incidence > 1).any()
                or (x[~gm] != 0).any() or (globals_[~cm] != 0).any()
                or (evidence[~cm] != 0).any() or (incidence[~cm] != 0).any()
                or (incidence.masked_select(~gm[:, None, :].expand_as(incidence)) != 0).any()
                or not torch.allclose(incidence.double().sum(-1)[cm],
                                      torch.ones_like(incidence.double().sum(-1)[cm]), rtol=0, atol=2e-7)):
            raise ValueError("invalid padding, incidence or finite support")
        first = torch.relu(self.group1(x))
        second = torch.relu(self.group2(first))
        aggregate = torch.bmm(incidence, second)
        hidden = torch.relu(self.partition1(torch.cat((aggregate, globals_, evidence), dim=-1)))
        output = torch.nn.functional.softplus(self.partition2(hidden), beta=1, threshold=20)
        if not torch.isfinite(output).all():
            raise ValueError("nonfinite partition prediction")
        if return_activations:
            return output, {"group1": first, "group2": second, "partition1": hidden}
        return output


__all__ = ["EvidenceHead", "collate", "partition_cost_loss"]
