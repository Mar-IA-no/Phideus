"""Small invariant partition-cost head; explicit CPU initialization and masks.

No dataset generation, artifact selection, checkpoint loading or CUDA setup.
The caller owns device authorization and the future audited training gate.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .learned_partition_core import ARMS, READER_SEEDS, initialization_seeds


class PartitionCostHead(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        if arm not in ARMS or seed not in READER_SEEDS:
            raise ValueError("arm or reader seed outside declared design")
        self.arm, self.seed = arm, seed
        self.input_dim, self.width = (8, 33) if arm == ARMS[0] else (9, 32)
        # Avoid consuming the caller's CPU RNG in discarded default initializers.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.group1 = nn.Linear(self.input_dim, self.width, device="cpu", dtype=torch.float32)
            self.group2 = nn.Linear(self.width, 16, device="cpu", dtype=torch.float32)
            self.partition1 = nn.Linear(22, 32, device="cpu", dtype=torch.float32)
            self.partition2 = nn.Linear(32, 2, device="cpu", dtype=torch.float32)
        self.reset_paired_parameters()

    def reset_paired_parameters(self):
        seeds = {r["block_id"]: r["torch_seed"] for r in initialization_seeds() if r["reader_seed"] == self.seed}

        def draw(block, out_size, in_size, fan_in, bias=True):
            rng = torch.Generator(device="cpu").manual_seed(seeds[block])
            limit = 1/math.sqrt(fan_in)
            weight = torch.empty((out_size, in_size), dtype=torch.float32).uniform_(-limit, limit, generator=rng)
            value = torch.empty(out_size, dtype=torch.float32).uniform_(-limit, limit, generator=rng) if bias else None
            return weight, value

        w1, b1 = draw(10, 32, 9, 9)
        w2, b2 = draw(20, 16, 32, 32)
        w3, b3 = draw(30, 32, 22, 22)
        w4, b4 = draw(40, 2, 32, 32)
        if self.arm == ARMS[0]:
            extra1, extra_bias = draw(11, 1, 8, 9)
            extra2, _ = draw(21, 16, 1, 32, bias=False)
            w1, b1 = torch.cat((w1[:, :8], extra1)), torch.cat((b1, extra_bias))
            w2 = torch.cat((w2, extra2), dim=1)
        with torch.no_grad():
            for layer, weight, bias in ((self.group1, w1, b1), (self.group2, w2, b2),
                                        (self.partition1, w3, b3), (self.partition2, w4, b4)):
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)

    def forward(self, batch, *, return_activations=False):
        required = {"groups", "globals", "incidence", "group_mask", "candidate_mask"}
        if set(batch) != required:
            raise ValueError("observable batch schema differs or contains supervision")
        x, global_rows, incidence, group_mask, candidate_mask = [batch[k] for k in
            ("groups", "globals", "incidence", "group_mask", "candidate_mask")]
        if (x.ndim != 3 or not x.shape[0] or not x.shape[1]
                or x.shape[-1] != self.input_dim or global_rows.ndim != 3 or not global_rows.shape[1]
                or global_rows.shape[0] != x.shape[0] or global_rows.shape[-1] != 6
                or incidence.shape != (*global_rows.shape[:2], x.shape[1])
                or group_mask.shape != x.shape[:2] or candidate_mask.shape != global_rows.shape[:2]
                or group_mask.dtype != torch.bool or candidate_mask.dtype != torch.bool
                or any(t.dtype != torch.float32 for t in (x, global_rows, incidence))
                or any(t.device != self.group1.weight.device for t in batch.values())):
            raise ValueError("invalid input shapes, masks, dtype or device")
        if (not group_mask.any(dim=1).all() or not candidate_mask.any(dim=1).all()
                or any(not torch.isfinite(t).all() for t in (x, global_rows, incidence))
                or (incidence < 0).any() or (incidence > 1).any()
                or (incidence.masked_select(~group_mask[:, None, :].expand_as(incidence)) != 0).any()
                or (incidence[~candidate_mask] != 0).any()
                or (x[~group_mask] != 0).any() or (global_rows[~candidate_mask] != 0).any()
                or not torch.allclose(incidence.sum(-1)[candidate_mask],
                                      torch.ones_like(incidence.sum(-1)[candidate_mask]), rtol=0, atol=2e-7)):
            raise ValueError("invalid padding, empty candidate or nonfinite input")
        first = torch.relu(self.group1(x))
        second = torch.relu(self.group2(first))
        aggregate = torch.bmm(incidence, second)
        hidden = torch.relu(self.partition1(torch.cat((aggregate, global_rows), dim=-1)))
        output = torch.nn.functional.softplus(self.partition2(hidden), beta=1, threshold=20)
        if not torch.isfinite(output).all():
            raise ValueError("nonfinite reader prediction")
        if return_activations:
            return output, {"group1": first, "group2": second, "partition1": hidden}
        return output


def partition_cost_loss(prediction, target, candidate_mask):
    if (prediction.shape != target.shape or prediction.ndim != 3 or not prediction.shape[0]
            or prediction.shape[-1] != 2
            or candidate_mask.shape != prediction.shape[:2] or candidate_mask.dtype != torch.bool
            or not candidate_mask.any(dim=1).all()
            or any(t.dtype != torch.float32 for t in (prediction, target))
            or any(not torch.isfinite(t).all() for t in (prediction, target))
            or (prediction < 0).any() or (target < 0).any() or (target > 1).any()):
        raise ValueError("invalid supervised cost loss inputs")
    squared = (prediction-target).square().mean(dim=-1)
    per_scene = (squared*candidate_mask).sum(dim=1)/candidate_mask.sum(dim=1)
    return per_scene.mean()
