"""Recoverable training kernel and ragged batches, without a campaign entrypoint.

The future stage runner must validate authorization, exact dataset roster,
resources and binding before using this kernel on any prospective data.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch

from .learned_partition_core import READER_SEEDS
from .learned_partition_model import PartitionCostHead, partition_cost_loss
from .learned_partition_state import validate_state


def epoch_batches(seed, epoch, *, count=4096):
    if seed not in READER_SEEDS or type(epoch) is not int or not 0 <= epoch < 50 or type(count) is not int or count < 32 or count % 32:
        raise ValueError("invalid paired epoch roster")
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(count)
    return order.reshape(-1, 32)


def collate_inputs(rows):
    if not rows or len(rows) > 32:
        raise ValueError("expected 1..32 scenes")
    if any(set(r) != {"groups", "globals", "incidence"} for r in rows):
        raise ValueError("observable-only rows required")
    dims = {r["groups"].shape[1] for r in rows if r["groups"].ndim == 2}
    if len(dims) != 1 or not dims <= {8, 9}:
        raise ValueError("incompatible feature shapes")
    dim = dims.pop()
    for r in rows:
        g, p, w = r["groups"], r["globals"], r["incidence"]
        if (g.ndim != 2 or not 1 <= len(g) <= 94 or g.shape[1] != dim
                or p.ndim != 2 or not 1 <= len(p) <= 64 or p.shape[1] != 6
                or w.shape != (len(p), len(g))
                or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in r.values())):
            raise ValueError("invalid ragged input or nonfinite values")
    batch_size, u, c = len(rows), max(len(r["groups"]) for r in rows), max(len(r["globals"]) for r in rows)
    result = {"groups": torch.zeros(batch_size, u, dim, dtype=torch.float32, device="cpu"),
              "globals": torch.zeros(batch_size, c, 6, dtype=torch.float32, device="cpu"),
              "incidence": torch.zeros(batch_size, c, u, dtype=torch.float32, device="cpu"),
              "group_mask": torch.zeros(batch_size, u, dtype=torch.bool, device="cpu"),
              "candidate_mask": torch.zeros(batch_size, c, dtype=torch.bool, device="cpu")}
    for i, row in enumerate(rows):
        ng, nc = len(row["groups"]), len(row["globals"])
        result["groups"][i, :ng] = torch.from_numpy(row["groups"])
        result["globals"][i, :nc] = torch.from_numpy(row["globals"])
        result["incidence"][i, :nc, :ng] = torch.from_numpy(row["incidence"])
        result["group_mask"][i, :ng] = True
        result["candidate_mask"][i, :nc] = True
    return result


def collate_targets(targets, candidate_mask):
    """Separate supervision port; never inserted into the model's input dict."""
    if len(targets) != len(candidate_mask):
        raise ValueError("supervision roster differs from inputs")
    if (candidate_mask.dtype != torch.bool or candidate_mask.device.type != "cpu"
            or candidate_mask.ndim != 2 or not candidate_mask.any(dim=1).all()
            or not torch.equal(candidate_mask, torch.arange(candidate_mask.shape[1])[None, :] < candidate_mask.sum(1)[:, None])):
        raise ValueError("target collation requires the CPU boolean input mask")
    result = torch.zeros((*candidate_mask.shape, 2), dtype=torch.float32, device="cpu")
    for i, target in enumerate(targets):
        a = np.asarray(target)
        count = int(candidate_mask[i].sum())
        if (a.shape != (count, 2) or a.dtype != np.float32
                or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1)):
            raise ValueError("invalid partition supervision")
        result[i, :count] = torch.from_numpy(a)
    return result


def _moments(values):
    a = values.detach().cpu().numpy().astype(np.float64)
    return {"count": len(a), "sum": a.sum(axis=0), "square_sum": (a*a).sum(axis=0)}


def _parameter_blocks(model, values):
    result = {}
    for prefix in ("group1", "group2", "partition1", "partition2"):
        result[prefix] = [values[f"{prefix}.weight"], values[f"{prefix}.bias"]]
    if model.input_dim == 9:
        result["physical_column"] = [values["group1.weight"][:, 8]]
    else:
        result["baseline_extra_row"] = [values["group1.weight"][32], values["group1.bias"][32:33]]
        result["baseline_extra_column"] = [values["group2.weight"][:, 32]]
    return {name: float(torch.sqrt(sum(v.detach().double().square().sum() for v in block)).cpu())
            for name, block in result.items()}


class TrainingKernel:
    """One initialized cell. Mechanical tests may use a shorter bound roster."""
    def __init__(self, arm, seed, *, binding, count=4096, device="cpu"):
        if type(count) is not int or count < 32 or count % 32:
            raise ValueError("roster must contain full scene batches")
        self.binding = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
        if not self.binding:
            raise ValueError("explicit provenance binding required")
        self.count, self.device = count, str(torch.device(device))
        self.model = PartitionCostHead(arm, seed).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, betas=(.9, .999),
                                           eps=1e-8, weight_decay=1e-4, amsgrad=False,
                                           foreach=False, fused=False)
        self.epoch, self.next_batch, self.steps = 0, 0, 0
        self.accumulator = {}
        self.history = []
        self.safe_boundary = True
        self.batch_hash = hashlib.sha256(np.stack([epoch_batches(seed, e, count=count) for e in range(50)])
                                         .astype("<i8").tobytes()).hexdigest()

    def expected_scene_ids(self):
        if self.epoch >= 50:
            raise ValueError("declared training already complete")
        return epoch_batches(self.model.seed, self.epoch, count=self.count)[self.next_batch]

    def step(self, inputs, targets, scene_ids):
        if not self.safe_boundary:
            raise ValueError("previous update was incomplete; restore a verified snapshot")
        if not np.array_equal(np.asarray(scene_ids), self.expected_scene_ids()):
            raise ValueError("batch order differs from paired schedule")
        if inputs["groups"].shape[0] != 32:
            raise ValueError("training requires exactly 32 scenes")
        self.safe_boundary = False
        b = {k: v.to(self.device) for k, v in inputs.items()}
        target = targets.to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        predicted, activations = self.model(b, return_activations=True)
        loss = partition_cost_loss(predicted, target, b["candidate_mask"])
        loss.backward()
        grads = {n: p.grad for n, p in self.model.named_parameters()}
        if any(g is None or not torch.isfinite(g).all() for g in grads.values()):
            raise ValueError("invalid training gradient")
        gradient_norms = _parameter_blocks(self.model, grads)
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("invalid parameter after update; do not snapshot this step")
        updates = {n: p.detach()-before[n] for n, p in self.model.named_parameters()}
        moments = {"group_inputs": _moments(b["groups"][b["group_mask"]]),
                   "global_inputs": _moments(b["globals"][b["candidate_mask"]]),
                   "targets": _moments(target[b["candidate_mask"]]),
                   "predictions": _moments(predicted[b["candidate_mask"]])}
        activity = {}
        for name, value in activations.items():
            mask = b["group_mask"] if name.startswith("group") else b["candidate_mask"]
            a = value.detach()[mask]
            activity[name] = {"count": len(a), "positive": (a > 0).sum(0).cpu().numpy()}
        self._accumulate(float(loss.detach().cpu()), moments, activity, gradient_norms,
                         _parameter_blocks(self.model, updates))
        self.steps += 1
        self.next_batch += 1
        if self.next_batch == self.count//32:
            self.history.append(self.epoch_diagnostics())
            self.epoch += 1
            self.next_batch = 0
            self.accumulator = {}
        self.safe_boundary = True
        return float(loss.detach().cpu())

    def _accumulate(self, loss, moments, activity, gradients, updates):
        if not self.accumulator:
            self.accumulator = {"updates": 0, "loss_sum": 0., "moments": copy.deepcopy(moments),
                                "activity": copy.deepcopy(activity), "gradients": dict(gradients),
                                "parameter_updates": dict(updates)}
        else:
            for name, row in moments.items():
                for key, value in row.items():
                    self.accumulator["moments"][name][key] += value
            for name, row in activity.items():
                for key, value in row.items():
                    self.accumulator["activity"][name][key] += value
            for field, values in (("gradients", gradients), ("parameter_updates", updates)):
                for name, value in values.items():
                    self.accumulator[field][name] += value
        self.accumulator["updates"] += 1
        self.accumulator["loss_sum"] += loss

    def epoch_diagnostics(self):
        a = self.accumulator
        if not a or not a["updates"]:
            raise ValueError("no completed update to summarize")
        moments = {}
        for name, row in a["moments"].items():
            mean = row["sum"]/row["count"]
            moments[name] = {"count": row["count"], "mean": mean.tolist(),
                             "variance": np.maximum(0., row["square_sum"]/row["count"]-mean*mean).tolist()}
        return {"epoch": self.epoch, "updates": a["updates"], "loss": a["loss_sum"]/a["updates"],
                "moments": moments,
                "activity": {name: {"count": row["count"], "positive_fraction": (row["positive"]/row["count"]).tolist()}
                             for name, row in a["activity"].items()},
                "gradient_norms": {k: v/a["updates"] for k, v in a["gradients"].items()},
                "update_norms": {k: v/a["updates"] for k, v in a["parameter_updates"].items()}}

    def state(self):
        if not self.safe_boundary:
            raise ValueError("cannot snapshot an incomplete update")
        return copy.deepcopy({"schema": "learned-partition-training-state-v1", "binding": self.binding,
                              "arm": self.model.arm, "reader_seed": self.model.seed, "count": self.count,
                              "device": self.device, "batch_hash": self.batch_hash,
                              "epoch": self.epoch, "next_batch": self.next_batch, "steps": self.steps,
                              "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                              "accumulator": self.accumulator, "history": self.history,
                              "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                              "cuda_rng": torch.cuda.get_rng_state() if self.device.startswith("cuda") else None})

    def restore(self, state):
        validate_state(state)  # Entire payload before any mutation, including RNG.
        expected = {"schema": "learned-partition-training-state-v1", "binding": self.binding,
                    "arm": self.model.arm, "reader_seed": self.model.seed, "count": self.count,
                    "device": self.device, "batch_hash": self.batch_hash}
        if any(state.get(k) != v for k, v in expected.items()):
            raise ValueError("snapshot binding, schedule or device differs")
        e, b, steps = state["epoch"], state["next_batch"], state["steps"]
        if (any(type(v) is not int for v in (e, b, steps)) or not 0 <= e <= 50
                or not 0 <= b < self.count//32 or (e == 50 and b != 0)
                or steps != e*(self.count//32)+b or len(state["history"]) != e
                or (b == 0 and state["accumulator"]) or (b and state["accumulator"].get("updates") != b)):
            raise ValueError("inconsistent snapshot update boundary")
        self.safe_boundary = False
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch, self.next_batch, self.steps = e, b, steps
        self.accumulator, self.history = copy.deepcopy(state["accumulator"]), copy.deepcopy(state["history"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        if self.device.startswith("cuda"):
            torch.cuda.set_rng_state(state["cuda_rng"])
        self.safe_boundary = True
