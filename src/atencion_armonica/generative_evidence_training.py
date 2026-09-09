"""Recoverable 50-epoch head kernel; partial batches and observable exclusions.

No dataset reader or campaign entry point. The runner binds exact source bytes,
eligible scene roster, resources and device before calling this kernel.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch

from .generative_evidence import ARMS, CHECKPOINTS, READER_SEEDS
from .generative_evidence_model import EvidenceHead, partition_cost_loss
from .learned_partition_training import collate_targets, _moments

SCHEMA = "generative-evidence-training-state-v1"


def epoch_batches(seed, epoch, scene_ids):
    if (type(seed) is not int or seed not in READER_SEEDS or type(epoch) is not int or not 0 <= epoch < 50
            or not isinstance(scene_ids, list) or not scene_ids or len(scene_ids) > 4096
            or any(type(i) is not int or not 0 <= i < 4096 for i in scene_ids)
            or scene_ids != sorted(set(scene_ids))):
        raise ValueError("invalid paired eligible-scene roster")
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(scene_ids))
    ordered = np.asarray(scene_ids, dtype=np.int64)[order]
    return [ordered[start:start+32] for start in range(0, len(ordered), 32)]


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8,
                             weight_decay=1e-4, amsgrad=False, foreach=False, fused=False)


def _norms(values):
    blocks = {name: [values[f"{name}.weight"], values[f"{name}.bias"]]
              for name in ("group1", "group2", "partition1", "partition2")}
    blocks["generative_columns"] = [values["partition1.weight"][:, 33:39]]
    return {name: float(torch.sqrt(sum(v.detach().double().square().sum() for v in parts)).cpu())
            for name, parts in blocks.items()}


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(v) for v in value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "fiub" and bool(np.isfinite(value).all())
    return value is None or type(value) in (str, bool, int) or (type(value) is float and np.isfinite(value))


class TrainingKernel:
    def __init__(self, arm, checkpoint_seed, reader_seed, *, binding, scene_ids, device):
        if arm not in ARMS or type(checkpoint_seed) is not int or checkpoint_seed not in CHECKPOINTS:
            raise ValueError("cell outside the 27-cell roster")
        if device not in ("cpu", "cuda:0"):
            raise ValueError("device must be an explicitly selected CPU or local GPU")
        canonical = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
        if not binding or canonical != binding:
            raise ValueError("explicit JSON provenance binding required")
        epoch_batches(reader_seed, 0, scene_ids)
        self.binding, self.scene_ids = copy.deepcopy(binding), scene_ids.copy()
        self.arm, self.checkpoint_seed, self.reader_seed, self.device = arm, checkpoint_seed, reader_seed, device
        self.schedule = [epoch_batches(reader_seed, epoch, scene_ids) for epoch in range(50)]
        self.batch_hash = hashlib.sha256(np.concatenate([np.concatenate(b) for b in self.schedule])
                                         .astype("<i8").tobytes()).hexdigest()
        self.model = EvidenceHead(reader_seed).to(device)
        self.optimizer = optimizer_for(self.model)
        self.epoch = self.next_batch = self.steps = 0
        self.history, self.accumulator, self.safe_boundary = [], {}, True

    def expected_scene_ids(self):
        if self.epoch == 50:
            raise ValueError("declared 50 epochs already complete")
        return self.schedule[self.epoch][self.next_batch].copy()

    def step(self, inputs, targets, scene_ids):
        if not self.safe_boundary:
            raise ValueError("incomplete update: restore a verified safe snapshot")
        ids = np.asarray(scene_ids)
        if ids.dtype != np.int64 or not np.array_equal(ids, self.expected_scene_ids()):
            raise ValueError("batch order differs from paired schedule")
        if inputs["groups"].shape[0] != len(ids):
            raise ValueError("partial batch must contain exactly the scheduled scenes")
        self.safe_boundary = False
        b, target = {k: v.to(self.device) for k, v in inputs.items()}, targets.to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        predicted, activations = self.model(b, return_activations=True)
        loss = partition_cost_loss(predicted, target, b["candidate_mask"])
        loss.backward()
        grads = {name: p.grad for name, p in self.model.named_parameters()}
        if any(g is None or not torch.isfinite(g).all() for g in grads.values()):
            raise ValueError("nonfinite training gradient")
        gradients = _norms(grads)
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("nonfinite parameter after optimizer update")
        updates = _norms({name: p.detach()-before[name] for name, p in self.model.named_parameters()})
        moments = {"group_inputs": _moments(b["groups"][b["group_mask"]]),
                   "global_inputs": _moments(b["globals"][b["candidate_mask"]]),
                   "evidence_inputs": _moments(b["evidence"][b["candidate_mask"]]),
                   "targets": _moments(target[b["candidate_mask"]]),
                   "predictions": _moments(predicted[b["candidate_mask"]])}
        activity = {}
        for name, value in activations.items():
            mask = b["group_mask"] if name.startswith("group") else b["candidate_mask"]
            v = value.detach()[mask]
            activity[name] = {"count": len(v), "positive": (v > 0).sum(0).cpu().numpy()}
        loss_value = float(loss.detach().cpu())
        if not self.accumulator:
            self.accumulator = {"updates": 0, "scene_count": 0, "loss_sum": 0., "moments": moments,
                                "activity": activity, "gradients": gradients, "parameter_updates": updates}
        else:
            for field, values in (("moments", moments), ("activity", activity)):
                for name, row in values.items():
                    for key, value in row.items():
                        self.accumulator[field][name][key] += value
            for field, values in (("gradients", gradients), ("parameter_updates", updates)):
                for name, value in values.items():
                    self.accumulator[field][name] += value
        self.accumulator["updates"] += 1
        self.accumulator["scene_count"] += len(ids)
        self.accumulator["loss_sum"] += loss_value*len(ids)
        self.steps += 1
        self.next_batch += 1
        if self.next_batch == len(self.schedule[self.epoch]):
            self.history.append(self.epoch_diagnostics())
            self.epoch += 1
            self.next_batch, self.accumulator = 0, {}
        self.safe_boundary = True
        return loss_value

    def epoch_diagnostics(self):
        a = self.accumulator
        if not a or not a["updates"]:
            raise ValueError("no completed update")
        moments = {}
        for name, row in a["moments"].items():
            mean = row["sum"]/row["count"]
            moments[name] = {"count": row["count"], "mean": mean.tolist(),
                             "variance": np.maximum(0., row["square_sum"]/row["count"]-mean*mean).tolist()}
        return {"epoch": self.epoch, "updates": a["updates"], "scene_count": a["scene_count"],
                "loss": a["loss_sum"]/a["scene_count"], "moments": moments,
                "moment_weighting": "EACH_VALID_ROW; loss uses equal scene mass",
                "activity": {name: {"count": row["count"], "positive_fraction": (row["positive"]/row["count"]).tolist()}
                             for name, row in a["activity"].items()},
                "gradient_norms": {k: v/a["updates"] for k, v in a["gradients"].items()},
                "update_norms": {k: v/a["updates"] for k, v in a["parameter_updates"].items()}}

    def _identity(self):
        return {"schema": SCHEMA, "binding": self.binding, "arm": self.arm,
                "checkpoint_seed": self.checkpoint_seed, "reader_seed": self.reader_seed,
                "scene_ids": self.scene_ids, "device": self.device, "batch_hash": self.batch_hash}

    def state(self):
        if not self.safe_boundary:
            raise ValueError("cannot snapshot an incomplete update")
        return copy.deepcopy({**self._identity(), "epoch": self.epoch, "next_batch": self.next_batch, "steps": self.steps,
                              "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                              "accumulator": self.accumulator, "history": self.history,
                              "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                              "cuda_rng": torch.cuda.get_rng_state(0) if self.device == "cuda:0" else None})

    def restore(self, state):
        # Validate before changing model, optimizer or global RNG. Identity and
        # structural checks do not replace the runner's hash of serialized bytes.
        extra = {"epoch", "next_batch", "steps", "model", "optimizer", "accumulator", "history",
                 "torch_rng", "numpy_rng", "cuda_rng"}
        if (set(state) != set(self._identity()) | extra
                or any(state[k] != v for k, v in self._identity().items()) or not _finite_tree(state)):
            raise ValueError("snapshot schema, binding or finite state differs")
        e, b, steps = (state[k] for k in ("epoch", "next_batch", "steps"))
        batches = len(self.schedule[0])
        if (any(type(v) is not int for v in (e, b, steps)) or not 0 <= e <= 50 or not 0 <= b < batches
                or (e == 50 and b != 0) or steps != e*batches+b or len(state["history"]) != e
                or (b == 0 and state["accumulator"] != {})
                or (b and (state["accumulator"].get("updates") != b
                           or state["accumulator"].get("scene_count") != 32*b))):
            raise ValueError("snapshot update boundary differs")
        for i, row in enumerate(state["history"]):
            if row["epoch"] != i or row["updates"] != batches or row["scene_count"] != len(self.scene_ids):
                raise ValueError("snapshot epoch history differs")
        expected = self.model.state_dict()
        if set(state["model"]) != set(expected):
            raise ValueError("snapshot model names differ")
        for name, p in expected.items():
            v = state["model"][name]
            if not isinstance(v, torch.Tensor) or v.shape != p.shape or v.dtype != torch.float32:
                raise ValueError("snapshot model tensor differs")
        opt = state["optimizer"]
        recipe = optimizer_for(EvidenceHead(self.reader_seed)).state_dict()["param_groups"]
        if (set(opt) != {"state", "param_groups"} or opt["param_groups"] != recipe
                or set(opt["state"]) != (set(range(len(expected))) if steps else set())):
            raise ValueError("snapshot optimizer recipe differs")
        for i, p in enumerate(expected.values()):
            if not steps:
                break
            v = opt["state"][i]
            if (set(v) != {"step", "exp_avg", "exp_avg_sq"} or not isinstance(v["step"], torch.Tensor)
                    or v["step"].shape != () or v["step"].dtype != torch.float32 or v["step"].item() != steps
                    or any(not isinstance(v[k], torch.Tensor) or v[k].shape != p.shape or v[k].dtype != p.dtype
                           for k in ("exp_avg", "exp_avg_sq")) or (v["exp_avg_sq"] < 0).any()):
                raise ValueError("snapshot optimizer moment differs")
        try:
            torch.Generator(device="cpu").set_state(state["torch_rng"])
            np.random.RandomState().set_state(state["numpy_rng"])
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("snapshot RNG differs") from exc
        rng = state["cuda_rng"]
        if ((self.device == "cpu" and rng is not None)
                or (self.device == "cuda:0" and (not isinstance(rng, torch.Tensor) or rng.device.type != "cpu"
                    or rng.dtype != torch.uint8 or rng.shape != (16,)))):
            raise ValueError("snapshot CUDA RNG differs")
        self.safe_boundary = False
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch, self.next_batch, self.steps = e, b, steps
        self.accumulator, self.history = copy.deepcopy(state["accumulator"]), copy.deepcopy(state["history"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        if self.device == "cuda:0":
            torch.cuda.set_rng_state(rng, 0)
        self.safe_boundary = True


__all__ = ["TrainingKernel", "epoch_batches", "collate_targets"]
