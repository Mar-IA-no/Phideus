"""Recoverable fixed-recipe cell, without corpus access or campaign admission.

Inputs must use the authenticated canonical candidate order. Durations and
resource accounting live in the supervisor, not the replayable numeric state.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch

from .generative_evidence import CHECKPOINTS
from .generative_evidence_training import _finite_tree
from .geometric_decision_core import ARMS, READER_SEEDS
from .geometric_decision_model import GeometricDecisionHead, component_mse, decision_loss
from .partial_compatibility_cache import encoded

SCHEMA = "geometric-decision-training-state-v1"
BLOCKS = ("group1", "group2", "partition1", "partition2")


def epoch_batches(seed, epoch, scene_ids):
    if (type(seed) is not int or seed not in READER_SEEDS or type(epoch) is not int or not 0 <= epoch < 50
            or not isinstance(scene_ids, list) or not scene_ids or len(scene_ids) > 4096
            or any(type(i) is not int or not 0 <= i < 4096 for i in scene_ids)
            or scene_ids != sorted(set(scene_ids))):
        raise ValueError("invalid seed, epoch or paired eligible TRAIN scene roster")
    permutation = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(scene_ids))
    ordered = np.asarray(scene_ids, np.int64)[permutation]
    return [ordered[i:i+32] for i in range(0, len(ordered), 32)]


def collate_targets(rows, mask):
    if (not isinstance(rows, list) or mask.dtype != torch.bool or mask.ndim != 2
            or len(rows) != len(mask) or not 1 <= len(rows) <= 32 or mask.device.type != "cpu"):
        raise ValueError("targets require the corresponding CPU candidate mask")
    target = torch.zeros((*mask.shape, 2), dtype=torch.float32, device="cpu")
    for i, row in enumerate(rows):
        n = int(mask[i].sum())
        if (not isinstance(row, np.ndarray) or row.dtype != np.float32 or row.shape != (n, 2)
                or n == 0 or not np.isfinite(row).all() or np.any(row < 0) or np.any(row > 1)
                or not torch.equal(mask[i], torch.arange(mask.shape[1], device="cpu") < n)):
            raise ValueError("target extent, value or canonical prefix mask differs")
        target[i, :n] = torch.from_numpy(row)
    return target


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8,
                            weight_decay=1e-4, amsgrad=False, foreach=False, fused=False)


def _norms(values):
    result = {name: float(torch.sqrt(sum(values[f"{name}.{part}"].detach().double().square().sum()
                                       for part in ("weight", "bias"))).cpu()) for name in BLOCKS}
    result["evidence_columns"] = float(values["partition1.weight"][:, 33:41].detach().double().norm().cpu())
    return result


def runtime_identity(device):
    return {"torch": str(torch.__version__), "numpy": np.__version__, "device": device,
            "threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32}


class TrainingKernel:
    def __init__(self, arm, checkpoint_seed, reader_seed, *, binding, scene_ids, device):
        if (arm not in ARMS or type(checkpoint_seed) is not int or checkpoint_seed not in CHECKPOINTS
                or device not in ("cpu", "cuda:0")):
            raise ValueError("cell outside the fixed 72-cell roster or backend")
        if not isinstance(binding, dict) or not binding or json.loads(encoded(binding)) != binding:
            raise ValueError("explicit canonical provenance binding required")
        epoch_batches(reader_seed, 0, scene_ids)
        runtime = runtime_identity(device)
        if runtime["threads"] != 1 or not runtime["deterministic"] or runtime["matmul_tf32"] or runtime["cudnn_tf32"]:
            raise ValueError("kernel requires deterministic one-thread runtime without TF32")
        self.binding, self.scene_ids = copy.deepcopy(binding), scene_ids.copy()
        self.arm, self.checkpoint_seed, self.reader_seed, self.device = arm, checkpoint_seed, reader_seed, device
        self.runtime = runtime
        self.schedule = [epoch_batches(reader_seed, e, scene_ids) for e in range(50)]
        self.batch_hash = hashlib.sha256(np.concatenate([np.concatenate(b) for b in self.schedule]).astype("<i8").tobytes()).hexdigest()
        route, self.objective = arm.rsplit("_", 1)
        self.model = GeometricDecisionHead(reader_seed, route).to(device)
        self.optimizer = optimizer_for(self.model)
        self.epoch = self.next_batch = self.steps = 0
        self.history, self.accumulator, self.safe_boundary = [], {}, True

    def expected_scene_ids(self):
        if self.epoch == 50:
            raise ValueError("all 50 epochs already complete")
        return self.schedule[self.epoch][self.next_batch].copy()

    def step(self, inputs, targets, scene_ids):
        if not self.safe_boundary:
            raise ValueError("incomplete update requires restoration of a verified snapshot")
        ids = np.asarray(scene_ids)
        if ids.dtype != np.int64 or not np.array_equal(ids, self.expected_scene_ids()):
            raise ValueError("batch order differs from paired schedule")
        if inputs["groups"].shape[0] != len(ids) or runtime_identity(self.device) != self.runtime:
            raise ValueError("batch extent or frozen runtime changed")
        self.safe_boundary = False
        batch, target = {k: v.to(self.device) for k, v in inputs.items()}, targets.to(self.device)
        mask = batch["candidate_mask"]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction, activations = self.model(batch, return_activations=True)
        mse = component_mse(prediction, target, mask)
        decision = decision_loss(prediction, target, mask)
        loss = mse if self.objective == "mse" else decision
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training objective")
        loss.backward()
        gradients = {name: p.grad for name, p in self.model.named_parameters()}
        if any(v is None or not torch.isfinite(v).all() for v in gradients.values()):
            raise ValueError("nonfinite or missing gradient")
        gradient_norms = _norms(gradients)
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("nonfinite parameter update")
        update_norms = _norms({name: p.detach()-before[name] for name, p in self.model.named_parameters()})
        parameter_norms = _norms(dict(self.model.named_parameters()))
        with torch.no_grad():
            energy = prediction.sum(-1).masked_fill(~mask, torch.inf)
            cost = target.double().sum(-1)
            choice = energy.argmin(-1)  # Input candidates are canonically ordered.
            regret = cost.gather(1, choice[:, None]).squeeze(1)-cost.masked_fill(~mask, torch.inf).amin(-1)
            scalars = {"loss": float(loss.cpu()), "mse": float(mse.cpu()),
                       "decision_surrogate": float(decision.cpu()), "regret_tD": float(regret.mean().cpu())}
        if not all(np.isfinite(v) for v in scalars.values()):
            raise ValueError("nonfinite training diagnostics")
        activity = {}
        for name, values in activations.items():
            valid = values.detach()[batch["group_mask"] if name.startswith("group") else mask]
            activity[name] = {"count": len(valid), "positive": (valid > 0).sum(0).cpu().numpy()}
        if not self.accumulator:
            self.accumulator = {"updates": 0, "scene_count": 0, "sums": {k: 0. for k in scalars},
                "activity": {k: {"count": 0, "positive": np.zeros_like(v["positive"])} for k, v in activity.items()},
                **{field: {k: 0. for k in gradient_norms} for field in ("gradients", "parameter_updates", "parameters")}}
        a = self.accumulator
        a["updates"] += 1
        a["scene_count"] += len(ids)
        for k, v in scalars.items():
            a["sums"][k] += v*len(ids)
        for name, v in activity.items():
            a["activity"][name]["count"] += v["count"]
            a["activity"][name]["positive"] += v["positive"]
        for field, values in (("gradients", gradient_norms), ("parameter_updates", update_norms), ("parameters", parameter_norms)):
            for name, v in values.items():
                a[field][name] += v
        self.steps += 1
        self.next_batch += 1
        if self.next_batch == len(self.schedule[self.epoch]):
            self.history.append(self.epoch_diagnostics())
            self.epoch += 1
            self.next_batch, self.accumulator = 0, {}
        self.safe_boundary = True
        return scalars["loss"]

    def epoch_diagnostics(self):
        a = self.accumulator
        if not a or not a["updates"]:
            raise ValueError("no completed update")
        return {"epoch": self.epoch, "updates": a["updates"], "scene_count": a["scene_count"],
                **{k: v/a["scene_count"] for k, v in a["sums"].items()},
                "activity": {name: {"count": v["count"], "positive_fraction": (v["positive"]/v["count"]).tolist(),
                                     "inactive_units": int(np.count_nonzero(v["positive"] == 0))}
                             for name, v in a["activity"].items()},
                **{name: {k: v/a["updates"] for k, v in a[field].items()}
                   for name, field in (("gradient_norms", "gradients"), ("update_norms", "parameter_updates"), ("parameter_norms", "parameters"))},
                "weighting": "metrics equal scene mass; norms equal update mass; activity valid rows"}

    def _identity(self):
        return {"schema": SCHEMA, "binding": self.binding, "arm": self.arm, "checkpoint_seed": self.checkpoint_seed,
                "reader_seed": self.reader_seed, "scene_ids": self.scene_ids, "device": self.device,
                "runtime": self.runtime, "batch_hash": self.batch_hash}

    def state(self):
        if not self.safe_boundary:
            raise ValueError("cannot snapshot an incomplete update")
        return copy.deepcopy({**self._identity(), "epoch": self.epoch, "next_batch": self.next_batch, "steps": self.steps,
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "accumulator": self.accumulator,
            "history": self.history, "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "cuda_rng": torch.cuda.get_rng_state(0) if self.device == "cuda:0" else None})

    def restore(self, state):
        extra = {"epoch", "next_batch", "steps", "model", "optimizer", "accumulator", "history", "torch_rng", "numpy_rng", "cuda_rng"}
        if (set(state) != set(self._identity()) | extra or not _finite_tree(state)
                or any(state[k] != v for k, v in self._identity().items()) or runtime_identity(self.device) != self.runtime):
            raise ValueError("snapshot identity, schema, finite state or runtime differs")
        e, b, steps = (state[k] for k in ("epoch", "next_batch", "steps"))
        batches = len(self.schedule[0])
        if (any(type(v) is not int for v in (e, b, steps)) or not 0 <= e <= 50 or not 0 <= b < batches
                or (e == 50 and b) or steps != e*batches+b or len(state["history"]) != e
                or (b == 0 and state["accumulator"] != {})
                or (b and (state["accumulator"].get("updates") != b or state["accumulator"].get("scene_count") != 32*b))):
            raise ValueError("snapshot is not a complete scheduled update boundary")
        for i, row in enumerate(state["history"]):
            if row["epoch"] != i or row["updates"] != batches or row["scene_count"] != len(self.scene_ids):
                raise ValueError("snapshot epoch history differs")
        expected = self.model.state_dict()
        if set(state["model"]) != set(expected):
            raise ValueError("snapshot model names differ")
        for name, p in expected.items():
            v = state["model"][name]
            if not isinstance(v, torch.Tensor) or v.shape != p.shape or v.dtype != torch.float32:
                raise ValueError("snapshot parameter shape/dtype differs")
        opt = state["optimizer"]
        recipe = optimizer_for(GeometricDecisionHead(self.reader_seed, self.model.route)).state_dict()["param_groups"]
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
                raise ValueError("snapshot optimizer moments differ")
        try:
            torch.Generator(device="cpu").set_state(state["torch_rng"])
            np.random.RandomState().set_state(state["numpy_rng"])
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("snapshot RNG state differs") from exc
        rng = state["cuda_rng"]
        if ((self.device == "cpu" and rng is not None) or
                (self.device == "cuda:0" and (not isinstance(rng, torch.Tensor) or rng.device.type != "cpu"
                 or rng.dtype != torch.uint8 or rng.ndim != 1 or rng.numel() == 0))):
            raise ValueError("snapshot CUDA RNG differs")
        if self.device == "cuda:0":
            try:
                # The selected backend validates its own state representation;
                # no fixed byte length and no mutation of the live generator.
                torch.Generator(device=self.device).set_state(rng)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError("snapshot CUDA RNG differs") from exc
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
