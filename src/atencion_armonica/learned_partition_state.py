"""Validate a recoverable kernel state without mutating a kernel or global RNG.

CPU validation also handles serialized CUDA states without initializing CUDA.
This is structural integrity, not authorization or proof of training provenance.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from .learned_partition_core import ARMS, READER_SEEDS
from .learned_partition_model import PartitionCostHead

SCHEMA = "learned-partition-training-state-v1"
KEYS = {"schema", "binding", "arm", "reader_seed", "count", "device", "batch_hash",
        "epoch", "next_batch", "steps", "model", "optimizer", "accumulator", "history",
        "torch_rng", "numpy_rng", "cuda_rng"}


def _exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("state dictionary schema differs")


def _array(value, size, *, nonnegative=False, upper=None):
    a = np.asarray(value)
    if (a.shape != (size,) or a.dtype.kind not in "fiu" or not np.isfinite(a).all()
            or (nonnegative and np.any(a < 0)) or (upper is not None and np.any(a > upper))):
        raise ValueError("invalid diagnostic vector")


def _number(value, *, nonnegative=True):
    if (type(value) not in (int, float) or not np.isfinite(value)
            or (nonnegative and value < 0)):
        raise ValueError("invalid diagnostic scalar")


def _json_native(value):
    if type(value) is dict:
        return all(type(k) is str and _json_native(v) for k, v in value.items())
    if type(value) is list:
        return all(_json_native(v) for v in value)
    return type(value) in (str, bool, int, float, type(None))


def _diagnostics(value, *, updates, epoch, width, dim, partial):
    moments = {"group_inputs": dim, "global_inputs": 6, "targets": 2, "predictions": 2}
    activity = {"group1": width, "group2": 16, "partition1": 32}
    blocks = {"group1", "group2", "partition1", "partition2"}
    blocks |= {"physical_column"} if dim == 9 else {"baseline_extra_row", "baseline_extra_column"}
    keys = {"updates", "moments", "activity"}
    keys |= {"loss_sum", "gradients", "parameter_updates"} if partial else {"epoch", "loss", "gradient_norms", "update_norms"}
    _exact(value, keys)
    if type(value["updates"]) is not int or value["updates"] != updates:
        raise ValueError("diagnostic update count differs")
    if not partial and (type(value["epoch"]) is not int or value["epoch"] != epoch):
        raise ValueError("diagnostic epoch differs")
    _number(value["loss_sum" if partial else "loss"])
    _exact(value["moments"], moments)
    counts = {}
    for name, size in moments.items():
        row = value["moments"][name]
        _exact(row, {"count", "sum", "square_sum"} if partial else {"count", "mean", "variance"})
        count = row["count"]
        maximum = 94 if name == "group_inputs" else 64
        if type(count) is not int or not 32*updates <= count <= 32*updates*maximum:
            raise ValueError("diagnostic row count differs")
        counts[name] = count
        _array(row["sum" if partial else "mean"], size)
        _array(row["square_sum" if partial else "variance"], size, nonnegative=True)
    if len({counts[k] for k in ("global_inputs", "targets", "predictions")}) != 1:
        raise ValueError("diagnostic candidate counts differ")
    _exact(value["activity"], activity)
    for name, size in activity.items():
        row = value["activity"][name]
        _exact(row, {"count", "positive"} if partial else {"count", "positive_fraction"})
        expected = counts["group_inputs" if name.startswith("group") else "global_inputs"]
        if type(row["count"]) is not int or row["count"] != expected:
            raise ValueError("activation count differs")
        _array(row["positive" if partial else "positive_fraction"], size,
               nonnegative=True, upper=expected if partial else 1)
    for field in (("gradients", "parameter_updates") if partial else ("gradient_norms", "update_norms")):
        _exact(value[field], blocks)
        for number in value[field].values():
            _number(number)


def validate_state(state):
    _exact(state, KEYS)
    if (state["schema"] != SCHEMA or state["arm"] not in ARMS
            or type(state["reader_seed"]) is not int or state["reader_seed"] not in READER_SEEDS
            or not isinstance(state["binding"], dict) or not state["binding"]
            or state["device"] not in ("cpu", "cuda", "cuda:0")):
        raise ValueError("invalid training state identity")
    binding = state["binding"]
    if not _json_native(binding) or json.loads(json.dumps(binding, sort_keys=True, allow_nan=False)) != binding:
        raise ValueError("binding must already have canonical JSON types and values")
    count, e, b, steps = [state[k] for k in ("count", "epoch", "next_batch", "steps")]
    if (any(type(v) is not int for v in (count, e, b, steps)) or count < 32 or count % 32
            or not 0 <= e <= 50 or not 0 <= b < count//32 or (e == 50 and b != 0)
            or steps != e*(count//32)+b):
        raise ValueError("inconsistent update boundary")
    order = np.stack([np.random.default_rng(np.random.SeedSequence([state["reader_seed"], i])).permutation(count)
                      for i in range(50)]).astype("<i8")
    if state["batch_hash"] != hashlib.sha256(order.tobytes()).hexdigest():
        raise ValueError("training batch schedule differs")
    model = PartitionCostHead(state["arm"], state["reader_seed"])
    expected = model.state_dict()
    _exact(state["model"], expected)
    for name, parameter in expected.items():
        value = state["model"][name]
        if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError("invalid model state tensor")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8,
                                 weight_decay=1e-4, amsgrad=False, foreach=False, fused=False)
    opt = state["optimizer"]
    _exact(opt, {"state", "param_groups"})
    if opt["param_groups"] != optimizer.state_dict()["param_groups"]:
        raise ValueError("optimizer recipe or parameter order differs")
    _exact(opt["state"], range(len(expected)) if steps else ())
    for i, parameter in enumerate(expected.values()):
        if not steps:
            break
        row = opt["state"][i]
        _exact(row, {"step", "exp_avg", "exp_avg_sq"})
        t = row["step"]
        if not isinstance(t, torch.Tensor) or t.shape != () or t.dtype != torch.float32 or t.item() != steps:
            raise ValueError("optimizer step differs from boundary")
        for name in ("exp_avg", "exp_avg_sq"):
            t = row[name]
            if (not isinstance(t, torch.Tensor) or t.shape != parameter.shape or t.dtype != torch.float32
                    or not torch.isfinite(t).all() or (name == "exp_avg_sq" and (t < 0).any())):
                raise ValueError("invalid optimizer moment")
    rng = state["torch_rng"]
    if not isinstance(rng, torch.Tensor) or rng.device.type != "cpu" or rng.dtype != torch.uint8 or rng.ndim != 1:
        raise ValueError("invalid CPU RNG tensor")
    try:
        torch.Generator(device="cpu").set_state(rng)
        np.random.RandomState().set_state(state["numpy_rng"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("invalid serialized RNG state") from exc
    rng = state["cuda_rng"]
    if state["device"] == "cpu":
        if rng is not None:
            raise ValueError("CPU state cannot carry a CUDA RNG")
    elif (not isinstance(rng, torch.Tensor) or rng.device.type != "cpu" or rng.dtype != torch.uint8
          or rng.shape != (16,)):
        # The bound PyTorch CUDA generator serializes uint64 seed and offset.
        raise ValueError("invalid serialized CUDA RNG")
    if not isinstance(state["history"], list) or len(state["history"]) != e:
        raise ValueError("history does not cover completed epochs")
    for i, row in enumerate(state["history"]):
        _diagnostics(row, updates=count//32, epoch=i, width=model.width, dim=model.input_dim, partial=False)
    if b:
        _diagnostics(state["accumulator"], updates=b, epoch=e, width=model.width, dim=model.input_dim, partial=True)
    elif state["accumulator"] != {}:
        raise ValueError("epoch boundary must have an empty accumulator")
    return state
