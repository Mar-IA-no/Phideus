"""Preserve initial/selected numeric head parameters from completed training.

The caller authenticates COMPLETE selection/training operators first. No
fresh observation, forward, optimizer, new initialization or CUDA call here.
Trusted local checkpoint pickle is authenticated before CPU-only loading.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from .geometric_decision_campaign import ROSTER
from .geometric_decision_campaign_selection import ReadOnlyCellStore
from .geometric_decision_core import ARMS
from .geometric_decision_scene_store import fixed_arrays
from .partial_compatibility_cache import encoded

SHAPES = {"group1.weight": (32, 9), "group1.bias": (32,),
          "group2.weight": (16, 32), "group2.bias": (16,),
          "partition1.weight": (32, 41), "partition1.bias": (32,),
          "partition2.weight": (2, 32), "partition2.bias": (2,)}


def model_arrays(state):
    params = state["model"]
    if not isinstance(params, dict) or set(params) != set(SHAPES):
        raise ValueError("head snapshot has unexpected parameters")
    for name, shape in SHAPES.items():
        value = params[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != "cpu"
                or tuple(value.shape) != shape or not torch.isfinite(value).all()):
            raise ValueError("head parameter dtype, shape, device or finiteness differs")
    if state["epoch"] == 0 and (torch.count_nonzero(params["partition2.weight"]) or torch.count_nonzero(params["partition2.bias"])):
        raise ValueError("initial residual is not exactly zero")
    return {k: v.detach().numpy().copy() for k, v in params.items()}


def selected_epochs(selection):
    result = selection["result"]
    if result["schema"] != "geometric-decision-calibration-selection-v1" or set(result["arms"]) != set(ARMS):
        raise ValueError("selection does not contain all eight arms")
    epochs = {}
    for arm in ARMS:
        rows = result["arms"][arm]["epochs"]
        if ([r["epoch"] for r in rows] != list(range(5, 51, 5))
                or any(type(r["mean_regret_tD"]) not in (int, float) or not np.isfinite(r["mean_regret_tD"]) for r in rows)):
            raise ValueError("selection epoch scores differ")
        chosen = min(rows, key=lambda r: (r["mean_regret_tD"], r["epoch"]))["epoch"]
        if result["arms"][arm]["selected_epoch"] != chosen:
            raise ValueError("selected epoch differs from the common declared criterion")
        epochs[arm] = chosen
    return epochs


def preserve_heads(selection_store, selection_ref, campaign, campaign_ref, output, *, check):
    """144 states, never a seed winner. Operator ownership/admission is external."""
    selection, complete = selection_store.json(selection_ref), campaign.json(campaign_ref)
    if (selection["binding"] != selection_store.binding or selection["campaign"] != campaign_ref
            or selection_store.binding["campaign_binding"] != campaign.binding
            or complete["binding"] != campaign.binding
            or complete["schema"] != "geometric-decision-campaign-complete-v1"
            or [(c["checkpoint_seed"], c["arm"], c["reader_seed"]) for c in complete["cells"]] != list(ROSTER)
            or output.binding.get("selection") != selection_ref
            or output.binding.get("selection_binding") != selection_store.binding):
        raise ValueError("head archive requires the authenticated complete training and selection")
    epochs = selected_epochs(selection)
    provenance = {(r["checkpoint_seed"], r["arm"], r["reader_seed"], r["epoch"]): r for r in selection["calibrations"]}
    expected_roster = {(cp, arm, seed, epoch) for cp, arm, seed in ROSTER for epoch in range(5, 51, 5)}
    if len(selection["calibrations"]) != 720 or set(provenance) != expected_roster:
        raise ValueError("selection calibration provenance is not the complete 720-member roster")
    records = []
    for entry in complete["cells"]:
        check()
        cp, arm, seed = entry["checkpoint_seed"], entry["arm"], entry["reader_seed"]
        root = f"cells/cp_{cp}/{arm}/seed_{seed}"
        if entry["root"] != root:
            raise ValueError("head source cell root differs")
        data = campaign.json(entry["data"])
        binding = {"schema": "geometric-decision-cell-binding-v1", "data": data["binding"],
                   "arm": arm, "checkpoint_seed": cp, "reader_seed": seed, "device": campaign.binding["device"]}
        raw_binding = encoded(binding)
        cell = ReadOnlyCellStore(campaign.root/root, binding_ref={"path": "binding.json", "bytes": len(raw_binding),
            "sha256": hashlib.sha256(raw_binding).hexdigest()})
        closed = cell.json(entry["complete"])
        steps_per_epoch = (len(data["eligible"]["train"])+31)//32
        if (closed["schema"] != "geometric-decision-cell-complete-v1" or closed["binding"] != binding
                or closed["last_epoch"] != 50 or closed["steps"] != steps_per_epoch*50
                or len(closed["calibration"]) != 11 or len(closed["history"]) != 50):
            raise ValueError("head source cell is not complete")
        for stage, epoch in (("initial", 0), ("selected", epochs[arm])):
            check()
            calibration_ref = closed["calibration"][epoch//5]
            calibration = cell.json(calibration_ref)
            if calibration["binding"] != binding or calibration["epoch"] != epoch:
                raise ValueError("calibration does not bind the requested head epoch")
            if stage == "selected":
                selected = provenance[cp, arm, seed, epoch]
                if selected["calibration"] != calibration_ref or selected["state"] != calibration["state"]:
                    raise ValueError("selected head is not the state used by calibrated selection")
            state = cell.load_state(calibration["state"])
            if (state["schema"] != "geometric-decision-training-state-v1" or state["binding"] != binding
                    or (state["checkpoint_seed"], state["arm"], state["reader_seed"]) != (cp, arm, seed)
                    or state["epoch"] != epoch or state["next_batch"] != 0 or state["steps"] != steps_per_epoch*epoch
                    or state["scene_ids"] != data["eligible"]["train"]):
                raise ValueError("loaded head state identity or update boundary differs")
            arrays = model_arrays(state)
            path = f"heads/cp_{cp}/{arm}/seed_{seed}/{stage}"
            array_ref = fixed_arrays(output, path+".npz", arrays)
            record = {"schema": "geometric-decision-frozen-head-v1", "binding": output.binding,
                "checkpoint_seed": cp, "arm": arm, "reader_seed": seed, "stage": stage, "epoch": epoch,
                "arrays": array_ref, "source": {"campaign": campaign_ref, "root": root,
                    "cell_binding": binding, "complete": entry["complete"], "calibration": calibration_ref,
                    "state": calibration["state"], "state_digest": cell.json(calibration["state"])["state_digest"]}}
            records.append(output.publish_json(path+".json", record))
    return output.publish_json("heads.json", {"schema": "geometric-decision-head-archive-v1",
        "binding": output.binding, "selection": selection_ref, "selected_epochs": epochs,
        "count": len(records), "records": records, "new_initializations": False, "forward": False})


def read_head_arrays(store, ref):
    record = store.json(ref)
    if (record["schema"] != "geometric-decision-frozen-head-v1" or record["binding"] != store.binding
            or (record["checkpoint_seed"], record["arm"], record["reader_seed"]) not in ROSTER
            or record["stage"] not in ("initial", "selected")
            or record["epoch"] not in ((0,) if record["stage"] == "initial" else tuple(range(5, 51, 5)))):
        raise ValueError("invalid archived head identity")
    arrays = store.arrays(record["arrays"])
    if set(arrays) != set(SHAPES) or any(v.dtype != np.float32 or v.shape != SHAPES[k] or not np.isfinite(v).all()
                                       for k, v in arrays.items()):
        raise ValueError("invalid archived head parameters")
    if record["stage"] == "initial" and (np.any(arrays["partition2.weight"]) or np.any(arrays["partition2.bias"])):
        raise ValueError("initial archived head lost zero residual")
    return record, arrays
