"""Learned-cell execution and recovery on fully authorized preserved data.

This module does not generate observations or open test sidecars. The parent
supervisor supplies a live, hash-bound cumulative budget permit before training.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import signal
import time

import numpy as np

from . import learned_partition_gate as gate
from . import learned_partition_provenance as p
from .learned_partition_budget import terminal_receipt, verify_permit
from .learned_partition_budget import recovery_status
from .learned_partition_core import ARMS, READER_SEEDS, model_inputs
from .learned_partition_data import ObservationShard, _bundle, scene_ids
from .learned_partition_metrics import EPOCHS, SEEDS, SPLITS
from .learned_partition_readout import choose_costs
from . import learned_partition_runner as runner
from .partial_compatibility_cache import sha_file
from .structured_source_artifacts import mark_failure, safe_member, seal_bundle, verify_bundle, write_json, write_npz
from .learned_partition_validation import boundary


def cell_binding(*, authorization, train, calibration, normalizers, normalized_train,
                 normalized_calibration, arm, checkpoint_seed, reader_seed):
    if (arm not in ARMS or type(checkpoint_seed) is not int or checkpoint_seed not in SEEDS
            or type(reader_seed) is not int or reader_seed not in READER_SEEDS):
        raise ValueError("undeclared learned training cell")
    auth = gate.verify_authorization(authorization, "train")
    return {"common": auth["common"], "authorization": authorization, "train": train, "calibration": calibration,
        "normalizers": normalizers, "normalized_train": normalized_train, "normalized_calibration": normalized_calibration,
        "arm": arm, "checkpoint_seed": checkpoint_seed, "reader_seed": reader_seed,
        "device": auth["training_device"], "count": 4096}


@boundary
def load_cell_data(binding):
    """Inputs and targets remain separate, with exact immutable corpus rosters."""
    authref, common, seed, arm = (binding[k] for k in ("authorization", "common", "checkpoint_seed", "arm"))
    auth, train_manifest, train_shards = runner.training_corpus(binding["train"], "train", authorization=authref)
    _, cal_manifest, cal_shards = runner.training_corpus(binding["calibration"], "calibration", authorization=authref)
    if auth["common"] != common or auth["training_device"] != binding["device"]:
        raise ValueError("training source or device binding changed")
    _, cal_data = _bundle(cal_manifest["binding"]["data"], "learned_observation_split", common)
    if cal_data["binding"]["previous"] != {"train": train_manifest["binding"]["data"]}:
        raise ValueError("calibration and training belong to different prospective chains")
    norms = runner.read_normalizers(binding["normalizers"], common, authorization=authref, train=binding["train"])
    result = {}
    for split, shards, normalized in (("train", train_shards, binding["normalized_train"]),
                                      ("calibration", cal_shards, binding["normalized_calibration"])):
        if not isinstance(normalized, list) or len(normalized) != len(shards):
            raise ValueError("normalized training/calibration shard roster differs")
        inputs, targets, candidates, ari = [], [], [], []
        for index, (entry, normref) in enumerate(zip(shards, normalized)):
            cache = ObservationShard(entry["data"], split, index, common)
            root = runner.normalized_shard(normref, cache, common, authorization=authref, data=entry["data"],
                logits=entry["logits"], scored=entry["scored"], normalizers=binding["normalizers"], train=binding["train"])
            scored_root = p.verify_reference(entry["scored"]).parent
            target_root = p.verify_reference(entry["targets"]).parent
            packed = runner.read_inputs(root/f"seed_{seed}/{arm}.npz", scene_ids=cache.scene_ids,
                                        dim=8 if arm == ARMS[0] else 9)
            for i, values in zip(cache.scene_ids, packed):
                row = runner.load_rows(scored_root/f"seed_{seed}/{i:05d}_rows.npz")
                expected = model_inputs(row, norms[seed], arm)
                if any(not np.array_equal(values[k], expected[k]) for k in expected):
                    raise ValueError("preserved normalized tensor differs from its train-only transformation")
                inputs.append(values)
                with np.load(target_root/f"seed_{seed}/{i:05d}_targets.npz", allow_pickle=False) as raw:
                    targets.append(raw["targets"])
                candidates.append(row.candidates)
                metrics = json.loads((target_root/f"seed_{seed}/{i:05d}_metrics.json").read_bytes())
                ari.append(np.asarray([m["ari"] for m in metrics["candidate_metrics"]], np.float64))
        if len(inputs) != SPLITS[split][0]:
            raise ValueError("cell data has an incomplete scene roster")
        result[split] = {"inputs": inputs, "targets": targets, "candidates": candidates, "ari": ari}
    return result


def save_calibration(root, epoch, snapshot, model, data, binding):
    from .learned_partition_inference import predict_inputs
    if epoch not in EPOCHS or len(data["inputs"]) != 512:
        raise ValueError("calibration requires a declared evaluation epoch and full scene roster")
    folder = Path(root)/f"calibration_{epoch:02d}"
    folder.mkdir()
    try:
        predicted = predict_inputs(model, data["inputs"])
        offsets = np.r_[np.int64(0), np.cumsum([len(v) for v in predicted], dtype=np.int64)]
        choices = np.asarray([choose_costs(values, pool)["candidate_index"]
                              for values, pool in zip(predicted, data["candidates"])], np.int64)
        ari = np.asarray([values[i] for values, i in zip(data["ari"], choices)], np.float64)
        write_npz(folder/"predictions.npz", components=np.concatenate(predicted), offsets=offsets,
                  choices=choices, ari=ari, scene_ids=np.arange(512, dtype=np.int64))
        seal_bundle(folder, role="learned_calibration_predictions", binding={"cell": binding,
            "epoch": epoch, "snapshot": snapshot}, resources={"operation": "calibration_no_updates"})
        return p.reference(folder/"manifest.json")
    except BaseException as exc:
        mark_failure(folder, exc)
        raise


def calibration_record(ref, binding, data, *, snapshot):
    path = p.verify_reference(ref)
    manifest = verify_bundle(path.parent, ref["sha256"], role="learned_calibration_predictions")
    epoch = manifest["binding"]["epoch"]
    if (epoch not in EPOCHS or manifest["binding"] != {"cell": binding, "epoch": epoch, "snapshot": snapshot}
            or set(manifest["artifacts_sha256"]) != {"predictions.npz"}):
        raise ValueError("calibration snapshot/epoch/cell identity differs")
    with np.load(path.parent/"predictions.npz", allow_pickle=False) as raw:
        if set(raw.files) != {"components", "offsets", "choices", "ari", "scene_ids"}:
            raise ValueError("calibration predictions schema differs")
        a = {k: raw[k] for k in raw.files}
    offsets = np.r_[np.int64(0), np.cumsum([len(v) for v in data["candidates"]], dtype=np.int64)]
    if (a["components"].dtype != np.float32 or a["components"].shape != (offsets[-1], 2)
            or a["offsets"].dtype != np.int64 or not np.array_equal(a["offsets"], offsets)
            or a["scene_ids"].dtype != np.int64 or not np.array_equal(a["scene_ids"], np.arange(512))
            or a["choices"].dtype != np.int64 or a["ari"].dtype != np.float64):
        raise ValueError("calibration candidate/scene roster or dtype differs")
    choices = np.asarray([choose_costs(a["components"][lo:hi], pool)["candidate_index"]
        for lo, hi, pool in zip(offsets[:-1], offsets[1:], data["candidates"])], np.int64)
    ari = np.asarray([values[i] for values, i in zip(data["ari"], choices)], np.float64)
    if not np.array_equal(a["choices"], choices) or not np.array_equal(a["ari"], ari):
        raise ValueError("calibration readout does not replay from preserved predictions")
    return {"arm": binding["arm"], "checkpoint_seed": binding["checkpoint_seed"], "reader_seed": binding["reader_seed"],
        "epoch": epoch, "split": "calibration", "split_seed": SPLITS["calibration"][1],
        "scene_ids": list(range(512)), "ari": ari.tolist()}


@boundary
def snapshot_chain(refs, binding, *, complete=False):
    from .learned_partition_snapshots import read_snapshot
    if not isinstance(refs, list) or not refs:
        raise ValueError("training requires its complete init-to-last snapshot chain")
    seen, positions, previous = set(), {}, None
    for index, ref in enumerate(refs):
        if ref["path"] in seen:
            raise ValueError("duplicate snapshot in training chain")
        seen.add(ref["path"])
        state, manifest = read_snapshot(ref, expected_binding=binding)
        if (state["count"] != binding["count"] or state["arm"] != binding["arm"] or state["reader_seed"] != binding["reader_seed"]
                or state["device"] != binding["device"] or manifest["parents"] != ([] if index == 0 else [previous])
                or (index == 0 and state["steps"] != 0)):
            raise ValueError("snapshot chain changed cell, initialization or direct ancestry")
        if state["next_batch"] == 0 and state["epoch"] in EPOCHS:
            positions[state["epoch"]] = ref
        previous = ref
    required = [epoch for epoch in EPOCHS if epoch < state["epoch"] or (complete and epoch <= state["epoch"])]
    if any(epoch not in positions for epoch in required) or (complete and state["steps"] != 50*(binding["count"]//32)):
        raise ValueError("snapshot chain omits a mandatory evaluation boundary")
    return state, positions


def resume_state(resume, binding, data, *, visiting=None):
    """Recover only a registered failed parent with an exact autosnapshot."""
    from .learned_partition_snapshots import read_snapshot
    if not isinstance(resume, dict) or set(resume) != {"request", "terminal", "snapshot"}:
        raise ValueError("resume requires request, supervisor terminal receipt and snapshot")
    terminal = terminal_receipt(resume["terminal"], request_ref=resume["request"])
    if terminal["status"] != "FAILED":
        raise PermissionError("a complete cell must be reused, not resumed")
    request = p.read_reference(resume["request"])
    visiting = set() if visiting is None else set(visiting)
    if resume["request"]["path"] in visiting:
        raise ValueError("cyclic attempt ancestry")
    visiting.add(resume["request"]["path"])
    args = request["arguments"]
    for key in ("authorization", "train", "calibration", "normalizers", "normalized_train", "normalized_calibration",
                "arm", "checkpoint_seed", "reader_seed"):
        if args[key] != binding[key]:
            raise ValueError("resume parent changes cell inputs or architecture")
    parent = safe_member(p.ROOT, request["output"])
    inherited = (None, [], []) if args["resume"] is None else resume_state(args["resume"], binding, data, visiting=visiting)
    expected_ancestry = {"resume": args["resume"], "snapshots": inherited[1], "calibrations": inherited[2]}
    ancestry_path = parent/"ancestry.json"
    if ancestry_path.is_file():
        ancestry = json.loads(ancestry_path.read_bytes())
    elif not parent.exists() or {f.name for f in parent.iterdir()} <= {"FAILURE.json"}:
        ancestry = expected_ancestry  # Verified failure before any scientific output.
    else:
        raise ValueError("attempt lost its ancestry despite retaining scientific output")
    if ancestry != expected_ancestry:
        raise ValueError("attempt ancestry differs from its verified resume parent")
    observed_status = recovery_status(parent, request)
    if terminal["recovery_status"] != observed_status:
        raise ValueError("terminal recovery classification contradicts preserved training state")
    if observed_status == "NO_UPDATES_NO_SNAPSHOT":
        if resume["snapshot"] is not None or inherited[1] or inherited[2]:
            raise ValueError("pre-initialization recovery cannot discard inherited learned state")
        return None, [], []
    valid = []
    for candidate in (parent/"snapshots").glob("*/manifest.json"):
        # Atomic publish names only: interrupted staging remains evidence, not a checkpoint.
        if candidate.parent.name.startswith("snapshot_staging_"):
            continue
        try:
            candidate_ref = p.reference(candidate)
            value, _ = read_snapshot(candidate_ref, expected_binding=binding)
            valid.append((value["steps"], candidate_ref))
        except (ValueError, OSError):
            continue
    snapshots = list(ancestry["snapshots"])
    snapshots.extend(ref for _, ref in sorted(valid, key=lambda item: item[0]) if ref not in snapshots)
    if not snapshots or snapshots[-1] != resume["snapshot"]:
        raise ValueError("resume must use the parent's latest complete own or inherited snapshot")
    state, positions = snapshot_chain(snapshots, binding)
    calibrations = list(ancestry["calibrations"])
    by_epoch = {p.read_reference(ref)["binding"]["epoch"]: ref for ref in calibrations}
    for epoch in EPOCHS:
        file = parent/f"calibration_{epoch:02d}/manifest.json"
        if file.is_file() and epoch <= state["epoch"]:
            ref = p.reference(file)
            try:
                calibration_record(ref, binding, data, snapshot=positions[epoch])
            except (ValueError, OSError):
                # A failed/partial calibration can be recomputed only if its
                # model state is exactly the boundary being resumed.
                if epoch < state["epoch"]:
                    raise
                continue
            if epoch in by_epoch:
                raise ValueError("duplicate calibration prefix across attempts")
            calibrations.append(ref)
            by_epoch[epoch] = ref
    for epoch in EPOCHS:
        if epoch < state["epoch"] and epoch not in by_epoch:
            raise ValueError("resume would lose an earlier calibration checkpoint")
    for epoch, ref in by_epoch.items():
        calibration_record(ref, binding, data, snapshot=positions[epoch])
    return state, snapshots, calibrations


def run_schedule(output, kernel, data, snapshots, calibrations, *, should_stop):
    """Same real schedule for campaign4096 and explicitly mechanical32 fixtures."""
    from .learned_partition_training import collate_inputs, collate_targets
    from .learned_partition_snapshots import write_snapshot, read_snapshot
    if len(data["train"]["inputs"]) != kernel.count or len(data["train"]["targets"]) != kernel.count:
        raise ValueError("schedule data differs from the kernel scene roster")
    evaluated = {p.read_reference(ref)["binding"]["epoch"] for ref in calibrations}
    while True:
        if kernel.next_batch == 0 and kernel.epoch in EPOCHS and kernel.epoch not in evaluated:
            calibrations.append(save_calibration(output, kernel.epoch, snapshots[-1], kernel.model,
                                                 data["calibration"], kernel.binding))
            evaluated.add(kernel.epoch)
        if kernel.epoch == 50:
            break
        if should_stop():
            previous, _ = read_snapshot(snapshots[-1], expected_binding=kernel.binding)
            if previous["steps"] != kernel.steps:
                snapshots.append(write_snapshot(Path(output)/"snapshots", f"interrupted_{kernel.steps}", kernel, parents=[snapshots[-1]]))
            raise InterruptedError("training stopped at a complete update boundary")
        ids = kernel.expected_scene_ids()
        inputs = collate_inputs([data["train"]["inputs"][i] for i in ids])
        targets = collate_targets([data["train"]["targets"][i] for i in ids], inputs["candidate_mask"])
        kernel.step(inputs, targets, ids)
        if kernel.next_batch == 0 and kernel.epoch in EPOCHS:
            snapshots.append(write_snapshot(Path(output)/"snapshots", f"epoch_{kernel.epoch:02d}", kernel, parents=[snapshots[-1]]))
    return snapshots, calibrations


def train_cell(output, *, authorization, train, calibration, normalizers, normalized_train, normalized_calibration,
               arm, checkpoint_seed, reader_seed, gpu_grant, resume, request_ref, permit):
    started = time.monotonic()
    allowance = verify_permit(permit, request_ref)
    binding = cell_binding(authorization=authorization, train=train, calibration=calibration, normalizers=normalizers,
        normalized_train=normalized_train, normalized_calibration=normalized_calibration,
        arm=arm, checkpoint_seed=checkpoint_seed, reader_seed=reader_seed)
    data = load_cell_data(binding)
    restored, snapshots, calibrations = (None, [], []) if resume is None else resume_state(resume, binding, data["calibration"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    stop = {"requested": False}
    old_handler = signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    try:
        write_json(output/"ancestry.json", {"resume": resume, "snapshots": snapshots, "calibrations": calibrations})
        (output/"snapshots").mkdir()
        from .structured_source_profile import gpu_lease
        if binding["device"] == "cpu" and gpu_grant is not None:
            raise ValueError("CPU training cannot occupy a GPU lease")
        lease = gpu_lease(gpu_grant) if binding["device"] == "cuda:0" else nullcontext(None)
        with lease as availability:
            import torch
            from .learned_partition_training import TrainingKernel
            from .learned_partition_snapshots import write_snapshot, read_snapshot
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if binding["device"] == "cuda:0":
                torch.cuda.reset_peak_memory_stats(0)
            kernel = TrainingKernel(arm, reader_seed, binding=binding, device=binding["device"])
            if restored is None:
                snapshots.append(write_snapshot(output/"snapshots", "initial", kernel))
            else:
                kernel.restore(restored)
                del restored
            write_json(output/"training_ready.json", {"snapshot": snapshots[-1], "steps": kernel.steps, "binding": binding})
            run_schedule(output, kernel, data, snapshots, calibrations,
                should_stop=lambda: stop["requested"] or time.monotonic()-started > allowance["remaining_seconds"]-3)
            peak = torch.cuda.max_memory_reserved(0) if binding["device"] == "cuda:0" else 0
            if peak >= 2*1024**3:
                raise RuntimeError("training VRAM envelope exceeded")
            write_json(output/"history.json", kernel.history)
            write_json(output/"chain.json", {"initial": snapshots[0], "last_epoch": snapshots[-1],
                "snapshots": snapshots, "calibrations": calibrations})
        if cell_binding(authorization=authorization, train=train, calibration=calibration, normalizers=normalizers,
                normalized_train=normalized_train, normalized_calibration=normalized_calibration,
                arm=arm, checkpoint_seed=checkpoint_seed, reader_seed=reader_seed) != binding:
            raise ValueError("training binding changed")
        load_cell_data(binding)
        verify_permit(permit, request_ref)
        if time.monotonic()-started > allowance["remaining_seconds"]:
            raise TimeoutError("training and final validation exceed remaining budget")
        seal_bundle(output, role="learned_training_cell", binding={"cell": binding, "request": request_ref,
            "permit": permit, "resume": resume}, resources={"seconds": time.monotonic()-started,
            "peak_reserved_bytes": peak, "availability": availability})
        return p.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
