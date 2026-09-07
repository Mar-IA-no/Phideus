"""Fixed-recipe training cell; orchestration/test unlocking belongs to the runner.

Production training explicitly requires CUDA. CPU unit tests exercise ordering,
schedule and persistence utilities, never the 50-epoch training loop.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .partial_compatibility_cache import load_supervision, sha_file
from .partial_compatibility_learning import (
    ARMS, collate_cached_observations, collate_targets, loss_components, objective, seed_cpu,
)
from .pairformer import build_model, MODEL_CONFIGS

SEEDS = (2026090721, 2026090722, 2026090723)
RECIPE = {"epochs": 50, "batch_size": 128, "lr": 3e-4, "weight_decay": 1e-4,
          "warmup_fraction": .05, "dtype": "float32", "tf32": False,
          "batch_order_salt": 2026090740}


def epoch_order(seed, epoch, count=8192):
    if (seed not in SEEDS or type(epoch) is not int or not 0 <= epoch < RECIPE["epochs"]
            or type(count) is not int or count <= 0):
        raise ValueError("invalid training seed or epoch")
    return np.random.default_rng(np.random.SeedSequence([RECIPE["batch_order_salt"], seed, epoch])).permutation(count)


def lr_factor(step, total_steps=3200):
    if (type(total_steps) is not int or total_steps < 20
            or type(step) is not int or not 0 <= step <= total_steps):
        raise ValueError("schedule step outside recipe")
    warmup = int(RECIPE["warmup_fraction"] * total_steps)
    if step < warmup:
        return step / warmup
    return .5*(1+math.cos(math.pi*(step-warmup)/(total_steps-warmup)))


def model_digest(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_checkpoint(path, value):
    """Replace only this cell's rolling checkpoint after the new file is complete."""
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_resume(old, binding):
    """Only verified epoch boundaries can resume; mid-step states are diagnostic."""
    epoch = old.get("next_epoch")
    if (old.get("binding") != binding or old.get("resumable") is not True
            or type(epoch) is not int or not 0 <= epoch < RECIPE["epochs"]
            or old.get("next_batch") != 0 or old.get("steps") != epoch*64
            or old.get("scheduler", {}).get("last_epoch") != epoch*64
            or not math.isfinite(old.get("elapsed_total_seconds", float("nan")))
            or old["elapsed_total_seconds"] < 0):
        raise ValueError("resume must match the recipe and an unfinished whole-epoch checkpoint")


def training_cell(output, cache, arm, seed, source_hashes, *, remaining_seconds, resume=None):
    """Resolve supervision from its verified cache, never from caller-supplied labels."""
    if arm not in ARMS or seed not in SEEDS or cache.manifest["split"] != "train" or len(cache) != 8192:
        raise ValueError("cell does not match the fixed experiment")
    truths = load_supervision(cache)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        return _training_cell(output, cache, truths, arm, seed, source_hashes,
                              remaining_seconds=remaining_seconds, resume=resume)
    except BaseException as exc:
        if not (output/"FAILURE.json").exists():
            with (output/"FAILURE.json").open("x") as handle:
                json.dump({"status": "INCOMPLETE", "error": repr(exc),
                           "seconds_this_attempt": time.monotonic()-started,
                           "last_epoch_available": (output/"last_epoch.pt").exists()}, handle, sort_keys=True)
        raise


def _training_cell(output, cache, truths, arm, seed, source_hashes, *, remaining_seconds, resume=None):
    """Run one 50-epoch cell, keeping completed epochs on interruption.

    `source_hashes` are frozen by the campaign coordinator before launch. A
    resumed attempt gets a NEW output directory and links its old checkpoint;
    it never overwrites a closed or incomplete previous attempt.
    """
    if arm not in ARMS or seed not in SEEDS or cache.manifest["split"] != "train" or len(cache) != 8192:
        raise ValueError("cell does not match the fixed experiment")
    if not 0 < remaining_seconds <= 24*3600:
        raise ValueError("invalid remaining campaign budget")
    if len(truths) != len(cache):
        raise ValueError("misaligned supervision")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training; no CPU fallback")
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_cpu(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(ARMS[arm][0]).to("cuda:0")
    initial_digest = model_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    binding = {"arm": arm, "seed": seed, "recipe": RECIPE, "model_config": MODEL_CONFIGS[ARMS[arm][0]],
               "train_manifest_sha256": sha_file(cache.root/"manifest.json"), "source_sha256": source_hashes,
               "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                           "device": torch.cuda.get_device_name(0)},
               "initial_model_sha256": initial_digest}
    next_epoch = next_batch = steps = 0
    previous_elapsed = 0.0
    if resume is not None:
        old = torch.load(resume, map_location="cpu", weights_only=False)
        validate_resume(old, binding)
        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        scheduler.load_state_dict(old["scheduler"])
        torch.set_rng_state(old["cpu_rng"])
        torch.cuda.set_rng_state(old["cuda_rng"], device=0)
        next_epoch, next_batch, steps = old["next_epoch"], old["next_batch"], old["steps"]
        previous_elapsed = old["elapsed_total_seconds"]
    meta = {**binding, "resume_from": None if resume is None else str(resume),
            "resume_sha256": None if resume is None else sha_file(Path(resume)),
            "torch_version": torch.__version__, "numpy_version": np.__version__}
    with (output/"config.json").open("x") as handle:
        json.dump(meta, handle, sort_keys=True, indent=2, allow_nan=False)

    def checkpoint():
        return {"binding": binding, "resumable": next_batch == 0,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device=0), "next_epoch": next_epoch,
                "next_batch": next_batch, "steps": steps,
                "elapsed_total_seconds": previous_elapsed + time.monotonic()-started}

    atomic_checkpoint(output/"last_epoch.pt", checkpoint())
    model.train()
    try:
        with (output/"curve.jsonl").open("x") as curve:
            for epoch in range(next_epoch, RECIPE["epochs"]):
                order = epoch_order(seed, epoch)
                start_batch = next_batch if epoch == next_epoch else 0
                for batch_index in range(start_batch, 64):
                    if time.monotonic()-started >= remaining_seconds:
                        raise TimeoutError("remaining aggregate GPU budget exhausted")
                    ids = order[batch_index*128:(batch_index+1)*128]
                    cpu_batch = collate_cached_observations([cache.records[i] for i in ids])
                    targets = collate_targets([truths[i]["source_ids"] for i in ids], cpu_batch).to("cuda:0")
                    batch = {key: value.to("cuda:0") for key, value in cpu_batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    components = loss_components(model(batch), batch, targets)
                    loss = objective(components, arm)
                    norms = None
                    if batch_index == 0:
                        norms = {}
                        for key, value in components.items():
                            gradients = torch.autograd.grad(value.mean(), tuple(model.parameters()), retain_graph=True, allow_unused=True)
                            norms[key] = float(torch.sqrt(sum(g.square().sum() for g in gradients if g is not None)))
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError("nonfinite training gradient")
                    applied_lr = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    steps += 1
                    next_epoch, next_batch = (epoch+1, 0) if batch_index == 63 else (epoch, batch_index+1)
                    row = {"epoch": epoch+1, "batch": batch_index, "step": steps, "lr": applied_lr,
                           "scene_ids": ids.tolist(), "objective": float(loss.detach()),
                           "components": {k: float(v.detach().mean()) for k, v in components.items()},
                           "gradient_norms": norms}
                    curve.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                curve.flush()
                atomic_checkpoint(output/"last_epoch.pt", checkpoint())
                if epoch+1 in (10, 25, 50):
                    atomic_checkpoint(output/f"epoch_{epoch+1:02d}.pt", checkpoint())
                print(json.dumps({"arm": arm, "seed": seed, "epoch": epoch+1, "steps": steps,
                                  "seconds": time.monotonic()-started}), flush=True)
        if steps != 3200:
            raise RuntimeError("wrong number of optimizer steps")
        return {"status": "TRAINED_NOT_EVALUATED", "binding": binding, "steps": steps,
                "seconds_this_attempt": time.monotonic()-started,
                "elapsed_total_seconds": previous_elapsed+time.monotonic()-started,
                "last_epoch_sha256": sha_file(output/"last_epoch.pt")}
    except BaseException as exc:
        # Never serialize a possibly half-updated optimizer as a resumable state.
        # The last_epoch file remains the last fully completed boundary. A new
        # attempt may replay at most one epoch and must count that spent budget.
        with (output/"FAILURE.json").open("x") as handle:
            json.dump({"status": "INCOMPLETE", "error": repr(exc), "steps_observed": steps,
                       "seconds_this_attempt": time.monotonic()-started,
                       "resume_only_from": "last_epoch.pt"}, handle, sort_keys=True)
        raise
