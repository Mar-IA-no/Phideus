"""Opt-in GPU resource profile using mechanical fixtures, never study splits.

Not a campaign launcher. A path documenting the actual GPU authorization/lease
is mandatory; this operational receipt is not itself authorization. The parent
does not import torch or query GPU and bounds its single child to 570 seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ARMS = ("pairs_descriptors", "pairs_compatibility", "pairs_sham",
        "pairs_transitivity", "tokens_descriptors")
TRAIN_STEPS = 50 * (8192 // 128)
EVAL_STEPS_PER_RUN = 6 * (1024 // 128)  # Validation plus five held-out splits.


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def project_resource_cost(rows):
    if {r["arm"] for r in rows} != set(ARMS) or len(rows) != len(ARMS):
        raise ValueError("profile must include every arm exactly once")
    train = 3 * TRAIN_STEPS * sum(max(r["train_step_seconds"]) for r in rows)
    evaluation = 3 * EVAL_STEPS_PER_RUN * sum(max(r["eval_step_seconds"]) for r in rows)
    peak = max(r["peak_reserved_bytes"] for r in rows)
    return {"training_seconds": train, "evaluation_seconds": evaluation,
            "total_compute_seconds": train + evaluation, "peak_reserved_bytes": peak,
            "status": "REVIEW_REQUIRED" if train + evaluation > 24*3600 or peak >= 8*1024**3 else "WITHIN_ESTIMATE",
            "limitations": "Maximum of three timed steps per arm, not a runtime upper bound. Includes cached-batch transfer and optimizer step; excludes corpus preparation, disk loading, checkpoints, CPU reader/metrics and scheduling. Full runner must also enforce the aggregate 24 GPU-hour budget."}


def verify_authorization(output, authorization):
    expected = json.loads((output/"request.json").read_bytes())["authorization_sha256"]
    if hashlib.sha256(authorization.read_bytes()).hexdigest() != expected:
        raise RuntimeError("authorization receipt changed since request")


def worker(output, authorization):
    # This function is entered ONLY by the explicitly launched GPU child.
    verify_authorization(output, authorization)
    import torch
    from src.atencion_armonica.shared_partial_data import mechanical_fixture
    from src.atencion_armonica.partial_compatibility_learning import (
        ARMS as definitions, collate_observations, collate_targets, loss_components,
        objective, seed_cpu,
    )
    from src.atencion_armonica.pairformer import build_model, MODEL_CONFIGS
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("authorized CUDA device unavailable; no CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    # Full precision; the same settings must be retained in the campaign runner.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fixtures, batches, targets = {}, {}, {}
    for purpose, k in (("train", 3), ("eval", 4)):
        obs, truth = mechanical_fixture(k)
        fixtures[purpose] = {"observation": obs, "sidecar": truth}
        single = collate_observations([obs])
        # Repeat a computed fixture, not 128 independent scenes or test draws.
        batches[purpose] = {key: tensor.repeat((128,) + (1,)*(tensor.ndim-1))
                            for key, tensor in single.items()}
        targets[purpose] = collate_targets([truth["source_ids"]]*128, batches[purpose])
    write_json(output/"fixtures.json", fixtures)
    rows = []
    for arm in ARMS:
        model_name = definitions[arm][0]
        seed_cpu(2026090721)
        torch.cuda.manual_seed_all(2026090721)
        model = build_model(model_name).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.cuda.reset_peak_memory_stats(device)
        train_times, eval_times, losses = [], [], []
        model.train()
        for step in range(4):  # One warmup and three measured maximum-size steps.
            torch.cuda.synchronize(device)
            started = time.monotonic()
            batch = {key: tensor.to(device) for key, tensor in batches["train"].items()}
            target = targets["train"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            components = loss_components(logits, batch, target)
            loss = objective(components, arm)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("nonfinite profile gradient")
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.monotonic()-started
            losses.append({key: value.detach().cpu().tolist() for key, value in components.items()})
            if step:
                train_times.append(elapsed)
        train_logits = logits.detach().cpu()
        del batch, target, logits, components, loss
        model.eval()
        with torch.no_grad():
            for step in range(4):
                torch.cuda.synchronize(device)
                started = time.monotonic()
                batch = {key: tensor.to(device) for key, tensor in batches["eval"].items()}
                logits = model(batch)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("nonfinite profile evaluation")
                torch.cuda.synchronize(device)
                if step:
                    eval_times.append(time.monotonic()-started)
        row = {"arm": arm, "model": model_name, "train_n": 24, "eval_n": 32,
               "batch_size": 128, "train_step_seconds": train_times,
               "eval_step_seconds": eval_times, "four_profile_steps_components": losses,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        with (output/f"{arm}_profile_state.pt").open("xb") as handle:
            torch.save({"scope": "mechanical_profile_four_steps_not_campaign_checkpoint",
                        "arm": arm, "config": MODEL_CONFIGS[model_name], "initial": initial,
                        "after_profile": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device),
                        "train_logits": train_logits, "eval_logits": logits.detach().cpu()}, handle)
        write_json(output/f"{arm}_profile.json", row)
        rows.append(row)
        del model, optimizer, initial, logits, batch, train_logits
        torch.cuda.empty_cache()
    result = {"scope": "mechanical_GPU_profile_not_experiment",
              "torch_version": torch.__version__, "device_name": torch.cuda.get_device_name(device),
              "device_total_memory": torch.cuda.get_device_properties(device).total_memory,
              "tf32": False, "dtype": "float32", "rows": rows,
              "projection": project_resource_cost(rows)}
    write_json(output/"profile.json", result)
    if result["projection"]["status"] != "WITHIN_ESTIMATE":
        raise RuntimeError("resource profile requires review before campaign")


def launch(output, authorization):
    # No model/GPU import before validating the operational inputs.
    if not authorization.is_file() or not authorization.read_bytes().strip():
        raise ValueError("a nonempty existing authorization/lease receipt is required")
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT/"experiments/atencion_armonica/PLAN_SHARED_PARTIAL_COMPATIBILITY.md",
               *[ROOT/"src/atencion_armonica"/name for name in (
                   "shared_partial_data.py", "partial_compatibility.py",
                   "partial_compatibility_learning.py", "pairformer.py", "peak_tokens.py")]]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(output/"request.json", {"scope": "mechanical_GPU_profile", "timeout_seconds": 570,
               "authorization_sha256": hashlib.sha256(authorization.read_bytes()).hexdigest(),
               "source_sha256": hashes})
    started = time.monotonic()
    try:
        subprocess.run([sys.executable, __file__, "--worker", "--output", str(output),
                        "--authorization", str(authorization)], timeout=570, check=True)
        verify_authorization(output, authorization)
        if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != sha for p, sha in hashes.items()):
            raise RuntimeError("source changed during resource profile")
        write_json(output/"manifest.json", {"scope": "GPU_profile_not_full_experiment", "status": "COMPLETE",
                   "seconds": time.monotonic()-started, "source_sha256": hashes,
                   "artifacts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(output.iterdir())}})
    except BaseException as exc:
        write_json(output/"FAILURE.json", {"status": "INCOMPLETE", "error": repr(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.authorization.is_file() or not args.authorization.read_bytes().strip():
            parser.error("missing authorization receipt")
        worker(args.output, args.authorization)
    else:
        launch(args.output, args.authorization)
