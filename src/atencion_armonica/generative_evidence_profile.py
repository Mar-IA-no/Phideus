"""Bounded CPU/GPU and lossless-I/O profile on OPEN train scenes 0..31.

No new observations, test access, campaign training or scientific selection.
The 25 mechanical updates per case are disposable timing states, not cells.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import time

import numpy as np

from . import generative_evidence as ge
from . import generative_evidence_cache as cache
from .generative_evidence_reuse import OpenReuse, ROOT
from .generative_evidence_storage import atomic_bytes, read_arrays, read_scene, write_arrays, write_json, write_scene
from .partial_compatibility_cache import encoded

PROTOCOL = "experiments/atencion_armonica/PROTOCOL_GENERATIVE_EVIDENCE_READER.md"
PROTOCOL_SHA = "5d37f74e3495f885ff4fc8b673010e2ef6deb786e76f95259309ea1cbb8541e4"
CORE_SHA = "e57b93f3894e14f1923e1111231041497b0d634678311ac1fb520bdc359b25f8"


def source_binding():
    # One inexpensive package snapshot per profile boundary. Include helpers
    # imported transitively, not merely the new wrappers with visible entrypoints.
    paths = sorted({PROTOCOL, "experiments/atencion_armonica/profile_generative_evidence.py",
                    *[p.relative_to(ROOT).as_posix() for p in (ROOT/"src/atencion_armonica").glob("*.py")]})
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    if hashes[PROTOCOL] != PROTOCOL_SHA or hashes["src/atencion_armonica/observable_source_rivals.py"] != CORE_SHA:
        raise ValueError("frozen protocol/core changed")
    return {"purpose": "OPEN_TRAIN_RESOURCE_PROFILE_NOT_CAMPAIGN", "sources": hashes,
            "scene_ids": list(range(32)), "split": "train", "split_seed": cache.SPLITS["train"][1]}


def envelope_inputs():
    """Conservative dense tensor envelope, not a realizable partition/observation."""
    return {"groups": np.sin(np.arange(328*9, dtype=np.float32)).reshape(328, 9),
            "globals": np.cos(np.arange(82*17, dtype=np.float32)).reshape(82, 17),
            "evidence": np.sin(np.arange(82*6, dtype=np.float32)).reshape(82, 6),
            "incidence": np.full((82, 328), 1/328, np.float32)}


def gpu_availability():
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                         "--format=csv,noheader"], text=True, timeout=5)
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.used",
                                         "--format=csv,noheader"], text=True, timeout=5)
    if processes.strip():
        raise RuntimeError("GPU has another compute owner; profile did not load CUDA")
    if "RTX 3090" not in inventory or len(inventory.strip().splitlines()) != 1:
        raise RuntimeError("local single RTX 3090 identity differs")
    return {"processes": processes, "inventory": inventory}


def run_profile(output, *, device):
    if device not in ("cpu", "cuda:0"):
        raise ValueError("explicit CPU or local CUDA device required")
    output = Path(output).resolve()
    temporary = ROOT/".agent-work/phideus-generative-evidence-20260909"
    if not output.is_relative_to(temporary) or output == temporary:
        raise ValueError("profile output must be a new owned project temporary directory")
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("profile requires the three CPU thread environment limits set before import")
    availability = gpu_availability() if device == "cuda:0" else None
    binding = source_binding()
    output.mkdir(exist_ok=False)
    write_json(output/"owner.json", {"owner": "Phideus Codex", "purpose": binding["purpose"], "device": device})
    started = time.monotonic()
    report = {"binding": binding, "device": device, "availability": availability,
              "fit_scenes": [], "heads": {}, "status": "RUNNING"}
    def stop(signum, frame):
        raise TimeoutError("bounded profile interrupted or reached120s; no campaign state exists")
    previous = {s: signal.signal(s, stop) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL, 118.)  # Leave receipt time within120s.
    try:
        import torch
        from .generative_evidence_model import collate
        from .generative_evidence_training import TrainingKernel, collate_targets
        from .generative_evidence_supervision import open_truths, candidate_supervision
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if device == "cuda:0":
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
                raise ValueError("deterministic CUDA workspace must be set before initialization")
            torch.cuda.set_device(0)
            torch.cuda.set_per_process_memory_fraction(.25, 0)
            torch.cuda.reset_peak_memory_stats(0)
        report["runtime"] = {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
                             "cuda": torch.version.cuda if device == "cuda:0" else None,
                             "cpu_threads": torch.get_num_threads(), "deterministic": True, "tf32": False}
        def check():
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            reserved = torch.cuda.max_memory_reserved(0) if device == "cuda:0" else 0
            if rss > 6*1024**3 or reserved > 6*1024**3 or time.monotonic()-started > 118:
                raise RuntimeError("profile resource envelope exceeded")
        def sync():
            if device == "cuda:0":
                torch.cuda.synchronize()
        def snapshot(folder, name, kernel):
            stream = BytesIO()
            torch.save(kernel.state(), stream)
            ref = atomic_bytes(folder/name, stream.getvalue())
            raw = (folder/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
                raise ValueError("mechanical snapshot bytes changed")
            kernel.restore(torch.load(BytesIO(raw), map_location="cpu", weights_only=False))
            return ref
        def head_case(name, inputs, targets):
            before = time.monotonic()
            folder = output/name
            folder.mkdir()
            kernel = TrainingKernel("generative", ge.CHECKPOINTS[0], ge.READER_SEEDS[0],
                                    binding={**binding, "case": name}, scene_ids=list(range(len(inputs))), device=device)
            initial = snapshot(folder, "initial.pt", kernel)
            setup = time.monotonic()-before
            updates = []
            for i in range(25):
                before = time.monotonic()
                ids = kernel.expected_scene_ids()
                batch = collate([inputs[j] for j in ids])
                target = collate_targets([targets[j] for j in ids], batch["candidate_mask"])
                kernel.step(batch, target, ids)
                sync()
                updates.append(time.monotonic()-before)
                check()
            evaluation = []
            for i in range(3):
                before = time.monotonic()
                batch = {k: v.to(device) for k, v in collate(inputs).items()}
                with torch.no_grad():
                    predicted = kernel.model(batch).cpu().numpy()
                ref = write_arrays(folder/f"predictions_{i}.npz", {"components": predicted,
                                   "candidate_mask": batch["candidate_mask"].cpu().numpy()})
                np.testing.assert_array_equal(read_arrays(folder/f"predictions_{i}.npz", ref)["components"], predicted)
                evaluation.append(time.monotonic()-before)
                check()
            before = time.monotonic()
            last = snapshot(folder, "last.pt", kernel)
            snapshot_seconds = time.monotonic()-before
            value = {"status": "MECHANICAL_ONLY_NOT_TRAINED_CELL", "scene_count": len(inputs),
                     "candidate_counts": [len(r["globals"]) for r in inputs],
                     "group_counts": [len(r["groups"]) for r in inputs], "setup_seconds": setup,
                     "all_update_seconds": updates, "steady_update_seconds": updates[5:],
                     "evaluation_batch_io_seconds": evaluation, "snapshot_io_seconds": snapshot_seconds,
                     "initial": initial, "last": last}
            write_json(folder/"timings.json", value)
            return value
        x = envelope_inputs()
        report["heads"]["envelope"] = head_case("envelope", [x]*32,
                                                 [np.linspace(0, 1, 164, dtype=np.float32).reshape(82, 2)]*32)
        before = time.monotonic()
        reuse = OpenReuse()
        shard = reuse.shard("train", 0)
        truths = open_truths(shard)  # OPEN development only, not passed to fitter/features.
        report["reuse_setup_seconds"] = time.monotonic()-before
        fitter = ge.law.GroupFitter(ge.law.Grid(257, 65, 4), device="cuda" if device == "cuda:0" else "cpu")
        rows, observations, targets = [], [], []
        for scene_id in range(32):
            before = time.monotonic()
            scene = shard.scene(scene_id)
            input_seconds = time.monotonic()-before
            before = time.monotonic()
            fitted = ge.law.fit_candidates(scene["q32"], scene["partitions"], fitter)
            sync()
            fit_seconds = time.monotonic()-before
            before = time.monotonic()
            f = scene["features"]
            row = ge.observable_rows(np.asarray(scene["observation"]["log_f"], np.float32), scene["logits"][ge.CHECKPOINTS[0]],
                                     f["triples"], f["residual_cents"], scene["partitions"], fitted["fits"])
            labels = truths[scene_id]["labels"]  # Already reconstructed in canonical order.
            targets.append(candidate_supervision(row["partitions"], labels)["targets"])
            rows.append(row)
            observations.append(scene["observation"])
            # Preserve complete factors/assignments/witnesses, not a reduced view.
            value = {"observation": scene["observation"], "inventory": scene["inventory"], **fitted}
            ref = write_scene(output/f"scene_{scene_id:05d}.json.gz", value)
            read_scene(output/f"scene_{scene_id:05d}.json.gz", ref)
            arrays = cache.pack_rows("train", ge.CHECKPOINTS[0], [scene["observation"]], [row], binding=binding)
            array_ref = write_arrays(output/f"row_{scene_id:05d}.npz", arrays)
            cache.unpack_rows(read_arrays(output/f"row_{scene_id:05d}.npz", array_ref), binding=binding,
                              split="train", checkpoint_seed=ge.CHECKPOINTS[0])
            timing = {"scene_id": scene_id, "n": row["n"], "candidates": len(row["partitions"]),
                      "groups": len(row["groups"]), "input_seconds": input_seconds, "fit_seconds": fit_seconds,
                      "features_supervision_io_validation_seconds": time.monotonic()-before,
                      "scene": ref, "arrays": array_ref}
            report["fit_scenes"].append(timing)
            write_json(output/f"receipt_{scene_id:05d}.json", timing)
            print(json.dumps({"device": device, "profile_scene": scene_id, "fit_seconds": fit_seconds}), flush=True)
            check()
        before = time.monotonic()
        common = ge.fit_common_normalizer(lambda: iter(rows))
        evidence = ge.fit_evidence_normalizer(lambda: iter(rows))
        write_json(output/"profile_only_normalizers.json", {"purpose": "32_SCENE_TIMING_ONLY_NOT_CAMPAIGN_NORMALIZERS",
                                                            "common": common, "evidence": evidence})
        actual, actual_targets = [], []
        for i, row in enumerate(rows):
            if row["partitions"]:
                actual.append(ge.model_inputs(row, common, evidence, "generative",
                    split_seed=cache.SPLITS["train"][1], scene_id=i))
                actual_targets.append(targets[i])
        report["profile_normalization_seconds"] = time.monotonic()-before
        report["heads"]["observed_train"] = head_case("observed_train", actual, actual_targets)
        report["reuse"] = reuse.receipt()
        if source_binding() != binding:
            raise ValueError("profile source snapshot changed during execution")
        check()
        report["status"] = "MEASURED"
    except BaseException as exc:
        report["status"] = "INCOMPLETE"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report["seconds"] = time.monotonic()-started
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if "torch" in locals() and device == "cuda:0" and torch.cuda.is_initialized():
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        write_json(output/"report.json", report)
        for s, handler in previous.items():
            signal.signal(s, handler)
    return report
