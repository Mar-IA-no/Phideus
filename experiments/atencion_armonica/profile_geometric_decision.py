"""Resource-accounted profile operator. Never creates fresh test observations."""
from __future__ import annotations

import time
LAUNCH_STARTED = time.monotonic()

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import subprocess

import numpy as np
import torch

from experiments.atencion_armonica.prepare_geometric_decision_open import (
    ROOT, PROTOCOL, PROTOCOL_SHA, SOURCE_REFS, reference, code_snapshot,
)
from src.atencion_armonica.geometric_decision_budget import StageBudget, BudgetExceeded
from src.atencion_armonica.geometric_decision_corpus import OpenPreparation
from src.atencion_armonica.geometric_decision_open import OpenSource, ReadOnlyStore
from src.atencion_armonica.geometric_decision_store import ArtifactStore, BASES
from src.atencion_armonica import geometric_decision_profile as profile
from src.atencion_armonica.generative_evidence_profile import gpu_availability

CONTROL_BINDING = {"path": "binding.json", "bytes": 873,
    "sha256": "4e1ca362e88db55a1a600e245899d1e85a57ab0e57952bb75bc379861d440953"}
OPEN_BINDING = {"path": "binding.json", "bytes": 6796,
    "sha256": "035697f7b6f737040150a95e5c15853bbd4251aca46446d83c2c5256a73544f4"}
OPEN_FINISH = {"path": "attempts/0000/finish.json", "bytes": 474,
    "sha256": "4ef4cadde727a933833cac5c14adb708decb55f74f0d8e592ba32736bebddabe"}


def admitted_open():
    """Output presence alone is insufficient: validate the pinned COMPLETE chain."""
    view = ReadOnlyStore(BASES[0]/"control", binding_ref=CONTROL_BINDING)
    finish = view.json(OPEN_FINISH)
    if finish["status"] != "COMPLETE" or finish["completion"] is None:
        raise ValueError("OPEN operation was not admitted complete")
    start = view.json(finish["start"])
    output = view.json(finish["completion"])
    manifest = view.json(start["manifest"])
    source = OpenSource(ROOT/"data/atencion_armonica/generative_evidence_reader_v1", **SOURCE_REFS)
    prepared_view = ReadOnlyStore(BASES[0]/"open", binding_ref=OPEN_BINDING)
    if (start["stage"] != "open" or start["binding"] != view.binding
            or output["manifest"] != start["manifest"] or output["root"] != str(prepared_view.root)
            or manifest["operation"] != "open" or manifest["preparation_binding"] != prepared_view.binding
            or manifest["source_refs"] != SOURCE_REFS):
        raise ValueError("OPEN finish, manifest, source or prepared binding differs")
    prep = OpenPreparation(source, prepared_view)
    prep.completion(output["complete"])
    return ArtifactStore(view.root, binding=view.binding), prep, output["complete"]


def admitted_profile_inputs(control):
    matches = []
    for path in sorted(control.path("attempts").glob("*/finish.json")):
        ref = control.reference(path)
        finish = control.json(ref)
        start = control.json(finish["start"])
        manifest = control.json(start["manifest"])
        if manifest.get("operation") == "profile-inputs" and finish["status"] == "COMPLETE":
            output = control.json(finish["completion"])
            if output["manifest"] != start["manifest"] or output["root"] != str(BASES[0]/"profiles/inputs"):
                raise ValueError("profile input completion has another manifest/root")
            matches.append((ref, output))
    if len(matches) != 1:
        raise ValueError("exactly one admitted profile input preparation required")
    finish_ref, output = matches[0]
    store = ReadOnlyStore(output["root"], binding_ref=output["binding"])
    rows, provenance = profile.read_batch(store, output["result"])
    load = store.json(output["checkpoint_load"])
    return rows, {"finish": finish_ref, "output": output, "provenance": provenance, "checkpoint_load": load}


def sources():
    return sorted([*code_snapshot(), reference("experiments/atencion_armonica/profile_geometric_decision.py"),
                   reference("experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_PROFILES.md")], key=lambda r: r["path"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("inputs", "head", "fitter"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = parser.parse_args()
    if (Path.cwd().resolve() != ROOT
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            or (args.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "")
            or (args.device == "cuda:0" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8")
            or (args.task == "inputs" and args.device != "cpu")):
        raise RuntimeError("explicit one-thread deterministic device environment required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    control, preparation, complete_ref = admitted_open()
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        name = "inputs" if args.task == "inputs" else f"{args.task}-{args.device.replace(':', '-')}"
        root = BASES[0]/"profiles"/name
        if root.exists():
            raise ValueError("profile attempt already exists; no silent repetition or overwrite")
        # Resolve inputs and device identity BEFORE publishing the immutable
        # attempt manifest. LAUNCH_STARTED charges these reads to the profile.
        rows, provenance = (None, None) if args.task == "inputs" else admitted_profile_inputs(control)
        availability = gpu_availability() if args.device == "cuda:0" else None
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
            text=True, timeout=5) if args.device == "cuda:0" else None
        protocol, code = reference(PROTOCOL), sources()
        if protocol["sha256"] != PROTOCOL_SHA:
            raise ValueError("frozen protocol changed")
        runtime = {"torch": str(torch.__version__), "numpy": np.__version__, "python": platform.python_version(),
                   "device": args.device, "threads": 1, "deterministic": True, "tf32": False,
                   "cuda_build": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version() if args.device == "cuda:0" else None,
                   "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "availability": availability, "driver": driver}
        binding = {"purpose": "RESOURCE_PROFILE_NOT_CAMPAIGN", "task": args.task,
            "runtime": runtime, "code": code, "protocol": protocol, "open_finish": OPEN_FINISH,
            "open_binding": OPEN_BINDING, "complete": complete_ref, "input_provenance": provenance}
        manifest_ref = control.publish_json(f"manifests/profile-{name}.json",
            {"operation": f"profile-{args.task}", "binding": binding, "root": str(root)})
        reservation = 100. if args.task == "inputs" else 120.
        def vram():
            return torch.cuda.max_memory_reserved(0) if args.device == "cuda:0" and torch.cuda.is_initialized() else 0
        budget = StageBudget(control, "profile", manifest_ref=manifest_ref, reservation_seconds=reservation,
            started_at=LAUNCH_STARTED, prior_charges=control.binding["prior_charges"],
            output_roots=[Path(p) for p in control.binding["output_roots"]], vram=vram)
        def stop(signum, frame):
            if signum == signal.SIGALRM:
                raise BudgetExceeded("profile deadline reached")
            raise InterruptedError("profile interrupted; evidence and costs retained")
        old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        try:
            for sig in old:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, max(.001, reservation-(time.monotonic()-LAUNCH_STARTED)))
            store = ArtifactStore(root, binding=binding)
            if args.device == "cuda:0":
                torch.cuda.set_device(0)
                torch.cuda.set_per_process_memory_fraction(.25, 0)
                torch.cuda.reset_peak_memory_stats(0)
            budget.check()
            load_ref = equality_ref = None
            if args.task == "inputs":
                before = time.monotonic()
                data = preparation.load_checkpoint(complete_ref, profile.CHECKPOINTS[0], check=budget.check)
                load_ref = store.publish_json("checkpoint-load.json", {"seconds": time.monotonic()-before,
                    "data_binding": data.binding, "checkpoint_seed": profile.CHECKPOINTS[0],
                    "eligible_train": len(data.eligible["train"]), "eligible_calibration": len(data.eligible["calibration"])})
                rows, provenance = profile.extract_first_batch(preparation, complete_ref, check=budget.check)
                equality = []
                for row in rows:
                    original = data.rows["train"][row["scene_id"]]
                    digest = profile.assert_same_arrays({**row["inputs"], "targets": row["targets"]},
                        {**original["inputs"], "targets": original["targets"]})
                    if row["identity"] != original["identity"] or row["partitions"] != original["partitions"]:
                        raise ValueError("profile subset differs from full-checkpoint source identity")
                    equality.append({"scene_id": row["scene_id"], "full_subset_digest": digest})
                result = profile.save_batch(store, rows, provenance)
                restored, restored_provenance = profile.read_batch(store, result)
                if restored_provenance != provenance:
                    raise ValueError("profile subset provenance changed during persistence")
                for original, recovered, record in zip(rows, restored, equality):
                    record["roundtrip_digest"] = profile.assert_same_arrays(
                        {**original["inputs"], "targets": original["targets"], "q32": original["q32"]},
                        {**recovered["inputs"], "targets": recovered["targets"], "q32": recovered["q32"]})
                equality_ref = store.publish_json("subset-equality.json", {"method": "dtype.str, shape, C-order bytes",
                    "batch": result, "checkpoint_load": load_ref, "scenes": equality})
            else:
                if args.task == "head":
                    cases = []
                    for kind, case_rows in (("envelope", profile.envelope_rows()), ("first_train_batch", rows)):
                        for objective in ("mse", "decision"):
                            budget.check()
                            case = ArtifactStore(root/f"{kind}-{objective}", binding={**binding,
                                "input_provenance": provenance, "case": kind, "objective": objective})
                            ref = profile.head_case(case, case_rows, objective, args.device, check=budget.check)
                            cases.append({"case": kind, "objective": objective, "root": str(case.root), "result": ref})
                            print(json.dumps({"profile": name, "case": kind, "objective": objective, "status": "MEASURED"}), flush=True)
                    result = store.publish_json("result.json", {"binding": binding, "cases": cases})
                else:
                    workloads = profile.fitter_workloads(rows)
                    store.publish_json("workloads.json", workloads)
                    result = profile.fit_profile(store, workloads, args.device, check=budget.check)
            budget.check(force_resources=True)
            if sources() != code or reference(PROTOCOL) != protocol:
                raise ValueError("profile source code changed during operation")
            output = control.publish_json(f"outputs/profile-{name}.json", {"manifest": manifest_ref,
                "root": str(root), "binding": store.reference(store.path("binding.json")), "result": result,
                "checkpoint_load": load_ref, "subset_equality": equality_ref})
            finish = budget.finish("COMPLETE", completion=output)
            print(json.dumps({"profile": name, "status": "COMPLETE", "output": output, "finish": finish}), flush=True)
        except BaseException as exc:
            if not budget.closed:
                status = "LIMIT_REACHED" if isinstance(exc, BudgetExceeded) else "PAUSED" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED"
                budget.finish(status)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            for sig, handler in old.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
