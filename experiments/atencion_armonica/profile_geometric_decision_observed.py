"""Bounded full-path GPU profile on already-open TRAIN observations only."""
from __future__ import annotations

import time
LAUNCH_STARTED = time.monotonic()

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal

import numpy as np
import torch

from experiments.atencion_armonica.prepare_geometric_decision_open import ROOT, PROTOCOL, PROTOCOL_SHA, reference, code_snapshot
from experiments.atencion_armonica.profile_geometric_decision import admitted_open
from experiments.atencion_armonica.select_geometric_decision import admitted_training
from src.atencion_armonica.geometric_decision_archive_admission import admitted_selection, verify_heads
from src.atencion_armonica.geometric_decision_budget import StageBudget, BudgetExceeded, owned_bytes
from src.atencion_armonica.geometric_decision_observed_run import run_observed
from src.atencion_armonica.geometric_decision_observables import validate_observation
from src.atencion_armonica.geometric_decision_open import ReadOnlyStore
from src.atencion_armonica.geometric_decision_store import ArtifactStore, BASES
from src.atencion_armonica.generative_evidence_reuse import OpenReuse, OPEN_SPLITS
from src.atencion_armonica.generative_evidence_profile import gpu_availability
from src.atencion_armonica.structured_source_runner import checkpoint_forward, gpu_runtime
from src.atencion_armonica import generative_evidence as ge
# Resolve the forward's local lazy imports before source pinning (no CUDA).
from src.atencion_armonica import pairformer, partial_compatibility_inference, partial_compatibility_learning

OPERATOR = "experiments/atencion_armonica/profile_geometric_decision_observed.py"
PLAN = "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_OBSERVED_PROFILE.md"
ARCHIVE_FINISH = {"path": "attempts/0009/finish.json", "bytes": 533,
    "sha256": "158fe1bc539f0c31e4baaa6191ba963d3cb6573531f8865dd146c1a933625e1c"}
COUNT, RESERVATION = 16, 480.
PHASE_UNITS = {
    "archive": {"units": 144, "unit_kind": "archived_head_state"},
    "original-observable": {"units": COUNT, "unit_kind": "observed_scene", "backbones": 3},
    "original-classical": {"units": COUNT, "unit_kind": "observed_scene", "references": 4},
    "original-readout": {"units": COUNT, "unit_kind": "observed_scene", "head_states": 144, "transport_scenes": 4},
    "roundtrip-observable": {"units": 4, "unit_kind": "derived_scene", "backbones": 3},
    "roundtrip-classical": {"units": 4, "unit_kind": "derived_scene", "references": 4},
    "roundtrip-readout": {"units": 4, "unit_kind": "derived_scene", "head_states": 144},
    "observable-recovery": {"units": COUNT+4, "unit_kind": "original_plus_derived_scene", "head_states": 144},
}


def sources():
    refs = [*code_snapshot(), *[reference(p) for p in (OPERATOR, PLAN,
        "experiments/atencion_armonica/profile_geometric_decision.py",
        "experiments/atencion_armonica/select_geometric_decision.py")]]
    return sorted(refs, key=lambda r: r["path"])


def admitted_archive(control, selection, selection_ref, selected):
    finish = control.json(ARCHIVE_FINISH)
    start = control.json(finish["start"])
    manifest = control.json(start["manifest"])
    if finish["status"] != "COMPLETE" or finish["completion"] is None:
        raise ValueError("archive operation is not COMPLETE")
    output = control.json(finish["completion"])
    if (start["binding"] != control.binding or start["stage"] != "fresh"
            or manifest["operation"] != "archive-and-exclusions" or output["manifest"] != start["manifest"]
            or output["root"] != str(BASES[0]/"archive") or manifest["root"] != output["root"]):
        raise ValueError("archive completion authority differs")
    archive = ReadOnlyStore(output["root"], binding_ref=output["binding"])
    complete = archive.json(output["complete"])
    if (archive.binding != manifest["binding"] or archive.binding["admitted_selection"] != selected
            or archive.binding["selection"] != selection_ref or archive.binding["selection_binding"] != selection.binding
            or set(complete) != {"schema", "binding", "heads", "exclusions", "catalog", "fresh_tests", "test_authority"}
            or complete["schema"] != "geometric-decision-archive-complete-v1" or complete["binding"] != archive.binding
            or complete["fresh_tests"] != "not opened" or complete["test_authority"] is not False):
        raise ValueError("archive output is not the admitted calibrated roster")
    return archive, complete, {"finish": ARCHIVE_FINISH, "output": output}


def projection(timings, *, profile_bytes, overhead_seconds):
    if (set(timings) != set(PHASE_UNITS) or any(not np.isfinite(v) or v <= 0 for v in timings.values())
            or not np.isfinite(overhead_seconds) or overhead_seconds < 0):
        raise ValueError("full nonempty original/roundtrip/recovery timings required")
    original = sum(timings[k] for k in ("original-observable", "original-classical", "original-readout"))
    derived = sum(timings[k] for k in ("roundtrip-observable", "roundtrip-classical", "roundtrip-readout"))
    return {"schema": "geometric-decision-observed-projection-v1", "margin": 1.25,
        "measured_scenes": COUNT, "fresh_scenes": 2048, "test_scenarios": 4,
        "phase_units": PHASE_UNITS, "measured_unallocated_overhead_seconds": overhead_seconds,
        "overhead_allocation": "four full setup/IO overhead allowances in each projected stage before margin",
        "observed_path_seconds": 1.25*(original*2048/COUNT+4*(derived+timings["archive"]+overhead_seconds)),
        "observable_recovery_seconds": 1.25*(timings["observable-recovery"]*2048/COUNT+4*overhead_seconds),
        "projected_bytes": int(np.ceil(1.25*profile_bytes*2048/COUNT)),
        "overcount": "original transport repeated per16 rather than per512; recovery includes extra probes",
        "not_measured": ["fresh sampler/draw IO", "global seal", "privileged metrics/bootstrap",
                         "final report/source verification/finish IO after overhead snapshot (charged in finish)"],
        "test_authority": False}


def execute(control, store, manifest, *, preparation, open_ref, reuse, observations, archive,
            archive_complete, selection, selection_ref, verify, started_at, reservation=RESERVATION):
    if control.json(manifest) != {"operation": "profile-observed", "root": str(store.root), "binding": store.binding}:
        raise ValueError("full-path profile manifest differs")
    budget = StageBudget(control, "profile", manifest_ref=manifest, reservation_seconds=reservation,
        started_at=started_at, prior_charges=control.binding["prior_charges"],
        output_roots=[Path(p) for p in control.binding["output_roots"]],
        vram=lambda: torch.cuda.max_memory_reserved(0) if torch.cuda.is_initialized() else 0)
    def stop(signum, frame):
        if signum == signal.SIGALRM:
            raise BudgetExceeded("full-path profile deadline reached")
        raise InterruptedError("full-path profile paused; retain partial evidence")
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    timings = {}
    deadline = started_at+reservation
    def measure(name, operation):
        budget.check()
        if name in timings:
            raise ValueError("profile phase cannot repeat silently")
        began = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, max(.001, min(deadline-began, 120.)))
        result = operation()
        torch.cuda.synchronize()
        seconds = time.monotonic()-began
        if seconds >= 120.:
            raise BudgetExceeded("profile phase exceeded120s")
        timings[name] = seconds
        store.publish_json("timings/"+name+".json", {"phase": name, "seconds": seconds, **PHASE_UNITS[name]})
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
        budget.check()
        print(json.dumps({"phase": name, "seconds": seconds}), flush=True)
        return result
    try:
        for sig in old:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
        verify()
        # GPU owner was checked before manifest; recheck immediately before initialization.
        gpu_availability()
        runtime = gpu_runtime()
        if any(str(runtime[k]) != str(reuse.common["runtime"][k]) for k in ("torch", "numpy")):
            raise ValueError("GPU/package runtime differs from OPEN checkpoints")
        torch.cuda.set_per_process_memory_fraction(.25, 0)
        runtime_ref = store.publish_json("runtime.json", runtime)
        budget.check()
        obs_refs = [store.publish_json(f"observations/{o['scene_id']:05d}.json", o) for o in observations]
        norms = {k: deepcopy(preparation.source.corpus.norm[k]) for k in ("common", "evidence")}
        normalizer = store.publish_json("normalizers.json", {"source_root": str(preparation.source.store.root),
            "source": preparation.source.corpus.normalizer_ref, "normalizers": norms})
        scale_ref = preparation.completion(open_ref)["scale"]
        scale = preparation.store.json(scale_ref)["scale"]
        store.publish_json("scale.json", {"source_root": str(preparation.store.root), "source": scale_ref, "scale": scale})
        fitter = ge.law.GroupFitter(ge.law.Grid(257, 65, 4), device="cuda", assignment_batch=8)
        def forward(cp, records):
            budget.check()
            result = checkpoint_forward(cp, records, runtime)
            budget.check()
            return result
        def fit(q, partitions):
            budget.check()
            result = ge.law.fit_candidates(q, partitions, fitter)
            budget.check()
            return result
        args = dict(expected_seed=OPEN_SPLITS["train"][1], observation_origin={"kind": "original", "records": obs_refs},
            checkpoints=reuse.common["checkpoints"], runtime=runtime, forward=forward,
            normalizers=norms, normalization_ref=normalizer, scale=scale,
            fit_origin={"grid": {"beta": 257, "gamma": 65, "coarse_stride": 4}, "assignment_batch": 8,
                        "device": "cuda", "runtime": runtime_ref}, fit_candidates=fit,
            head_store=archive, archive_ref=archive_complete["heads"], selection_store=selection,
            selection_ref=selection_ref, device="cuda:0", check=budget.check)
        result = run_observed(store, "observed", "train", observations, **args, measure=measure)
        def forbidden(*args, **kwargs):
            raise AssertionError("observable recovery cannot forward or fit")
        recovered = measure("observable-recovery", lambda: run_observed(store, "observed", "train", observations,
            **{**args, "forward": forbidden, "fit_candidates": forbidden}, recovery_only=True))
        if recovered != result:
            raise ValueError("observable recovery changed the preserved complete result")
        value = store.json(result)
        if len(value["roundtrip_scene_ids"]) != 4:
            raise ValueError("fixed TRAIN profile lacks four eligible probes; do not substitute scenes")
        size = owned_bytes([store.root])
        elapsed = time.monotonic()-started_at
        overhead = elapsed-sum(timings.values())
        forecast = projection(timings, profile_bytes=size, overhead_seconds=overhead)
        report = store.publish_json("result.json", {"schema": "geometric-decision-observed-profile-v1",
            "binding": store.binding, "result": result, "recovery": recovered, "runtime": runtime_ref,
            "timings": timings, "phase_units": PHASE_UNITS, "profile_bytes": size, "forecast": forecast,
            "elapsed_to_forecast_seconds": elapsed, "unallocated_overhead_seconds": overhead,
            "vram_peak_bytes": torch.cuda.max_memory_reserved(0), "rss_peak_bytes": budget.rss(),
            "roundtrip_scene_ids": value["roundtrip_scene_ids"], "new_test_observations": 0,
            "exclusion_extension_required": True})
        verify()
        output = control.publish_json("outputs/profile-observed.json", {"manifest": manifest,
            "root": str(store.root), "binding": store.reference(store.path("binding.json")), "result": report})
        finish = budget.finish("COMPLETE", completion=output)
        return {"output": output, "finish": finish, "forecast": forecast}
    except BaseException as exc:
        if not budget.closed:
            budget.finish("LIMIT_REACHED" if isinstance(exc, BudgetExceeded) else "PAUSED"
                          if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        for sig, handler in old.items():
            signal.signal(sig, handler)


def main():
    if (Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0")
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))):
        raise ValueError("full profile requires explicit one-thread deterministic local GPU environment")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    control, preparation, open_ref = admitted_open()
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = BASES[0]/"profiles/observed-cuda-0"
        if root.exists():
            raise ValueError("full-path profile already attempted; no silent repetition")
        campaign, campaign_ref, training = admitted_training(control, root=BASES[0]/"training")
        selection, selection_ref, selected = admitted_selection(control, root=BASES[0]/"selection",
            campaign=campaign, campaign_ref=campaign_ref, training=training)
        archive, complete, archive_evidence = admitted_archive(control, selection, selection_ref, selected)
        reuse = OpenReuse()
        data = reuse.bundle(reuse.shards["train"][0]["data"], "learned_observation_shard")
        all_obs = [json.loads(line) for line in data.read("observations.jsonl").splitlines()]
        if len(all_obs) != 512:
            raise ValueError("OPEN profile shard is incomplete")
        for sid, obs in enumerate(all_obs):
            validate_observation(obs, scene_id=sid, split_seed=OPEN_SPLITS["train"][1])
        observations = all_obs[:COUNT]
        code, protocol = sources(), reference(PROTOCOL)
        if protocol["sha256"] != PROTOCOL_SHA:
            raise ValueError("frozen scientific protocol changed")
        availability = gpu_availability()
        binding = {"schema": "geometric-decision-observed-profile-binding-v1", "code": code, "protocol": protocol,
            "archive": archive_evidence, "open": open_ref, "observations": data.reference("observations.jsonl"),
            "scene_ids": list(range(COUNT)), "reuse": reuse.receipt(), "checkpoints": reuse.common["checkpoints"],
            "availability": availability, "test_authority": False}
        store = ArtifactStore(root, binding=binding)
        manifest = control.publish_json("manifests/profile-observed.json", {"operation": "profile-observed",
            "root": str(root), "binding": binding})
        def verify():
            if sources() != code or reference(PROTOCOL) != protocol:
                raise ValueError("profile code/protocol changed during execution")
        result = execute(control, store, manifest, preparation=preparation, open_ref=open_ref, reuse=reuse,
            observations=observations, archive=archive, archive_complete=complete, selection=selection,
            selection_ref=selection_ref, verify=verify, started_at=LAUNCH_STARTED)
        print(json.dumps({"status": "OBSERVED_PROFILE_COMPLETE_NOT_TEST_AUTHORITY", **result}), flush=True)


if __name__ == "__main__":
    main()
