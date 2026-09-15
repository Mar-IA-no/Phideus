"""CPU-only remaining-cost profile on16already-knownTRAIN scenes and4probes."""
from __future__ import annotations

import time
LAUNCH_STARTED = time.monotonic()

import fcntl
import json
import os
from pathlib import Path
import signal

import numpy as np
import torch

from experiments.atencion_armonica.prepare_geometric_decision_open import ROOT, PROTOCOL, PROTOCOL_SHA, reference, code_snapshot
from experiments.atencion_armonica.profile_geometric_decision import admitted_open, CONTROL_BINDING
from experiments.atencion_armonica.select_geometric_decision import admitted_training
from experiments.atencion_armonica.profile_geometric_decision_observed import admitted_archive
from src.atencion_armonica.geometric_decision_archive_admission import admitted_selection
from src.atencion_armonica.geometric_decision_observed_admission import admit_profile
from src.atencion_armonica.geometric_decision_budget import StageBudget, BudgetExceeded, owned_bytes
from src.atencion_armonica.geometric_decision_store import ArtifactStore, BASES
from src.atencion_armonica.geometric_decision_open import ReadOnlyStore
from src.atencion_armonica.geometric_decision_release import file_inventory
from src.atencion_armonica.geometric_decision_evaluation import evaluate_batch, labels_by_event, NAMES, save_arrays, save_json
from src.atencion_armonica.geometric_decision_metrics import primary
from src.atencion_armonica.generative_evidence_reuse import OpenReuse, VerifiedBytes
from src.atencion_armonica.generative_evidence_supervision import reconstruct_truth
from src.atencion_armonica.learned_partition_data import _draw_scene
from src.atencion_armonica.partial_compatibility_cache import encoded

OPERATOR = "experiments/atencion_armonica/profile_geometric_decision_closing.py"
PLAN = "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_CLOSING_PROFILE.md"
RESERVATION = 360.
UNITS = {"admission-setup": (1, "authenticated_preparation"),
    "open-truth": (16, "TRAIN_parses_from_two_complete_512row_files"), "known-draw-io": (16, "identical_TRAIN_reconstruction"),
    "original-metrics": (16, "scene_144heads"), "probe-metrics": (4, "probe_144heads"),
    "original-replay": (16, "scene_144heads"), "probe-replay": (4, "probe_144heads"),
    "bootstrap512": (512, "repeated_metric_fixture_not_observations"),
    "observable-inventory": (20, "original_plus_probe_file_tree")}


def sources():
    extra = (OPERATOR, PLAN, "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_RELEASE.md",
        "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_METRICS.md",
        "experiments/atencion_armonica/profile_geometric_decision_observed.py",
        "experiments/atencion_armonica/profile_geometric_decision.py",
        "experiments/atencion_armonica/select_geometric_decision.py")
    return sorted([*code_snapshot(), *[reference(p) for p in extra]], key=lambda r: r["path"])


def projection(timings, observed, *, overhead, profile_bytes):
    if (set(timings) != set(UNITS) or any(type(t) not in (float, int) or not np.isfinite(t) or t <= 0 for t in timings.values())
            or not np.isfinite(overhead) or overhead < 0):
        raise ValueError("remaining-cost projection needs all finite phases and overhead")
    scale, margin = 2048/16, 1.25
    inventory = timings["observable-inventory"]*scale
    setup = timings["admission-setup"]+overhead
    fresh_extra = margin*(4*scale*timings["known-draw-io"]+inventory+4*setup)
    eval_extra = margin*(scale*(timings["original-metrics"]+timings["original-replay"]+timings["open-truth"])
        +4*(timings["probe-metrics"]+timings["probe-replay"])+5*inventory+2*timings["bootstrap512"]+4*setup)
    return {"schema": "geometric-decision-closing-projection-v1", "margin": margin,
        "fresh_scenes": 2048, "original_profile_scenes": 16, "probe_profile_scenes": 4,
        "fresh_seconds": observed["observed_path_with_closing_seconds"]+observed["observable_recovery_with_closing_seconds"]+fresh_extra,
        "evaluation_seconds": observed["observable_recovery_with_closing_seconds"]+eval_extra,
        "projected_new_bytes": observed["projected_profile_bytes"]+int(np.ceil(margin*profile_bytes*scale)),
        "fresh_additional_seconds": fresh_extra, "evaluation_additional_seconds": eval_extra,
        "preseal_recovery_count": 1, "postseal_recovery_count": 1,
        "evaluation_inventory_count": 5,
        "units": {k: {"count": n, "kind": kind} for k, (n, kind) in UNITS.items()},
        "not_a_worst_case_bound": True, "test_authority": False,
        "closing_tail": "add this operator finish minus forecast snapshot before admission"}


def admit_inputs(control, *, check):
    """Material admission runs only inside the measured, guarded operation."""
    check()
    opened, _, _ = admitted_open()
    if opened.root != control.root or opened.binding != control.binding:
        raise ValueError("OPEN returned a different control")
    check()
    campaign, campaign_ref, training = admitted_training(control, root=BASES[0]/"training")
    check()
    selection, selection_ref, selected = admitted_selection(control, root=BASES[0]/"selection",
        campaign=campaign, campaign_ref=campaign_ref, training=training)
    check()
    archive, complete, evidence = admitted_archive(control, selection, selection_ref, selected)
    check()
    admitted = admit_profile(control, archive, complete, evidence, VerifiedBytes(ROOT), check=check)
    reuse = OpenReuse()
    check()
    data = reuse.bundle(reuse.shards["train"][0]["data"], "learned_observation_shard")
    check()
    return admitted, data


def execute(control, store, manifest, *, started_at, verify, admit=admit_inputs):
    if control.json(manifest) != {"operation": "profile-closing", "binding": store.binding, "root": str(store.root)}:
        raise ValueError("closing profile manifest differs")
    budget = StageBudget(control, "profile", manifest_ref=manifest, reservation_seconds=RESERVATION,
        started_at=started_at, prior_charges=control.binding["prior_charges"],
        output_roots=[Path(p) for p in control.binding["output_roots"]])
    deadline, timings = started_at+RESERVATION, {}
    def stop(signum, frame):
        if signum == signal.SIGALRM:
            raise BudgetExceeded("closing profile deadline")
        raise InterruptedError("closing profile paused; preserve partial evidence")
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    def measure(name, fn):
        budget.check()
        if name in timings:
            raise ValueError("profile phase cannot repeat silently")
        start = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, max(.001, min(120., deadline-start)))
        value = fn()
        seconds = time.monotonic()-start
        if seconds >= 120.:
            raise BudgetExceeded("closing profile phase exceeded120s")
        timings[name] = seconds
        n, kind = UNITS[name]
        store.publish_json(f"timings/{name}.json", {"seconds": seconds, "units": n, "kind": kind})
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
        budget.check()
        print(json.dumps({"phase": name, "seconds": seconds}), flush=True)
        return value
    try:
        for sig in old:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
        verify()
        def setup():
            admitted, data = admit(control, check=budget.check)
            receipt = store.publish_json("admission.json", {"binding": store.binding,
                "observed_profile": admitted["forecast"], "open_train_data": data.ref,
                "observations": data.reference("observations.jsonl"), "sidecars": data.reference("sidecars.jsonl")})
            return admitted, data, receipt
        admitted, data, admission_ref = measure("admission-setup", setup)
        profile, report = admitted["profile"], admitted["report"]
        observed = profile.json(report["result"])
        originals = profile.json(observed["sources"])["sources"]
        def open_labels():
            obs_lines, truth_lines = (data.read(name).splitlines() for name in ("observations.jsonl", "sidecars.jsonl"))
            if len(obs_lines) != 512 or len(truth_lines) != 512:
                raise ValueError("known TRAIN shard is incomplete")
            observations, truths, labels, scenes = [], [], {}, {}
            for sid in range(16):
                budget.check()
                observation, truth = json.loads(obs_lines[sid]), json.loads(truth_lines[sid])
                scene = profile.json(originals[sid])["scene"]
                if scene["observation"] != observation or observation["scene_id"] != sid or observation["split_seed"] != 2026090880:
                    raise ValueError("closing profile observation differs from knownTRAIN")
                labels[sid] = reconstruct_truth(observation, truth, "train")["labels"]
                observations.append(observation)
                truths.append(truth)
                scenes[sid] = scene
            return observations, truths, labels, scenes
        observations, truths, labels, scenes = measure("open-truth", open_labels)
        def known_draws():
            refs = []
            for sid in range(16):
                budget.check()
                intent = store.publish_json(f"known-draw/{sid}/intent.json", {"split": "train", "scene_id": sid, "seed": 2026090880})
                obs, truth = _draw_scene("train", sid, 2026090880)
                if encoded(obs) != encoded(observations[sid]) or encoded(truth) != encoded(truths[sid]):
                    raise ValueError("known TRAIN reconstruction changed; no substitute")
                o = store.publish_json(f"known-draw/{sid}/observation.json", obs)
                t = store.publish_json(f"known-draw/{sid}/sidecar.json", truth)
                for ref in (intent, o, t):
                    store.read(ref)
                refs.append(store.publish_json(f"known-draw/{sid}/receipt.json", {"intent": intent, "observation": o, "sidecar": t}))
            return refs
        known = measure("known-draw-io", known_draws)
        first, learned, eligible = measure("original-metrics", lambda: evaluate_batch(
            profile, observed, labels, store, "metrics/original", check=budget.check))
        derived = profile.json(observed["roundtrip"]["sources"])
        moved = {sid: labels_by_event(scenes[sid], profile.json(ref)["scene"], labels[sid])
            for sid, ref in zip(derived["scene_ids"], derived["sources"])}
        second, _, _ = measure("probe-metrics", lambda: evaluate_batch(profile, observed["roundtrip"], moved,
            store, "metrics/probe", check=budget.check))
        replay_first, _, _ = measure("original-replay", lambda: evaluate_batch(profile, observed, labels,
            store, "metrics/original", check=budget.check, replay=True))
        replay_second, _, _ = measure("probe-replay", lambda: evaluate_batch(profile, observed["roundtrip"], moved,
            store, "metrics/probe", check=budget.check, replay=True))
        if first != replay_first or second != replay_second:
            raise ValueError("metric profile replay changed its final receipts")
        def bootstrap():
            fixture = np.tile(learned[:, 1, :, :, :, NAMES.index("regret_tM")], (32, 1, 1, 1))
            result = primary(fixture, np.tile(eligible, 32), check=budget.check)
            ref = save_arrays(store, "bootstrap-fixture.npz", result["arrays"], replay=False)
            return save_json(store, "bootstrap-fixture.json", {"authority": "repeated metrics for timing only; NOT test evidence",
                "arrays": ref, "summary": result["summary"]}, replay=False)
        boot = measure("bootstrap512", bootstrap)
        inventory = measure("observable-inventory", lambda: file_inventory(profile, check=budget.check))
        inventory_ref = store.publish_json("observed-inventory.json", {"root": str(profile.root), "files": inventory})
        exclusions = store.publish_json("exclusions.json", admitted["exclusions"])
        size = owned_bytes([store.root])
        elapsed = time.monotonic()-started_at
        overhead = elapsed-sum(timings.values())
        forecast = projection(timings, admitted["forecast"], overhead=overhead, profile_bytes=size)
        result = store.publish_json("result.json", {"schema": "geometric-decision-closing-profile-v1",
            "binding": store.binding, "admission": admission_ref, "timings": timings, "forecast": forecast,
            "elapsed_to_forecast_seconds": elapsed, "overhead_seconds": overhead, "profile_bytes": size,
            "rss_peak_bytes": budget.rss(), "original": first, "probe": second,
            "original_replay": replay_first, "probe_replay": replay_second, "bootstrap_fixture": boot,
            "known_reconstructions": known, "observable_inventory": inventory_ref,
            "exclusions": exclusions, "new_observations": 0, "new_test_access": False})
        verify()
        output = control.publish_json("outputs/profile-closing.json", {"manifest": manifest,
            "root": str(store.root), "binding": store.reference(store.path("binding.json")), "result": result})
        return {"finish": budget.finish("COMPLETE", completion=output), "output": output, "forecast": forecast}
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
    if (Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))):
        raise ValueError("closing profile requires CPU-only one-thread project environment")
    torch.set_num_threads(1)
    view = ReadOnlyStore(BASES[0]/"control", binding_ref=CONTROL_BINDING)
    control = ArtifactStore(view.root, binding=view.binding)
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = BASES[0]/"profiles/closing-cpu-0"
        if root.exists():
            raise ValueError("closing profile already attempted; no silent repetition")
        code, protocol = sources(), reference(PROTOCOL)
        if protocol["sha256"] != PROTOCOL_SHA:
            raise ValueError("frozen protocol changed")
        binding = {"schema": "geometric-decision-closing-profile-binding-v1", "code": code, "protocol": protocol,
            "runtime": {"numpy": np.__version__, "torch": str(torch.__version__), "device": "cpu", "threads": 1},
            "new_test_authority": False}
        store = ArtifactStore(root, binding=binding)
        manifest = control.publish_json("manifests/profile-closing.json", {"operation": "profile-closing",
            "binding": binding, "root": str(root)})
        def verify():
            if sources() != code or reference(PROTOCOL) != protocol:
                raise ValueError("closing profile code/protocol changed")
        result = execute(control, store, manifest, started_at=LAUNCH_STARTED, verify=verify)
        print(json.dumps({"status": "CLOSING_PROFILE_COMPLETE_NOT_TEST_AUTHORITY", **result}), flush=True)


if __name__ == "__main__":
    main()
