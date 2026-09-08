"""Frozen-model forward and CPU structured readout; no training entry point."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from . import structured_source_gate as gate
from .partial_compatibility_cache import encoded, sha_file
from .source_coherence import GroupFitCache, SourceFitter
from .structured_source_artifacts import mark_failure, seal_bundle, write_json, write_npz
from .structured_source_data import StructuredObservations, cpu_resources, load_supervision, prior_corpus
from .structured_source_metrics import (SEEDS, SPLIT_SEEDS, bootstrap_indices, calibration_grid,
                                        evaluate_scene, read_selected, select_gammas, summarize_test)
from .structured_source_reader import score_scene


def stage_inputs(authorization, split, data_ref, logits_ref=None):
    auth = gate.verify_authorization(authorization, split)
    cache = StructuredObservations(data_ref, split, auth["common"])
    binding = cache.manifest["binding"]
    if binding.get("authorization") != authorization:
        raise ValueError("data was not produced under this exact stage authorization")
    if prior_corpus(auth, split, binding["previous"]) & cache.fingerprints:
        raise ValueError("fresh observations duplicate earlier corpus")
    raw = None
    if logits_ref is not None:
        root, manifest = gate.bundle_reference(logits_ref, f"{split}_logits", auth["common"])
        b = manifest["binding"]
        if (b.get("data") != data_ref or b.get("authorization") != authorization or b.get("split") != split
                or b.get("split_seed") != SPLIT_SEEDS[split] or b.get("count") != 256):
            raise ValueError("forward has a different data/authorization identity")
        raw = gate.ordered_forward(root, manifest, cache.observations, auth["common"])
    return auth, cache, raw


def gpu_runtime():
    """Called only after a verified GPU grant and availability check."""
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("requires the authorized RTX 3090")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(0)
    return {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(0)}


def checkpoint_forward(checkpoint, records, runtime):
    """Observables only. The pinned checkpoint is the authority, not filename."""
    import torch
    from .partial_compatibility_inference import collect_logits
    from .partial_compatibility_learning import ARMS
    from .pairformer import MODEL_CONFIGS, build_model
    path = gate.verify_reference(checkpoint["checkpoint"])
    state = torch.load(path, map_location="cpu", weights_only=False)
    binding = state["binding"]
    if (binding["runtime"] != runtime or binding["arm"] != "pairs_descriptors"
            or binding["seed"] != checkpoint["seed"] or binding["model_config"] != MODEL_CONFIGS[ARMS["pairs_descriptors"][0]]
            or state["steps"] != 3200 or state["next_epoch"] != 50 or state["next_batch"] != 0):
        raise ValueError("checkpoint is not the declared unchanged last-epoch state/runtime")
    model = build_model(ARMS["pairs_descriptors"][0]).to("cuda:0")
    model.load_state_dict(state["model"])
    try:
        result = collect_logits(model, records, device="cuda:0")
        gate.verify_reference(checkpoint["checkpoint"])
        if torch.cuda.max_memory_reserved(0) >= 2*1024**3:
            raise RuntimeError("forward exceeds the predeclared 2GiB envelope")
        return result
    finally:
        del model, state
        torch.cuda.empty_cache()


def forward_split(output, split, *, authorization, data, gpu_grant):
    started = time.monotonic()
    auth, cache, _ = stage_inputs(authorization, split, data)
    from .structured_source_profile import gpu_lease
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        with gpu_lease(gpu_grant) as availability:
            runtime = gpu_runtime()
            import torch
            from .partial_compatibility_inference import save_logits
            rows = []
            for checkpoint in auth["common"]["checkpoints"]:
                if time.monotonic()-started >= 600:
                    raise TimeoutError("forward time envelope exhausted")
                matrices = checkpoint_forward(checkpoint, cache.records, runtime)
                seed = checkpoint["seed"]
                name = f"seed_{seed}.npz"
                save_logits(output/name, matrices, cache.observations)
                rows.append({"seed": seed, "checkpoint": checkpoint["checkpoint"], "path": name, "count": 256})
                del matrices
            peak = torch.cuda.max_memory_reserved(0)
            torch.cuda.synchronize()
        write_json(output/"forward.json", {"rows": rows, "runtime": runtime})
        after, _, _ = stage_inputs(authorization, split, data)
        if after != auth:
            raise ValueError("forward inputs changed during execution")
        resources = {"seconds": time.monotonic()-started, "peak_reserved_bytes": peak,
                     "runtime": runtime, "availability": availability}
        if resources["seconds"] > 600 or peak >= 2*1024**3:
            raise RuntimeError("forward resource envelope exceeded")
        binding = {"common": auth["common"], "authorization": authorization, "data": data,
                   "gpu_grant": gpu_grant, "split": split, "split_seed": SPLIT_SEEDS[split], "count": 256}
        seal_bundle(output, role=f"{split}_logits", binding=binding, resources=resources)
        ref = gate.reference(output/"manifest.json")
        root, manifest = gate.bundle_reference(ref, f"{split}_logits", auth["common"])
        gate.ordered_forward(root, manifest, cache.observations, auth["common"])
        return ref
    except BaseException as exc:
        mark_failure(output, exc)
        raise


def scene_payload(observation, record, matrices, checkpoints, source_ids, *, fitter):
    """Fit/cache observable groups first; labels enter only evaluation below."""
    q = np.asarray(observation["log_f"], np.float32)
    cache = GroupFitCache(fitter)
    scored = {}
    for checkpoint in checkpoints:
        seed = checkpoint["seed"]
        scored[seed] = score_scene(q, matrices[seed], record["pair_support"], record["triples"],
                                   record["residual_cents"], split_seed=observation["split_seed"],
                                   scene_id=observation["scene_id"], fit_cache=cache)
    # All candidates/costs for all checkpoint pools already exist at this point.
    return {c["seed"]: {"scored": scored[c["seed"]],
                        "evaluation": evaluate_scene(scored[c["seed"]], source_ids,
                                                     matrices[c["seed"]], c["threshold"])} for c in checkpoints}


def analyze_split(output, split, *, authorization, data, logits, replay_of=None):
    started = time.monotonic()
    auth, cache, raw = stage_inputs(authorization, split, data, logits)
    truths = load_supervision(cache)
    frozen = None if split == "calibration" else gate.read_reference(auth["freeze"])
    gammas = None if frozen is None else {k: v["gamma"] for k, v in frozen["selection"]["factors"].items()}
    original = None
    if replay_of is not None:
        _, original = gate.bundle_reference(replay_of, f"{split}_analysis", auth["common"])
        expected = {"authorization": authorization, "data": data, "logits": logits, "split": split,
                    "split_seed": SPLIT_SEEDS[split], "count": 256, "replay_of": None}
        if any(original["binding"].get(k) != v for k, v in expected.items()):
            raise ValueError("replay must use the primary analysis's exact inputs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output/"scenes").mkdir()
        fitter, rows = SourceFitter(), []
        for i, (obs, features, truth) in enumerate(zip(cache.observations, cache.records, truths)):
            payloads = scene_payload(obs, features, {s: raw[s][i] for s in SEEDS},
                                     auth["common"]["checkpoints"], truth["source_ids"], fitter=fitter)
            for seed in SEEDS:
                payload = payloads[seed]
                identity = {"scene_id": i, "seed": seed, "split_role": split, "split_seed": SPLIT_SEEDS[split]}
                write_json(output/f"scenes/{i:05d}__seed_{seed}.json", {**identity, **payload})
                row = (calibration_grid(**payload, observation=obs, seed=seed) if split == "calibration"
                       else {**identity, **read_selected(**payload, gammas=gammas)})
                rows.append(row)
            cpu_resources(started)
        if split == "calibration":
            selection = select_gammas(rows, split=split)
            write_json(output/"selection.json", selection)
            name = "gamma_grid.jsonl"
        else:
            indices = bootstrap_indices()
            write_npz(output/"bootstrap_indices.npz", indices=indices)
            write_json(output/"summary.json", summarize_test(rows, split=split, indices=indices))
            name = "readouts.jsonl"
        with (output/name).open("xb") as handle:
            for row in rows:
                handle.write(encoded(row))
        after, _, _ = stage_inputs(authorization, split, data, logits)
        if after != auth:
            raise ValueError("analysis inputs changed during execution")
        if original is not None:
            actual = {p.relative_to(output).as_posix(): sha_file(p) for p in output.rglob("*") if p.is_file()}
            if actual != original["artifacts_sha256"]:
                raise ValueError("replay differs from primary scientific artifacts")
        binding = {"common": auth["common"], "authorization": authorization, "data": data, "logits": logits,
                   "split": split, "split_seed": SPLIT_SEEDS[split], "count": 256, "replay_of": replay_of}
        resources = {**cpu_resources(started), "replay_status": "EXACT_SCIENTIFIC_BYTES" if original else "PRIMARY"}
        seal_bundle(output, role=f"{split}_analysis", binding=binding, resources=resources)
        return gate.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise
