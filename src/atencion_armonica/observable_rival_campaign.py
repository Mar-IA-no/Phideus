"""Small, immutable per-scene campaign over an already released observable roster."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time
import uuid

import numpy as np

from . import observable_source_rivals as math

SPLITS = ("iid", "ood_beta", "ood_polyphony", "deformed_family")
RELEASE = Path("data/atencion_armonica/learned_partition_reader_v1/evaluation_release_recovery_v2/tests_01.json")
RELEASE_SHA = "7f4f368cf22ea60a2d02faa92099933cdbec2dcb13a69dc1df2a603f45c66153"
PROTOCOL = Path("experiments/atencion_armonica/PROTOCOL_OBSERVABLE_SOURCE_RIVALS.md")
PROTOCOL_SHA = "905d67e42de3b39a5898ee7127073ceeaf87ccef7046020d11dbf18b5a051f3b"
CHECKPOINTS = (2026090721, 2026090722, 2026090723)
SOURCES = (
    "src/atencion_armonica/observable_source_rivals.py",
    "src/atencion_armonica/observable_rival_campaign.py",
    "src/atencion_armonica/observable_rival_evaluation.py",
    "experiments/atencion_armonica/run_observable_source_rivals.py",
    "experiments/atencion_armonica/test_observable_source_rivals.py",
    "experiments/atencion_armonica/test_observable_rival_campaign.py",
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")+b"\n"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference(path):
    return {"path": str(path), "sha256": sha(path)}


def checked_json(ref):
    if sha(ref["path"]) != ref["sha256"]:
        raise ValueError(f"hash mismatch: {ref['path']}")
    return json.loads(Path(ref["path"]).read_bytes())


def write_once(path, value):
    """Atomic publish without replacement; an existing identical payload is reusable."""
    path = Path(path)
    payload = encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
        return reference(path)
    temporary = path.parent / ("."+path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("xb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return reference(path)


def selected_ids():
    return {split: sorted(np.random.default_rng(np.random.SeedSequence([2026090901, i]))
                          .choice(512, 24, replace=False).tolist()) for i, split in enumerate(SPLITS)}


def validate_scene_roster(scenes, ids):
    expected = [(split, sid) for split, values in selected_ids().items() for sid in values]
    actual = [(row["split"], row["scene_id"]) for row in scenes]
    if (ids != selected_ids() or len(scenes) != 96 or len(set(actual)) != 96
            or any(type(sid) is not int for _, sid in actual) or actual != expected):
        raise ValueError("campaign requires the exact ordered 96-scene roster without replacement")


def runtime_fingerprint():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "packages": {p: importlib.metadata.version(p) for p in ("scipy", "scikit-learn", "torch")},
            "system": platform.system(), "machine": platform.machine(), "kernel": platform.release(),
            "arithmetic": "float64", "tf32": False}


def source_binding():
    checked_json({"path": str(RELEASE), "sha256": RELEASE_SHA})
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("protocol differs from independently reviewed version")
    return {"sources": {p: sha(p) for p in SOURCES}, "protocol": reference(PROTOCOL),
            "release": reference(RELEASE), "grid": asdict(math.Grid()),
            "runtime": runtime_fingerprint(),
            "scene_ids": selected_ids()}


def read_bundle_member(manifest_ref, name):
    manifest = checked_json(manifest_ref)
    if manifest.get("status") != "COMPLETE":
        raise ValueError("upstream bundle incomplete")
    path = Path(manifest_ref["path"]).parent/name
    ref = {"path": str(path), "sha256": manifest["artifacts_sha256"][name]}
    if sha(path) != ref["sha256"]:
        raise ValueError(f"upstream member changed: {path}")
    return path, ref, manifest


def load_observable_inventory():
    """Parse observations/pools only; sidecars and metrics are not opened here."""
    release = checked_json({"path": str(RELEASE), "sha256": RELEASE_SHA})
    output = []
    for split, ids in selected_ids().items():
        entry = release["splits"][split]
        path, obs_ref, manifest = read_bundle_member(entry["data"], "observations.jsonl")
        observations = [json.loads(line) for line in path.read_bytes().splitlines()]
        if len(observations) != 512 or manifest["binding"]["split"] != split:
            raise ValueError("upstream observation roster differs")
        for scene_id in ids:
            obs = observations[scene_id]
            if (set(obs) != {"scene_id", "split_seed", "log_f"} or obs["scene_id"] != scene_id
                    or obs["split_seed"] != manifest["binding"]["split_seed"]):
                raise ValueError("observation identity differs")
            q = np.asarray(obs["log_f"], dtype=np.float64)
            if (q.ndim != 1 or not 8 <= len(q) <= 32 or not np.isfinite(q).all()
                    or not np.array_equal(q, q.astype(np.float32).astype(np.float64))):
                raise ValueError("invalid q32 observation")
            order = np.argsort(q, kind="stable")
            canonical = q[order]
            pools, refs = {}, []
            for seed in CHECKPOINTS:
                name = f"seed_{seed}/{scene_id:05d}_pool.json"
                pp, pref, pm = read_bundle_member(entry["scored"], name)
                if pm["binding"]["data"] != entry["data"] or pm["binding"]["split"] != split:
                    raise ValueError("pool belongs to another observation shard")
                pool = json.loads(pp.read_bytes())["pool"]
                if (pool["canonical_to_observed"] != order.tolist()
                        or pool["canonical_q32"] != canonical.tolist()
                        or pool["q_tie_count"] != int(np.sum(np.diff(canonical) == 0))):
                    raise ValueError("pool canonicalization differs from observation")
                pools[str(seed)] = pool["partitions"]
                refs.append(pref)
            inventory = math.candidate_inventory(pools, len(q))
            output.append({"split": split, "scene_id": scene_id, "q32": canonical.tolist(),
                           "canonical_to_observed": order.tolist(),
                           "q_tie_count": int(np.sum(np.diff(canonical) == 0)),
                           "observation_reference": obs_ref, "pool_references": refs, **inventory})
    return output


def profile(device):
    """Mechanical deterministic groups, no test observations or source labels."""
    fitter = math.GroupFitter(device=device)
    t0 = time.monotonic()
    rows = []
    for branch in math.BRANCHES:
        for m in (4, 6, 8):
            beta, gamma = fitter.grid.values(branch)
            y = math.template(range(1, m+1), beta[128], gamma[len(gamma)//2])
            y += np.linspace(-2, 2, m)*math.CENTS_TO_LOG
            start = time.monotonic()
            factors = fitter.fit([y]*8, branch)
            duration = time.monotonic()-start
            rows.append({"branch": branch, "size": m, "group_count": 8,
                         "seconds": duration, "factor": factors[0]})
            if time.monotonic()-t0 > 120:
                raise TimeoutError("mechanical profile exceeded 120 seconds")
    return {"device": device, "seconds": time.monotonic()-t0, "rows": rows,
            "core_sha256": sha(SOURCES[0]), "grid": asdict(fitter.grid),
            "runtime": runtime_fingerprint(),
            "gpu": ({"name": fitter.torch.cuda.get_device_name(),
                     "capability": list(fitter.torch.cuda.get_device_capability()),
                     "cuda_build": fitter.torch.version.cuda} if fitter.torch is not None else None),
            "rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_peak_bytes": (fitter.torch.cuda.max_memory_allocated()
                                if fitter.torch is not None else 0)}


def compare_profiles(first_ref, second_ref):
    a, b = checked_json(first_ref), checked_json(second_ref)
    if (a["core_sha256"] != b["core_sha256"] or a["grid"] != b["grid"]
            or a.get("runtime") != b.get("runtime") or a.get("runtime") != runtime_fingerprint()
            or a["core_sha256"] != sha(SOURCES[0]) or {a["device"], b["device"]} != {"cpu", "cuda"}):
        raise ValueError("profiles do not contrast identical current grids on CPU/CUDA")
    max_j, max_template = 0., 0.
    for left, right in zip(a["rows"], b["rows"], strict=True):
        if (left["branch"], left["size"]) != (right["branch"], right["size"]):
            raise ValueError("mechanical profile roster differs")
        for resolution in ("fine", "coarse"):
            for x, y in zip(left["factor"][resolution], right["factor"][resolution], strict=True):
                if x["indices"] != y["indices"]:
                    raise ValueError("mechanical assignment roster differs")
                max_j = max(max_j, abs(x["sse"]-y["sse"])/(2*math.SIGMA**2))
                max_template = max(max_template, float(np.max(np.abs(
                    np.asarray(x["template"])-np.asarray(y["template"])))))
    if max_j > math.TOL_J or max_template > 1e-10:
        raise ArithmeticError(f"backend differential exceeds tolerances: {max_j}, {max_template}")
    return {"status": "PASS", "profiles": [first_ref, second_ref],
            "max_J_difference": max_j, "max_template_difference": max_template,
            "seconds": {p["device"]: p["seconds"] for p in (a, b)}}


def verify_binding(manifest):
    if manifest["binding"] != source_binding():
        raise ValueError("campaign implementation, protocol, runtime or roster changed")
    validate_scene_roster(manifest["scenes"], manifest["binding"]["scene_ids"])


def initialize(root, device, profile_ref, review_refs):
    """Called only after human-readable review and mechanical resource projection."""
    root = Path(root)
    binding = source_binding()
    prof = checked_json(profile_ref)
    if prof.get("runtime") != binding["runtime"]:
        raise ValueError("profile does not bind the current runtime/backend arithmetic")
    if prof["core_sha256"] != binding["sources"][SOURCES[0]] or prof["grid"] != binding["grid"]:
        raise ValueError("profile does not bind the current mathematical implementation")
    if prof["device"] != device or prof["seconds"] > 120 or prof["rss_kib"] > 4*1024**2:
        raise ValueError("backend profile or CPU resource budget differs")
    if prof["cuda_peak_bytes"] > 6*1024**3:
        raise ValueError("GPU resource budget exceeded")
    # Reviews are immutable evidence references, not verdicts inferred by a parser.
    for ref in review_refs:
        if sha(ref["path"]) != ref["sha256"]:
            raise ValueError("review changed")
    scenes = load_observable_inventory()
    validate_scene_roster(scenes, binding["scene_ids"])
    group_counts = {}
    for branch, (_, _, ks) in math.BRANCHES.items():
        for m in range(4, 9):
            group_counts[f"{branch}/{m}"] = sum(len({tuple(g)
                for row in scene["candidates"] if row["status"] == "SUPPORTED"
                and len(row["partition"]) in ks for g in row["partition"] if len(g) == m})
                for scene in scenes)
    # Unprofiled sizes 5/7 use the maximum observed per-group cost, conservatively.
    max_cost = max(row["seconds"]/row["group_count"] for row in prof["rows"])
    projection = sum(count*max_cost for count in group_counts.values())
    # Factor four covers joint composition, reference fits, replay and non-sweep overhead.
    projection *= 4
    if projection > 7200:
        raise TimeoutError(f"projected campaign exceeds two-hour budget: {projection:.1f}s")
    return write_once(root/"campaign.json", {"schema": "observable-rivals-v1", "binding": binding,
        "device": device, "profile": profile_ref, "reviews": review_refs, "scenes": scenes,
        "group_counts": group_counts, "projected_seconds": projection})


def elapsed_budget(root):
    total = 0.
    for p in Path(root).glob("attempts/*.json"):
        record = json.loads(p.read_bytes())
        end = p.with_suffix(".end.json")
        if p.name.endswith(".end.json"):
            continue
        if not end.exists():
            raise RuntimeError(f"unclosed attempt requires resource reconciliation: {p}")
        total += json.loads(end.read_bytes())["seconds"]
    return total


def run(root):
    root = Path(root)
    manifest = json.loads((root/"campaign.json").read_bytes())
    verify_binding(manifest)
    spent = elapsed_budget(root)
    if spent >= 7200:
        raise TimeoutError("cumulative campaign time exhausted")
    fitter = math.GroupFitter(device=manifest["device"])
    attempt = root/"attempts"/(uuid.uuid4().hex+".json")
    write_once(attempt, {"pid": os.getpid(), "started_unix": time.time(), "kind": "observable"})
    start = time.monotonic()
    refs = []
    try:
        for scene in manifest["scenes"]:
            if spent+time.monotonic()-start >= 7200:
                raise TimeoutError("cumulative campaign time exhausted")
            path = root/"observable"/scene["split"]/f"{scene['scene_id']:05d}.json"
            if path.exists():
                record = json.loads(path.read_bytes())
                if record["input_sha256"] != hashlib.sha256(encoded(scene)).hexdigest():
                    raise ValueError("existing scene has a different observable input")
            else:
                t0 = time.monotonic()
                partitions = [row["partition"] for row in scene["candidates"]]
                result = math.fit_candidates(scene["q32"], partitions, fitter)
                record = {"input_sha256": hashlib.sha256(encoded(scene)).hexdigest(),
                    "split": scene["split"], "scene_id": scene["scene_id"],
                    "status": "COMPLETE" if any(f["status"] == "FITTED" for f in result["fits"])
                              else "NO_OBSERVABLE_CANDIDATE", **result,
                    "resources": {"seconds": time.monotonic()-t0,
                        "rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        "cuda_peak_bytes": (fitter.torch.cuda.max_memory_allocated()
                                            if fitter.torch is not None else 0)}}
                write_once(path, record)
                print(json.dumps({"split": scene["split"], "scene_id": scene["scene_id"],
                      "status": record["status"], **record["resources"]}), flush=True)
            refs.append(reference(path))
            if (record["resources"]["rss_kib"] > 4*1024**2
                    or record["resources"]["cuda_peak_bytes"] > 6*1024**3):
                raise MemoryError("campaign resource ceiling exceeded")
            if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > 5*1024**3:
                raise OSError("campaign artifact budget exceeded")
        verify_binding(manifest)
        return write_once(root/"observable_seal.json", {"schema": "observable-rivals-seal-v1",
            "campaign": reference(root/"campaign.json"), "scenes": refs, "count": len(refs)})
    finally:
        write_once(attempt.with_suffix(".end.json"), {"seconds": time.monotonic()-start})


def verify_seal(root):
    root = Path(root)
    seal = json.loads((root/"observable_seal.json").read_bytes())
    manifest = checked_json(seal["campaign"])
    verify_binding(manifest)
    validate_scene_roster(manifest["scenes"], manifest["binding"]["scene_ids"])
    expected = [root/"observable"/scene["split"]/f"{scene['scene_id']:05d}.json"
                for scene in manifest["scenes"]]
    if (seal["count"] != 96 or len(seal["scenes"]) != 96 or len(manifest["scenes"]) != 96
            or [r["path"] for r in seal["scenes"]] != [str(p) for p in expected]):
        raise ValueError("observable seal lacks the exact complete roster")
    for scene, ref in zip(manifest["scenes"], seal["scenes"]):
        record = checked_json(ref)
        if (record["input_sha256"] != hashlib.sha256(encoded(scene)).hexdigest()
                or record["scene_id"] != scene["scene_id"] or record["split"] != scene["split"]):
            raise ValueError("sealed record identity differs")
    return manifest, seal["scenes"]


def replay(root):
    root = Path(root)
    manifest, refs = verify_seal(root)
    spent = elapsed_budget(root)
    if spent >= 7200:
        raise TimeoutError("cumulative campaign time exhausted")
    attempt = root/"attempts"/(uuid.uuid4().hex+".json")
    write_once(attempt, {"kind": "observable_replay", "started_unix": time.time()})
    t0 = time.monotonic()
    try:
        for scene, ref in zip(manifest["scenes"], refs):
            if spent+time.monotonic()-t0 >= 7200:
                raise TimeoutError("cumulative campaign time exhausted")
            record = checked_json(ref)
            fits = math.replay_fits(scene["q32"], record["group_factors"],
                                   [c["partition"] for c in scene["candidates"]])
            if encoded(fits) != encoded(record["fits"]):
                raise ValueError("observable witness replay differs")
        return {"status": "PASS", "count": len(refs), "seconds": time.monotonic()-t0,
                "seal": reference(root/"observable_seal.json")}
    finally:
        write_once(attempt.with_suffix(".end.json"), {"seconds": time.monotonic()-t0})
