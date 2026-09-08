"""Fail-closed learned-reader phase and dataset provenance checks.

Receipts record integration of independent audits, not authentication against
an operator forging both reports and receipts. No Torch, model or RNG setup.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np

from . import learned_partition_provenance as p
from . import learned_partition_resources as resources
from .learned_partition_data import ROLES, SHARD_SIZE, _bundle, scene_ids, validate_observation
from .learned_partition_metrics import SPLITS
from .partial_compatibility_cache import observation_fingerprint, sha_file
from .structured_source_artifacts import mark_failure, seal_bundle, write_json
from .learned_partition_validation import boundary, memoized

PROFILE_ROLES = {"geometry": "learned_geometry_profile", "cpu": "learned_training_cpu_profile",
                 "gpu": "learned_training_gpu_profile"}


@memoized
def common_binding():
    tests = sorted((p.ROOT/"experiments/atencion_armonica").glob("test_learned_partition_*.py"))
    return {"plan": p.PLAN, "protocol": p.PROTOCOL, "source_sha256": p.current_sources(),
            "test_sha256": {f.relative_to(p.ROOT).as_posix(): sha_file(f) for f in tests},
            "runtime": p.runtime_versions(), "checkpoints": p.historical.checkpoints()}


def verify_audit(ref, common, *, scope, target=None):
    record = p.read_reference(ref)
    if (set(record) != {"status", "scope", "common", "target", "reports"}
            or record["status"] != "PASS" or record["scope"] != scope
            or record["common"] != common or record["target"] != target
            or not isinstance(record["reports"], list) or not record["reports"]):
        raise ValueError("missing or mismatched independent integration audit")
    if len({r["path"] for r in record["reports"]}) != len(record["reports"]):
        raise ValueError("duplicate audit report")
    for report in record["reports"]:
        if not p.verify_reference(report).read_text().strip():
            raise ValueError("empty independent audit report")


def _bounded(value, bound):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < bound:
        raise ValueError("resource measurement or projection outside envelope")


@memoized
def _profiles(record, common):
    if set(record["profiles"]) != set(PROFILE_ROLES):
        raise ValueError("all three mechanical profiles required")
    reports = {}
    for name, role in PROFILE_ROLES.items():
        root, manifest = _bundle(record["profiles"][name], role, common)
        if manifest["binding"].get("implementation_audit") != record["implementation_audit"]:
            raise ValueError("profile uses another implementation audit")
        r = json.loads((root/"report.json").read_bytes())
        if r.get("status") != "MEASURED" or r.get("namespace") != "MECHANICAL_NOT_PROSPECTIVE":
            raise ValueError("profile does not contain completed measurements")
        observations = r["observations"]
        if not isinstance(observations, list) or (name != "cpu" and not observations):
            raise ValueError("profile omitted its observable fixture roster")
        for obs in observations:
            if set(obs) != {"scene_id", "split_seed", "log_f"}:
                raise ValueError("profile observation schema differs")
            q = np.asarray(obs["log_f"])
            if (q.ndim != 1 or not 8 <= len(q) <= 32 or not np.isfinite(q).all()
                    or not np.array_equal(q, q.astype(np.float32).astype(np.float64))):
                raise ValueError("profile observation is not exact finite q32")
        _bounded(r["seconds"], 120.000001)
        _bounded(r["peak_rss_bytes"], (1 if name == "geometry" else 4)*1024**3)
        if name == "geometry":
            if manifest["binding"] != {"common": common, "implementation_audit": record["implementation_audit"]}:
                raise ValueError("geometry profile binding differs")
            resources.validate_geometry(r)
            _bounded(r["projected_shard_seconds"], 2400.000001)
        else:
            if set(manifest["binding"]) != {"common", "implementation_audit", "geometry", "gpu_grant"}:
                raise ValueError("training profile binding schema differs")
            if manifest["binding"].get("geometry") != record["profiles"]["geometry"]:
                raise ValueError("training measurements use another geometry profile")
            resources.validate_training(r, reports["geometry"], gpu=name == "gpu")
            if any(r["runtime"][key] != common["runtime"][key] for key in ("torch", "numpy")):
                raise ValueError("profile runtime differs from the audited implementation")
            if (r["batch_size"], r["candidate_padding"], r["group_padding"]) != (32, 64, 94):
                raise ValueError("training profile shape or recipe differs")
            _bounded(r["projected_cell_seconds"], math.inf)
            if name == "gpu":
                grant = p.read_reference(manifest["binding"]["gpu_grant"])
                if (grant.get("status") != "AUTHORIZED" or grant.get("project") != "Phideus"
                        or grant.get("device") != r["device"] or not isinstance(grant.get("user_directive"), str)
                        or not grant["user_directive"].strip()):
                    raise PermissionError("profile GPU grant is missing or inconsistent")
                if r["device"] != "NVIDIA GeForce RTX 3090":
                    raise ValueError("profile uses an unauthorized GPU")
                _bounded(r["peak_reserved_bytes"], 2*1024**3)
                _bounded(r["projected_forward_shard_seconds"], 1200.000001)
            elif r["device"] != "cpu" or manifest["binding"]["gpu_grant"] is not None:
                raise ValueError("CPU training profile device differs")
        reports[name] = r
    device = "cpu" if reports["cpu"]["projected_cell_seconds"] <= reports["gpu"]["projected_cell_seconds"] else "cuda:0"
    if record["training_device"] != device:
        raise ValueError("training device was not selected by the paired resource projections")
    _bounded(reports["cpu" if device == "cpu" else "gpu"]["projected_cell_seconds"], 1200.000001)
    selected = reports["cpu" if device == "cpu" else "gpu"]
    projected_stages = resources.stage_projection(selected["heads"], reports["geometry"],
        forward_shard_seconds=reports["gpu"]["projected_forward_shard_seconds"])
    if record["projected_stages"] != projected_stages:
        raise ValueError("stage validation/compute projections differ")
    for name, value in projected_stages.items():
        _bounded(value["worker_total_seconds"], 2400.000001 if name == "score" else 1200.000001)
    _bounded(36*selected["projected_cell_seconds"], 43200.000001)
    _bounded(record["projected_disk_bytes"], math.inf)
    if record["projected_disk_bytes"] <= 0 or record["projected_disk_bytes"] != reports["geometry"]["projected_disk_bytes"]:
        raise ValueError("missing or inconsistent preserved-artifact disk projection")
    return reports


@memoized
def _verify_data_authorization(record, common):
    if (set(record) != {"status", "common", "implementation_audit", "profiles", "training_device",
                       "prior_corpus", "projected_disk_bytes", "projected_stages"}
            or record["status"] != "TRAIN_CALIBRATION_READY" or record["common"] != common):
        raise PermissionError("train/calibration require the complete audited implementation and profiles")
    verify_audit(record["implementation_audit"], common, scope="FULL_IMPLEMENTATION")
    reports = _profiles(record, common)
    if record["prior_corpus"] != p.prior_corpus():
        raise ValueError("prior corpus was changed or omitted")
    return reports


def create_data_authorization(output, *, implementation_audit, profiles):
    common = common_binding()
    # Read measured profiles only; the shared verifier checks every binding.
    raw = {}
    for name, role in PROFILE_ROLES.items():
        root, _ = _bundle(profiles[name], role, common)
        raw[name] = json.loads((root/"report.json").read_bytes())
    device = "cpu" if raw["cpu"]["projected_cell_seconds"] <= raw["gpu"]["projected_cell_seconds"] else "cuda:0"
    record = {"status": "TRAIN_CALIBRATION_READY", "common": common, "implementation_audit": implementation_audit,
              "profiles": profiles, "training_device": device, "prior_corpus": p.prior_corpus(),
              "projected_disk_bytes": raw["geometry"]["projected_disk_bytes"],
              "projected_stages": resources.stage_projection(raw["cpu" if device == "cpu" else "gpu"]["heads"], raw["geometry"],
                  forward_shard_seconds=raw["gpu"]["projected_forward_shard_seconds"])}
    _verify_data_authorization(record, common)
    if shutil.disk_usage(p.ROOT).free <= 2*record["projected_disk_bytes"]:
        raise RuntimeError("insufficient disk margin for the preserved campaign")
    if common_binding() != common:
        raise ValueError("implementation changed during authorization")
    write_json(output, record)
    return p.reference(output)


@boundary
@memoized
def verify_authorization(ref, split):
    if split not in SPLITS:
        raise ValueError("unknown campaign role")
    common = common_binding()
    record = p.read_reference(ref)
    if split in ("train", "calibration"):
        _verify_data_authorization(record, common)
        return record
    if (set(record) != {"status", "common", "freeze", "freeze_audit"}
            or record["status"] != "TEST_READY" or record["common"] != common):
        raise PermissionError("test requires its independently audited selection freeze")
    verify_audit(record["freeze_audit"], common, scope="SELECTION_FREEZE", target=record["freeze"])
    from .learned_partition_selection import verify_selection_chain
    verify_selection_chain(record["freeze"], common)
    return record


@memoized
def _shard_observations(ref, split, shard, common, *, authorization, previous, earlier_shards):
    root, m = _bundle(ref, "learned_observation_shard", common)
    ids = scene_ids(split, shard)
    expected = {"common": common, "authorization": authorization, "previous": previous,
                "earlier_shards": earlier_shards, "split": split, "split_seed": SPLITS[split][1],
                "shard": shard, "count": SHARD_SIZE, "split_count": SPLITS[split][0], "scene_ids": ids}
    if m["binding"] != expected:
        raise ValueError("shard has a different authorization, role or complete prefix")
    files = {"observations.jsonl", "sidecars.jsonl", "rows.json", "deduplication.json"}
    files.update(f"features/{i:05d}.npz" for i in ids)
    if set(m["artifacts_sha256"]) != files:
        raise ValueError("shard scientific inventory differs")
    observations = [json.loads(line) for line in (root/"observations.jsonl").read_bytes().splitlines()]
    if len(observations) != SHARD_SIZE:
        raise ValueError("incomplete shard observation roster")
    fingerprints = set()
    for i, obs in zip(ids, observations):
        validate_observation(obs, i, split)
        fp = observation_fingerprint(obs)
        if fp in fingerprints:
            raise ValueError("duplicate within shard")
        fingerprints.add(fp)
    return fingerprints


@memoized
def split_fingerprints(ref, split, common, *, authorization, previous):
    root, m = _bundle(ref, "learned_observation_split", common)
    if m["binding"] != {"common": common, "authorization": authorization, "previous": previous,
                        "split": split, "split_seed": SPLITS[split][1], "count": SPLITS[split][0]}:
        raise ValueError("split aggregation identity differs")
    if set(m["artifacts_sha256"]) != {"shards.json", "fingerprints.json"}:
        raise ValueError("split aggregation inventory differs")
    shards = json.loads((root/"shards.json").read_bytes())
    if not isinstance(shards, list) or len(shards) != SPLITS[split][0]//SHARD_SIZE:
        raise ValueError("split aggregation omitted or added shards")
    seen = set()
    for i, shard in enumerate(shards):
        values = _shard_observations(shard, split, i, common, authorization=authorization,
                                    previous=previous, earlier_shards=shards[:i])
        if seen & values:
            raise ValueError("observation duplicated across shards")
        seen.update(values)
    if sorted(seen) != json.loads((root/"fingerprints.json").read_bytes()) or len(seen) != SPLITS[split][0]:
        raise ValueError("split fingerprint aggregation differs")
    return seen


@boundary
def verify_data_stage(authorization, split, shard, previous, earlier_shards):
    scene_ids(split, shard)
    auth = verify_authorization(authorization, split)
    common = auth["common"]
    if not isinstance(earlier_shards, list) or len(earlier_shards) != shard:
        raise ValueError("every earlier shard reference is required")
    roles = ROLES[:ROLES.index(split)]
    if set(previous) != set(roles):
        raise ValueError("every earlier complete role is required")
    if split in ("train", "calibration"):
        base, base_ref = auth, authorization
    else:
        frozen = p.read_reference(auth["freeze"])
        base_ref = frozen["data_authorization"]
        base = p.read_reference(base_ref)
        _verify_data_authorization(base, common)
        if previous["train"] != frozen["train_data"] or previous["calibration"] != frozen["calibration_data"]:
            raise ValueError("test preceding data differs from the frozen training/selection corpus")
    seen = set(base["prior_corpus"]["fingerprints"])
    for name, report in _profiles(base, common).items():
        observations = report["observations"]
        if not isinstance(observations, list) or (name != "cpu" and not observations):
            raise ValueError("profile omitted its observable fixture roster")
        for obs in observations:
            if set(obs) != {"scene_id", "split_seed", "log_f"}:
                raise ValueError("profile observation schema differs")
            seen.add(observation_fingerprint(obs))
    for role in roles:
        values = split_fingerprints(previous[role], role, common,
            authorization=base_ref if role in ("train", "calibration") else authorization,
            previous={r: previous[r] for r in ROLES[:ROLES.index(role)]})
        if seen & values:
            raise ValueError("earlier role duplicates prior observations")
        seen.update(values)
    for i, ref in enumerate(earlier_shards):
        values = _shard_observations(ref, split, i, common, authorization=authorization,
                                    previous=previous, earlier_shards=earlier_shards[:i])
        if seen & values:
            raise ValueError("earlier shard duplicates prior observations")
        seen.update(values)
    return auth, seen


def aggregate_split(output, split, *, authorization, previous, shards):
    expected = SPLITS[split][0]//SHARD_SIZE
    if not isinstance(shards, list) or len(shards) != expected:
        raise ValueError("all fixed shards are required before aggregation")
    auth, prior = verify_data_stage(authorization, split, 0, previous, [])
    seen = set()
    for i, ref in enumerate(shards):
        values = _shard_observations(ref, split, i, auth["common"], authorization=authorization,
                                    previous=previous, earlier_shards=shards[:i])
        if (prior | seen) & values:
            raise ValueError("split duplicates an earlier observation")
        seen.update(values)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output/"shards.json", shards)
        write_json(output/"fingerprints.json", sorted(seen))
        if verify_data_stage(authorization, split, 0, previous, []) != (auth, prior):
            raise ValueError("aggregation authorization changed")
        seal_bundle(output, role="learned_observation_split", binding={"common": auth["common"],
            "authorization": authorization, "previous": previous, "split": split,
            "split_seed": SPLITS[split][1], "count": SPLITS[split][0]}, resources={"operation": "aggregate_no_draws"})
        return p.reference(output/"manifest.json")
    except BaseException as exc:
        mark_failure(output, exc)
        raise
