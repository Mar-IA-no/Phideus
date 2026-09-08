"""One declared prepared-cohort import; no generic source compatibility.

Original payloads keep their identity. New manifests attest import/revalidation,
not another draw, forward, scoring pass or normalizer fit. Ordinary consumers
continue to require exact current bindings.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from . import learned_partition_provenance as p
from .partial_compatibility_cache import encoded, sha_file
from .structured_source_artifacts import (safe_member, verify_bundle, seal_bundle,
    write_json, mark_failure)
from .learned_partition_validation import memoized, boundary

PREPARED_FIELDS = {"authorization", "train", "calibration", "normalizers",
                   "normalized_train", "normalized_calibration"}
ROLES = {"learned_observation_shard": 9, "learned_logits_shard": 9,
    "learned_scored_shard": 9, "learned_supervised_targets_shard": 9,
    "learned_normalized_shard": 9, "learned_observation_split": 2,
    "learned_training_corpus": 2, "learned_train_normalizers": 1}
INDEX_ROLES = {"learned_observation_split", "learned_training_corpus"}
ROOT = p.ROOT/"data/atencion_armonica/learned_partition_reader_v1"
EDITED = {f"src/atencion_armonica/learned_partition_{name}.py" for name in
          ("model", "gate", "campaign", "selection", "data", "supervisor", "provenance")}


def _ref(value):
    return isinstance(value, dict) and set(value) == {"path", "sha256"}


@memoized
def origin():
    declaration = p.read_reference(p.REUSE_DECLARATION)
    prepared = p.read_reference(declaration["prepared"])
    if set(prepared) != PREPARED_FIELDS or len(prepared["normalized_train"]) != 8 or len(prepared["normalized_calibration"]) != 1:
        raise ValueError("reuse origin must be the complete declared prepared cohort")
    authorization = p.read_reference(prepared["authorization"])
    common = authorization["common"]
    if hashlib.sha256(encoded(common)).hexdigest() != declaration["producer_common_sha256"]:
        raise ValueError("reuse producer identity differs")
    from . import learned_partition_gate as gate
    if "reuse" in authorization:
        raise ValueError("only the declared original cohort may be imported")
    gate._verify_data_authorization(authorization, common)
    return declaration, prepared, authorization


def validate_source_change(current, producer):
    if any(current[k] != producer[k] for k in ("plan", "protocol", "runtime", "checkpoints")):
        raise ValueError("reuse changed scientific protocol, runtime or checkpoints")
    old, new = producer["source_sha256"], current["source_sha256"]
    added = {"src/atencion_armonica/learned_partition_reuse.py", p.AMENDMENT["path"], p.REUSE_DECLARATION["path"]}
    if (set(old)-set(new) or set(new)-set(old) != added
            or {k for k in old if old[k] != new[k]} != EDITED):
        raise ValueError("source changes exceed the independently audited guard/import amendment")
    old_tests, new_tests = producer["test_sha256"], current["test_sha256"]
    if (set(old_tests)-set(new_tests)
            or set(new_tests)-set(old_tests) != {"experiments/atencion_armonica/test_learned_partition_reuse.py"}
            or any(new_tests[k] != v for k, v in old_tests.items())):
        raise ValueError("reuse must preserve original test hashes and add only the declared regression suite")


@memoized
def verify_reuse_authorization(record, common):
    from . import learned_partition_gate as gate
    reuse = record["reuse"]
    if set(reuse) != {"declaration", "audit"} or reuse["declaration"] != p.REUSE_DECLARATION:
        raise ValueError("undeclared cohort reuse")
    gate.verify_audit(reuse["audit"], common, scope="PREPARED_REUSE_PLAN", target=p.REUSE_DECLARATION)
    _, _, old_auth = origin()
    validate_source_change(common, old_auth["common"])
    if record["training_device"] != old_auth["training_device"]:
        raise ValueError("initial-state continuity cannot change device")


def create_reuse_authorization(output, *, implementation_audit, profiles, reuse_plan_audit):
    from .learned_partition_gate import create_data_authorization
    return create_data_authorization(output, implementation_audit=implementation_audit, profiles=profiles,
        reuse={"declaration": p.REUSE_DECLARATION, "audit": reuse_plan_audit})


def _old_location(ref):
    path = p.verify_reference(ref)
    if (path.name != "manifest.json" or not any(path.is_relative_to(ROOT/name)
            for name in ("train", "calibration", "normalizers"))):
        raise ValueError("import source is outside the exact prepared cohort roots")
    return path


def _rewrite(value, old_common, new_common, resolve):
    if value == old_common:
        return copy.deepcopy(new_common)
    if _ref(value):
        return resolve(value)
    if isinstance(value, dict):
        return {k: _rewrite(v, old_common, new_common, resolve) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite(v, old_common, new_common, resolve) for v in value]
    return value


def import_prepared(output, *, authorization):
    from . import learned_partition_gate as gate
    started = time.monotonic()
    auth = gate.verify_authorization(authorization, "train")
    if "reuse" not in auth:
        raise PermissionError("import requires a non-generative reuse authorization")
    declaration, prepared, old_auth = origin()
    output = Path(output)
    if output != safe_member(p.ROOT, declaration["destination"]):
        raise ValueError("import destination differs from its prior declaration")
    old_common, common = old_auth["common"], auth["common"]
    output.mkdir(parents=True, exist_ok=False)
    mapping, active, entries, counts = {prepared["authorization"]["path"]: (prepared["authorization"], authorization)}, set(), [], Counter()
    payloads = byte_count = 0
    try:
        def resolve(ref):
            nonlocal payloads, byte_count
            if ref["path"] in mapping:
                original, imported = mapping[ref["path"]]
                if ref != original:
                    raise ValueError("same import path claimed with different SHA")
                return imported
            location = safe_member(p.ROOT, ref["path"])
            if not location.is_relative_to(ROOT):
                p.verify_reference(ref)
                return ref
            old_path = _old_location(ref)
            if ref["path"] in active:
                raise ValueError("cyclic prepared-cohort dependency")
            active.add(ref["path"])
            original = p.read_reference(ref)
            role = original["role"]
            if role not in ROLES or original["binding"].get("common") != old_common:
                raise ValueError("prepared role or original producer differs")
            verify_bundle(old_path.parent, ref["sha256"], role=role)
            binding = _rewrite(original["binding"], old_common, common, resolve)
            rewritten = {}
            if role in INDEX_ROLES:
                rewritten["shards.json"] = _rewrite(json.loads((old_path.parent/"shards.json").read_bytes()),
                    old_common, common, resolve)
            destination = output/old_path.parent.relative_to(ROOT)
            destination.mkdir(parents=True, exist_ok=False)
            for name, digest in original["artifacts_sha256"].items():
                source, target = safe_member(old_path.parent, name), safe_member(destination, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                if name in rewritten:
                    write_json(target, rewritten[name])
                else:
                    if target.exists():
                        raise FileExistsError("import payload already exists")
                    with source.open("rb") as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                    if sha_file(target) != digest:
                        raise ValueError("import changed scientific payload bytes")
                payloads += 1
                byte_count += source.stat().st_size
                if time.monotonic()-started > 1195:
                    raise TimeoutError("prepared import exhausted its CPU stage allowance")
            resources = {"operation": "IMPORT_REVALIDATION_NO_DRAWS_NO_FORWARDS",
                "producer_manifest": ref, "producer_resources": {
                    "path": (old_path.parent/"resources.json").relative_to(p.ROOT).as_posix(),
                    "sha256": original["resources_sha256"]}, "executor_common": common}
            seal_bundle(destination, role=role, binding=binding, resources=resources)
            imported = p.reference(destination/"manifest.json")
            mapping[ref["path"]] = (ref, imported)
            entries.append({"original": ref, "imported": imported, "rewritten_payloads": sorted(rewritten)})
            counts[role] += 1
            active.remove(ref["path"])
            return imported

        new_prepared = _rewrite(prepared, old_common, common, resolve)
        if (counts != Counter(ROLES) or len(entries) != declaration["expected_bundles"]
                or payloads != declaration["expected_payloads"] or byte_count != declaration["expected_payload_bytes"]):
            raise ValueError("import must cover all original bundles and payloads exactly once")
        write_json(output/"prepared.json", new_prepared)
        write_json(output/"mapping.json", entries)
        seal_bundle(output, role="learned_prepared_import", binding={"common": common,
            "authorization": authorization, "declaration": p.REUSE_DECLARATION,
            "producer_authorization": prepared["authorization"]}, resources={"operation": "PREPARED_COHORT_IMPORT",
            "seconds": time.monotonic()-started, "bundle_count": len(entries),
            "payload_count": payloads, "original_payload_bytes": byte_count})
        ref = p.reference(output/"manifest.json")
        verify_import(ref, common, full_bytes=True)
        return ref
    except BaseException as exc:
        mark_failure(output, exc)
        raise


@boundary
@memoized
def verify_import(ref, common, *, full_bytes=False):
    """Check all declared identities; original payload bytes only at import/audit."""
    declaration, prepared, old_auth = origin()
    path = p.verify_reference(ref)
    if path != safe_member(p.ROOT, declaration["destination"])/"manifest.json":
        raise ValueError("prepared import has an undeclared root")
    outer = p.read_reference(ref)
    if (outer.get("role") != "learned_prepared_import" or outer.get("status") != "COMPLETE"
            or outer.get("schema") != "structured-source-bundle-v1"
            or set(outer) != {"schema", "status", "role", "binding", "artifacts_sha256", "resources_sha256"}):
        raise ValueError("incomplete prepared import")
    b = outer["binding"]
    if (set(b) != {"common", "authorization", "declaration", "producer_authorization"}
            or b["common"] != common or b["declaration"] != p.REUSE_DECLARATION
            or b["producer_authorization"] != prepared["authorization"]):
        raise ValueError("import producer/executor or declaration differs")
    from . import learned_partition_gate as gate
    auth = gate.verify_authorization(b["authorization"], "train")
    if "reuse" not in auth:
        raise ValueError("import lost its reuse authorization")
    if full_bytes:
        verify_bundle(path.parent, ref["sha256"], role="learned_prepared_import")
    resource = p.read_reference({"path": (path.parent/"resources.json").relative_to(p.ROOT).as_posix(),
                                 "sha256": outer["resources_sha256"]})
    if (set(resource) != {"operation", "seconds", "bundle_count", "payload_count", "original_payload_bytes"}
            or resource["operation"] != "PREPARED_COHORT_IMPORT"
            or resource["bundle_count"] != declaration["expected_bundles"]
            or resource["payload_count"] != declaration["expected_payloads"]
            or resource["original_payload_bytes"] != declaration["expected_payload_bytes"]
            or type(resource["seconds"]) not in (int, float) or not math.isfinite(resource["seconds"])
            or not 0 < resource["seconds"] <= 1200):
        raise ValueError("import resource receipt differs from declared work")
    def member(name):
        return p.read_reference({"path": (path.parent/name).relative_to(p.ROOT).as_posix(),
                                 "sha256": outer["artifacts_sha256"][name]})
    entries, result = member("mapping.json"), member("prepared.json")
    if not isinstance(entries, list) or len(entries) != declaration["expected_bundles"]:
        raise ValueError("prepared import mapping count differs")
    mapping = {prepared["authorization"]["path"]: (prepared["authorization"], b["authorization"])}
    for entry in entries:
        if set(entry) != {"original", "imported", "rewritten_payloads"} or entry["original"]["path"] in mapping:
            raise ValueError("duplicate or malformed prepared import mapping")
        mapping[entry["original"]["path"]] = (entry["original"], entry["imported"])
    def resolve(value):
        if value["path"] in mapping:
            old, new = mapping[value["path"]]
            if value != old:
                raise ValueError("conflicting origin digest in import mapping")
            return new
        if safe_member(p.ROOT, value["path"]).is_relative_to(ROOT):
            raise ValueError("unmapped prepared-cohort dependency")
        return value
    counts, payloads, expected_outer = Counter(), 0, {"mapping.json", "prepared.json"}
    byte_count = 0
    for entry in entries:
        original, imported = entry["original"], entry["imported"]
        old_path = _old_location(original)
        new_path = p.verify_reference(imported)
        if new_path != path.parent/old_path.relative_to(ROOT):
            raise ValueError("import mapping moved a bundle outside its declared destination")
        old, new = p.read_reference(original), p.read_reference(imported)
        for manifest in (old, new):
            if (set(manifest) != {"schema", "status", "role", "binding", "artifacts_sha256", "resources_sha256"}
                    or manifest["schema"] != "structured-source-bundle-v1" or manifest["status"] != "COMPLETE"):
                raise ValueError("import mapping contains an incomplete or foreign bundle")
        if old["role"] not in ROLES or new["role"] != old["role"] or old["binding"].get("common") != old_auth["common"]:
            raise ValueError("import role or producer differs")
        if new["binding"] != _rewrite(old["binding"], old_auth["common"], common, resolve):
            raise ValueError("import changed bindings beyond declared source/reference updates")
        rewritten = ["shards.json"] if old["role"] in INDEX_ROLES else []
        if entry["rewritten_payloads"] != rewritten or set(new["artifacts_sha256"]) != set(old["artifacts_sha256"]):
            raise ValueError("import changed payload inventory or rewrite policy")
        if full_bytes:
            verify_bundle(old_path.parent, original["sha256"], role=old["role"])
            verify_bundle(new_path.parent, imported["sha256"], role=new["role"])
        for name, digest in old["artifacts_sha256"].items():
            if name in rewritten:
                old_index = p.read_reference({"path": (old_path.parent/name).relative_to(p.ROOT).as_posix(), "sha256": digest})
                new_index = p.read_reference({"path": (new_path.parent/name).relative_to(p.ROOT).as_posix(), "sha256": new["artifacts_sha256"][name]})
                if new_index != _rewrite(old_index, old_auth["common"], common, resolve):
                    raise ValueError("import shard topology changed")
            elif new["artifacts_sha256"][name] != digest:
                raise ValueError("scientific payload hash differs across import")
            payloads += 1
            if full_bytes:
                byte_count += (old_path.parent/name).stat().st_size
        resource_ref = {"path": (new_path.parent/"resources.json").relative_to(p.ROOT).as_posix(), "sha256": new["resources_sha256"]}
        if p.read_reference(resource_ref) != {"operation": "IMPORT_REVALIDATION_NO_DRAWS_NO_FORWARDS",
                "producer_manifest": original, "producer_resources": {
                    "path": (old_path.parent/"resources.json").relative_to(p.ROOT).as_posix(),
                    "sha256": old["resources_sha256"]}, "executor_common": common}:
            raise ValueError("import pretends to be a new scientific computation")
        counts[old["role"]] += 1
        prefix = new_path.parent.relative_to(path.parent).as_posix()+"/"
        expected_outer.update(prefix+n for n in new["artifacts_sha256"])
        expected_outer.update({prefix+"manifest.json", prefix+"resources.json"})
        if (outer["artifacts_sha256"].get(prefix+"manifest.json") != imported["sha256"]
                or outer["artifacts_sha256"].get(prefix+"resources.json") != new["resources_sha256"]
                or any(outer["artifacts_sha256"].get(prefix+n) != h for n, h in new["artifacts_sha256"].items())):
            raise ValueError("outer import inventory differs from its children")
    if (counts != Counter(ROLES) or payloads != declaration["expected_payloads"]
            or set(outer["artifacts_sha256"]) != expected_outer
            or (full_bytes and byte_count != declaration["expected_payload_bytes"])
            or result != _rewrite(prepared, old_auth["common"], common, resolve)):
        raise ValueError("prepared import omitted, added or changed cohort members")
    return result


@memoized
def verify_completion(audit, authorization, prepared):
    """Exact audit target/arguments, not a mutable flag allowing training."""
    from . import learned_partition_gate as gate
    auth = gate.verify_authorization(authorization, "train")
    if "reuse" not in auth:
        if audit is not None:
            raise ValueError("normal training cannot claim a reuse audit")
        return None
    if audit is None:
        raise PermissionError("reused preparation requires its independent completion audit")
    receipt = p.read_reference(audit)
    target = receipt.get("target")
    if not isinstance(target, dict) or set(target) != {"import", "prepared"}:
        raise ValueError("reuse audit must bind both the import and prepared index")
    gate.verify_audit(audit, auth["common"], scope="PREPARED_REUSE_COMPLETE", target=target)
    actual = verify_import(target["import"], auth["common"])
    root = p.verify_reference(target["import"]).parent
    if (p.verify_reference(target["prepared"]) != root/"prepared.json"
            or p.read_reference(target["prepared"]) != actual
            or actual != prepared or actual["authorization"] != authorization):
        raise ValueError("training arguments differ from the exact audited imported preparation")
    return target


def initial_continuity(resume, binding):
    """Return an explicitly rebound initial state, or None for ordinary resume."""
    declaration, prepared, old_auth = origin()
    if resume != declaration["failed_parent"]:
        return None
    verify_completion(binding.get("reuse_audit"), binding["authorization"],
                      {k: binding[k] for k in PREPARED_FIELDS})
    from .learned_partition_budget import terminal_receipt, accounting, recovery_status
    from .learned_partition_snapshots import read_snapshot
    terminal = terminal_receipt(resume["terminal"], request_ref=resume["request"])
    attempts, _, _ = accounting()
    if not any(row["terminal"] == resume["terminal"] for row in attempts):
        raise ValueError("reuse initial lost its original budget debit")
    request = p.read_reference(resume["request"])
    args = request["arguments"]
    original_binding = {"common": old_auth["common"], **prepared,
        **{k: args[k] for k in ("arm", "checkpoint_seed", "reader_seed")},
        "device": old_auth["training_device"], "count": 4096}
    if (any(args[k] != prepared[k] for k in PREPARED_FIELDS)
            or any(binding[k] != original_binding[k] for k in ("arm", "checkpoint_seed", "reader_seed", "device", "count"))
            or args["resume"] is not None or terminal["status"] != "FAILED"
            or terminal["recovery_status"] != "SNAPSHOT_REQUIRED"):
        raise ValueError("initial continuity changed its original cell or failure")
    parent = safe_member(p.ROOT, request["output"])
    if (recovery_status(parent, request) != "SNAPSHOT_REQUIRED"
            or json.loads((parent/"ancestry.json").read_bytes()) != {"resume": None, "snapshots": [], "calibrations": []}
            or list(parent.glob("calibration_*"))
            or sorted(f.relative_to(parent).as_posix() for f in parent.glob("snapshots/*/manifest.json")) != ["snapshots/initial/manifest.json"]):
        raise ValueError("initial continuity would discard additional training state")
    state, manifest = read_snapshot(resume["snapshot"], expected_binding=original_binding)
    if any(state[k] != 0 for k in ("epoch", "next_batch", "steps")) or manifest["parents"]:
        raise ValueError("only an untrained initial may cross this source amendment")
    restored = copy.deepcopy(state)
    restored["binding"] = copy.deepcopy(binding)
    return restored, [], []
