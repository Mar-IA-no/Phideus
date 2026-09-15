"""Final independent audit operator for geometric-decision energy.

One invocation holds the common lock across two irreversible phases. PRECOMMIT
authenticates only terminal metadata, the prospective freeze/seal, and sealed
observable metadata, then publishes the deterministic structural cut. VERIFY
re-authenticates that receipt before opening supervision, metrics, or report
payloads. No training, model load, forward, fitting, sampling, or CUDA path is
imported or called.

Do not run this operator until its source and fixtures receive the separate
review required by PLAN_GEOMETRIC_DECISION_FINAL_AUDIT.md.
"""
from __future__ import annotations

import time
LAUNCH_STARTED = time.monotonic()

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
from typing import Any

import numpy as np

from experiments.atencion_armonica import audit_geometric_decision_core as core
from src.atencion_armonica.geometric_decision_budget import LIMITS, STAGES as BUDGET_STAGES, StageBudget, BudgetExceeded
from src.atencion_armonica.geometric_decision_store import ArtifactStore


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/atencion_armonica/geometric_decision_energy_v1"
AUDIT_ROOT = BASE / "audit-final"
PREFLIGHT_ROOT = AUDIT_ROOT / "preflight"
PLAN = "experiments/atencion_armonica/PLAN_GEOMETRIC_DECISION_FINAL_AUDIT.md"
PROTOCOL = "experiments/atencion_armonica/PROTOCOL_GEOMETRIC_DECISION_ENERGY.md"
OPERATOR = "experiments/atencion_armonica/audit_geometric_decision.py"
CORE = "experiments/atencion_armonica/audit_geometric_decision_core.py"
PLAN_SHA = "7d3f085b955ab271d186ab1704a5f1a6f137e1fa779accae562eb2b3859620f2"
PROTOCOL_SHA = "72ef0c7b0e6e5ae1edc8f78cb8701f7bd5c31c2299c1ba756cd41ccdd5b62fc6"
CONTROL_BINDING = {"path": "binding.json", "bytes": 873,
    "sha256": "4e1ca362e88db55a1a600e245899d1e85a57ab0e57952bb75bc379861d440953"}
RESERVATION = 1500.0
PREFLIGHT_RESERVATION = 60.0
REQUIRED_COMPLETE = ("open", "training", "calibration-selection", "archive-and-exclusions",
    "profile-observed", "profile-closing", "admission-fresh", "prospective-observables",
    "admission-evaluate", "evaluate", "admission-replay", "observable-replay", "replay",
    "post-replay-report")

# Reviewed dispatch: schema -> required receipt-bearing fields and their owner.
# Fields not listed here are scalar metadata or are handled by their enclosing
# schema-specific walker; ownership is never inferred from a dict shaped like a receipt.
REFERENCE_DISPATCH = {
    "geometric-decision-attempt-v1": {"control": ("binding", "manifest")},
    "geometric-decision-attempt-finish-v1": {"control": ("start", "completion")},
    "geometric-decision-prospective-freeze-v1": {"control": ("exclusions",)},
    "geometric-decision-global-seal-v1": {"control": ("test_freeze",), "fresh": ("fresh_binding", "batches", "files")},
    "geometric-decision-postseal-source-contract-v1": {"control": ("test_freeze", "closing_origin")},
    "geometric-decision-open-prepared-v1": {"open": ("scale", "entries")},
    "geometric-decision-delivered-shard-v1": {"open": ("arrays",)},
    "generative-evidence-open-preparation-v1": {"historical-open": ("splits", "reuse")},
    "generative-evidence-prepared-split-v1": {"historical-open": ("shards",)},
    "generative-evidence-prepared-shard-v1": {"historical-open": ("raw", "targets", "metrics", "records")},
    "generative-evidence-fit-v1": {"historical-open": ("artifact",)},
    "generative-evidence-observable-v1": {"historical-open": ("fit", "raw")},
    "generative-evidence-supervision-v1": {"historical-open": ("observable", "targets", "metrics")},
    "generative-evidence-delivered-open-v1": {"historical-open": ("prepared", "normalizers", "entries")},
    "generative-evidence-delivered-index-v1": {"historical-open": ("prepared", "normalizers", "prepared_shard", "raw", "inputs")},
    "geometric-decision-campaign-complete-v1": {"training": ("cells",)},
    "geometric-decision-cell-complete-v1": {"training-cell": ("last_state", "calibration")},
    "geometric-decision-calibration-v1": {"training-cell": ("state", "predictions")},
    "geometric-decision-calibration-selection-v1": {"selection": ("arms",)},
    "geometric-decision-archive-complete-v1": {"archive": ("heads", "exclusions", "catalog")},
    "geometric-decision-head-archive-v1": {"archive": ("records",)},
    "geometric-decision-frozen-head-v1": {"archive": ("arrays", "source")},
    "geometric-decision-observed-run-v1": {"fresh": ("sources", "inputs", "records", "classical", "roundtrip")},
    "geometric-decision-source-batch-v1": {"fresh": ("sources",)},
    "geometric-decision-source-v1": {"fresh": ("arrays", "coordinates")},
    "geometric-decision-input-batch-v1": {"fresh": ("sources", "records")},
    "geometric-decision-fit-v1": {"fresh": ("source", "artifact")},
    "geometric-decision-inputs-v1": {"fresh": ("source", "fit", "arrays")},
    "geometric-decision-classical-index-v1": {"fresh": ("inputs", "records")},
    "geometric-decision-classical-v1": {"fresh": ("inputs", "source", "fit", "arrays")},
    "geometric-decision-prediction-v1": {"fresh": ("inputs", "head", "arrays")},
    "geometric-decision-transport-index-v1": {"fresh": ("prediction", "records")},
    "geometric-decision-transport-v1": {"fresh": ("prediction", "arrays")},
    "geometric-decision-draw-index-v1": {"fresh": ("records",)},
    "geometric-decision-draw-v1": {"fresh": ("observation", "sidecar")},
    "geometric-decision-evaluation-complete-v1": {"evaluation": ("results", "primary")},
    "geometric-decision-batch-evaluation-v1": {"evaluation": ("targets", "heads", "classical", "arrays")},
    "geometric-decision-report-v1": {"report": ("scenarios", "primary_source")},
    "geometric-decision-observed-profile-v1": {"profile-observed": ("result", "recovery", "runtime")},
    "geometric-decision-closing-profile-v1": {"profile-closing": (
        "admission", "original", "original_replay", "probe", "probe_replay", "bootstrap_fixture",
        "known_reconstructions", "observable_inventory", "exclusions")},
    "geometric-decision-profile-batch-v1": {"profile-inputs": ("rows",)},
    "geometric-decision-selection-profile-v1": {"profile-selection": ("io", "fixture_result", "snapshots")},
    "geometric-decision-snapshot-v1": {"profile": ("state", "previous")},
}


def source_reference(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    digest, size = core.sha256_path(path)
    return {"path": relative, "sha256": digest, "bytes": size}


def sources() -> list[dict[str, Any]]:
    return [source_reference(path) for path in (OPERATOR, CORE)]


def binding_ref(binding: dict[str, Any]) -> dict[str, Any]:
    raw = core.encoded(binding)
    return {"path": "binding.json", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def expect_schema(value: Any, schema: str, *, label: str) -> dict[str, Any]:
    if schema not in REFERENCE_DISPATCH:
        raise ValueError("auditor has no reviewed dispatch for required schema " + schema)
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{label} has unknown or wrong required schema")
    for fields in REFERENCE_DISPATCH[schema].values():
        if any(field not in value for field in fields):
            raise ValueError(f"{label} omits a required dispatched field")
    return value


def expect_binding(value: dict[str, Any], expected: dict[str, Any], *, label: str) -> None:
    """A hashed record must also identify the authenticated store it belongs to."""
    if value.get("binding") != expected:
        raise ValueError(f"{label} internal binding differs")


def nested_equal(actual: Any, expected: Any, label: str, *, atol: float = core.ATOL) -> None:
    """Exact for discrete/shape/dtype, declared tolerance only for floats."""
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray):
            raise ValueError(label + " type differs")
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(label + " dtype/shape differs")
        if actual.dtype.kind == "f":
            core.assert_close(actual, expected, label, atol=atol)
        else:
            core.assert_exact(actual, expected, label)
        return
    if isinstance(actual, dict) or isinstance(expected, dict):
        if not isinstance(actual, dict) or not isinstance(expected, dict) or set(actual) != set(expected):
            raise ValueError(label + " keys differ")
        for key in actual:
            nested_equal(actual[key], expected[key], label + "." + str(key), atol=atol)
        return
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or not isinstance(expected, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(label + " sequence differs")
        for i, (left, right) in enumerate(zip(actual, expected)):
            nested_equal(left, right, f"{label}[{i}]", atol=atol)
        return
    if type(actual) is not type(expected):
        raise ValueError(label + " scalar type differs")
    if isinstance(actual, float) or isinstance(expected, float):
        if actual is None or expected is None:
            if actual is not expected:
                raise ValueError(label + " null differs")
        elif not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
            raise ValueError(label + " float differs")
        return
    if actual != expected:
        raise ValueError(label + " differs")


def compressed_ref(value: Any) -> dict[str, Any]:
    required = {"path", "bytes", "sha256", "decoded_sha256", "decoded_bytes", "codec"}
    if (not isinstance(value, dict) or set(value) != required
            or value["codec"] != "canonical-json-gzip3-mtime0"
            or type(value["decoded_bytes"]) is not int or value["decoded_bytes"] < 0
            or type(value["decoded_sha256"]) is not str or len(value["decoded_sha256"]) != 64):
        raise ValueError("compressed artifact receipt differs")
    return {key: value[key] for key in ("path", "bytes", "sha256")}


def read_ledger(control: core.AuditStore, binding: dict[str, Any], *, active_start=None,
                check=lambda: None) -> dict[str, Any]:
    """Independent replay of the cumulative ledger, including failed/crashed attempts."""
    expected_stages = {"profile": 600.0, "open": 1800.0, "training": 14400.0,
        "fresh": 21600.0, "evaluation": 7200.0, "audit": 3600.0}
    if binding.get("limits") != LIMITS or LIMITS["stages"] != expected_stages or LIMITS["total_seconds"] != 49200.0:
        raise ValueError("control binding does not retain fixed budget")
    charged = {name: 0.0 for name in expected_stages}
    for row in binding.get("prior_charges", []):
        if set(row) != {"stage", "seconds", "source"} or row["stage"] not in charged:
            raise ValueError("prior charge schema differs")
        core.project_reference(ROOT, row["source"], check=check)
        charged[row["stage"]] += row["seconds"]
    attempts, operations = [], {}
    starts = sorted(control.path("attempts").glob("*/start.json"))
    for i, path in enumerate(starts):
        check()
        if path.parent.name != f"{i:04d}":
            raise ValueError("attempt ledger is not contiguous")
        sref = control.current_reference(f"attempts/{i:04d}/start.json", check=check)
        start = expect_schema(control.json(sref, check=check), "geometric-decision-attempt-v1", label="attempt start")
        if (set(start) != {"schema", "binding", "manifest", "stage", "reservation_seconds", "charged_before"}
                or start["binding"] != binding or start["stage"] not in charged
                or start["charged_before"] != charged or not 0 < start["reservation_seconds"] <= expected_stages[start["stage"]]):
            raise ValueError("attempt start loses cumulative authority")
        manifest = control.json(start["manifest"], check=check)
        if not isinstance(manifest, dict) or type(manifest.get("operation")) is not str:
            raise ValueError("attempt manifest lacks explicit operation")
        finish_path = control.path(f"attempts/{i:04d}/finish.json")
        if not finish_path.exists():
            if active_start is not None and sref == active_start and i == len(starts) - 1:
                attempts.append({"start_ref": sref, "start": start, "manifest": manifest,
                                 "finish_ref": None, "finish": None, "output": None})
                continue
            charged[start["stage"]] += start["reservation_seconds"]
            attempts.append({"start_ref": sref, "start": start, "manifest": manifest,
                             "finish_ref": None, "finish": None, "output": None})
            continue
        fref = control.current_reference(f"attempts/{i:04d}/finish.json", check=check)
        finish = expect_schema(control.json(fref, check=check), "geometric-decision-attempt-finish-v1", label="attempt finish")
        expected_after = {**charged, start["stage"]: charged[start["stage"]] + finish["seconds"]}
        if (set(finish) != {"schema", "start", "status", "seconds", "charged_after", "completion"}
                or finish["start"] != sref or finish["status"] not in ("COMPLETE", "PAUSED", "FAILED", "LIMIT_REACHED")
                or not math.isfinite(finish["seconds"]) or finish["seconds"] < 0
                or (finish["seconds"] >= start["reservation_seconds"] and finish["status"] != "LIMIT_REACHED")
                or finish["charged_after"] != expected_after
                or (finish["completion"] is None) != (finish["status"] != "COMPLETE")):
            raise ValueError("attempt finish arithmetic or authority differs")
        output = control.json(finish["completion"], check=check) if finish["completion"] is not None else None
        if output is not None and (not isinstance(output, dict) or output.get("manifest") != start["manifest"]):
            raise ValueError("COMPLETE output does not link to its start manifest")
        charged = expected_after
        record = {"start_ref": sref, "start": start, "manifest": manifest,
                  "finish_ref": fref, "finish": finish, "output": output}
        attempts.append(record)
        if finish["status"] == "COMPLETE":
            operations.setdefault(manifest["operation"], []).append(record)
    if any(charged[name] > expected_stages[name] for name in charged) or sum(charged.values()) > LIMITS["total_seconds"]:
        raise ValueError("ledger exceeds fixed stage or total cap")
    return {"attempts": attempts, "operations": operations, "charged": charged}


def unique_operation(ledger: dict[str, Any], name: str) -> dict[str, Any]:
    rows = ledger["operations"].get(name, [])
    if len(rows) != 1:
        raise ValueError(f"requires exactly one COMPLETE {name}")
    return rows[0]


def open_bound(label: str, root: Path, binding: dict[str, Any], coverage: core.Coverage,
               *, check=lambda: None) -> core.AuditStore:
    store = core.AuditStore(label, root, coverage)
    actual = store.json(binding_ref(binding), check=check)
    if actual != binding:
        raise ValueError(f"{label} binding differs")
    return store


def authenticate_sources(value: dict[str, Any], *, check=lambda: None) -> None:
    for field in ("code", "original_code", "late_sources", "execution_sources", "catalog_sources"):
        if field in value:
            for ref in value[field]:
                core.project_reference(ROOT, ref, check=check)
    if "protocol" in value and isinstance(value["protocol"], dict) and "path" in value["protocol"]:
        current = core.project_reference(ROOT, value["protocol"], check=check)
        if current["sha256"] != PROTOCOL_SHA:
            raise ValueError("scientific protocol hash differs")


def historical_reference_inventory(ledger: dict[str, Any], *, check=lambda: None) -> dict[str, Any]:
    """Walk historical receipt metadata only; payload hashing stays in VERIFY."""
    open_row = unique_operation(ledger, "open")
    source = open_row["manifest"]["preparation_binding"]["source"]
    store = open_bound("historical-projection", Path(source["root"]), source["binding"],
                       core.Coverage(), check=check)
    records: dict[tuple[str, str, int, str], dict[str, Any]] = {}

    def add(ref: dict[str, Any], kind: str, authenticated: bool) -> None:
        item = core.receipt(ref)
        key = ("historical-open", item["path"], item["bytes"], item["sha256"])
        previous = records.get(key)
        if previous is None or authenticated:
            records[key] = {**item, "store": "historical-open", "kind": kind,
                            "authenticated_in_preflight": authenticated}

    binding_receipt = {**binding_ref(source["binding"])}
    add(binding_receipt, "metadata", True)
    external_candidates: list[dict[str, Any]] = []

    def metadata(ref: dict[str, Any], schema: str | None = None) -> dict[str, Any]:
        value = store.json(ref, check=check)
        add(ref, "metadata", True)
        return expect_schema(value, schema, label="historical projection") if schema else value

    prepared = metadata(source["prepared"], "generative-evidence-open-preparation-v1")
    normalizers = metadata(source["normalizers"])
    if prepared["normalizers"] != source["normalizers"] or normalizers["prepared_train"] != prepared["splits"]["train"]:
        raise ValueError("historical projection entry lineage differs")
    for split, split_ref in prepared["splits"].items():
        split_row = metadata(split_ref, "generative-evidence-prepared-split-v1")
        if split_row["split"] != split:
            raise ValueError("historical projection split identity differs")
        for shard_ref in split_row["shards"]:
            shard = metadata(shard_ref, "generative-evidence-prepared-shard-v1")
            for ref in shard["raw"].values(): add(ref, "array-payload", False)
            add(shard["targets"], "array-payload", False)
            add(shard["metrics"], "metric-payload", False)
            for group in shard["records"]:
                fit = metadata(group["fit"], "generative-evidence-fit-v1")
                add(compressed_ref(fit["artifact"]), "gzip-payload", False)
                observable = metadata(group["observable"], "generative-evidence-observable-v1")
                if observable["fit"] != group["fit"]:
                    raise ValueError("historical projection observable fit link differs")
                for ref in observable["raw"].values(): add(ref, "array-payload", False)
                supervision = metadata(group["supervision"], "generative-evidence-supervision-v1")
                if supervision["observable"] != group["observable"]:
                    raise ValueError("historical projection supervision link differs")
                add(supervision["targets"], "array-payload", False)
                add(supervision["metrics"], "metric-payload", False)
                for field in ("authorization", "import", "data", "sidecars"):
                    if field in supervision.get("source", {}):
                        external_candidates.append(supervision["source"][field])
    delivered = metadata(source["delivered"], "generative-evidence-delivered-open-v1")
    if delivered["prepared"] != source["prepared"] or delivered["normalizers"] != source["normalizers"]:
        raise ValueError("historical projection delivered lineage differs")
    for entry in delivered["entries"]:
        index = metadata(entry["index"], "generative-evidence-delivered-index-v1")
        for field in ("prepared_shard", "raw", "inputs"):
            add(index[field], "array-payload" if field != "prepared_shard" else "metadata-payload", False)

    external: dict[tuple[str, str], dict[str, Any]] = {}
    reuse = prepared["reuse"]
    candidates = [reuse[field] for field in ("authorization", "import")]
    candidates.extend(reuse.get("corpora", {}).values())
    candidates.extend({"path": path, "sha256": digest} for path, digest in reuse.get("consumed_sha256", {}).items())
    candidates.extend(external_candidates)
    for ref in candidates:
        validated = core.receipt(ref, sizes="bytes" in ref)
        path = ROOT / validated["path"]
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve())):
            raise ValueError("historical projection external source path differs")
        size = validated.get("bytes", path.stat().st_size)
        external[(validated["path"], validated["sha256"])] = {
            "path": validated["path"], "sha256": validated["sha256"], "bytes": size,
            "authenticated_in_preflight": False}
    values = list(records.values())
    return {"schema": "geometric-decision-historical-reference-inventory-v1",
        "root": str(store.root), "files": len(values), "bytes": sum(row["bytes"] for row in values),
        "metadata_authenticated_files": sum(row["authenticated_in_preflight"] for row in values),
        "metadata_authenticated_bytes": sum(row["bytes"] for row in values if row["authenticated_in_preflight"]),
        "payload_hash_deferred_files": sum(not row["authenticated_in_preflight"] for row in values),
        "external_declared_files": len(external),
        "external_declared_bytes": sum(row["bytes"] for row in external.values()),
        "records": values, "external": list(external.values()),
        "authority": "receipt metadata inventory; payload bytes are authenticated only in VERIFY"}


def measure_independent_kernels(*, scene_units: int = 20, head_units: int = 144,
                                bootstrap_scenes: int = 512, check=lambda: None,
                                clock=time.monotonic) -> dict[str, Any]:
    """CPU fixture timing for this checker, not a scientific observation."""
    if any(type(value) is not int or value <= 0 for value in (scene_units, head_units, bootstrap_scenes)):
        raise ValueError("independent projection fixture units differ")
    partitions: list[tuple[tuple[int, ...], ...]] = [((0,),)]
    for event in range(1, 8):
        expanded = []
        for partition in partitions:
            for group_index in range(len(partition)):
                groups = [list(group) for group in partition]
                groups[group_index].append(event)
                expanded.append(tuple(tuple(group) for group in groups))
            expanded.append((*partition, (event,)))
        partitions = expanded
    partitions = sorted(partitions)[:82]
    target = core.independent_targets(partitions, np.asarray([0, 0, 1, 1, 2, 2, 3, 3], np.int64))
    components = target["u32"].astype(np.float64)
    energy = components.sum(1, dtype=np.float64)
    started = clock()
    for _ in range(scene_units):
        for _ in range(head_units):
            core.describe(partitions, target, energy, components)
        check()
    scene_seconds = clock() - started
    values = np.zeros((bootstrap_scenes, 8, 3, 3), np.float64)
    started = clock()
    core.primary(values, np.ones(bootstrap_scenes, np.bool_), check=check)
    bootstrap_seconds = clock() - started
    if any(not math.isfinite(value) or value <= 0 for value in (scene_seconds, bootstrap_seconds)):
        raise ValueError("independent projection fixture timing differs")
    return {"schema": "geometric-decision-independent-kernel-profile-v1",
        "scene_units": scene_units, "head_units": head_units, "candidate_units": 82,
        "scene_head_seconds": scene_seconds, "bootstrap_scenes": bootstrap_scenes,
        "bootstrap_seconds": bootstrap_seconds,
        "authority": "abstract CPU timing fixture; no campaign data or scientific evidence"}


def inventory_projection(ledger: dict[str, Any], seal: dict[str, Any], basis: dict[str, Any],
                         historical_inventory: dict[str, Any], audit_measurement: dict[str, Any],
                         *, check=lambda: None) -> dict[str, Any]:
    files = seal.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("sealed file inventory required for projection")
    fresh_bytes = sum(core.receipt(row)["bytes"] for row in files)
    roots = {BASE / "control", Path(seal["fresh_root"])}
    for name in ("open", "training", "calibration-selection", "archive-and-exclusions",
                 "profile-observed", "profile-closing", "evaluate", "post-replay-report"):
        record = unique_operation(ledger, name)
        root = record["manifest"].get("root", record["output"].get("root"))
        if type(root) is not str:
            raise ValueError("projection lacks an explicit operation root")
        roots.add(Path(root))
    for operation in ("profile-inputs", "profile-head", "profile-fitter", "profile-selection"):
        for record in ledger["operations"].get(operation, []):
            root = record["manifest"].get("root")
            if type(root) is not str or record["output"].get("root") != root:
                raise ValueError("projection lacks an explicit profile root")
            roots.add(Path(root))
    inventories, total_files, total_bytes = [], 0, 0
    for root in sorted(roots, key=str):
        files_here = bytes_here = 0
        for path in root.rglob("*"):
            check()
            if path.is_file() and not path.is_symlink():
                files_here += 1
                bytes_here += path.stat().st_size
        inventories.append({"root": str(root), "files": files_here, "bytes": bytes_here})
        total_files += files_here
        total_bytes += bytes_here
    if (historical_inventory.get("schema") != "geometric-decision-historical-reference-inventory-v1"
            or historical_inventory.get("files") != len(historical_inventory.get("records", []))):
        raise ValueError("historical projection inventory differs")
    historical_bytes = historical_inventory["bytes"] + historical_inventory["external_declared_bytes"]
    historical_files = historical_inventory["files"] + historical_inventory["external_declared_files"]
    timings, measured_bytes = basis["timings"], basis["inventory_bytes"]
    overhead = basis.get("overhead_seconds")
    if (set(("observable-inventory", "original-metrics", "probe-metrics", "bootstrap512")) - set(timings)
            or any(type(timings[key]) not in (int, float) or not math.isfinite(timings[key]) or timings[key] <= 0
                   for key in ("observable-inventory", "original-metrics", "probe-metrics", "bootstrap512"))
            or type(measured_bytes) is not int or measured_bytes <= 0
            or type(overhead) not in (int, float) or not math.isfinite(overhead) or overhead < 0):
        raise ValueError("audit projection lacks authenticated measured closing-profile basis")
    scan = timings["observable-inventory"] * (total_bytes + historical_bytes) / measured_bytes
    # Producer timings and this checker's abstract fixture are distinct scale
    # observations. Use the slower projection after normalizing both workloads.
    producer_scaled = (timings["original-metrics"] + timings["probe-metrics"]) * (48 / 20)
    if (audit_measurement.get("schema") != "geometric-decision-independent-kernel-profile-v1"
            or audit_measurement.get("candidate_units") != 82
            or audit_measurement.get("scene_units", 0) <= 0
            or audit_measurement.get("head_units", 0) <= 0
            or audit_measurement.get("bootstrap_scenes", 0) <= 0
            or any(type(audit_measurement.get(field)) not in (int, float)
                   or not math.isfinite(audit_measurement[field]) or audit_measurement[field] <= 0
                   for field in ("scene_head_seconds", "bootstrap_seconds"))):
        raise ValueError("independent audit kernel measurement differs")
    independent_scaled = (audit_measurement["scene_head_seconds"]
        * (48 * 144) / (audit_measurement["scene_units"] * audit_measurement["head_units"]))
    cut_compute = max(producer_scaled, independent_scaled)
    bootstrap = max(timings["bootstrap512"], audit_measurement["bootstrap_seconds"]
                    * 512 / audit_measurement["bootstrap_scenes"])
    raw_seconds = 3 * scan + cut_compute + bootstrap + overhead
    projected = 1.25 * raw_seconds
    return {"schema": "geometric-decision-final-audit-projection-v1", "inventories": inventories,
        "total_files": total_files, "total_bytes": total_bytes,
        "historical_declared_files": historical_files, "historical_declared_bytes": historical_bytes,
        "historical_inventory": historical_inventory,
        "sealed_fresh_files": len(files), "sealed_fresh_bytes": fresh_bytes,
        "basis": basis, "audit_measurement": audit_measurement, "scan_passes": 3,
        "maximum_cut_original_scenes": 32, "maximum_cut_derived_scenes": 16,
        "producer_scaled_seconds": producer_scaled, "independent_scaled_seconds": independent_scaled,
        "cut_compute_seconds": cut_compute, "bootstrap_seconds": bootstrap,
        "base_seconds": raw_seconds, "margin": 1.25,
        "projected_seconds": projected,
        "scope": "measured closing-profile inventory/metric/bootstrap rates scaled to explicit audit inventory and maximum structural cut"}


def verify_seal_inventory(fresh: core.AuditStore, seal: dict[str, Any], *, check=lambda: None) -> None:
    expected = seal["files"]
    actual_paths = []
    for path in sorted(fresh.root.rglob("*")):
        check()
        if path.is_symlink():
            raise ValueError("fresh inventory contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file() or path.suffix not in (".json", ".npz", ".gz"):
            raise ValueError("fresh inventory contains an unknown file type")
        actual_paths.append(path.relative_to(fresh.root).as_posix())
    if actual_paths != [row["path"] for row in expected]:
        raise ValueError("current fresh inventory path roster differs from seal")
    for ref in expected:
        fresh.authenticate(ref, kind="sealed-file", check=check)


def verify_fresh_reference_closure(fresh: core.AuditStore, seal: dict[str, Any], *, check=lambda: None) -> dict[str, int]:
    """Schema/field walk of the sealed fresh graph; truth sidecars remain opaque."""
    counts = {"sources": 0, "draws": 0, "inputs": 0, "predictions": 0,
              "transports": 0, "classical": 0, "sidecars_opaque": 0}
    binding = fresh.json(seal["fresh_binding"], check=check)
    original_heads = None
    for sealed in seal["batches"]:
        draw_index = expect_schema(fresh.json(sealed["draws"], check=check),
                                   "geometric-decision-draw-index-v1", label="draw closure")
        observed = expect_schema(fresh.json(sealed["observed"], check=check), "geometric-decision-observed-run-v1", label="observed closure")
        if draw_index["binding"] != binding or observed["binding"] != binding:
            raise ValueError("fresh closure binding differs")
        for batch, expected_ids in ((observed, list(range(512))),
                                    (observed["roundtrip"], observed["roundtrip_scene_ids"])):
            if batch is None:
                if expected_ids: raise ValueError("missing roundtrip graph")
                continue
            sources = expect_schema(fresh.json(batch["sources"], check=check), "geometric-decision-source-batch-v1", label="source closure")
            inputs = expect_schema(fresh.json(batch["inputs"], check=check),
                                   "geometric-decision-input-batch-v1", label="input closure")
            classical = expect_schema(fresh.json(batch["classical"], check=check),
                                      "geometric-decision-classical-index-v1", label="classical closure")
            if (sources["scene_ids"] != expected_ids or inputs["scene_ids"] != expected_ids
                    or classical["scene_ids"] != expected_ids or len(sources["sources"]) != len(expected_ids)
                    or len(inputs["records"]) != len(expected_ids) or len(classical["records"]) != len(expected_ids)):
                raise ValueError("fresh source/input/classical roster differs")
            if any(row["binding"] != binding for row in (sources, inputs, classical)):
                raise ValueError("fresh batch child binding differs")
            for i, sid in enumerate(expected_ids):
                source = expect_schema(fresh.json(sources["sources"][i], check=check), "geometric-decision-source-v1", label="source scene closure")
                inp = expect_schema(fresh.json(inputs["records"][i], check=check),
                                    "geometric-decision-inputs-v1", label="scene input closure")
                classic = expect_schema(fresh.json(classical["records"][i], check=check),
                                        "geometric-decision-classical-v1", label="scene classical closure")
                if (source["scene"]["observation"]["scene_id"] != sid or inp["source"] != sources["sources"][i]
                        or classic["source"] != sources["sources"][i] or classic["inputs"] != inputs["records"][i]
                        or any(row["binding"] != binding for row in (source, inp, classic))):
                    raise ValueError("fresh scene graph cross-link differs")
                fresh.authenticate(source["arrays"], kind="npz-reference", check=check)
                if source["coordinates"] is not None:
                    fresh.authenticate(source["coordinates"], kind="npz-reference", check=check)
                fit = expect_schema(fresh.json(inp["fit"], check=check),
                                    "geometric-decision-fit-v1", label="fit closure")
                if fit["source"] != sources["sources"][i] or fit["binding"] != binding:
                    raise ValueError("fresh fit source differs")
                fresh.authenticate(compressed_ref(fit["artifact"]), kind="gzip", check=check)
                fresh.authenticate(inp["arrays"], kind="npz-reference", check=check)
                fresh.authenticate(classic["arrays"], kind="npz-reference", check=check)
                counts["sources"] += 1; counts["inputs"] += 1; counts["classical"] += 1
            if len(batch["records"]) != 144:
                raise ValueError("fresh readout roster does not contain 144 heads")
            heads = [record["head"] for record in batch["records"]]
            if len({core.encoded(ref) for ref in heads}) != 144:
                raise ValueError("fresh readout head roster contains duplicates")
            if original_heads is None:
                original_heads = heads
            elif heads != original_heads:
                raise ValueError("fresh original/roundtrip head roster differs")
            for record in batch["records"]:
                prediction = expect_schema(fresh.json(record["prediction"], check=check),
                    "geometric-decision-prediction-v1", label="prediction closure")
                if (prediction["inputs"] != batch["inputs"] or prediction["head"] != record["head"]
                        or prediction["binding"] != binding):
                    raise ValueError("fresh prediction graph differs")
                fresh.authenticate(prediction["arrays"], kind="npz-reference", check=check)
                counts["predictions"] += 1
                if record["transport"] is not None:
                    transports = expect_schema(fresh.json(record["transport"], check=check),
                        "geometric-decision-transport-index-v1", label="transport closure")
                    if transports["prediction"] != record["prediction"] or transports["scene_ids"] != observed["roundtrip_scene_ids"]:
                        raise ValueError("fresh transport index differs")
                    for tref in transports["records"]:
                        transport = expect_schema(fresh.json(tref, check=check),
                            "geometric-decision-transport-v1", label="transport record closure")
                        if (transport["prediction"] != record["prediction"]
                                or transport["binding"] != binding):
                            raise ValueError("fresh transport parent differs")
                        fresh.authenticate(transport["arrays"], kind="npz-reference", check=check)
                        counts["transports"] += 1
        if draw_index["scene_ids"] != list(range(512)) or len(draw_index["records"]) != 512:
            raise ValueError("fresh draw roster differs")
        for sid, ref in enumerate(draw_index["records"]):
            draw = expect_schema(fresh.json(ref, check=check), "geometric-decision-draw-v1", label="draw record closure")
            observation = fresh.json(draw["observation"], check=check)
            if observation["scene_id"] != sid or draw["binding"] != binding:
                raise ValueError("fresh draw identity differs")
            fresh.authenticate(draw["sidecar"], kind="sidecar-opaque", check=check)
            counts["draws"] += 1; counts["sidecars_opaque"] += 1
    return counts


def admit_metadata(control: core.AuditStore, ledger: dict[str, Any], *, check=lambda: None) -> dict[str, Any]:
    for name in REQUIRED_COMPLETE:
        unique_operation(ledger, name)
    prospective = unique_operation(ledger, "prospective-observables")
    evaluate = unique_operation(ledger, "evaluate")
    replay = unique_operation(ledger, "replay")
    recovery = unique_operation(ledger, "observable-replay")
    report = unique_operation(ledger, "post-replay-report")
    freeze_ref = prospective["output"]["test_freeze"]
    seal_ref = prospective["output"]["seal"]
    freeze = expect_schema(control.json(freeze_ref, check=check), "geometric-decision-prospective-freeze-v1", label="freeze")
    seal = expect_schema(control.json(seal_ref, check=check), "geometric-decision-global-seal-v1", label="seal")
    if (freeze["test_roster"] != [{"split": split, "split_seed": seed, "count": 512} for split, seed in core.TESTS]
            or len(freeze["head_roster"]) != 144 or seal["test_freeze"] != freeze_ref
            or seal["original_scenes"] != 2048 or len(seal["batches"]) != 4):
        raise ValueError("freeze/seal scope differs")
    contract_ref = evaluate["manifest"].get("execution_contract")
    contract = expect_schema(control.json(contract_ref, check=check), "geometric-decision-postseal-source-contract-v1", label="postseal contract")
    if (contract["test_freeze"] != freeze_ref or replay["manifest"].get("execution_contract") != contract_ref
            or replay["output"].get("execution_contract") != contract_ref
            or recovery["finish_ref"] != replay["manifest"].get("observable_replay_finish")):
        raise ValueError("postseal evaluation/replay chain differs")
    authenticate_sources(freeze, check=check)
    authenticate_sources(contract, check=check)
    closing = unique_operation(ledger, "profile-closing")
    origin = contract.get("closing_origin")
    if (origin != {"finish": closing["finish_ref"], "root": closing["output"]["root"],
                   "binding": closing["output"]["binding"], "result": closing["output"]["result"]}
            or freeze["budget"]["forecast"]["profile_finish"] != closing["finish_ref"]
            or freeze["budget"]["forecast"]["profile_report"] != closing["output"]["result"]):
        raise ValueError("late-source closing profile origin differs from freeze")
    closing_binding = closing["manifest"]["binding"]
    closing_store = open_bound("profile-closing-preflight", Path(closing["output"]["root"]),
                               closing_binding, core.Coverage(), check=check)
    closing_report = expect_schema(closing_store.json(closing["output"]["result"], check=check),
        "geometric-decision-closing-profile-v1", label="closing projection basis")
    inventory = closing_store.json(closing_report["observable_inventory"], check=check)
    measured_files = inventory.get("files")
    if not isinstance(measured_files, list) or not measured_files:
        raise ValueError("closing profile inventory basis differs")
    projection_basis = {"profile_result": closing["output"]["result"],
        "observable_inventory": closing_report["observable_inventory"],
        "timings": {key: closing_report["timings"][key] for key in
                    ("observable-inventory", "original-metrics", "probe-metrics", "bootstrap512")},
        "inventory_bytes": sum(core.receipt(ref)["bytes"] for ref in measured_files),
        "overhead_seconds": closing_report["overhead_seconds"]}
    report_output = report["output"]
    if report_output.get("root") != str(BASE / "report") or "result" not in report_output:
        raise ValueError("report terminal output differs")
    return {"freeze_ref": freeze_ref, "seal_ref": seal_ref, "contract_ref": contract_ref,
        "terminal_receipts": {name: unique_operation(ledger, name)["finish_ref"] for name in REQUIRED_COMPLETE},
        "report_result_ref": report_output["result"], "fresh_binding_ref": seal["fresh_binding"],
        "projection_basis": projection_basis}


def precommit_inputs(control: core.AuditStore, ledger: dict[str, Any], coverage: core.Coverage,
                     metadata: dict[str, Any], *, check=lambda: None) -> dict[str, Any]:
    """Only sealed observable metadata; called after the audit attempt starts."""
    seal = control.json(metadata["seal_ref"], check=check)
    fresh = core.AuditStore("fresh", Path(seal["fresh_root"]), coverage)
    fresh_binding = fresh.json(seal["fresh_binding"], check=check)
    if fresh_binding != {"test_freeze": metadata["freeze_ref"]}:
        raise ValueError("fresh binding differs")
    verify_seal_inventory(fresh, seal, check=check)
    cuts = []
    for batch, (split, seed) in zip(seal["batches"], core.TESTS):
        observed = expect_schema(fresh.json(batch["observed"], check=check), "geometric-decision-observed-run-v1", label="observed batch")
        if observed["split"] != split or observed["split_seed"] != seed or observed["scene_ids"] != list(range(512)):
            raise ValueError("sealed observed roster differs")
        source_index = expect_schema(fresh.json(observed["sources"], check=check), "geometric-decision-source-batch-v1", label="source batch")
        rows, refs = [], source_index["sources"]
        if len(refs) != 512 or source_index["scene_ids"] != list(range(512)):
            raise ValueError("source batch does not cover 512 scenes")
        for sid, ref in enumerate(refs):
            source = expect_schema(fresh.json(ref, check=check), "geometric-decision-source-v1", label="source scene")
            scene = source["scene"]
            if scene["observation"]["scene_id"] != sid:
                raise ValueError("source scene identity differs")
            rows.append({"scene_id": sid, "partitions": scene["partitions"]})
        cut = core.select_structural_cut(rows, observed["roundtrip_scene_ids"])
        derived = []
        if observed["roundtrip"] is not None:
            derived_index = fresh.json(observed["roundtrip"]["sources"], check=check)
            if derived_index["scene_ids"] != observed["roundtrip_scene_ids"]:
                raise ValueError("derived source roster differs from probe roster")
            derived = derived_index["sources"]
        cut.update(split=split, split_seed=seed, observed=batch["observed"], sources=observed["sources"],
                   source_refs=[refs[sid] for sid in cut["scene_ids"]], derived_source_refs=derived)
        cuts.append(cut)
    return {**metadata, "cuts": cuts}


def walk_historical_open(store: core.AuditStore, prepared_ref: dict[str, Any], delivered_ref: dict[str, Any],
                         normalizer_ref: dict[str, Any], *, check=lambda: None) -> None:
    """Explicit schema/field traversal of the reused TRAIN/calibration aggregate."""
    prepared = expect_schema(store.json(prepared_ref, check=check), "generative-evidence-open-preparation-v1", label="historical prepared")
    normalizers = store.json(normalizer_ref, check=check)
    if prepared["normalizers"] != normalizer_ref or normalizers["prepared_train"] != prepared["splits"]["train"]:
        raise ValueError("historical normalizer/prepared lineage differs")
    for split, split_ref in prepared["splits"].items():
        split_row = expect_schema(store.json(split_ref, check=check), "generative-evidence-prepared-split-v1", label="prepared split")
        if split_row["split"] != split:
            raise ValueError("historical split identity differs")
        for shard_ref in split_row["shards"]:
            shard = expect_schema(store.json(shard_ref, check=check), "generative-evidence-prepared-shard-v1", label="prepared shard")
            for ref in shard["raw"].values():
                store.arrays(ref, check=check)
            store.arrays(shard["targets"], check=check)
            store.json(shard["metrics"], check=check)
            for records in shard["records"]:
                for kind in ("fit", "observable", "supervision"):
                    row = expect_schema(store.json(records[kind], check=check), f"generative-evidence-{kind}-v1", label="prepared scene")
                    if kind == "fit":
                        store.authenticate(compressed_ref(row["artifact"]), kind="gzip", check=check)
                    elif kind == "observable":
                        store.json(row["fit"], check=check)
                        for ref in row["raw"].values():
                            store.arrays(ref, check=check)
                    else:
                        store.json(row["observable"], check=check)
                        store.arrays(row["targets"], check=check)
                        store.json(row["metrics"], check=check)
                        source = row.get("source")
                        if isinstance(source, dict):
                            for field in ("authorization", "import", "data", "sidecars"):
                                if field in source:
                                    core.project_reference(ROOT, source[field], check=check)
    delivered = expect_schema(store.json(delivered_ref, check=check), "generative-evidence-delivered-open-v1", label="historical delivered")
    if delivered["prepared"] != prepared_ref or delivered["normalizers"] != normalizer_ref:
        raise ValueError("historical delivered parent differs")
    for entry in delivered["entries"]:
        index = expect_schema(store.json(entry["index"], check=check), "generative-evidence-delivered-index-v1", label="delivered index")
        if (index["prepared"] != prepared_ref or index["normalizers"] != normalizer_ref
                or index["split"] != entry["split"] or index["checkpoint_seed"] != entry["checkpoint_seed"]
                or index["shard"] != entry["shard"]):
            raise ValueError("historical delivered identity differs")
        store.json(index["prepared_shard"], check=check)
        store.arrays(index["raw"], check=check)
        store.arrays(index["inputs"], check=check)
    reuse = prepared["reuse"]
    for field in ("authorization", "import"):
        core.project_reference(ROOT, reuse[field], check=check)
    for ref in reuse.get("corpora", {}).values():
        core.project_reference(ROOT, ref, check=check)
    for path, digest in reuse.get("consumed_sha256", {}).items():
        core.project_reference(ROOT, {"path": path, "sha256": digest}, check=check)


def verify_open(ledger: dict[str, Any], coverage: core.Coverage, *, check=lambda: None) -> dict[str, Any]:
    row = unique_operation(ledger, "open")
    root = row["output"].get("root")
    if root != str(BASE / "open") or row["manifest"].get("root", root) != root:
        raise ValueError("open root differs")
    binding = row["manifest"]["preparation_binding"]
    store = open_bound("open", Path(root), binding, coverage, check=check)
    authenticate_sources(binding, check=check)
    complete = expect_schema(store.json(row["output"]["complete"], check=check), "geometric-decision-open-prepared-v1", label="open completion")
    expect_binding(complete, binding, label="open completion")
    store.json(complete["scale"], check=check)
    if len(complete["entries"]) != 27:
        raise ValueError("open adaptation does not contain 27 entries")
    for entry in complete["entries"]:
        delivered = expect_schema(store.json(entry["index"], check=check), "geometric-decision-delivered-shard-v1", label="open delivered shard")
        expect_binding(delivered, binding, label="open delivered shard")
        if (delivered["split"] != entry["split"] or delivered["checkpoint_seed"] != entry["checkpoint_seed"]
                or delivered["shard"] != entry["shard"] or delivered["scale"] != complete["scale"]):
            raise ValueError("open delivered shard identity differs")
        store.arrays(delivered["arrays"], check=check)
    source = binding["source"]
    historical = open_bound("historical-open", Path(source["root"]), source["binding"], coverage, check=check)
    walk_historical_open(historical, source["prepared"], source["delivered"], source["normalizers"], check=check)
    return {"store": store, "complete_ref": row["output"]["complete"], "binding": binding}


def authenticate_owned_tree(store: core.AuditStore, *, check=lambda: None) -> dict[str, int]:
    """Authenticate every artifact in an exact single-purpose profile root."""
    counts = {"json": 0, "npz": 0, "opaque": 0}
    for path in sorted(store.root.rglob("*")):
        check()
        if path.is_symlink():
            raise ValueError("owned profile tree contains a symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(store.root).as_posix()
        ref = store.current_reference(relative, check=check)
        if path.suffix == ".json":
            store.json(ref, check=check); counts["json"] += 1
        elif path.suffix == ".npz":
            store.arrays(ref, check=check); counts["npz"] += 1
        elif path.suffix in (".gz", ".pt"):
            store.authenticate(ref, kind="profile-opaque", check=check); counts["opaque"] += 1
        else:
            raise ValueError("unknown artifact type in owned profile root")
    return counts


def profile_batch_closure(store: core.AuditStore, ref: dict[str, Any], binding: dict[str, Any],
                          *, check=lambda: None) -> None:
    batch = expect_schema(store.json(ref, check=check), "geometric-decision-profile-batch-v1",
                          label="profile input batch")
    if (batch["binding"] != binding or len(batch["rows"]) != 32
            or len({core.encoded(row["arrays"]) for row in batch["rows"]}) != 32):
        raise ValueError("profile input batch binding/roster differs")
    for row in batch["rows"]:
        store.authenticate(row["arrays"], kind="profile-input-arrays", check=check)


def profile_snapshot_closure(store: core.AuditStore, refs: list[dict[str, Any]],
                             *, check=lambda: None) -> None:
    seen = set()
    previous_ref, previous_steps = None, -1
    for ref in refs:
        identity = (ref["path"], ref["bytes"], ref["sha256"])
        if identity in seen:
            raise ValueError("profile snapshot roster contains duplicates")
        seen.add(identity)
        snapshot = expect_schema(store.json(ref, check=check), "geometric-decision-snapshot-v1",
                                 label="profile snapshot")
        if (snapshot["previous"] != previous_ref or type(snapshot["steps"]) is not int
                or snapshot["steps"] <= previous_steps):
            raise ValueError("profile snapshot parent chain differs")
        store.authenticate(snapshot["state"], kind="pt", check=check)
        previous_ref, previous_steps = ref, snapshot["steps"]


def closing_batch_closure(store: core.AuditStore, ref: dict[str, Any], *, check=lambda: None) -> None:
    batch = expect_schema(store.json(ref, check=check), "geometric-decision-batch-evaluation-v1",
                          label="closing profile metric batch")
    if (len(batch["targets"]) != len(batch["scene_ids"]) or len(batch["heads"]) != 144
            or len({core.encoded(ref) for ref in batch["heads"]}) != 144):
        raise ValueError("closing profile metric batch roster differs")
    for target_ref in batch["targets"]:
        target = store.json(target_ref, check=check)
        store.authenticate(target["arrays"], kind="profile-target-arrays", check=check)
    for head_ref in batch["heads"]:
        store.json(head_ref, check=check)
    store.json(batch["classical"], check=check)
    store.authenticate(batch["arrays"], kind="profile-metric-arrays", check=check)


def heterogeneous_profile_closure(operation: str, row: dict[str, Any], store: core.AuditStore,
                                  binding: dict[str, Any], coverage: core.Coverage,
                                  *, check=lambda: None) -> None:
    """Consume typed links rooted at each heterogeneous terminal result."""
    if operation == "profile-inputs":
        profile_batch_closure(store, row["output"]["result"], binding, check=check)
        load = store.json(row["output"]["checkpoint_load"], check=check)
        equality = store.json(row["output"]["subset_equality"], check=check)
        if (equality["batch"] != row["output"]["result"]
                or equality["checkpoint_load"] != row["output"]["checkpoint_load"]
                or load["checkpoint_seed"] != core.CHECKPOINTS[0]):
            raise ValueError("profile input terminal links differ")
        return
    if operation == "profile-head":
        outer = store.json(row["output"]["result"], check=check)
        expected = [(case, objective) for case in ("envelope", "first_train_batch")
                    for objective in ("mse", "decision")]
        if [(case["case"], case["objective"]) for case in outer["cases"]] != expected:
            raise ValueError("profile head case roster differs")
        for case in outer["cases"]:
            case_root = store.root / f"{case['case']}-{case['objective']}"
            if case["root"] != str(case_root):
                raise ValueError("profile head case root differs")
            case_binding = {**binding, "case": case["case"], "objective": case["objective"]}
            child = open_bound("profile-head-case", case_root, case_binding, coverage, check=check)
            result = child.json(case["result"], check=check)
            if result["binding"] != case_binding or result["objective"] != case["objective"]:
                raise ValueError("profile head result binding differs")
            profile_snapshot_closure(child, [result["initial"], result["middle"], result["last"]], check=check)
            for output in result["outputs"]:
                child.authenticate(output, kind="profile-head-arrays", check=check)
        return
    if operation == "profile-fitter":
        result = store.json(row["output"]["result"], check=check)
        if result["binding"] != binding or result["device"] != binding["runtime"]["device"]:
            raise ValueError("profile fitter result binding differs")
        for item in result["results"]:
            if item["status"] == "MEASURED":
                store.json(item["factors"], check=check)
            elif item["status"] != "NO_GROUPS_IN_BATCH":
                raise ValueError("profile fitter result status differs")
        return
    if operation == "profile-selection":
        result = expect_schema(store.json(row["output"]["profile"], check=check),
            "geometric-decision-selection-profile-v1", label="selection profile")
        if result["binding"] != binding or len(result["io"]) != 11 or len(result["snapshots"]) != 11:
            raise ValueError("selection profile binding/roster differs")
        for item in result["io"]:
            record = store.json(item["record"], check=check)
            store.authenticate(record["predictions"], kind="profile-selection-arrays", check=check)
        store.json(result["fixture_result"], check=check)
        profile_snapshot_closure(store, result["snapshots"], check=check)
        return
    raise ValueError("no heterogeneous profile closure for " + operation)


def verify_profiles(ledger: dict[str, Any], coverage: core.Coverage, *, check=lambda: None) -> dict[str, Any]:
    expected = {
        "profile-observed": (BASE / "profiles/observed-cuda-0", "geometric-decision-observed-profile-binding-v1",
                             "geometric-decision-observed-profile-v1"),
        "profile-closing": (BASE / "profiles/closing-cpu-0", "geometric-decision-closing-profile-binding-v1",
                            "geometric-decision-closing-profile-v1"),
    }
    result = {}
    for operation, (root, binding_schema, result_schema) in expected.items():
        row = unique_operation(ledger, operation)
        if row["manifest"].get("root") != str(root) or row["output"].get("root") != str(root):
            raise ValueError(operation + " root differs from declared profile root")
        binding = row["manifest"]["binding"]
        if binding.get("schema") != binding_schema:
            raise ValueError(operation + " binding schema differs")
        store = open_bound(operation, root, binding, coverage, check=check)
        authenticate_sources(binding, check=check)
        profile = expect_schema(store.json(row["output"]["result"], check=check), result_schema, label=operation)
        if profile["binding"] != binding:
            raise ValueError(operation + " result binding differs")
        if operation == "profile-observed" and (profile["new_test_observations"] != 0
                or binding["test_authority"] is not False):
            raise ValueError("observed profile acquired test authority")
        if operation == "profile-closing" and (profile["new_observations"] != 0
                or profile["new_test_access"] is not False or binding["new_test_authority"] is not False):
            raise ValueError("closing profile acquired test authority")
        if operation == "profile-observed":
            observed = expect_schema(store.json(profile["result"], check=check),
                "geometric-decision-observed-run-v1", label="observed profile payload")
            if profile["result"] != profile["recovery"] or observed["truth_access"] is not False:
                raise ValueError("observed profile recovery/result differs")
            store.json(profile["runtime"], check=check)
        else:
            admission = store.json(profile["admission"], check=check)
            if (profile["original"] != profile["original_replay"] or profile["probe"] != profile["probe_replay"]
                    or admission["binding"] != binding):
                raise ValueError("closing profile replay/admission differs")
            for field in ("original", "probe"):
                closing_batch_closure(store, profile[field], check=check)
            bootstrap = store.json(profile["bootstrap_fixture"], check=check)
            store.arrays(bootstrap["arrays"], check=check)
            if len(profile["known_reconstructions"]) != 16:
                raise ValueError("closing profile known reconstruction roster differs")
            for receipt_ref in profile["known_reconstructions"]:
                receipt = store.json(receipt_ref, check=check)
                for field in ("intent", "observation", "sidecar"):
                    store.json(receipt[field], check=check)
            inventory = store.json(profile["observable_inventory"], check=check)
            observed_root = BASE / "profiles/observed-cuda-0"
            observed_profile = core.AuditStore("profile-observed-inventory", observed_root, coverage)
            current_paths = sorted(path.relative_to(observed_root).as_posix()
                                   for path in observed_root.rglob("*") if path.is_file() and not path.is_symlink())
            if (inventory.get("root") != str(observed_root)
                    or [ref["path"] for ref in inventory["files"]] != current_paths):
                raise ValueError("closing profile observed inventory roster differs")
            for ref in inventory["files"]:
                observed_profile.authenticate(ref, kind="profile-inventory", check=check)
            store.json(profile["exclusions"], check=check)
        result[operation] = authenticate_owned_tree(store, check=check)
    profile_roots = {
        "profile-inputs": [BASE / "profiles/inputs"],
        "profile-head": [BASE / "profiles/head-cpu", BASE / "profiles/head-cuda-0"],
        "profile-fitter": [BASE / "profiles/fitter-cpu", BASE / "profiles/fitter-cuda-0"],
        "profile-selection": [BASE / "profiles/selection-cpu"],
    }
    present_profiles = {name for name in ledger["operations"] if name.startswith("profile-")}
    unexpected = present_profiles - set(expected) - set(profile_roots)
    if unexpected:
        raise ValueError("unknown profile operation roots: " + ",".join(sorted(unexpected)))
    # Heterogeneous profiles still have exact root rosters, and every terminal
    # output receipt is checked against the bytes named by the ledger output.
    for operation, roots in profile_roots.items():
        rows = ledger["operations"].get(operation, [])
        if len(rows) != len(roots):
            raise ValueError(operation + " profile roster differs")
        result[operation] = []
        by_root = {row["manifest"].get("root"): row for row in rows}
        if set(by_root) != {str(root) for root in roots}:
            raise ValueError(operation + " exact profile roots differ")
        for index, root in enumerate(roots):
            row = by_root[str(root)]
            if (not root.is_relative_to(BASE / "profiles") or root == BASE / "profiles"
                    or row["output"].get("root") != str(root)
                    or row["output"].get("manifest") != row["start"]["manifest"]):
                raise ValueError(operation + " profile store transition differs")
            binding = row["manifest"].get("binding")
            if not isinstance(binding, dict) or row["output"].get("binding") != binding_ref(binding):
                raise ValueError(operation + " profile binding transition differs")
            store = open_bound(f"{operation}/{index}", root, binding, coverage, check=check)
            authenticate_sources(binding, check=check)
            terminal_fields = (("profile",) if operation == "profile-selection"
                               else ("result", "checkpoint_load", "subset_equality"))
            consumed = 0
            for field in terminal_fields:
                ref = row["output"].get(field)
                if ref is not None:
                    store.authenticate(ref, kind="profile-terminal", check=check)
                    consumed += 1
            if consumed == 0:
                raise ValueError(operation + " lacks an authenticated terminal result")
            heterogeneous_profile_closure(operation, row, store, binding, coverage, check=check)
            result[operation].append(authenticate_owned_tree(store, check=check))
    return result


def verify_training(ledger: dict[str, Any], coverage: core.Coverage, *, check=lambda: None) -> dict[str, Any]:
    training_row = unique_operation(ledger, "training")
    if (training_row["manifest"].get("root") != str(BASE / "training")
            or training_row["output"].get("root") != str(BASE / "training")):
        raise ValueError("training root differs")
    binding = training_row["manifest"]["campaign_binding"]
    store = open_bound("training", Path(training_row["manifest"]["root"]), binding, coverage, check=check)
    authenticate_sources(binding, check=check)
    admission = binding["admission"]
    inputs_profiles = ledger["operations"].get("profile-inputs", [])
    head_profiles = ledger["operations"].get("profile-head", [])
    if (len(inputs_profiles) != 1 or len(head_profiles) != 2
            or admission["input_provenance"]["finish"] != inputs_profiles[0]["finish_ref"]
            or admission["input_provenance"]["output"] != inputs_profiles[0]["output"]
            or set(admission["profiles"]) != {"cpu", "cuda:0"}):
        raise ValueError("training profile admission roster differs")
    profiles_by_root = {row["output"]["root"]: row for row in head_profiles}
    for device in ("cpu", "cuda:0"):
        root = str(BASE / "profiles" / f"head-{device.replace(':', '-')}")
        evidence = admission["profiles"][device]
        profile_row = profiles_by_root.get(root)
        if (profile_row is None or evidence["finish"] != profile_row["finish_ref"]
                or evidence["output"] != profile_row["output"]):
            raise ValueError("training head profile receipt differs")
    complete_ref = training_row["output"]["complete"]
    complete = expect_schema(store.json(complete_ref, check=check), "geometric-decision-campaign-complete-v1", label="training completion")
    expect_binding(complete, binding, label="training completion")
    roster = [(cp, arm, seed) for cp in core.CHECKPOINTS for arm in core.ARMS for seed in core.READER_SEEDS]
    if len(complete["cells"]) != 72 or [(r["checkpoint_seed"], r["arm"], r["reader_seed"]) for r in complete["cells"]] != roster:
        raise ValueError("training roster differs from 72 declared cells")
    prepared = open_bound("open-from-training", BASE / "open", binding["preparation"], coverage, check=check)
    open_complete = prepared.json(binding["open_complete"], check=check)
    historical_identity = binding["preparation"]["source"]
    historical = open_bound("historical-open-from-training", Path(historical_identity["root"]),
                            historical_identity["binding"], coverage, check=check)
    cells = {}
    data_by_cp = {}
    for entry in complete["cells"]:
        check()
        cp, arm, seed = entry["checkpoint_seed"], entry["arm"], entry["reader_seed"]
        data = store.json(entry["data"], check=check)
        if cp not in data_by_cp:
            data_binding = data["binding"]
            if (data_binding.get("prepared_binding") != binding["preparation"]
                    or data_binding.get("completion") != binding["open_complete"]
                    or data_binding.get("scale") != open_complete["scale"]
                    or data_binding.get("checkpoint_seed") != cp
                    or len(data_binding.get("supervision", [])) != 9):
                raise ValueError("training aggregate data lineage differs")
            for source in data_binding["supervision"]:
                if source["split"] not in ("train", "calibration"):
                    raise ValueError("training supervision split differs")
                historical.arrays(source["targets"], check=check)
            data_by_cp[cp] = data
        elif data != data_by_cp[cp]:
            raise ValueError("training cells do not share checkpoint aggregate")
        relative = f"cells/cp_{cp}/{arm}/seed_{seed}"
        if entry["root"] != relative:
            raise ValueError("training cell root differs")
        cell_binding = {"schema": "geometric-decision-cell-binding-v1", "data": data["binding"],
                        "arm": arm, "checkpoint_seed": cp, "reader_seed": seed, "device": binding["device"]}
        cell = open_bound("training/" + relative, store.root / relative, cell_binding, coverage, check=check)
        closed = expect_schema(cell.json(entry["complete"], check=check), "geometric-decision-cell-complete-v1", label="cell completion")
        expect_binding(closed, cell_binding, label="cell completion")
        if closed["last_epoch"] != 50 or len(closed["history"]) != 50 or len(closed["calibration"]) != 11:
            raise ValueError("cell lacks final epoch/history/calibration roster")
        state_refs, previous = {}, None
        for epoch, cref in zip(range(0, 51, 5), closed["calibration"]):
            calibration = expect_schema(cell.json(cref, check=check), "geometric-decision-calibration-v1", label="calibration")
            expect_binding(calibration, cell_binding, label="calibration")
            if calibration["epoch"] != epoch:
                raise ValueError("calibration epoch roster differs")
            predictions = cell.arrays(calibration["predictions"], check=check)
            if (set(predictions) != {"components", "energy", "offsets"}
                    or predictions["components"].dtype != np.float64
                    or predictions["energy"].dtype != np.float64
                    or predictions["offsets"].dtype != np.int64
                    or predictions["components"].shape != (len(predictions["energy"]), 2)
                    or not np.array_equal(predictions["energy"], predictions["components"].sum(1, dtype=np.float64))
                    or predictions["offsets"].shape != (513,) or predictions["offsets"][0] != 0
                    or predictions["offsets"][-1] != len(predictions["energy"])
                    or np.any(np.diff(predictions["offsets"]) < 0)):
                raise ValueError("calibration arrays or signed energy arithmetic differ")
            sref = calibration["state"]
            state = cell.json(sref, check=check)
            if state.get("schema") != "geometric-decision-snapshot-v1" or state["epoch"] != epoch or state["next_batch"] != 0:
                raise ValueError("calibration snapshot boundary differs")
            cell.authenticate(state["state"], kind="pt", check=check)
            state_refs[epoch] = sref
        if closed["last_state"] != state_refs[50]:
            raise ValueError("cell final state differs from epoch50 calibration")
        # Authenticate the complete linked snapshot chain without unpickling.
        cursor, seen = closed["last_state"], set()
        while cursor is not None:
            key = (cursor["path"], cursor["sha256"])
            if key in seen:
                raise ValueError("snapshot parent cycle")
            seen.add(key)
            state = cell.json(cursor, check=check)
            cell.authenticate(state["state"], kind="pt", check=check)
            if previous is None:
                previous = state["steps"]
            elif state["steps"] >= previous:
                raise ValueError("snapshot parent does not precede child")
            previous, cursor = state["steps"], state["previous"]
        if not seen or state["steps"] != 0 or state["previous"] is not None:
            raise ValueError("snapshot chain does not terminate at initial state")
        cells[(cp, arm, seed)] = {"store": cell, "entry": entry, "closed": closed,
                                  "states": state_refs}
    return {"store": store, "binding": binding, "complete_ref": complete_ref,
            "complete": complete, "cells": cells}


def verify_selection(ledger: dict[str, Any], training: dict[str, Any], coverage: core.Coverage,
                     *, check=lambda: None) -> dict[str, Any]:
    row = unique_operation(ledger, "calibration-selection")
    if row["manifest"].get("root") != str(BASE / "selection") or row["output"].get("root") != str(BASE / "selection"):
        raise ValueError("selection root differs")
    binding = row["manifest"]["binding"]
    store = open_bound("selection", Path(row["manifest"]["root"]), binding, coverage, check=check)
    authenticate_sources(binding, check=check)
    profile_rows = ledger["operations"].get("profile-selection", [])
    if (len(profile_rows) != 1 or binding["admission"]["finish"] != profile_rows[0]["finish_ref"]
            or binding["admission"]["output"] != profile_rows[0]["output"]):
        raise ValueError("selection profile admission receipt differs")
    selected_ref = row["output"]["selection"]
    selected = store.json(selected_ref, check=check)
    expect_binding(selected, binding, label="selection")
    result = expect_schema(selected["result"], "geometric-decision-calibration-selection-v1", label="selection result")
    target_arrays = store.arrays(selected["targets"], check=check)
    if set(target_arrays) != {"targets", "offsets"} or target_arrays["targets"].dtype != np.float32:
        raise ValueError("selection target arrays differ")
    offsets = target_arrays["offsets"]
    if offsets.dtype != np.int64 or offsets.shape != (513,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("selection offsets differ")
    targets = target_arrays["targets"]
    eligible = np.flatnonzero(np.diff(offsets) > 0).tolist()
    if result["eligible_scene_ids"] != eligible:
        raise ValueError("selection eligible roster differs")
    provenance = {(r["checkpoint_seed"], r["arm"], r["reader_seed"], r["epoch"]): r
                  for r in selected["calibrations"]}
    expected = {(cp, arm, seed, epoch) for cp in core.CHECKPOINTS for arm in core.ARMS
                for seed in core.READER_SEEDS for epoch in range(5, 51, 5)}
    if len(selected["calibrations"]) != 720 or set(provenance) != expected:
        raise ValueError("selection calibration provenance differs from 720 rows")
    energies = {}
    for key, source in provenance.items():
        cp, arm, seed, epoch = key
        cell = training["cells"][(cp, arm, seed)]["store"]
        calibration = cell.json(source["calibration"], check=check)
        if calibration["state"] != source["state"] or calibration["epoch"] != epoch:
            raise ValueError("selection calibration/state link differs")
        arrays = cell.arrays(calibration["predictions"], check=check)
        if not np.array_equal(arrays["offsets"], offsets):
            raise ValueError("selection cell offsets differ")
        energies[key] = arrays["energy"]
    tD = targets.astype(np.float64).sum(1, dtype=np.float64)
    for arm in core.ARMS:
        rows = []
        for epoch in range(5, 51, 5):
            regrets = np.empty((9, len(eligible)), np.float64)
            for cell_id, (cp, seed) in enumerate((cp, seed) for cp in core.CHECKPOINTS for seed in core.READER_SEEDS):
                energy = energies[(cp, arm, seed, epoch)]
                for j, sid in enumerate(eligible):
                    a, b = offsets[sid:sid + 2]
                    choice = int(np.argmin(energy[a:b]))
                    regrets[cell_id, j] = tD[a + choice] - np.min(tD[a:b])
            scene = regrets.mean(0, dtype=np.float64)
            rows.append({"epoch": epoch, "mean_regret_tD": float(scene.mean(dtype=np.float64)),
                         "scene_mean_regret_tD": scene.tolist()})
        nested_equal(result["arms"][arm]["epochs"], rows, "independent selection " + arm)
        chosen = min(rows, key=lambda value: (value["mean_regret_tD"], value["epoch"]))["epoch"]
        if result["arms"][arm]["selected_epoch"] != chosen:
            raise ValueError("selected epoch differs from independent selector")
    return {"store": store, "binding": binding, "selected_ref": selected_ref,
            "selected": selected, "epochs": {arm: result["arms"][arm]["selected_epoch"] for arm in core.ARMS}}


def verify_archive(ledger: dict[str, Any], selection: dict[str, Any], training: dict[str, Any],
                   coverage: core.Coverage, *, check=lambda: None) -> dict[str, Any]:
    row = unique_operation(ledger, "archive-and-exclusions")
    if row["manifest"].get("root") != str(BASE / "archive") or row["output"].get("root") != str(BASE / "archive"):
        raise ValueError("archive root differs")
    binding = row["manifest"]["binding"]
    store = open_bound("archive", Path(row["manifest"]["root"]), binding, coverage, check=check)
    authenticate_sources(binding, check=check)
    complete = expect_schema(store.json(row["output"]["complete"], check=check), "geometric-decision-archive-complete-v1", label="archive completion")
    expect_binding(complete, binding, label="archive completion")
    if complete.get("fresh_tests") != "not opened" or complete.get("test_authority") is not False:
        raise ValueError("archive acquired fresh-test authority")
    exclusions = store.json(complete["exclusions"], check=check)
    catalog = store.json(complete["catalog"], check=check)
    if catalog.get("schema") != "geometric-decision-mechanical-exclusions-v1":
        raise ValueError("archive catalog schema differs")
    core.project_reference(ROOT, catalog["incident_receipt"], check=check)
    for entry in catalog["records"]:
        core.project_reference(ROOT, entry["source"], check=check)
    if exclusions.get("schema") != "generative-evidence-observed-exclusions-v1":
        raise ValueError("archive exclusion inventory schema differs")
    for path, digest in exclusions["consumed_sha256"].items():
        core.project_reference(ROOT, {"path": path, "sha256": digest}, check=check)
    extension = exclusions["extension"]
    for field in ("parent", "closed_campaign", "mechanical_catalog"):
        core.project_reference(ROOT, extension[field], check=check)
    heads = expect_schema(store.json(complete["heads"], check=check), "geometric-decision-head-archive-v1", label="head archive")
    expect_binding(heads, binding, label="head archive")
    expected = [(cp, arm, seed, stage) for cp in core.CHECKPOINTS for arm in core.ARMS
                for seed in core.READER_SEEDS for stage in core.STAGES]
    if len(heads["records"]) != 144:
        raise ValueError("archive lacks 144 heads")
    records = []
    shapes = {"group1.weight": (32, 9), "group1.bias": (32,), "group2.weight": (16, 32),
        "group2.bias": (16,), "partition1.weight": (32, 41), "partition1.bias": (32,),
        "partition2.weight": (2, 32), "partition2.bias": (2,)}
    for ref, identity in zip(heads["records"], expected):
        record = expect_schema(store.json(ref, check=check), "geometric-decision-frozen-head-v1", label="archived head")
        expect_binding(record, binding, label="archived head")
        cp, arm, seed, stage = identity
        if (record["checkpoint_seed"], record["arm"], record["reader_seed"], record["stage"]) != identity:
            raise ValueError("archived head order/identity differs")
        epoch = 0 if stage == "initial" else selection["epochs"][arm]
        if record["epoch"] != epoch:
            raise ValueError("archived selected epoch differs")
        arrays = store.arrays(record["arrays"], check=check)
        if set(arrays) != set(shapes) or any(v.dtype != np.float32 or v.shape != shapes[k] for k, v in arrays.items()):
            raise ValueError("archived head array schema differs")
        if stage == "initial" and (np.any(arrays["partition2.weight"]) or np.any(arrays["partition2.bias"])):
            raise ValueError("initial archived residual is not zero")
        cell = training["cells"][(cp, arm, seed)]["store"]
        entry = training["cells"][(cp, arm, seed)]["entry"]
        closed = training["cells"][(cp, arm, seed)]["closed"]
        source = record["source"]
        if (source["campaign"] != training["complete_ref"] or source["root"] != entry["root"]
                or source["cell_binding"] != cell.json(binding_ref(source["cell_binding"]), check=check)
                or source["complete"] != entry["complete"]
                or source["calibration"] != closed["calibration"][epoch // 5]):
            raise ValueError("archived head source lineage differs")
        cell.json(source["complete"], check=check)
        calibration = cell.json(source["calibration"], check=check)
        if calibration["epoch"] != epoch or calibration["state"] != source["state"]:
            raise ValueError("archived head calibration/state epoch differs")
        state = cell.json(record["source"]["state"], check=check)
        cell.authenticate(state["state"], kind="pt", check=check)
        if state["state_digest"] != record["source"]["state_digest"]:
            raise ValueError("archived source state digest differs")
        records.append(record)
    return {"store": store, "binding": binding, "complete": complete,
            "heads_ref": complete["heads"], "head_refs": heads["records"], "heads": records}


def load_evaluation(ledger: dict[str, Any], precommit: dict[str, Any], coverage: core.Coverage,
                    *, check=lambda: None) -> dict[str, Any]:
    evaluate, replay = unique_operation(ledger, "evaluate"), unique_operation(ledger, "replay")
    if any(row["output"].get("root") != str(BASE / "evaluation") or row["manifest"].get("root") != str(BASE / "evaluation")
           for row in (evaluate, replay)):
        raise ValueError("evaluation/replay root differs")
    if evaluate["output"]["complete"] != replay["output"]["complete"]:
        raise ValueError("evaluation replay changed completion ref")
    binding = {"test_freeze": precommit["freeze_ref"], "prediction_seal": precommit["seal_ref"]}
    store = open_bound("evaluation", Path(evaluate["output"]["root"]), binding, coverage, check=check)
    complete = expect_schema(store.json(evaluate["output"]["complete"], check=check), "geometric-decision-evaluation-complete-v1", label="evaluation completion")
    expect_binding(complete, binding, label="evaluation completion")
    if complete["original_scenes"] != 2048 or len(complete["results"]) != 4:
        raise ValueError("evaluation completion scope differs")
    seal = core.AuditStore("control", BASE / "control", coverage).json(precommit["seal_ref"], check=check)
    fresh = core.AuditStore("fresh", Path(seal["fresh_root"]), coverage)
    batches = {}
    for row, sealed_row, (split, _) in zip(complete["results"], seal["batches"], core.TESTS):
        if row["split"] != split:
            raise ValueError("evaluation split order differs")
        batch = expect_schema(store.json(row["original"], check=check), "geometric-decision-batch-evaluation-v1", label="evaluation batch")
        expect_binding(batch, binding, label="evaluation batch")
        if batch["scene_ids"] != list(range(512)) or len(batch["targets"]) != 512 or len(batch["heads"]) != 144:
            raise ValueError("evaluation batch roster differs")
        observed = fresh.json(sealed_row["observed"], check=check)
        source_index = fresh.json(observed["sources"], check=check)
        if batch["observed_sources"] != observed["sources"]:
            raise ValueError("evaluation sources differ from sealed observation")
        targets = []
        for sid, ref in enumerate(batch["targets"]):
            target = store.json(ref, check=check)
            arrays = store.arrays(target["arrays"], check=check)
            count, events = target["candidate_count"], target["event_count"]
            required = {"raw", "u64", "u32", "ari", "exact", "k_error", "tM", "tD", "labels"}
            if (target["scene_id"] != sid or target["source"] != source_index["sources"][sid]
                    or set(arrays) != required or arrays["labels"].dtype != np.int64
                    or arrays["labels"].shape != (events,) or arrays["u64"].shape != (count, 2)
                    or arrays["raw"].dtype != np.float64 or arrays["raw"].shape != (count, 2)
                    or arrays["ari"].dtype != np.float64 or arrays["ari"].shape != (count,)
                    or arrays["exact"].dtype != np.bool_ or arrays["exact"].shape != (count,)
                    or arrays["k_error"].dtype != np.int64 or arrays["k_error"].shape != (count,)):
                raise ValueError("evaluation target schema/identity differs")
            core.canonical_target_arithmetic(arrays)
            if not np.array_equal(arrays["u64"], arrays["raw"] / np.log(events)):
                raise ValueError("canonical normalized target arithmetic differs")
            targets.append((target, arrays))
        heads = [store.json(ref, check=check) for ref in batch["heads"]]
        expected_heads = [(cp, arm, seed, stage) for cp in core.CHECKPOINTS for arm in core.ARMS
                          for seed in core.READER_SEEDS for stage in core.STAGES]
        if any(head["identity"] != list(identity) or head["scene_ids"] != list(range(512))
               or head["prediction"] != observed["records"][i]["prediction"]
               or [scene["scene_id"] for scene in head["rows"]] != list(range(512))
               for i, (head, identity) in enumerate(zip(heads, expected_heads))):
            raise ValueError("evaluation head identity/scene closure differs")
        classical = store.json(batch["classical"], check=check)
        arrays = store.arrays(batch["arrays"], check=check)
        if (len(classical) != 512 or [item["scene_id"] for item in classical] != list(range(512))
                or set(arrays) != {"learned", "classical", "eligible", "scene_ids"}
                or arrays["learned"].shape != (512, 2, 8, 3, 3, 9)
                or arrays["classical"].shape != (512, 4, 9)
                or arrays["eligible"].dtype != np.bool_ or arrays["eligible"].shape != (512,)
                or not np.array_equal(arrays["scene_ids"], np.arange(512, dtype=np.int64))
                or not np.array_equal(arrays["eligible"], np.asarray([meta["candidate_count"] > 0 for meta, _ in targets]))):
            raise ValueError("evaluation aggregate arrays/eligibility differ")
        independent_summary = core.aggregate_summary(arrays["learned"], arrays["classical"], arrays["eligible"],
            [meta["planted_presence"] for meta, _ in targets])
        nested_equal(batch["summary"], independent_summary, "independent evaluation summary " + split)
        transformed = None
        if row["roundtrip"] is not None:
            transformed_batch = expect_schema(store.json(row["roundtrip"], check=check), "geometric-decision-batch-evaluation-v1", label="roundtrip evaluation")
            expect_binding(transformed_batch, binding, label="roundtrip evaluation")
            derived_observed = observed["roundtrip"]
            derived_sources = fresh.json(derived_observed["sources"], check=check)
            if (transformed_batch["scene_ids"] != observed["roundtrip_scene_ids"]
                    or transformed_batch["observed_sources"] != derived_observed["sources"]):
                raise ValueError("roundtrip evaluation source roster differs")
            transformed_targets = []
            for j, ref in enumerate(transformed_batch["targets"]):
                target = store.json(ref, check=check)
                values = store.arrays(target["arrays"], check=check)
                if target["scene_id"] != observed["roundtrip_scene_ids"][j] or target["source"] != derived_sources["sources"][j]:
                    raise ValueError("roundtrip target identity differs")
                core.canonical_target_arithmetic(values)
                if (set(values) != {"raw", "u64", "u32", "ari", "exact", "k_error", "tM", "tD", "labels"}
                        or values["labels"].dtype != np.int64
                        or not np.array_equal(values["u64"], values["raw"] / np.log(target["event_count"]))):
                    raise ValueError("roundtrip target schema/arithmetic differs")
                transformed_targets.append((target, values))
            transformed = {"batch": transformed_batch, "targets": transformed_targets,
                "heads": [store.json(ref, check=check) for ref in transformed_batch["heads"]],
                "classical": store.json(transformed_batch["classical"], check=check),
                "arrays": store.arrays(transformed_batch["arrays"], check=check)}
            if (len(transformed["heads"]) != 144 or len(transformed["classical"]) != len(transformed_targets)
                    or set(transformed["arrays"]) != {"learned", "classical", "eligible", "scene_ids"}
                    or transformed["arrays"]["learned"].shape != (len(transformed_targets), 2, 8, 3, 3, 9)
                    or transformed["arrays"]["classical"].shape != (len(transformed_targets), 4, 9)
                    or transformed["arrays"]["eligible"].dtype != np.bool_
                    or not np.array_equal(transformed["arrays"]["scene_ids"],
                                          np.asarray(observed["roundtrip_scene_ids"], np.int64))
                    or not np.array_equal(transformed["arrays"]["eligible"],
                        np.asarray([meta["candidate_count"] > 0 for meta, _ in transformed_targets]))):
                raise ValueError("roundtrip evaluation closure differs")
            transformed_summary = core.aggregate_summary(transformed["arrays"]["learned"],
                transformed["arrays"]["classical"], transformed["arrays"]["eligible"],
                [meta["planted_presence"] for meta, _ in transformed_targets])
            nested_equal(transformed_batch["summary"], transformed_summary,
                         "independent roundtrip evaluation summary " + split)
        batches[split] = {"row": row, "batch": batch, "targets": targets,
                          "heads": heads, "classical": classical, "arrays": arrays,
                          "roundtrip": transformed}
    primary_record = store.json(complete["primary"], check=check)
    primary_arrays = store.arrays(primary_record["arrays"], check=check)
    deformed = batches["deformed_family"]
    learned = deformed["arrays"]["learned"]
    eligible = deformed["arrays"]["eligible"]
    independent = core.primary(learned[:, 1, :, :, :, core.METRICS.index("regret_tM")], eligible, check=check)
    nested_equal(primary_record["summary"], independent["summary"], "primary summary")
    for key, expected in independent["arrays"].items():
        if key == "bootstrap_indices":
            core.assert_exact(primary_arrays[key], expected, "bootstrap indices")
        else:
            core.assert_close(primary_arrays[key], expected, "primary " + key)
    return {"store": store, "complete": complete, "complete_ref": evaluate["output"]["complete"],
            "batches": batches, "primary": primary_record}


def reconstruct_cut(precommit: dict[str, Any], ledger: dict[str, Any], evaluation: dict[str, Any],
                    coverage: core.Coverage, *, check=lambda: None) -> dict[str, Any]:
    """Independent targets/decisions for every structural scene and all sealed probes."""
    seal = core.AuditStore("control", BASE / "control", coverage).json(precommit["seal_ref"], check=check)
    fresh = core.AuditStore("fresh", Path(seal["fresh_root"]), coverage)
    # Lazy privileged import only after PRECOMMIT is re-authenticated.
    from src.atencion_armonica.generative_evidence_supervision import reconstruct_truth
    observations_checked = heads_checked = classics_checked = probes_checked = 0
    expected_probe_reports = {}
    for cut, sealed_batch in zip(precommit["cuts"], seal["batches"]):
        split = cut["split"]
        observed = fresh.json(sealed_batch["observed"], check=check)
        sources = fresh.json(observed["sources"], check=check)
        inputs = fresh.json(observed["inputs"], check=check)
        draws = fresh.json(sealed_batch["draws"], check=check)
        classical_index = fresh.json(observed["classical"], check=check)
        ev = evaluation["batches"][split]
        expected_probe_reports[split] = {"scene_ids": [], "coordinates": [], "heads": []}
        target_by_id = {row[0]["scene_id"]: row for row in ev["targets"]}
        for sid in cut["scene_ids"]:
            check()
            source_record = fresh.json(sources["sources"][sid], check=check)
            scene = source_record["scene"]
            draw = fresh.json(draws["records"][sid], check=check)
            observation = fresh.json(draw["observation"], check=check)
            if observation != scene["observation"]:
                raise ValueError("draw/source observation identity differs")
            sidecar = fresh.json(draw["sidecar"], check=check)
            labels = reconstruct_truth(observation, sidecar, split)["labels"]
            independent = core.independent_targets(scene["partitions"], labels)
            target_meta, target_arrays = target_by_id[sid]
            core.canonical_target_arithmetic(target_arrays)
            core.assert_exact(target_arrays["labels"], np.asarray(labels, np.int64), "cut reconstructed labels")
            core.assert_close(target_arrays["u64"], independent["u64"], "independent VI")
            core.assert_close(target_arrays["tM"], independent["tM"], "independent tM")
            core.assert_close(target_arrays["ari"], independent["ari"], "independent ARI")
            for key in ("exact", "k_error"):
                core.assert_exact(target_arrays[key], independent[key], "cut " + key)
            input_record = fresh.json(inputs["records"][sid], check=check)
            input_arrays = fresh.arrays(input_record["arrays"], check=check)
            classical_record = fresh.json(classical_index["records"][sid], check=check)
            independent_strata = core.observable_strata(scene["partitions"],
                input_arrays[f"raw/{core.CHECKPOINTS[0]}/available"],
                classical_record["upper_branches"]["extended"])
            if target_meta["strata"] != independent_strata:
                raise ValueError("target stratum definitions differ from observable sources")
            # Donor/mask/stratum integrity for all three observable backbones.
            for cp in core.CHECKPOINTS:
                diagnosis = target_meta["diagnostics"][str(cp)]
                donors = diagnosis["six_channel_sham"]["donors"]
                count = target_meta["candidate_count"]
                sham = diagnosis["six_channel_sham"]
                if (sorted(donors) != list(range(count)) or any(type(i) is not int for i in donors)):
                    raise ValueError("donors do not permute candidate support")
                evidence = input_arrays[f"inputs/{cp}/evidence"]
                core.validate_sham(scene["partitions"], evidence[:, :6], sham)
                z = evidence[:, 6]
                expected_mask = (z != z[np.asarray(donors, np.int64)]).tolist()
                if diagnosis["scalar_changed_mask"] != expected_mask:
                    raise ValueError("scalar donor mask differs")
                six_mask = np.any(evidence[:, :6] != evidence[np.asarray(donors, np.int64), :6], axis=1).tolist()
                if sham["changed_mask"] != six_mask:
                    raise ValueError("six-channel donor mask differs")
                covered = []
                for stratum in sham["strata"]:
                    ids = stratum["candidate_ids"]
                    if (not ids or stratum["donors"] != [donors[i] for i in ids]
                            or sorted(stratum["donors"]) != sorted(ids)):
                        raise ValueError("donor stratum membership differs")
                    covered.extend(ids)
                if sorted(covered) != list(range(count)) or len(covered) != len(set(covered)):
                    raise ValueError("donor strata do not partition support")
            planted = tuple(sorted(tuple(np.flatnonzero(labels == value).tolist()) for value in np.unique(labels)))
            origins = {candidate["origin"] for candidate in scene["inventory"]["candidates"]
                       if tuple(tuple(group) for group in candidate["partition"]) == planted}
            presence = "pool" if "pool" in origins else "neighbor" if "neighbor" in origins else "absent"
            if target_meta["planted_presence"] != presence:
                raise ValueError("planted presence differs")
            observations_checked += 1
        # Every one of 144 initial/selected predictions is checked on the cut.
        for position, (prediction_row, evaluated_head) in enumerate(zip(observed["records"], ev["heads"])):
            prediction = fresh.json(prediction_row["prediction"], check=check)
            arrays = fresh.arrays(prediction["arrays"], check=check)
            cp = prediction["checkpoint_seed"]
            for sid in cut["scene_ids"]:
                scene = fresh.json(sources["sources"][sid], check=check)["scene"]
                a, b = arrays["offsets"][sid:sid + 2]
                target = target_by_id[sid][1]
                expected = core.describe(scene["partitions"], target, arrays["energy"][a:b], arrays["components"][a:b])
                if prediction["choices"][sid] != expected["chosen"]:
                    raise ValueError("sealed original choice differs from canonical selector")
                actual = evaluated_head["rows"][sid]["full"]
                nested_equal(actual, expected, f"head {position} scene {sid}")
                for scheme, definitions in target_by_id[sid][0]["strata"].items():
                    if scheme == "full":
                        continue
                    stratified = core.stratum_metrics(scene["partitions"], target, arrays["energy"][a:b],
                                                      definitions, arrays["components"][a:b])
                    nested_equal(evaluated_head["rows"][sid]["strata"][scheme], stratified,
                                 f"head strata {position} {sid} {scheme}")
                heads_checked += 1
            if prediction_row["transport"] is not None:
                transport_index = fresh.json(prediction_row["transport"], check=check)
                if transport_index["scene_ids"] != cut["probe_scene_ids"]:
                    raise ValueError("transport probe roster differs")
                for tref, sid in zip(transport_index["records"], cut["probe_scene_ids"]):
                    transport = fresh.json(tref, check=check)
                    arrays_t = fresh.arrays(transport["arrays"], check=check)
                    if transport["scene_id"] != sid or transport["prediction"] != prediction_row["prediction"]:
                        raise ValueError("transport identity differs")
                    input_row = fresh.arrays(fresh.json(inputs["records"][sid], check=check)["arrays"], check=check)
                    routed = {key: input_row[f"inputs/{cp}/{key}"].copy()
                              for key in ("groups", "globals", "incidence", "evidence")}
                    route = prediction["arm"].rsplit("_", 1)[0]
                    if route == "local":
                        routed["evidence"][:] = 0
                    bypass = 1 if route == "geometric" else 0 if route == "decoupled" else None
                    scene = fresh.json(sources["sources"][sid], check=check)["scene"]
                    a, b = arrays["offsets"][sid:sid + 2]
                    independent_transport = core.transport_diagnostic(
                        arrays_t, scene["partitions"], routed, bypass, arrays["components"][a:b])
                    nested_equal(transport["diagnostic"], independent_transport, "transport diagnostic")
                    probes_checked += 1
        for sid in cut["scene_ids"]:
            record = fresh.json(classical_index["records"][sid], check=check)
            scores = fresh.arrays(record["arrays"], check=check)
            scene = fresh.json(sources["sources"][sid], check=check)["scene"]
            target = target_by_id[sid][1]
            actual = ev["classical"][sid]["values"]
            for name in core.CLASSICAL:
                expected = core.describe(scene["partitions"], target, scores[name])
                if record["choices"][name] != expected["chosen"]:
                    raise ValueError("sealed classical choice differs from canonical selector")
                nested_equal(actual[name]["full"], expected, f"classical {name} scene {sid}")
                for scheme, definitions in target_by_id[sid][0]["strata"].items():
                    if scheme != "full":
                        nested_equal(actual[name]["strata"][scheme],
                            core.stratum_metrics(scene["partitions"], target, scores[name], definitions),
                            f"classical strata {name} {sid} {scheme}")
                classics_checked += 1
        # Roundtrip topology/coordinate math for every sealed probe and head.
        if observed["roundtrip"] is not None:
            derived_sources = fresh.json(observed["roundtrip"]["sources"], check=check)
            derived_inputs = fresh.json(observed["roundtrip"]["inputs"], check=check)
            coordinates_expected = []
            for j, dref in enumerate(derived_sources["sources"]):
                derived = fresh.json(dref, check=check)
                coords = fresh.arrays(derived["coordinates"], check=check)
                coordinates_expected.append({"scene_id": cut["probe_scene_ids"][j],
                    "coordinate_source": derived["coordinates"],
                    "coordinates": core.coordinate_comparison(coords)})
            transformed = ev["roundtrip"]
            if transformed is None:
                raise ValueError("sealed roundtrip lacks evaluation")
            derived_target_by_id = {meta["scene_id"]: (meta, arrays)
                                    for meta, arrays in transformed["targets"]}
            derived_classical_index = fresh.json(observed["roundtrip"]["classical"], check=check)
            for j, sid in enumerate(cut["probe_scene_ids"]):
                original_scene = fresh.json(sources["sources"][sid], check=check)["scene"]
                derived_scene = fresh.json(derived_sources["sources"][j], check=check)["scene"]
                original_labels = target_by_id[sid][1]["labels"]
                moved_labels = core.labels_by_event(original_scene, derived_scene, original_labels)
                target = derived_target_by_id[sid][1]
                core.canonical_target_arithmetic(target)
                independent = core.independent_targets(derived_scene["partitions"], moved_labels)
                core.assert_exact(target["labels"], moved_labels, "derived labels")
                for key in ("u64", "tM", "ari"):
                    core.assert_close(target[key], independent[key], "derived independent " + key)
                for key in ("exact", "k_error"):
                    core.assert_exact(target[key], independent[key], "derived " + key)
                target_meta = derived_target_by_id[sid][0]
                input_arrays = fresh.arrays(fresh.json(derived_inputs["records"][j], check=check)["arrays"], check=check)
                classical_record = fresh.json(derived_classical_index["records"][j], check=check)
                independent_strata = core.observable_strata(derived_scene["partitions"],
                    input_arrays[f"raw/{core.CHECKPOINTS[0]}/available"],
                    classical_record["upper_branches"]["extended"])
                if target_meta["strata"] != independent_strata:
                    raise ValueError("derived target stratum definitions differ")
                for cp in core.CHECKPOINTS:
                    diagnosis = target_meta["diagnostics"][str(cp)]
                    evidence = input_arrays[f"inputs/{cp}/evidence"]
                    core.validate_sham(derived_scene["partitions"], evidence[:, :6],
                                       diagnosis["six_channel_sham"])
                    donors = np.asarray(diagnosis["six_channel_sham"]["donors"], np.int64)
                    if diagnosis["scalar_changed_mask"] != (evidence[:, 6] != evidence[donors, 6]).tolist():
                        raise ValueError("derived scalar donor mask differs")
                planted = tuple(sorted(tuple(np.flatnonzero(moved_labels == value).tolist())
                                       for value in np.unique(moved_labels)))
                origins = {candidate["origin"] for candidate in derived_scene["inventory"]["candidates"]
                           if tuple(tuple(group) for group in candidate["partition"]) == planted}
                presence = "pool" if "pool" in origins else "neighbor" if "neighbor" in origins else "absent"
                if target_meta["planted_presence"] != presence:
                    raise ValueError("derived planted presence differs")
            head_probe_rows = []
            for before_row, after_row in zip(observed["records"], observed["roundtrip"]["records"]):
                before = fresh.json(before_row["prediction"], check=check)
                after = fresh.json(after_row["prediction"], check=check)
                before_arrays, after_arrays = fresh.arrays(before["arrays"], check=check), fresh.arrays(after["arrays"], check=check)
                rows = []
                after_evaluated = transformed["heads"][len(head_probe_rows)]
                identity = [before[key] for key in ("checkpoint_seed", "arm", "reader_seed", "stage")]
                if (before_row["head"] != after_row["head"] or before["head"] != after["head"]
                        or [after[key] for key in ("checkpoint_seed", "arm", "reader_seed", "stage")] != identity
                        or after_evaluated["identity"] != identity):
                    raise ValueError("roundtrip head identity differs")
                transport_index = fresh.json(before_row["transport"], check=check)
                for j, sid in enumerate(cut["probe_scene_ids"]):
                    original_scene = fresh.json(sources["sources"][sid], check=check)["scene"]
                    derived_scene = fresh.json(derived_sources["sources"][j], check=check)["scene"]
                    a, b = before_arrays["offsets"][sid:sid + 2]
                    c, d = after_arrays["offsets"][j:j + 2]
                    before_input = fresh.arrays(fresh.json(inputs["records"][sid], check=check)["arrays"], check=check)
                    after_input = fresh.arrays(fresh.json(derived_inputs["records"][j], check=check)["arrays"], check=check)
                    comparison = core.roundtrip_comparison(original_scene, derived_scene,
                        energy_before=before_arrays["energy"][a:b], energy_after=after_arrays["energy"][c:d],
                        evidence_before=before_input[f"inputs/{before['checkpoint_seed']}/evidence"],
                        evidence_after=after_input[f"inputs/{after['checkpoint_seed']}/evidence"],
                        choice_before=before["choices"][sid], choice_after=after["choices"][j])
                    target = derived_target_by_id[sid][1]
                    expected_after = core.describe(derived_scene["partitions"], target,
                        after_arrays["energy"][c:d], after_arrays["components"][c:d])
                    if after["choices"][j] != expected_after["chosen"]:
                        raise ValueError("sealed derived choice differs from canonical selector")
                    nested_equal(after_evaluated["rows"][j]["full"],
                        expected_after,
                        "derived head decision")
                    for scheme, definitions in derived_target_by_id[sid][0]["strata"].items():
                        if scheme != "full":
                            nested_equal(after_evaluated["rows"][j]["strata"][scheme],
                                core.stratum_metrics(derived_scene["partitions"], target,
                                    after_arrays["energy"][c:d], definitions, after_arrays["components"][c:d]),
                                "derived head strata")
                    transport = fresh.json(transport_index["records"][j], check=check)
                    rows.append({"scene_id": sid, "transport_source": transport_index["records"][j],
                        "transport": transport["diagnostic"], "roundtrip": comparison})
                    probes_checked += 1
                head_probe_rows.append({"identity": {k: before[k] for k in ("checkpoint_seed", "reader_seed", "arm", "stage")},
                    "before_prediction": before_row["prediction"], "after_prediction": after_row["prediction"], "rows": rows})
            for j, sid in enumerate(cut["probe_scene_ids"]):
                scene = fresh.json(derived_sources["sources"][j], check=check)["scene"]
                target = derived_target_by_id[sid][1]
                record = fresh.json(derived_classical_index["records"][j], check=check)
                scores = fresh.arrays(record["arrays"], check=check)
                actual = transformed["classical"][j]["values"]
                for name in core.CLASSICAL:
                    expected = core.describe(scene["partitions"], target, scores[name])
                    if record["choices"][name] != expected["chosen"]:
                        raise ValueError("sealed derived classical choice differs from canonical selector")
                    nested_equal(actual[name]["full"], expected,
                                 "derived classical decision")
                    for scheme, definitions in derived_target_by_id[sid][0]["strata"].items():
                        if scheme != "full":
                            nested_equal(actual[name]["strata"][scheme],
                                core.stratum_metrics(scene["partitions"], target, scores[name], definitions),
                                "derived classical strata")
            expected_probe_reports[split] = {"scene_ids": cut["probe_scene_ids"],
                "coordinates": coordinates_expected, "heads": head_probe_rows}
    return {"cut_original_scenes": observations_checked, "head_scene_checks": heads_checked,
            "classical_scene_checks": classics_checked, "probe_checks": probes_checked,
            "expected_probe_reports": expected_probe_reports}


def verify_report(ledger: dict[str, Any], precommit: dict[str, Any], evaluation: dict[str, Any],
                  coverage: core.Coverage, expected_probes: dict[str, Any], *, check=lambda: None) -> dict[str, Any]:
    row = unique_operation(ledger, "post-replay-report")
    if row["manifest"].get("root") != str(BASE / "report") or row["output"].get("root") != str(BASE / "report"):
        raise ValueError("report root differs")
    binding = row["manifest"]["code"], row["manifest"]["protocol"]
    authenticate_sources({"code": binding[0], "protocol": binding[1]}, check=check)
    store = core.AuditStore("report", Path(row["output"]["root"]), coverage)
    actual_binding = store.json(row["output"]["binding"], check=check)
    if actual_binding != {"code": binding[0], "protocol": binding[1]}:
        raise ValueError("report binding differs")
    complete = expect_schema(store.json(precommit["report_result_ref"], check=check), "geometric-decision-report-v1", label="report completion")
    expect_binding(complete, actual_binding, label="report completion")
    nested_equal(complete["primary"], evaluation["primary"]["summary"], "report primary")
    expected_origins = {"fresh_finish": precommit["terminal_receipts"]["prospective-observables"],
        "evaluation_finish": precommit["terminal_receipts"]["evaluate"],
        "replay_finish": precommit["terminal_receipts"]["replay"],
        "evaluation_complete": evaluation["complete_ref"], "test_freeze": precommit["freeze_ref"],
        "prediction_seal": precommit["seal_ref"], "execution_contract": precommit["contract_ref"],
        "observable_replay_finish": precommit["terminal_receipts"]["observable-replay"]}
    if complete["origins"] != expected_origins or complete["primary_source"] != evaluation["complete"]["primary"]:
        raise ValueError("report origins/primary source differ")
    if len(complete["scenarios"]) != 4:
        raise ValueError("report scenario roster differs")
    files_checked = 1
    for scenario, (split, _) in zip(complete["scenarios"], core.TESTS):
        if scenario["split"] != split:
            raise ValueError("report scenario order differs")
        summary = store.json(scenario["summary"], check=check)
        if summary["source"] != evaluation["complete"]["results"][core.TESTS.index((split, dict(core.TESTS)[split]))]:
            raise ValueError("report scenario source differs")
        nested_equal(summary["summary"], evaluation["batches"][split]["batch"]["summary"], "report summary " + split)
        probes = store.json(summary["probe_report"], check=check)
        probe_ids = next(c["probe_scene_ids"] for c in precommit["cuts"] if c["split"] == split)
        if len(probes["heads"]) != (144 if probe_ids else 0) or probes["scene_ids"] != probe_ids:
            raise ValueError("report probe roster differs")
        nested_equal(probes, expected_probes[split], "report probes " + split)
        strata = store.json(summary["stratified_report"], check=check)
        if len(strata["heads"]) != 144 or set(strata["classical"]) != set(core.CLASSICAL):
            raise ValueError("report strata roster differs")
        source_batch = evaluation["batches"][split]
        targets = {meta["scene_id"]: meta for meta, _ in source_batch["targets"]}
        def rows_expected(rows):
            output = []
            for row in rows:
                target = targets[row["scene_id"]]
                support = {}
                for scheme, definitions in target["strata"].items():
                    if scheme == "full": continue
                    groups = row["strata"][scheme]
                    support[scheme] = {"strata": len(groups),
                        "candidate_denominator": sum(len(group["candidate_ids"]) for group in groups.values()),
                        "singleton_strata": sum(len(group["candidate_ids"]) == 1 for group in groups.values()),
                        "defined_strata": {name: sum(group["metrics"][name] is not None for group in groups.values()) for name in core.METRICS},
                        "undefined_strata": {name: sum(group["metrics"][name] is None for group in groups.values()) for name in core.METRICS}}
                output.append({"scene_id": row["scene_id"], "candidate_denominator": target["candidate_count"],
                    "eligible_scene": target["candidate_count"] > 0, "full": row["full"],
                    "strata": row["strata"], "support": support})
            return output
        for i, ref in enumerate(strata["heads"]):
            actual = store.json(ref, check=check)
            source = source_batch["heads"][i]
            nested_equal(actual, {"source": source_batch["batch"]["heads"][i], "identity": source["identity"],
                                  "rows": rows_expected(source["rows"])}, "report stratum head")
            files_checked += 1
        for name, ref in strata["classical"].items():
            actual = store.json(ref, check=check)
            rows = [{"scene_id": row["scene_id"], **row["values"][name]} for row in source_batch["classical"]]
            nested_equal(actual, {"source": source_batch["batch"]["classical"], "name": name,
                                  "rows": rows_expected(rows)}, "report stratum classical")
            files_checked += 1
        sham = summary["sham"]
        if sham.get("authority") != "descriptive within-scene correspondence change, not independent replications":
            raise ValueError("report sham authority differs")
        # Every published sham row is checked against the authenticated target diagnostics.
        expected_rows = {(meta["scene_id"], cp): meta["diagnostics"][cp]
                         for meta, _ in source_batch["targets"] for cp in map(str, core.CHECKPOINTS)}
        if len(sham["scene_checkpoint_rows"]) != 512 * 3:
            raise ValueError("report sham lacks scene/checkpoint closure")
        for item in sham["scene_checkpoint_rows"]:
            diagnosis = expected_rows[(item["scene_id"], item["checkpoint_seed"])]
            if item["diagnostics"] != diagnosis:
                raise ValueError("report sham diagnostic differs")
            first, second = diagnosis["scalar_changed_mask"], diagnosis["six_channel_sham"]["changed_mask"]
            expected_support = [{"sizes": group["sizes"], "candidate_denominator": len(group["candidate_ids"]),
                "singleton": len(group["candidate_ids"]) == 1,
                "scalar_changed": sum(first[i] for i in group["candidate_ids"]),
                "six_channels_changed": sum(second[i] for i in group["candidate_ids"])}
                for group in diagnosis["six_channel_sham"]["strata"]]
            if (item["candidate_denominator"] != len(first) or item["stratum_support"] != expected_support):
                raise ValueError("report sham support differs")
        aggregates = {}
        for cp in map(str, core.CHECKPOINTS):
            rows_cp = [item for item in sham["scene_checkpoint_rows"] if item["checkpoint_seed"] == cp]
            denominators = [item["candidate_denominator"] for item in rows_cp]
            scalar = [sum(item["diagnostics"]["scalar_changed_mask"]) for item in rows_cp]
            six = [sum(item["diagnostics"]["six_channel_sham"]["changed_mask"]) for item in rows_cp]
            eligible_ids = [i for i, n in enumerate(denominators) if n]
            aggregates[cp] = {"scenes": 512, "eligible_scenes": len(eligible_ids),
                "candidates": sum(denominators), "scalar_changed": sum(scalar), "six_channels_changed": sum(six),
                "scalar_scene_mean": (math.fsum(scalar[i] / denominators[i] for i in eligible_ids) / len(eligible_ids)) if eligible_ids else None,
                "six_channel_scene_mean": (math.fsum(six[i] / denominators[i] for i in eligible_ids) / len(eligible_ids)) if eligible_ids else None}
        nested_equal(sham["checkpoint_cells"], aggregates, "report sham aggregates")
        files_checked += 3
    return {"report_files_checked": files_checked, "scenarios": 4,
            "primary_source": complete["primary_source"]}


def verify_all(control_write: ArtifactStore, control_read: core.AuditStore, ledger: dict[str, Any], active_start: dict[str, Any],
               precommit_ref: dict[str, Any], audit_store: ArtifactStore, coverage: core.Coverage,
               *, check=lambda: None) -> dict[str, Any]:
    audit_read = core.AuditStore("audit", audit_store.root, coverage)
    precommit = audit_read.json(precommit_ref, check=check)
    if precommit.get("schema") != "geometric-decision-final-audit-precommit-v1":
        raise ValueError("audit precommit schema differs")
    if (precommit["binding"] != audit_store.binding or precommit["preflight"] != audit_store.binding["preflight"]
            or precommit["plan"] != source_reference(PLAN) or precommit["checker"] != sources()):
        raise ValueError("audit plan/checker changed after PRECOMMIT")
    current_ledger = read_ledger(control_read, control_write.binding, active_start=active_start, check=check)
    # During VERIFY the current audit start is the only unfinished last attempt.
    if current_ledger["attempts"][-1]["manifest"].get("operation") != "final-technical-audit":
        raise ValueError("current audit attempt is not ledger tail")
    preflight_row = unique_operation(current_ledger, "final-technical-audit-preflight")
    preflight_authority = precommit["preflight"]
    if (preflight_row["finish_ref"] != preflight_authority["finish"]
            or preflight_row["output"] != control_read.json(preflight_authority["output"], check=check)):
        raise ValueError("audit preflight receipt changed before VERIFY")
    preflight_store = open_bound("audit-preflight-verify", Path(preflight_row["output"]["root"]),
                                 preflight_row["manifest"]["binding"], coverage, check=check)
    preflight_store.json(preflight_authority["admission"], check=check)
    preflight_store.json(preflight_authority["projection"], check=check)
    for name, ref in precommit["terminal_receipts"].items():
        if unique_operation(current_ledger, name)["finish_ref"] != ref:
            raise ValueError("terminal receipt changed after PRECOMMIT")
    opened = verify_open(current_ledger, coverage, check=check)
    profiles = verify_profiles(current_ledger, coverage, check=check)
    training = verify_training(current_ledger, coverage, check=check)
    selection = verify_selection(current_ledger, training, coverage, check=check)
    archive = verify_archive(current_ledger, selection, training, coverage, check=check)
    if archive["head_refs"] != json.loads(json.dumps(control_read.json(precommit["freeze_ref"])["head_roster"])):
        raise ValueError("freeze head roster differs from authenticated archive")
    seal = control_read.json(precommit["seal_ref"], check=check)
    fresh_store = core.AuditStore("fresh", Path(seal["fresh_root"]), coverage)
    verify_seal_inventory(fresh_store, seal, check=check)
    fresh_closure = verify_fresh_reference_closure(fresh_store, seal, check=check)
    evaluation = load_evaluation(current_ledger, precommit, coverage, check=check)
    cut = reconstruct_cut(precommit, current_ledger, evaluation, coverage, check=check)
    report = verify_report(current_ledger, precommit, evaluation, coverage,
                           cut["expected_probe_reports"], check=check)
    cut = {k: v for k, v in cut.items() if k != "expected_probe_reports"}
    return {"schema": "geometric-decision-final-audit-result-v1", "status": "COMPLETE",
        "precommit": precommit_ref, "coverage": coverage.summary(), "open": {"complete": opened["complete_ref"]},
        "profiles": profiles,
        "training": {"cells": len(training["cells"]), "states": "initial/final and complete parent chains authenticated"},
        "selection": {"calibrations": 720, "epochs": selection["epochs"]},
        "archive": {"heads": len(archive["heads"])}, "evaluation": {"scenarios": 4, "original_scenes": 2048},
        "fresh_reference_closure": fresh_closure,
        "cut": cut, "report": report,
        "authority": "technical provenance/arithmetic audit; no architecture promotion or GO/NO-GO"}


def execution_reservation(ledger: dict[str, Any], preflight_seconds: float,
                          projected_seconds: float) -> float:
    if (type(preflight_seconds) not in (int, float) or not math.isfinite(preflight_seconds)
            or not 0 <= preflight_seconds <= PREFLIGHT_RESERVATION
            or type(projected_seconds) not in (int, float) or not math.isfinite(projected_seconds)
            or projected_seconds <= 0):
        raise ValueError("audit preflight/projection timing differs")
    remaining = min(BUDGET_STAGES["audit"] - ledger["charged"]["audit"],
                    LIMITS["total_seconds"] - sum(ledger["charged"].values()),
                    RESERVATION - preflight_seconds)
    reservation = min(RESERVATION, remaining)
    if reservation <= 0 or projected_seconds > reservation:
        raise BudgetExceeded("audit projection does not fit 1500s minus charged preflight/common ledger")
    return reservation


def main() -> None:
    if (Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))):
        raise ValueError("final audit requires project cwd, hidden CUDA and one-thread CPU environment")
    if source_reference(PLAN)["sha256"] != PLAN_SHA or source_reference(PROTOCOL)["sha256"] != PROTOCOL_SHA:
        raise ValueError("accepted plan or scientific protocol changed")
    coverage = core.Coverage()
    control_read = core.AuditStore("control", BASE / "control", coverage)
    binding = control_read.json(CONTROL_BINDING)
    if binding.get("schema") != "geometric-decision-control-v1":
        raise ValueError("control binding schema differs")
    control = ArtifactStore(control_read.root, binding=binding)
    with control.path("operator.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (control.path("manifests/final-technical-audit-preflight.json").exists()
                or control.path("manifests/final-technical-audit.json").exists()):
            raise ValueError("final audit already attempted; preserve receipts and do not retry")
        plan_ref, code = source_reference(PLAN), sources()
        preflight_binding = {"schema": "geometric-decision-final-audit-preflight-binding-v1",
            "plan": plan_ref, "code": code, "protocol": source_reference(PROTOCOL),
            "control_binding": CONTROL_BINDING, "maximum_combined_seconds": RESERVATION}
        preflight_manifest = control.publish_json("manifests/final-technical-audit-preflight.json", {
            "operation": "final-technical-audit-preflight", "root": str(PREFLIGHT_ROOT),
            "binding": preflight_binding, "reservation_seconds": PREFLIGHT_RESERVATION})
        preflight_budget = StageBudget(control, "audit", manifest_ref=preflight_manifest,
            reservation_seconds=PREFLIGHT_RESERVATION, started_at=LAUNCH_STARTED,
            prior_charges=binding["prior_charges"],
            output_roots=[Path(path) for path in binding["output_roots"]])
        preflight_store = None
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        def stop_preflight(signum, frame):
            if signum == signal.SIGALRM:
                raise BudgetExceeded("final audit preflight reservation exhausted")
            raise InterruptedError("final audit preflight interrupted; no retry")
        try:
            for sig in old:
                signal.signal(sig, stop_preflight)
            signal.setitimer(signal.ITIMER_REAL,
                max(0.001, PREFLIGHT_RESERVATION - (time.monotonic() - LAUNCH_STARTED)))
            preflight_store = ArtifactStore(PREFLIGHT_ROOT, binding=preflight_binding)
            ledger = read_ledger(control_read, binding, active_start=preflight_budget.start_ref,
                                 check=preflight_budget.check)
            metadata = admit_metadata(control_read, ledger, check=preflight_budget.check)
            historical_inventory = historical_reference_inventory(ledger, check=preflight_budget.check)
            metadata["historical_inventory"] = historical_inventory
            audit_measurement = measure_independent_kernels(check=preflight_budget.check)
            metadata["audit_measurement"] = audit_measurement
            projection = inventory_projection(ledger, control_read.json(metadata["seal_ref"]),
                metadata["projection_basis"], historical_inventory, audit_measurement,
                check=preflight_budget.check)
            metadata_ref = preflight_store.publish_json("admission.json", metadata)
            projection_ref = preflight_store.publish_json("projection.json", projection)
            preflight_output = control.publish_json("outputs/final-technical-audit-preflight.json", {
                "manifest": preflight_manifest, "root": str(PREFLIGHT_ROOT),
                "binding": preflight_store.reference(preflight_store.path("binding.json")),
                "admission": metadata_ref, "projection": projection_ref})
            preflight_finish = preflight_budget.finish("COMPLETE", completion=preflight_output)
        except BaseException as exc:
            if preflight_store is not None:
                try:
                    preflight_store.publish_json("discrepancy.json", {
                        "schema": "geometric-decision-final-audit-preflight-discrepancy-v1",
                        "exception_type": type(exc).__name__, "message": str(exc),
                        "authority": "bounded metadata-only preflight discrepancy"})
                except BaseException:
                    pass
            if not preflight_budget.closed:
                status = ("LIMIT_REACHED" if isinstance(exc, BudgetExceeded) else "PAUSED"
                          if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED")
                preflight_budget.finish(status)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            for sig, handler in old.items():
                signal.signal(sig, handler)

        final_started = time.monotonic()
        preflight_seconds = control.json(preflight_finish)["seconds"]
        ledger = read_ledger(control_read, binding)
        preflight_record = unique_operation(ledger, "final-technical-audit-preflight")
        if (preflight_record["finish_ref"] != preflight_finish
                or preflight_record["output"] != control.json(preflight_output)):
            raise ValueError("completed audit preflight authority differs")
        preflight_read = open_bound("audit-preflight", PREFLIGHT_ROOT, preflight_binding,
                                    coverage, check=lambda: None)
        authenticated_metadata = preflight_read.json(metadata_ref)
        authenticated_projection = preflight_read.json(projection_ref)
        if authenticated_metadata != metadata or authenticated_projection != projection:
            raise ValueError("completed audit preflight payload differs")
        metadata, projection = authenticated_metadata, authenticated_projection
        reservation = execution_reservation(ledger, preflight_seconds, projection["projected_seconds"])
        audit_binding = {"schema": "geometric-decision-final-audit-binding-v1", "plan": plan_ref,
            "code": code, "protocol": source_reference(PROTOCOL), "inputs": metadata["terminal_receipts"],
            "preflight": {"finish": preflight_finish, "output": preflight_output,
                          "admission": metadata_ref, "projection": projection_ref}}
        audit_store = ArtifactStore(AUDIT_ROOT, binding=audit_binding)
        manifest = control.publish_json("manifests/final-technical-audit.json", {
            "operation": "final-technical-audit", "root": str(AUDIT_ROOT), "binding": audit_binding,
            "preflight_finish": preflight_finish, "projection": projection_ref,
            "reservation_seconds": reservation})
        budget = StageBudget(control, "audit", manifest_ref=manifest, reservation_seconds=reservation,
            started_at=final_started, prior_charges=binding["prior_charges"],
            output_roots=[Path(path) for path in binding["output_roots"]])
        def stop(signum, frame):
            if signum == signal.SIGALRM:
                raise BudgetExceeded("final audit reservation exhausted")
            raise InterruptedError("final audit interrupted; no retry")
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        try:
            for sig in old:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, reservation - (time.monotonic() - final_started)))
            budget.check()
            metadata = precommit_inputs(control_read, ledger, coverage, metadata, check=budget.check)
            precommit = {"schema": "geometric-decision-final-audit-precommit-v1", "binding": audit_binding,
                "plan": plan_ref, "checker": code, "terminal_receipts": metadata["terminal_receipts"],
                "preflight": audit_binding["preflight"],
                "freeze_ref": metadata["freeze_ref"], "seal_ref": metadata["seal_ref"],
                "contract_ref": metadata["contract_ref"], "report_result_ref": metadata["report_result_ref"],
                "cuts": metadata["cuts"], "result_payloads_opened": False}
            precommit_ref = audit_store.publish_json("precommit.json", precommit)
            budget.check(force_resources=True)
            result = verify_all(control, control_read, ledger, budget.start_ref, precommit_ref, audit_store, coverage, check=budget.check)
            if sources() != code or source_reference(PLAN) != plan_ref:
                raise ValueError("checker or accepted plan changed during audit")
            result_ref = audit_store.publish_json("result.json", result)
            output = control.publish_json("outputs/final-technical-audit.json", {"manifest": manifest,
                "root": str(AUDIT_ROOT), "binding": audit_store.reference(audit_store.path("binding.json")),
                "precommit": precommit_ref, "result": result_ref})
            finish = budget.finish("COMPLETE", completion=output)
            print(json.dumps({"status": "FINAL_TECHNICAL_AUDIT_COMPLETE", "finish": finish,
                              "result": result_ref}), flush=True)
        except BaseException as exc:
            try:
                audit_store.publish_json("discrepancy.json", {"schema": "geometric-decision-final-audit-discrepancy-v1",
                    "exception_type": type(exc).__name__, "message": str(exc), "coverage": coverage.summary(),
                    "authority": "raw technical discrepancy; never a corrected scientific result"})
            except BaseException:
                pass
            if not budget.closed:
                status = "LIMIT_REACHED" if isinstance(exc, BudgetExceeded) else "PAUSED" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED"
                budget.finish(status)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            for sig, handler in old.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
