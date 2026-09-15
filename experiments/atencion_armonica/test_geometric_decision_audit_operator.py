"""Abstract control/guard fixtures; never opens a campaign store."""
from __future__ import annotations

import ast
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from experiments.atencion_armonica import audit_geometric_decision as audit
from experiments.atencion_armonica import audit_geometric_decision_core as core
from src.atencion_armonica.geometric_decision_budget import LIMITS


def publish(root: Path, relative: str, value):
    raw = core.encoded(value)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def publish_raw(root: Path, relative: str, raw: bytes):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def publish_arrays(root: Path, relative: str):
    stream = BytesIO(); np.savez(stream, empty=np.empty(0, np.float32))
    return publish_raw(root, relative, stream.getvalue())


def binding():
    return {"schema": "geometric-decision-control-v1", "limits": LIMITS, "prior_charges": [],
            "protocol": {"path": "protocol", "sha256": "0" * 64}, "output_roots": ["fixture"]}


def start_fixture(root: Path, *, finish=None):
    b = binding()
    manifest = publish(root, "manifest.json", {"operation": "fixture"})
    zero = {name: 0.0 for name in ("profile", "open", "training", "fresh", "evaluation", "audit")}
    start = publish(root, "attempts/0000/start.json", {"schema": "geometric-decision-attempt-v1",
        "binding": b, "manifest": manifest, "stage": "audit", "reservation_seconds": 7.0,
        "charged_before": zero})
    if finish is not None:
        publish(root, "attempts/0000/finish.json", {"schema": "geometric-decision-attempt-finish-v1",
            "start": start, "status": "FAILED", "seconds": finish,
            "charged_after": {**zero, "audit": finish}, "completion": None})
    return b


def test_ledger_charges_missing_finish_at_full_reservation_and_validates_finish(tmp_path):
    b = start_fixture(tmp_path)
    ledger = audit.read_ledger(core.AuditStore("control-fixture", tmp_path, core.Coverage()), b)
    assert ledger["charged"]["audit"] == 7.0
    # A completed failure charges measured seconds, not its reservation.
    other = tmp_path / "finished"; other.mkdir()
    b = start_fixture(other, finish=3.5)
    ledger = audit.read_ledger(core.AuditStore("control-fixture", other, core.Coverage()), b)
    assert ledger["charged"]["audit"] == 3.5


def test_ledger_rejects_noncontiguous_attempts_and_broken_output_parent(tmp_path):
    b = binding(); manifest = publish(tmp_path, "manifest.json", {"operation": "fixture"})
    zero = {name: 0.0 for name in ("profile", "open", "training", "fresh", "evaluation", "audit")}
    publish(tmp_path, "attempts/0001/start.json", {"schema": "geometric-decision-attempt-v1",
        "binding": b, "manifest": manifest, "stage": "audit", "reservation_seconds": 1., "charged_before": zero})
    with pytest.raises(ValueError, match="contiguous"):
        audit.read_ledger(core.AuditStore("control-fixture", tmp_path, core.Coverage()), b)


def test_operator_top_level_imports_have_no_model_truth_fitter_or_cuda_capability():
    tree = ast.parse(Path(audit.__file__).read_text())
    modules = []
    for node in tree.body:
        if isinstance(node, ast.Import): modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): modules.append(node.module or "")
    forbidden = ("geometric_decision_metrics", "geometric_decision_model", "geometric_decision_inference",
                 "generative_evidence_supervision", "torch", "cuda")
    assert not any(token in module for module in modules for token in forbidden)
    # The one semantic parser is a late import scoped inside VERIFY reconstruction.
    source = Path(audit.__file__).read_text()
    assert source.count("from src.atencion_armonica.generative_evidence_supervision import reconstruct_truth") == 1
    assert source.index("def reconstruct_cut") < source.index("generative_evidence_supervision import reconstruct_truth")


def test_main_preserves_launch_lock_no_retry_precommit_then_verify_barriers():
    source = Path(audit.__file__).read_text()
    assert source.index("LAUNCH_STARTED = time.monotonic()") < source.index("import numpy as np")
    assert source.index("fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)") < source.index(
        'if (control.path("manifests/final-technical-audit-preflight.json").exists()')
    assert source.index("budget = StageBudget(") < source.index(
        "metadata = precommit_inputs(control_read")
    assert source.count("= StageBudget(") == 2
    assert source.index('preflight_budget.finish("COMPLETE"') < source.index(
        "final_started = time.monotonic()")
    assert source.index('audit_store.publish_json("precommit.json"') < source.index(
        "result = verify_all(")
    assert audit.RESERVATION == 1500.0
    assert audit.PREFLIGHT_RESERVATION <= 60.0
    assert "RESERVATION - preflight_seconds" in source


def test_nested_comparison_never_uses_tolerance_for_choices_or_ties():
    with pytest.raises(ValueError): audit.nested_equal({"chosen": 1}, {"chosen": 2}, "choice")
    with pytest.raises(ValueError): audit.nested_equal({"optima": [0]}, {"optima": [0, 1]}, "ties")
    audit.nested_equal({"value": 1.0}, {"value": 1.0 + 5e-13}, "derived")
    with pytest.raises(ValueError): audit.nested_equal({"value": 1.0}, {"value": 1.0 + 2e-12}, "derived")


def test_projection_uses_measured_basis_explicit_roots_and_independent_kernel_margin(tmp_path, monkeypatch):
    roots = {}
    for name in ("open", "training", "calibration-selection", "archive-and-exclusions",
                 "profile-observed", "profile-closing", "evaluate", "post-replay-report"):
        root = tmp_path / name; root.mkdir(); (root / "x").write_bytes(b"x")
        roots[name] = [{"manifest": {"root": str(root)}, "output": {"root": str(root)}}]
    roots["open"][0]["manifest"]["preparation_binding"] = {"source": {}}
    monkeypatch.setattr(audit, "BASE", tmp_path)
    historical = {"schema": "geometric-decision-historical-reference-inventory-v1",
        "files": 2, "bytes": 30, "external_declared_files": 1,
        "external_declared_bytes": 7, "records": [{}, {}]}
    measured = {"schema": "geometric-decision-independent-kernel-profile-v1",
        "candidate_units": 82, "scene_units": 20, "head_units": 144,
        "scene_head_seconds": .4, "bootstrap_scenes": 512, "bootstrap_seconds": .08}
    projection = audit.inventory_projection({"operations": roots}, {"files": [
        {"path": "x", "bytes": 1, "sha256": "0" * 64}], "fresh_root": str(tmp_path / "open")},
        {"timings": {"observable-inventory": .01, "original-metrics": .2,
                     "probe-metrics": .1, "bootstrap512": .05},
         "inventory_bytes": 100, "overhead_seconds": .1, "profile_result": {}, "observable_inventory": {}},
        historical, measured)
    assert 0 < projection["projected_seconds"] < audit.RESERVATION
    assert projection["margin"] == 1.25
    assert projection["cut_compute_seconds"] == max(
        projection["producer_scaled_seconds"], projection["independent_scaled_seconds"])
    assert projection["historical_declared_files"] == 3
    assert projection["historical_declared_bytes"] == 37
    assert "measured" in projection["scope"]


def test_execution_reservation_enforces_preflight_plus_run_1500_cap():
    charged = {name: 0.0 for name in ("profile", "open", "training", "fresh", "evaluation", "audit")}
    ledger = {"charged": charged}
    assert audit.execution_reservation(ledger, 50.0, 1400.0) == 1450.0
    with pytest.raises(audit.BudgetExceeded, match="1500s"):
        audit.execution_reservation(ledger, 50.0, 1450.0001)
    with pytest.raises(ValueError, match="preflight"):
        audit.execution_reservation(ledger, 60.0001, 1.0)


def test_independent_projection_fixture_is_measured_without_campaign_inputs():
    measured = audit.measure_independent_kernels(scene_units=1, head_units=1, bootstrap_scenes=2)
    assert measured["candidate_units"] == 82
    assert measured["scene_head_seconds"] > 0 and measured["bootstrap_seconds"] > 0
    assert measured["authority"].startswith("abstract CPU timing fixture")


def test_fresh_closure_walks_positive_schema_topology_and_authenticates_source_arrays(tmp_path):
    binding = {"test_freeze": {"path": "freeze.json", "bytes": 1, "sha256": "0" * 64}}
    binding_ref = publish(tmp_path, "binding.json", binding)
    arrays = publish_arrays(tmp_path, "shared/arrays.npz")
    artifact = publish_raw(tmp_path, "shared/factors.json.gz", b"abstract-gzip-bytes")
    sidecar = publish_raw(tmp_path, "shared/sidecar.json", b"opaque-sidecar-bytes")

    source_refs, input_refs, classical_refs, draw_refs = [], [], [], []
    for scene_id in range(512):
        source = publish(tmp_path, f"sources/{scene_id}.json", {
            "schema": "geometric-decision-source-v1", "binding": binding,
            "scene": {"observation": {"scene_id": scene_id}}, "arrays": arrays, "coordinates": None})
        fit = publish(tmp_path, f"fits/{scene_id}.json", {
            "schema": "geometric-decision-fit-v1", "binding": binding, "source": source,
            "artifact": {**artifact, "decoded_sha256": "1" * 64, "decoded_bytes": 1,
                         "codec": "canonical-json-gzip3-mtime0"}})
        inputs = publish(tmp_path, f"inputs/{scene_id}.json", {
            "schema": "geometric-decision-inputs-v1", "binding": binding,
            "source": source, "fit": fit, "arrays": arrays})
        classical = publish(tmp_path, f"classical/{scene_id}.json", {
            "schema": "geometric-decision-classical-v1", "binding": binding,
            "inputs": inputs, "source": source, "fit": fit, "arrays": arrays})
        observation = publish(tmp_path, f"draws/{scene_id}/observation.json", {"scene_id": scene_id})
        draw = publish(tmp_path, f"draws/{scene_id}/draw.json", {
            "schema": "geometric-decision-draw-v1", "binding": binding,
            "observation": observation, "sidecar": sidecar})
        source_refs.append(source); input_refs.append(inputs); classical_refs.append(classical); draw_refs.append(draw)

    sources = publish(tmp_path, "sources/index.json", {
        "schema": "geometric-decision-source-batch-v1", "binding": binding,
        "scene_ids": list(range(512)), "sources": source_refs})
    inputs = publish(tmp_path, "inputs/index.json", {
        "schema": "geometric-decision-input-batch-v1", "binding": binding,
        "scene_ids": list(range(512)), "sources": sources, "records": input_refs})
    classical = publish(tmp_path, "classical/index.json", {
        "schema": "geometric-decision-classical-index-v1", "binding": binding,
        "scene_ids": list(range(512)), "inputs": inputs, "records": classical_refs})
    prediction_records = []
    for i in range(144):
        head = {"path": f"heads/{i}.json", "bytes": 1,
                "sha256": hashlib.sha256(str(i).encode()).hexdigest()}
        prediction = publish(tmp_path, f"predictions/{i}.json", {
            "schema": "geometric-decision-prediction-v1", "binding": binding,
            "inputs": inputs, "head": head, "arrays": arrays})
        prediction_records.append({"head": head, "prediction": prediction, "transport": None})
    observed = publish(tmp_path, "observed.json", {
        "schema": "geometric-decision-observed-run-v1", "binding": binding,
        "sources": sources, "inputs": inputs, "records": prediction_records,
        "classical": classical, "roundtrip_scene_ids": [], "roundtrip": None})
    draws = publish(tmp_path, "draws/index.json", {
        "schema": "geometric-decision-draw-index-v1", "binding": binding,
        "scene_ids": list(range(512)), "records": draw_refs})
    seal = {"fresh_binding": binding_ref, "batches": [{"draws": draws, "observed": observed}]}
    coverage = core.Coverage(); store = core.AuditStore("fresh-fixture", tmp_path, coverage)
    assert audit.verify_fresh_reference_closure(store, seal) == {
        "sources": 512, "draws": 512, "inputs": 512, "predictions": 144,
        "transports": 0, "classical": 512, "sidecars_opaque": 512}
    (tmp_path / arrays["path"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        audit.verify_fresh_reference_closure(store, seal)


def test_profile_terminal_closures_follow_named_internal_receipts(tmp_path):
    input_root = tmp_path / "inputs-profile"; input_root.mkdir()
    binding = {"purpose": "abstract-profile"}
    publish(input_root, "binding.json", binding)
    rows = []
    for i in range(32):
        rows.append({"scene_id": i, "arrays": publish_arrays(input_root, f"rows/{i}.npz")})
    batch_ref = publish(input_root, "batch.json", {
        "schema": "geometric-decision-profile-batch-v1", "binding": binding, "rows": rows})
    input_store = core.AuditStore("profile-input-fixture", input_root, core.Coverage())
    audit.profile_batch_closure(input_store, batch_ref, binding)
    (input_root / rows[0]["arrays"]["path"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        audit.profile_batch_closure(input_store, batch_ref, binding)

    closing_root = tmp_path / "closing-profile"; closing_root.mkdir()
    closing_binding = {"purpose": "abstract-closing"}
    publish(closing_root, "binding.json", closing_binding)
    targets = []
    for i in range(2):
        target_arrays = publish_arrays(closing_root, f"targets/{i}.npz")
        targets.append(publish(closing_root, f"targets/{i}.json", {"arrays": target_arrays}))
    heads = [publish(closing_root, f"heads/{i}.json", {"identity": i}) for i in range(144)]
    classical = publish(closing_root, "classical.json", [{"scene_id": 0}, {"scene_id": 1}])
    metric_arrays = publish_arrays(closing_root, "metrics.npz")
    complete = publish(closing_root, "complete.json", {
        "schema": "geometric-decision-batch-evaluation-v1", "binding": closing_binding,
        "observed_sources": {"path": "external", "bytes": 1, "sha256": "0" * 64},
        "scene_ids": [0, 1], "targets": targets, "heads": heads,
        "classical": classical, "arrays": metric_arrays, "summary": {}})
    closing_store = core.AuditStore("closing-fixture", closing_root, core.Coverage())
    audit.closing_batch_closure(closing_store, complete)
    (closing_root / target_arrays["path"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        audit.closing_batch_closure(closing_store, complete)


def test_historical_projection_walks_transitive_metadata_but_defers_payload_hashes(tmp_path):
    root = tmp_path / "historical"; root.mkdir()
    historical_binding = {"purpose": "abstract-historical"}
    publish(root, "binding.json", historical_binding)
    payload = publish_raw(root, "payload.bin", b"opaque-payload")
    artifact = publish_raw(root, "fit.gz", b"opaque-gzip")
    source_ref = audit.source_reference(audit.PLAN)
    fit = publish(root, "fit.json", {"schema": "generative-evidence-fit-v1",
        "artifact": {**artifact, "decoded_sha256": "1" * 64, "decoded_bytes": 1,
                     "codec": "canonical-json-gzip3-mtime0"}})
    observable = publish(root, "observable.json", {"schema": "generative-evidence-observable-v1",
        "fit": fit, "raw": {"x": payload}})
    supervision = publish(root, "supervision.json", {"schema": "generative-evidence-supervision-v1",
        "observable": observable, "targets": payload, "metrics": payload, "source": {}})
    shard = publish(root, "shard.json", {"schema": "generative-evidence-prepared-shard-v1",
        "raw": {"x": payload}, "targets": payload, "metrics": payload,
        "records": [{"fit": fit, "observable": observable, "supervision": supervision}]})
    split = publish(root, "split.json", {"schema": "generative-evidence-prepared-split-v1",
        "split": "train", "shards": [shard]})
    normalizers = publish(root, "normalizers.json", {"prepared_train": split})
    prepared = publish(root, "prepared.json", {"schema": "generative-evidence-open-preparation-v1",
        "normalizers": normalizers, "splits": {"train": split},
        "reuse": {"authorization": source_ref, "import": source_ref, "corpora": {}, "consumed_sha256": {}}})
    index = publish(root, "delivered-index.json", {"schema": "generative-evidence-delivered-index-v1",
        "prepared": prepared, "normalizers": normalizers, "split": "train", "checkpoint_seed": 1,
        "shard": 0, "prepared_shard": shard, "raw": payload, "inputs": payload})
    delivered = publish(root, "delivered.json", {"schema": "generative-evidence-delivered-open-v1",
        "prepared": prepared, "normalizers": normalizers,
        "entries": [{"split": "train", "checkpoint_seed": 1, "shard": 0, "index": index}]})
    source = {"root": str(root), "binding": historical_binding, "prepared": prepared,
              "delivered": delivered, "normalizers": normalizers}
    ledger = {"operations": {"open": [{"manifest": {"preparation_binding": {"source": source}}}]}}
    inventory = audit.historical_reference_inventory(ledger)
    assert inventory["metadata_authenticated_files"] >= 8
    assert inventory["payload_hash_deferred_files"] >= 2
    assert inventory["external_declared_files"] == 1
    (root / payload["path"]).write_bytes(b"changed-but-still-deferred")
    assert audit.historical_reference_inventory(ledger)["payload_hash_deferred_files"] >= 2
    (root / split["path"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        audit.historical_reference_inventory(ledger)
