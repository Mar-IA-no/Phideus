"""Metadata boundary tests, not a substitute for campaign VERIFY.

Real AuditStore hashes exercise terminal binding checks. Scientific payloads
are deliberately absent; sentinels prove where each adapter stops.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from experiments.atencion_armonica import audit_geometric_decision as audit
from experiments.atencion_armonica import audit_geometric_decision_core as core
from experiments.atencion_armonica.test_geometric_decision_audit_operator import publish


class ReachedBoundary(Exception):
    pass


def metadata_case(base, name):
    """Use the producers' manifest/output shapes, including rootless OPEN."""
    operations = {}
    binding = {"fixture": "expected"}
    precommit = {"freeze_ref": {"fixture": "freeze"}, "seal_ref": {"fixture": "seal"}}
    configs = {
        "open": ("open", "preparation_binding", "geometric-decision-open-prepared-v1", "open completion"),
        "training": ("training", "campaign_binding", "geometric-decision-campaign-complete-v1", "training completion"),
        "calibration-selection": ("selection", "binding", None, "selection"),
        "archive-and-exclusions": ("archive", "binding", "geometric-decision-archive-complete-v1", "archive completion"),
        "evaluate": ("evaluation", None, "geometric-decision-evaluation-complete-v1", "evaluation completion"),
        "post-replay-report": ("report", None, "geometric-decision-report-v1", "report completion"),
    }
    folder, key, schema, label = configs[name]
    root = base / folder
    if name == "training":
        inputs = {"finish_ref": {"fixture": "inputs"}, "output": {"root": str(base / "profiles/inputs")}}
        heads = [{"finish_ref": {"fixture": device}, "output": {
            "root": str(base / "profiles" / f"head-{device.replace(':', '-')}")}}
            for device in ("cpu", "cuda:0")]
        operations.update({"profile-inputs": [inputs], "profile-head": heads})
        binding["admission"] = {"input_provenance": {"finish": inputs["finish_ref"], "output": inputs["output"]},
            "profiles": {device: {"finish": row["finish_ref"], "output": row["output"]}
                         for device, row in zip(("cpu", "cuda:0"), heads)}}
    elif name == "calibration-selection":
        profile = {"finish_ref": {"fixture": "profile"}, "output": {"fixture": "output"}}
        operations["profile-selection"] = [profile]
        binding["admission"] = {"finish": profile["finish_ref"], "output": profile["output"]}
    elif name == "evaluate":
        binding = {"test_freeze": precommit["freeze_ref"], "prediction_seal": precommit["seal_ref"]}
    elif name == "post-replay-report":
        binding = {"code": [], "protocol": {"fixture": "protocol"}}
    manifest = {} if name == "open" else {"root": str(root)}
    if key:
        manifest[key] = binding
    elif name == "post-replay-report":
        manifest.update(binding)
    record = {"binding": binding}
    if schema:
        record["schema"] = schema
        for fields in audit.REFERENCE_DISPATCH[schema].values():
            record.update({field: [] for field in fields})
    row = {"manifest": manifest, "output": {"root": str(root)}}
    operations[name] = [row]
    if name == "evaluate":
        operations["replay"] = [row]  # Same completion identity, as in the producer.
    return {"operations": operations}, precommit, root, binding, record, label


def invoke(name, ledger, precommit):
    coverage = core.Coverage()
    if name == "open":
        return audit.verify_open(ledger, coverage)
    if name == "training":
        return audit.verify_training(ledger, coverage)
    if name == "calibration-selection":
        return audit.verify_selection(ledger, {}, coverage)
    if name == "archive-and-exclusions":
        return audit.verify_archive(ledger, {}, {}, coverage)
    if name == "evaluate":
        return audit.load_evaluation(ledger, precommit, coverage)
    return audit.verify_report(ledger, precommit, {}, coverage, {})


OPERATIONS = ("open", "training", "calibration-selection", "archive-and-exclusions",
              "evaluate", "post-replay-report")


@pytest.mark.parametrize("name", OPERATIONS)
@pytest.mark.parametrize("mutation", ("valid", "wrong", "missing"))
def test_terminal_adapter_checks_real_hashed_internal_binding(tmp_path, monkeypatch, name, mutation):
    monkeypatch.setattr(audit, "BASE", tmp_path)
    monkeypatch.setattr(audit, "authenticate_sources", lambda *a, **k: None)
    ledger, precommit, root, binding, record, label = metadata_case(tmp_path, name)
    row = ledger["operations"][name][0]
    bref = publish(root, "binding.json", binding)
    record = deepcopy(record)
    if mutation == "wrong":
        record["binding"] = {"fixture": "another campaign"}
    elif mutation == "missing":
        del record["binding"]
    ref = publish(root, "complete.json", record)
    row["output"].update({"complete": ref, "selection": ref, "binding": bref})
    precommit["report_result_ref"] = ref
    checked = []
    original = audit.expect_binding

    def stop_after_binding(value, expected, *, label):
        checked.append(label)
        original(value, expected, label=label)
        raise ReachedBoundary(label)

    monkeypatch.setattr(audit, "expect_binding", stop_after_binding)
    if mutation == "valid":
        with pytest.raises(ReachedBoundary, match=label):
            invoke(name, ledger, precommit)
    else:
        with pytest.raises(ValueError, match=label + " internal binding differs"):
            invoke(name, ledger, precommit)
    assert checked == [label]


@pytest.mark.parametrize("name", OPERATIONS)
@pytest.mark.parametrize("field", ("manifest", "output"))
def test_terminal_roots_reject_inconsistent_or_wrong_location(tmp_path, monkeypatch, name, field):
    monkeypatch.setattr(audit, "BASE", tmp_path)
    ledger, precommit, *_ = metadata_case(tmp_path, name)
    ledger["operations"][name][0][field]["root"] = str(tmp_path / "foreign")
    with pytest.raises(ValueError, match="root differs"):
        invoke(name, ledger, precommit)


@pytest.mark.parametrize("manifest_root", ("absent", "matching", "null"))
def test_open_passes_resolved_output_root_to_store(tmp_path, monkeypatch, manifest_root):
    monkeypatch.setattr(audit, "BASE", tmp_path)
    ledger, _, root, binding, *_ = metadata_case(tmp_path, "open")
    manifest = ledger["operations"]["open"][0]["manifest"]
    if manifest_root != "absent":
        manifest["root"] = str(root) if manifest_root == "matching" else None
    seen = []

    def sentinel(label, path, actual_binding, coverage, **kwargs):
        seen.append((label, path, actual_binding))
        raise ReachedBoundary()

    monkeypatch.setattr(audit, "open_bound", sentinel)
    if manifest_root == "null":
        with pytest.raises(ValueError, match="open root differs"):
            audit.verify_open(ledger, core.Coverage())
        assert seen == []
    else:
        with pytest.raises(ReachedBoundary):
            audit.verify_open(ledger, core.Coverage())
        assert seen == [("open", Path(root), binding)]


@pytest.mark.parametrize("mutation", ("none", "string_order", "duplicate", "missing", "content", "symlink"))
def test_profile_inventory_keeps_producer_path_order_and_authenticates(tmp_path, mutation):
    child = publish(tmp_path, "a/child.json", {"fixture": "child"})
    sibling = publish(tmp_path, "a.json", {"fixture": "sibling"})
    refs = [child, sibling]  # Path sort, not lexicographic full relative string.
    assert [p.relative_to(tmp_path).as_posix() for p in sorted(tmp_path.rglob("*")) if p.is_file()] == [r["path"] for r in refs]
    assert sorted(r["path"] for r in refs) != [r["path"] for r in refs]
    if mutation == "string_order":
        refs = [sibling, child]
    elif mutation == "duplicate":
        refs = [child, child]
    elif mutation == "missing":
        refs = [child]
    elif mutation == "content":
        (tmp_path / "a.json").write_text("changed")
    elif mutation == "symlink":
        (tmp_path / "extra.json").symlink_to("a.json")
    inventory = {"root": str(tmp_path), "files": refs}
    if mutation == "none":
        audit.verify_profile_inventory(inventory, tmp_path, core.Coverage())
    else:
        with pytest.raises(ValueError):
            audit.verify_profile_inventory(inventory, tmp_path, core.Coverage())


@pytest.mark.parametrize("mutation", ("valid", "wrong", "missing"))
def test_open_delivered_child_binding_is_checked_before_payload(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(audit, "BASE", tmp_path)
    monkeypatch.setattr(audit, "authenticate_sources", lambda *a, **k: None)
    ledger, _, root, binding, complete, _ = metadata_case(tmp_path, "open")
    publish(root, "binding.json", binding)
    scale = publish(root, "scale.json", {"fixture": "scale"})
    child = {"schema": "geometric-decision-delivered-shard-v1", "binding": binding,
             "split": "train", "checkpoint_seed": core.CHECKPOINTS[0], "shard": 0,
             "scale": scale, "arrays": {"path": "never-opened.npz"}}
    if mutation == "wrong":
        child["binding"] = {"fixture": "another campaign"}
    elif mutation == "missing":
        del child["binding"]
    ref = publish(root, "delivered.json", child)
    complete.update(scale=scale, entries=[{"index": ref, "split": "train",
        "checkpoint_seed": core.CHECKPOINTS[0], "shard": 0}] * 27)
    ledger["operations"]["open"][0]["output"]["complete"] = publish(root, "complete.json", complete)
    seen = []

    def payload_sentinel(*args, **kwargs):
        seen.append("payload")
        raise ReachedBoundary()

    monkeypatch.setattr(core.AuditStore, "arrays", payload_sentinel)
    if mutation == "valid":
        with pytest.raises(ReachedBoundary):
            audit.verify_open(ledger, core.Coverage())
        assert seen == ["payload"]
    else:
        with pytest.raises(ValueError, match="open delivered shard internal binding differs"):
            audit.verify_open(ledger, core.Coverage())
        assert seen == []
