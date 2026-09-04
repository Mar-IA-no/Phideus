from __future__ import annotations

import builtins
import copy
import itertools
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments/geometria_proporcional"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import _wave57_phase_worker as worker  # noqa: E402
import prepare_wave56_fresh as preparer  # noqa: E402
import run_wave56_contextual_gate as coordinator  # noqa: E402
from geometria_proporcional.wave52_policy import constrained_regret  # noqa: E402
from geometria_proporcional.wave56_contextual_gate import disagreement_weights  # noqa: E402
from geometria_proporcional.wave57_tail_guard import validate_wave57_frozen_config  # noqa: E402


CONFIG_PATH = EXPERIMENTS / "configs/wave57_contextual_tail_guard_fresh.json"
FAILED_ORIGIN = (
    REPO_ROOT
    / "data/geometria_proporcional/"
    "wave57_contextual_tail_guard_fresh_v1.failed_20260904T215941566314Z"
)
SYNTHETIC_PREPARER = "prep.py"
SYNTHETIC_TEST = "test.py"
SYNTHETIC_PLAN = "plan.md"
SYNTHETIC_AUDIT_DIR = "audits"
SYNTHETIC_PLAN_AUDIT = "audits/plan_audit.md"
SYNTHETIC_IMPLEMENTATION_AUDIT = "audits/implementation_audit.md"
SYNTHETIC_AMENDMENT = "amendment.json"
SYNTHETIC_FINAL_AUDIT = "audits/final_audit.md"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def real_content_blind_case() -> tuple[dict, dict]:
    freeze = json.loads(
        (FAILED_ORIGIN / preparer.FREEZE_NAME).read_text(encoding="utf-8")
    )
    inventory = preparer.physical_tree_inventory(FAILED_ORIGIN)
    by_path = {record["path"]: record for record in inventory}
    execution_contract = copy.deepcopy(freeze["contract"])
    execution_contract["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    old_sources = freeze["contract"]["sources"]
    new_hashes = {
        preparer.PREPARER_RELATIVE: preparer.sha256_file(
            REPO_ROOT / preparer.PREPARER_RELATIVE
        ),
        preparer.WAVE57_RECOVERY_TEST_RELATIVE: preparer.sha256_file(
            REPO_ROOT / preparer.WAVE57_RECOVERY_TEST_RELATIVE
        ),
    }
    execution_contract["sources"].update(new_hashes)
    amendment = {
        "schema_version": preparer.WAVE57_RECOVERY_AMENDMENT_SCHEMA,
        "implementation": {
            label: {
                "path": relative,
                "old_sha256": old_sources[relative],
                "new_sha256": new_hashes[relative],
            }
            for label, relative in (
                ("preparer", preparer.PREPARER_RELATIVE),
                ("test", preparer.WAVE57_RECOVERY_TEST_RELATIVE),
            )
        },
        "escrow_origin": {
            "failed_attempt_basename": FAILED_ORIGIN.name,
            "contract_git_commit": freeze["contract"]["git_commit"],
            "contract_sha256": preparer.compact_json_sha256(freeze["contract"]),
            "escrow_sha256": by_path[preparer.ESCROW_NAME]["sha256"],
            "pre_generation_freeze_sha256": by_path[preparer.FREEZE_NAME]["sha256"],
            "failure_sha256": by_path["FAILURE.json"]["sha256"],
            "benchmark_manifest_sha256": by_path["benchmark/manifest.json"]["sha256"],
            "inventory": inventory,
        },
    }
    return amendment, execution_contract


def synthetic_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def synthetic_commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return synthetic_git(repo, "rev-parse", "HEAD")


def write_canonical_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode()
    path.write_bytes(encoded)
    return preparer.sha256_bytes(encoded)


def write_audit_report(
    path: Path, title: str, fields: list[str], result: str
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"# {title}\n\n"
        + "\n".join(fields)
        + "\n\n## Evidence\n\nSynthetic authority fixture.\n\n"
        + "## Machine-verifiable decision\n\n"
        + f"**Final decision:** `{result}`\n"
    )
    path.write_text(text, encoding="utf-8")
    return preparer.sha256_file(path)


def build_wave57_authority_repo(
    tmp_path: Path,
    *,
    amendment_mutator=None,
    mixed_commit: str | None = None,
    break_direct_parent: bool = False,
    plan_audit_result: str = "PASS",
    implementation_audit_result: str = "PASS",
    final_audit_result: str = "PASS",
    head_after_final: bool = False,
    dirty_after_final: bool = False,
) -> dict:
    repo = tmp_path / "repo"
    repo.mkdir()
    failed = tmp_path / "origins/failed"
    failed.mkdir(parents=True)
    (failed / preparer.ESCROW_NAME).write_text("synthetic escrow\n", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "wave57@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Wave 57 fixture"],
        cwd=repo,
        check=True,
    )
    (repo / SYNTHETIC_PREPARER).write_text("old preparer\n", encoding="utf-8")
    (repo / SYNTHETIC_TEST).write_text("old test\n", encoding="utf-8")
    origin_commit = synthetic_commit(repo, "origin")
    old_hashes = {
        SYNTHETIC_PREPARER: preparer.sha256_file(repo / SYNTHETIC_PREPARER),
        SYNTHETIC_TEST: preparer.sha256_file(repo / SYNTHETIC_TEST),
    }
    origin_contract = {
        "git_commit": origin_commit,
        "config_sha256": "config",
        "prospective_config": {"schema_version": preparer.WAVE57_CONFIG_SCHEMA},
        "sources": copy.deepcopy(old_hashes),
        "upstream": [{"sha256": "upstream"}],
        "historical_preflight": {"status": "PASS"},
        "source_bindings": {"binding": "fixed"},
    }

    (repo / SYNTHETIC_PLAN).write_text("# Synthetic recovery plan\n", encoding="utf-8")
    if mixed_commit == "plan":
        (repo / "plan-extra").write_text("mixed\n", encoding="utf-8")
    plan_sha = preparer.sha256_file(repo / SYNTHETIC_PLAN)
    plan_commit = synthetic_commit(repo, "plan")

    plan_fields = [
        f"**Plan commit:** `{plan_commit}`",
        f"**Plan SHA:** `{plan_sha}`",
        f"**Result:** `{plan_audit_result}`",
    ]
    plan_audit_sha = write_audit_report(
        repo / SYNTHETIC_PLAN_AUDIT,
        "Synthetic plan audit",
        plan_fields,
        plan_audit_result,
    )
    plan_audit_commit = synthetic_commit(repo, "plan audit")
    if break_direct_parent:
        (repo / "intervening").write_text("break DAG\n", encoding="utf-8")
        synthetic_commit(repo, "intervening commit")

    (repo / SYNTHETIC_PREPARER).write_text("new preparer\n", encoding="utf-8")
    (repo / SYNTHETIC_TEST).write_text("new test\n", encoding="utf-8")
    if mixed_commit == "implementation":
        (repo / "implementation-extra").write_text("mixed\n", encoding="utf-8")
    implementation_commit = synthetic_commit(repo, "implementation")
    new_hashes = {
        SYNTHETIC_PREPARER: preparer.sha256_file(repo / SYNTHETIC_PREPARER),
        SYNTHETIC_TEST: preparer.sha256_file(repo / SYNTHETIC_TEST),
    }
    implementation = {
        "commit": implementation_commit,
        "preparer": {
            "path": SYNTHETIC_PREPARER,
            "old_sha256": old_hashes[SYNTHETIC_PREPARER],
            "new_sha256": new_hashes[SYNTHETIC_PREPARER],
        },
        "test": {
            "path": SYNTHETIC_TEST,
            "old_sha256": old_hashes[SYNTHETIC_TEST],
            "new_sha256": new_hashes[SYNTHETIC_TEST],
        },
    }
    implementation_fields = [
        f"**Implementation commit:** `{implementation_commit}`",
        f"**Preparer SHA-256:** `{new_hashes[SYNTHETIC_PREPARER]}`",
        f"**Test SHA-256:** `{new_hashes[SYNTHETIC_TEST]}`",
        f"**Result:** `{implementation_audit_result}`",
    ]
    implementation_audit_sha = write_audit_report(
        repo / SYNTHETIC_IMPLEMENTATION_AUDIT,
        "Synthetic implementation audit",
        implementation_fields,
        implementation_audit_result,
    )
    implementation_audit_commit = synthetic_commit(repo, "implementation audit")

    expected_population = {
        "rows": 4992,
        "total_unique_pair_tokens": 1152,
        "eligible_unique_pair_tokens": 768,
        "out_of_catalog_unique_pair_tokens": 384,
        "noncanonical_unique_pair_tokens": 192,
        "eligible_intersection_noncanonical_unique_pair_tokens": 192,
    }
    amendment = {
        "schema_version": preparer.WAVE57_RECOVERY_AMENDMENT_SCHEMA,
        "status": "APPROVED_PREORACLE_RECOVERY",
        "plan": {
            "commit": plan_commit,
            "path": SYNTHETIC_PLAN,
            "sha256": plan_sha,
        },
        "plan_audit": {
            "commit": plan_audit_commit,
            "path": SYNTHETIC_PLAN_AUDIT,
            "sha256": plan_audit_sha,
        },
        "implementation": implementation,
        "implementation_audit": {
            "commit": implementation_audit_commit,
            "path": SYNTHETIC_IMPLEMENTATION_AUDIT,
            "sha256": implementation_audit_sha,
        },
        "final_audit_path": SYNTHETIC_FINAL_AUDIT,
        "escrow_origin": {
            "failed_attempt_basename": "failed",
            "contract_git_commit": origin_commit,
            "contract_sha256": preparer.compact_json_sha256(origin_contract),
            "escrow_sha256": preparer.sha256_file(failed / preparer.ESCROW_NAME),
            "pre_generation_freeze_sha256": "2" * 64,
            "failure_sha256": "3" * 64,
            "benchmark_manifest_sha256": "4" * 64,
            "inventory": [],
        },
        "population_contract": {
            "eligibility_predicate": {
                "is_out_of_catalog": False,
                "calibration_population": "canonical_preserving",
                "filter_rows_before_deduplicating_pair_tokens": True,
            },
            "counts_by_split": {
                split: copy.deepcopy(expected_population) for split in preparer.SPLITS
            },
        },
        "assertions": {
            "no_redraw": True,
            "no_inference_in_origin": True,
            "no_materialized_oracle_in_origin": True,
            "no_authorized_labels_in_origin": True,
        },
    }
    if amendment_mutator is not None:
        amendment_mutator(amendment)
    amendment_sha = write_canonical_json(repo / SYNTHETIC_AMENDMENT, amendment)
    amendment_commit = synthetic_commit(repo, "amendment")

    final_fields = [
        f"**Audited package commit:** `{amendment_commit}`",
        f"**Amendment SHA-256:** `{amendment_sha}`",
        f"**Result:** `{final_audit_result}`",
    ]
    write_audit_report(
        repo / SYNTHETIC_FINAL_AUDIT,
        "Synthetic final audit",
        final_fields,
        final_audit_result,
    )
    final_commit = synthetic_commit(repo, "final audit")
    if head_after_final:
        (repo / "post-final").write_text("later\n", encoding="utf-8")
        synthetic_commit(repo, "post final")
    if dirty_after_final:
        (repo / "dirty-untracked").write_text("dirty\n", encoding="utf-8")

    execution_contract = copy.deepcopy(origin_contract)
    execution_contract["git_commit"] = synthetic_git(repo, "rev-parse", "HEAD")
    execution_contract["sources"] = copy.deepcopy(new_hashes)
    return {
        "repo": repo,
        "failed": failed,
        "amendment_path": repo / SYNTHETIC_AMENDMENT,
        "amendment": amendment,
        "origin_contract": origin_contract,
        "execution_contract": execution_contract,
        "final_commit": final_commit,
    }


def patch_synthetic_authority_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preparer, "PREPARER_RELATIVE", SYNTHETIC_PREPARER)
    monkeypatch.setattr(
        preparer, "WAVE57_RECOVERY_TEST_RELATIVE", SYNTHETIC_TEST
    )
    monkeypatch.setattr(preparer, "WAVE57_RECOVERY_PLAN_RELATIVE", SYNTHETIC_PLAN)
    monkeypatch.setattr(
        preparer, "WAVE57_RECOVERY_AMENDMENT_RELATIVE", SYNTHETIC_AMENDMENT
    )
    monkeypatch.setattr(
        preparer, "AUDIT_REPORTS_RELATIVE_DIR", SYNTHETIC_AUDIT_DIR
    )


def run_synthetic_authority_validator(
    case: dict, monkeypatch: pytest.MonkeyPatch, amendment_path: Path | None = None
) -> tuple[dict, list[str]]:
    patch_synthetic_authority_paths(monkeypatch)
    events: list[str] = []

    def content_blind(*_args, **_kwargs):
        events.append("content-blind")
        return case["failed"], [], case["origin_contract"]

    def semantic(*_args, **_kwargs):
        events.append("semantic")
        assert events == ["content-blind", "semantic"]
        return {"authorized": True}

    def escrow(*_args, **_kwargs):
        events.append("escrow")
        assert events == ["content-blind", "semantic", "escrow"]
        return {"contract": case["origin_contract"]}

    monkeypatch.setattr(
        preparer, "validate_wave57_failed_origin_content_blind", content_blind
    )
    monkeypatch.setattr(
        preparer, "validate_wave57_failed_origin_semantic", semantic
    )
    monkeypatch.setattr(preparer, "read_escrow", escrow)
    result = preparer._validate_wave57_recovery_amendment(
        amendment_path or case["amendment_path"],
        case["failed"],
        case["execution_contract"],
        "recovery",
        repo_root=case["repo"],
        trusted_public_key_path=case["repo"] / "public.pem",
    )
    return result, events


def test_recovery_dispatch_is_typed_by_prospective_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    amendment = tmp_path / "amendment.json"
    amendment.write_text("{}\n", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        preparer,
        "_validate_wave57_recovery_amendment",
        lambda *_args, **_kwargs: calls.append("wave57") or {"route": "wave57"},
    )
    monkeypatch.setattr(
        preparer,
        "_validate_wave56_recovery_amendment",
        lambda *_args, **_kwargs: calls.append("wave56") or {"route": "wave56"},
    )
    wave57 = {"prospective_config": {"schema_version": preparer.WAVE57_CONFIG_SCHEMA}}
    assert preparer.validate_recovery_amendment(
        amendment, source, wave57, "recovery", repo_root=tmp_path
    ) == {"route": "wave57"}
    wave56 = {
        "prospective_config": {
            "schema_version": "wave56-contextual-residual-gate-stage1-v1"
        }
    }
    assert preparer.validate_recovery_amendment(
        amendment, source, wave56, "recovery", repo_root=tmp_path
    ) == {"route": "wave56"}
    assert calls == ["wave57", "wave56"]


def test_wave57_authority_validator_executes_complete_positive_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = build_wave57_authority_repo(tmp_path)
    result, events = run_synthetic_authority_validator(case, monkeypatch)
    assert result["amendment"]["schema_version"] == preparer.WAVE57_RECOVERY_AMENDMENT_SCHEMA
    assert result["implementation_commit"] == case["amendment"]["implementation"]["commit"]
    assert events == ["content-blind", "semantic", "escrow"]


@pytest.mark.parametrize(
    "fault",
    (
        "alternate_path",
        "schema",
        "status",
        "assertion",
        "population",
        "new_hash",
        "old_hash",
        "mixed_plan",
        "mixed_implementation",
        "broken_parent",
        "plan_revise",
        "implementation_revise",
        "final_revise",
        "missing_implementation_audit",
        "head_after_final",
        "dirty_worktree",
    ),
)
def test_wave57_authority_validator_rejects_each_authority_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    def mutate(amendment: dict) -> None:
        if fault == "schema":
            amendment["schema_version"] = "wrong-schema"
        elif fault == "status":
            amendment["status"] = "DRAFT"
        elif fault == "assertion":
            amendment["assertions"]["no_redraw"] = False
        elif fault == "population":
            amendment["population_contract"]["counts_by_split"]["val"][
                "eligible_unique_pair_tokens"
            ] = 767
        elif fault == "new_hash":
            amendment["implementation"]["preparer"]["new_sha256"] = "f" * 64
        elif fault == "old_hash":
            amendment["implementation"]["test"]["old_sha256"] = "e" * 64

    case = build_wave57_authority_repo(
        tmp_path,
        amendment_mutator=mutate,
        mixed_commit=(
            "plan"
            if fault == "mixed_plan"
            else "implementation" if fault == "mixed_implementation" else None
        ),
        break_direct_parent=fault == "broken_parent",
        plan_audit_result="REVISE" if fault == "plan_revise" else "PASS",
        implementation_audit_result=(
            "REVISE" if fault == "implementation_revise" else "PASS"
        ),
        final_audit_result="REVISE" if fault == "final_revise" else "PASS",
        head_after_final=fault == "head_after_final",
        dirty_after_final=fault == "dirty_worktree",
    )
    path = case["amendment_path"]
    if fault == "alternate_path":
        path = case["repo"] / "alternate.json"
        path.write_bytes(case["amendment_path"].read_bytes())
    if fault == "missing_implementation_audit":
        (case["repo"] / SYNTHETIC_IMPLEMENTATION_AUDIT).unlink()
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        run_synthetic_authority_validator(case, monkeypatch, amendment_path=path)


def test_wave57_public_parser_rejects_every_sensitive_path() -> None:
    for relative in (
        preparer.ESCROW_NAME,
        "benchmark/sealed/train.jsonl",
        "benchmark/sealed/generation_secret.json",
    ):
        with pytest.raises(RuntimeError, match="forbids JSON parsing"):
            preparer._wave57_public_json(FAILED_ORIGIN, relative)


@pytest.mark.skipif(
    not FAILED_ORIGIN.is_dir(),
    reason="the immutable Wave 57 failed origin is required for its final content-blind probe",
)
def test_real_wave57_preflight_hashes_sensitive_files_without_semantic_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amendment, execution_contract = real_content_blind_case()
    sensitive_root = (FAILED_ORIGIN / "benchmark/sealed").resolve()
    escrow_path = (FAILED_ORIGIN / preparer.ESCROW_NAME).resolve()
    opaque_opens: set[str] = set()
    opaque_descriptors: dict[int, dict[str, object]] = {}
    completed_descriptors: dict[str, dict[str, object]] = {}
    original_open = os.open
    original_fstat = os.fstat
    original_read = os.read
    original_close = os.close
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    original_json_loads = json.loads

    def sensitive_relative(path) -> str | None:
        if not isinstance(path, (str, os.PathLike)):
            return None
        candidate = Path(path).resolve()
        if candidate == escrow_path or candidate.is_relative_to(sensitive_root):
            return str(candidate.relative_to(FAILED_ORIGIN.resolve()))
        return None

    def guarded_open(path, flags, *args, **kwargs):
        relative = sensitive_relative(path)
        descriptor = original_open(path, flags, *args, **kwargs)
        if relative is not None:
            assert sys._getframe(1).f_code.co_name == "_secure_file_record"
            assert flags & getattr(os, "O_NOFOLLOW", 0)
            opaque_opens.add(relative)
            opaque_descriptors[descriptor] = {
                "path": relative,
                "fstat": False,
                "reads": 0,
            }
        return descriptor

    def guarded_fstat(descriptor):
        if descriptor in opaque_descriptors:
            assert sys._getframe(1).f_code.co_name == "_secure_file_record"
            opaque_descriptors[descriptor]["fstat"] = True
        return original_fstat(descriptor)

    def guarded_read(descriptor, size):
        if descriptor in opaque_descriptors:
            assert sys._getframe(1).f_code.co_name == "_secure_file_record"
            assert opaque_descriptors[descriptor]["fstat"] is True
            opaque_descriptors[descriptor]["reads"] = int(
                opaque_descriptors[descriptor]["reads"]
            ) + 1
        return original_read(descriptor, size)

    def guarded_close(descriptor):
        if descriptor in opaque_descriptors:
            state = opaque_descriptors.pop(descriptor)
            assert state["fstat"] is True
            assert int(state["reads"]) >= 2
            completed_descriptors[str(state["path"])] = state
        return original_close(descriptor)

    def guarded_builtin_open(path, *args, **kwargs):
        if sensitive_relative(path) is not None:
            raise AssertionError(f"alternate builtins.open reached sensitive path: {path}")
        return original_builtin_open(path, *args, **kwargs)

    def guarded_io_open(path, *args, **kwargs):
        if sensitive_relative(path) is not None:
            raise AssertionError(f"alternate io.open reached sensitive path: {path}")
        return original_io_open(path, *args, **kwargs)

    def guarded_read_text(path: Path, *args, **kwargs):
        if sensitive_relative(path) is not None:
            raise AssertionError(f"semantic read_text reached sensitive path: {path}")
        return original_read_text(path, *args, **kwargs)

    def guarded_read_bytes(path: Path, *args, **kwargs):
        if sensitive_relative(path) is not None:
            raise AssertionError(f"semantic read_bytes reached sensitive path: {path}")
        return original_read_bytes(path, *args, **kwargs)

    def guarded_json_loads(payload, *args, **kwargs):
        if isinstance(payload, (bytes, bytearray, memoryview)):
            raise AssertionError("json.loads received bytes during content-blind preflight")
        return original_json_loads(payload, *args, **kwargs)

    def forbid_fromhex(_frame, event, function):
        if event == "c_call" and getattr(function, "__qualname__", "") == "bytes.fromhex":
            raise AssertionError("bytes.fromhex ran during content-blind preflight")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("content-blind preflight invoked a semantic helper")

    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(os, "fstat", guarded_fstat)
    monkeypatch.setattr(os, "read", guarded_read)
    monkeypatch.setattr(os, "close", guarded_close)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(io, "open", guarded_io_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(json, "loads", guarded_json_loads)
    monkeypatch.setattr(preparer, "read_escrow", forbidden)
    monkeypatch.setattr(preparer, "keys_from_escrow", forbidden)
    monkeypatch.setattr(preparer, "validate_semantic_attestation", forbidden)
    monkeypatch.setattr(preparer, "sealed_population_counts", forbidden)
    monkeypatch.setattr(preparer, "read_jsonl", forbidden)

    sys.setprofile(forbid_fromhex)
    try:
        failed, inventory, public_contract = (
            preparer.validate_wave57_failed_origin_content_blind(
                amendment,
                FAILED_ORIGIN.parent,
                execution_contract,
                preparer.PUBLIC_KEY,
            )
        )
    finally:
        sys.setprofile(None)
    assert failed == FAILED_ORIGIN
    assert inventory == amendment["escrow_origin"]["inventory"]
    assert public_contract["git_commit"] == "379229f1cae0f1b713fe5c293f303ed60ed7f187"
    assert preparer.ESCROW_NAME in opaque_opens
    assert {
        "benchmark/sealed/train.jsonl",
        "benchmark/sealed/val.jsonl",
        "benchmark/sealed/lockbox.jsonl",
        "benchmark/sealed/calibration_null.jsonl",
        "benchmark/sealed/generation_secret.json",
        "benchmark/sealed/identity_secret.json",
        "benchmark/sealed/semantic_commitment_secret.json",
    }.issubset(opaque_opens)
    assert not opaque_descriptors
    assert set(completed_descriptors) == opaque_opens


@pytest.mark.skipif(not FAILED_ORIGIN.is_dir(), reason="Wave 57 failed origin unavailable")
@pytest.mark.parametrize("fault", ("extra", "ownership", "mode", "hash"))
def test_wave57_content_blind_preflight_rejects_inventory_and_forbidden_material(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    amendment, execution_contract = real_content_blind_case()
    observed = copy.deepcopy(amendment["escrow_origin"]["inventory"])
    if fault == "extra":
        observed.append(
            {
                "path": "inference",
                "type": "directory",
                "mode": "0700",
                "uid": 0,
                "gid": 0,
            }
        )
    elif fault == "ownership":
        next(row for row in observed if row["path"] == preparer.ESCROW_NAME)["uid"] = 1
    elif fault == "mode":
        next(row for row in observed if row["path"] == preparer.ESCROW_NAME)["mode"] = "0644"
    elif fault == "hash":
        next(row for row in observed if row["path"] == preparer.ESCROW_NAME)["sha256"] = "0" * 64
    amendment["escrow_origin"]["inventory"] = observed
    monkeypatch.setattr(preparer, "physical_tree_inventory", lambda _root: observed)
    with pytest.raises((RuntimeError, PermissionError)):
        preparer.validate_wave57_failed_origin_content_blind(
            amendment,
            FAILED_ORIGIN.parent,
            execution_contract,
            preparer.PUBLIC_KEY,
        )


@pytest.mark.skipif(not FAILED_ORIGIN.is_dir(), reason="Wave 57 failed origin unavailable")
def test_wave57_contract_delta_rejects_every_unapproved_change() -> None:
    amendment, execution_contract = real_content_blind_case()
    origin_contract = json.loads(
        (FAILED_ORIGIN / preparer.FREEZE_NAME).read_text(encoding="utf-8")
    )["contract"]
    preparer._validate_wave57_contract_delta(
        origin_contract, execution_contract, amendment, REPO_ROOT
    )

    changed_config = copy.deepcopy(execution_contract)
    changed_config["prospective_config"]["cpu_threads"] = 99
    with pytest.raises(RuntimeError, match="changed frozen field"):
        preparer._validate_wave57_contract_delta(
            origin_contract, changed_config, amendment, REPO_ROOT
        )

    changed_source_set = copy.deepcopy(execution_contract)
    changed_source_set["sources"]["unauthorized.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="source set differs"):
        preparer._validate_wave57_contract_delta(
            origin_contract, changed_source_set, amendment, REPO_ROOT
        )

    wrong_hash = copy.deepcopy(amendment)
    wrong_hash["implementation"]["test"]["new_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="test source delta differs"):
        preparer._validate_wave57_contract_delta(
            origin_contract, execution_contract, wrong_hash, REPO_ROOT
        )


def test_authorized_wave57_revalidation_orders_content_blind_before_semantic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed = tmp_path / "failed"
    failed.mkdir()
    amendment = {"schema_version": preparer.WAVE57_RECOVERY_AMENDMENT_SCHEMA}
    context = {
        "amendment": amendment,
        "failed_attempt": failed,
        "repo_root": tmp_path,
    }
    inventory = [{"path": ".", "type": "directory"}]
    calls: list[str] = []

    def blind(*_args, **_kwargs):
        calls.append("content-blind")
        return failed, inventory, {"contract": "public"}

    def semantic(*_args, **_kwargs):
        calls.append("semantic")
        assert calls == ["content-blind", "semantic"]
        return {"keys": "authorized"}

    monkeypatch.setattr(
        preparer, "validate_wave57_failed_origin_content_blind", blind
    )
    monkeypatch.setattr(
        preparer, "validate_wave57_failed_origin_semantic", semantic
    )
    assert preparer.revalidate_authorized_recovery_origin(
        context, {"execution": "contract"}, tmp_path / "public.pem"
    ) == (failed, inventory)
    assert calls == ["content-blind", "semantic"]


def test_wave57_recovery_provenance_uses_typed_amendment_schema() -> None:
    context = {
        "amendment": {"schema_version": preparer.WAVE57_RECOVERY_AMENDMENT_SCHEMA},
        "amendment_path": preparer.WAVE57_RECOVERY_AMENDMENT_RELATIVE,
        "amendment_sha256": "a" * 64,
        "implementation_commit": "implementation",
        "implementation_audit": {"path": "audit", "sha256": "b" * 64},
        "final_audit": {"path": "final", "sha256": "c" * 64},
        "failed_attempt_basename": "failed",
        "escrow_origin_contract_sha256": "d" * 64,
        "benchmark_manifest_sha256": "e" * 64,
    }
    provenance = preparer.recovery_provenance(context, {"contract": "execution"})
    assert provenance["schema_version"] == preparer.WAVE57_RECOVERY_AMENDMENT_SCHEMA


def test_wave57_report_parser_accepts_verbatim_r401_but_rejects_extra_blanks(
    tmp_path: Path,
) -> None:
    report = (
        REPO_ROOT
        / "Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/"
        "401_wave57_preoracle_recovery_plan_final_reaudit.md"
    )
    fields = [
        "**Plan commit:** `d21f3c96e0313077f8e121dab731c9f8a7a93320`",
        "**Plan SHA:** `8485b07e8632f8a247f68db5eadabdc90dfe9944479a274d25fdb01a36165d52`",
        "**Result:** `PASS`",
    ]
    preparer._require_report_fields(
        report,
        fields,
        "Wave 57 plan audit",
        allow_one_terminal_blank=True,
    )
    invalid = tmp_path / "triple-terminal-newline.md"
    invalid.write_bytes(report.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="canonical attestation block"):
        preparer._require_report_fields(
            invalid,
            fields,
            "Wave 57 plan audit",
            allow_one_terminal_blank=True,
        )


def synthetic_data(n_tokens: int = 420, seed: int = 5700) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    permutations = np.asarray(list(itertools.permutations(range(4))), dtype=np.int64)
    utilities = np.asarray([-0.3, 0.2, 0.8, 1.5], dtype=np.float64)[permutations]
    token_index = np.arange(n_tokens)[:, None]
    policy_index = np.arange(24)[None, :]
    hard = ((token_index + policy_index) % 4).astype(np.int64)
    disagreement = ((token_index + 2 * policy_index) % 5) != 0
    posterior = np.where(disagreement, (hard + 1 + policy_index % 2) % 4, hard)
    target = np.zeros((n_tokens, 4), dtype=bool)
    target[np.arange(n_tokens), np.arange(n_tokens) % 4] = True
    target[np.arange(n_tokens), (np.arange(n_tokens) + 1 + np.arange(n_tokens) % 2) % 4] = True
    hard_regret = constrained_regret(hard, target, utilities, 1.25)
    posterior_regret = constrained_regret(posterior, target, utilities, 1.25)
    gain = hard_regret - posterior_regret
    design = rng.normal(size=(n_tokens, 24, 17))
    # Make mean gain and harm separately learnable without copying truth exactly.
    design[..., 0] = gain + rng.normal(scale=0.18, size=gain.shape)
    design[..., 1] = (gain < -1e-12) + rng.normal(scale=0.35, size=gain.shape)
    ensemble = rng.normal(size=(n_tokens, 4))
    per_seed = ensemble[None, ...] + rng.normal(scale=0.1, size=(3, n_tokens, 4))
    hard_set = np.zeros((n_tokens, 4), dtype=bool)
    hard_set[np.arange(n_tokens), np.argmax(ensemble, axis=1)] = True
    tokens = np.asarray([f"wave57-{seed}-{index:05d}" for index in range(n_tokens)])
    return {
        "pair_token": tokens,
        "cluster_id": tokens.copy(),
        "target": target,
        "per_seed_logits": per_seed,
        "ensemble_logits": ensemble,
        "design_stratum": np.asarray(["NEAR_RIVAL"] * n_tokens),
        "cardinality": np.full(n_tokens, 2, dtype=np.int64),
        "split_role": np.asarray(["synthetic"] * n_tokens),
        "design": design,
        "gain": gain,
        "disagreement": disagreement,
        "weights": disagreement_weights(disagreement),
        "primary": np.ones(n_tokens, dtype=bool),
        "hard_actions": hard,
        "posterior_actions": posterior,
        "hard_set": hard_set,
        "advantage": np.maximum(design[..., 0], 0.0),
        "absent_support": np.zeros(n_tokens, dtype=bool),
    }, utilities


def test_frozen_config_is_accepted_by_preparer_and_worker() -> None:
    prospective = config()
    preparer.validate_prospective_config(prospective)
    validate_wave57_frozen_config(prospective)
    worker.base.validate_frozen_config(prospective)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("hard_set_tau",), 0.25),
        (("incompatible_regret_penalty",), 9.0),
        (("selection", "accuracy_noninferiority_margin"), 0.2),
        (("diagnostic_criteria", "regret_reduction_vs_hard_min"), 0.2),
        (("quantile_method",), "linear"),
        (("harm_epsilon",), 1e-6),
        (("bootstrap", "interval"), [5.0, 95.0]),
        (("shards", "salt"), "wrong-salt"),
        (("shared_boundary_interface", "compatibility_semantics"), "silent-alias"),
        (
            ("phase_worker_relative",),
            "experiments/geometria_proporcional/_wave56_phase_worker.py",
        ),
        (("scalar_gamma_grid",), [0.0, "hard_only"]),
        (("phase_runtime_sources",), []),
    ),
)
def test_complete_frozen_config_rejects_each_drift_family(
    path: tuple[str, ...], replacement: object
) -> None:
    prospective = config()
    target = prospective
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(RuntimeError, match="complete frozen config drifted"):
        validate_wave57_frozen_config(prospective)
    with pytest.raises(RuntimeError, match="complete frozen config drifted"):
        preparer.validate_prospective_config(prospective)


def test_runtime_hook_stages_only_bound_sources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    worker_path = coordinator.build_phase_runtime(source, config())
    assert worker_path.name == "_wave57_phase_worker.py"
    assert (source / "_wave56_phase_worker.py").is_file()
    assert (source / "geometria_proporcional/wave57_tail_guard.py").is_file()
    staged = {path.name for path in source.iterdir() if path.is_file()}
    assert staged == {
        "_wave56_phase_worker.py",
        "_wave57_phase_worker.py",
        "run_wave56_retrospective.py",
    }
    completed = subprocess.run(
        [sys.executable, str(worker_path), "--help"],
        cwd=source,
        env={"PATH": str(Path(sys.executable).parent), "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_unbound_runtime_hook_is_rejected(tmp_path: Path) -> None:
    prospective = config()
    prospective["phase_runtime_sources"].append(
        "experiments/geometria_proporcional/run_wave55_policy_bridge.py"
    )
    try:
        coordinator.build_phase_runtime(tmp_path / "source", prospective)
    except RuntimeError as error:
        assert "not bound" in str(error)
    else:
        raise AssertionError("unbound runtime source was accepted")


def test_proposer_hard_only_terminates_selection_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, utilities = synthetic_data(n_tokens=20)
    terminal = {
        "selected": {"threshold": "hard_only", "support": {"tokens": 0}},
        "grid": [],
    }
    monkeypatch.setattr(worker, "select_proposer", lambda *args, **kwargs: terminal)
    result = worker._selection_sequence(
        np.zeros_like(data["gain"]),
        np.zeros_like(data["gain"]),
        [np.zeros_like(data["gain"])] * 5,
        ["PASS"] * 5,
        data,
        utilities,
        config(),
        data["primary"],
        40,
        25,
    )
    assert result["status"] == "NOT_EVALUABLE"
    assert result["reason"] == "proposer_identity_or_low_support"


def test_guard_cells_fail_individually_on_authorized_token_support() -> None:
    data, utilities = synthetic_data(n_tokens=60)
    data["disagreement"][:, 0] = True
    proposals = np.zeros_like(data["disagreement"], dtype=bool)
    proposals[:, 0] = True
    probabilities = np.full(proposals.shape, np.nan)
    probabilities[:, 0] = np.linspace(0.0, 1.0, len(data["primary"]))
    result = worker.select_guard(
        probabilities,
        proposals,
        data,
        utilities,
        config(),
        data["primary"],
        30,
    )
    nonterminal = result["grid"][:-1]
    assert any(not row["evaluable"] for row in nonterminal)
    assert any(row["evaluable"] for row in nonterminal)
    assert result["grid"][-1]["threshold"] == "hard_only"


def test_one_invalid_sham_does_not_terminate_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prospective = config()
    data, utilities = synthetic_data(seed=5711)
    original = worker.conditional_harm_shuffle
    calls = 0

    def one_invalid(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        if calls == 0:
            result["diagnostics"]["permutable_fraction"] = 0.0
            result["diagnostics"]["hamming_global"] = 0.0
            result["diagnostics"]["hamming_weighted"] = 0.0
        calls += 1
        return result

    monkeypatch.setattr(worker, "conditional_harm_shuffle", one_invalid)
    stage = tmp_path / "stage"
    output = tmp_path / "output"
    stage.mkdir()
    output.mkdir()
    (stage / "phase_request.json").write_text(
        json.dumps({"phase": "fit", "synthetic": True}), encoding="utf-8"
    )
    assert worker.run_fit(stage, output, prospective, utilities, data) == "FIT_COMPLETE"
    rows = json.loads((output / "fit_core.json").read_text(encoding="utf-8"))["shams"]
    assert [row["status"] for row in rows] == ["NOT_EVALUABLE"] + ["PASS"] * 4
    assert (output / "fit_arrays.npz").is_file()

    select_data, _ = synthetic_data(seed=5712)
    select_stage = tmp_path / "select-stage"
    (select_stage / "previous").mkdir(parents=True)
    (select_stage / "phase_request.json").write_text(
        json.dumps({"phase": "select", "synthetic": True}), encoding="utf-8"
    )
    shutil.copy2(output / "fit_core.json", select_stage / "previous/fit_core.json")
    select_output = tmp_path / "select-output"
    select_output.mkdir()
    assert worker.run_select(
        select_stage, select_output, prospective, utilities, select_data
    ) == "SELECT_COMPLETE"
    selection = json.loads(
        (select_output / "selection_core.json").read_text(encoding="utf-8")
    )
    assert selection["sequence"]["guard"]["selected"] is not None
    assert [row["status"] for row in selection["sequence"]["shams"]] == [
        "NOT_EVALUABLE",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
    ]

    monitor_data, _ = synthetic_data(seed=5713)
    adjudicate_stage = tmp_path / "adjudicate-stage"
    (adjudicate_stage / "previous").mkdir(parents=True)
    for name in ("fit_core.json", "fit_arrays.npz"):
        shutil.copy2(output / name, adjudicate_stage / "previous" / name)
    for name in ("selection_core.json", "selection_freeze.json", "selection_arrays.npz"):
        shutil.copy2(select_output / name, adjudicate_stage / "previous" / name)
    adjudicate_output = tmp_path / "adjudicate-output"
    adjudicate_output.mkdir()
    assert worker.run_adjudicate(
        adjudicate_stage, adjudicate_output, prospective, utilities, monitor_data
    ) == "COMPLETE"
    analysis = json.loads(
        (adjudicate_output / "analysis_core.json").read_text(encoding="utf-8")
    )
    assert analysis["arms"]["mean_plus_harm_guard"]["status"] == "PASS"
    assert analysis["arms"]["mean_plus_shuffled_harm_guard"] == {
        "status": "NOT_EVALUABLE",
        "summary": None,
        "replicate_statuses": ["NOT_EVALUABLE"] + ["PASS"] * 4,
    }
    assert analysis["contrasts"]["main_minus_shuffled"]["status"] == "NOT_EVALUABLE"
    assert analysis["diagnostic_pattern"]["conditions"]["diagnostic_condition_5"] == "NOT_EVALUABLE"
    with np.load(adjudicate_output / "result_arrays.npz", allow_pickle=False) as arrays:
        assert not any(name.startswith("sham__average_metric__") for name in arrays.files)


def test_one_low_support_shard_is_granular_and_uses_shard_minima(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prospective = config()
    fit_data, utilities = synthetic_data(seed=5721)
    select_data, _ = synthetic_data(seed=5722)
    tokens_by_shard = {0: [], 1: []}
    candidate = 0
    while len(tokens_by_shard[0]) < 39 or len(tokens_by_shard[1]) < 381:
        token = f"imbalanced-{candidate:06d}"
        shard = int(worker._shard_assignment(np.asarray([token]))[0])
        limit = 39 if shard == 0 else 381
        if len(tokens_by_shard[shard]) < limit:
            tokens_by_shard[shard].append(token)
        candidate += 1
    tokens = np.asarray(tokens_by_shard[0] + tokens_by_shard[1])
    select_data["pair_token"] = tokens
    select_data["cluster_id"] = tokens.copy()

    fit_stage = tmp_path / "fit-stage"
    fit_output = tmp_path / "fit-output"
    fit_stage.mkdir()
    fit_output.mkdir()
    (fit_stage / "phase_request.json").write_text(
        json.dumps({"phase": "fit", "synthetic": True}), encoding="utf-8"
    )
    assert worker.run_fit(
        fit_stage, fit_output, prospective, utilities, fit_data
    ) == "FIT_COMPLETE"

    calls: list[tuple[int, int]] = []
    original = worker._selection_sequence

    def recording_sequence(*args, **kwargs):
        calls.append((int(args[-2]), int(args[-1])))
        return original(*args, **kwargs)

    monkeypatch.setattr(worker, "_selection_sequence", recording_sequence)
    select_stage = tmp_path / "select-stage"
    (select_stage / "previous").mkdir(parents=True)
    (select_stage / "phase_request.json").write_text(
        json.dumps({"phase": "select", "synthetic": True}), encoding="utf-8"
    )
    shutil.copy2(fit_output / "fit_core.json", select_stage / "previous/fit_core.json")
    select_output = tmp_path / "select-output"
    select_output.mkdir()
    assert worker.run_select(
        select_stage, select_output, prospective, utilities, select_data
    ) == "SELECT_COMPLETE"
    core = json.loads((select_output / "selection_core.json").read_text(encoding="utf-8"))
    assert core["shards"]["0"]["status"] == "NOT_EVALUABLE"
    assert core["shards"]["1"]["status"] == "PASS"
    assert calls[0] == (40, 25)
    assert calls[1:] == [(20, 12)]


def run_synthetic_package(root: Path) -> None:
    prospective = config()
    fit_data, utilities = synthetic_data(seed=5701)
    select_data, _ = synthetic_data(seed=5702)
    monitor_data, _ = synthetic_data(seed=5703)
    monitor_data["target"][:40] = np.asarray([True, False, True, False])
    hard_regret = constrained_regret(
        monitor_data["hard_actions"], monitor_data["target"], utilities, 1.25
    )
    posterior_regret = constrained_regret(
        monitor_data["posterior_actions"], monitor_data["target"], utilities, 1.25
    )
    monitor_data["gain"] = hard_regret - posterior_regret
    fit_stage = root / "fit-stage"
    fit_stage.mkdir(parents=True)
    (fit_stage / "phase_request.json").write_text(
        json.dumps({"phase": "fit", "synthetic": True}), encoding="utf-8"
    )
    fit = root / "phases/fit.complete"
    fit.mkdir(parents=True)
    assert worker.run_fit(fit_stage, fit, prospective, utilities, fit_data) == "FIT_COMPLETE"

    select_stage = root / "select-stage"
    (select_stage / "previous").mkdir(parents=True)
    (select_stage / "phase_request.json").write_text(
        json.dumps({"phase": "select", "synthetic": True}), encoding="utf-8"
    )
    shutil.copy2(fit / "fit_core.json", select_stage / "previous/fit_core.json")
    select = root / "phases/select.complete"
    select.mkdir(parents=True)
    assert worker.run_select(select_stage, select, prospective, utilities, select_data) == "SELECT_COMPLETE"
    freeze = json.loads((select / "selection_freeze.json").read_text(encoding="utf-8"))["selected"]
    assert freeze["proposer"]["threshold"] != "hard_only"

    adjudicate_stage = root / "adjudicate-stage"
    (adjudicate_stage / "previous").mkdir(parents=True)
    for name in ("fit_core.json", "fit_arrays.npz"):
        shutil.copy2(fit / name, adjudicate_stage / "previous" / name)
    for name in ("selection_core.json", "selection_freeze.json", "selection_arrays.npz"):
        shutil.copy2(select / name, adjudicate_stage / "previous" / name)
    adjudicate = root / "phases/adjudicate.complete"
    adjudicate.mkdir(parents=True)
    assert worker.run_adjudicate(
        adjudicate_stage, adjudicate, prospective, utilities, monitor_data
    ) == "COMPLETE"


def test_synthetic_fit_select_adjudicate_and_exact_replay(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    run_synthetic_package(primary)
    run_synthetic_package(replay)
    report = json.loads(
        (primary / "phases/adjudicate.complete/analysis_core.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["estimand"]["unit"] == "pair_token"
    assert len(report["diagnostic_pattern"]["conditions"]) == 6
    assert set(report["absent_support_by_set"]) == {"0", "4", "8", "10", "12"}
    support = report["absent_support_by_set"]["4"]
    assert support["status"] == "EVALUABLE"
    assert support["arm_statuses"]["mean_plus_shuffled_harm_guard"] == "PASS"
    assert support["summaries"]["mean_plus_shuffled_harm_guard"] is not None
    assert support["sham_replicate_statuses"] == ["PASS"] * 5
    assert "main_minus_shuffled" in support["contrasts"]
    assert "scalar_advantage_gate" in report["secondary_diagnostics"]
    assert "all_in_catalog_by_observational_slice" in report["secondary_diagnostics"][
        "mean_plus_harm_guard_sensitivities"
    ]

    for relative in (
        "phases/fit.complete/fit_core.json",
        "phases/fit.complete/fit_freeze.json",
        "phases/fit.complete/fit_arrays.npz",
        "phases/select.complete/selection_core.json",
        "phases/select.complete/selection_freeze.json",
        "phases/select.complete/selection_arrays.npz",
        "phases/adjudicate.complete/analysis_core.json",
        "phases/adjudicate.complete/result_arrays.npz",
    ):
        assert (primary / relative).read_bytes() == (replay / relative).read_bytes()
    preparation = {
        "config_sha256": coordinator.sha256_file(CONFIG_PATH),
        "prospective_config": config(),
    }
    for root in (primary, replay):
        (root / "preparation_freeze.json").write_text(
            json.dumps(preparation, sort_keys=True), encoding="utf-8"
        )
    comparison = coordinator.compare_reference(replay, primary)
    assert comparison["all_exact"] is True
    assert comparison["checks"]["preparation/prospective_config"] is True
    assert comparison["checks"]["fit/fit_freeze.json"] is True
    assert comparison["checks"]["select/selection_freeze.json"] is True


@pytest.mark.parametrize("relative", ("fit/fit_freeze.json", "select/selection_freeze.json"))
def test_replay_rejects_each_mutated_phase_freeze(tmp_path: Path, relative: str) -> None:
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    run_synthetic_package(primary)
    shutil.copytree(primary, replay)
    path = replay / "phases" / f"{relative.split('/')[0]}.complete" / relative.split('/')[1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["adversarial_mutation"] = relative
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="replay mismatch"):
        coordinator.compare_reference(replay, primary)


def test_monitor_without_disagreement_is_not_evaluable(tmp_path: Path) -> None:
    data, utilities = synthetic_data()
    data["disagreement"][:] = False
    data["weights"][:] = 0.0
    output = tmp_path / "output"
    output.mkdir()
    failure = worker._global_failure(data, config(), "adjudicate")
    assert failure is not None
    assert "disagreement_tokens" in failure["failed"]
