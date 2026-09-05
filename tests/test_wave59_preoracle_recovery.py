from __future__ import annotations

import builtins
import copy
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments/geometria_proporcional"
SRC = REPO_ROOT / "src"
for candidate in (EXPERIMENTS, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import prepare_wave56_fresh as preparer  # noqa: E402
import run_wave59_hgb_guard_bracket as runner  # noqa: E402


FAILED_ORIGIN = (
    REPO_ROOT
    / "data/geometria_proporcional/"
    "wave59_fresh_hgb_guard_bracket_v1.failed_20260905T071003529517Z"
)


def _commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=repo, check=True)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _canonical_report(title: str, fields: list[str], body: str = "Checked.") -> str:
    result = next(
        field.removeprefix("**Result:** `").removesuffix("`")
        for field in fields
        if field.startswith("**Result:** `")
    )
    return (
        f"# {title}\n\n"
        + "\n".join(fields)
        + f"\n\n## Review\n\n{body}\n\n"
        + "## Machine-verifiable decision\n\n"
        + f"**Final decision:** `{result}`\n"
    )


def synthetic_repository_authority(
    tmp_path: Path,
    *,
    amendment_mutator: Callable[[dict], None] | None = None,
    implementation_extra: bool = False,
    implementation_result: str = "PASS",
) -> tuple[Path, dict, dict, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "wave59-recovery@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Wave 59 Recovery Test"],
        cwd=repo,
        check=True,
    )
    source_specs = {
        preparer.PREPARER_RELATIVE: ("old preparer\n", "new preparer\n"),
        preparer.WAVE59_RUNNER_RELATIVE: ("old runner\n", "new runner\n"),
        preparer.WAVE59_PROSPECTIVE_TEST_RELATIVE: (
            "old prospective test\n",
            "new prospective test\n",
        ),
    }
    for relative, (old, _) in source_specs.items():
        _write(repo / relative, old)
    _commit(repo, "baseline")
    expected = {
        relative: preparer.sha256_file(repo / relative)
        for relative in source_specs
    }

    for relative in (
        preparer.WAVE59_RUNNER_RELATIVE,
        preparer.WAVE59_PROSPECTIVE_TEST_RELATIVE,
    ):
        _write(repo / relative, source_specs[relative][1])
    _commit(repo, "previous audited implementation")

    plan_path = repo / preparer.WAVE59_RECOVERY_PLAN_RELATIVE
    _write(plan_path, "# Synthetic Wave 59 recovery plan\n")
    plan_commit = _commit(repo, "plan")
    plan_sha = preparer.sha256_file(plan_path)

    plan_audit_relative = (
        f"{preparer.AUDIT_REPORTS_RELATIVE_DIR}/synthetic_plan_audit.md"
    )
    plan_audit_path = repo / plan_audit_relative
    _write(
        plan_audit_path,
        _canonical_report(
            "Synthetic plan audit",
            [
                f"**Plan commit:** `{plan_commit}`",
                f"**Plan SHA:** `{plan_sha}`",
                "**Result:** `PASS`",
            ],
        ),
    )
    plan_audit_commit = _commit(repo, "plan audit")
    plan_audit_sha = preparer.sha256_file(plan_audit_path)

    _write(
        repo / preparer.PREPARER_RELATIVE,
        source_specs[preparer.PREPARER_RELATIVE][1],
    )
    recovery_test_path = repo / preparer.WAVE59_RECOVERY_TEST_RELATIVE
    _write(recovery_test_path, "# synthetic recovery coverage\n")
    if implementation_extra:
        _write(repo / "unexpected.txt", "mixed implementation\n")
    implementation_commit = _commit(repo, "implementation")
    observed = {
        relative: preparer.sha256_file(repo / relative)
        for relative in source_specs
    }
    recovery_test_sha = preparer.sha256_file(recovery_test_path)

    implementation_audit_relative = (
        f"{preparer.AUDIT_REPORTS_RELATIVE_DIR}/synthetic_implementation_audit.md"
    )
    implementation_audit_path = repo / implementation_audit_relative
    implementation_fields = [
        f"**Implementation commit:** `{implementation_commit}`",
        f"**Preparer SHA-256:** `{observed[preparer.PREPARER_RELATIVE]}`",
        f"**Runner SHA-256:** `{observed[preparer.WAVE59_RUNNER_RELATIVE]}`",
        "**Prospective test SHA-256:** "
        f"`{observed[preparer.WAVE59_PROSPECTIVE_TEST_RELATIVE]}`",
        f"**Recovery test SHA-256:** `{recovery_test_sha}`",
        f"**Result:** `{implementation_result}`",
    ]
    _write(
        implementation_audit_path,
        _canonical_report("Synthetic implementation audit", implementation_fields),
    )
    implementation_audit_commit = _commit(repo, "implementation audit")
    implementation_audit_sha = preparer.sha256_file(implementation_audit_path)

    final_audit_relative = (
        f"{preparer.AUDIT_REPORTS_RELATIVE_DIR}/synthetic_final_audit.md"
    )
    implementation = {
        "commit": implementation_commit,
        "preparer": {
            "path": preparer.PREPARER_RELATIVE,
            "old_sha256": expected[preparer.PREPARER_RELATIVE],
            "new_sha256": observed[preparer.PREPARER_RELATIVE],
        },
        "runner": {
            "path": preparer.WAVE59_RUNNER_RELATIVE,
            "old_sha256": expected[preparer.WAVE59_RUNNER_RELATIVE],
            "new_sha256": observed[preparer.WAVE59_RUNNER_RELATIVE],
        },
        "prospective_test": {
            "path": preparer.WAVE59_PROSPECTIVE_TEST_RELATIVE,
            "old_sha256": expected[preparer.WAVE59_PROSPECTIVE_TEST_RELATIVE],
            "new_sha256": observed[preparer.WAVE59_PROSPECTIVE_TEST_RELATIVE],
        },
        "recovery_test": {
            "path": preparer.WAVE59_RECOVERY_TEST_RELATIVE,
            "sha256": recovery_test_sha,
            "introduced_commit": implementation_commit,
        },
    }
    amendment = {
        "schema_version": preparer.WAVE59_RECOVERY_AMENDMENT_SCHEMA,
        "status": "APPROVED_PREORACLE_RECOVERY",
        "plan": {
            "commit": plan_commit,
            "path": preparer.WAVE59_RECOVERY_PLAN_RELATIVE,
            "sha256": plan_sha,
        },
        "plan_audit": {
            "commit": plan_audit_commit,
            "path": plan_audit_relative,
            "sha256": plan_audit_sha,
        },
        "implementation": implementation,
        "implementation_audit": {
            "commit": implementation_audit_commit,
            "path": implementation_audit_relative,
            "sha256": implementation_audit_sha,
        },
        "final_audit_path": final_audit_relative,
        "escrow_origin": {},
        "population_contract": {},
        "assertions": {
            "no_redraw": True,
            "no_inference_in_origin": True,
            "no_materialized_oracle_in_origin": True,
            "no_authorized_labels_in_origin": True,
        },
    }
    if amendment_mutator is not None:
        amendment_mutator(amendment)
    amendment_path = repo / preparer.WAVE59_RECOVERY_AMENDMENT_RELATIVE
    amendment_path.parent.mkdir(parents=True, exist_ok=True)
    preparer.atomic_write_json(amendment_path, amendment, mode=0o644)
    amendment_commit = _commit(repo, "amendment")
    amendment_sha = preparer.sha256_file(amendment_path)

    final_path = repo / final_audit_relative
    _write(
        final_path,
        _canonical_report(
            "Synthetic final audit",
            [
                f"**Audited package commit:** `{amendment_commit}`",
                f"**Amendment SHA-256:** `{amendment_sha}`",
                "**Result:** `PASS`",
            ],
        ),
    )
    _commit(repo, "final audit")
    return repo, expected, observed, amendment_path


def test_wave59_repository_authority_accepts_only_the_declared_three_deltas(
    tmp_path: Path,
) -> None:
    repo, expected, observed, amendment_path = synthetic_repository_authority(tmp_path)
    amendment, digest = preparer.validate_wave59_repository_recovery_authority(
        amendment_path, expected, observed, repo_root=repo
    )
    assert amendment["schema_version"] == preparer.WAVE59_RECOVERY_AMENDMENT_SCHEMA
    assert digest == preparer.sha256_file(amendment_path)
    forged = dict(observed)
    forged["unapproved.py"] = "f" * 64
    with pytest.raises(RuntimeError, match="source set differs"):
        preparer.validate_wave59_repository_recovery_authority(
            amendment_path, expected, forged, repo_root=repo
        )
    alias = tmp_path / "amendment-alias.json"
    alias.symlink_to(amendment_path)
    with pytest.raises(RuntimeError, match="canonical repository path"):
        preparer.validate_wave59_repository_recovery_authority(
            alias, expected, observed, repo_root=repo
        )


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda amendment: amendment.__setitem__("schema_version", "forged"),
            "schema differs",
        ),
        (
            lambda amendment: amendment.__setitem__("status", "PENDING"),
            "not approved",
        ),
        (
            lambda amendment: amendment["assertions"].__setitem__("no_redraw", False),
            "assertions differ",
        ),
    ],
)
def test_wave59_repository_authority_rejects_schema_status_and_assertions(
    tmp_path: Path, mutator: Callable[[dict], None], match: str
) -> None:
    repo, expected, observed, amendment_path = synthetic_repository_authority(
        tmp_path, amendment_mutator=mutator
    )
    with pytest.raises(RuntimeError, match=match):
        preparer.validate_wave59_repository_recovery_authority(
            amendment_path, expected, observed, repo_root=repo
        )


def test_wave59_repository_authority_rejects_mixed_implementation(
    tmp_path: Path,
) -> None:
    repo, expected, observed, amendment_path = synthetic_repository_authority(
        tmp_path, implementation_extra=True
    )
    with pytest.raises(RuntimeError, match="unauthorized paths"):
        preparer.validate_wave59_repository_recovery_authority(
            amendment_path, expected, observed, repo_root=repo
        )


def test_wave59_repository_authority_rejects_contradictory_audit(
    tmp_path: Path,
) -> None:
    repo, expected, observed, amendment_path = synthetic_repository_authority(
        tmp_path, implementation_result="REVISE"
    )
    with pytest.raises(RuntimeError, match="canonical attestation block"):
        preparer.validate_wave59_repository_recovery_authority(
            amendment_path, expected, observed, repo_root=repo
        )


def test_wave59_repository_authority_rejects_post_audit_head(
    tmp_path: Path,
) -> None:
    repo, expected, observed, amendment_path = synthetic_repository_authority(tmp_path)
    _write(repo / "later.txt", "unauthorized later commit\n")
    _commit(repo, "later")
    with pytest.raises(RuntimeError, match="final-audit commit"):
        preparer.validate_wave59_repository_recovery_authority(
            amendment_path, expected, observed, repo_root=repo
        )


def test_wave59_contract_delta_rejects_every_undeclared_change() -> None:
    amendment, execution = _real_content_blind_case()
    origin = json.loads(
        (FAILED_ORIGIN / preparer.FREEZE_NAME).read_text(encoding="utf-8")
    )["contract"]
    preparer._validate_wave59_contract_delta(origin, execution, amendment, REPO_ROOT)
    changed_field = copy.deepcopy(execution)
    changed_field["source_bindings"] = {"forged": True}
    with pytest.raises(RuntimeError, match="changed frozen field"):
        preparer._validate_wave59_contract_delta(
            origin, changed_field, amendment, REPO_ROOT
        )
    changed_set = copy.deepcopy(execution)
    changed_set["sources"]["unapproved.py"] = "a" * 64
    with pytest.raises(RuntimeError, match="source set differs"):
        preparer._validate_wave59_contract_delta(
            origin, changed_set, amendment, REPO_ROOT
        )
    missing_delta = copy.deepcopy(execution)
    missing_delta["sources"][preparer.WAVE59_RUNNER_RELATIVE] = origin["sources"][
        preparer.WAVE59_RUNNER_RELATIVE
    ]
    with pytest.raises(RuntimeError, match="permits only"):
        preparer._validate_wave59_contract_delta(
            origin, missing_delta, amendment, REPO_ROOT
        )


def test_wave59_dispatch_is_typed_and_does_not_fall_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    marker = {"wave": 59}
    monkeypatch.setattr(
        preparer, "_validate_wave59_recovery_amendment", lambda *_a, **_k: marker
    )
    monkeypatch.setattr(
        preparer,
        "_validate_wave56_recovery_amendment",
        lambda *_a, **_k: pytest.fail("Wave 59 fell through to Wave 56"),
    )
    observed = preparer.validate_recovery_amendment(
        tmp_path / "amendment.json",
        source,
        {"prospective_config": {"schema_version": preparer.WAVE59_CONFIG_SCHEMA}},
        "recovery",
    )
    assert observed is marker


def test_wave59_full_validator_rejects_population_drift_before_origin_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = EXPERIMENTS / "configs/wave59_fresh_hgb_guard_bracket.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = {
        "prospective_config": config,
        "sources": {
            relative: preparer.sha256_file(REPO_ROOT / relative)
            for relative in config["required_execution_sources"]
        },
    }
    amendment = {
        "population_contract": {
            "eligibility_predicate": {
                "is_out_of_catalog": False,
                "calibration_population": "canonical_preserving",
                "filter_rows_before_deduplicating_pair_tokens": True,
            },
            "counts_by_split": {
                split: {
                    "rows": 4992,
                    "total_unique_pair_tokens": 1152,
                    "eligible_unique_pair_tokens": 769,
                    "out_of_catalog_unique_pair_tokens": 384,
                    "noncanonical_unique_pair_tokens": 192,
                    "eligible_intersection_noncanonical_unique_pair_tokens": 192,
                }
                for split in preparer.SPLITS
            },
        }
    }
    monkeypatch.setattr(
        preparer,
        "validate_wave59_repository_recovery_authority",
        lambda *_args, **_kwargs: (amendment, "a" * 64),
    )
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(RuntimeError, match="populations differ"):
        preparer._validate_wave59_recovery_amendment(
            config_path,
            source,
            contract,
            "recovery",
        )


def _real_content_blind_case() -> tuple[dict, dict]:
    freeze = json.loads((FAILED_ORIGIN / preparer.FREEZE_NAME).read_text())
    origin_contract = freeze["contract"]
    execution_contract = copy.deepcopy(origin_contract)
    execution_contract["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    implementation: dict[str, dict[str, str]] = {}
    for label, relative in (
        ("preparer", preparer.PREPARER_RELATIVE),
        ("runner", preparer.WAVE59_RUNNER_RELATIVE),
        ("prospective_test", preparer.WAVE59_PROSPECTIVE_TEST_RELATIVE),
    ):
        new_sha = preparer.sha256_file(REPO_ROOT / relative)
        implementation[label] = {
            "path": relative,
            "old_sha256": origin_contract["sources"][relative],
            "new_sha256": new_sha,
        }
        execution_contract["sources"][relative] = new_sha
    amendment = {
        "implementation": implementation,
        "escrow_origin": {
            "failed_attempt_basename": FAILED_ORIGIN.name,
            "contract_git_commit": origin_contract["git_commit"],
            "contract_sha256": preparer.compact_json_sha256(origin_contract),
            "escrow_sha256": preparer.sha256_file(
                FAILED_ORIGIN / preparer.ESCROW_NAME
            ),
            "pre_generation_freeze_sha256": preparer.sha256_file(
                FAILED_ORIGIN / preparer.FREEZE_NAME
            ),
            "failure_sha256": preparer.sha256_file(FAILED_ORIGIN / "FAILURE.json"),
            "failure_inventory_sha256": preparer.sha256_file(
                FAILED_ORIGIN / "failure_inventory.json"
            ),
            "failure_attestation_sha256": preparer.sha256_file(
                FAILED_ORIGIN / "failure_attestation.json"
            ),
            "benchmark_manifest_sha256": preparer.sha256_file(
                FAILED_ORIGIN / "benchmark/manifest.json"
            ),
            "inventory": preparer.physical_tree_inventory(FAILED_ORIGIN),
        },
    }
    return amendment, execution_contract


def test_wave59_public_parser_rejects_sensitive_paths() -> None:
    for relative in (
        preparer.ESCROW_NAME,
        "benchmark/commitments/semantic.jsonl",
        "benchmark/sealed/train.jsonl",
        "benchmark/sealed/generation_secret.json",
    ):
        with pytest.raises(RuntimeError, match="forbids JSON parsing"):
            preparer._wave59_public_json(FAILED_ORIGIN, relative)


@pytest.mark.skipif(
    not FAILED_ORIGIN.is_dir(),
    reason="the immutable Wave 59 failed origin is required for this probe",
)
def test_real_wave59_content_blind_preflight_never_parses_sensitive_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amendment, execution_contract = _real_content_blind_case()
    sensitive_root = (FAILED_ORIGIN / "benchmark/sealed").resolve()
    commitments_path = (
        FAILED_ORIGIN / "benchmark/commitments/semantic.jsonl"
    ).resolve()
    escrow_path = (FAILED_ORIGIN / preparer.ESCROW_NAME).resolve()
    opaque_descriptors: dict[int, dict[str, object]] = {}
    completed: set[str] = set()
    original_open = os.open
    original_fstat = os.fstat
    original_read = os.read
    original_close = os.close
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    original_json_loads = json.loads

    def sensitive_relative(path: object) -> str | None:
        if not isinstance(path, (str, os.PathLike)):
            return None
        candidate = Path(path).resolve()
        if (
            candidate == escrow_path
            or candidate == commitments_path
            or candidate.is_relative_to(sensitive_root)
        ):
            return str(candidate.relative_to(FAILED_ORIGIN.resolve()))
        return None

    def guarded_open(path, flags, *args, **kwargs):
        relative = sensitive_relative(path)
        descriptor = original_open(path, flags, *args, **kwargs)
        if relative is not None:
            assert sys._getframe(1).f_code.co_name == "_secure_file_record"
            assert flags & getattr(os, "O_NOFOLLOW", 0)
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
            assert opaque_descriptors[descriptor]["fstat"] is True
            opaque_descriptors[descriptor]["reads"] = int(
                opaque_descriptors[descriptor]["reads"]
            ) + 1
        return original_read(descriptor, size)

    def guarded_close(descriptor):
        if descriptor in opaque_descriptors:
            state = opaque_descriptors.pop(descriptor)
            assert state["fstat"] is True and int(state["reads"]) >= 2
            completed.add(str(state["path"]))
        return original_close(descriptor)

    def guarded_builtin_open(path, *args, **kwargs):
        if sensitive_relative(path) is not None:
            raise AssertionError(f"builtins.open reached sensitive path: {path}")
        return original_builtin_open(path, *args, **kwargs)

    def guarded_io_open(path, *args, **kwargs):
        if sensitive_relative(path) is not None:
            raise AssertionError(f"io.open reached sensitive path: {path}")
        return original_io_open(path, *args, **kwargs)

    def guarded_read_text(path: Path, *args, **kwargs):
        if sensitive_relative(path) is not None:
            raise AssertionError(f"read_text reached sensitive path: {path}")
        return original_read_text(path, *args, **kwargs)

    def guarded_read_bytes(path: Path, *args, **kwargs):
        if sensitive_relative(path) is not None:
            raise AssertionError(f"read_bytes reached sensitive path: {path}")
        return original_read_bytes(path, *args, **kwargs)

    def guarded_json_loads(payload, *args, **kwargs):
        if isinstance(payload, (bytes, bytearray, memoryview)):
            raise AssertionError("json.loads received opaque bytes")
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
            preparer.validate_wave59_failed_origin_content_blind(
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
    assert public_contract["git_commit"] == amendment["escrow_origin"][
        "contract_git_commit"
    ]
    assert not opaque_descriptors
    assert {
        preparer.ESCROW_NAME,
        "benchmark/commitments/semantic.jsonl",
        "benchmark/sealed/train.jsonl",
        "benchmark/sealed/val.jsonl",
        "benchmark/sealed/lockbox.jsonl",
        "benchmark/sealed/calibration_null.jsonl",
        "benchmark/sealed/generation_secret.json",
        "benchmark/sealed/identity_secret.json",
        "benchmark/sealed/semantic_commitment_secret.json",
    }.issubset(completed)


def test_wave59_semantic_stage_reinventories_before_parsing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failed = tmp_path / "failed"
    failed.mkdir()
    expected = [{"path": ".", "type": "directory"}]
    events: list[str] = []

    def changed(_path: Path):
        events.append("inventory")
        return [{"path": ".", "type": "directory", "changed": True}]

    def forbidden(_path: Path):
        events.append("parse")
        raise AssertionError("semantic parser ran before identity continuity")

    monkeypatch.setattr(preparer, "physical_tree_inventory", changed)
    monkeypatch.setattr(preparer, "read_escrow", forbidden)
    with pytest.raises(RuntimeError, match="changed before semantic"):
        preparer.validate_wave59_failed_origin_semantic(
            {"population_contract": {"counts_by_split": {}}},
            failed,
            preparer.PUBLIC_KEY,
            expected,
        )
    assert events == ["inventory"]


def test_wave59_normal_execution_rejects_recovery_source_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.loads(
        (
            EXPERIMENTS / "configs/wave59_fresh_hgb_guard_bracket.json"
        ).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(runner, "require_clean_head_source", lambda *_args: None)
    with pytest.raises(RuntimeError, match="execution source drifted"):
        runner.validate_execution_bindings(config, runner.CONFIG_DEFAULT)


def _signed_preparation_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict, dict, Path, Path]:
    private = tmp_path / "private.pem"
    public = tmp_path / "public.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private)],
        check=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
        check=True,
    )
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "TRUSTED_PUBLIC_KEY", public)
    config = {
        "schema_version": preparer.WAVE59_CONFIG_SCHEMA,
        "primary_output": "primary",
        "replay_output": "replay",
        "source_binding": {"bound": "value"},
        "required_execution_sources": [
            "experiments/geometria_proporcional/configs/"
            "wave59_fresh_hgb_guard_bracket.json"
        ],
    }
    root = tmp_path / "primary"
    root.mkdir(mode=0o700)
    for relative, mode in {
        "benchmark": 0o700,
        "benchmark/visible": 0o700,
        "inference": 0o700,
        "inference/logits": 0o700,
        "prepared": 0o700,
        "journals": 0o755,
    }.items():
        (root / relative).mkdir(mode=mode)
        (root / relative).chmod(mode)
    amendment = {"schema_version": preparer.WAVE59_RECOVERY_AMENDMENT_SCHEMA}
    preparer.atomic_write_json(root / "recovery_amendment.json", amendment, mode=0o644)
    provenance = {
        "amendment_sha256": preparer.digest(root / "recovery_amendment.json")
    }
    preparer.atomic_write_json(
        root / "pre_generation_freeze.json",
        {
            "schema_version": "wave56-key-escrow-v1",
            "phase": "keys-escrowed-and-contract-frozen-before-generation",
            "contains_secrets": False,
            "generator_invoked": False,
            "contract": {},
            "key_commitments": {"k": "v"},
        },
        mode=0o644,
    )
    visible = root / "benchmark/visible/train.jsonl"
    visible.write_bytes(b'{"fixture":true}\n')
    visible.chmod(0o600)
    manifest = {
        "schema_version": "wave49-relational-benchmark-v2",
        "generator": "wave49_generator",
        "software": {},
        "files": {
            "visible/train.jsonl": {
                "bytes": visible.stat().st_size,
                "sha256": preparer.digest(visible),
            }
        },
        "counts": {}, "catalog_families": [],
        "out_of_catalog_families": [], "generation_key_commitment": "g",
        "identity_key_commitment": "i", "semantic_commitment_key_commitment": "s",
        "calibration_contract": {}, "semantic_attestation": {},
    }
    preparer.atomic_write_json(root / "benchmark/manifest.json", manifest, mode=0o600)
    preparer.atomic_write_json(root / "config.snapshot.json", config, mode=0o644)
    preparer.atomic_write_json(root / "source_bindings.json", config["source_binding"], mode=0o644)
    inference_file = root / "inference/logits/seed17__train.npz"
    inference_file.write_bytes(b"signed-inference")
    inference_file.chmod(0o600)
    bundle_modes = {
        "gate_fit_bundle.npz": 0o600,
        "gate_select_truth_bundle.npz": 0o600,
        "gate_select_inference_bundle.npz": 0o644,
        "sealed_monitor_truth_bundle.npz": 0o600,
        "sealed_monitor_inference_bundle.npz": 0o644,
    }
    for name, mode in bundle_modes.items():
        path = root / "prepared" / name
        path.write_bytes(name.encode())
        path.chmod(mode)
    inference_hashes = preparer.inventory_hashes(root / "inference")
    bundle_hashes = {
        f"prepared/{name}": preparer.digest(root / "prepared" / name)
        for name in bundle_modes
    }
    generation = {
        "phase": "fresh-benchmark-generated-after-verified-escrow-freeze",
        "execution_mode": "recovery", "escrow_sha256": "0" * 64,
        "key_commitments": {"k": "v"},
        "manifest_sha256": preparer.digest(root / "benchmark/manifest.json"),
        "visible_sha256": {}, "sealed_population_counts": {},
        "sealed_pair_token_counts_total": {}, "sealed_eligible_pair_token_counts": {},
        "sealed_root_owner": 0, "sealed_root_mode": "0700",
        "oracle_materialized": False, "recovery_provenance": provenance,
    }
    preparer.atomic_write_json(root / "generation_receipt.json", generation, mode=0o644)
    freeze = {
        "schema_version": preparer.WAVE59_CONFIG_SCHEMA,
        "phase": "prepared-with-blind-inference-before-any-oracle",
        "git_commit": "c" * 40,
        "config_sha256": preparer.digest(root / "config.snapshot.json"),
        "prospective_config": config, "sources": {}, "upstream": {},
        "historical_preflight": {}, "source_bindings": config["source_binding"],
        "key_commitments": {"k": "v"},
        "benchmark_manifest_sha256": preparer.digest(root / "benchmark/manifest.json"),
        "protocol_config_sha256": "1" * 64, "visible_sha256": {},
        "staging_input_hashes": {}, "inference_runtime_hashes": {},
        "checkpoint_receipts": [], "inference_hashes": inference_hashes,
        "inference_uid": 65534, "inference_gid": 65534,
        "negative_truth_probe": {}, "oracle_materialized": False,
        "authorized_labels_present": False, "bundles_present": True,
        "fit_operations": False, "physical_splits": {},
        "prepared_bundle_hashes": bundle_hashes, "recovery_provenance": provenance,
    }
    preparer.atomic_write_json(root / "preparation_freeze.json", freeze, mode=0o644)
    preparation_receipt = {
        "phase": "wave59-stage1-preparation-complete", "timestamp_utc": "test",
        "execution_mode": "recovery",
        "preparation_freeze_sha256": preparer.digest(root / "preparation_freeze.json"),
        "generation_receipt_sha256": preparer.digest(root / "generation_receipt.json"),
        "replay_exact": None, "next_state": "PREPARED",
        "recovery_provenance": provenance, "superseded_output": None,
        "coordinator_budget": {},
    }
    preparer.atomic_write_json(root / "preparation_receipt.json", preparation_receipt, mode=0o644)
    preparer.atomic_write_json(
        root / "journals/prepare.json",
        {
            "schema_version": "wave59-phase-journal-v1", "phase": "prepare",
            "status": "PREPARED", "execution_mode": "recovery",
            "preparation_freeze_sha256": preparer.digest(root / "preparation_freeze.json"),
            "prepared_bundle_hashes": bundle_hashes,
            "maximum_truth_materialized": "prepared_all_splits_root_only",
            "next_state": "PREPARED",
        },
        mode=0o644,
    )
    preparer.publish_wave59_preparation_attestation(root, "recovery", private, public)
    return root, config, amendment, public, private


def _convert_signed_package_to_fresh_successor(
    root: Path,
    config: dict,
    execution_mode: str,
    private: Path,
    public: Path,
) -> None:
    config = json.loads(json.dumps(config))
    config["required_execution_sources"] = [
        "experiments/geometria_proporcional/configs/"
        "wave59_fresh_hgb_guard_bracket_replay_normalized.json"
    ]
    config["primary_output"] = "primary"
    config["replay_output"] = "replay"
    (root / "recovery_amendment.json").unlink(missing_ok=True)
    preparer.atomic_write_json(root / "config.snapshot.json", config, mode=0o644)

    generation = json.loads((root / "generation_receipt.json").read_text())
    generation["execution_mode"] = execution_mode
    generation.pop("recovery_provenance", None)
    preparer.atomic_write_json(root / "generation_receipt.json", generation, mode=0o644)

    freeze = json.loads((root / "preparation_freeze.json").read_text())
    freeze.pop("recovery_provenance", None)
    freeze["prospective_config"] = config
    freeze["config_sha256"] = preparer.digest(root / "config.snapshot.json")
    preparer.atomic_write_json(root / "preparation_freeze.json", freeze, mode=0o644)

    receipt = json.loads((root / "preparation_receipt.json").read_text())
    receipt["execution_mode"] = execution_mode
    receipt["generation_receipt_sha256"] = preparer.digest(
        root / "generation_receipt.json"
    )
    receipt["preparation_freeze_sha256"] = preparer.digest(
        root / "preparation_freeze.json"
    )
    receipt["replay_exact"] = True if execution_mode == "replay" else None
    receipt.pop("recovery_provenance", None)
    preparer.atomic_write_json(root / "preparation_receipt.json", receipt, mode=0o644)

    journal = json.loads((root / "journals/prepare.json").read_text())
    journal["execution_mode"] = execution_mode
    journal["preparation_freeze_sha256"] = preparer.digest(
        root / "preparation_freeze.json"
    )
    preparer.atomic_write_json(root / "journals/prepare.json", journal, mode=0o644)
    preparer.publish_wave59_preparation_attestation(
        root, execution_mode, private, public
    )


def test_wave59_signed_preparation_package_accepts_authentic_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, amendment, _, _ = _signed_preparation_fixture(tmp_path, monkeypatch)
    assert runner.validate_signed_preparation_package(
        root, config, amendment, preparer.digest(root / "recovery_amendment.json")
    ) == "recovery"
    payload = json.loads((root / "preparation_attestation.json").read_text())["payload"]
    assert payload["schema_version"] == preparer.WAVE59_PREPARATION_ATTESTATION_SCHEMA
    assert "recovery_provenance" in payload


def test_wave59_signed_preparation_package_accepts_fresh_primary_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, config, _, public, private = _signed_preparation_fixture(
        tmp_path, monkeypatch
    )
    _convert_signed_package_to_fresh_successor(
        primary, config, "primary", private, public
    )
    successor = json.loads((primary / "config.snapshot.json").read_text())
    assert runner.validate_signed_preparation_package(primary, successor) == "primary"
    payload = json.loads((primary / "preparation_attestation.json").read_text())["payload"]
    assert payload["schema_version"] == preparer.WAVE59_FRESH_PREPARATION_ATTESTATION_SCHEMA
    assert "recovery_provenance" not in payload
    assert "recovery_amendment.json" not in payload["records"]

    replay = tmp_path / "replay"
    shutil.copytree(primary, replay)
    _convert_signed_package_to_fresh_successor(
        replay, successor, "replay", private, public
    )
    assert runner.validate_signed_preparation_package(replay, successor) == "replay"

    primary_generation, primary_receipt = runner._linked_preparation_receipts(primary)
    replay_generation, replay_receipt = runner._linked_preparation_receipts(replay)
    runner._verified_preparation_attestation_invariants(primary)
    runner._verified_preparation_attestation_invariants(replay)
    assert runner._normalized_preparation_receipts(
        primary_generation, primary_receipt
    ) == runner._normalized_preparation_receipts(replay_generation, replay_receipt)
    assert preparer.digest(primary / "generation_receipt.json") != preparer.digest(
        replay / "generation_receipt.json"
    )


def test_wave59_fresh_preparation_rejects_recovery_mix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, amendment, public, private = _signed_preparation_fixture(
        tmp_path, monkeypatch
    )
    _convert_signed_package_to_fresh_successor(
        root, config, "primary", private, public
    )
    preparer.atomic_write_json(
        root / "recovery_amendment.json", amendment, mode=0o644
    )
    with pytest.raises(RuntimeError, match="mixed recovery|contains a recovery"):
        preparer.publish_wave59_preparation_attestation(
            root, "primary", private, public
        )


def test_wave59_normalized_preparation_rejects_broken_raw_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, _, public, private = _signed_preparation_fixture(
        tmp_path, monkeypatch
    )
    _convert_signed_package_to_fresh_successor(
        root, config, "primary", private, public
    )
    receipt = json.loads((root / "preparation_receipt.json").read_text())
    receipt["generation_receipt_sha256"] = "0" * 64
    preparer.atomic_write_json(root / "preparation_receipt.json", receipt, mode=0o644)
    with pytest.raises(RuntimeError, match="broken local generation link"):
        runner._linked_preparation_receipts(root)


@pytest.mark.parametrize("mutation", ("generation_semantic", "preparation_semantic", "extra_key"))
def test_wave59_normalized_preparation_preserves_nonoperational_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    primary, config, _, public, private = _signed_preparation_fixture(
        tmp_path, monkeypatch
    )
    _convert_signed_package_to_fresh_successor(
        primary, config, "primary", private, public
    )
    replay = tmp_path / "replay"
    shutil.copytree(primary, replay)
    successor = json.loads((primary / "config.snapshot.json").read_text())
    _convert_signed_package_to_fresh_successor(
        replay, successor, "replay", private, public
    )
    if mutation == "generation_semantic":
        generation = json.loads((replay / "generation_receipt.json").read_text())
        generation["manifest_sha256"] = "f" * 64
        preparer.atomic_write_json(
            replay / "generation_receipt.json", generation, mode=0o644
        )
        receipt = json.loads((replay / "preparation_receipt.json").read_text())
        receipt["generation_receipt_sha256"] = preparer.digest(
            replay / "generation_receipt.json"
        )
        preparer.atomic_write_json(
            replay / "preparation_receipt.json", receipt, mode=0o644
        )
        preparer.publish_wave59_preparation_attestation(
            replay, "replay", private, public
        )
    elif mutation == "preparation_semantic":
        receipt = json.loads((replay / "preparation_receipt.json").read_text())
        receipt["next_state"] = "DIFFERENT"
        preparer.atomic_write_json(
            replay / "preparation_receipt.json", receipt, mode=0o644
        )
        preparer.publish_wave59_preparation_attestation(
            replay, "replay", private, public
        )
    else:
        receipt = json.loads((replay / "preparation_receipt.json").read_text())
        receipt["unexpected"] = True
        preparer.atomic_write_json(
            replay / "preparation_receipt.json", receipt, mode=0o644
        )
        with pytest.raises(RuntimeError, match="keys drifted"):
            runner._linked_preparation_receipts(replay)
        return

    primary_pair = runner._linked_preparation_receipts(primary)
    replay_pair = runner._linked_preparation_receipts(replay)
    runner._verified_preparation_attestation_invariants(primary)
    runner._verified_preparation_attestation_invariants(replay)
    assert runner._normalized_preparation_receipts(
        *primary_pair
    ) != runner._normalized_preparation_receipts(*replay_pair)


def test_wave59_compare_runs_accepts_real_fresh_primary_replay_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer._validate_wave59_antecedent_sentinels(REPO_ROOT)
    source_primary = (
        REPO_ROOT
        / "data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_v1"
    )
    source_replay = (
        REPO_ROOT
        / "data/geometria_proporcional/"
        "wave59_fresh_hgb_guard_bracket_v1_replay.failed_20260905T102929791142Z"
    )
    if not source_primary.is_dir() or not source_replay.is_dir():
        pytest.skip("Wave 59 primary/replay diagnostic antecedent is unavailable")
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    shutil.copytree(source_primary, primary, copy_function=os.link)
    shutil.copytree(source_replay, replay, copy_function=os.link)

    private = tmp_path / "private.pem"
    public = tmp_path / "public.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private)],
        check=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
        check=True,
    )
    monkeypatch.setattr(runner, "TRUSTED_PUBLIC_KEY", public)
    config = json.loads((primary / "config.snapshot.json").read_text())
    _convert_signed_package_to_fresh_successor(
        primary, config, "primary", private, public
    )
    successor = json.loads((primary / "config.snapshot.json").read_text())
    _convert_signed_package_to_fresh_successor(
        replay, successor, "replay", private, public
    )

    comparison = runner.compare_runs(replay, primary)
    assert comparison["all_exact"] is True
    assert comparison["operational_semantic"]["generation_receipt.json"] is True
    assert comparison["operational_semantic"]["preparation_receipt.json"] is True
    assert preparer.digest(primary / "generation_receipt.json") != preparer.digest(
        replay / "generation_receipt.json"
    )
    preparer._validate_wave59_antecedent_sentinels(REPO_ROOT)


@pytest.mark.parametrize(
    "mutation",
    [
        "world_writable", "alternate_root", "freeze_extra", "receipt_unbound",
        "journal_unbound", "provenance_false", "bundle_mode", "inference_changed",
        "benchmark_changed", "benchmark_extra", "benchmark_symlink", "benchmark_fifo",
        "benchmark_empty_dir", "caller_symlink",
        "missing_attestation", "bad_signature",
    ],
)
def test_wave59_signed_preparation_package_rejects_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root, config, amendment, _, _ = _signed_preparation_fixture(tmp_path, monkeypatch)
    if mutation == "world_writable":
        root.chmod(0o777)
    elif mutation == "alternate_root":
        alternate = root.with_name("forged-root")
        root.rename(alternate)
        root = alternate
    elif mutation == "caller_symlink":
        alias = root.with_name("caller-alias")
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    elif mutation == "bundle_mode":
        (root / "prepared/gate_select_inference_bundle.npz").chmod(0o666)
    elif mutation == "inference_changed":
        (root / "inference/logits/seed17__train.npz").write_bytes(b"changed")
    elif mutation == "benchmark_changed":
        (root / "benchmark/visible/train.jsonl").write_bytes(b"changed")
    elif mutation == "benchmark_extra":
        (root / "benchmark/extra.bin").write_bytes(b"extra")
    elif mutation == "benchmark_symlink":
        visible = root / "benchmark/visible/train.jsonl"
        visible.unlink()
        visible.symlink_to(root / "benchmark/manifest.json")
    elif mutation == "benchmark_fifo":
        os.mkfifo(root / "benchmark/extra.pipe")
    elif mutation == "benchmark_empty_dir":
        (root / "benchmark/empty").mkdir()
    elif mutation == "missing_attestation":
        (root / "preparation_attestation.json").unlink()
    elif mutation == "bad_signature":
        receipt = json.loads((root / "preparation_attestation.json").read_text())
        receipt["signature_base64"] = "AAAA"
        preparer.atomic_write_json(root / "preparation_attestation.json", receipt, mode=0o644)
    elif mutation == "provenance_false":
        freeze = json.loads((root / "preparation_freeze.json").read_text())
        freeze["recovery_provenance"]["amendment_sha256"] = "f" * 64
        preparer.atomic_write_json(root / "preparation_freeze.json", freeze, mode=0o644)
    else:
        relative = {
            "freeze_extra": "preparation_freeze.json",
            "receipt_unbound": "preparation_receipt.json",
            "journal_unbound": "journals/prepare.json",
        }[mutation]
        payload = json.loads((root / relative).read_text())
        payload["forged"] = True
        preparer.atomic_write_json(root / relative, payload, mode=0o644)
    with pytest.raises((RuntimeError, PermissionError, FileNotFoundError)):
        runner.validate_signed_preparation_package(
            root, config, amendment, preparer.digest(root / "recovery_amendment.json")
        )


@pytest.mark.parametrize("failure_kind", ["sign", "publication"])
def test_wave59_failure_archive_retires_canonical_root_without_signing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str
) -> None:
    run_dir = tmp_path / "canonical"
    run_dir.mkdir(mode=0o700)
    if failure_kind == "sign":
        monkeypatch.setattr(
            runner,
            "sign_attestation",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("sign failed")),
        )
    else:
        monkeypatch.setattr(
            runner,
            "sign_attestation",
            lambda *_args, **_kwargs: {"synthetic": "attestation"},
        )
        original_write = runner.write_json

        def fail_attestation_only(path: Path, payload: object, mode: int = 0o444) -> None:
            if path.name == "failure_attestation.json":
                raise OSError("attestation publication failed")
            original_write(path, payload, mode=mode)

        monkeypatch.setattr(runner, "write_json", fail_attestation_only)
    archived = runner.archive_failed_attempt(
        run_dir,
        RuntimeError("original failure"),
        run_role="primary",
        recovery_context=True,
        attestation_private_key=tmp_path / "missing-private.pem",
        trusted_public_key=tmp_path / "missing-public.pem",
    )
    assert not run_dir.exists()
    assert archived.is_dir()
    marker = json.loads((archived / "failure_attestation_error.json").read_text())
    assert marker["status"] == "UNATTESTED_SIGNING_FAILURE"
    assert marker["authoritative"] is False
    inventory = json.loads((archived / "failure_inventory.json").read_text())
    assert inventory["failure_records"] == [
        "FAILURE.json",
        "failure_inventory.json",
        "failure_attestation_error.json",
    ]


def test_wave59_fresh_replay_late_failure_is_not_archived_as_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private.pem"
    public = tmp_path / "public.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private)],
        check=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
        check=True,
    )
    config_path = tmp_path / "successor.json"
    preparer.atomic_write_json(
        config_path,
        {"schema_version": preparer.WAVE59_CONFIG_SCHEMA},
        mode=0o644,
    )
    replay_source = tmp_path / "primary"
    replay_source.mkdir()
    output = tmp_path / "fresh-replay"
    args = SimpleNamespace(
        config=config_path,
        output_dir=output,
        replay_secrets_from=replay_source,
        recovery_secrets_from=None,
        recovery_amendment=None,
        attestation_private_key=private,
        force=False,
    )

    @contextmanager
    def synthetic_budget(_config: dict):
        yield {"synthetic": True}

    def publish_then_fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected after fresh replay preparation")

    def materialize_fresh_replay(*_args: object, **_kwargs: object) -> None:
        output.mkdir(mode=0o700)
        preparer.atomic_write_json(
            output / "preparation_receipt.json",
            {"execution_mode": "replay"},
            mode=0o644,
        )

    monkeypatch.setattr(preparer, "parse_args", lambda: args)
    monkeypatch.setattr(preparer, "validate_invocation", lambda *_args: "replay")
    monkeypatch.setattr(preparer, "preparation_preflight", lambda *_args: {})
    monkeypatch.setattr(preparer, "validate_reused_escrow", lambda *_args: {})
    monkeypatch.setattr(
        preparer, "run_preparation_transaction", materialize_fresh_replay
    )
    monkeypatch.setattr(preparer, "wave59_coordinator_budget", synthetic_budget)
    monkeypatch.setattr(
        preparer, "publish_wave59_preparation_attestation", publish_then_fail
    )
    monkeypatch.setattr(preparer, "PUBLIC_KEY", public)

    with pytest.raises(RuntimeError, match="injected after fresh replay preparation"):
        preparer.main()

    archived = list(tmp_path.glob("fresh-replay.failed_*"))
    assert len(archived) == 1
    failure = json.loads((archived[0] / "FAILURE.json").read_text(encoding="utf-8"))
    assert failure["run_role"] == "replay"
    assert failure["recovery_context"] is False
    assert not (archived[0] / "recovery_amendment.json").exists()
    inventory = json.loads(
        (archived[0] / "failure_inventory.json").read_text(encoding="utf-8")
    )
    inventoried = {record["path"] for record in inventory["records"]}
    assert "recovery_amendment.json" not in inventoried
    assert "recovery_amendment.json" not in inventory[
        "missing_required_through_last_journal"
    ]


def test_wave59_publisher_rejects_benchmark_root_symlink_before_signing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    external = tmp_path / "external-benchmark"
    external.mkdir()
    preparer.atomic_write_json(external / "manifest.json", {"files": {}}, mode=0o600)
    (root / "benchmark").symlink_to(external, target_is_directory=True)
    called = False

    def forbidden_sign(*_args: object, **_kwargs: object) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(preparer, "sign_attestation", forbidden_sign)
    with pytest.raises(RuntimeError, match="physical directory"):
        preparer.publish_wave59_preparation_attestation(
            root, "recovery", tmp_path / "private.pem", tmp_path / "public.pem"
        )
    assert called is False
