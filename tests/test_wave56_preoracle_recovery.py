from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from geometria_proporcional.wave49_schema import sha256_file, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP_PATH = REPO_ROOT / "experiments/geometria_proporcional/prepare_wave56_fresh.py"
PROSPECTIVE_TEST_PATH = REPO_ROOT / "tests/test_wave56_prospective.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load_module(PREP_PATH, "wave56_preoracle_recovery_test")
prospective = load_module(PROSPECTIVE_TEST_PATH, "wave56_prospective_recovery_helpers")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def commit_all(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True, capture_output=True)
    return git(repo, "rev-parse", "HEAD")


def write_canonical(path: Path, payload: object, mode: int = 0o644) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
    path.write_bytes(encoded)
    path.chmod(mode)
    return prep.sha256_bytes(encoded)


def population_rows() -> list[dict[str, object]]:
    return [
        {
            "pair_token": "eligible-overlap",
            "is_out_of_catalog": False,
            "calibration_population": "canonical_preserving",
        },
        {
            "pair_token": "eligible-overlap",
            "is_out_of_catalog": False,
            "calibration_population": "origin_translation_break",
        },
        {
            "pair_token": "eligible-only",
            "is_out_of_catalog": False,
            "calibration_population": "canonical_preserving",
        },
        {
            "pair_token": "out-of-catalog",
            "is_out_of_catalog": True,
            "calibration_population": "canonical_preserving",
        },
    ]


def test_population_filter_precedes_dedup_and_preserves_overlap(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed.jsonl"
    write_jsonl(sealed, population_rows())

    assert prep.sealed_population_counts(sealed) == {
        "rows": 4,
        "total_unique_pair_tokens": 3,
        "eligible_unique_pair_tokens": 2,
        "out_of_catalog_unique_pair_tokens": 1,
        "noncanonical_unique_pair_tokens": 1,
        "eligible_intersection_noncanonical_unique_pair_tokens": 1,
    }


def test_recovery_amendment_cannot_authorize_a_fresh_primary(tmp_path: Path) -> None:
    config = prospective.prospective_config()
    config["output_parent_relative"] = "."
    args = SimpleNamespace(
        replay_secrets_from=None,
        recovery_secrets_from=None,
        recovery_amendment=tmp_path / "amendment.json",
        reference_dir=None,
        force=False,
    )
    with pytest.raises(ValueError, match="cannot authorize a fresh primary"):
        prep.validate_invocation(
            args,
            tmp_path / config["primary_output_name"],
            config,
            repo_root=tmp_path,
        )


def test_physical_inventory_is_closed_and_rejects_links_and_special_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "origin"
    nested = root / "benchmark"
    nested.mkdir(parents=True)
    (root / "escrow.json").write_text("escrow", encoding="utf-8")
    (nested / "manifest.json").write_text("manifest", encoding="utf-8")

    baseline = prep.physical_tree_inventory(root)
    assert [entry["path"] for entry in baseline] == [
        ".",
        "benchmark",
        "benchmark/manifest.json",
        "escrow.json",
    ]

    link = root / "link"
    link.symlink_to(root / "escrow.json")
    with pytest.raises(RuntimeError, match="symlink"):
        prep.physical_tree_inventory(root)
    link.unlink()

    fifo = root / "fifo"
    os.mkfifo(fifo)
    try:
        with pytest.raises(RuntimeError, match="special file"):
            prep.physical_tree_inventory(root)
    finally:
        fifo.unlink()

    (root / "extra").write_text("extra", encoding="utf-8")
    assert prep.physical_tree_inventory(root) != baseline


def test_read_escrow_rejects_symlink_even_when_target_is_valid(tmp_path: Path) -> None:
    target_dir = tmp_path / "target"
    source_dir = tmp_path / "source"
    target_dir.mkdir()
    source_dir.mkdir()
    escrow = prep.make_escrow(
        minimal_contract("origin", "old"),
        (b"g" * 32, b"i" * 32, b"c" * 32),
    )
    prep.atomic_write_json(target_dir / prep.ESCROW_NAME, escrow, mode=0o600)
    (source_dir / prep.ESCROW_NAME).symlink_to(target_dir / prep.ESCROW_NAME)
    with pytest.raises(PermissionError, match="physical regular file"):
        prep.read_escrow(source_dir)


def minimal_contract(commit: str, preparer_sha256: str) -> dict[str, object]:
    return {
        "git_commit": commit,
        "config_sha256": "config",
        "prospective_config": {"frozen": True},
        "sources": {"prep.py": preparer_sha256},
        "upstream": [{"sha256": "upstream"}],
        "historical_preflight": {"status": "PASS"},
        "source_bindings": {"binding": "fixed"},
    }


def build_provenance_repo(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bad_preparer_hash: bool = False,
    extra_implementation_path: bool = False,
    bad_audit_fields: bool = False,
) -> SimpleNamespace:
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Wave 56 Test"], cwd=repo, check=True)

    (repo / "prep.py").write_text("old\n", encoding="utf-8")
    (repo / "plan.md").write_text("approved recovery plan\n", encoding="utf-8")
    plan_commit = commit_all(repo, "P")
    old_sha = sha256_file(repo / "prep.py")
    plan_sha = sha256_file(repo / "plan.md")

    (repo / "prep.py").write_text("new\n", encoding="utf-8")
    (repo / "test.py").write_text("def test_recovery():\n    assert True\n", encoding="utf-8")
    if extra_implementation_path:
        (repo / "unrelated.txt").write_text("not allowed\n", encoding="utf-8")
    implementation_commit = commit_all(repo, "I")
    new_sha = sha256_file(repo / "prep.py")
    test_sha = sha256_file(repo / "test.py")

    audit_text = (
        f"**Implementation commit:** `{implementation_commit}`\n"
        f"**Preparer SHA-256:** `{new_sha}`\n"
        f"**Test SHA-256:** `{test_sha}`\n"
        "**Result:** `PASS`\n"
    )
    if bad_audit_fields:
        audit_text = "**Result:** `PASS`\n"
    (repo / "audit.md").write_text(audit_text, encoding="utf-8")
    commit_all(repo, "A")
    audit_sha = sha256_file(repo / "audit.md")

    source = root / "failed"
    source.mkdir()
    old_contract = minimal_contract(plan_commit, old_sha)
    escrow = prep.make_escrow(old_contract, (b"g" * 32, b"i" * 32, b"c" * 32))
    prep.atomic_write_json(source / prep.ESCROW_NAME, escrow, mode=0o600)

    counts = {
        "rows": 4,
        "total_unique_pair_tokens": 3,
        "eligible_unique_pair_tokens": 2,
        "out_of_catalog_unique_pair_tokens": 1,
        "noncanonical_unique_pair_tokens": 1,
        "eligible_intersection_noncanonical_unique_pair_tokens": 1,
    }
    amendment = {
        "schema_version": prep.RECOVERY_AMENDMENT_SCHEMA,
        "status": "APPROVED_PREORACLE_RECOVERY",
        "plan": {"path": "plan.md", "sha256": plan_sha},
        "implementation": {
            "commit": implementation_commit,
            "preparer": {
                "path": "prep.py",
                "old_sha256": old_sha,
                "new_sha256": "0" * 64 if bad_preparer_hash else new_sha,
            },
            "test": {"path": "test.py", "sha256": test_sha},
        },
        "implementation_audit": {"path": "audit.md", "sha256": audit_sha},
        "final_audit_path": "final.md",
        "escrow_origin": {
            "failed_attempt_basename": source.name,
            "contract_git_commit": plan_commit,
            "contract_sha256": prep.compact_json_sha256(old_contract),
            "escrow_sha256": sha256_file(source / prep.ESCROW_NAME),
            "pre_generation_freeze_sha256": "freeze",
            "failure_sha256": "failure",
            "benchmark_manifest_sha256": "manifest",
            "inventory": [],
        },
        "population_contract": {
            "eligibility_predicate": {
                "is_out_of_catalog": False,
                "calibration_population": "canonical_preserving",
                "filter_rows_before_deduplicating_pair_tokens": True,
            },
            "counts_by_split": {split: counts for split in prep.SPLITS},
        },
        "assertions": {
            "no_redraw": True,
            "no_inference_in_origin": True,
            "no_oracle_in_origin": True,
            "no_labels_in_origin": True,
        },
    }
    amendment_sha = write_canonical(repo / "amendment.json", amendment)
    amendment_commit = commit_all(repo, "J")
    (repo / "final.md").write_text(
        f"**Audited package commit:** `{amendment_commit}`\n"
        f"**Amendment SHA-256:** `{amendment_sha}`\n"
        "**Result:** `PASS`\n",
        encoding="utf-8",
    )
    final_commit = commit_all(repo, "F")

    monkeypatch.setattr(prep, "RECOVERY_AMENDMENT_RELATIVE", "amendment.json")
    monkeypatch.setattr(prep, "PREPARER_RELATIVE", "prep.py")
    monkeypatch.setattr(prep, "RECOVERY_TEST_RELATIVE", "test.py")
    monkeypatch.setattr(
        prep,
        "validate_failed_recovery_origin",
        lambda _amendment, _parent, _key: (source, []),
    )
    execution_contract = minimal_contract(final_commit, new_sha)
    return SimpleNamespace(
        repo=repo,
        source=source,
        amendment=repo / "amendment.json",
        execution_contract=execution_contract,
        implementation_commit=implementation_commit,
    )


def test_recovery_amendment_enforces_git_dag_and_blob_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = build_provenance_repo(tmp_path, monkeypatch)
    context = prep.validate_recovery_amendment(
        case.amendment,
        case.source,
        case.execution_contract,
        "recovery",
        repo_root=case.repo,
        trusted_public_key_path=tmp_path / "unused.pem",
    )
    assert context["implementation_commit"] == case.implementation_commit
    assert context["repo_root"] == case.repo


def test_recovery_amendment_rejects_dirty_artifact_and_symlinked_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = build_provenance_repo(tmp_path, monkeypatch)
    original = case.amendment.read_text(encoding="utf-8")
    case.amendment.write_text(original + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from HEAD"):
        prep.validate_recovery_amendment(
            case.amendment,
            case.source,
            case.execution_contract,
            "recovery",
            repo_root=case.repo,
            trusted_public_key_path=tmp_path / "unused.pem",
        )
    case.amendment.write_text(original, encoding="utf-8")
    source_link = tmp_path / "failed-link"
    source_link.symlink_to(case.source, target_is_directory=True)
    with pytest.raises(RuntimeError, match="physical directory"):
        prep.validate_recovery_amendment(
            case.amendment,
            source_link,
            case.execution_contract,
            "recovery",
            repo_root=case.repo,
            trusted_public_key_path=tmp_path / "unused.pem",
        )


@pytest.mark.parametrize(
    ("builder_kwargs", "message"),
    [
        ({"bad_preparer_hash": True}, "preparer blob differs"),
        ({"extra_implementation_path": True}, "outside preparer and recovery test"),
        ({"bad_audit_fields": True}, "does not attest"),
    ],
)
def test_recovery_amendment_rejects_broken_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    builder_kwargs: dict[str, bool],
    message: str,
) -> None:
    case = build_provenance_repo(tmp_path, monkeypatch, **builder_kwargs)
    with pytest.raises(RuntimeError, match=message):
        prep.validate_recovery_amendment(
            case.amendment,
            case.source,
            case.execution_contract,
            "recovery",
            repo_root=case.repo,
            trusted_public_key_path=tmp_path / "unused.pem",
        )


def test_contract_delta_rejects_every_change_beyond_the_preparer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Wave 56 Test"], cwd=repo, check=True)
    (repo / "tracked").write_text("x", encoding="utf-8")
    head = commit_all(repo, "head")
    monkeypatch.setattr(prep, "PREPARER_RELATIVE", "prep.py")
    old = minimal_contract("origin", "old")
    old["sources"]["other.py"] = "same"
    new = copy.deepcopy(old)
    new["git_commit"] = head
    new["sources"]["prep.py"] = "new"
    amendment = {
        "escrow_origin": {
            "contract_sha256": prep.compact_json_sha256(old),
            "contract_git_commit": "origin",
        },
        "implementation": {
            "preparer": {"path": "prep.py", "old_sha256": "old", "new_sha256": "new"}
        },
    }
    prep._validate_contract_delta(old, new, amendment, repo)
    new["sources"]["other.py"] = "changed"
    with pytest.raises(RuntimeError, match="only the preparer source delta"):
        prep._validate_contract_delta(old, new, amendment, repo)


@pytest.mark.skipif(
    not prospective.PHYSICAL_TEST_REQUIREMENTS,
    reason="physical recovery/replay requires root and setpriv",
)
def test_amended_recovery_reuses_keys_and_replays_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = prospective.make_full_preparation_inputs(tmp_path)
    inputs.config["fresh_benchmark"]["expected_eligible_pair_tokens_per_split"] = 80
    prospective.write_json(inputs.config_path, inputs.config)
    inputs.contract["config_sha256"] = sha256_file(inputs.config_path)
    inputs.contract["prospective_config"] = inputs.config
    args = SimpleNamespace(
        wave51_dir=inputs.wave51,
        attestation_private_key=inputs.private_key,
        reference_dir=None,
    )
    failed_output = tmp_path / "failed-source"
    keys = (b"g" * 32, b"i" * 32, b"c" * 32)
    with pytest.raises(RuntimeError, match="pair-token count differs"):
        prep.run_preparation_transaction(
            args,
            failed_output,
            inputs.config_path,
            inputs.config,
            "primary",
            inputs.contract,
            None,
            force=False,
            keys_override=keys,
            protocol_override=inputs.protocol,
            trusted_public_key_path=inputs.public_key,
        )
    failed = next(tmp_path.glob("failed-source.failed_*"))
    population = {
        split: prep.sealed_population_counts(failed / "benchmark/sealed" / f"{split}.jsonl")
        for split in prep.SPLITS
    }
    origin = {
        "failed_attempt_basename": failed.name,
        "contract_git_commit": inputs.contract["git_commit"],
        "contract_sha256": prep.compact_json_sha256(inputs.contract),
        "escrow_sha256": sha256_file(failed / prep.ESCROW_NAME),
        "pre_generation_freeze_sha256": sha256_file(failed / prep.FREEZE_NAME),
        "failure_sha256": sha256_file(failed / "FAILURE.json"),
        "benchmark_manifest_sha256": sha256_file(failed / "benchmark/manifest.json"),
        "inventory": prep.physical_tree_inventory(failed),
    }
    amendment = {
        "escrow_origin": origin,
        "population_contract": {"counts_by_split": population},
    }
    context = {
        "amendment": amendment,
        "amendment_sha256": prep.canonical_json_sha256(amendment),
        "amendment_path": "synthetic-amendment.json",
        "implementation_commit": "synthetic-I",
        "implementation_audit": {"path": "synthetic-A", "sha256": "a"},
        "final_audit": {"path": "synthetic-F", "sha256": "f"},
        "escrow_origin_contract_sha256": origin["contract_sha256"],
        "failed_attempt": failed,
        "failed_attempt_basename": failed.name,
        "benchmark_manifest_sha256": origin["benchmark_manifest_sha256"],
        "origin_inventory": origin["inventory"],
        "repo_root": REPO_ROOT,
    }
    escrow = prep.validate_reused_escrow(failed, inputs.contract, context)
    monkeypatch.setattr(
        prep.secrets,
        "token_bytes",
        lambda _size: (_ for _ in ()).throw(AssertionError("recovery attempted a redraw")),
    )

    def mutate_origin_after_generation(*generation_args, **generation_kwargs):
        result = prep.generate_benchmark(*generation_args, **generation_kwargs)
        (failed / "late-extra").write_text("mutation", encoding="utf-8")
        return result

    with pytest.raises(RuntimeError, match="physical whitelist differs"):
        prep.run_preparation_transaction(
            args,
            tmp_path / "mutated-origin-attempt",
            inputs.config_path,
            inputs.config,
            "recovery",
            inputs.contract,
            escrow,
            force=False,
            recovery_context=context,
            protocol_override=inputs.protocol,
            trusted_public_key_path=inputs.public_key,
            generation_fn=mutate_origin_after_generation,
        )
    (failed / "late-extra").unlink()
    assert prep.physical_tree_inventory(failed) == context["origin_inventory"]

    primary = tmp_path / "recovered-primary"
    prep.run_preparation_transaction(
        args,
        primary,
        inputs.config_path,
        inputs.config,
        "recovery",
        inputs.contract,
        escrow,
        force=False,
        recovery_context=context,
        protocol_override=inputs.protocol,
        trusted_public_key_path=inputs.public_key,
    )
    assert prep.keys_from_escrow(prep.read_escrow(primary)) == keys
    assert sha256_file(primary / prep.ESCROW_NAME) == origin["escrow_sha256"]
    assert sha256_file(primary / prep.FREEZE_NAME) == origin["pre_generation_freeze_sha256"]
    assert sha256_file(primary / prep.RECOVERY_AMENDMENT_COPY_NAME) == context["amendment_sha256"]
    assert json.loads((primary / "preparation_receipt.json").read_text())["next_state"] == "PREPARED"

    replay = tmp_path / "replay"
    replay_args = SimpleNamespace(
        wave51_dir=inputs.wave51,
        attestation_private_key=inputs.private_key,
        reference_dir=primary,
    )
    replay_escrow = prep.validate_reused_escrow(primary, inputs.contract, context)
    prep.run_preparation_transaction(
        replay_args,
        replay,
        inputs.config_path,
        inputs.config,
        "replay",
        inputs.contract,
        replay_escrow,
        force=False,
        recovery_context=context,
        protocol_override=inputs.protocol,
        trusted_public_key_path=inputs.public_key,
    )
    replay_receipt = json.loads((replay / "preparation_replay.json").read_text())
    assert replay_receipt["all_exact"] is True
    assert all(replay_receipt["checks"].values())
