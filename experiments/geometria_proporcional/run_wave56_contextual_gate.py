#!/usr/bin/env python3
"""Transactional root coordinator for prospective Wave 56 Stage 1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_STAGE1_PROSPECTIVE_IMPLEMENTATION_PLAN.md"
CONFIG_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/wave56_contextual_gate_fresh.json"
WORKER_PATH = REPO_ROOT / "experiments/geometria_proporcional/_wave56_phase_worker.py"
MATERIALIZER_PATH = REPO_ROOT / "experiments/geometria_proporcional/_wave56_oracle_materializer.py"
RETROSPECTIVE_PATH = REPO_ROOT / "experiments/geometria_proporcional/run_wave56_retrospective.py"
PRIMITIVES_PATH = REPO_ROOT / "src/geometria_proporcional/wave56_contextual_gate.py"
PHASE_RUNTIME_SOURCES = (
    "__init__.py",
    "wave49_schema.py",
    "wave50_model.py",
    "wave50_neural.py",
    "wave52_policy.py",
    "wave53_uncertainty.py",
    "wave54_joint_set.py",
    "wave55_policy_bridge.py",
    "wave56_contextual_gate.py",
)

PHASES = ("fit", "select", "adjudicate")
PHASE_TO_SPLIT = {"fit": "train", "select": "val", "adjudicate": "lockbox"}
PHASE_TO_ROLE = {"fit": "gate_fit", "select": "gate_select", "adjudicate": "sealed_monitor"}
SUCCESS_STATE = {"fit": "FIT_COMPLETE", "select": "SELECT_COMPLETE", "adjudicate": "COMPLETE"}
NOT_EVALUABLE_STATE = {
    "fit": "FIT_NOT_EVALUABLE",
    "select": "SELECT_NOT_EVALUABLE",
    "adjudicate": "MONITOR_NOT_EVALUABLE",
}
NOT_EVALUABLE_FILE = {
    "fit": "fit_not_evaluable.json",
    "select": "selection_not_evaluable.json",
    "adjudicate": "monitor_not_evaluable.json",
}
CORE_FILES = {
    "fit": ("fit_core.json", "fit_arrays.npz", "fit_freeze.json", "feature_schema.json"),
    "select": ("selection_core.json", "selection_arrays.npz", "selection_freeze.json"),
    "adjudicate": ("analysis_core.json", "result_arrays.npz"),
}
ANALYTICS_DIR = "analytics.complete"


def parse_args() -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-dir", "--output-dir", dest="run_dir", type=Path, required=True)
    common.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    common.add_argument("--wave52-dir", type=Path)
    common.add_argument("--wave54-dir", type=Path)
    common.add_argument("--policy-manifest", type=Path)
    common.add_argument("--wave54-selection-freeze", type=Path)
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    subparsers.add_parser("fit", parents=[common])
    subparsers.add_parser("select", parents=[common])
    adjudicate = subparsers.add_parser("adjudicate", parents=[common])
    adjudicate.add_argument("--reference-dir", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_json(value: Any, location: str = "$") -> None:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"non-finite JSON value at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json(item, f"{location}[{index}]")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path: Path, payload: Any, mode: int = 0o600) -> None:
    _validate_json(payload)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def file_inventory(root: Path, *, excluded: set[str] | None = None) -> dict[str, dict[str, Any]]:
    excluded = excluded or set()
    return {
        str(path.relative_to(root)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in excluded
    }


def hash_inventory(root: Path) -> dict[str, str]:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"symlink found in frozen tree: {root}")
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def public_run_inventory(root: Path) -> dict[str, dict[str, Any]]:
    """Hash operational/public state without reading escrow or sealed truth."""
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        text = str(relative)
        if text == "generation_escrow.json" or text.startswith("benchmark/sealed/"):
            continue
        result[text] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def validate_prepared_package(
    run_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    preparation: dict[str, Any],
) -> None:
    """Revalidate every public/frozen preparation input before opening labels."""
    if preparation.get("prospective_config") != config:
        raise RuntimeError("prepared package embeds a different prospective config")
    if preparation.get("sources") != execution_source_hashes(config_path):
        raise RuntimeError("prepared execution-source inventory differs from current HEAD")
    benchmark = run_dir / "benchmark"
    fixed = {
        benchmark / "manifest.json": preparation.get("benchmark_manifest_sha256"),
        benchmark / "protocol_config.json": preparation.get("protocol_config_sha256"),
    }
    for path, expected in fixed.items():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"prepared fixed input changed: {path}")

    visible_root = benchmark / "visible"
    expected_visible = {
        f"{split}.jsonl": preparation.get("visible_sha256", {}).get(split)
        for split in PHASE_TO_SPLIT.values()
    }
    if hash_inventory(visible_root) != expected_visible:
        raise RuntimeError("prepared visible inventory differs from preparation freeze")

    inference_root = run_dir / "inference"
    expected_inference = preparation.get("inference_hashes")
    if not isinstance(expected_inference, dict) or hash_inventory(inference_root) != expected_inference:
        raise RuntimeError("prepared inference inventory differs from preparation freeze")

    preparation_receipt = json.loads(
        (run_dir / "preparation_receipt.json").read_text(encoding="utf-8")
    )
    if (
        preparation_receipt.get("preparation_freeze_sha256")
        != sha256_file(run_dir / "preparation_freeze.json")
        or preparation_receipt.get("generation_receipt_sha256")
        != sha256_file(run_dir / "generation_receipt.json")
        or preparation_receipt.get("next_state") != "PREPARED"
    ):
        raise RuntimeError("preparation receipt does not authenticate the prepared package")
    generation = json.loads((run_dir / "generation_receipt.json").read_text(encoding="utf-8"))
    if (
        generation.get("manifest_sha256") != preparation.get("benchmark_manifest_sha256")
        or generation.get("visible_sha256") != preparation.get("visible_sha256")
        or generation.get("key_commitments") != preparation.get("key_commitments")
    ):
        raise RuntimeError("generation receipt differs from preparation freeze")


def verify_inventory(root: Path, expected: dict[str, dict[str, Any]]) -> None:
    for relative, row in expected.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"transaction input disappeared: {relative}")
        actual = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        if actual != row:
            raise RuntimeError(f"transaction input changed: {relative}")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def require_sources_at_head(config_path: Path) -> str:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = config.get("required_execution_sources")
    paths = (
        [REPO_ROOT / relative for relative in required]
        if required
        else [Path(__file__), WORKER_PATH, MATERIALIZER_PATH, RETROSPECTIVE_PATH, PRIMITIVES_PATH, PLAN_PATH, config_path]
    )
    if RETROSPECTIVE_PATH not in paths:
        paths.append(RETROSPECTIVE_PATH)
    for path in paths:
        relative = path.resolve(strict=True).relative_to(REPO_ROOT)
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(relative)], cwd=REPO_ROOT, text=True
        ).strip()
        if dirty:
            raise RuntimeError(f"execution source differs from HEAD: {relative}")
    return git_head()


def execution_source_hashes(config_path: Path) -> dict[str, str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    relatives = list(config.get("required_execution_sources", ()))
    retrospective_relative = str(RETROSPECTIVE_PATH.relative_to(REPO_ROOT))
    if retrospective_relative not in relatives:
        relatives.append(retrospective_relative)
    return {relative: sha256_file(REPO_ROOT / relative) for relative in relatives}


def _phase_dir(run_dir: Path, phase: str, suffix: str) -> Path:
    return run_dir / "phases" / f"{phase}.{suffix}"


def current_state(run_dir: Path) -> str:
    if not (run_dir / "preparation_freeze.json").is_file():
        raise RuntimeError("run is not PREPARED")
    present: set[tuple[str, str]] = set()
    for phase in PHASES:
        for suffix in ("pending", "complete", "not_evaluable"):
            path = _phase_dir(run_dir, phase, suffix)
            if path.exists():
                present.add((phase, suffix))
    legal = {
        frozenset(): "PREPARED",
        frozenset({("fit", "pending")}): "FIT_PENDING",
        frozenset({("fit", "not_evaluable")}): "FIT_NOT_EVALUABLE",
        frozenset({("fit", "complete")}): "FIT_COMPLETE",
        frozenset({("fit", "complete"), ("select", "pending")}): "SELECT_PENDING",
        frozenset({("fit", "complete"), ("select", "not_evaluable")}): "SELECT_NOT_EVALUABLE",
        frozenset({("fit", "complete"), ("select", "complete")}): "SELECT_COMPLETE",
        frozenset({("fit", "complete"), ("select", "complete"), ("adjudicate", "pending")}): "ADJUDICATE_PENDING",
        frozenset({("fit", "complete"), ("select", "complete"), ("adjudicate", "not_evaluable")}): "MONITOR_NOT_EVALUABLE",
        frozenset({("fit", "complete"), ("select", "complete"), ("adjudicate", "complete")}): "COMPLETE",
    }
    try:
        return legal[frozenset(present)]
    except KeyError as error:
        raise RuntimeError(f"incompatible Wave 56 phase topology: {sorted(present)}") from error


def validate_transition(state: str, phase: str) -> None:
    allowed = {
        "fit": {"PREPARED", "FIT_PENDING"},
        "select": {"FIT_COMPLETE", "SELECT_PENDING"},
        "adjudicate": {"SELECT_COMPLETE", "ADJUDICATE_PENDING"},
    }
    if state not in allowed[phase]:
        raise RuntimeError(f"cannot run {phase} from terminal/current state {state}")


def _assert_no_future_material(run_dir: Path, phase: str) -> None:
    phase_index = PHASES.index(phase)
    future_roles = {PHASE_TO_ROLE[name] for name in PHASES[phase_index + 1:]}
    future_splits = {PHASE_TO_SPLIT[name] for name in PHASES[phase_index + 1:]}
    forbidden_names = future_roles | future_splits
    for path in (run_dir / "phases").rglob("*") if (run_dir / "phases").exists() else ():
        lowered = path.name.lower()
        if path.is_file() and any(name in lowered for name in forbidden_names):
            raise RuntimeError(f"future material already exists before {phase}: {path}")
    benchmark_oracle = run_dir / "benchmark/sealed/oracle"
    if benchmark_oracle.exists() and any((benchmark_oracle / f"{split}.jsonl").exists() for split in future_splits):
        raise RuntimeError(f"future benchmark oracle already exists before {phase}")


def _load_freeze(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files", {})
    for filename, expected in files.items():
        if sha256_file(path.parent / filename) != expected:
            raise RuntimeError(f"freeze hash mismatch for {filename}")
    return payload


def phase_artifact_root(phase_root: Path) -> Path:
    artifact_root = phase_root / ANALYTICS_DIR
    return artifact_root if artifact_root.is_dir() else phase_root


def validate_completed_phase(run_dir: Path, phase: str) -> Path:
    root = _phase_dir(run_dir, phase, "complete")
    if not root.is_dir():
        raise RuntimeError(f"{phase} is not complete")
    artifacts = phase_artifact_root(root)
    for filename in CORE_FILES[phase]:
        if not (artifacts / filename).is_file():
            raise RuntimeError(f"completed {phase} lacks {filename}")
    freeze_name = {"fit": "fit_freeze.json", "select": "selection_freeze.json"}.get(phase)
    if freeze_name:
        _load_freeze(artifacts / freeze_name)
    return root


def _source_candidate(
    run_dir: Path,
    relatives: tuple[str, ...],
    explicit: Path | None = None,
    *,
    expected_sha256: str | None = None,
) -> Path:
    candidates = ([explicit] if explicit else []) + [run_dir / relative for relative in relatives]
    found = [path.resolve(strict=True) for path in candidates if path is not None and path.exists()]
    if not found:
        raise FileNotFoundError(f"prepared package lacks all candidates: {relatives}")
    source = found[0]
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"staged source must be a regular non-symlink file: {source}")
    if expected_sha256 is not None and sha256_file(source) != expected_sha256:
        raise RuntimeError(f"staged source differs from frozen hash: {source}")
    return source


def _bound_upstream_path(
    preparation: dict[str, Any], expected_sha256: str, basename: str
) -> Path:
    matches = []
    for row in preparation.get("upstream", []):
        if row.get("sha256") == expected_sha256 and Path(row.get("path", "")).name == basename:
            candidate = Path(row["path"])
            if candidate.is_file() and not candidate.is_symlink():
                matches.append(candidate.resolve(strict=True))
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen upstream {basename}; got {matches}")
    if sha256_file(matches[0]) != expected_sha256:
        raise RuntimeError(f"frozen upstream changed: {matches[0]}")
    return matches[0]


def historical_pair_tokens(preparation: dict[str, Any]) -> list[str]:
    tokens: set[str] = set()
    accepted_names = {"posterior_state.npz", "decision_select.npz", "sealed_monitor.npz"}
    for row in preparation.get("upstream", []):
        source = Path(row.get("path", ""))
        if source.name not in accepted_names:
            continue
        source = source.resolve(strict=True)
        if source.is_symlink() or sha256_file(source) != row.get("sha256"):
            raise RuntimeError(f"historical token source changed: {source}")
        with np.load(source, allow_pickle=False) as data:
            if "pair_token" in data.files:
                tokens.update(data["pair_token"].astype(str).tolist())
    if not tokens:
        raise RuntimeError("prepared freeze exposes no Wave 54-55 token inventory")
    return sorted(tokens)


def resolve_prepared_inputs(
    run_dir: Path,
    phase: str,
    preparation: dict[str, Any],
    config: dict[str, Any],
    policy_manifest: Path | None,
    wave54_selection_freeze: Path | None,
) -> dict[str, Path]:
    split = PHASE_TO_SPLIT[phase]
    binding = config["source_binding"]
    return {
        "visible": _source_candidate(run_dir, (
            f"analysis_staging/visible/{split}.jsonl",
            f"staging/visible/{split}.jsonl",
            f"visible/{split}.jsonl",
            f"benchmark/visible/{split}.jsonl",
        ), expected_sha256=preparation["visible_sha256"][split]),
        "protocol": _source_candidate(run_dir, (
            "analysis_staging/protocol_config.json",
            "staging/protocol_config.json",
            "protocol_config.json",
            "benchmark/protocol_config.json",
        ), expected_sha256=preparation["protocol_config_sha256"]),
        "policy_manifest": _source_candidate(run_dir, (
            "analysis_staging/frozen/policy_manifest.json",
            "staging/frozen/policy_manifest.json",
            "frozen/policy_manifest.json",
            "policy_manifest.json",
        ), policy_manifest, expected_sha256=binding["wave52_policy_manifest_sha256"]),
        "wave54_selection_freeze": _source_candidate(run_dir, (
            "analysis_staging/frozen/wave54_selection_freeze.json",
            "staging/frozen/wave54_selection_freeze.json",
            "frozen/wave54_selection_freeze.json",
            "wave54_selection_freeze.json",
        ), wave54_selection_freeze, expected_sha256=binding["wave54_selection_freeze_sha256"]),
    }


def _copy_regular(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"refusing non-regular staging source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def validate_materialization(
    run_dir: Path,
    config_path: Path,
    phase: str,
    destination: Path,
) -> None:
    split = PHASE_TO_SPLIT[phase]
    role = PHASE_TO_ROLE[phase]
    expected_names = {f"{split}.jsonl", "materialization_receipt.json"}
    if destination.is_symlink() or any(path.is_symlink() for path in destination.rglob("*")):
        raise RuntimeError("authorized-label materialization contains a symlink")
    actual_names = {
        str(path.relative_to(destination))
        for path in destination.rglob("*")
        if path.is_file()
    }
    if actual_names != expected_names:
        raise RuntimeError(f"incomplete materialization in pending phase: {sorted(actual_names)}")
    label = destination / f"{split}.jsonl"
    receipt = json.loads((destination / "materialization_receipt.json").read_text(encoding="utf-8"))
    sealed_truth = run_dir / "benchmark/sealed" / f"{split}.jsonl"
    protocol = run_dir / "benchmark/protocol_config.json"
    expected = {
        "phase": "wave56-split-scoped-oracle-materialization",
        "effective_uid": 0,
        "effective_gid": 0,
        "split": split,
        "role": role,
        "sealed_truth_sha256": sha256_file(sealed_truth),
        "protocol_config_sha256": sha256_file(protocol),
        "prospective_config_sha256": sha256_file(config_path),
        "authorized_labels_sha256": sha256_file(label),
        "other_splits_materialized": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"materialization receipt mismatch for {key}")
    row_count = sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())
    if receipt.get("count") != row_count:
        raise RuntimeError("materialization receipt row count mismatch")


def materialize_labels(run_dir: Path, config_path: Path, phase: str, pending: Path) -> Path:
    destination = pending / "authorized_labels"
    if destination.exists():
        validate_materialization(run_dir, config_path, phase, destination)
        return destination
    command = [
        sys.executable,
        str(MATERIALIZER_PATH),
        "--benchmark-root", str((run_dir / "benchmark").resolve(strict=True)),
        "--config", str(config_path),
        "--split", PHASE_TO_SPLIT[phase],
        "--role", PHASE_TO_ROLE[phase],
        "--destination", str(destination),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    validate_materialization(run_dir, config_path, phase, destination)
    return destination


def _copy_logits(
    run_dir: Path,
    split: str,
    destination: Path,
    preparation: dict[str, Any],
    config: dict[str, Any],
) -> None:
    source_root = run_dir / "inference/logits"
    names = {f"seed{int(seed)}__{split}.npz" for seed in config["seeds"]}
    matches = sorted(source_root / name for name in names)
    for source in matches:
        relative = str(source.relative_to(run_dir / "inference"))
        expected = preparation["inference_hashes"].get(relative)
        if source.is_symlink() or not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"frozen inference logit changed or is absent: {relative}")
        _copy_regular(source, destination / source.name)


def _make_stage_readable(stage: Path) -> None:
    for path in [stage, *stage.rglob("*")]:
        if path.is_symlink():
            raise RuntimeError(f"phase staging cannot contain symlinks: {path}")
        path.chmod(0o755 if path.is_dir() else 0o644)


def build_phase_runtime(source: Path) -> Path:
    package = source / "geometria_proporcional"
    for name in PHASE_RUNTIME_SOURCES:
        _copy_regular(REPO_ROOT / "src/geometria_proporcional" / name, package / name)
    _copy_regular(WORKER_PATH, source / WORKER_PATH.name)
    _copy_regular(RETROSPECTIVE_PATH, source / RETROSPECTIVE_PATH.name)
    _make_stage_readable(source)
    return source / WORKER_PATH.name


def build_worker_stage(
    stage: Path,
    run_dir: Path,
    config_path: Path,
    phase: str,
    labels: Path,
    policy_manifest: Path | None,
    wave54_selection_freeze: Path | None,
    commit: str,
    historical_tokens: list[str],
    preparation: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    inputs = resolve_prepared_inputs(
        run_dir,
        phase,
        preparation,
        config,
        policy_manifest,
        wave54_selection_freeze,
    )
    split = PHASE_TO_SPLIT[phase]
    stage.mkdir()
    _copy_regular(config_path, stage / "config.json")
    _copy_regular(inputs["protocol"], stage / "protocol_config.json")
    _copy_regular(inputs["visible"], stage / "visible" / f"{split}.jsonl")
    _copy_regular(labels / f"{split}.jsonl", stage / "labels" / f"{split}.jsonl")
    _copy_regular(inputs["policy_manifest"], stage / "frozen/policy_manifest.json")
    _copy_regular(
        inputs["wave54_selection_freeze"],
        stage / "frozen/wave54_selection_freeze.json",
    )
    write_json_atomic(
        stage / "frozen/historical_pair_tokens.json",
        {"pair_tokens": historical_tokens},
        mode=0o644,
    )
    _copy_logits(run_dir, split, stage / "inference/logits", preparation, config)
    if phase in {"select", "adjudicate"}:
        fit = phase_artifact_root(validate_completed_phase(run_dir, "fit"))
        for filename in CORE_FILES["fit"]:
            _copy_regular(fit / filename, stage / "previous" / filename)
    if phase == "adjudicate":
        select = phase_artifact_root(validate_completed_phase(run_dir, "select"))
        for filename in CORE_FILES["select"]:
            _copy_regular(select / filename, stage / "previous" / filename)
    staged_input_hashes = hash_inventory(stage)
    write_json_atomic(
        stage / "phase_request.json",
        {
            "phase": phase,
            "split": split,
            "role": PHASE_TO_ROLE[phase],
            "git_commit": commit,
            "config_sha256": sha256_file(config_path),
            "preparation_freeze_sha256": sha256_file(run_dir / "preparation_freeze.json"),
            "execution_sources": execution_source_hashes(config_path),
            "policy_manifest_sha256": sha256_file(inputs["policy_manifest"]),
            "wave54_selection_freeze_sha256": sha256_file(
                inputs["wave54_selection_freeze"]
            ),
            "staged_input_hashes": staged_input_hashes,
        },
        mode=0o644,
    )
    source = stage.parent / "source"
    _make_stage_readable(stage)
    return build_phase_runtime(source)


def validate_worker_receipt(receipt: dict[str, Any], stage: Path) -> None:
    request = json.loads((stage / "phase_request.json").read_text(encoding="utf-8"))
    stage_hashes = hash_inventory(stage)
    if receipt.get("phase") != request.get("phase"):
        raise RuntimeError("restricted worker receipt phase differs from its request")
    if receipt.get("stage_inventory") != sorted(stage_hashes):
        raise RuntimeError("restricted worker receipt stage inventory differs")
    if receipt.get("stage_hashes") != stage_hashes:
        raise RuntimeError("restricted worker receipt is not bound to staged inputs")


def run_restricted_worker(stage: Path, worker_output: Path, worker: Path, probes: list[Path]) -> None:
    if os.geteuid() != 0:
        raise PermissionError("Wave 56 phase coordinator must run as root")
    account = pwd.getpwnam("nobody")
    worker_output.mkdir(mode=0o700)
    os.chown(worker_output, account.pw_uid, account.pw_gid)
    command = [
        "setpriv",
        "--reuid", str(account.pw_uid),
        "--regid", str(account.pw_gid),
        "--clear-groups",
        "--no-new-privs",
        sys.executable,
        str(worker),
        "--stage", str(stage),
        "--output", str(worker_output),
    ]
    for probe in probes:
        command.extend(["--forbidden-probe", str(probe)])
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(stage.parent / "source"),
        "HOME": "/tmp",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    })
    completed = subprocess.run(command, cwd=stage, env=env, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"restricted {stage.name} worker failed ({completed.returncode}): {completed.stderr.strip()}")
    receipt = json.loads((worker_output / "access_receipt.json").read_text(encoding="utf-8"))
    validate_worker_receipt(receipt, stage)
    if receipt.get("effective_uid") == 0 or receipt.get("benchmark_root_received") is not False:
        raise RuntimeError("restricted worker boundary was not demonstrated")
    security = receipt.get("process_security", {})
    if (
        security.get("effective_capabilities_hex") != "0000000000000000"
        or security.get("no_new_privileges") != 1
        or security.get("supplementary_groups") != []
    ):
        raise RuntimeError("restricted worker privilege state was not demonstrated")
    if not all(row.get("denied") for row in receipt.get("sealed_probes", [])):
        raise RuntimeError("restricted worker did not prove sealed access denial")


def _phase_probes(run_dir: Path, phase: str) -> list[Path]:
    split_index = PHASES.index(phase)
    splits = [PHASE_TO_SPLIT[name] for name in PHASES[split_index:]]
    return [(run_dir / "benchmark/sealed" / f"{split}.jsonl").resolve(strict=True) for split in splits]


def _journal_path(pending: Path) -> Path:
    return pending / "transaction_journal.json"


def begin_or_resume(run_dir: Path, phase: str, commit: str, config_path: Path) -> Path:
    pending = _phase_dir(run_dir, phase, "pending")
    if pending.exists():
        journal = json.loads(_journal_path(pending).read_text(encoding="utf-8"))
        if (
            journal["git_commit"] != commit
            or journal["config_sha256"] != sha256_file(config_path)
            or journal["execution_sources"] != execution_source_hashes(config_path)
        ):
            raise RuntimeError("pending phase can only resume with the same HEAD and config")
        verify_inventory(pending, journal.get("durable_inventory", {}))
        return pending
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.mkdir()
    write_json_atomic(
        _journal_path(pending),
        {
            "phase": phase,
            "git_commit": commit,
            "config_sha256": sha256_file(config_path),
            "execution_sources": execution_source_hashes(config_path),
            "step": "PENDING_CREATED",
            "durable_inventory": {},
        },
    )
    fsync_directory(pending.parent)
    return pending


def update_journal(pending: Path, step: str) -> None:
    previous = json.loads(_journal_path(pending).read_text(encoding="utf-8"))
    inventory = file_inventory(pending, excluded={"transaction_journal.json", "inventory_before.json", "inventory_after.json"})
    previous.update({"step": step, "durable_inventory": inventory})
    write_json_atomic(_journal_path(pending), previous)


def promote_phase(run_dir: Path, phase: str, pending: Path, state: str) -> Path:
    suffix = "complete" if state == SUCCESS_STATE[phase] else "not_evaluable"
    destination = _phase_dir(run_dir, phase, suffix)
    if destination.exists():
        raise FileExistsError(destination)
    for path in sorted(pending.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    fsync_directory(pending)
    os.replace(pending, destination)
    fsync_directory(destination.parent)
    return destination


def validate_worker_results(root: Path) -> dict[str, Any]:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("worker result tree contains a symlink")
    receipt_path = root / "access_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("worker result tree lacks access receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = receipt.get("output_inventory")
    if not isinstance(expected, dict):
        raise RuntimeError("worker access receipt lacks output inventory")
    actual = file_inventory(root, excluded={"access_receipt.json"})
    if actual != expected:
        raise RuntimeError("worker result tree differs from its exact output inventory")
    return receipt


def publish_worker_results(worker_output: Path, pending: Path) -> Path:
    staging = pending / "analytics.pending"
    destination = pending / ANALYTICS_DIR
    if destination.exists():
        validate_worker_results(destination)
        return destination
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    try:
        for source in sorted(worker_output.rglob("*")):
            relative = source.relative_to(worker_output)
            target = staging / relative
            if source.is_symlink():
                raise RuntimeError(f"worker result contains symlink: {relative}")
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif source.is_file():
                _copy_regular(source, target)
        validate_worker_results(staging)
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        fsync_directory(staging)
        os.replace(staging, destination)
        fsync_directory(pending)
        validate_worker_results(destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def compare_reference(
    run_dir: Path,
    reference_dir: Path,
    *,
    adjudicate_root: Path | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for phase, json_names, array_name in (
        ("fit", ("fit_core.json", "feature_schema.json"), "fit_arrays.npz"),
        ("select", ("selection_core.json",), "selection_arrays.npz"),
        ("adjudicate", ("analysis_core.json",), "result_arrays.npz"),
    ):
        left_phase = adjudicate_root if phase == "adjudicate" and adjudicate_root else validate_completed_phase(run_dir, phase)
        left_root = phase_artifact_root(left_phase)
        right_root = phase_artifact_root(validate_completed_phase(reference_dir, phase))
        for name in json_names:
            checks[f"{phase}/{name}"] = (left_root / name).read_bytes() == (right_root / name).read_bytes()
        with np.load(left_root / array_name, allow_pickle=False) as left, np.load(right_root / array_name, allow_pickle=False) as right:
            exact = set(left.files) == set(right.files)
            if exact:
                for key in left.files:
                    if left[key].dtype.kind in "fc" and right[key].dtype.kind in "fc":
                        equal = np.array_equal(left[key], right[key], equal_nan=True)
                    else:
                        equal = np.array_equal(left[key], right[key])
                    exact = exact and bool(equal) and left[key].dtype == right[key].dtype and left[key].shape == right[key].shape
            checks[f"{phase}/{array_name}"] = exact
        bundle_name = {
            "fit": "gate_fit_bundle.npz",
            "select": "gate_select_bundle.npz",
            "adjudicate": "sealed_monitor_bundle.npz",
        }[phase]
        with np.load(left_root / bundle_name, allow_pickle=False) as left, np.load(
            right_root / bundle_name, allow_pickle=False
        ) as right:
            checks[f"{phase}/{bundle_name}"] = set(left.files) == set(right.files) and all(
                left[key].dtype == right[key].dtype
                and left[key].shape == right[key].shape
                and bool(
                    np.array_equal(left[key], right[key], equal_nan=left[key].dtype.kind in "fc")
                )
                for key in left.files
            )
    if not all(checks.values()):
        raise RuntimeError(f"Wave 56 Stage 1 replay mismatch: {checks}")
    return {"all_exact": True, "checks": checks}


def validate_replay_reference(
    run_dir: Path,
    reference_dir: Path | None,
    config: dict[str, Any],
    preparation: dict[str, Any],
) -> Path | None:
    primary_name = str(config["primary_output_name"])
    replay_name = str(config["replay_output_name"])
    receipt = json.loads((run_dir / "preparation_receipt.json").read_text(encoding="utf-8"))
    mode = receipt.get("execution_mode")
    if run_dir.name == primary_name:
        if reference_dir is not None:
            raise ValueError("canonical primary cannot claim replay equivalence")
        if mode not in {"primary", "recovery"}:
            raise RuntimeError("canonical primary has an invalid preparation mode")
        return None
    if run_dir.name != replay_name or mode != "replay":
        raise RuntimeError("run directory/preparation mode is not a canonical primary or replay")
    if reference_dir is None:
        return None
    reference = reference_dir.resolve(strict=True)
    if reference.parent != run_dir.parent or reference.name != primary_name:
        raise ValueError("replay reference must be the canonical sibling primary")
    replay_receipt = json.loads((run_dir / "preparation_replay.json").read_text(encoding="utf-8"))
    reference_receipt = json.loads(
        (reference / "preparation_receipt.json").read_text(encoding="utf-8")
    )
    reference_freeze = json.loads(
        (reference / "preparation_freeze.json").read_text(encoding="utf-8")
    )
    if replay_receipt.get("all_exact") is not True or receipt.get("replay_exact") is not True:
        raise RuntimeError("canonical replay preparation was not exact")
    if reference_receipt.get("execution_mode") not in {"primary", "recovery"}:
        raise RuntimeError("replay reference is not the canonical primary preparation")
    if preparation.get("key_commitments") != reference_freeze.get("key_commitments"):
        raise RuntimeError("replay and primary key commitments differ")
    if preparation != reference_freeze:
        raise RuntimeError("replay and primary preparation freezes differ")
    return reference


def write_public_artifact_manifest(run_dir: Path) -> None:
    manifest = run_dir / "artifact_manifest.json"
    manifest.unlink(missing_ok=True)
    files = public_run_inventory(run_dir)
    write_json_atomic(
        manifest,
        {
            "scope": "complete public Wave 56 Stage 1 package",
            "files": files,
            "excluded": ["generation_escrow.json", "benchmark/sealed/**", "artifact_manifest.json"],
            "excluded_reason": "secrets/sealed truth are not read into the public manifest; self excluded",
        },
        mode=0o644,
    )


def write_diagnostic_outcome(pending: Path, replay_exact: bool | None) -> None:
    core = json.loads((phase_artifact_root(pending) / "analysis_core.json").read_text(encoding="utf-8"))
    conditions = dict(core["diagnostic_pattern"]["conditions"])
    selector_condition = conditions.pop("diagnostic_condition_6_without_replay")
    conditions["diagnostic_condition_6"] = (
        bool(selector_condition and replay_exact)
        if replay_exact is not None
        else "PENDING_EXACT_REPLAY"
    )
    observed = list(conditions.values())
    aggregate = bool(all(observed)) if all(isinstance(value, bool) for value in observed) else None
    write_json_atomic(
        pending / "diagnostic_outcome.json",
        {
            "conditions": conditions,
            "prospective_pattern_observed": aggregate,
            "replay_exact": replay_exact,
            "scientific_decision": None,
            "decision_authority": "user",
        },
    )


def run_phase(
    run_dir: Path,
    config_path: Path,
    phase: str,
    *,
    policy_manifest: Path | None = None,
    wave54_selection_freeze: Path | None = None,
    reference_dir: Path | None = None,
    enforce_sources: bool = True,
) -> str:
    run_dir = run_dir.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    commit = require_sources_at_head(config_path) if enforce_sources else git_head()
    preparation = json.loads((run_dir / "preparation_freeze.json").read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if enforce_sources and preparation.get("git_commit") != commit:
        raise RuntimeError("prepared package was not created from current HEAD")
    if enforce_sources and preparation.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("prepared package config differs from current frozen config")
    if enforce_sources:
        validate_prepared_package(run_dir, config_path, config, preparation)
    canonical_reference = validate_replay_reference(
        run_dir,
        reference_dir if phase == "adjudicate" else None,
        config,
        preparation,
    ) if enforce_sources else reference_dir
    if (
        enforce_sources
        and phase == "adjudicate"
        and run_dir.name == str(config["replay_output_name"])
        and canonical_reference is None
    ):
        raise ValueError("canonical replay adjudication requires its canonical primary reference")
    binding = config["source_binding"]
    if policy_manifest is None:
        policy_manifest = _bound_upstream_path(
            preparation, binding["wave52_policy_manifest_sha256"], "policy_manifest.json"
        )
    if wave54_selection_freeze is None:
        wave54_selection_freeze = _bound_upstream_path(
            preparation,
            binding["wave54_selection_freeze_sha256"],
            "selection_freeze.json",
        )
    else:
        wave54_selection_freeze = _source_candidate(
            run_dir,
            (),
            wave54_selection_freeze,
            expected_sha256=binding["wave54_selection_freeze_sha256"],
        )
    if policy_manifest is not None:
        policy_manifest = _source_candidate(
            run_dir,
            (),
            policy_manifest,
            expected_sha256=binding["wave52_policy_manifest_sha256"],
        )
    old_tokens = historical_pair_tokens(preparation)
    state = current_state(run_dir)
    validate_transition(state, phase)
    _assert_no_future_material(run_dir, phase)
    pending = begin_or_resume(run_dir, phase, commit, config_path)
    if not (pending / "inventory_before.json").exists():
        write_json_atomic(
            pending / "inventory_before.json", {"run": public_run_inventory(run_dir)}
        )
    journal = json.loads(_journal_path(pending).read_text(encoding="utf-8"))
    labels = materialize_labels(run_dir, config_path, phase, pending)
    if journal["step"] == "PENDING_CREATED":
        update_journal(pending, "ORACLE_MATERIALIZED")
        journal = json.loads(_journal_path(pending).read_text(encoding="utf-8"))

    if journal["step"] == "ORACLE_MATERIALIZED":
        # Rebuild the deterministic stage even when analytics were already
        # promoted: recovery authenticates the prior receipt against the same inputs.
        with tempfile.TemporaryDirectory(prefix=f"wave56-{phase}-", dir="/tmp") as raw:
            temporary = Path(raw)
            temporary.chmod(0o711)
            stage = temporary / "stage"
            worker = build_worker_stage(
                stage,
                run_dir,
                config_path,
                phase,
                labels,
                policy_manifest,
                wave54_selection_freeze,
                commit,
                old_tokens,
                preparation,
                config,
            )
            analytics = pending / ANALYTICS_DIR
            if analytics.exists():
                receipt = validate_worker_results(analytics)
                validate_worker_receipt(receipt, stage)
            else:
                worker_output = temporary / "worker-output"
                run_restricted_worker(stage, worker_output, worker, _phase_probes(run_dir, phase))
                publish_worker_results(worker_output, pending)
        update_journal(pending, "ANALYTICS_COMPLETE")
        journal = json.loads(_journal_path(pending).read_text(encoding="utf-8"))

    analytics = pending / ANALYTICS_DIR
    if journal["step"] in {"ANALYTICS_COMPLETE", "READY_TO_PROMOTE"} or analytics.exists():
        validate_worker_results(analytics)

    if (analytics / NOT_EVALUABLE_FILE[phase]).is_file():
        terminal_state = NOT_EVALUABLE_STATE[phase]
    else:
        terminal_state = SUCCESS_STATE[phase]
        for filename in CORE_FILES[phase]:
            if not (analytics / filename).is_file():
                raise RuntimeError(f"phase worker omitted terminal artifact {filename}")
    if phase == "adjudicate":
        if terminal_state == "COMPLETE":
            # Replay metadata stays outside the byte-stable analytical core.
            if canonical_reference:
                replay = compare_reference(
                    run_dir, canonical_reference, adjudicate_root=pending
                )
                write_json_atomic(pending / "replay_receipt.json", replay)
            write_diagnostic_outcome(pending, True if canonical_reference else None)
        write_json_atomic(
            pending / "runtime.json",
            {
                "python": sys.version.split()[0],
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
                "device": "cpu",
            },
        )
    write_json_atomic(
        pending / "inventory_after.json",
        {
            "phase": file_inventory(
                pending,
                excluded={
                    "transaction_journal.json",
                    "inventory_before.json",
                    "inventory_after.json",
                },
            )
        },
    )
    update_journal(pending, "READY_TO_PROMOTE")
    promote_phase(run_dir, phase, pending, terminal_state)
    if phase == "adjudicate":
        write_public_artifact_manifest(run_dir)
    return terminal_state


def main() -> None:
    args = parse_args()
    policy_manifest = args.policy_manifest
    if policy_manifest is None and args.wave52_dir is not None:
        policy_manifest = args.wave52_dir / "policy_manifest.json"
    wave54_selection_freeze = args.wave54_selection_freeze
    if wave54_selection_freeze is None and args.wave54_dir is not None:
        wave54_selection_freeze = args.wave54_dir / "selection_freeze.json"
    state = run_phase(
        args.run_dir,
        args.config,
        args.phase,
        policy_manifest=policy_manifest,
        wave54_selection_freeze=wave54_selection_freeze,
        reference_dir=getattr(args, "reference_dir", None),
    )
    print(json.dumps({"phase": args.phase, "state": state}, sort_keys=True))


if __name__ == "__main__":
    main()
