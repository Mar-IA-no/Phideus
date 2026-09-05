#!/usr/bin/env python3
"""Coordinate the physically separated Wave 59 analytical phases.

This runner consumes a previously prepared package.  It never generates a draw
or opens a sealed truth file outside the phase transition that authorizes it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import pwd
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from datetime import UTC, datetime
import hashlib

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometria_proporcional.wave49_schema import sha256_file  # noqa: E402
from geometria_proporcional.wave59_hgb_guard_bracket import (  # noqa: E402
    HARM_CONTROL_SEEDS,
    INCOMPATIBILITY_CONTROL_SEEDS,
    inference_safe_view,
    model_id,
    validate_pre_draw_config,
)


CONFIG_DEFAULT = REPO_ROOT / "experiments/geometria_proporcional/configs/wave59_fresh_hgb_guard_bracket.json"
WORKER_SOURCE = REPO_ROOT / "experiments/geometria_proporcional/_wave59_phase_worker.py"
RUNTIME_MODULES = (
    "__init__.py",
    "wave49_schema.py",
    "wave52_policy.py",
    "wave53_uncertainty.py",
    "wave55_policy_bridge.py",
    "wave56_contextual_gate.py",
    "wave58_open_diagnostic.py",
    "wave59_hgb_guard_bracket.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, payload: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode()
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_npz(path: Path, arrays: dict[str, np.ndarray], *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        np.savez(temporary, **{key: np.asarray(value) for key, value in sorted(arrays.items())})
        temporary.chmod(mode)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def materialize_prepared_bundles(
    run_dir: Path,
    config: dict[str, Any],
    *,
    policy_manifest: Path,
    wave54_selection_freeze: Path,
) -> dict[str, str]:
    """Create Wave 59 safe/truth bundles during PREPARE under root control.

    The shared preparer has already sealed the benchmark and completed blind
    inference.  This root-only step derives the frozen ``primary`` indicator
    and all result-affecting arrays once, publishes only the allowlisted safe
    projections for calibration/monitor application, and keeps full truth
    bundles mode 0600 for phase-scoped coordinator access.
    """
    if os.geteuid() != 0 or os.getegid() != 0:
        raise PermissionError("Wave 59 bundle preparation requires root")
    if run_dir.is_symlink():
        raise RuntimeError("Wave 59 preparation root cannot be a symlink")

    # These imports are deliberately local: the analytical workers never stage
    # either the oracle materializer or the legacy bundle constructor.
    from geometria_proporcional.wave49_oracle import compute_oracle_splits
    from geometria_proporcional.wave49_schema import ProtocolConfig
    import _wave56_phase_worker as wave56_worker
    import run_wave56_retrospective as retrospective

    benchmark = run_dir / "benchmark"
    protocol_path = benchmark / "protocol_config.json"
    protocol = ProtocolConfig.from_dict(read_json(protocol_path))
    utilities, _ = retrospective.load_utilities(policy_manifest.resolve(strict=True))
    selection = read_json(wave54_selection_freeze.resolve(strict=True))
    theta = np.asarray(
        selection["selected_models"]["joint_full"]["theta"], dtype=np.float64
    )
    seeds = [int(seed) for seed in config["seeds"]]
    prepared = run_dir / "prepared"
    if prepared.exists():
        raise FileExistsError(prepared)
    prepared.mkdir(mode=0o700)

    specs = {
        "train": ("gate_fit_bundle.npz", None),
        "val": ("gate_select_truth_bundle.npz", "gate_select_inference_bundle.npz"),
        "lockbox": (
            "sealed_monitor_truth_bundle.npz",
            "sealed_monitor_inference_bundle.npz",
        ),
    }
    hashes: dict[str, str] = {}
    try:
        for split, (truth_name, safe_name) in specs.items():
            with tempfile.TemporaryDirectory(prefix=f"wave59-prepare-{split}-", dir="/tmp") as raw:
                stage = Path(raw) / "stage"
                (stage / "visible").mkdir(parents=True)
                (stage / "labels").mkdir()
                (stage / "inference/logits").mkdir(parents=True)
                shutil.copyfile(benchmark / "visible" / f"{split}.jsonl", stage / "visible" / f"{split}.jsonl")
                shutil.copyfile(protocol_path, stage / "protocol_config.json")
                compute_oracle_splits(benchmark, protocol, (split,), stage / "labels")
                for seed in seeds:
                    source = run_dir / "inference/logits" / f"seed{seed}__{split}.npz"
                    shutil.copyfile(source, stage / "inference/logits" / source.name)

                role = config["physical_splits"][split]
                bundle = wave56_worker.build_bundle(stage, split, role, seeds)
                dataset_config = {
                    "hard_set_tau": float(config["hard_set_tau"]),
                    "incompatible_regret_penalty": float(config["penalty"]),
                    "primary_population": {
                        "design_stratum": "NEAR_RIVAL",
                        "minimum_true_cardinality": 2,
                    },
                }
                data = retrospective.make_dataset(bundle, theta, utilities, dataset_config)
                arrays = {
                    key: np.asarray(value)
                    for key, value in data.items()
                    if isinstance(value, np.ndarray)
                }
                truth_path = prepared / truth_name
                write_npz(truth_path, arrays, mode=0o600)
                hashes[f"prepared/{truth_name}"] = sha256_file(truth_path)
                if safe_name is not None:
                    safe_path = prepared / safe_name
                    write_npz(safe_path, inference_safe_view(arrays), mode=0o644)
                    hashes[f"prepared/{safe_name}"] = sha256_file(safe_path)
    except BaseException:
        shutil.rmtree(prepared, ignore_errors=True)
        raise
    _fsync_directory(prepared)
    return hashes


def _copy(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def load_utilities(policy_manifest: Path) -> np.ndarray:
    payload = read_json(policy_manifest.resolve(strict=True))
    levels = np.asarray(payload["levels"], dtype=np.float64)
    permutations = np.asarray(payload["rank_permutations"], dtype=np.int64)
    if permutations.shape != (24, 4) or set(map(tuple, permutations)) != set(
        itertools.permutations(range(4))
    ):
        raise ValueError("policy manifest utility_matrix is invalid")
    utilities = levels[permutations]
    if utilities.shape != (24, 4) or not np.all(np.isfinite(utilities)):
        raise ValueError("policy manifest utilities are invalid")
    return utilities


def build_runtime(root: Path) -> Path:
    source = root / "source"
    package = source / "geometria_proporcional"
    package.mkdir(parents=True)
    for name in RUNTIME_MODULES:
        _copy(SRC_ROOT / "geometria_proporcional" / name, package / name)
    _copy(WORKER_SOURCE, source / WORKER_SOURCE.name)
    for path in source.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o444)
    source.chmod(0o755)
    return source / WORKER_SOURCE.name


def _stage_request(stage: Path, phase: str) -> None:
    files = sorted(path.name for path in stage.iterdir() if path.is_file())
    hashes = {name: sha256_file(stage / name) for name in files}
    request = {
        "phase": phase,
        "allowed_files": sorted([*files, "phase_request.json"]),
        "sha256": hashes,
    }
    write_json(stage / "phase_request.json", request)
    for path in stage.iterdir():
        path.chmod(0o444)
    stage.chmod(0o555)


def run_worker(
    temporary: Path,
    stage: Path,
    phase: str,
    probes: list[Path],
) -> tuple[Path, dict[str, Any], float, int]:
    worker = build_runtime(temporary)
    _stage_request(stage, phase)
    account = pwd.getpwnam("nobody")
    output = temporary / "worker-output"
    output.mkdir(mode=0o700)
    os.chown(output, account.pw_uid, account.pw_gid)
    command = [
        "setpriv",
        "--reuid",
        str(account.pw_uid),
        "--regid",
        str(account.pw_gid),
        "--clear-groups",
        "--no-new-privs",
        sys.executable,
        str(worker),
        "--stage",
        str(stage),
        "--output",
        str(output),
    ]
    for probe in probes:
        command.extend(("--forbidden-probe", str(probe)))
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": str(temporary / "source"),
        "WAVE59_STAGED_RUNTIME": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "PYTHONHASHSEED": "0",
    }
    started = time.monotonic()
    completed = subprocess.run(
        command, cwd=stage, env=env, text=True, capture_output=True, check=False
    )
    duration = time.monotonic() - started
    max_rss_bytes = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024
    if completed.returncode:
        raise RuntimeError(f"Wave 59 {phase} worker failed: {completed.stderr.strip()}")
    receipt = read_json(output / "access_receipt.json")
    if receipt["effective_uid"] != 65534 or receipt["effective_gid"] != 65534:
        raise RuntimeError("Wave 59 worker identity drifted")
    security = receipt["process_security"]
    if security != {
        "effective_capabilities_hex": "0000000000000000",
        "no_new_privileges": 1,
        "supplementary_groups": [],
    }:
        raise RuntimeError("Wave 59 worker privilege boundary failed")
    if not all(row["denied"] for row in receipt["forbidden_probes"]):
        raise RuntimeError("Wave 59 worker could access a forbidden truth probe")
    threadpools = receipt.get("threadpools", [])
    if any(int(row.get("num_threads", 0)) > 4 for row in threadpools):
        raise RuntimeError("Wave 59 worker exceeded the four-thread contract")
    return output, receipt, duration, max_rss_bytes


def _publish(worker_output: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name(destination.name + ".pending")
    if staging.exists():
        raise FileExistsError(staging)
    shutil.copytree(
        worker_output,
        staging,
        symlinks=False,
        ignore=shutil.ignore_patterns("access_receipt.json"),
    )
    for path in staging.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    staging.chmod(0o555)
    os.replace(staging, destination)


def _run_phase(
    run_dir: Path,
    phase: str,
    inputs: dict[str, Path],
    probes: list[Path],
    *,
    destination_name: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix=f"wave59-{phase}-", dir="/tmp") as raw:
        temporary = Path(raw)
        temporary.chmod(0o711)
        stage = temporary / "stage"
        stage.mkdir()
        for name, source in inputs.items():
            _copy(source, stage / name)
        output, receipt, duration, max_rss_bytes = run_worker(
            temporary, stage, phase, probes
        )
        destination = run_dir / (destination_name or phase)
        _publish(output, destination)
    journal = {
        "schema_version": "wave59-phase-journal-v1",
        "phase": phase,
        "status": receipt["status"],
        "input_sha256": {
            name: sha256_file(path.resolve(strict=True)) for name, path in sorted(inputs.items())
        },
        "output_sha256": {
            str(path.relative_to(destination)): sha256_file(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        },
        "access_receipt": receipt,
        "duration_seconds": duration,
        "max_rss_bytes": max_rss_bytes,
        "maximum_truth_materialized": {
            "fit": "train",
            "calibrate_scores": "train",
            "validate": "validation",
            "monitor_apply": "validation",
            "monitor_evaluate": "monitor",
        }[phase],
    }
    write_json(run_dir / "journals" / f"{phase}.json", journal)
    return destination, journal


def _reuse_or_run_phase(
    run_dir: Path,
    phase: str,
    inputs: dict[str, Path],
    probes: list[Path],
    *,
    destination_name: str,
) -> tuple[Path, dict[str, Any]]:
    destination = run_dir / destination_name
    journal_path = run_dir / "journals" / f"{phase}.json"
    if journal_path.is_file():
        journal = read_json(journal_path)
        if journal.get("phase") != phase:
            raise RuntimeError(f"Wave 59 {phase} resume journal drifted")
        expected_inputs = {
            name: sha256_file(path.resolve(strict=True)) for name, path in sorted(inputs.items())
        }
        if journal.get("input_sha256") != expected_inputs:
            raise RuntimeError(f"Wave 59 {phase} resume inputs differ")
        for relative, expected in journal.get("output_sha256", {}).items():
            path = destination / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"Wave 59 {phase} resumed output differs: {relative}")
        return destination, journal
    if destination.exists():
        raise RuntimeError(f"Wave 59 {phase} output exists without a durable journal")
    return _run_phase(
        run_dir,
        phase,
        inputs,
        probes,
        destination_name=destination_name,
    )


def restore_identical_hash_attempt(
    archived: Path, output: Path, config_path: Path
) -> Path:
    archived = archived.resolve(strict=True)
    output = output.resolve(strict=False)
    if output.exists():
        raise FileExistsError(output)
    if not (archived / "FAILURE.json").is_file():
        raise RuntimeError("resume source is not a Wave 59 failed attempt")
    failure = read_json(archived / "FAILURE.json")
    if failure.get("schema_version") != "wave59-failed-attempt-v1":
        raise RuntimeError("resume source failure schema drifted")
    if Path(str(failure.get("original_path", ""))).resolve() != output:
        raise RuntimeError("identical-hash resume must restore the original canonical path")
    snapshot = archived / "config.snapshot.json"
    if sha256_file(snapshot) != sha256_file(config_path.resolve(strict=True)):
        raise RuntimeError("identical-hash resume config differs")
    shutil.copytree(
        archived,
        output,
        symlinks=False,
        ignore=shutil.ignore_patterns(
            "FAILURE.json",
            "failure_inventory.json",
            "artifact_manifest.json",
        ),
    )
    _fsync_directory(output.parent)
    return output


def _merge_evaluation(adjudication: Path, evaluation: Path, run_dir: Path) -> None:
    mapping = {
        evaluation / "bootstrap_indices.npz": adjudication / "bootstrap_indices.npz",
        evaluation / "analysis_arrays.npz": adjudication / "analysis_arrays.npz",
        evaluation / "analysis.json": run_dir / "analysis.json",
    }
    for source, destination in mapping.items():
        if not source.is_file() or source.is_symlink() or destination.exists():
            raise RuntimeError(f"invalid monitor-evaluate promotion: {source}")
        temporary = destination.with_name(f".{destination.name}.pending")
        shutil.copyfile(source, temporary)
        temporary.chmod(0o444)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    shutil.rmtree(evaluation)


def _write_report(run_dir: Path) -> None:
    analysis = read_json(run_dir / "analysis.json")
    patterns = analysis["prospective_patterns"]
    lines = [
        "# Wave 59 — fresh HGB guard bracket",
        "",
        "Estado operativo: `COMPLETE`. Decisión científica: reservada al usuario.",
        "",
        "Los dos patrones se informan por separado y no seleccionan una arquitectura:",
        "",
        f"- incompatibility: `{patterns['incompatibility']['aggregate_with_replay']}`",
        f"- harm: `{patterns['harm']['aggregate_with_replay']}`",
        "",
    ]
    path = run_dir / "REPORT.md"
    encoded = "\n".join(lines).encode("utf-8")
    descriptor, raw = tempfile.mkstemp(prefix=".REPORT.md.", dir=run_dir)
    temporary = Path(raw)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(run_dir)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _array_exact(left: Path, right: Path) -> bool:
    with np.load(left, allow_pickle=False) as lhs, np.load(right, allow_pickle=False) as rhs:
        if set(lhs.files) != set(rhs.files):
            return False
        for key in lhs.files:
            a = lhs[key]
            b = rhs[key]
            if a.dtype != b.dtype or a.shape != b.shape:
                return False
            if a.dtype.kind in "fc":
                if not np.array_equal(a, b, equal_nan=True):
                    return False
            elif not np.array_equal(a, b):
                return False
    return True


def _scientific_paths(canonical: bool) -> tuple[list[str], list[str]]:
    exact = [
        "config.snapshot.json",
        "source_bindings.json",
        "fit/feature_schema.json",
        "fit/fit_freeze.json",
        "fit/max_displacement_diagnostics.json",
        "calibration/calibration_freeze.json",
        "validation/validation_freeze.json",
        "validation/validation_summary.json",
        "adjudication/monitor_action_freeze.json",
        "analysis.json",
        "REPORT.md",
    ]
    arrays = [
        "fit/model_state_arrays.npz",
        "fit/train_scores.npz",
        "fit/max_displacement_mappings.npz",
        "calibration/validation_scores.npz",
        "calibration/validation_policy_arrays.npz",
        "validation/validation_metrics.npz",
        "adjudication/monitor_scores.npz",
        "adjudication/monitor_policy_arrays.npz",
        "adjudication/bootstrap_indices.npz",
        "adjudication/analysis_arrays.npz",
    ]
    if canonical:
        exact.extend(
            [
                "pre_generation_freeze.json",
                "preparation_freeze.json",
                "benchmark/manifest.json",
                "benchmark/protocol_config.json",
                "benchmark/attestations/semantic_root.json",
                "benchmark/commitments/semantic.jsonl",
                *(f"benchmark/visible/{split}.jsonl" for split in ("calibration_null", "train", "val", "lockbox")),
            ]
        )
        arrays.extend(
            [
                *(f"inference/logits/seed{seed}__{split}.npz" for seed in (17, 29, 43) for split in ("train", "val", "lockbox")),
                "prepared/gate_select_inference_bundle.npz",
                "prepared/sealed_monitor_inference_bundle.npz",
            ]
        )
    return sorted(exact), sorted(arrays)


def _portable_joblib_check(run_dir: Path) -> dict[str, bool]:
    manifest = read_json(run_dir / "fit/model_states/manifest.json")
    states = manifest["portable_states"]
    bundle_root = run_dir / "prepared" if (run_dir / "prepared").is_dir() else None
    if bundle_root is None:
        return {identifier: True for identifier in states}
    split_specs = (
        (bundle_root / "gate_fit_bundle.npz", run_dir / "fit/train_scores.npz"),
        (bundle_root / "gate_select_inference_bundle.npz", run_dir / "calibration/validation_scores.npz"),
        (bundle_root / "sealed_monitor_inference_bundle.npz", run_dir / "adjudication/monitor_scores.npz"),
    )
    checks: dict[str, bool] = {}
    for identifier, state in states.items():
        model_path = run_dir / "fit" / manifest["models"][identifier]["path"]
        if sha256_file(model_path) != manifest["models"][identifier]["sha256"]:
            checks[identifier] = False
            continue
        model = joblib.load(model_path)
        exact = True
        for bundle_path, scores_path in split_specs:
            with np.load(bundle_path, allow_pickle=False) as bundle, np.load(
                scores_path, allow_pickle=False
            ) as scores:
                active = np.asarray(bundle["disagreement"], dtype=bool)
                design = np.asarray(bundle["design"], dtype=np.float64)[active]
                if state["kind"] in {"ridge", "logistic"} and not getattr(
                    model, "_wave59_portable", False
                ):
                    design = (design - np.asarray(state["mean"])) / np.asarray(state["scale"])
                direct = (
                    model.predict_proba(design)[:, 1]
                    if state["kind"] in {"logistic", "hgb_classifier"}
                    else model.predict(design)
                )
                expected = np.asarray(scores[identifier], dtype=np.float64)[active]
                if not np.array_equal(np.asarray(direct, dtype=np.float64), expected):
                    exact = False
                    break
        checks[identifier] = exact
    return checks


def _normalize_operational(value: Any) -> Any:
    omitted = {
        "timestamp_utc",
        "execution_mode",
        "replay_exact",
        "superseded_output",
        "duration_seconds",
        "max_rss_bytes",
        "output_sha256",
        "output_inventory_before_receipt",
        "path_sha256",
    }
    if isinstance(value, dict):
        return {
            key: _normalize_operational(item)
            for key, item in value.items()
            if key not in omitted
        }
    if isinstance(value, list):
        return [_normalize_operational(item) for item in value]
    return value


def compare_runs(replay: Path, primary: Path) -> dict[str, Any]:
    replay = replay.resolve(strict=True)
    primary = primary.resolve(strict=True)
    if replay == primary:
        raise ValueError("Wave 59 replay cannot compare itself")
    canonical = (replay / "benchmark").is_dir() or (primary / "benchmark").is_dir()
    if canonical and not ((replay / "benchmark").is_dir() and (primary / "benchmark").is_dir()):
        raise RuntimeError("Wave 59 primary/replay canonical scope differs")
    exact_paths, array_paths = _scientific_paths(canonical)
    exact = {
        path: (replay / path).is_file()
        and (primary / path).is_file()
        and sha256_file(replay / path) == sha256_file(primary / path)
        for path in exact_paths
    }
    arrays = {
        path: (replay / path).is_file()
        and (primary / path).is_file()
        and _array_exact(replay / path, primary / path)
        for path in array_paths
    }
    secrets: dict[str, bool] = {}
    if canonical:
        for path in [
            "generation_escrow.json",
            *(f"benchmark/sealed/{name}" for name in (
                "calibration_null.jsonl",
                "train.jsonl",
                "val.jsonl",
                "lockbox.jsonl",
                "generation_secret.json",
                "identity_secret.json",
                "semantic_commitment_secret.json",
            )),
            "prepared/gate_fit_bundle.npz",
            "prepared/gate_select_truth_bundle.npz",
            "prepared/sealed_monitor_truth_bundle.npz",
        ]:
            secrets[path] = (replay / path).is_file() and (primary / path).is_file() and sha256_file(replay / path) == sha256_file(primary / path)
    functional = {
        "primary": _portable_joblib_check(primary),
        "replay": _portable_joblib_check(replay),
    }
    operational_paths = [
        "runtime.json",
        *(f"journals/{phase}.json" for phase in (
            "fit",
            "calibrate_scores",
            "validate",
            "monitor_apply",
            "monitor_evaluate",
        )),
    ]
    if canonical:
        operational_paths.extend(
            [
                "generation_receipt.json",
                "preparation_receipt.json",
                "inference/access_receipt.json",
                "journals/prepare.json",
            ]
        )
    operational = {
        path: (replay / path).is_file()
        and (primary / path).is_file()
        and _normalize_operational(read_json(replay / path))
        == _normalize_operational(read_json(primary / path))
        for path in operational_paths
    }
    all_exact = (
        all(exact.values())
        and all(arrays.values())
        and all(secrets.values())
        and all(functional["primary"].values())
        and all(functional["replay"].values())
        and all(operational.values())
    )
    result = {
        "schema_version": "wave59-replay-comparison-v1",
        "scientific_exact": exact,
        "scientific_array_exact": arrays,
        "secret_sha256_exact": secrets,
        "functional_state_exact": functional,
        "operational_semantic": operational,
        "all_exact": bool(all_exact),
    }
    if not all_exact:
        raise RuntimeError("Wave 59 replay differs from primary")
    return result


def _finalize_replay_condition(run_dir: Path, replay_exact: bool) -> None:
    analysis = read_json(run_dir / "analysis.json")
    for payload in analysis["prospective_patterns"].values():
        payload["conditions"]["replay_exact"] = bool(replay_exact)
        values = list(payload["conditions"].values())
        payload["replay_exact"] = bool(replay_exact)
        payload["aggregate_with_replay"] = (
            "NOT_EVALUABLE"
            if "NOT_EVALUABLE" in values or "PENDING" in values
            else bool(all(values))
        )
    write_json(run_dir / "analysis.json", analysis, mode=0o444)
    _write_report(run_dir)


def _artifact_classes(
    run_dir: Path, *, run_role: str, recovery_context: bool
) -> dict[str, list[str]]:
    exact, arrays = _scientific_paths(canonical=True)
    if recovery_context:
        exact.append("recovery_amendment.json")
    model_ids = [
        model_id("ridge"),
        model_id("hgb"),
        *(model_id(guard, target) for guard in ("logistic", "hgb") for target in ("harm", "posterior_incompatibility")),
        *(model_id("control-hgb", "harm", seed) for seed in HARM_CONTROL_SEEDS),
        *(model_id("control-hgb", "posterior_incompatibility", seed) for seed in INCOMPATIBILITY_CONTROL_SEEDS),
    ]
    operational = [
        "generation_receipt.json",
        "preparation_receipt.json",
        "inference/access_receipt.json",
        *(f"journals/{phase}.json" for phase in (
            "prepare",
            "fit",
            "calibrate_scores",
            "validate",
            "monitor_apply",
            "monitor_evaluate",
        )),
        "runtime.json",
    ]
    if run_role == "replay":
        operational.append("preparation_replay.json")
    secret = [
        "generation_escrow.json",
        *(f"benchmark/sealed/{name}" for name in (
            "calibration_null.jsonl",
            "train.jsonl",
            "val.jsonl",
            "lockbox.jsonl",
            "generation_secret.json",
            "identity_secret.json",
            "semantic_commitment_secret.json",
        )),
        "prepared/gate_fit_bundle.npz",
        "prepared/gate_select_truth_bundle.npz",
        "prepared/sealed_monitor_truth_bundle.npz",
    ]
    self_reference = ["artifact_manifest.json"]
    if run_role == "replay":
        self_reference.append("replay_comparison.json")
    return {
        "scientific_exact": sorted(exact),
        "scientific_array_exact": sorted(arrays),
        "functional_state": sorted(
            ["fit/model_states/manifest.json", *(f"fit/model_states/{identifier}.joblib" for identifier in model_ids)]
        ),
        "operational_semantic": sorted(operational),
        "secret_excluded_from_public_manifest": sorted(secret),
        "self_reference": sorted(self_reference),
    }


def write_artifact_manifest(run_dir: Path, *, run_role: str) -> dict[str, Any]:
    recovery_context = (run_dir / "recovery_amendment.json").is_file()
    classes = _artifact_classes(
        run_dir, run_role=run_role, recovery_context=recovery_context
    )
    flattened = [path for values in classes.values() for path in values]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Wave 59 artifact classes overlap")
    actual = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    expected = set(flattened)
    missing = expected - (actual | {"artifact_manifest.json"})
    extra = actual - expected
    if missing or extra:
        raise RuntimeError(
            f"Wave 59 closed artifact inventory mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    public: dict[str, Any] = {}
    for class_name, paths in classes.items():
        if class_name == "secret_excluded_from_public_manifest":
            public[class_name] = {"count": len(paths), "paths_redacted": True}
            continue
        if class_name == "self_reference":
            public[class_name] = {"paths": paths, "hashes_omitted": True}
            continue
        public[class_name] = {
            path: {"sha256": sha256_file(run_dir / path), "bytes": (run_dir / path).stat().st_size}
            for path in paths
        }
    config = read_json(run_dir / "config.snapshot.json")
    manifest = {
        "schema_version": "wave59-artifact-manifest-v1",
        "closed_world": True,
        "run_role": run_role,
        "recovery_context": recovery_context,
        "plan_sha256": config["plan"]["sha256"],
        "accepted_plan_audit_sha256": config["accepted_plan_audit"]["sha256"],
        "config_sha256": sha256_file(run_dir / "config.snapshot.json"),
        "classes": public,
        "coverage": {
            "missing": [],
            "extra": [],
            "overlap": [],
            "unclassified": [],
            "file_count": len(expected),
        },
    }
    write_json(run_dir / "artifact_manifest.json", manifest, mode=0o444)
    return manifest


def archive_failed_attempt(
    run_dir: Path,
    error: BaseException,
    *,
    run_role: str,
    recovery_context: bool,
) -> Path:
    """Preserve a failed canonical attempt without exposing escrow contents."""
    run_dir = run_dir.resolve(strict=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archived = run_dir.with_name(f"{run_dir.name}.failed_{stamp}")
    journals = []
    journal_root = run_dir / "journals"
    if journal_root.is_dir():
        for phase in (
            "prepare",
            "fit",
            "calibrate_scores",
            "validate",
            "monitor_apply",
            "monitor_evaluate",
        ):
            path = journal_root / f"{phase}.json"
            if path.is_file():
                journals.append(read_json(path))
    last = journals[-1] if journals else None
    message_hash = hashlib.sha256(str(error).encode("utf-8")).hexdigest()
    write_json(
        run_dir / "FAILURE.json",
        {
            "schema_version": "wave59-failed-attempt-v1",
            "error_type": type(error).__name__,
            "error_message_sha256": message_hash,
            "last_state": last.get("status") if last else None,
            "maximum_truth_materialized": (
                last.get("maximum_truth_materialized") if last else "none"
            ),
            "run_role": run_role,
            "recovery_context": bool(recovery_context),
            "original_path": str(run_dir),
            "archived_path": str(archived),
        },
        mode=0o600,
    )
    classes = _artifact_classes(
        run_dir, run_role=run_role, recovery_context=recovery_context
    )
    class_by_path = {
        path: class_name for class_name, paths in classes.items() for path in paths
    }
    records = []
    unknown = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "failure_inventory.json":
            continue
        relative = str(path.relative_to(run_dir))
        class_name = "failure_record" if relative == "FAILURE.json" else class_by_path.get(relative)
        if class_name is None:
            unknown.append(relative)
            continue
        record: dict[str, Any] = {
            "path": relative,
            "class": class_name,
            "bytes": path.stat().st_size,
        }
        if class_name != "secret_excluded_from_public_manifest":
            record["sha256"] = sha256_file(path)
        records.append(record)
    if unknown:
        raise RuntimeError(f"failed-attempt inventory has unclassified paths: {unknown}")
    write_json(
        run_dir / "failure_inventory.json",
        {
            "schema_version": "wave59-failure-inventory-v1",
            "records": records,
            "failure_records": ["FAILURE.json", "failure_inventory.json"],
            "missing_required_through_last_journal": [],
            "extra": [],
            "overlap": [],
            "unclassified": [],
        },
        mode=0o600,
    )
    os.replace(run_dir, archived)
    _fsync_directory(archived.parent)
    return archived


def _terminal_runtime(output: Path, status: str, started: float) -> None:
    write_json(
        output / "runtime.json",
        {
            "status": status,
            "device": "cpu",
            "cuda_visible_devices": "",
            "duration_seconds": time.monotonic() - started,
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            * 1024,
            "phases": [
                "fit",
                "calibrate_scores",
                "validate",
                "monitor_apply",
                "monitor_evaluate",
            ],
        },
    )


def execute(
    prepared: Path,
    policy_manifest: Path,
    output: Path,
    config_path: Path,
    reference_dir: Path | None = None,
) -> Path:
    started = time.monotonic()
    config = read_json(config_path)
    validate_pre_draw_config(config)
    prepared = prepared.resolve(strict=True)
    output = output.resolve(strict=False)
    canonical_existing = output == prepared and (prepared / "prepared").is_dir()
    bundle_root = prepared / "prepared" if canonical_existing else prepared
    if output.exists() and not canonical_existing:
        raise FileExistsError(output)
    if not output.exists():
        output.mkdir(parents=True)
    snapshot = output / "config.snapshot.json"
    if snapshot.exists():
        if sha256_file(snapshot) != sha256_file(config_path):
            raise RuntimeError("Wave 59 config snapshot differs from execution config")
    else:
        _copy(config_path, snapshot)
    if not (output / "source_bindings.json").exists():
        write_json(output / "source_bindings.json", config["source_binding"])
    utilities = load_utilities(policy_manifest)
    with tempfile.TemporaryDirectory(prefix="wave59-utilities-", dir="/tmp") as raw:
        utilities_path = Path(raw) / "utilities.npy"
        np.save(utilities_path, utilities)
        fit, fit_journal = _reuse_or_run_phase(
            output,
            "fit",
            {
                "config.json": config_path,
                "bundle.npz": bundle_root / "gate_fit_bundle.npz",
                "utilities.npy": utilities_path,
            },
            [bundle_root / "gate_select_truth_bundle.npz", bundle_root / "sealed_monitor_truth_bundle.npz"],
            destination_name="fit",
        )
        if fit_journal["status"] != "FIT_COMPLETE":
            _terminal_runtime(output, fit_journal["status"], started)
            return output
        calibration, calibration_journal = _reuse_or_run_phase(
            output,
            "calibrate_scores",
            {
                "config.json": config_path,
                "inference_bundle.npz": bundle_root / "gate_select_inference_bundle.npz",
                "model_states_manifest.json": fit / "model_states/manifest.json",
                "model_state_arrays.npz": fit / "model_state_arrays.npz",
                "fit_freeze.json": fit / "fit_freeze.json",
            },
            [bundle_root / "gate_select_truth_bundle.npz", bundle_root / "sealed_monitor_truth_bundle.npz"],
            destination_name="calibration",
        )
        if calibration_journal["status"] != "CALIBRATION_FROZEN":
            _terminal_runtime(output, calibration_journal["status"], started)
            return output
        validation, validation_journal = _reuse_or_run_phase(
            output,
            "validate",
            {
                "config.json": config_path,
                "inference_bundle.npz": bundle_root / "gate_select_inference_bundle.npz",
                "truth_bundle.npz": bundle_root / "gate_select_truth_bundle.npz",
                "validation_scores.npz": calibration / "validation_scores.npz",
                "validation_policy_arrays.npz": calibration / "validation_policy_arrays.npz",
                "calibration_freeze.json": calibration / "calibration_freeze.json",
                "utilities.npy": utilities_path,
            },
            [bundle_root / "sealed_monitor_truth_bundle.npz"],
            destination_name="validation",
        )
        if validation_journal["status"] != "VALIDATION_COMPLETE":
            _terminal_runtime(output, validation_journal["status"], started)
            return output
        adjudication, apply_journal = _reuse_or_run_phase(
            output,
            "monitor_apply",
            {
                "config.json": config_path,
                "inference_bundle.npz": bundle_root / "sealed_monitor_inference_bundle.npz",
                "model_states_manifest.json": fit / "model_states/manifest.json",
                "model_state_arrays.npz": fit / "model_state_arrays.npz",
                "fit_freeze.json": fit / "fit_freeze.json",
                "calibration_freeze.json": calibration / "calibration_freeze.json",
            },
            [bundle_root / "sealed_monitor_truth_bundle.npz"],
            destination_name="adjudication",
        )
        if apply_journal["status"] != "MONITOR_ACTIONS_FROZEN":
            _terminal_runtime(output, apply_journal["status"], started)
            return output
        evaluate_inputs = {
            "config.json": config_path,
            "truth_bundle.npz": bundle_root / "sealed_monitor_truth_bundle.npz",
            "monitor_policy_arrays.npz": adjudication / "monitor_policy_arrays.npz",
            "monitor_action_freeze.json": adjudication / "monitor_action_freeze.json",
            "utilities.npy": utilities_path,
        }
        evaluate_journal_path = output / "journals/monitor_evaluate.json"
        evaluation_already_promoted = (
            evaluate_journal_path.is_file()
            and (output / "analysis.json").is_file()
            and (adjudication / "bootstrap_indices.npz").is_file()
            and (adjudication / "analysis_arrays.npz").is_file()
        )
        if evaluation_already_promoted:
            evaluate_journal = read_json(evaluate_journal_path)
            if evaluate_journal.get("input_sha256") != {
                name: sha256_file(path.resolve(strict=True))
                for name, path in sorted(evaluate_inputs.items())
            }:
                raise RuntimeError("Wave 59 promoted monitor evaluation inputs differ")
            evaluation = None
        else:
            evaluation, evaluate_journal = _reuse_or_run_phase(
                output,
                "monitor_evaluate",
                evaluate_inputs,
                [],
                destination_name=".monitor_evaluate.complete",
            )
        if evaluate_journal["status"] != "COMPLETE":
            _terminal_runtime(output, evaluate_journal["status"], started)
            return output
        if evaluation is not None:
            _merge_evaluation(adjudication, evaluation, output)
    _terminal_runtime(output, "COMPLETE", started)
    _write_report(output)
    canonical = (output / "benchmark").is_dir()
    if reference_dir is not None:
        reference = reference_dir.resolve(strict=True)
        comparison = compare_runs(output, reference)
        write_json(output / "replay_comparison.json", comparison, mode=0o444)
        _finalize_replay_condition(reference, comparison["all_exact"])
        _finalize_replay_condition(output, comparison["all_exact"])
        if canonical:
            write_artifact_manifest(reference, run_role="primary")
            write_artifact_manifest(output, run_role="replay")
    elif canonical:
        write_artifact_manifest(output, run_role="primary")
    return output


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve(strict=False)
    prepared_arg = args.prepared_dir.resolve(strict=True)
    if args.resume_from is not None:
        restore_identical_hash_attempt(
            args.resume_from.resolve(strict=True),
            output,
            args.config.resolve(strict=True),
        )
        prepared_arg = output.resolve(strict=True)
    try:
        execute(
            prepared_arg,
            args.policy_manifest.resolve(strict=True),
            output,
            args.config.resolve(strict=True),
            args.reference_dir.resolve(strict=True) if args.reference_dir else None,
        )
    except BaseException as error:
        if output.is_dir() and (output / "config.snapshot.json").is_file():
            recovery_context = (output / "recovery_amendment.json").is_file()
            run_role = "replay" if args.reference_dir else "primary"
            archived = archive_failed_attempt(
                output,
                error,
                run_role=run_role,
                recovery_context=recovery_context,
            )
            print(
                json.dumps(
                    {"status": "FAILED", "archived_attempt": str(archived)},
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        raise


if __name__ == "__main__":
    main()
