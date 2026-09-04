#!/usr/bin/env python3
"""CPU-only functional ablation of frozen proportional-graph outputs.

This runner never imports torch, loads a checkpoint, or performs a neural
forward.  It combines the observed/corrected relations and unit/learned base
weights already preserved by the neural smoke, then sends them through the
same frozen WLS and Huber-IRLS solvers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import resource
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geometria_proporcional.proportional_graph_contract import (  # noqa: E402
    PublicGraphObservation,
    SolverResult,
    incidence_matrix,
    public_schema_hash,
    solve_huber_irls,
    solve_weighted_least_squares,
)

PLAN_PATH = (
    REPO_ROOT
    / "experiments/geometria_proporcional/PLAN_PROPORTIONAL_SOLVER_DISENTANGLEMENT_CPU.md"
)
RUNNER_PATH = Path(__file__).resolve()
SOLVER_PATH = REPO_ROOT / "src/geometria_proporcional/proportional_graph_contract.py"
CANONICAL_CONFIG_PATH = (
    REPO_ROOT
    / "experiments/geometria_proporcional/configs/proportional_graph_solver_disentanglement_v1.json"
)

RELATIONS = ("observed", "corrected")
BASE_WEIGHTS = ("unit", "learned")
SOLVERS = ("wls", "irls")
EFFECTS = (
    "relation_at_unit_weight",
    "weight_on_observed_relation",
    "relation_weight_interaction",
    "total_delivered_effect",
)
SLICES = ("test|iid", "test|grouped", "grouped_minus_iid")
METRICS = (
    "quotient_rmse",
    "relation_rmse",
    "weighted_residual_rmse",
    "laplacian_rank",
    "laplacian_condition",
    "converged",
    "iterations",
)
EXPECTED_ARMS = (
    "raw_generic",
    "raw_typed",
    "closure_generic",
    "closure_typed",
    "closure_typed_path_shuffle",
    "pair_state_no_mix",
    "generic_message_passing",
    "edge_mlp",
)
EXPECTED_SEEDS = (104729, 130363)
EXPECTED_PRIMARY_ARMS = EXPECTED_ARMS[:4]
EXPECTED_SECONDARY_ARMS = EXPECTED_ARMS[4:]
EXPECTED_REUSE_INDICES = (0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 630)
EXPECTED_SOLVER = {
    "weight_floor": 0.001,
    "huber_delta": 1.5,
    "irls_damping": 1.0,
    "irls_iterations": 7500,
    "tolerance": 1e-6,
    "reuse_atol": 1e-12,
}
EXPECTED_EXECUTION = {
    "device": "cpu",
    "numeric_threads": 1,
    "max_seconds": 7200,
    "max_rss_gib": 8.0,
}


@dataclass(frozen=True)
class CompletedState:
    x_hat: np.ndarray
    reconstructed_log_ratio: np.ndarray
    weights: np.ndarray
    quotient_rmse: float
    relation_rmse: float
    weighted_residual_rmse: float
    laplacian_rank: int
    laplacian_condition: float
    converged: bool
    iterations: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CANONICAL_CONFIG_PATH)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--development",
        action="store_true",
        help="permit untracked runner inputs; source hashes and CPU guards still apply",
    )
    return parser.parse_args()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(data: Any) -> str:
    return json.dumps(
        _json_safe(data),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(data),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_clean_records(paths: Iterable[Path], development: bool) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for path in paths:
        path = path.resolve(strict=True)
        try:
            relative = str(path.relative_to(REPO_ROOT))
        except ValueError:
            if not development:
                raise RuntimeError(f"official source must live in repository: {path}")
            relative = str(path)
        tracked = (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", relative],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )
        dirty = tracked and (
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", relative],
                cwd=REPO_ROOT,
            ).returncode
            != 0
        )
        if not development and (not tracked or dirty):
            raise RuntimeError(f"official source must be tracked and clean: {relative}")
        records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "tracked": tracked,
            "dirty": dirty,
        }
    return records


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected_top = {
        "schema_version",
        "artifact_schema_version",
        "source",
        "arms",
        "seeds",
        "solver",
        "analysis",
        "execution",
    }
    if set(config) != expected_top:
        raise ValueError(f"config keys must be exactly {sorted(expected_top)}")
    if config["schema_version"] != "proportional-graph-solver-disentanglement-v1":
        raise ValueError("unexpected config schema")
    if config["artifact_schema_version"] != (
        "proportional-graph-solver-disentanglement-artifact-v1"
    ):
        raise ValueError("unexpected artifact schema")
    source_keys = {
        "path",
        "manifest_sha256",
        "artifact_schema_version",
        "git_head",
        "public_schema_sha256",
    }
    if set(config["source"]) != source_keys:
        raise ValueError("unexpected source contract")
    expected_source = {
        "path": "data/geometria_proporcional/proportional_graph_neural_smoke_v1",
        "manifest_sha256": "7e982a92bd366a4c22fe95c0cc9c8fd5f7a1773bd78c4e8e0781b5c2259624e1",
        "artifact_schema_version": "proportional-graph-neural-artifact-v1",
        "git_head": "3a683ac9a7ef444e344b746cdd83062ff03ff30a",
        "public_schema_sha256": "0a219cb4991b7696291512d5aa307684bcb4960058caf88e273f8af138edf6af",
    }
    if config["source"] != expected_source:
        raise ValueError("source contract differs from the frozen protocol")
    if tuple(config["arms"]) != EXPECTED_ARMS:
        raise ValueError("arms differ from the frozen protocol")
    if tuple(config["seeds"]) != EXPECTED_SEEDS:
        raise ValueError("seeds differ from the frozen protocol")
    analysis_keys = {
        "relations",
        "base_weights",
        "solvers",
        "primary_arms",
        "secondary_arms",
        "effect_order",
        "slice_order",
        "reuse_verification_indices",
        "bootstrap_replicates",
        "expected_validation_views",
        "expected_test_views",
        "expected_test_masters",
        "strict_irls_failures",
        "inference_regime",
    }
    if set(config["analysis"]) != analysis_keys:
        raise ValueError("unexpected analysis contract")
    if tuple(config["analysis"]["relations"]) != RELATIONS:
        raise ValueError("relation order must be observed, corrected")
    if tuple(config["analysis"]["base_weights"]) != BASE_WEIGHTS:
        raise ValueError("weight order must be unit, learned")
    if tuple(config["analysis"]["solvers"]) != SOLVERS:
        raise ValueError("solver order must be wls, irls")
    if tuple(config["analysis"]["effect_order"]) != EFFECTS:
        raise ValueError("effect order differs from the frozen plan")
    if tuple(config["analysis"]["slice_order"]) != SLICES:
        raise ValueError("slice order differs from the frozen plan")
    if tuple(config["analysis"]["primary_arms"]) != EXPECTED_PRIMARY_ARMS:
        raise ValueError("primary arms differ from the frozen protocol")
    if tuple(config["analysis"]["secondary_arms"]) != EXPECTED_SECONDARY_ARMS:
        raise ValueError("secondary arms differ from the frozen protocol")
    if tuple(config["analysis"]["reuse_verification_indices"]) != EXPECTED_REUSE_INDICES:
        raise ValueError("reuse verification indices differ from the frozen protocol")
    expected_analysis_scalars = {
        "bootstrap_replicates": 2000,
        "expected_validation_views": 127,
        "expected_test_views": 504,
        "expected_test_masters": 252,
        "strict_irls_failures": True,
        "inference_regime": "exploratory_posthoc_marginal_ci_no_pvalues",
    }
    for key, expected in expected_analysis_scalars.items():
        if config["analysis"][key] != expected:
            raise ValueError(f"analysis value differs from frozen protocol: {key}")
    solver = config["solver"]
    if set(solver) != set(EXPECTED_SOLVER):
        raise ValueError("unexpected solver contract")
    if solver != EXPECTED_SOLVER:
        raise ValueError("solver values differ from the frozen protocol")
    execution = config["execution"]
    if set(execution) != set(EXPECTED_EXECUTION) or execution != EXPECTED_EXECUTION:
        raise ValueError("execution contract differs from the frozen protocol")
    return config


def _guard_cpu_environment(config: dict[str, Any]) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be set to the empty string")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be set to 1")
    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported by this CPU-only runner")
    if config["execution"]["numeric_threads"] != 1:
        raise RuntimeError("numeric thread contract changed")


def _prepare_output(output: Path, source: Path) -> None:
    output = output.resolve()
    source = source.resolve(strict=True)
    if output == source or output.is_relative_to(source):
        raise ValueError("output may not equal or descend from the source package")
    if source.is_relative_to(output):
        raise ValueError("output may not contain the source package")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _verify_member(source: Path, manifest: dict[str, Any], relative: str) -> dict[str, Any]:
    if relative not in manifest["files"]:
        raise ValueError(f"source manifest does not attest {relative}")
    path = source / relative
    record = manifest["files"][relative]
    actual = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    if actual != record:
        raise ValueError(f"source member mismatch: {relative}")
    return actual


def _exact_equal(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
        raise ValueError(f"unaligned source array: {label}")


def _validate_ragged_paths(arrays: dict[str, np.ndarray], label: str) -> None:
    required = {"path_offsets", "path_index", "path_sign", "path_valid", "edge_offsets"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"{label}: missing path arrays {sorted(missing)}")
    offsets = arrays["path_offsets"]
    edge_offsets = arrays["edge_offsets"]
    indices = arrays["path_index"]
    signs = arrays["path_sign"]
    valid = arrays["path_valid"]
    if offsets.dtype != np.int64 or offsets.ndim != 1:
        raise ValueError(f"{label}: path_offsets must be int64 rank one")
    if edge_offsets.dtype != np.int64 or edge_offsets.shape != offsets.shape:
        raise ValueError(f"{label}: path and edge offsets must align by view")
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(indices):
        raise ValueError(f"{label}: invalid path offset endpoints")
    if np.any(np.diff(offsets) < 0):
        raise ValueError(f"{label}: path offsets must be monotone")
    if indices.dtype != np.int64 or indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"{label}: path_index must be int64 [n,3]")
    if signs.shape != indices.shape or not np.issubdtype(signs.dtype, np.floating):
        raise ValueError(f"{label}: path_sign must be floating and path-aligned")
    if not np.all(np.isfinite(signs)) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError(f"{label}: path_sign must contain finite orientations")
    if valid.dtype != np.bool_ or valid.shape != (len(indices),):
        raise ValueError(f"{label}: path_valid must be boolean and path-aligned")
    for view_index in range(len(offsets) - 1):
        path_slice = slice(int(offsets[view_index]), int(offsets[view_index + 1]))
        edge_count = int(edge_offsets[view_index + 1] - edge_offsets[view_index])
        local = indices[path_slice]
        if len(local) and (np.any(local < 0) or np.any(local >= edge_count)):
            raise ValueError(f"{label}: path index outside view {view_index}")


def _validate_raw_alignment(
    canonical: dict[str, np.ndarray], raw: dict[str, np.ndarray], arm: str
) -> dict[str, Any]:
    _validate_ragged_paths(canonical, "canonical")
    _validate_ragged_paths(raw, arm)
    exact_keys = (
        "edge_offsets",
        "node_offsets",
        "n_nodes",
        "view_id",
        "master_id",
        "split",
        "mechanism",
        "edge_index",
        "edge_valid",
        "observed_log_ratio",
        "clean_log_ratio",
        "x_true",
        "path_valid",
    )
    for key in exact_keys:
        _exact_equal(canonical[key], raw[key], f"{arm}:{key}")
    if raw["edge_variance"].dtype != np.float32 or not np.array_equal(
        raw["edge_variance"], canonical["edge_variance"].astype(np.float32)
    ):
        raise ValueError(f"{arm}: edge_variance is not the expected float32 projection")
    shuffled = arm == "closure_typed_path_shuffle"
    if not shuffled:
        _exact_equal(canonical["path_offsets"], raw["path_offsets"], f"{arm}:path_offsets")
        _exact_equal(canonical["path_index"], raw["path_index"], f"{arm}:path_index")
        if raw["path_sign"].dtype != np.float32 or not np.array_equal(
            raw["path_sign"], canonical["path_sign"].astype(np.float32)
        ):
            raise ValueError(f"{arm}: path_sign is not the expected projection")
    else:
        if raw["path_offsets"].shape != canonical["path_offsets"].shape:
            raise ValueError("path-shuffle offsets have the wrong shape")
        if np.array_equal(raw["path_index"], canonical["path_index"]):
            raise ValueError("path-shuffle raw unexpectedly retained canonical paths")
    required = {
        "corrected_log_ratio",
        "reliability",
        "x_hat_wls",
        "x_hat_irls",
        "irls_weights",
        "wls_laplacian_rank",
        "wls_condition",
        "irls_laplacian_rank",
        "irls_condition",
        "irls_converged",
        "irls_iterations",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{arm}: missing raw arrays {sorted(missing)}")
    n_edges = len(canonical["observed_log_ratio"])
    n_nodes = len(canonical["x_true"])
    if any(len(raw[key]) != n_edges for key in ("corrected_log_ratio", "reliability", "irls_weights")):
        raise ValueError(f"{arm}: edge-aligned raw array has wrong length")
    if any(len(raw[key]) != n_nodes for key in ("x_hat_wls", "x_hat_irls")):
        raise ValueError(f"{arm}: node-aligned raw array has wrong length")
    if not np.all(np.isfinite(raw["corrected_log_ratio"])):
        raise ValueError(f"{arm}: non-finite corrected relation")
    if not np.all(np.isfinite(raw["reliability"])) or np.any(raw["reliability"] <= 0) or np.any(raw["reliability"] > 1):
        raise ValueError(f"{arm}: reliability outside (0, 1]")
    return {
        "canonical_exact_fields": list(exact_keys),
        "edge_variance_projection": "float64_to_float32_exact",
        "path_policy": "intervened_shuffle" if shuffled else "canonical_with_float32_sign",
    }


def _verify_source(
    source: Path,
    config: dict[str, Any],
    *,
    started: float | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    source = source.resolve(strict=True)
    manifest_path = source / "manifest.json"
    if _sha256_file(manifest_path) != config["source"]["manifest_sha256"]:
        raise ValueError("source manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != config["source"]["artifact_schema_version"]:
        raise ValueError("source artifact schema mismatch")
    if manifest["git"]["head"] != config["source"]["git_head"] or manifest["git"]["dirty"]:
        raise ValueError("source git attestation mismatch")
    if manifest["public_schema_sha256"] != config["source"]["public_schema_sha256"]:
        raise ValueError("source public schema mismatch")
    if public_schema_hash() != config["source"]["public_schema_sha256"]:
        raise ValueError("current solver public schema differs from source")
    solver_relative = "src/geometria_proporcional/proportional_graph_contract.py"
    if manifest["source_files"][solver_relative]["sha256"] != _sha256_file(SOLVER_PATH):
        raise ValueError("current solver implementation differs from source solver")

    resolved_relative = "resolved_config.json"
    control_relative = "raw_eval/observed_unweighted.npz"
    bootstrap_relative = "bootstrap_indices.npz"
    used_records = {
        relative: _verify_member(source, manifest, relative)
        for relative in (resolved_relative, control_relative, bootstrap_relative)
    }
    source_config = json.loads((source / resolved_relative).read_text(encoding="utf-8"))
    source_arms = [arm["name"] for arm in source_config["arms"]]
    source_seeds = source_config["training"]["seeds"]
    if source_arms != config["arms"] or source_seeds != config["seeds"]:
        raise ValueError("arms or seeds differ from the frozen source")
    graph = source_config["graph"]
    expected_solver = {
        "weight_floor": graph["weight_floor"],
        "huber_delta": graph["huber_delta"],
        "irls_damping": graph["irls_damping"],
        "irls_iterations": graph["irls_iterations"],
    }
    for key, value in expected_solver.items():
        if config["solver"][key] != value:
            raise ValueError(f"solver setting differs from source: {key}")

    canonical = _load_npz(source / control_relative)
    raw_by_key: dict[str, dict[str, np.ndarray]] = {}
    alignment: dict[str, Any] = {}
    for arm in config["arms"]:
        for seed in config["seeds"]:
            key = f"{arm}|seed={seed}"
            relative = f"raw_eval/{key}.npz"
            used_records[relative] = _verify_member(source, manifest, relative)
            raw = _load_npz(source / relative)
            alignment[key] = _validate_raw_alignment(canonical, raw, arm)
            raw_by_key[key] = raw
            if started is not None:
                _enforce_budget(started, config, f"source attestation {key}")

    bootstrap = _load_npz(source / bootstrap_relative)
    expected_replicates = config["analysis"]["bootstrap_replicates"]
    master_ids = bootstrap.get("master_id")
    index_key = f"complete_n_{config['analysis']['expected_test_masters']}"
    if master_ids is None or index_key not in bootstrap:
        raise ValueError("source bootstrap lacks frozen master IDs or indices")
    indices = bootstrap[index_key]
    if indices.shape != (expected_replicates, len(master_ids)):
        raise ValueError("bootstrap index shape mismatch")
    if np.any(indices < 0) or np.any(indices >= len(master_ids)):
        raise ValueError("bootstrap index outside master universe")

    splits, counts = np.unique(canonical["split"], return_counts=True)
    split_counts = dict(zip(splits.tolist(), counts.tolist()))
    if split_counts != {
        "test": config["analysis"]["expected_test_views"],
        "validation": config["analysis"]["expected_validation_views"],
    }:
        raise ValueError(f"unexpected source split counts: {split_counts}")
    test_pairs: dict[str, set[str]] = {}
    for master_id, split, mechanism in zip(
        canonical["master_id"], canonical["split"], canonical["mechanism"]
    ):
        if split == "test":
            test_pairs.setdefault(str(master_id), set()).add(str(mechanism))
    if set(test_pairs) != set(master_ids.tolist()) or any(
        mechanisms != {"iid", "grouped"} for mechanisms in test_pairs.values()
    ):
        raise ValueError("test master universe does not match bootstrap pairing")

    attestation = {
        "source_path": str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source),
        "manifest_sha256": _sha256_file(manifest_path),
        "artifact_schema_version": manifest["schema_version"],
        "source_git": manifest["git"],
        "public_schema_sha256": manifest["public_schema_sha256"],
        "used_files": dict(sorted(used_records.items())),
        "alignment": alignment,
        "split_counts": split_counts,
        "test_master_order": "bootstrap_indices.npz:master_id",
        "test_masters": len(master_ids),
    }
    return canonical, raw_by_key, bootstrap, attestation


def _write_deterministic_npz(path: Path, **arrays: Any) -> None:
    """Write an NPZ whose member order and ZIP metadata are replay-stable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asanyarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _resource_observation(started: float) -> dict[str, float]:
    return {
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024.0 * 1024.0),
    }


def _enforce_budget(started: float, config: dict[str, Any], stage: str) -> None:
    observed = _resource_observation(started)
    if observed["elapsed_seconds"] > float(config["execution"]["max_seconds"]):
        raise RuntimeError(f"CPU time budget exceeded during {stage}: {observed}")
    if observed["peak_rss_gib"] > float(config["execution"]["max_rss_gib"]):
        raise RuntimeError(f"RSS budget exceeded during {stage}: {observed}")


def _view_slices(canonical: dict[str, np.ndarray], index: int) -> tuple[slice, slice, slice]:
    return (
        slice(int(canonical["edge_offsets"][index]), int(canonical["edge_offsets"][index + 1])),
        slice(int(canonical["node_offsets"][index]), int(canonical["node_offsets"][index + 1])),
        slice(int(canonical["path_offsets"][index]), int(canonical["path_offsets"][index + 1])),
    )


def _public_view(canonical: dict[str, np.ndarray], index: int) -> PublicGraphObservation:
    edge_slice, _, path_slice = _view_slices(canonical, index)
    return PublicGraphObservation(
        n_nodes=int(canonical["n_nodes"][index]),
        edge_index=canonical["edge_index"][edge_slice],
        observed_log_ratio=canonical["observed_log_ratio"][edge_slice],
        edge_valid=canonical["edge_valid"][edge_slice],
        path_index=canonical["path_index"][path_slice],
        path_sign=canonical["path_sign"][path_slice],
        path_valid=canonical["path_valid"][path_slice],
        edge_variance=canonical["edge_variance"][edge_slice],
    )


def _score_state(
    result: SolverResult,
    x_true: np.ndarray,
    clean_relation: np.ndarray,
) -> CompletedState:
    target = np.asarray(x_true, dtype=np.float64)
    target = target - target.mean()
    raw_quotient = float(np.sqrt(np.mean((result.x_hat - target) ** 2)))
    raw_relation = float(
        np.sqrt(np.mean((result.reconstructed_log_ratio - clean_relation) ** 2))
    )
    inferential_quotient = raw_quotient if result.converged else float("nan")
    inferential_relation = raw_relation if result.converged else float("nan")
    inferential_residual = (
        result.weighted_residual_rmse if result.converged else float("nan")
    )
    return CompletedState(
        x_hat=np.asarray(result.x_hat, dtype=np.float64),
        reconstructed_log_ratio=np.asarray(
            result.reconstructed_log_ratio, dtype=np.float64
        ),
        weights=np.asarray(result.weights, dtype=np.float64),
        quotient_rmse=inferential_quotient,
        relation_rmse=inferential_relation,
        weighted_residual_rmse=inferential_residual,
        laplacian_rank=int(result.laplacian_rank),
        laplacian_condition=float(result.laplacian_condition),
        converged=bool(result.converged),
        iterations=int(result.iterations),
    )


def _solve_cell(
    observation: PublicGraphObservation,
    values: np.ndarray,
    base_weights: np.ndarray,
    solver_name: str,
    config: dict[str, Any],
    x_true: np.ndarray,
    clean_relation: np.ndarray,
) -> CompletedState:
    solver = config["solver"]
    if solver_name == "wls":
        result = solve_weighted_least_squares(
            observation,
            values=values,
            weights=base_weights,
            weight_floor=float(solver["weight_floor"]),
        )
    elif solver_name == "irls":
        result = solve_huber_irls(
            observation,
            values=values,
            base_weights=base_weights,
            delta=float(solver["huber_delta"]),
            max_iterations=int(solver["irls_iterations"]),
            damping=float(solver["irls_damping"]),
            weight_floor=float(solver["weight_floor"]),
            tolerance=float(solver["tolerance"]),
        )
    else:
        raise ValueError(f"unknown solver: {solver_name}")
    return _score_state(result, x_true, clean_relation)


def _stored_state(
    canonical: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    index: int,
    solver_name: str,
    values: np.ndarray,
) -> CompletedState:
    edge_slice, node_slice, _ = _view_slices(canonical, index)
    observation = _public_view(canonical, index)
    x_hat = np.asarray(raw[f"x_hat_{solver_name}"][node_slice], dtype=np.float64)
    reconstructed = incidence_matrix(observation.n_nodes, observation.edge_index) @ x_hat
    if solver_name == "wls":
        valid = np.asarray(observation.edge_valid, dtype=bool)
        weights = np.zeros(len(valid), dtype=np.float64)
        weights[valid] = 1.0
        rank = int(raw["wls_laplacian_rank"][index])
        condition = float(raw["wls_condition"][index])
        converged = True
        iterations = 1
    elif solver_name == "irls":
        weights = np.asarray(raw["irls_weights"][edge_slice], dtype=np.float64)
        rank = int(raw["irls_laplacian_rank"][index])
        condition = float(raw["irls_condition"][index])
        converged = bool(raw["irls_converged"][index])
        iterations = int(raw["irls_iterations"][index])
    else:
        raise ValueError(f"unknown stored solver: {solver_name}")
    valid = np.asarray(observation.edge_valid, dtype=bool)
    residual = reconstructed[valid] - np.asarray(values, dtype=np.float64)[valid]
    residual_rmse = float(np.sqrt(np.average(residual**2, weights=weights[valid])))
    result = SolverResult(
        x_hat=x_hat,
        reconstructed_log_ratio=reconstructed,
        weights=weights,
        quotient_rmse=float("nan"),
        relation_rmse=float("nan"),
        weighted_residual_rmse=residual_rmse,
        laplacian_rank=rank,
        laplacian_condition=condition,
        converged=converged,
        iterations=iterations,
    )
    return _score_state(
        result,
        canonical["x_true"][node_slice],
        canonical["clean_log_ratio"][edge_slice],
    )


def _assert_state_close(
    actual: CompletedState,
    expected: CompletedState,
    atol: float,
    label: str,
) -> None:
    for field in ("x_hat", "reconstructed_log_ratio", "weights"):
        if not np.allclose(
            getattr(actual, field), getattr(expected, field), rtol=0.0, atol=atol
        ):
            delta = float(np.max(np.abs(getattr(actual, field) - getattr(expected, field))))
            raise AssertionError(f"stored-state mismatch {label}:{field}; max_abs={delta}")
    if actual.converged != expected.converged or actual.iterations != expected.iterations:
        raise AssertionError(f"stored-state status mismatch {label}")


CELL_IDS = tuple(
    f"{relation}|{weight}|{solver}"
    for solver in SOLVERS
    for relation in RELATIONS
    for weight in BASE_WEIGHTS
)


def _cell_index(relation: str, weight: str, solver: str) -> int:
    return CELL_IDS.index(f"{relation}|{weight}|{solver}")


def _state_metric(state: CompletedState, metric: str) -> float | int | bool:
    return getattr(state, metric)


def _assert_wls_source(
    state: CompletedState,
    source_raw: dict[str, np.ndarray],
    view_index: int,
    node_slice: slice,
    atol: float,
    label: str,
) -> None:
    stored_xhat = source_raw["x_hat_wls"][node_slice]
    if not np.allclose(state.x_hat, stored_xhat, rtol=0.0, atol=atol):
        delta = float(np.max(np.abs(state.x_hat - stored_xhat)))
        raise AssertionError(f"stored WLS mismatch {label}; max_abs={delta}")
    if not np.isclose(
        state.quotient_rmse,
        float(source_raw["wls_quotient_rmse"][view_index]),
        rtol=0.0,
        atol=atol,
    ):
        raise AssertionError(f"stored WLS quotient metric mismatch {label}")
    if state.laplacian_rank != int(source_raw["wls_laplacian_rank"][view_index]):
        raise AssertionError(f"stored WLS rank mismatch {label}")
    if not np.isclose(
        state.laplacian_condition,
        float(source_raw["wls_condition"][view_index]),
        rtol=0.0,
        atol=atol,
    ):
        raise AssertionError(f"stored WLS condition mismatch {label}")


def _run_model_matrix(
    canonical: dict[str, np.ndarray],
    source_control: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    arm: str,
    seed: int,
    config: dict[str, Any],
    started: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    n_views = len(canonical["view_id"])
    total_nodes = len(canonical["x_true"])
    total_edges = len(canonical["observed_log_ratio"])
    state_arrays = {
        "x_hat": np.full((len(CELL_IDS), total_nodes), np.nan, dtype=np.float64),
        "reconstructed_log_ratio": np.full(
            (len(CELL_IDS), total_edges), np.nan, dtype=np.float64
        ),
        "final_weights": np.full(
            (len(CELL_IDS), total_edges), np.nan, dtype=np.float64
        ),
    }
    metric_arrays: dict[str, np.ndarray] = {
        metric: np.full(
            (len(CELL_IDS), n_views),
            False if metric == "converged" else -1 if metric in {"laplacian_rank", "iterations"} else np.nan,
            dtype=bool if metric == "converged" else np.int64 if metric in {"laplacian_rank", "iterations"} else np.float64,
        )
        for metric in METRICS
    }
    raw_finite_scores = {
        "last_quotient_rmse": np.full((len(CELL_IDS), n_views), np.nan),
        "last_relation_rmse": np.full((len(CELL_IDS), n_views), np.nan),
    }
    reused_counts = {"wls_verified": 0, "irls_verified": 0}
    verification_indices = set(config["analysis"]["reuse_verification_indices"])
    atol = float(config["solver"]["reuse_atol"])

    for view_index in range(n_views):
        edge_slice, node_slice, _ = _view_slices(canonical, view_index)
        observation = _public_view(canonical, view_index)
        relations = {
            "observed": canonical["observed_log_ratio"][edge_slice],
            "corrected": raw["corrected_log_ratio"][edge_slice],
        }
        weights = {
            "unit": np.ones(edge_slice.stop - edge_slice.start, dtype=np.float64),
            "learned": raw["reliability"][edge_slice],
        }
        x_true = canonical["x_true"][node_slice]
        clean = canonical["clean_log_ratio"][edge_slice]
        for solver_name in SOLVERS:
            for relation_name in RELATIONS:
                for weight_name in BASE_WEIGHTS:
                    cell = _cell_index(relation_name, weight_name, solver_name)
                    is_reused_irls = solver_name == "irls" and (
                        (relation_name, weight_name) == ("observed", "unit")
                        or (relation_name, weight_name) == ("corrected", "learned")
                    )
                    if is_reused_irls:
                        source_raw = (
                            source_control
                            if (relation_name, weight_name) == ("observed", "unit")
                            else raw
                        )
                        state = _stored_state(
                            canonical,
                            source_raw,
                            view_index,
                            "irls",
                            relations[relation_name],
                        )
                        if view_index in verification_indices:
                            recomputed = _solve_cell(
                                observation,
                                relations[relation_name],
                                weights[weight_name],
                                solver_name,
                                config,
                                x_true,
                                clean,
                            )
                            _assert_state_close(
                                recomputed,
                                state,
                                atol,
                                f"{arm}|seed={seed}|view={view_index}|{CELL_IDS[cell]}",
                            )
                            reused_counts["irls_verified"] += 1
                    else:
                        state = _solve_cell(
                            observation,
                            relations[relation_name],
                            weights[weight_name],
                            solver_name,
                            config,
                            x_true,
                            clean,
                        )
                    if solver_name == "wls":
                        if (relation_name, weight_name) == ("observed", "unit"):
                            _assert_wls_source(
                                state,
                                source_control,
                                view_index,
                                node_slice,
                                atol,
                                f"control|view={view_index}",
                            )
                            reused_counts["wls_verified"] += 1
                        elif (relation_name, weight_name) == ("corrected", "learned"):
                            _assert_wls_source(
                                state,
                                raw,
                                view_index,
                                node_slice,
                                atol,
                                f"{arm}|seed={seed}|view={view_index}",
                            )
                            reused_counts["wls_verified"] += 1

                    state_arrays["x_hat"][cell, node_slice] = state.x_hat
                    state_arrays["reconstructed_log_ratio"][cell, edge_slice] = (
                        state.reconstructed_log_ratio
                    )
                    state_arrays["final_weights"][cell, edge_slice] = state.weights
                    for metric in METRICS:
                        metric_arrays[metric][cell, view_index] = _state_metric(state, metric)
                    target = x_true - x_true.mean()
                    raw_finite_scores["last_quotient_rmse"][cell, view_index] = float(
                        np.sqrt(np.mean((state.x_hat - target) ** 2))
                    )
                    raw_finite_scores["last_relation_rmse"][cell, view_index] = float(
                        np.sqrt(np.mean((state.reconstructed_log_ratio - clean) ** 2))
                    )
        if view_index % 16 == 0:
            _enforce_budget(started, config, f"{arm}|seed={seed}|view={view_index}")

    raw_payload = {
        **state_arrays,
        **metric_arrays,
        **raw_finite_scores,
        "cell_id": np.asarray(CELL_IDS),
        "edge_offsets": canonical["edge_offsets"],
        "node_offsets": canonical["node_offsets"],
        "view_id": canonical["view_id"],
        "master_id": canonical["master_id"],
        "split": canonical["split"],
        "mechanism": canonical["mechanism"],
        "arm": np.asarray(arm),
        "seed": np.asarray(seed, dtype=np.int64),
    }
    return raw_payload, metric_arrays, reused_counts


def _strict_seed_mean(values: np.ndarray) -> np.ndarray:
    """Average the seed axis only when every seed is finite."""
    array = np.asarray(values, dtype=np.float64)
    return np.where(np.all(np.isfinite(array), axis=0), np.mean(array, axis=0), np.nan)


def _bootstrap_summary(
    values: np.ndarray,
    bootstrap_indices: np.ndarray,
    *,
    strict: bool,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    diagnostic = {
        "n_total": int(len(values)),
        "n_finite": int(finite.sum()),
        "failure_rate": float(np.mean(~finite)),
        "finite_mean": float(values[finite].mean()) if np.any(finite) else None,
    }
    if strict and not np.all(finite):
        return {
            "status": "NOT_EVALUABLE_SOLVER_FAILURE",
            "mean": None,
            "ci95_marginal": [None, None],
            "n": 0,
            "diagnostic_finite_survivors": diagnostic,
        }
    if not np.all(finite):
        raise AssertionError("non-IRLS estimand unexpectedly contains non-finite values")
    draws = values[bootstrap_indices].mean(axis=1)
    return {
        "status": "ESTIMATED",
        "mean": float(values.mean()),
        "ci95_marginal": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
        "n": int(len(values)),
        "bootstrap_replicates": int(len(draws)),
        "diagnostic_finite_survivors": diagnostic,
    }


def _master_view_indices(
    canonical: dict[str, np.ndarray], master_order: np.ndarray
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for mechanism in ("iid", "grouped"):
        lookup: dict[str, int] = {}
        for index, (master, split, actual_mechanism) in enumerate(
            zip(canonical["master_id"], canonical["split"], canonical["mechanism"])
        ):
            if split == "test" and actual_mechanism == mechanism:
                if str(master) in lookup:
                    raise ValueError(f"duplicate test master/mechanism: {master}/{mechanism}")
                lookup[str(master)] = index
        if set(lookup) != set(master_order.tolist()):
            raise ValueError(f"master mapping differs for {mechanism}")
        output[mechanism] = np.asarray(
            [lookup[str(master)] for master in master_order], dtype=np.int64
        )
    return output


def _slices(iid: np.ndarray, grouped: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "test|iid": iid,
        "test|grouped": grouped,
        "grouped_minus_iid": grouped - iid,
    }


def _factorial_effects(cell_values: np.ndarray) -> dict[str, np.ndarray]:
    """Return the four frozen effects from a [relation, weight, unit] tensor."""
    q00 = cell_values[0, 0]
    q10 = cell_values[1, 0]
    q01 = cell_values[0, 1]
    q11 = cell_values[1, 1]
    output = {
        "relation_at_unit_weight": q10 - q00,
        "weight_on_observed_relation": q01 - q00,
        "relation_weight_interaction": q11 - q10 - q01 + q00,
        "total_delivered_effect": q11 - q00,
    }
    identity_delta = output["total_delivered_effect"] - (
        output["relation_at_unit_weight"]
        + output["weight_on_observed_relation"]
        + output["relation_weight_interaction"]
    )
    finite = np.isfinite(identity_delta)
    if np.any(finite) and np.max(np.abs(identity_delta[finite])) > 1e-12:
        raise AssertionError("factorial identity failed")
    return output


def _build_summary(
    config: dict[str, Any],
    canonical: dict[str, np.ndarray],
    metric_tensor: dict[str, np.ndarray],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(config["arms"]):
        for seed_index, seed in enumerate(config["seeds"]):
            for cell, cell_id in enumerate(CELL_IDS):
                relation, weight, solver = cell_id.split("|")
                for split in ("validation", "test"):
                    mechanisms = (
                        ("all",) if split == "validation" else ("iid", "grouped")
                    )
                    for mechanism in mechanisms:
                        mask = canonical["split"] == split
                        if mechanism != "all":
                            mask &= canonical["mechanism"] == mechanism
                        record: dict[str, Any] = {
                            "arm": arm,
                            "seed": int(seed),
                            "relation": relation,
                            "base_weight": weight,
                            "solver": solver,
                            "split": split,
                            "mechanism": mechanism,
                            "n_views": int(mask.sum()),
                        }
                        for metric in METRICS:
                            values = metric_tensor[metric][arm_index, seed_index, cell, mask]
                            if metric == "converged":
                                record["converged_count"] = int(np.sum(values))
                                record["failure_count"] = int(np.sum(~values))
                                record["failure_rate"] = float(np.mean(~values))
                            elif metric in {"laplacian_rank", "iterations"}:
                                record[f"{metric}_mean"] = float(np.mean(values))
                                record[f"{metric}_max"] = int(np.max(values))
                            else:
                                finite = np.isfinite(values)
                                record[f"{metric}_n_finite"] = int(finite.sum())
                                record[f"{metric}_mean_finite"] = (
                                    float(values[finite].mean()) if np.any(finite) else None
                                )
                        records.append(record)
    return {
        "schema_version": "proportional-graph-solver-disentanglement-summary-v1",
        "records": records,
    }


def _build_failure_diagnostics(
    config: dict[str, Any],
    canonical: dict[str, np.ndarray],
    converged: np.ndarray,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(config["arms"]):
        for seed_index, seed in enumerate(config["seeds"]):
            for cell, cell_id in enumerate(CELL_IDS):
                relation, weight, solver = cell_id.split("|")
                for split in ("validation", "test"):
                    mechanisms = (
                        ("all",) if split == "validation" else ("iid", "grouped")
                    )
                    for mechanism in mechanisms:
                        mask = canonical["split"] == split
                        if mechanism != "all":
                            mask &= canonical["mechanism"] == mechanism
                        values = converged[arm_index, seed_index, cell, mask]
                        records.append(
                            {
                                "arm": arm,
                                "seed": int(seed),
                                "relation": relation,
                                "base_weight": weight,
                                "solver": solver,
                                "split": split,
                                "mechanism": mechanism,
                                "n_total": int(len(values)),
                                "n_failed": int(np.sum(~values)),
                                "failure_rate": float(np.mean(~values)),
                            }
                        )
    return {
        "schema_version": "proportional-graph-solver-failures-v1",
        "policy": "nonconverged IRLS states are preserved raw but excluded from strict inferential metrics",
        "records": records,
    }


def _build_effects(
    config: dict[str, Any],
    canonical: dict[str, np.ndarray],
    bootstrap: dict[str, np.ndarray],
    quotient_rmse: np.ndarray,
) -> dict[str, Any]:
    master_order = bootstrap["master_id"]
    view_indices = _master_view_indices(canonical, master_order)
    bootstrap_indices = bootstrap[
        f"complete_n_{config['analysis']['expected_test_masters']}"
    ]
    arms: dict[str, Any] = {}
    for arm_index, arm in enumerate(config["arms"]):
        seed_mean = _strict_seed_mean(quotient_rmse[arm_index])
        cells = np.full((len(SOLVERS), 2, 2, len(canonical["view_id"])), np.nan)
        for solver_index, solver in enumerate(SOLVERS):
            for relation_index, relation in enumerate(RELATIONS):
                for weight_index, weight in enumerate(BASE_WEIGHTS):
                    cells[solver_index, relation_index, weight_index] = seed_mean[
                        _cell_index(relation, weight, solver)
                    ]
        arm_payload: dict[str, Any] = {
            "family_role": (
                "primary"
                if arm in config["analysis"]["primary_arms"]
                else "secondary_control"
            ),
            "levels": {},
            "effects": {},
            "solver_interactions": {},
        }
        solver_effect_values: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for solver_index, solver in enumerate(SOLVERS):
            strict = solver == "irls"
            arm_payload["levels"][solver] = {}
            for relation_index, relation in enumerate(RELATIONS):
                for weight_index, weight in enumerate(BASE_WEIGHTS):
                    level_name = f"{relation}|{weight}"
                    vector = cells[solver_index, relation_index, weight_index]
                    sliced = _slices(
                        vector[view_indices["iid"]],
                        vector[view_indices["grouped"]],
                    )
                    arm_payload["levels"][solver][level_name] = {
                        name: _bootstrap_summary(
                            values, bootstrap_indices, strict=strict
                        )
                        for name, values in sliced.items()
                    }

            effects = _factorial_effects(cells[solver_index])
            solver_effect_values[solver] = {}
            arm_payload["effects"][solver] = {}
            for effect_name in EFFECTS:
                vector = effects[effect_name]
                sliced = _slices(
                    vector[view_indices["iid"]],
                    vector[view_indices["grouped"]],
                )
                solver_effect_values[solver][effect_name] = sliced
                arm_payload["effects"][solver][effect_name] = {
                    name: _bootstrap_summary(values, bootstrap_indices, strict=strict)
                    for name, values in sliced.items()
                }

        for effect_name in EFFECTS:
            arm_payload["solver_interactions"][effect_name] = {}
            for slice_name in SLICES:
                values = (
                    solver_effect_values["irls"][effect_name][slice_name]
                    - solver_effect_values["wls"][effect_name][slice_name]
                )
                arm_payload["solver_interactions"][effect_name][slice_name] = (
                    _bootstrap_summary(values, bootstrap_indices, strict=True)
                )
        arms[arm] = arm_payload
    return {
        "schema_version": "proportional-graph-solver-disentanglement-effects-v1",
        "estimand": "seed mean within arm/view/cell, followed by master-paired effects",
        "bootstrap": {
            "source": "bootstrap_indices.npz",
            "master_order": "explicit master_id mapping",
            "replicates": int(len(bootstrap_indices)),
            "interval": "marginal_95_percentile",
            "p_values": "not_reported_posthoc_exploratory",
        },
        "sign_convention": "negative favors the named intervention",
        "strict_irls_policy": "any non-finite master or seed makes the entire slice NOT_EVALUABLE_SOLVER_FAILURE",
        "arms": arms,
    }


def _write_replay(
    output: Path,
    config_path: Path,
    source: Path,
    source_records: dict[str, Any],
    development: bool,
) -> None:
    replay = output / "replay.sh"
    expected_head = _git_output("rev-parse", "HEAD")
    checks = []
    for relative, record in sorted(source_records.items()):
        if not record["tracked"]:
            continue
        checks.append(
            f"[[ $(sha256sum {shlex.quote(relative)} | cut -d' ' -f1) == "
            f"{shlex.quote(record['sha256'])} ]] || "
            f"{{ echo {shlex.quote('executable hash mismatch: ' + relative)} >&2; exit 3; }}"
        )
    development_flag = " --development" if development else ""
    config_argument = (
        str(config_path.relative_to(REPO_ROOT))
        if config_path.is_relative_to(REPO_ROOT)
        else str(config_path)
    )
    source_argument = (
        str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source)
    )
    script = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "if [[ $# -ne 1 ]]; then echo 'usage: replay.sh OUTPUT_DIR' >&2; exit 2; fi",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"[[ $(git rev-parse HEAD) == {shlex.quote(expected_head)} ]] || "
        "{ echo 'git HEAD differs; recover the manifest-recorded commit before replay' >&2; exit 3; }",
        *checks,
        "CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\",
        "venv/bin/python experiments/geometria_proporcional/run_proportional_graph_solver_disentanglement.py \\",
        f"  --config {shlex.quote(config_argument)} --source {shlex.quote(source_argument)} \\",
        f'  --output "$1"{development_flag}',
    ]
    replay.write_text("\n".join(script) + "\n", encoding="utf-8")
    replay.chmod(0o755)


def _report(effects: dict[str, Any], config: dict[str, Any]) -> str:
    lines = [
        "# Desentrelazado CPU de relación, peso y solver",
        "",
        "Esta ablación funcional recombina outputs congelados del smoke neuronal. "
        "No reentrena, no ejecuta forwards y no identifica efectos causales del entrenamiento.",
        "",
        "## Lectura del efecto total entregado",
        "",
        "Los valores son diferencias de RMSE de cociente; un valor negativo favorece "
        "la combinación corregida×aprendida frente a observada×unitaria. Los intervalos "
        "son bootstrap marginales post hoc, sin p-values.",
        "",
        "| brazo | solver | slice | estado | media | IC95 marginal |",
        "|---|---|---|---|---:|---:|",
    ]
    for arm in config["analysis"]["primary_arms"]:
        for solver in SOLVERS:
            payload = effects["arms"][arm]["effects"][solver][
                "total_delivered_effect"
            ]
            for slice_name in SLICES:
                result = payload[slice_name]
                mean = "—" if result["mean"] is None else f"{result['mean']:.6f}"
                ci = (
                    "—"
                    if result["ci95_marginal"][0] is None
                    else "["
                    + ", ".join(f"{value:.6f}" for value in result["ci95_marginal"])
                    + "]"
                )
                lines.append(
                    f"| `{arm}` | `{solver}` | `{slice_name}` | "
                    f"`{result['status']}` | {mean} | {ci} |"
                )
    lines.extend(
        [
            "",
            "## Fronteras de interpretación",
            "",
            "Los niveles, cuatro efectos factoriales, interacciones con solver, "
            "diagnósticos de fallo y estados raw viven en los artefactos estructurados. "
            "Una no convergencia IRLS invalida de forma conservadora el slice inferencial "
            "completo, aunque el último estado finito se conserva para diagnóstico.",
            "",
            "El resultado localiza compatibilidades funcionales entre outputs congelados "
            "y solvers. No promueve una arquitectura, no acredita geometría natural y no "
            "declara GO/NO-GO.",
        ]
    )
    return "\n".join(lines) + "\n"


def _manifest(
    output: Path,
    config: dict[str, Any],
    source_records: dict[str, Any],
    threadpools: list[dict[str, Any]],
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "runtime_observation.json"}:
            files[str(path.relative_to(output))] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    status = _git_output("status", "--porcelain")
    return {
        "schema_version": config["artifact_schema_version"],
        "git": {
            "head": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
            "dirty": bool(status),
            "status_sha256": _sha256_bytes(status.encode("utf-8")),
        },
        "resolved_config": config,
        "public_schema_sha256": public_schema_hash(),
        "official_source_files": source_records,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "threadpoolctl": importlib.metadata.version("threadpoolctl"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "threadpools_inside_limit": threadpools,
            "device": "cpu",
            "torch_imported": "torch" in sys.modules,
        },
        "replay_scope": {
            "byte_exact_files": "all manifest-listed files",
            "excluded_nondeterministic_files": ["runtime_observation.json"],
        },
        "files": files,
    }


def _global_metric_payload(
    config: dict[str, Any],
    canonical: dict[str, np.ndarray],
    metric_tensor: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        **metric_tensor,
        "arm": np.asarray(config["arms"]),
        "seed": np.asarray(config["seeds"], dtype=np.int64),
        "cell_id": np.asarray(CELL_IDS),
        "view_id": canonical["view_id"],
        "master_id": canonical["master_id"],
        "split": canonical["split"],
        "mechanism": canonical["mechanism"],
    }


def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    config_path = args.config.resolve(strict=True)
    config = _load_config(config_path)
    _guard_cpu_environment(config)
    configured_source = (REPO_ROOT / config["source"]["path"]).resolve(strict=True)
    source = (args.source.resolve(strict=True) if args.source else configured_source)
    if source != configured_source:
        raise ValueError("source must equal the frozen path in the resolved configuration")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    source_records = _tracked_clean_records(
        (PLAN_PATH, config_path, RUNNER_PATH, SOLVER_PATH), args.development
    )
    canonical, raw_by_key, bootstrap, attestation = _verify_source(
        source, config, started=started
    )
    _enforce_budget(started, config, "source attestation")
    _prepare_output(output, source)

    shape = (len(config["arms"]), len(config["seeds"]), len(CELL_IDS), len(canonical["view_id"]))
    metric_tensor: dict[str, np.ndarray] = {
        metric: np.full(
            shape,
            False if metric == "converged" else -1 if metric in {"laplacian_rank", "iterations"} else np.nan,
            dtype=bool if metric == "converged" else np.int64 if metric in {"laplacian_rank", "iterations"} else np.float64,
        )
        for metric in METRICS
    }
    reuse_records: dict[str, Any] = {}
    with threadpool_limits(limits=1):
        limited_threadpools = threadpool_info()
        if any(int(pool.get("num_threads", 1)) != 1 for pool in limited_threadpools):
            raise RuntimeError(f"numeric thread limit failed: {limited_threadpools}")
        for arm_index, arm in enumerate(config["arms"]):
            for seed_index, seed in enumerate(config["seeds"]):
                key = f"{arm}|seed={seed}"
                raw_payload, model_metrics, reuse = _run_model_matrix(
                    canonical,
                    canonical,
                    raw_by_key[key],
                    arm,
                    int(seed),
                    config,
                    started,
                )
                for metric in METRICS:
                    metric_tensor[metric][arm_index, seed_index] = model_metrics[metric]
                reuse_records[key] = reuse
                _write_deterministic_npz(
                    output / "raw_solver" / f"{key}.npz", **raw_payload
                )
                _enforce_budget(started, config, f"serialization {key}")

        summary = _build_summary(config, canonical, metric_tensor)
        effects = _build_effects(
            config, canonical, bootstrap, metric_tensor["quotient_rmse"]
        )
        failures = _build_failure_diagnostics(
            config, canonical, metric_tensor["converged"]
        )

    attestation["official_source_files"] = source_records
    attestation["reuse_verification"] = reuse_records
    attestation["solver_input_authority"] = (
        "all PublicGraphObservation fields from raw_eval/observed_unweighted.npz; "
        "neural raws contribute corrected_log_ratio and reliability only"
    )
    _write_json(output / "resolved_config.json", config)
    _write_json(output / "source_attestation.json", attestation)
    _write_deterministic_npz(
        output / "per_view_metrics.npz",
        **_global_metric_payload(config, canonical, metric_tensor),
    )
    _write_json(output / "summary.json", summary)
    _write_json(output / "effects.json", effects)
    _write_json(output / "failure_diagnostics.json", failures)
    shutil.copyfile(source / "bootstrap_indices.npz", output / "bootstrap_indices.npz")
    (output / "DISENTANGLEMENT_REPORT.md").write_text(
        _report(effects, config), encoding="utf-8"
    )
    _write_replay(
        output, config_path, source, source_records, args.development
    )
    _enforce_budget(started, config, "artifact finalization")
    runtime = _resource_observation(started)
    _write_json(
        output / "runtime_observation.json",
        {
            **runtime,
            "included_in_byte_exact_replay": False,
            "reason": "wall time and peak RSS are run-specific observations",
        },
    )
    _write_json(
        output / "manifest.json",
        _manifest(output, config, source_records, limited_threadpools),
    )
    print(
        json.dumps(
            {
                "device": "cpu",
                "arms": len(config["arms"]),
                "seeds": len(config["seeds"]),
                "views_per_model": len(canonical["view_id"]),
                "cells": len(CELL_IDS),
                **runtime,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
