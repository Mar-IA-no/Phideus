#!/usr/bin/env python3
"""CPU-only diagnostic of learned reliability semantics across frozen solvers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.geometria_proporcional import (  # noqa: E402
    run_proportional_graph_solver_disentanglement as dis,
)


PLAN_PATH = (
    REPO_ROOT
    / "experiments/geometria_proporcional/PLAN_PROPORTIONAL_SOLVER_INTERFACE_DIAGNOSTIC_CPU.md"
)
RUNNER_PATH = Path(__file__).resolve()
SOLVER_PATH = REPO_ROOT / "src/geometria_proporcional/proportional_graph_contract.py"
BASE_RUNNER_PATH = (
    REPO_ROOT
    / "experiments/geometria_proporcional/run_proportional_graph_solver_disentanglement.py"
)

ARMS = (
    "raw_generic",
    "raw_typed",
    "closure_generic",
    "closure_typed",
    "closure_typed_path_shuffle",
    "pair_state_no_mix",
    "generic_message_passing",
    "edge_mlp",
)
SEEDS = (104729, 130363)
RELATIONS = ("observed", "corrected")
SOLVERS = ("wls", "irls")
STATIC_CELLS = (
    "observed|unit",
    "observed|learned",
    "corrected|unit",
    "corrected|learned",
)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
RIDGE_LAMBDAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
FEATURE_ORDER = (
    "n_nodes",
    "n_valid_edges",
    "edge_density",
    "n_valid_paths",
    "edge_variance_mean",
    "edge_variance_std",
    "edge_variance_max",
    "observed_rms",
    "observed_abs_max",
    "correction_rms",
    "correction_abs_max",
    "observed_closure_rms",
    "observed_closure_abs_median",
    "corrected_closure_rms",
    "corrected_closure_abs_median",
    "reliability_mean",
    "reliability_std",
    "reliability_min",
    "reliability_median",
    "reliability_max",
    "reliability_entropy_normalized",
    "reliability_effective_fraction",
    "reliability_correction_correlation",
)
PRIMARY_ARMS = ARMS[:4]
SECONDARY_ARMS = ARMS[4:]
SLICES = ("test|iid", "test|grouped", "grouped_minus_iid")
EXPECTED_EXECUTION = {
    "device": "cpu",
    "numeric_threads": 1,
    "max_seconds": 900,
    "max_rss_gib": 4.0,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "experiments/geometria_proporcional/configs/proportional_graph_solver_interface_diagnostic_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != {
        "schema_version",
        "artifact_schema_version",
        "sources",
        "arms",
        "seeds",
        "solver",
        "analysis",
        "execution",
    }:
        raise ValueError("unexpected top-level config keys")
    if config["schema_version"] != "proportional-graph-solver-interface-diagnostic-v1":
        raise ValueError("unexpected config schema")
    if config["artifact_schema_version"] != (
        "proportional-graph-solver-interface-diagnostic-artifact-v1"
    ):
        raise ValueError("unexpected artifact schema")
    expected_sources = {
        "smoke": {
            "path": "data/geometria_proporcional/proportional_graph_neural_smoke_v1",
            "manifest_sha256": "7e982a92bd366a4c22fe95c0cc9c8fd5f7a1773bd78c4e8e0781b5c2259624e1",
            "artifact_schema_version": "proportional-graph-neural-artifact-v1",
        },
        "disentanglement": {
            "path": "data/geometria_proporcional/proportional_graph_solver_disentanglement_v1",
            "manifest_sha256": "17d09bdee0100ebc1d5a7ee4f2f3c826e51b6e04ef4ecbc33c8f290fb5cf0347",
            "artifact_schema_version": "proportional-graph-solver-disentanglement-artifact-v1",
            "source_git_head": "0731bc054fa9c716483ac3d09b12fb922446a784",
        },
        "disentanglement_config": "experiments/geometria_proporcional/configs/proportional_graph_solver_disentanglement_v1.json",
    }
    if config["sources"] != expected_sources:
        raise ValueError("source contract differs from frozen plan")
    if tuple(config["arms"]) != ARMS or tuple(config["seeds"]) != SEEDS:
        raise ValueError("arm or seed order differs from frozen plan")
    if config["solver"] != dis.EXPECTED_SOLVER:
        raise ValueError("solver contract differs from frozen plan")
    analysis = config["analysis"]
    if set(analysis) != {
        "relations",
        "solvers",
        "static_cell_order",
        "alphas",
        "shuffle_replicates",
        "shuffle_seed_root",
        "ridge_lambdas",
        "ridge_folds",
        "feature_order",
        "new_solve_arms",
        "temperature_scope",
        "shuffle_scope",
        "primary_arms",
        "secondary_arms",
        "slice_order",
        "bootstrap_replicates",
        "expected_validation_views",
        "expected_test_views",
        "expected_test_masters",
        "strict_irls_failures",
        "inference_regime",
    }:
        raise ValueError("unexpected analysis contract")
    exact = {
        "relations": list(RELATIONS),
        "solvers": list(SOLVERS),
        "static_cell_order": list(STATIC_CELLS),
        "alphas": list(ALPHAS),
        "shuffle_replicates": 8,
        "shuffle_seed_root": 350903,
        "ridge_lambdas": list(RIDGE_LAMBDAS),
        "ridge_folds": 5,
        "feature_order": list(FEATURE_ORDER),
        "new_solve_arms": list(PRIMARY_ARMS),
        "temperature_scope": {"wls": list(RELATIONS), "irls": ["observed"]},
        "shuffle_scope": {"wls": ["observed"], "irls": []},
        "primary_arms": list(PRIMARY_ARMS),
        "secondary_arms": list(SECONDARY_ARMS),
        "slice_order": list(SLICES),
        "bootstrap_replicates": 2000,
        "expected_validation_views": 127,
        "expected_test_views": 504,
        "expected_test_masters": 252,
        "strict_irls_failures": True,
        "inference_regime": "exploratory_posthoc_transport_and_weight_semantics_no_pvalues",
    }
    if analysis != exact:
        raise ValueError("analysis values differ from frozen plan")
    if config["execution"] != EXPECTED_EXECUTION:
        raise ValueError("execution contract differs from frozen plan")
    return config


def _prepare_output(output: Path, sources: Iterable[Path]) -> None:
    target = output.resolve()
    resolved_sources = [source.resolve(strict=True) for source in sources]
    for source in resolved_sources:
        if target == source or target.is_relative_to(source):
            raise ValueError("output may not equal or descend from a source")
        if source.is_relative_to(target):
            raise ValueError("output may not contain a source")
    if target.exists():
        raise FileExistsError(f"output already exists: {target}")
    target.mkdir(parents=True)


def _verify_manifest_package(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if dis._sha256_file(manifest_path) != expected["manifest_sha256"]:
        raise ValueError(f"manifest hash mismatch: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != expected["artifact_schema_version"]:
        raise ValueError(f"artifact schema mismatch: {path}")
    if "source_git_head" in expected:
        if manifest.get("git", {}).get("head") != expected["source_git_head"]:
            raise ValueError("disentanglement source git head mismatch")
        if manifest.get("git", {}).get("dirty") is not False:
            raise ValueError("disentanglement package was not produced cleanly")
    records: dict[str, Any] = {}
    for relative, record in sorted(manifest.get("files", {}).items()):
        member = path / relative
        if not member.is_file():
            raise FileNotFoundError(member)
        actual = {"bytes": member.stat().st_size, "sha256": dis._sha256_file(member)}
        if actual != record:
            raise ValueError(f"manifest member mismatch: {relative}")
        records[relative] = actual
    return {"manifest": manifest, "verified_files": records}


def _verify_sources(
    config: dict[str, Any], started: float
) -> tuple[
    Path,
    Path,
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
]:
    smoke = (REPO_ROOT / config["sources"]["smoke"]["path"]).resolve(strict=True)
    disentangle = (
        REPO_ROOT / config["sources"]["disentanglement"]["path"]
    ).resolve(strict=True)
    base_config_path = (
        REPO_ROOT / config["sources"]["disentanglement_config"]
    ).resolve(strict=True)
    base_config = dis._load_config(base_config_path)
    canonical, neural_raws, bootstrap, smoke_attestation = dis._verify_source(
        smoke, base_config, started=started
    )
    package = _verify_manifest_package(
        disentangle, config["sources"]["disentanglement"]
    )
    global_metrics = dis._load_npz(disentangle / "per_view_metrics.npz")
    required_global = {
        "arm",
        "seed",
        "cell_id",
        "view_id",
        "master_id",
        "split",
        "mechanism",
        "quotient_rmse",
        "relation_rmse",
        "weighted_residual_rmse",
        "laplacian_rank",
        "laplacian_condition",
        "converged",
        "iterations",
    }
    if set(global_metrics) != required_global:
        raise ValueError("unexpected disentanglement per-view schema")
    dis._exact_equal(global_metrics["arm"], np.asarray(ARMS), "arm axis")
    dis._exact_equal(global_metrics["seed"], np.asarray(SEEDS), "seed axis")
    dis._exact_equal(global_metrics["cell_id"], np.asarray(dis.CELL_IDS), "cell axis")
    for key in ("view_id", "master_id", "split", "mechanism"):
        dis._exact_equal(global_metrics[key], canonical[key], f"global {key}")
    expected_shape = (len(ARMS), len(SEEDS), len(dis.CELL_IDS), len(canonical["view_id"]))
    for key in required_global - {
        "arm",
        "seed",
        "cell_id",
        "view_id",
        "master_id",
        "split",
        "mechanism",
    }:
        if global_metrics[key].shape != expected_shape:
            raise ValueError(f"unexpected global metric shape: {key}")
    solver_raws: dict[str, dict[str, np.ndarray]] = {}
    for arm_index, arm in enumerate(ARMS):
        for seed_index, seed in enumerate(SEEDS):
            key = f"{arm}|seed={seed}"
            raw = dis._load_npz(disentangle / "raw_solver" / f"{key}.npz")
            dis._exact_equal(raw["cell_id"], np.asarray(dis.CELL_IDS), f"{key} cells")
            for field in ("view_id", "master_id", "split", "mechanism"):
                dis._exact_equal(raw[field], canonical[field], f"{key} {field}")
            dis._exact_equal(raw["edge_offsets"], canonical["edge_offsets"], f"{key} edges")
            dis._exact_equal(raw["node_offsets"], canonical["node_offsets"], f"{key} nodes")
            for metric in dis.METRICS:
                left = raw[metric]
                right = global_metrics[metric][arm_index, seed_index]
                if not np.array_equal(left, right, equal_nan=True):
                    raise ValueError(f"raw/global metric mismatch: {key}:{metric}")
            solver_raws[key] = raw
    if (smoke / "bootstrap_indices.npz").read_bytes() != (
        disentangle / "bootstrap_indices.npz"
    ).read_bytes():
        raise ValueError("bootstrap copies differ")
    validation = canonical["split"] == "validation"
    test = canonical["split"] == "test"
    if int(validation.sum()) != 127 or int(test.sum()) != 504:
        raise ValueError("unexpected validation/test counts")
    if set(canonical["mechanism"][validation].tolist()) != {"iid"}:
        raise ValueError("validation must be IID-only")
    val_masters = canonical["master_id"][validation]
    if len(np.unique(val_masters)) != len(val_masters):
        raise ValueError("validation must contain one view per master")
    attestation = {
        "smoke": smoke_attestation,
        "disentanglement": package,
        "bootstrap_byte_exact": True,
        "validation_views": int(validation.sum()),
        "validation_masters": int(len(np.unique(val_masters))),
        "test_views": int(test.sum()),
        "test_masters": int(len(np.unique(canonical["master_id"][test]))),
    }
    return (
        smoke,
        disentangle,
        canonical,
        neural_raws,
        global_metrics,
        solver_raws,
        {"indices": bootstrap, "attestation": attestation},
    )


def _alpha_label(alpha: float) -> str:
    return f"{alpha:.2f}"


TEMP_VARIANTS = tuple(
    f"{relation}|alpha={_alpha_label(alpha)}|{solver}"
    for solver in SOLVERS
    for relation in (RELATIONS if solver == "wls" else ("observed",))
    for alpha in ALPHAS
)
SHUFFLE_VARIANTS = tuple(
    f"observed|shuffle={replicate:02d}|wls"
    for replicate in range(8)
)
VARIANTS = TEMP_VARIANTS + SHUFFLE_VARIANTS


def _tempered_weights(weights: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    source = np.asarray(weights, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if source.shape != mask.shape or not np.all(np.isfinite(source)):
        raise ValueError("weights must be finite and edge-aligned")
    result = np.zeros_like(source)
    if alpha == 0.0:
        result[mask] = 1.0
        return result
    if alpha == 1.0:
        result[mask] = source[mask]
        return result
    values = np.exp(alpha * np.log(np.maximum(source[mask], 0.001)))
    values /= values.mean()
    result[mask] = values
    return result


def _shuffle_weights(
    weights: np.ndarray,
    valid: np.ndarray,
    *,
    arm: str,
    seed: int,
    view_index: int,
    replicate: int,
    seed_root: int,
) -> np.ndarray:
    source = np.asarray(weights, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    result = source.copy()
    valid_indices = np.flatnonzero(mask)
    token = f"{seed_root}|{arm}|{seed}|{view_index}|{replicate}".encode("utf-8")
    rng_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    rng = np.random.default_rng(rng_seed)
    permutation = rng.permutation(len(valid_indices))
    result[valid_indices] = source[valid_indices[permutation]]
    if not np.array_equal(np.sort(result[mask]), np.sort(source[mask])):
        raise AssertionError("shuffle failed to preserve weight multiset")
    return result


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values))) if len(values) else 0.0


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _public_features(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_valid: np.ndarray,
    edge_variance: np.ndarray,
    observed_log_ratio: np.ndarray,
    path_index: np.ndarray,
    path_sign: np.ndarray,
    path_valid: np.ndarray,
    corrected_log_ratio: np.ndarray,
    reliability: np.ndarray,
) -> np.ndarray:
    valid = np.asarray(edge_valid, dtype=bool)
    edges = np.asarray(edge_index, dtype=np.int64)
    variance = np.asarray(edge_variance, dtype=np.float64)[valid]
    observed = np.asarray(observed_log_ratio, dtype=np.float64)
    corrected = np.asarray(corrected_log_ratio, dtype=np.float64)
    learned = np.asarray(reliability, dtype=np.float64)[valid]
    paths = np.asarray(path_index, dtype=np.int64)
    signs = np.asarray(path_sign, dtype=np.float64)
    pvalid = np.asarray(path_valid, dtype=bool)
    if edges.shape != (len(valid), 2) or observed.shape != valid.shape:
        raise ValueError("invalid public edge arrays")
    if corrected.shape != valid.shape or np.asarray(reliability).shape != valid.shape:
        raise ValueError("invalid model output arrays")
    if paths.ndim != 2 or paths.shape[1] != 3 or signs.shape != paths.shape:
        raise ValueError("invalid public path arrays")
    if pvalid.shape != (len(paths),) or np.any(paths < 0) or np.any(paths >= len(valid)):
        raise ValueError("invalid public path indices")
    if not all(np.all(np.isfinite(x)) for x in (variance, observed, corrected, learned, signs)):
        raise ValueError("public features require finite inputs")
    selected_paths = paths[pvalid]
    selected_signs = signs[pvalid]
    observed_closure = (
        np.sum(selected_signs * observed[selected_paths], axis=1)
        if len(selected_paths)
        else np.empty(0)
    )
    corrected_closure = (
        np.sum(selected_signs * corrected[selected_paths], axis=1)
        if len(selected_paths)
        else np.empty(0)
    )
    delta = corrected[valid] - observed[valid]
    total = float(learned.sum())
    probabilities = learned / total
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    entropy_normalized = entropy / math.log(len(learned)) if len(learned) > 1 else 1.0
    effective_fraction = float(total * total / np.sum(learned * learned) / len(learned))
    possible_edges = max(n_nodes * (n_nodes - 1) / 2, 1.0)
    values = {
        "n_nodes": float(n_nodes),
        "n_valid_edges": float(valid.sum()),
        "edge_density": float(valid.sum() / possible_edges),
        "n_valid_paths": float(pvalid.sum()),
        "edge_variance_mean": float(variance.mean()),
        "edge_variance_std": float(variance.std()),
        "edge_variance_max": float(variance.max()),
        "observed_rms": _rms(observed[valid]),
        "observed_abs_max": float(np.max(np.abs(observed[valid]))),
        "correction_rms": _rms(delta),
        "correction_abs_max": float(np.max(np.abs(delta))),
        "observed_closure_rms": _rms(observed_closure),
        "observed_closure_abs_median": float(np.median(np.abs(observed_closure))) if len(observed_closure) else 0.0,
        "corrected_closure_rms": _rms(corrected_closure),
        "corrected_closure_abs_median": float(np.median(np.abs(corrected_closure))) if len(corrected_closure) else 0.0,
        "reliability_mean": float(learned.mean()),
        "reliability_std": float(learned.std()),
        "reliability_min": float(learned.min()),
        "reliability_median": float(np.median(learned)),
        "reliability_max": float(learned.max()),
        "reliability_entropy_normalized": entropy_normalized,
        "reliability_effective_fraction": effective_fraction,
        "reliability_correction_correlation": _safe_correlation(learned, np.abs(delta)),
    }
    result = np.asarray([values[key] for key in FEATURE_ORDER], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite public feature")
    return result


def _features_for_raw(
    canonical: dict[str, np.ndarray], neural_raw: dict[str, np.ndarray]
) -> np.ndarray:
    features = np.full(
        (len(canonical["view_id"]), len(FEATURE_ORDER)), np.nan, dtype=np.float64
    )
    for view_index in range(len(canonical["view_id"])):
        edge_slice, _, path_slice = dis._view_slices(canonical, view_index)
        features[view_index] = _public_features(
            n_nodes=int(canonical["n_nodes"][view_index]),
            edge_index=canonical["edge_index"][edge_slice],
            edge_valid=canonical["edge_valid"][edge_slice],
            edge_variance=canonical["edge_variance"][edge_slice],
            observed_log_ratio=canonical["observed_log_ratio"][edge_slice],
            path_index=neural_raw["path_index"][path_slice],
            path_sign=neural_raw["path_sign"][path_slice],
            path_valid=neural_raw["path_valid"][path_slice],
            corrected_log_ratio=neural_raw["corrected_log_ratio"][edge_slice],
            reliability=neural_raw["reliability"][edge_slice],
        )
    return features


def _assert_endpoint(
    state: dis.CompletedState,
    raw: dict[str, np.ndarray],
    cell: int,
    view_index: int,
    edge_slice: slice,
    node_slice: slice,
    atol: float,
    label: str,
) -> None:
    for actual, expected, field in (
        (state.x_hat, raw["x_hat"][cell, node_slice], "x_hat"),
        (state.weights, raw["final_weights"][cell, edge_slice], "weights"),
    ):
        if not np.allclose(actual, expected, rtol=0.0, atol=atol, equal_nan=True):
            raise AssertionError(f"endpoint mismatch {label}:{field}")
    for field in dis.METRICS:
        actual = getattr(state, field)
        expected = raw[field][cell, view_index]
        if field in {"converged", "iterations", "laplacian_rank"}:
            if actual != expected:
                raise AssertionError(f"endpoint mismatch {label}:{field}")
        elif not np.isclose(actual, expected, rtol=0.0, atol=atol, equal_nan=True):
            raise AssertionError(f"endpoint mismatch {label}:{field}")


def _run_variants(
    canonical: dict[str, np.ndarray],
    neural_raw: dict[str, np.ndarray],
    solver_raw: dict[str, np.ndarray],
    arm: str,
    seed: int,
    config: dict[str, Any],
    started: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n_views = len(canonical["view_id"])
    total_nodes = len(canonical["x_true"])
    total_edges = len(canonical["observed_log_ratio"])
    x_hat = np.full((len(VARIANTS), total_nodes), np.nan, dtype=np.float64)
    final_weights = np.full((len(VARIANTS), total_edges), np.nan, dtype=np.float64)
    metrics: dict[str, np.ndarray] = {
        field: np.full(
            (len(VARIANTS), n_views),
            False if field == "converged" else -1 if field in {"iterations", "laplacian_rank"} else np.nan,
            dtype=bool if field == "converged" else np.int64 if field in {"iterations", "laplacian_rank"} else np.float64,
        )
        for field in dis.METRICS
    }
    features = _features_for_raw(canonical, neural_raw)
    atol = float(config["solver"]["reuse_atol"])
    for view_index in range(n_views):
        edge_slice, node_slice, path_slice = dis._view_slices(canonical, view_index)
        observation = dis._public_view(canonical, view_index)
        observed = canonical["observed_log_ratio"][edge_slice]
        corrected = neural_raw["corrected_log_ratio"][edge_slice]
        learned = neural_raw["reliability"][edge_slice]
        valid = canonical["edge_valid"][edge_slice]
        relations = {"observed": observed, "corrected": corrected}
        x_true = canonical["x_true"][node_slice]
        clean = canonical["clean_log_ratio"][edge_slice]
        for variant_index, variant in enumerate(VARIANTS):
            relation_name, transform, solver_name = variant.split("|")
            if transform.startswith("alpha="):
                alpha = float(transform.split("=", 1)[1])
                base = _tempered_weights(learned, valid, alpha)
            else:
                replicate = int(transform.split("=", 1)[1])
                base = _shuffle_weights(
                    learned,
                    valid,
                    arm=arm,
                    seed=seed,
                    view_index=view_index,
                    replicate=replicate,
                    seed_root=int(config["analysis"]["shuffle_seed_root"]),
                )
            state = dis._solve_cell(
                observation,
                relations[relation_name],
                base,
                solver_name,
                config,
                x_true,
                clean,
            )
            x_hat[variant_index, node_slice] = state.x_hat
            final_weights[variant_index, edge_slice] = state.weights
            for field in dis.METRICS:
                metrics[field][variant_index, view_index] = getattr(state, field)
            if transform in {"alpha=0.00", "alpha=1.00"}:
                weight_name = "unit" if transform == "alpha=0.00" else "learned"
                endpoint = dis._cell_index(relation_name, weight_name, solver_name)
                _assert_endpoint(
                    state,
                    solver_raw,
                    endpoint,
                    view_index,
                    edge_slice,
                    node_slice,
                    atol,
                    f"{arm}|seed={seed}|view={view_index}|{variant}",
                )
        if view_index % 8 == 0:
            dis._enforce_budget(started, config, f"{arm}|seed={seed}|view={view_index}")
    payload = {
        "variant_id": np.asarray(VARIANTS),
        "x_hat": x_hat,
        "final_weights": final_weights,
        **metrics,
        "edge_offsets": canonical["edge_offsets"],
        "node_offsets": canonical["node_offsets"],
        "view_id": canonical["view_id"],
        "master_id": canonical["master_id"],
        "split": canonical["split"],
        "mechanism": canonical["mechanism"],
        "arm": np.asarray(arm),
        "seed": np.asarray(seed, dtype=np.int64),
    }
    return payload, features


def _master_mean(values: np.ndarray, master_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.asarray(sorted(set(master_ids.tolist())))
    means = np.asarray([np.mean(values[master_ids == master]) for master in order])
    return order, means


def _validation_selection(
    canonical: dict[str, np.ndarray], global_metrics: dict[str, np.ndarray]
) -> dict[str, Any]:
    validation = canonical["split"] == "validation"
    result: dict[str, Any] = {"selection_basis": "validation_master_mean", "arms": {}}
    for arm_index, arm in enumerate(ARMS):
        result["arms"][arm] = {}
        for solver in SOLVERS:
            candidates = []
            for order, cell_name in enumerate(STATIC_CELLS):
                relation, weight = cell_name.split("|")
                cell = dis._cell_index(relation, weight, solver)
                raw = global_metrics["quotient_rmse"][arm_index, :, cell, :]
                seed_mean = dis._strict_seed_mean(raw)
                _, master_values = _master_mean(
                    seed_mean[validation], canonical["master_id"][validation]
                )
                eligible = bool(np.all(np.isfinite(master_values)))
                candidates.append(
                    {
                        "cell": cell_name,
                        "eligible": eligible,
                        "validation_mean": float(master_values.mean()) if eligible else None,
                        "tie_order": order,
                    }
                )
            eligible = [x for x in candidates if x["eligible"]]
            if not eligible:
                raise RuntimeError(f"no validation-eligible static cell: {arm}:{solver}")
            best_mean = min(x["validation_mean"] for x in eligible)
            chosen = min(
                (x for x in eligible if x["validation_mean"] <= best_mean + 1e-12),
                key=lambda x: x["tie_order"],
            )
            result["arms"][arm][solver] = {
                "selected_cell": chosen["cell"],
                "candidates": candidates,
            }
    return result


def _temperature_selection(
    canonical: dict[str, np.ndarray], quotient: np.ndarray
) -> dict[str, Any]:
    validation = canonical["split"] == "validation"
    result: dict[str, Any] = {"selection_basis": "validation_master_mean", "arms": {}}
    for arm_index, arm in enumerate(PRIMARY_ARMS):
        result["arms"][arm] = {}
        for solver in SOLVERS:
            result["arms"][arm][solver] = {}
            for relation in (RELATIONS if solver == "wls" else ("observed",)):
                candidates = []
                for order, alpha in enumerate(ALPHAS):
                    variant = VARIANTS.index(
                        f"{relation}|alpha={_alpha_label(alpha)}|{solver}"
                    )
                    seed_mean = dis._strict_seed_mean(quotient[arm_index, :, variant])
                    _, master_values = _master_mean(
                        seed_mean[validation], canonical["master_id"][validation]
                    )
                    eligible = bool(np.all(np.isfinite(master_values)))
                    candidates.append(
                        {
                            "alpha": alpha,
                            "eligible": eligible,
                            "validation_mean": float(master_values.mean()) if eligible else None,
                            "tie_order": order,
                        }
                    )
                eligible = [x for x in candidates if x["eligible"]]
                if not eligible:
                    raise RuntimeError(f"no eligible alpha: {arm}:{solver}:{relation}")
                best_mean = min(x["validation_mean"] for x in eligible)
                chosen = min(
                    (x for x in eligible if x["validation_mean"] <= best_mean + 1e-12),
                    key=lambda x: x["tie_order"],
                )
                result["arms"][arm][solver][relation] = {
                    "selected_alpha": chosen["alpha"],
                    "candidates": candidates,
                }
    return result


def _fold_ids(master_ids: np.ndarray, folds: int) -> np.ndarray:
    result = []
    for master in master_ids:
        digest = hashlib.sha256(str(master).encode("utf-8")).digest()
        result.append(int.from_bytes(digest[:8], "little") % folds)
    return np.asarray(result, dtype=np.int64)


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> dict[str, np.ndarray | float]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), standardized])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    if ridge == 0.0:
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    else:
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {
        "mean": mean,
        "scale": scale,
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:],
    }


def _ridge_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return model["intercept"] + ((x - model["mean"]) / model["scale"]) @ model[
        "coefficients"
    ]


def _fit_transport_models(
    canonical: dict[str, np.ndarray],
    features: np.ndarray,
    global_metrics: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    validation = canonical["split"] == "validation"
    feature_mean = features.mean(axis=1)
    predictions = np.full((len(ARMS), len(SOLVERS), len(canonical["view_id"])), np.nan)
    decisions = np.zeros_like(predictions, dtype=bool)
    report: dict[str, Any] = {"feature_order": list(FEATURE_ORDER), "arms": {}}
    val_master = canonical["master_id"][validation]
    folds = _fold_ids(val_master, int(config["analysis"]["ridge_folds"]))
    if set(folds.tolist()) != set(range(int(config["analysis"]["ridge_folds"]))):
        raise ValueError("ridge folds are unexpectedly empty")
    for arm_index, arm in enumerate(ARMS):
        report["arms"][arm] = {}
        x_val = feature_mean[arm_index, validation]
        for solver_index, solver in enumerate(SOLVERS):
            unit_cell = dis._cell_index("observed", "unit", solver)
            learned_cell = dis._cell_index("observed", "learned", solver)
            delta = dis._strict_seed_mean(
                global_metrics["quotient_rmse"][arm_index, :, learned_cell]
                - global_metrics["quotient_rmse"][arm_index, :, unit_cell]
            )
            y_val = delta[validation]
            if not np.all(np.isfinite(y_val)):
                report["arms"][arm][solver] = {"status": "NOT_EVALUABLE_VALIDATION_FAILURE"}
                continue
            candidates = []
            for ridge in RIDGE_LAMBDAS:
                fold_losses = []
                for fold in range(int(config["analysis"]["ridge_folds"])):
                    train = folds != fold
                    held = folds == fold
                    model = _ridge_fit(x_val[train], y_val[train], ridge)
                    predicted = _ridge_predict(model, x_val[held])
                    fold_losses.append(float(np.mean((predicted - y_val[held]) ** 2)))
                candidates.append({"lambda": ridge, "cv_mse": float(np.mean(fold_losses))})
            best_mse = min(x["cv_mse"] for x in candidates)
            best = max(
                (x for x in candidates if x["cv_mse"] <= best_mse + 1e-15),
                key=lambda x: x["lambda"],
            )
            model = _ridge_fit(x_val, y_val, float(best["lambda"]))
            predicted = _ridge_predict(model, feature_mean[arm_index])
            predictions[arm_index, solver_index] = predicted
            decisions[arm_index, solver_index] = predicted < 0.0
            report["arms"][arm][solver] = {
                "status": "FITTED_VALIDATION_ONLY",
                "selected_lambda": best["lambda"],
                "candidates": candidates,
                "fold_id": folds.tolist(),
                "feature_mean": model["mean"].tolist(),
                "feature_scale": model["scale"].tolist(),
                "intercept": model["intercept"],
                "coefficients": model["coefficients"].tolist(),
            }
    return report, predictions, decisions


def _test_pair_indices(
    canonical: dict[str, np.ndarray], bootstrap: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    master_order = np.asarray(bootstrap["master_id"])
    view_indices = dis._master_view_indices(canonical, master_order)
    indices = np.asarray(bootstrap["complete_n_252"], dtype=np.int64)
    return view_indices["iid"], view_indices["grouped"], indices


def _slice_summaries(
    iid_values: np.ndarray,
    grouped_values: np.ndarray,
    bootstrap_indices: np.ndarray,
    *,
    strict: bool,
) -> dict[str, Any]:
    return {
        "test|iid": dis._bootstrap_summary(iid_values, bootstrap_indices, strict=strict),
        "test|grouped": dis._bootstrap_summary(grouped_values, bootstrap_indices, strict=strict),
        "grouped_minus_iid": dis._bootstrap_summary(
            grouped_values - iid_values, bootstrap_indices, strict=strict
        ),
    }


def _evaluate_effects(
    canonical: dict[str, np.ndarray],
    bootstrap: dict[str, np.ndarray],
    global_metrics: dict[str, np.ndarray],
    quotient: np.ndarray,
    static_selection: dict[str, Any],
    temperature_selection: dict[str, Any],
    predictions: np.ndarray,
    decisions: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    iid, grouped, bootstrap_indices = _test_pair_indices(canonical, bootstrap)
    effects: dict[str, Any] = {"arms": {}}
    transport: dict[str, Any] = {"arms": {}}
    for arm_index, arm in enumerate(ARMS):
        effects["arms"][arm] = {"static": {}, "temperature": {}, "shuffle": {}}
        transport["arms"][arm] = {}
        for solver_index, solver in enumerate(SOLVERS):
            strict = solver == "irls"
            selected_name = static_selection["arms"][arm][solver]["selected_cell"]
            relation, weight = selected_name.split("|")
            selected_cell = dis._cell_index(relation, weight, solver)
            unit_cell = dis._cell_index("observed", "unit", solver)
            learned_cell = dis._cell_index("observed", "learned", solver)
            delivered_cell = dis._cell_index("corrected", "learned", solver)
            base_seed_mean = {
                "selected": dis._strict_seed_mean(global_metrics["quotient_rmse"][arm_index, :, selected_cell]),
                "unit": dis._strict_seed_mean(global_metrics["quotient_rmse"][arm_index, :, unit_cell]),
                "learned": dis._strict_seed_mean(global_metrics["quotient_rmse"][arm_index, :, learned_cell]),
                "delivered": dis._strict_seed_mean(global_metrics["quotient_rmse"][arm_index, :, delivered_cell]),
            }
            effects["arms"][arm]["static"][solver] = {
                "selected_cell": selected_name,
                "level": _slice_summaries(
                    base_seed_mean["selected"][iid],
                    base_seed_mean["selected"][grouped],
                    bootstrap_indices,
                    strict=strict,
                ),
                "delta_vs_unit": _slice_summaries(
                    (base_seed_mean["selected"] - base_seed_mean["unit"])[iid],
                    (base_seed_mean["selected"] - base_seed_mean["unit"])[grouped],
                    bootstrap_indices,
                    strict=strict,
                ),
                "delta_vs_learned": _slice_summaries(
                    (base_seed_mean["selected"] - base_seed_mean["learned"])[iid],
                    (base_seed_mean["selected"] - base_seed_mean["learned"])[grouped],
                    bootstrap_indices,
                    strict=strict,
                ),
                "delta_vs_delivered": _slice_summaries(
                    (base_seed_mean["selected"] - base_seed_mean["delivered"])[iid],
                    (base_seed_mean["selected"] - base_seed_mean["delivered"])[grouped],
                    bootstrap_indices,
                    strict=strict,
                ),
            }
            if arm in PRIMARY_ARMS:
                primary_index = PRIMARY_ARMS.index(arm)
                effects["arms"][arm]["temperature"][solver] = {}
                supported_relations = RELATIONS if solver == "wls" else ("observed",)
                for relation_name in supported_relations:
                    alpha = temperature_selection["arms"][arm][solver][relation_name]["selected_alpha"]
                    variant = VARIANTS.index(
                        f"{relation_name}|alpha={_alpha_label(alpha)}|{solver}"
                    )
                    selected = dis._strict_seed_mean(quotient[primary_index, :, variant])
                    unit_variant = VARIANTS.index(f"{relation_name}|alpha=0.00|{solver}")
                    learned_variant = VARIANTS.index(f"{relation_name}|alpha=1.00|{solver}")
                    unit = dis._strict_seed_mean(quotient[primary_index, :, unit_variant])
                    learned = dis._strict_seed_mean(quotient[primary_index, :, learned_variant])
                    effects["arms"][arm]["temperature"][solver][relation_name] = {
                        "selected_alpha": alpha,
                        "delta_vs_unit": _slice_summaries(
                            (selected - unit)[iid], (selected - unit)[grouped], bootstrap_indices, strict=strict
                        ),
                        "delta_vs_learned": _slice_summaries(
                            (selected - learned)[iid], (selected - learned)[grouped], bootstrap_indices, strict=strict
                        ),
                    }
                if solver == "wls":
                    learned_variant = VARIANTS.index("observed|alpha=1.00|wls")
                    learned = dis._strict_seed_mean(quotient[primary_index, :, learned_variant])
                    shuffled_replicates = []
                    for replicate in range(8):
                        variant = VARIANTS.index(f"observed|shuffle={replicate:02d}|wls")
                        shuffled_replicates.append(
                            dis._strict_seed_mean(quotient[primary_index, :, variant])
                        )
                    shuffled_matrix = np.asarray(shuffled_replicates)
                    shuffled = np.where(
                        np.all(np.isfinite(shuffled_matrix), axis=0),
                        np.mean(shuffled_matrix, axis=0),
                        np.nan,
                    )
                    unit = base_seed_mean["unit"]
                    effects["arms"][arm]["shuffle"][solver] = {
                        "learned_minus_shuffled": _slice_summaries(
                            (learned - shuffled)[iid], (learned - shuffled)[grouped], bootstrap_indices, strict=False
                        ),
                        "shuffled_minus_unit": _slice_summaries(
                            (shuffled - unit)[iid], (shuffled - unit)[grouped], bootstrap_indices, strict=False
                        ),
                        "replicate_sd_mean": {
                            "test|iid": float(np.nanmean(np.nanstd(shuffled_matrix[:, iid], axis=0))),
                            "test|grouped": float(np.nanmean(np.nanstd(shuffled_matrix[:, grouped], axis=0))),
                        },
                    }
            predicted = predictions[arm_index, solver_index]
            choose_learned = decisions[arm_index, solver_index]
            actual_delta = base_seed_mean["learned"] - base_seed_mean["unit"]
            policy = np.where(choose_learned, base_seed_mean["learned"], base_seed_mean["unit"])
            failed_selected = choose_learned & ~np.isfinite(base_seed_mean["learned"])
            rescue = policy.copy()
            rescue[failed_selected] = base_seed_mean["unit"][failed_selected]
            oracle = np.fmin(base_seed_mean["unit"], base_seed_mean["learned"])
            solver_report: dict[str, Any] = {
                "selected_fraction": {
                    "validation": float(np.mean(choose_learned[canonical["split"] == "validation"])),
                    "test|iid": float(np.mean(choose_learned[iid])),
                    "test|grouped": float(np.mean(choose_learned[grouped])),
                },
                "prediction": {},
                "policy_delta_vs_unit": _slice_summaries(
                    (policy - base_seed_mean["unit"])[iid],
                    (policy - base_seed_mean["unit"])[grouped],
                    bootstrap_indices,
                    strict=strict,
                ),
                "policy_regret_vs_two_cell_oracle": _slice_summaries(
                    (policy - oracle)[iid], (policy - oracle)[grouped], bootstrap_indices, strict=strict
                ),
                "rescue_delta_vs_unit": _slice_summaries(
                    (rescue - base_seed_mean["unit"])[iid],
                    (rescue - base_seed_mean["unit"])[grouped],
                    bootstrap_indices,
                    strict=False,
                ),
                "retry_rate": {
                    "test|iid": float(np.mean(failed_selected[iid])),
                    "test|grouped": float(np.mean(failed_selected[grouped])),
                },
            }
            for name, mask in (
                ("validation", canonical["split"] == "validation"),
                ("test|iid", np.isin(np.arange(len(predicted)), iid)),
                ("test|grouped", np.isin(np.arange(len(predicted)), grouped)),
            ):
                finite = mask & np.isfinite(actual_delta) & np.isfinite(predicted)
                solver_report["prediction"][name] = {
                    "n": int(finite.sum()),
                    "mae": float(np.mean(np.abs(predicted[finite] - actual_delta[finite]))),
                    "correlation": _safe_correlation(predicted[finite], actual_delta[finite]),
                }
            transport["arms"][arm][solver] = solver_report
    return effects, transport


def _effective_fraction(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    total = float(values.sum())
    return float(total * total / np.sum(values * values) / len(values))


def _bottom_overlap(left: np.ndarray, right: np.ndarray) -> float:
    k = max(1, int(math.ceil(0.1 * len(left))))
    left_ids = set(np.argsort(left, kind="stable")[:k].tolist())
    right_ids = set(np.argsort(right, kind="stable")[:k].tolist())
    return float(len(left_ids & right_ids) / k)


def _weight_semantics(
    canonical: dict[str, np.ndarray],
    neural_raws: dict[str, dict[str, np.ndarray]],
    solver_raws: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    shape = (len(ARMS), len(SEEDS), len(canonical["view_id"]))
    names = (
        "base_effective_fraction",
        "unit_irls_effective_fraction",
        "learned_irls_effective_fraction",
        "tail_overlap",
        "base_corrupt_mass",
        "unit_irls_corrupt_mass",
        "learned_irls_corrupt_mass",
        "learned_minus_unit_irls_rmse",
        "effective_fraction_change",
    )
    arrays = {name: np.full(shape, np.nan) for name in names}
    for arm_index, arm in enumerate(ARMS):
        for seed_index, seed in enumerate(SEEDS):
            key = f"{arm}|seed={seed}"
            neural = neural_raws[key]
            states = solver_raws[key]
            unit_cell = dis._cell_index("observed", "unit", "irls")
            learned_cell = dis._cell_index("observed", "learned", "irls")
            for view_index in range(len(canonical["view_id"])):
                edge_slice, _, _ = dis._view_slices(canonical, view_index)
                valid = canonical["edge_valid"][edge_slice]
                base = neural["reliability"][edge_slice][valid].astype(np.float64)
                unit_final = states["final_weights"][unit_cell, edge_slice][valid]
                learned_final = states["final_weights"][learned_cell, edge_slice][valid]
                mask = neural["causal_corruption_mask"][edge_slice][valid]
                relative = learned_final / base
                relative /= relative.mean()
                reconstructed = base * relative
                reconstructed /= reconstructed.mean()
                expected = learned_final / learned_final.mean()
                if not np.allclose(reconstructed, expected, rtol=0.0, atol=1e-12):
                    raise AssertionError("effective robust multiplier reconstruction failed")
                arrays["base_effective_fraction"][arm_index, seed_index, view_index] = _effective_fraction(base)
                arrays["unit_irls_effective_fraction"][arm_index, seed_index, view_index] = _effective_fraction(unit_final)
                arrays["learned_irls_effective_fraction"][arm_index, seed_index, view_index] = _effective_fraction(learned_final)
                arrays["tail_overlap"][arm_index, seed_index, view_index] = _bottom_overlap(base, unit_final)
                arrays["base_corrupt_mass"][arm_index, seed_index, view_index] = float(base[mask].sum() / base.sum())
                arrays["unit_irls_corrupt_mass"][arm_index, seed_index, view_index] = float(unit_final[mask].sum() / unit_final.sum())
                arrays["learned_irls_corrupt_mass"][arm_index, seed_index, view_index] = float(learned_final[mask].sum() / learned_final.sum())
                learned_q = states["quotient_rmse"][learned_cell, view_index]
                unit_q = states["quotient_rmse"][unit_cell, view_index]
                arrays["learned_minus_unit_irls_rmse"][arm_index, seed_index, view_index] = learned_q - unit_q
                arrays["effective_fraction_change"][arm_index, seed_index, view_index] = (
                    _effective_fraction(learned_final) - _effective_fraction(unit_final)
                )
    report: dict[str, Any] = {"arms": {}}
    for arm_index, arm in enumerate(ARMS):
        report["arms"][arm] = {}
        seed_means = {name: dis._strict_seed_mean(values[arm_index]) for name, values in arrays.items()}
        for split_name, mask in (
            ("validation", canonical["split"] == "validation"),
            ("test|iid", (canonical["split"] == "test") & (canonical["mechanism"] == "iid")),
            ("test|grouped", (canonical["split"] == "test") & (canonical["mechanism"] == "grouped")),
        ):
            report["arms"][arm][split_name] = {
                name: float(np.nanmean(values[mask])) for name, values in seed_means.items()
            }
            x = seed_means["effective_fraction_change"][mask]
            y = seed_means["learned_minus_unit_irls_rmse"][mask]
            finite = np.isfinite(x) & np.isfinite(y)
            report["arms"][arm][split_name]["ess_change_rmse_delta_correlation"] = _safe_correlation(x[finite], y[finite])
    return report, arrays


def _failure_diagnostics(
    canonical: dict[str, np.ndarray], converged: np.ndarray
) -> dict[str, Any]:
    records = []
    for arm_index, arm in enumerate(PRIMARY_ARMS):
        for seed_index, seed in enumerate(SEEDS):
            for variant_index, variant in enumerate(VARIANTS):
                for split, mechanism in (
                    ("validation", "iid"),
                    ("test", "iid"),
                    ("test", "grouped"),
                ):
                    mask = (canonical["split"] == split) & (
                        canonical["mechanism"] == mechanism
                    )
                    values = converged[arm_index, seed_index, variant_index, mask]
                    records.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "variant": variant,
                            "split": split,
                            "mechanism": mechanism,
                            "n_total": int(len(values)),
                            "n_failed": int((~values).sum()),
                            "failure_rate": float(np.mean(~values)),
                        }
                    )
    return {
        "schema_version": "proportional-graph-solver-interface-failures-v1",
        "policy": "any required IRLS nonconvergence makes the affected strict slice not evaluable",
        "records": records,
    }


def _report_text(
    static: dict[str, Any],
    temperature: dict[str, Any],
    effects: dict[str, Any],
    transport: dict[str, Any],
    semantics: dict[str, Any],
) -> str:
    def estimate(entry: dict[str, Any]) -> str:
        return "NE" if entry["status"] != "ESTIMATED" else f"{entry['mean']:.6f}"

    lines = [
        "# Diagnóstico CPU de interfaz solver-condicionada",
        "",
        "Este seguimiento reutiliza estados congelados, ejecuta perturbaciones de peso y permanece exploratorio porque test ya había sido abierto.",
        "",
        "## Selección estática y temperado",
        "",
        "| brazo | solver | celda estática validation | delta vs unit IID | delta vs unit grouped | alpha observado | alpha corregido |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for solver in SOLVERS:
            selected = temperature.get("arms", {}).get(arm, {}).get(solver, {})
            observed_alpha = (
                f"{selected['observed']['selected_alpha']:.2f}"
                if "observed" in selected
                else "—"
            )
            corrected_alpha = (
                f"{selected['corrected']['selected_alpha']:.2f}"
                if "corrected" in selected
                else "—"
            )
            static_effect = effects["arms"][arm]["static"][solver]["delta_vs_unit"]
            lines.append(
                f"| `{arm}` | `{solver}` | `{static['arms'][arm][solver]['selected_cell']}` | "
                f"{estimate(static_effect['test|iid'])} | "
                f"{estimate(static_effect['test|grouped'])} | "
                f"{observed_alpha} | {corrected_alpha} |"
            )
    lines.extend(
        [
            "",
            "## Temperado en brazos primarios",
            "",
            "| brazo | solver | relación | alpha validation | delta vs unit IID | delta vs unit grouped |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for arm in PRIMARY_ARMS:
        for solver in SOLVERS:
            for relation, payload in effects["arms"][arm]["temperature"].get(solver, {}).items():
                lines.append(
                    f"| `{arm}` | `{solver}` | `{relation}` | {payload['selected_alpha']:.2f} | "
                    f"{estimate(payload['delta_vs_unit']['test|iid'])} | "
                    f"{estimate(payload['delta_vs_unit']['test|grouped'])} |"
                )
    lines.extend(
        [
            "",
            "## Ubicación del peso bajo WLS",
            "",
            "| brazo | learned−shuffled IID | learned−shuffled grouped | shuffled−unit IID | shuffled−unit grouped |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in PRIMARY_ARMS:
        payload = effects["arms"][arm]["shuffle"]["wls"]
        lines.append(
            f"| `{arm}` | {estimate(payload['learned_minus_shuffled']['test|iid'])} | "
            f"{estimate(payload['learned_minus_shuffled']['test|grouped'])} | "
            f"{estimate(payload['shuffled_minus_unit']['test|iid'])} | "
            f"{estimate(payload['shuffled_minus_unit']['test|grouped'])} |"
        )
    lines.extend(
        [
            "",
            "## Transporte del predictor público",
            "",
            "| brazo | solver | fracción learned IID/grouped | correlación IID/grouped | política−unit IID/grouped |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for arm in PRIMARY_ARMS:
        for solver in SOLVERS:
            payload = transport["arms"][arm][solver]
            lines.append(
                f"| `{arm}` | `{solver}` | {payload['selected_fraction']['test|iid']:.3f}/"
                f"{payload['selected_fraction']['test|grouped']:.3f} | "
                f"{payload['prediction']['test|iid']['correlation']:.3f}/"
                f"{payload['prediction']['test|grouped']['correlation']:.3f} | "
                f"{estimate(payload['policy_delta_vs_unit']['test|iid'])}/"
                f"{estimate(payload['policy_delta_vs_unit']['test|grouped'])} |"
            )
    lines.extend(
        [
            "",
            "## Diagnóstico IRLS de masa efectiva",
            "",
            "| brazo | slice | ESS unit→learned | solapamiento de cola | learned−unit RMSE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for arm in PRIMARY_ARMS:
        for slice_name in ("test|iid", "test|grouped"):
            payload = semantics["arms"][arm][slice_name]
            lines.append(
                f"| `{arm}` | `{slice_name}` | "
                f"{payload['unit_irls_effective_fraction']:.3f}→"
                f"{payload['learned_irls_effective_fraction']:.3f} | "
                f"{payload['tail_overlap']:.3f} | "
                f"{payload['learned_minus_unit_irls_rmse']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Fronteras",
            "",
            "Los JSON estructurados conservan efectos, transporte, shuffles, fallos y semántica de pesos. Los intervalos son bootstrap marginales, sin p-values. Los rescates IRLS son diagnósticos operacionales y no sustituyen el estimando estricto.",
            "",
            "El resultado no promueve arquitectura, no acredita geometría natural y no declara GO/NO-GO.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_replay(output: Path, config_path: Path, development: bool) -> None:
    replay = output / "replay.sh"
    development_flag = " --development" if development else ""
    text = f"""#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python {RUNNER_PATH.relative_to(REPO_ROOT)} \\
  --config {config_path.relative_to(REPO_ROOT)} \\
  --output data/geometria_proporcional/proportional_graph_solver_interface_diagnostic_v1_replay{development_flag}
"""
    replay.write_text(text, encoding="utf-8")
    replay.chmod(0o755)


def _manifest(
    output: Path,
    config: dict[str, Any],
    source_records: dict[str, Any],
    pools: list[dict[str, Any]],
) -> dict[str, Any]:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "runtime_observation.json"}:
            relative = str(path.relative_to(output))
            files[relative] = {"bytes": path.stat().st_size, "sha256": dis._sha256_file(path)}
    status = dis._git_output("status", "--porcelain")
    return {
        "schema_version": config["artifact_schema_version"],
        "git": {
            "head": dis._git_output("rev-parse", "HEAD"),
            "branch": dis._git_output("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "status_sha256": dis._sha256_bytes(status.encode("utf-8")),
        },
        "official_source_files": source_records,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_imported": "torch" in sys.modules,
            "threadpools": pools,
        },
        "resolved_config": config,
        "files": files,
        "replay_scope": {
            "deterministic_files": sorted(files),
            "excluded": ["runtime_observation.json"],
        },
    }


def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    config_path = args.config.resolve(strict=True)
    config = _load_config(config_path)
    dis._guard_cpu_environment(config)
    (
        smoke,
        disentangle,
        canonical,
        neural_raws,
        global_metrics,
        solver_raws,
        source_bundle,
    ) = _verify_sources(config, started)
    output = args.output.resolve()
    source_records = dis._tracked_clean_records(
        (PLAN_PATH, config_path, RUNNER_PATH, BASE_RUNNER_PATH, SOLVER_PATH),
        args.development,
    )
    _prepare_output(output, (smoke, disentangle))
    static_selection = _validation_selection(canonical, global_metrics)
    metric_shape = (
        len(PRIMARY_ARMS),
        len(SEEDS),
        len(VARIANTS),
        len(canonical["view_id"]),
    )
    quotient = np.full(metric_shape, np.nan)
    converged = np.zeros(metric_shape, dtype=bool)
    iterations = np.full(metric_shape, -1, dtype=np.int64)
    features = np.full(
        (len(ARMS), len(SEEDS), len(canonical["view_id"]), len(FEATURE_ORDER)), np.nan
    )
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        if any(int(pool.get("num_threads", 1)) != 1 for pool in pools):
            raise RuntimeError(f"numeric thread limit failed: {pools}")
        for arm_index, arm in enumerate(PRIMARY_ARMS):
            for seed_index, seed in enumerate(SEEDS):
                key = f"{arm}|seed={seed}"
                payload, model_features = _run_variants(
                    canonical,
                    neural_raws[key],
                    solver_raws[key],
                    arm,
                    seed,
                    config,
                    started,
                )
                quotient[arm_index, seed_index] = payload["quotient_rmse"]
                converged[arm_index, seed_index] = payload["converged"]
                iterations[arm_index, seed_index] = payload["iterations"]
                features[arm_index, seed_index] = model_features
                dis._write_deterministic_npz(
                    output / "raw_solver" / f"{key}.npz", **payload
                )
                dis._enforce_budget(started, config, f"serialization {key}")
        for arm_index, arm in enumerate(SECONDARY_ARMS, start=len(PRIMARY_ARMS)):
            for seed_index, seed in enumerate(SEEDS):
                features[arm_index, seed_index] = _features_for_raw(
                    canonical, neural_raws[f"{arm}|seed={seed}"]
                )
        temperature_selection = _temperature_selection(canonical, quotient)
        ridge_models, predictions, decisions = _fit_transport_models(
            canonical, features, global_metrics, config
        )
        effects, transport = _evaluate_effects(
            canonical,
            source_bundle["indices"],
            global_metrics,
            quotient,
            static_selection,
            temperature_selection,
            predictions,
            decisions,
        )
        semantics, semantic_arrays = _weight_semantics(
            canonical, neural_raws, solver_raws
        )
        failures = _failure_diagnostics(canonical, converged)
    dis._write_json(output / "resolved_config.json", config)
    source_bundle["attestation"]["official_source_files"] = source_records
    source_bundle["attestation"]["input_boundary"] = (
        "selectors and transformations use the frozen public whitelist; private targets score outputs, "
        "and the causal mask appears only in post-solver weight diagnostics"
    )
    dis._write_json(output / "source_attestation.json", source_bundle["attestation"])
    dis._write_json(output / "static_selection.json", static_selection)
    dis._write_json(output / "temperature_selection.json", temperature_selection)
    dis._write_deterministic_npz(
        output / "public_features.npz",
        feature_order=np.asarray(FEATURE_ORDER),
        features=features,
        arm=np.asarray(ARMS),
        seed=np.asarray(SEEDS),
        view_id=canonical["view_id"],
        master_id=canonical["master_id"],
        split=canonical["split"],
    )
    dis._write_deterministic_npz(
        output / "per_view_metrics.npz",
        solve_arm=np.asarray(PRIMARY_ARMS),
        diagnostic_arm=np.asarray(ARMS),
        seed=np.asarray(SEEDS),
        variant_id=np.asarray(VARIANTS),
        view_id=canonical["view_id"],
        master_id=canonical["master_id"],
        split=canonical["split"],
        mechanism=canonical["mechanism"],
        quotient_rmse=quotient,
        converged=converged,
        iterations=iterations,
        ridge_prediction=predictions,
        ridge_choose_learned=decisions,
        **semantic_arrays,
    )
    dis._write_json(output / "ridge_models.json", ridge_models)
    dis._write_json(output / "effects.json", effects)
    dis._write_json(output / "transport_diagnostic.json", transport)
    dis._write_json(output / "weight_semantics.json", semantics)
    dis._write_json(output / "failure_diagnostics.json", failures)
    shutil.copyfile(disentangle / "bootstrap_indices.npz", output / "bootstrap_indices.npz")
    (output / "SOLVER_INTERFACE_REPORT.md").write_text(
        _report_text(
            static_selection,
            temperature_selection,
            effects,
            transport,
            semantics,
        ),
        encoding="utf-8",
    )
    _write_replay(output, config_path, args.development)
    dis._enforce_budget(started, config, "artifact finalization")
    dis._write_json(output / "manifest.json", _manifest(output, config, source_records, pools))
    runtime = dis._resource_observation(started)
    dis._write_json(
        output / "runtime_observation.json",
        {
            **runtime,
            "included_in_byte_exact_replay": False,
            "reason": "wall time and peak RSS are run-specific observations",
        },
    )
    print(dis._canonical_json({"output": str(output), **runtime}))


if __name__ == "__main__":
    main()
