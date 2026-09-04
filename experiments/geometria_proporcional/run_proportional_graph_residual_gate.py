#!/usr/bin/env python3
"""Evaluate an identity-preserving public residual gate, CPU-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/geometria_proporcional"))

import run_proportional_graph_frozen_adapters as frozen  # noqa: E402
import run_proportional_graph_neural_smoke as smoke  # noqa: E402
import run_proportional_graph_solver_interface_diagnostic as interface  # noqa: E402
import torch  # noqa: E402
from geometria_proporcional.proportional_graph_contract import (  # noqa: E402
    ProportionalGraphConfig,
    generate_graph_views,
    score_solver,
    solve_huber_irls,
)

DEFAULT_CONFIG = (
    ROOT
    / "experiments/geometria_proporcional/configs/proportional_graph_residual_gate_v1.json"
)
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_RESIDUAL_GATE_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_residual_gate_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_residual_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_irls_loss_contrast.py",
    "experiments/geometria_proporcional/run_proportional_graph_solver_interface_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py",
    "src/geometria_proporcional/proportional_graph_contract.py",
)
FEATURE_ORDER = tuple(interface.FEATURE_ORDER[:15])
REDUCED_FEATURES = ("correction_rms", "correction_abs_max")
POLICIES = (
    "identity",
    "always_post",
    "constant_validation",
    "correction_scale_ridge",
    "public_ridge_gate",
    "shuffled_target_ridge",
    "oracle_per_view",
)


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    expected = {
        "schema_version",
        "source_loss_contrast",
        "source_manifest_sha256",
        "source_smoke_config",
        "arms",
        "seeds",
        "alphas",
        "ridge_lambdas",
        "ridge_folds",
        "shuffle_replicates",
        "shuffle_seed",
        "bootstrap_replicates",
        "bootstrap_seed",
        "execution",
    }
    if (
        set(cfg) != expected
        or cfg["schema_version"] != "proportional-graph-residual-gate-v1"
    ):
        raise ValueError("invalid residual gate schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms must remain frozen")
    if cfg["seeds"] != [104729, 130363] or cfg["alphas"] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("seed and alpha contracts must remain frozen")
    if cfg["ridge_lambdas"] != [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
        raise ValueError("ridge grid must remain frozen")
    if cfg["execution"]["torch_threads"] != 1:
        raise ValueError("runner must remain single-threaded")
    return cfg


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def resource_sample(started: float, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }


def enforce(sample: dict[str, Any], cfg: dict[str, Any]) -> None:
    execution = cfg["execution"]
    if sample["elapsed_seconds"] > execution["max_seconds"]:
        raise RuntimeError("runtime budget exceeded")
    if sample["max_rss_gib"] > execution["max_rss_gib"]:
        raise RuntimeError("RSS budget exceeded")


def public_features(view: Any, corrected: np.ndarray) -> np.ndarray:
    """Whitelisted pre-solver features; private graph state is inaccessible."""
    observation = view.public
    values = interface._public_features(
        n_nodes=observation.n_nodes,
        edge_index=observation.edge_index,
        edge_valid=observation.edge_valid,
        edge_variance=observation.edge_variance,
        observed_log_ratio=observation.observed_log_ratio,
        path_index=observation.path_index,
        path_sign=observation.path_sign,
        path_valid=observation.path_valid,
        corrected_log_ratio=np.asarray(corrected, dtype=np.float64),
        reliability=np.ones(len(observation.observed_log_ratio), dtype=np.float64),
    )
    return values[: len(FEATURE_ORDER)]


def load_post_relations(
    source: Path, arm: str, seed: int
) -> tuple[list[str], dict[str, np.ndarray]]:
    with np.load(
        source / "raw_eval" / f"{arm}|seed={seed}.npz", allow_pickle=False
    ) as saved:
        ids = [str(x) for x in saved["view_id"]]
        offsets = saved["edge_offset"]
        values = saved["post_irls_corrected_relation"]
        return ids, {
            view_id: values[offsets[i] : offsets[i + 1]].astype(np.float64)
            for i, view_id in enumerate(ids)
        }


def verify_source(source: Path, expected_manifest_hash: str) -> None:
    if sha(source / "manifest.json") != expected_manifest_hash:
        raise AssertionError("source manifest hash mismatch")
    manifest = json.loads((source / "manifest.json").read_text())
    deterministic = manifest.get("deterministic_files", {})
    required = [
        "bootstrap_indices.npz",
        *[
            f"raw_eval/{arm}|seed={seed}.npz"
            for arm in ("raw_generic", "raw_typed", "closure_generic", "closure_typed")
            for seed in (104729, 130363)
        ],
    ]
    for relative in required:
        if deterministic.get(relative) != sha(source / relative):
            raise AssertionError(f"source hash mismatch: {relative}")


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def fit_gate(
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_all: np.ndarray,
    master_ids: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    folds = interface._fold_ids(master_ids, cfg["ridge_folds"])
    candidates = []
    cached: dict[float, np.ndarray] = {}
    for ridge in cfg["ridge_lambdas"]:
        oof = np.full_like(y_validation, np.nan)
        for fold in range(cfg["ridge_folds"]):
            train, held = folds != fold, folds == fold
            for column in range(y_validation.shape[1]):
                model = interface._ridge_fit(
                    x_validation[train], y_validation[train, column], ridge
                )
                oof[held, column] = interface._ridge_predict(model, x_validation[held])
        mse = float(np.mean((oof - y_validation) ** 2))
        candidates.append({"lambda": ridge, "cv_mse": mse})
        cached[float(ridge)] = oof
    best_mse = min(row["cv_mse"] for row in candidates)
    selected = max(
        (row for row in candidates if row["cv_mse"] <= best_mse + 1e-15),
        key=lambda row: row["lambda"],
    )
    ridge = float(selected["lambda"])
    predictions = np.full((len(x_all), y_validation.shape[1]), np.nan)
    models = []
    for column in range(y_validation.shape[1]):
        model = interface._ridge_fit(x_validation, y_validation[:, column], ridge)
        predictions[:, column] = interface._ridge_predict(model, x_all)
        models.append(
            {
                "intercept": model["intercept"],
                "mean": model["mean"].tolist(),
                "scale": model["scale"].tolist(),
                "coefficients": model["coefficients"].tolist(),
            }
        )
    return (
        {
            "selected_lambda": ridge,
            "candidates": candidates,
            "fold_id": folds.tolist(),
            "models": models,
        },
        predictions,
        cached[ridge],
    )


def choose_alpha(predicted_nonzero_deltas: np.ndarray) -> np.ndarray:
    with_identity = np.column_stack(
        (np.zeros(len(predicted_nonzero_deltas)), predicted_nonzero_deltas)
    )
    return np.argmin(with_identity, axis=1).astype(np.int64)


def stratified_rotation(n_nodes: np.ndarray, seed: int) -> np.ndarray:
    order = np.arange(len(n_nodes))
    result = order.copy()
    rng = np.random.default_rng(seed)
    for value in sorted(set(n_nodes.tolist())):
        group = np.flatnonzero(n_nodes == value)
        if len(group) < 2:
            raise ValueError("shuffle stratum cannot be deranged")
        shift = int(rng.integers(1, len(group)))
        result[group] = np.roll(group, shift)
    if np.any(result == order):
        raise AssertionError("target shuffle retained an identity")
    return result


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(
    quotient: np.ndarray,
    decisions: dict[str, np.ndarray],
    shuffle_decisions: np.ndarray,
    views: list[Any],
    bootstrap_masters: np.ndarray,
    bootstrap_indices: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {"arms": {}}
    by_master = {
        (str(view.private.master_id), view.private.corruption_mechanism): i
        for i, view in enumerate(views)
        if view.private.split == "test"
    }
    for arm_index, arm in enumerate(cfg["arms"]):
        result["arms"][arm] = {}
        policy_values: dict[str, np.ndarray] = {}
        for policy, action in decisions.items():
            chosen = np.take_along_axis(
                quotient[arm_index],
                action[arm_index][None, None, :].repeat(len(cfg["seeds"]), axis=0),
                axis=1,
            )[:, 0, :]
            policy_values[policy] = chosen
        shuffled = []
        for replicate in range(cfg["shuffle_replicates"]):
            action = shuffle_decisions[arm_index, replicate]
            shuffled.append(
                np.take_along_axis(
                    quotient[arm_index],
                    action[None, None, :].repeat(len(cfg["seeds"]), axis=0),
                    axis=1,
                )[:, 0, :]
            )
        policy_values["shuffled_target_ridge"] = np.mean(shuffled, axis=0)
        for mechanism in ("iid", "grouped"):
            indices = np.asarray(
                [by_master[(str(master), mechanism)] for master in bootstrap_masters],
                dtype=np.int64,
            )
            result["arms"][arm][f"test_{mechanism}"] = {}
            for policy in POLICIES:
                seed_values = policy_values[policy][:, indices]
                master_values = seed_values.mean(axis=0)
                draws = master_values[bootstrap_indices].mean(axis=1)
                payload = {
                    "mean_rmse": float(master_values.mean()),
                    "ci95_mean": [
                        float(np.percentile(draws, 2.5)),
                        float(np.percentile(draws, 97.5)),
                    ],
                    "seed_means": seed_values.mean(axis=1).tolist(),
                }
                for baseline in (
                    "identity",
                    "always_post",
                    "constant_validation",
                    "correction_scale_ridge",
                    "shuffled_target_ridge",
                ):
                    if policy == baseline:
                        continue
                    delta = master_values - policy_values[baseline][:, indices].mean(
                        axis=0
                    )
                    delta_draws = delta[bootstrap_indices].mean(axis=1)
                    payload[f"minus_{baseline}"] = {
                        "mean": float(delta.mean()),
                        "ci95": [
                            float(np.percentile(delta_draws, 2.5)),
                            float(np.percentile(delta_draws, 97.5)),
                        ],
                    }
                result["arms"][arm][f"test_{mechanism}"][policy] = payload
        slice_masks = {
            "validation_refit_in_sample": np.asarray(
                [view.private.split == "validation" for view in views]
            ),
            "test_iid": np.asarray(
                [
                    view.private.split == "test"
                    and view.private.corruption_mechanism == "iid"
                    for view in views
                ]
            ),
            "test_grouped": np.asarray(
                [
                    view.private.split == "test"
                    and view.private.corruption_mechanism == "grouped"
                    for view in views
                ]
            ),
        }
        result["arms"][arm]["action_fractions"] = {}
        for label, mask in slice_masks.items():
            result["arms"][arm]["action_fractions"][label] = {
                policy: {
                    str(alpha): float(np.mean(action[arm_index, mask] == alpha_index))
                    for alpha_index, alpha in enumerate(cfg["alphas"])
                }
                for policy, action in decisions.items()
            }
            result["arms"][arm]["action_fractions"][label]["shuffled_target_ridge"] = {
                str(alpha): float(
                    np.mean(shuffle_decisions[arm_index, :, mask] == alpha_index)
                )
                for alpha_index, alpha in enumerate(cfg["alphas"])
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if (
        not args.development
        and subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    ):
        raise RuntimeError("official run requires clean worktree")
    output.mkdir(parents=True)
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    started = time.monotonic()
    samples = [resource_sample(started, "start")]
    source = ROOT / cfg["source_loss_contrast"]
    verify_source(source, cfg["source_manifest_sha256"])
    source_cfg = smoke._load_config((ROOT / cfg["source_smoke_config"]).resolve())
    graph = source_cfg["graph"]
    all_views = generate_graph_views(ProportionalGraphConfig.from_dict(graph))
    shuffle_arm = next(
        a for a in source_cfg["arms"] if a["name"] == "closure_typed_path_shuffle"
    )
    eligible = {
        view.private.master_id: bool(
            smoke._view_tensors(view, shuffle_arm, cfg["seeds"][0])[
                "path_shuffle_eligible"
            ]
        )
        for view in all_views
    }
    eligible_views = [
        view
        for view in all_views
        if eligible[view.private.master_id]
        and view.private.split in {"validation", "test"}
    ]
    canonical_ids, _ = load_post_relations(source, cfg["arms"][0], cfg["seeds"][0])
    view_by_id = {view.private.view_id: view for view in eligible_views}
    if set(canonical_ids) != set(view_by_id):
        raise AssertionError("source and regenerated view universes differ")
    views = [view_by_id[view_id] for view_id in canonical_ids]
    view_ids = [view.private.view_id for view in views]
    n_arms, n_seeds, n_alpha, n_views = (
        len(cfg["arms"]),
        len(cfg["seeds"]),
        len(cfg["alphas"]),
        len(views),
    )
    features = np.full((n_arms, n_seeds, n_views, len(FEATURE_ORDER)), np.nan)
    quotient = np.full((n_arms, n_seeds, n_alpha, n_views), np.nan)
    relation_rmse = np.full_like(quotient, np.nan)
    converged = np.zeros_like(quotient, dtype=bool)
    iterations = np.zeros_like(quotient, dtype=np.int64)
    conditions = np.full_like(quotient, np.nan)
    for arm_index, arm in enumerate(cfg["arms"]):
        for seed_index, seed in enumerate(cfg["seeds"]):
            ids, post = load_post_relations(source, arm, seed)
            if ids != view_ids:
                raise AssertionError("source raw view order mismatch")
            x_states, weight_states = [[] for _ in cfg["alphas"]], [
                [] for _ in cfg["alphas"]
            ]
            node_offsets, edge_offsets = [0], [0]
            for view_index, view in enumerate(views):
                observed = np.asarray(view.public.observed_log_ratio, dtype=np.float64)
                corrected = post[view.private.view_id]
                features[arm_index, seed_index, view_index] = public_features(
                    view, corrected
                )
                node_offsets.append(node_offsets[-1] + view.public.n_nodes)
                edge_offsets.append(edge_offsets[-1] + len(observed))
                for alpha_index, alpha in enumerate(cfg["alphas"]):
                    relation = observed + alpha * (corrected - observed)
                    solved = solve_huber_irls(
                        view.public,
                        values=relation,
                        base_weights=None,
                        delta=graph["huber_delta"],
                        max_iterations=graph["irls_iterations"],
                        damping=graph["irls_damping"],
                        weight_floor=graph["weight_floor"],
                    )
                    score = score_solver(solved, view.private)
                    converged[arm_index, seed_index, alpha_index, view_index] = (
                        solved.converged
                    )
                    iterations[arm_index, seed_index, alpha_index, view_index] = (
                        solved.iterations
                    )
                    conditions[arm_index, seed_index, alpha_index, view_index] = (
                        solved.laplacian_condition
                    )
                    relation_rmse[arm_index, seed_index, alpha_index, view_index] = (
                        float(
                            np.sqrt(
                                np.mean((relation - view.private.clean_log_ratio) ** 2)
                            )
                        )
                    )
                    if solved.converged:
                        quotient[arm_index, seed_index, alpha_index, view_index] = (
                            score.quotient_rmse
                        )
                    x_states[alpha_index].append(solved.x_hat)
                    weight_states[alpha_index].append(solved.weights)
                if (view_index + 1) % cfg["execution"]["sample_every_views"] == 0:
                    sample = resource_sample(
                        started, f"{arm}|{seed}|view={view_index + 1}"
                    )
                    samples.append(sample)
                    enforce(sample, cfg)
            frozen.save_npz(
                output / "raw_solver" / f"{arm}|seed={seed}.npz",
                {
                    "view_id": np.asarray(view_ids),
                    "node_offset": np.asarray(node_offsets, dtype=np.int64),
                    "edge_offset": np.asarray(edge_offsets, dtype=np.int64),
                    "x_hat": np.stack([np.concatenate(values) for values in x_states]),
                    "final_weights": np.stack(
                        [np.concatenate(values) for values in weight_states]
                    ),
                    "converged": converged[arm_index, seed_index],
                    "iterations": iterations[arm_index, seed_index],
                    "condition": conditions[arm_index, seed_index],
                },
            )
    if not np.all(converged) or not np.all(np.isfinite(quotient)):
        raise RuntimeError("strict IRLS convergence contract failed")
    validation = np.asarray([view.private.split == "validation" for view in views])
    test = ~validation
    if validation.sum() != 127 or test.sum() != 504:
        raise AssertionError("unexpected validation/test universe")
    feature_mean = features.mean(axis=1)
    quotient_mean = quotient.mean(axis=1)
    y_delta = (
        quotient_mean[:, 1:, :][:, :, validation]
        - quotient_mean[:, :1, :][:, :, validation]
    )
    y_delta = np.transpose(y_delta, (0, 2, 1))
    validation_masters = np.asarray([view.private.master_id for view in views])[
        validation
    ]
    n_nodes_validation = np.asarray([view.public.n_nodes for view in views])[validation]
    gates: dict[str, Any] = {"feature_order": list(FEATURE_ORDER), "arms": {}}
    decisions = {
        policy: np.zeros((n_arms, n_views), dtype=np.int64)
        for policy in POLICIES
        if policy != "shuffled_target_ridge"
    }
    decisions["always_post"][:] = n_alpha - 1
    shuffle_decisions = np.zeros(
        (n_arms, cfg["shuffle_replicates"], n_views), dtype=np.int64
    )
    predictions = np.full((n_arms, 2, n_views, n_alpha - 1), np.nan)
    oof_predictions = np.full((n_arms, 2, validation.sum(), n_alpha - 1), np.nan)
    oof_actions = np.zeros((n_arms, 2, validation.sum()), dtype=np.int64)
    for arm_index, arm in enumerate(cfg["arms"]):
        validation_means = quotient_mean[arm_index][:, validation].mean(axis=1)
        constant = int(
            np.flatnonzero(validation_means <= validation_means.min() + 1e-12)[0]
        )
        decisions["constant_validation"][arm_index] = constant
        gates["arms"][arm] = {"constant_alpha_index": constant, "models": {}}
        for family_index, (family, columns) in enumerate(
            (
                (
                    "correction_scale_ridge",
                    [FEATURE_ORDER.index(x) for x in REDUCED_FEATURES],
                ),
                ("public_ridge_gate", list(range(len(FEATURE_ORDER)))),
            )
        ):
            report, predicted, oof = fit_gate(
                feature_mean[arm_index][validation][:, columns],
                y_delta[arm_index],
                feature_mean[arm_index][:, columns],
                validation_masters,
                cfg,
            )
            gates["arms"][arm]["models"][family] = report
            predictions[arm_index, family_index] = predicted
            oof_predictions[arm_index, family_index] = oof
            oof_actions[arm_index, family_index] = choose_alpha(oof)
            decisions[family][arm_index] = choose_alpha(predicted)
        gates["arms"][arm]["shuffled_target_ridge"] = []
        for replicate in range(cfg["shuffle_replicates"]):
            permutation = stratified_rotation(
                n_nodes_validation,
                stable_seed(cfg["shuffle_seed"], arm, replicate),
            )
            report, predicted, _ = fit_gate(
                feature_mean[arm_index][validation],
                y_delta[arm_index, permutation],
                feature_mean[arm_index],
                validation_masters,
                cfg,
            )
            shuffle_decisions[arm_index, replicate] = choose_alpha(predicted)
            gates["arms"][arm]["shuffled_target_ridge"].append(
                {
                    "replicate": replicate,
                    "permutation": permutation.tolist(),
                    "model": report,
                }
            )
        decisions["oracle_per_view"][arm_index] = np.argmin(
            quotient_mean[arm_index], axis=0
        )
    diagnostics: dict[str, Any] = {"arms": {}}
    for arm_index, arm in enumerate(cfg["arms"]):
        diagnostics["arms"][arm] = {}
        for family_index, family in enumerate(
            ("correction_scale_ridge", "public_ridge_gate")
        ):
            diagnostics["arms"][arm][family] = {}
            for label, mask in (
                ("validation_oof", validation),
                (
                    "test_iid",
                    np.asarray(
                        [
                            v.private.split == "test"
                            and v.private.corruption_mechanism == "iid"
                            for v in views
                        ]
                    ),
                ),
                (
                    "test_grouped",
                    np.asarray(
                        [
                            v.private.split == "test"
                            and v.private.corruption_mechanism == "grouped"
                            for v in views
                        ]
                    ),
                ),
            ):
                predicted = (
                    oof_predictions[arm_index, family_index]
                    if label == "validation_oof"
                    else predictions[arm_index, family_index, mask]
                )
                actual = (
                    y_delta[arm_index]
                    if label == "validation_oof"
                    else (
                        quotient_mean[arm_index][1:, mask]
                        - quotient_mean[arm_index][:1, mask]
                    ).T
                )
                diagnostics["arms"][arm][family][label] = {
                    "mse": float(np.mean((predicted - actual) ** 2)),
                    "correlations": [
                        correlation(predicted[:, i], actual[:, i])
                        for i in range(n_alpha - 1)
                    ],
                }
                if label == "validation_oof":
                    action = oof_actions[arm_index, family_index]
                    selected = quotient_mean[arm_index][:, validation][
                        action, np.arange(validation.sum())
                    ]
                    identity = quotient_mean[arm_index][0, validation]
                    diagnostics["arms"][arm][family][label].update(
                        {
                            "mean_rmse": float(selected.mean()),
                            "minus_identity": float(np.mean(selected - identity)),
                            "action_fractions": {
                                str(alpha): float(np.mean(action == alpha_index))
                                for alpha_index, alpha in enumerate(cfg["alphas"])
                            },
                        }
                    )
    with np.load(source / "bootstrap_indices.npz", allow_pickle=False) as saved:
        bootstrap_masters = saved["master_id"].copy()
        bootstrap_indices = saved["complete_n_252"].copy()
    effects = summarize(
        quotient,
        decisions,
        shuffle_decisions,
        views,
        bootstrap_masters,
        bootstrap_indices,
        cfg,
    )
    frozen.save_npz(
        output / "features.npz",
        {"feature_order": np.asarray(FEATURE_ORDER), "features": features},
    )
    frozen.save_npz(
        output / "alpha_metrics.npz",
        {
            "quotient_rmse": quotient,
            "relation_rmse": relation_rmse,
            "converged": converged,
            "iterations": iterations,
            "condition": conditions,
            "view_id": np.asarray(view_ids),
        },
    )
    frozen.save_npz(
        output / "decisions.npz",
        {
            **decisions,
            "shuffled_target_ridge": shuffle_decisions,
            "oof_validation_actions": oof_actions,
        },
    )
    frozen.save_npz(
        output / "bootstrap_indices.npz",
        {"master_id": bootstrap_masters, "complete_n_252": bootstrap_indices},
    )
    write_json(output / "resolved_config.json", cfg)
    write_json(output / "gate_models.json", gates)
    write_json(output / "prediction_diagnostics.json", diagnostics)
    write_json(output / "effects.json", effects)
    samples.append(resource_sample(started, "complete"))
    write_json(output / "resource_samples.json", samples)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    source_inputs = [
        source / "manifest.json",
        source / "bootstrap_indices.npz",
        *sorted((source / "raw_eval").glob("*.npz")),
    ]
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / path)}' \"$repo/{path}\" | sha256sum -c -"
        for path in SOURCE_FILES
    )
    checks += "\n" + "\n".join(
        f"printf '%s  %s\\n' '{sha(path)}' \"$repo/{path.relative_to(ROOT)}\" | sha256sum -c -"
        for path in source_inputs
    )
    replay = f"""#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || {{ echo "wrong git HEAD" >&2; exit 2; }}
[ -z "$(git -C "$repo" status --porcelain)" ] || {{ echo "dirty worktree" >&2; exit 2; }}
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR to a new path}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_residual_gate.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_residual_gate_v1.json" --output "$OUTPUT_DIR"
"""
    (output / "replay.sh").write_text(replay)
    (output / "replay.sh").chmod(0o755)
    deterministic = sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and path.name
        not in {"manifest.json", "runtime_observation.json", "resource_samples.json"}
    )
    manifest = {
        "schema_version": cfg["schema_version"],
        "git_head": head,
        "source_hashes": {path: sha(ROOT / path) for path in SOURCE_FILES},
        "input_hashes": {
            str(path.relative_to(ROOT)): sha(path) for path in source_inputs
        },
        "deterministic_files": {
            str(path.relative_to(output)): sha(path) for path in deterministic
        },
        "runtime_exclusions": ["resource_samples.json", "runtime_observation.json"],
    }
    write_json(output / "manifest.json", manifest)
    runtime = resource_sample(started, "complete")
    write_json(output / "runtime_observation.json", runtime)
    print(json.dumps(runtime, sort_keys=True))


if __name__ == "__main__":
    main()
