#!/usr/bin/env python3
"""Calibrate and evaluate a fresh mixed-mechanism residual gate, CPU-only."""

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
import run_proportional_graph_residual_gate as historical_gate  # noqa: E402
import torch  # noqa: E402
from geometria_proporcional.proportional_graph_contract import (  # noqa: E402
    ProportionalGraphConfig,
    generate_graph_views,
    score_solver,
    solve_huber_irls,
)

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_fresh_mixed_gate_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_FRESH_MIXED_GATE_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_fresh_mixed_gate_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_residual_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_irls_loss_contrast.py",
    "experiments/geometria_proporcional/run_proportional_graph_solver_interface_diagnostic.py",
    "experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py",
    "src/geometria_proporcional/proportional_graph_contract.py",
)
FEATURE_ORDER = historical_gate.FEATURE_ORDER
REDUCED_COLUMNS = tuple(FEATURE_ORDER.index(name) for name in historical_gate.REDUCED_FEATURES)
MECHANISMS = ("iid", "grouped")
POLICIES = (
    "identity",
    "always_post",
    "constant_balanced",
    "correction_scale_ridge",
    "public_mixed_ridge_gate",
    "historical_iid_ridge_gate",
    "shuffled_target_mixed_ridge",
    "oracle_per_view",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    expected = {
        "schema_version", "source_loss_contrast", "source_loss_manifest_sha256",
        "source_historical_gate", "source_historical_gate_manifest_sha256",
        "source_smoke_config", "arms", "seeds", "alphas", "realizations",
        "ridge_lambdas", "ridge_folds", "shuffle_replicates", "shuffle_seed",
        "bootstrap_replicates", "bootstrap_seed", "execution",
    }
    if set(cfg) != expected or cfg["schema_version"] != "proportional-graph-fresh-mixed-gate-v1":
        raise ValueError("invalid fresh mixed gate schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms must remain frozen")
    if cfg["seeds"] != [104729, 130363] or cfg["alphas"] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("neural seeds and alpha grid must remain frozen")
    if cfg["ridge_lambdas"] != [0.0, 0.01, 0.1, 1.0, 10.0, 100.0] or cfg["ridge_folds"] != 5:
        raise ValueError("ridge contract must remain frozen")
    realization = cfg["realizations"]
    if set(realization) != {"calibration_seed", "adjudication_seed", "min_eligible_masters"}:
        raise ValueError("invalid realization contract")
    if realization["calibration_seed"] == realization["adjudication_seed"]:
        raise ValueError("fresh realizations must use different seeds")
    if realization["min_eligible_masters"] < 2:
        raise ValueError("minimum eligible master count is too small")
    execution = cfg["execution"]
    if set(execution) != {"torch_threads", "max_seconds_per_phase", "max_rss_gib", "sample_every_views"}:
        raise ValueError("invalid execution contract")
    if execution["torch_threads"] != 1:
        raise ValueError("PyTorch must remain single-threaded")
    return cfg


def resource_sample(started: float, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }


def enforce(sample: dict[str, Any], cfg: dict[str, Any]) -> None:
    execution = cfg["execution"]
    if sample["elapsed_seconds"] > execution["max_seconds_per_phase"]:
        raise RuntimeError("phase runtime budget exceeded")
    if sample["max_rss_gib"] > execution["max_rss_gib"]:
        raise RuntimeError("phase RSS budget exceeded")


def require_clean(development: bool) -> None:
    if development:
        return
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
        text=True, check=True,
    ).stdout
    if dirty:
        raise RuntimeError("official phase requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True,
    ).stdout.strip()


def source_inputs(cfg: dict[str, Any]) -> list[Path]:
    loss = ROOT / cfg["source_loss_contrast"]
    old_gate = ROOT / cfg["source_historical_gate"]
    return [
        loss / "manifest.json",
        loss / "resolved_config.json",
        *[
            loss / "checkpoints" / f"{arm}|seed={seed}|post_irls.npz"
            for arm in cfg["arms"] for seed in cfg["seeds"]
        ],
        old_gate / "manifest.json",
        old_gate / "gate_models.json",
    ]


def verify_sources(cfg: dict[str, Any]) -> None:
    loss = ROOT / cfg["source_loss_contrast"]
    old_gate = ROOT / cfg["source_historical_gate"]
    if sha(loss / "manifest.json") != cfg["source_loss_manifest_sha256"]:
        raise AssertionError("loss contrast manifest hash mismatch")
    if sha(old_gate / "manifest.json") != cfg["source_historical_gate_manifest_sha256"]:
        raise AssertionError("historical gate manifest hash mismatch")
    loss_manifest = json.loads((loss / "manifest.json").read_text())
    gate_manifest = json.loads((old_gate / "manifest.json").read_text())
    for path in source_inputs(cfg):
        if path.name == "manifest.json":
            continue
        owner = loss if loss in path.parents else old_gate
        relative = str(path.relative_to(owner))
        manifest = loss_manifest if owner == loss else gate_manifest
        if manifest["deterministic_files"].get(relative) != sha(path):
            raise AssertionError(f"source file hash mismatch: {relative}")


def source_context(cfg: dict[str, Any]) -> tuple[dict[str, Any], float, dict[str, dict[int, torch.nn.Module]]]:
    verify_sources(cfg)
    source_cfg = smoke._load_config((ROOT / cfg["source_smoke_config"]).resolve())
    original_views = generate_graph_views(ProportionalGraphConfig.from_dict(source_cfg["graph"]))
    shuffle_arm = next(a for a in source_cfg["arms"] if a["name"] == "closure_typed_path_shuffle")
    eligible = {
        view.private.master_id: bool(
            smoke._view_tensors(view, shuffle_arm, cfg["seeds"][0])["path_shuffle_eligible"]
        )
        for view in original_views
    }
    train = [
        view for view in original_views
        if view.private.split == "train" and eligible[view.private.master_id]
    ]
    scale = smoke._input_scale(train)
    loss = ROOT / cfg["source_loss_contrast"]
    loss_cfg = json.loads((loss / "resolved_config.json").read_text())
    models: dict[str, dict[int, torch.nn.Module]] = {}
    for arm_name in cfg["arms"]:
        arm = next(a for a in source_cfg["arms"] if a["name"] == arm_name)
        models[arm_name] = {}
        for seed in cfg["seeds"]:
            model = smoke._model_for_arm(arm, source_cfg, scale)
            path = loss / "checkpoints" / f"{arm_name}|seed={seed}|post_irls.npz"
            with np.load(path, allow_pickle=False) as saved:
                if saved["format_version"].item() != "proportional-frozen-adapter-v1":
                    raise ValueError("unsupported post-IRLS adapter format")
                metadata = json.loads(saved["metadata_json"].item())
                if metadata != {
                    "arm": arm_name, "last_epoch": loss_cfg["training"]["epochs"],
                    "loss_multiplier": metadata["loss_multiplier"],
                    "mode": "post_irls", "seed": seed,
                }:
                    raise AssertionError("post-IRLS adapter metadata mismatch")
                state = {
                    key.removeprefix("model::"): torch.from_numpy(saved[key].copy())
                    for key in saved.files if key.startswith("model::")
                }
            model.load_state_dict(state)
            expected_scale = float(np.float32(scale))
            if float(model.input_scale.detach().cpu()) != expected_scale:
                raise AssertionError("adapter input scale differs from regenerated source train scale")
            model.eval()
            models[arm_name][seed] = model
    return source_cfg, scale, models


def fresh_views(
    cfg: dict[str, Any], source_cfg: dict[str, Any], seed: int
) -> list[Any]:
    graph = dict(source_cfg["graph"])
    graph["seed"] = int(seed)
    generated = generate_graph_views(ProportionalGraphConfig.from_dict(graph))
    shuffle_arm = next(a for a in source_cfg["arms"] if a["name"] == "closure_typed_path_shuffle")
    test = [view for view in generated if view.private.split == "test"]
    eligible = {
        view.private.master_id: bool(
            smoke._view_tensors(view, shuffle_arm, cfg["seeds"][0])["path_shuffle_eligible"]
        )
        for view in test if view.private.corruption_mechanism == "iid"
    }
    views = [view for view in test if eligible[view.private.master_id]]
    mechanism_order = {name: index for index, name in enumerate(MECHANISMS)}
    views.sort(key=lambda view: (view.private.master_id, mechanism_order[view.private.corruption_mechanism]))
    by_master: dict[str, list[Any]] = {}
    for view in views:
        by_master.setdefault(view.private.master_id, []).append(view)
    if len(by_master) < cfg["realizations"]["min_eligible_masters"]:
        raise RuntimeError("too few eligible masters in fresh realization")
    for pair in by_master.values():
        if [view.private.corruption_mechanism for view in pair] != list(MECHANISMS):
            raise AssertionError("fresh master is not an ordered IID/grouped pair")
        if not np.array_equal(pair[0].public.edge_index, pair[1].public.edge_index):
            raise AssertionError("paired topology mismatch")
    return views


def view_index(views: list[Any], realization_seed: int) -> list[dict[str, Any]]:
    result = []
    for view in views:
        public = view.public
        digest = hashlib.sha256()
        for array in (
            public.edge_index, public.observed_log_ratio, public.edge_valid,
            public.path_index, public.path_sign, public.path_valid, public.edge_variance,
        ):
            value = np.ascontiguousarray(array)
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes())
        result.append({
            "view_id": view.private.view_id,
            "master_id": view.private.master_id,
            "mechanism": view.private.corruption_mechanism,
            "n_nodes": public.n_nodes,
            "n_edges": len(public.edge_index),
            "realization_seed": realization_seed,
            "public_sha256": digest.hexdigest(),
        })
    return result


def run_universe(
    role: str,
    views: list[Any],
    models: dict[str, dict[int, torch.nn.Module]],
    source_cfg: dict[str, Any],
    cfg: dict[str, Any],
    output: Path,
    started: float,
    samples: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    n_arms, n_seeds, n_alpha, n_views = (
        len(cfg["arms"]), len(cfg["seeds"]), len(cfg["alphas"]), len(views)
    )
    features = np.full((n_arms, n_seeds, n_views, len(FEATURE_ORDER)), np.nan)
    quotient = np.full((n_arms, n_seeds, n_alpha, n_views), np.nan)
    relation_rmse = np.full_like(quotient, np.nan)
    converged = np.zeros_like(quotient, dtype=bool)
    iterations = np.zeros_like(quotient, dtype=np.int64)
    condition = np.full_like(quotient, np.nan)
    view_ids = np.asarray([view.private.view_id for view in views])
    for arm_index, arm_name in enumerate(cfg["arms"]):
        arm = next(a for a in source_cfg["arms"] if a["name"] == arm_name)
        for seed_index, seed in enumerate(cfg["seeds"]):
            model = models[arm_name][seed]
            corrected_rows, edge_offsets = [], [0]
            x_states = [[] for _ in cfg["alphas"]]
            weight_states = [[] for _ in cfg["alphas"]]
            node_offsets = [0]
            with torch.no_grad():
                for view_number, view in enumerate(views, start=1):
                    corrected = model(smoke._view_tensors(view, arm, seed)).corrected_log_ratio.cpu().numpy().astype(np.float64)
                    observed = np.asarray(view.public.observed_log_ratio, dtype=np.float64)
                    corrected_rows.append(corrected)
                    edge_offsets.append(edge_offsets[-1] + len(observed))
                    node_offsets.append(node_offsets[-1] + view.public.n_nodes)
                    features[arm_index, seed_index, view_number - 1] = historical_gate.public_features(view, corrected)
                    for alpha_index, alpha in enumerate(cfg["alphas"]):
                        relation = observed + alpha * (corrected - observed)
                        solved = solve_huber_irls(
                            view.public, values=relation, base_weights=None,
                            delta=source_cfg["graph"]["huber_delta"],
                            max_iterations=source_cfg["graph"]["irls_iterations"],
                            damping=source_cfg["graph"]["irls_damping"],
                            weight_floor=source_cfg["graph"]["weight_floor"],
                        )
                        scored = score_solver(solved, view.private)
                        converged[arm_index, seed_index, alpha_index, view_number - 1] = solved.converged
                        iterations[arm_index, seed_index, alpha_index, view_number - 1] = solved.iterations
                        condition[arm_index, seed_index, alpha_index, view_number - 1] = solved.laplacian_condition
                        relation_rmse[arm_index, seed_index, alpha_index, view_number - 1] = float(
                            np.sqrt(np.mean((relation - view.private.clean_log_ratio) ** 2))
                        )
                        if solved.converged:
                            quotient[arm_index, seed_index, alpha_index, view_number - 1] = scored.quotient_rmse
                        x_states[alpha_index].append(solved.x_hat)
                        weight_states[alpha_index].append(solved.weights)
                    if view_number % cfg["execution"]["sample_every_views"] == 0:
                        sample = resource_sample(started, f"{role}|{arm_name}|{seed}|view={view_number}")
                        samples.append(sample)
                        enforce(sample, cfg)
            frozen.save_npz(output / role / "forward" / f"{arm_name}|seed={seed}.npz", {
                "view_id": view_ids,
                "edge_offset": np.asarray(edge_offsets, dtype=np.int64),
                "post_irls_corrected_relation": np.concatenate(corrected_rows).astype(np.float32),
            })
            frozen.save_npz(output / role / "solver" / f"{arm_name}|seed={seed}.npz", {
                "view_id": view_ids,
                "node_offset": np.asarray(node_offsets, dtype=np.int64),
                "edge_offset": np.asarray(edge_offsets, dtype=np.int64),
                "x_hat": np.stack([np.concatenate(rows) for rows in x_states]),
                "final_weights": np.stack([np.concatenate(rows) for rows in weight_states]),
                "converged": converged[arm_index, seed_index],
                "iterations": iterations[arm_index, seed_index],
                "condition": condition[arm_index, seed_index],
            })
    if not np.all(converged) or not np.all(np.isfinite(quotient)):
        raise RuntimeError(f"strict IRLS convergence failed in {role}")
    frozen.save_npz(output / role / "features.npz", {
        "feature_order": np.asarray(FEATURE_ORDER), "features": features,
        "view_id": view_ids,
    })
    frozen.save_npz(output / role / "alpha_metrics.npz", {
        "view_id": view_ids, "quotient_rmse": quotient,
        "relation_rmse": relation_rmse, "converged": converged,
        "iterations": iterations, "condition": condition,
    })
    return {
        "features": features, "quotient_rmse": quotient,
        "relation_rmse": relation_rmse, "converged": converged,
        "iterations": iterations, "condition": condition,
    }


def predict_report(report: dict[str, Any], x: np.ndarray) -> np.ndarray:
    columns = []
    for saved in report["models"]:
        model = {
            "intercept": float(saved["intercept"]),
            "mean": np.asarray(saved["mean"], dtype=np.float64),
            "scale": np.asarray(saved["scale"], dtype=np.float64),
            "coefficients": np.asarray(saved["coefficients"], dtype=np.float64),
        }
        columns.append(historical_gate.interface._ridge_predict(model, x))
    return np.column_stack(columns)


def paired_master_permutation(views: list[Any], seed: int) -> np.ndarray:
    if len(views) % 2:
        raise ValueError("paired shuffle requires an even number of views")
    result = np.arange(len(views))
    rng = np.random.default_rng(seed)
    masters = [views[index].private.master_id for index in range(0, len(views), 2)]
    n_nodes = np.asarray([views[index].public.n_nodes for index in range(0, len(views), 2)])
    for value in sorted(set(n_nodes.tolist())):
        recipients = np.flatnonzero(n_nodes == value)
        if len(recipients) < 2:
            raise ValueError("master shuffle stratum cannot be deranged")
        shift = int(rng.integers(1, len(recipients)))
        donors = np.roll(recipients, shift)
        for recipient, donor in zip(recipients, donors, strict=True):
            result[2 * recipient : 2 * recipient + 2] = [2 * donor, 2 * donor + 1]
    if any(masters[index // 2] == masters[result[index] // 2] for index in range(len(result))):
        raise AssertionError("paired target shuffle retained a master identity")
    return result


def calibration_fit(
    data: dict[str, np.ndarray], views: list[Any], cfg: dict[str, Any], historical: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    feature_mean = data["features"].mean(axis=1)
    quotient_mean = data["quotient_rmse"].mean(axis=1)
    targets = np.transpose(quotient_mean[:, 1:, :] - quotient_mean[:, :1, :], (0, 2, 1))
    masters = np.asarray([view.private.master_id for view in views])
    models: dict[str, Any] = {"feature_order": list(FEATURE_ORDER), "arms": {}}
    decisions = {
        name: np.zeros((len(cfg["arms"]), len(views)), dtype=np.int64)
        for name in POLICIES if name not in {"shuffled_target_mixed_ridge", "oracle_per_view"}
    }
    decisions["always_post"][:] = len(cfg["alphas"]) - 1
    shuffle_actions = np.zeros((len(cfg["arms"]), cfg["shuffle_replicates"], len(views)), dtype=np.int64)
    oof_actions = np.zeros((len(cfg["arms"]), 2, len(views)), dtype=np.int64)
    oof_predictions = np.full((len(cfg["arms"]), 2, len(views), len(cfg["alphas"]) - 1), np.nan)
    for arm_index, arm in enumerate(cfg["arms"]):
        means = quotient_mean[arm_index].mean(axis=1)
        constant = int(np.flatnonzero(means <= means.min() + 1e-12)[0])
        decisions["constant_balanced"][arm_index] = constant
        arm_record: dict[str, Any] = {"constant_alpha_index": constant, "models": {}}
        for family_index, (name, columns) in enumerate((
            ("correction_scale_ridge", list(REDUCED_COLUMNS)),
            ("public_mixed_ridge_gate", list(range(len(FEATURE_ORDER)))),
        )):
            report, predicted, oof = historical_gate.fit_gate(
                feature_mean[arm_index][:, columns], targets[arm_index],
                feature_mean[arm_index][:, columns], masters, cfg,
            )
            arm_record["models"][name] = report
            decisions[name][arm_index] = historical_gate.choose_alpha(predicted)
            oof_predictions[arm_index, family_index] = oof
            oof_actions[arm_index, family_index] = historical_gate.choose_alpha(oof)
        old_report = historical["arms"][arm]["models"]["public_ridge_gate"]
        if historical["feature_order"] != list(FEATURE_ORDER):
            raise AssertionError("historical feature order mismatch")
        arm_record["models"]["historical_iid_ridge_gate"] = old_report
        decisions["historical_iid_ridge_gate"][arm_index] = historical_gate.choose_alpha(
            predict_report(old_report, feature_mean[arm_index])
        )
        arm_record["shuffled_target_mixed_ridge"] = []
        for replicate in range(cfg["shuffle_replicates"]):
            permutation = paired_master_permutation(
                views, stable_seed(cfg["shuffle_seed"], arm, replicate)
            )
            report, predicted, _ = historical_gate.fit_gate(
                feature_mean[arm_index], targets[arm_index, permutation],
                feature_mean[arm_index], masters, cfg,
            )
            shuffle_actions[arm_index, replicate] = historical_gate.choose_alpha(predicted)
            arm_record["shuffled_target_mixed_ridge"].append({
                "replicate": replicate, "permutation": permutation.tolist(), "model": report,
            })
        models["arms"][arm] = arm_record
    return models, {
        **decisions,
        "shuffled_target_mixed_ridge": shuffle_actions,
        "oof_actions": oof_actions,
        "oof_predictions": oof_predictions,
        "target_delta": targets,
    }


def calibration_diagnostics(
    arrays: dict[str, np.ndarray], data: dict[str, np.ndarray], cfg: dict[str, Any]
) -> dict[str, Any]:
    quotient_mean = data["quotient_rmse"].mean(axis=1)
    targets = arrays["target_delta"]
    result: dict[str, Any] = {"arms": {}}
    for arm_index, arm in enumerate(cfg["arms"]):
        result["arms"][arm] = {}
        for family_index, family in enumerate(("correction_scale_ridge", "public_mixed_ridge_gate")):
            predicted = arrays["oof_predictions"][arm_index, family_index]
            action = arrays["oof_actions"][arm_index, family_index]
            selected = quotient_mean[arm_index, action, np.arange(len(action))]
            identity = quotient_mean[arm_index, 0]
            result["arms"][arm][family] = {
                "status": "TUNING_DIAGNOSTIC_NOT_INDEPENDENT",
                "mse": float(np.mean((predicted - targets[arm_index]) ** 2)),
                "correlations": [
                    historical_gate.correlation(predicted[:, index], targets[arm_index, :, index])
                    for index in range(predicted.shape[1])
                ],
                "mean_rmse": float(selected.mean()),
                "minus_identity": float(np.mean(selected - identity)),
                "action_fractions": {
                    str(alpha): float(np.mean(action == index))
                    for index, alpha in enumerate(cfg["alphas"])
                },
            }
    return result


def calibration_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if output.exists():
        raise FileExistsError(output)
    if not development and (ROOT / "data/geometria_proporcional").resolve() not in output.resolve().parents:
        raise ValueError("official output must be below data/geometria_proporcional")
    output.mkdir(parents=True)
    started = time.monotonic()
    samples = [resource_sample(started, "calibrate_start")]
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    source_cfg, scale, models = source_context(cfg)
    seed = cfg["realizations"]["calibration_seed"]
    views = fresh_views(cfg, source_cfg, seed)
    write_json(output / "calibration/view_index.json", view_index(views, seed))
    data = run_universe("calibration", views, models, source_cfg, cfg, output, started, samples)
    historical = json.loads((ROOT / cfg["source_historical_gate"] / "gate_models.json").read_text())
    gate_models, arrays = calibration_fit(data, views, cfg, historical)
    write_json(output / "calibration/gate_models.json", gate_models)
    write_json(output / "calibration/diagnostics.json", calibration_diagnostics(arrays, data, cfg))
    frozen.save_npz(output / "calibration/gate_fit.npz", {
        key: value for key, value in arrays.items() if isinstance(value, np.ndarray)
    })
    write_json(output / "resolved_config.json", cfg)
    deterministic_calibration = sorted(
        path for path in (output / "calibration").rglob("*") if path.is_file()
    )
    calibration_manifest = {
        "schema_version": "fresh-mixed-gate-calibration-manifest-v1",
        "git_head": git_head(),
        "source_hashes": {path: sha(ROOT / path) for path in SOURCE_FILES},
        "input_hashes": {str(path.relative_to(ROOT)): sha(path) for path in source_inputs(cfg)},
        "input_scale": scale,
        "calibration_seed": seed,
        "calibration_files": {
            str(path.relative_to(output)): sha(path) for path in deterministic_calibration
        },
    }
    write_json(output / "calibration_manifest.json", calibration_manifest)
    freeze = {
        "schema_version": "fresh-mixed-gate-policy-freeze-v1",
        "git_head": calibration_manifest["git_head"],
        "config_sha256": sha(output / "resolved_config.json"),
        "calibration_manifest_sha256": sha(output / "calibration_manifest.json"),
        "gate_models_sha256": sha(output / "calibration/gate_models.json"),
        "future_materialized": False,
        "allowed_evaluation_operation": "apply frozen policies to adjudication realization",
    }
    write_json(output / "policy_freeze.json", freeze)
    write_json(output / "phase_receipt.json", {
        "phase": "calibrate", "policy_freeze_sha256": sha(output / "policy_freeze.json"),
        "calibration_manifest_sha256": sha(output / "calibration_manifest.json"),
        "adjudication_seed_materialized": False,
    })
    samples.append(resource_sample(started, "calibrate_complete"))
    enforce(samples[-1], cfg)
    write_json(output / "resource_samples_calibrate.json", samples)
    print(json.dumps(samples[-1], sort_keys=True))


def verify_freeze(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    receipt = json.loads((output / "phase_receipt.json").read_text())
    if receipt["phase"] != "calibrate" or receipt["adjudication_seed_materialized"]:
        raise AssertionError("invalid calibration phase receipt")
    if receipt["policy_freeze_sha256"] != sha(output / "policy_freeze.json"):
        raise AssertionError("policy freeze hash mismatch")
    if receipt["calibration_manifest_sha256"] != sha(output / "calibration_manifest.json"):
        raise AssertionError("calibration manifest receipt mismatch")
    freeze = json.loads((output / "policy_freeze.json").read_text())
    manifest = json.loads((output / "calibration_manifest.json").read_text())
    if freeze["git_head"] != git_head() or manifest["git_head"] != git_head():
        raise AssertionError("git HEAD changed after policy freeze")
    if freeze["config_sha256"] != sha(output / "resolved_config.json"):
        raise AssertionError("resolved config changed after freeze")
    if json.loads((output / "resolved_config.json").read_text()) != cfg:
        raise AssertionError("requested config differs from frozen config")
    if freeze["calibration_manifest_sha256"] != sha(output / "calibration_manifest.json"):
        raise AssertionError("freeze does not bind calibration manifest")
    if freeze["gate_models_sha256"] != sha(output / "calibration/gate_models.json"):
        raise AssertionError("gate models changed after freeze")
    for relative, expected in manifest["calibration_files"].items():
        if sha(output / relative) != expected:
            raise AssertionError(f"calibration artifact changed: {relative}")
    return freeze


def decisions_from_freeze(
    gate_models: dict[str, Any], features: np.ndarray, quotient: np.ndarray,
    cfg: dict[str, Any]
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    feature_mean = features.mean(axis=1)
    quotient_mean = quotient.mean(axis=1)
    n_arms, _, n_views = quotient_mean.shape
    decisions = {
        name: np.zeros((n_arms, n_views), dtype=np.int64)
        for name in POLICIES if name != "shuffled_target_mixed_ridge"
    }
    decisions["always_post"][:] = len(cfg["alphas"]) - 1
    shuffle = np.zeros((n_arms, cfg["shuffle_replicates"], n_views), dtype=np.int64)
    predictions: dict[str, np.ndarray] = {
        "correction_scale_ridge": np.full((n_arms, n_views, len(cfg["alphas"]) - 1), np.nan),
        "public_mixed_ridge_gate": np.full((n_arms, n_views, len(cfg["alphas"]) - 1), np.nan),
        "historical_iid_ridge_gate": np.full((n_arms, n_views, len(cfg["alphas"]) - 1), np.nan),
    }
    for arm_index, arm in enumerate(cfg["arms"]):
        record = gate_models["arms"][arm]
        decisions["constant_balanced"][arm_index] = record["constant_alpha_index"]
        for family, columns in (
            ("correction_scale_ridge", list(REDUCED_COLUMNS)),
            ("public_mixed_ridge_gate", list(range(len(FEATURE_ORDER)))),
            ("historical_iid_ridge_gate", list(range(len(FEATURE_ORDER)))),
        ):
            predicted = predict_report(record["models"][family], feature_mean[arm_index][:, columns])
            predictions[family][arm_index] = predicted
            decisions[family][arm_index] = historical_gate.choose_alpha(predicted)
        for replicate, shuffled in enumerate(record["shuffled_target_mixed_ridge"]):
            predicted = predict_report(shuffled["model"], feature_mean[arm_index])
            shuffle[arm_index, replicate] = historical_gate.choose_alpha(predicted)
        decisions["oracle_per_view"][arm_index] = np.argmin(quotient_mean[arm_index], axis=0)
    return decisions, shuffle, predictions


def selected_values(
    quotient: np.ndarray, action: np.ndarray
) -> np.ndarray:
    return np.take_along_axis(
        quotient, action[None, None, :].repeat(quotient.shape[0], axis=0), axis=1
    )[:, 0, :]


def summarize_effects(
    quotient: np.ndarray, decisions: dict[str, np.ndarray], shuffle: np.ndarray,
    views: list[Any], bootstrap: np.ndarray, cfg: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {"arms": {}}
    master_order = [views[index].private.master_id for index in range(0, len(views), 2)]
    for arm_index, arm in enumerate(cfg["arms"]):
        policy_values = {
            policy: selected_values(quotient[arm_index], action[arm_index])
            for policy, action in decisions.items()
        }
        shuffled_values = [
            selected_values(quotient[arm_index], shuffle[arm_index, replicate])
            for replicate in range(cfg["shuffle_replicates"])
        ]
        policy_values["shuffled_target_mixed_ridge"] = np.mean(shuffled_values, axis=0)
        arm_result: dict[str, Any] = {}
        cell_indices = {
            "iid": np.arange(0, len(views), 2),
            "grouped": np.arange(1, len(views), 2),
        }
        for cell in ("iid", "grouped", "balanced"):
            arm_result[cell] = {}
            per_policy: dict[str, np.ndarray] = {}
            for policy in POLICIES:
                values = policy_values[policy]
                if cell == "balanced":
                    master_values = 0.5 * (values[:, cell_indices["iid"]] + values[:, cell_indices["grouped"]])
                else:
                    master_values = values[:, cell_indices[cell]]
                per_policy[policy] = master_values.mean(axis=0)
            for policy in POLICIES:
                values = per_policy[policy]
                draws = values[bootstrap].mean(axis=1)
                payload: dict[str, Any] = {
                    "n_masters": len(master_order), "mean_rmse": float(values.mean()),
                    "ci95_mean": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                }
                for baseline in (
                    "identity", "always_post", "constant_balanced",
                    "correction_scale_ridge", "historical_iid_ridge_gate",
                    "shuffled_target_mixed_ridge",
                ):
                    if policy == baseline:
                        continue
                    delta = values - per_policy[baseline]
                    delta_draws = delta[bootstrap].mean(axis=1)
                    payload[f"minus_{baseline}"] = {
                        "mean": float(delta.mean()),
                        "ci95": [float(np.percentile(delta_draws, 2.5)), float(np.percentile(delta_draws, 97.5))],
                    }
                arm_result[cell][policy] = payload
        public_effect_iid = (
            np.mean(policy_values["public_mixed_ridge_gate"][:, cell_indices["iid"]], axis=0)
            - np.mean(policy_values["identity"][:, cell_indices["iid"]], axis=0)
        )
        public_effect_grouped = (
            np.mean(policy_values["public_mixed_ridge_gate"][:, cell_indices["grouped"]], axis=0)
            - np.mean(policy_values["identity"][:, cell_indices["grouped"]], axis=0)
        )
        interaction = public_effect_grouped - public_effect_iid
        interaction_draws = interaction[bootstrap].mean(axis=1)
        arm_result["grouped_minus_iid_public_effect"] = {
            "mean": float(interaction.mean()),
            "ci95": [float(np.percentile(interaction_draws, 2.5)), float(np.percentile(interaction_draws, 97.5))],
        }
        arm_result["action_fractions"] = {}
        for cell, indices in cell_indices.items():
            arm_result["action_fractions"][cell] = {
                policy: {
                    str(alpha): float(np.mean(action[arm_index, indices] == alpha_index))
                    for alpha_index, alpha in enumerate(cfg["alphas"])
                }
                for policy, action in decisions.items()
            }
            arm_result["action_fractions"][cell]["shuffled_target_mixed_ridge"] = {
                str(alpha): float(np.mean(shuffle[arm_index, :, indices] == alpha_index))
                for alpha_index, alpha in enumerate(cfg["alphas"])
            }
        result["arms"][arm] = arm_result
    return result


def prediction_diagnostics(
    predictions: dict[str, np.ndarray], quotient: np.ndarray, cfg: dict[str, Any]
) -> dict[str, Any]:
    actual = np.transpose(quotient.mean(axis=1)[:, 1:, :] - quotient.mean(axis=1)[:, :1, :], (0, 2, 1))
    result: dict[str, Any] = {"arms": {}}
    for arm_index, arm in enumerate(cfg["arms"]):
        result["arms"][arm] = {}
        for family, predicted_all in predictions.items():
            result["arms"][arm][family] = {}
            for mechanism, indices in (("iid", np.arange(0, actual.shape[1], 2)), ("grouped", np.arange(1, actual.shape[1], 2))):
                predicted = predicted_all[arm_index, indices]
                target = actual[arm_index, indices]
                result["arms"][arm][family][mechanism] = {
                    "mse": float(np.mean((predicted - target) ** 2)),
                    "correlations": [
                        historical_gate.correlation(predicted[:, index], target[:, index])
                        for index in range(predicted.shape[1])
                    ],
                }
    return result


def write_replay(output: Path, cfg: dict[str, Any], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / path)}' \"$repo/{path}\" | sha256sum -c -"
        for path in SOURCE_FILES
    )
    checks += "\n" + "\n".join(
        f"printf '%s  %s\\n' '{sha(path)}' \"$repo/{path.relative_to(ROOT)}\" | sha256sum -c -"
        for path in source_inputs(cfg)
    )
    replay = f'''#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || {{ echo "wrong git HEAD" >&2; exit 2; }}
[ -z "$(git -C "$repo" status --porcelain)" ] || {{ echo "dirty worktree" >&2; exit 2; }}
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR to a new path}}"
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py" --phase calibrate --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_fresh_mixed_gate_v1.json" --output "$OUTPUT_DIR"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_fresh_mixed_gate.py" --phase evaluate --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_fresh_mixed_gate_v1.json" --output "$OUTPUT_DIR"
'''
    (output / "replay.sh").write_text(replay)
    (output / "replay.sh").chmod(0o755)


def evaluation_phase(cfg: dict[str, Any], output: Path, development: bool) -> None:
    require_clean(development)
    if not output.is_dir() or (output / "adjudication").exists():
        raise RuntimeError("evaluation requires a calibrated, unevaluated output")
    verify_freeze(cfg, output)
    started = time.monotonic()
    samples = [resource_sample(started, "evaluate_start")]
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    source_cfg, _, models = source_context(cfg)
    seed = cfg["realizations"]["adjudication_seed"]
    views = fresh_views(cfg, source_cfg, seed)
    calibration_index = json.loads((output / "calibration/view_index.json").read_text())
    calibration_masters = {row["master_id"] for row in calibration_index}
    if calibration_masters & {view.private.master_id for view in views}:
        raise AssertionError("calibration and adjudication masters overlap")
    write_json(output / "adjudication/view_index.json", view_index(views, seed))
    data = run_universe("adjudication", views, models, source_cfg, cfg, output, started, samples)
    gate_models = json.loads((output / "calibration/gate_models.json").read_text())
    decisions, shuffle, predictions = decisions_from_freeze(
        gate_models, data["features"], data["quotient_rmse"], cfg
    )
    frozen.save_npz(output / "adjudication/decisions.npz", {
        **decisions, "shuffled_target_mixed_ridge": shuffle,
    })
    frozen.save_npz(output / "adjudication/predictions.npz", predictions)
    n_masters = len(views) // 2
    bootstrap = np.random.default_rng(cfg["bootstrap_seed"]).integers(
        0, n_masters, size=(cfg["bootstrap_replicates"], n_masters)
    )
    frozen.save_npz(output / "bootstrap_indices.npz", {
        "master_id": np.asarray([views[index].private.master_id for index in range(0, len(views), 2)]),
        f"complete_n_{n_masters}": bootstrap,
    })
    write_json(output / "effects.json", summarize_effects(
        data["quotient_rmse"], decisions, shuffle, views, bootstrap, cfg
    ))
    write_json(output / "prediction_diagnostics.json", prediction_diagnostics(
        predictions, data["quotient_rmse"], cfg
    ))
    samples.append(resource_sample(started, "evaluate_complete"))
    enforce(samples[-1], cfg)
    write_json(output / "resource_samples_evaluate.json", samples)
    write_replay(output, cfg, git_head())
    deterministic = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name not in {
            "manifest.json", "resource_samples_calibrate.json",
            "resource_samples_evaluate.json", "runtime_observation.json",
        }
    )
    manifest = {
        "schema_version": cfg["schema_version"], "git_head": git_head(),
        "source_hashes": {path: sha(ROOT / path) for path in SOURCE_FILES},
        "input_hashes": {str(path.relative_to(ROOT)): sha(path) for path in source_inputs(cfg)},
        "deterministic_files": {str(path.relative_to(output)): sha(path) for path in deterministic},
        "runtime_exclusions": [
            "resource_samples_calibrate.json", "resource_samples_evaluate.json",
            "runtime_observation.json",
        ],
    }
    write_json(output / "manifest.json", manifest)
    runtime = resource_sample(started, "evaluate_complete")
    write_json(output / "runtime_observation.json", runtime)
    print(json.dumps(runtime, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = args.output.resolve()
    if args.phase == "calibrate":
        calibration_phase(cfg, output, args.development)
    else:
        evaluation_phase(cfg, output, args.development)


if __name__ == "__main__":
    main()
