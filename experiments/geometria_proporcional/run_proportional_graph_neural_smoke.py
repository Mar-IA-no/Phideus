#!/usr/bin/env python3
"""Run the CPU neural factorial for local proportional graph coherence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss
from torch.utils.flop_counter import FlopCounterMode

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometria_proporcional.proportional_graph_contract import (  # noqa: E402
    GraphView,
    ProportionalGraphConfig,
    public_schema_hash,
    score_solver,
    solve_huber_irls,
    solve_weighted_least_squares,
)
from geometria_proporcional.proportional_graph_contract import (
    generate_graph_views,
)  # noqa: E402
from geometria_proporcional.proportional_graph_neural import (  # noqa: E402
    EdgewiseMLP,
    GenericMessagePassing,
    ProportionalPathMixer,
    direct_centered_decoder,
    differentiable_wls,
    exact_closure_only,
    local_closure_loss,
    materialize_closure_evidence,
    observation_tensors,
    parameter_count,
    shuffled_path_tensors,
)

DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/geometria_proporcional/configs/proportional_graph_neural_smoke_v1.json"
)
SOURCE_PATHS = (
    "experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py",
    "experiments/geometria_proporcional/configs/proportional_graph_neural_smoke_v1.json",
    "src/geometria_proporcional/proportional_graph_contract.py",
    "src/geometria_proporcional/proportional_graph_neural.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    return parser.parse_args()


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_records(config_path: Path, development: bool) -> dict[str, dict[str, Any]]:
    expected_config = DEFAULT_CONFIG.resolve()
    if not development and config_path.resolve() != expected_config:
        raise ValueError("official run requires the canonical tracked config")
    records: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        tracked = (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", relative],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )
        dirty = (
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", relative], cwd=REPO_ROOT
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
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "graph",
        "training",
        "arms",
        "loss",
        "artifact_schema_version",
        "analysis_universe",
        "inference",
    }
    if set(data) != required:
        raise ValueError(f"config keys must be exactly {sorted(required)}")
    if data["schema_version"] != "proportional-graph-neural-smoke-v1":
        raise ValueError("unexpected neural smoke schema")
    if data["analysis_universe"] != {"require_path_shuffle_eligible": True}:
        raise ValueError(
            "official analysis universe must require a feasible path shuffle"
        )
    graph = ProportionalGraphConfig.from_dict(data["graph"])
    train = data["training"]
    if train["epochs"] < 1 or train["batch_size"] < 1 or len(train["seeds"]) < 1:
        raise ValueError("invalid training schedule")
    if len(train["seeds"]) != len(set(train["seeds"])):
        raise ValueError("training seeds must be unique")
    if train["device"] != "cpu":
        raise ValueError("this smoke is CPU-only")
    if train["max_seconds"] <= 0 or train["max_rss_gib"] <= 0:
        raise ValueError("resource budgets must be positive")
    if train["latency_repeats"] < 1:
        raise ValueError("latency_repeats must be positive")
    names = [arm["name"] for arm in data["arms"]]
    if len(names) != len(set(names)):
        raise ValueError("arm names must be unique")
    required_names = {
        "raw_generic",
        "raw_typed",
        "closure_generic",
        "closure_typed",
        "closure_typed_path_shuffle",
        "pair_state_no_mix",
        "generic_message_passing",
        "edge_mlp",
    }
    if set(names) != required_names:
        raise ValueError(f"arm names must be exactly {sorted(required_names)}")
    factorial = {
        (arm["evidence"], arm["mixer"])
        for arm in data["arms"]
        if arm["architecture"] == "path_mixer"
        and not arm.get("path_shuffle", False)
        and arm.get("mix_paths", True)
    }
    if factorial != {
        ("raw", "generic"),
        ("raw", "typed"),
        ("closure", "generic"),
        ("closure", "typed"),
    }:
        raise ValueError("arms must contain the complete unshuffled 2x2 factorial")
    shuffled = [arm for arm in data["arms"] if arm.get("path_shuffle", False)]
    if len(shuffled) != 1 or any(
        (arm["evidence"], arm["mixer"]) != ("closure", "typed") for arm in shuffled
    ):
        raise ValueError("path shuffle control must use the closure-typed arm")
    no_mix = [
        arm
        for arm in data["arms"]
        if arm["architecture"] == "path_mixer" and not arm.get("mix_paths", True)
    ]
    if len(no_mix) != 1 or (no_mix[0]["evidence"], no_mix[0]["mixer"]) != (
        "raw",
        "generic",
    ):
        raise ValueError(
            "exactly one raw-generic pair-state-no-mix control is required"
        )
    controls = [arm for arm in data["arms"] if arm["architecture"] != "path_mixer"]
    if sorted(arm["architecture"] for arm in controls) != [
        "edge_mlp",
        "generic_message_passing",
    ]:
        raise ValueError("generic message passing and edge MLP controls are required")
    if any(
        arm["evidence"] != "raw" or arm["mixer"] != "generic" or arm["path_shuffle"]
        for arm in controls
    ):
        raise ValueError("neural controls must use raw evidence without path shuffle")
    allowed_arm_keys = {
        "name",
        "architecture",
        "evidence",
        "mixer",
        "mix_paths",
        "path_shuffle",
    }
    if any(set(arm) != allowed_arm_keys for arm in data["arms"]):
        raise ValueError(f"arm keys must be exactly {sorted(allowed_arm_keys)}")
    expected_specs = {
        "raw_generic": ("path_mixer", "raw", "generic", True, False),
        "raw_typed": ("path_mixer", "raw", "typed", True, False),
        "closure_generic": ("path_mixer", "closure", "generic", True, False),
        "closure_typed": ("path_mixer", "closure", "typed", True, False),
        "closure_typed_path_shuffle": (
            "path_mixer",
            "closure",
            "typed",
            True,
            True,
        ),
        "pair_state_no_mix": ("path_mixer", "raw", "generic", False, False),
        "generic_message_passing": (
            "generic_message_passing",
            "raw",
            "generic",
            True,
            False,
        ),
        "edge_mlp": ("edge_mlp", "raw", "generic", False, False),
    }
    for arm in data["arms"]:
        actual = (
            arm["architecture"],
            arm["evidence"],
            arm["mixer"],
            arm["mix_paths"],
            arm["path_shuffle"],
        )
        if actual != expected_specs[arm["name"]]:
            raise ValueError(f"arm specification mismatch: {arm['name']}")
    expected_inference = {
        "contrast_direction": "negative_favors_positive_term_first_named_arm",
        "primary_order": [
            "typed_effect_raw",
            "typed_effect_closure",
            "factorial_interaction",
        ],
        "secondary_family_order": [
            "closure_effect_generic",
            "closure_effect_typed",
            "typed_closure_vs_path_shuffle",
            "path_mixing_vs_pair_state",
            "typed_path_vs_generic_message_passing",
        ],
        "secondary_multiplicity_method": (
            "holm_all_metrics_slices_and_solver_interactions"
        ),
        "secondary_family_scope": (
            "all_estimable_two_sided_bootstrap_tail_probabilities"
        ),
    }
    if data["inference"] != expected_inference:
        raise ValueError("inference family, order and multiplicity must stay frozen")
    data["graph"] = graph.to_dict()
    return data


def _prepare_output(path: Path, development: bool) -> None:
    resolved = path.resolve()
    allowed = (REPO_ROOT / "data/geometria_proporcional").resolve()
    if not development and allowed not in resolved.parents:
        raise ValueError(f"output must be below {allowed}")
    if resolved.exists():
        raise FileExistsError(f"output exists; refusing to overwrite: {resolved}")
    resolved.mkdir(parents=True)


def _stable_seed(*parts: Any) -> int:
    payload = ":".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def _arrays_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, raw in sorted(arrays.items()):
        array = np.ascontiguousarray(raw)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    return _arrays_digest(
        {name: tensor.detach().cpu().numpy() for name, tensor in state.items()}
    )


def _input_scale(views: list[GraphView]) -> float:
    values = np.concatenate([view.public.observed_log_ratio for view in views])
    return max(float(np.sqrt(np.mean(values * values))), 1e-6)


def _resource_observation(started: float) -> dict[str, float]:
    return {
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024.0 * 1024.0),
    }


def _enforce_budget(started: float, config: dict[str, Any], stage: str) -> None:
    observed = _resource_observation(started)
    if observed["elapsed_seconds"] > float(config["training"]["max_seconds"]):
        raise RuntimeError(
            f"runtime budget exceeded during {stage}: "
            f"{observed['elapsed_seconds']:.1f}s"
        )
    if observed["max_rss_gib"] > float(config["training"]["max_rss_gib"]):
        raise RuntimeError(
            f"RSS budget exceeded during {stage}: {observed['max_rss_gib']:.2f} GiB"
        )


def _model_for_arm(
    arm: dict[str, Any], config: dict[str, Any], scale: float
) -> torch.nn.Module:
    train = config["training"]
    common = {
        "hidden_dim": int(train["hidden_dim"]),
        "weight_floor": float(config["graph"]["weight_floor"]),
        "input_scale": scale,
    }
    if arm["architecture"] == "generic_message_passing":
        return GenericMessagePassing(**common)
    if arm["architecture"] == "edge_mlp":
        return EdgewiseMLP(**common)
    if arm["architecture"] != "path_mixer":
        raise ValueError(f"unknown architecture: {arm['architecture']}")
    return ProportionalPathMixer(
        **common,
        evidence=arm["evidence"],
        mixer=arm["mixer"],
        mix_paths=arm.get("mix_paths", True),
    )


def _view_tensors(
    view: GraphView, arm: dict[str, Any], seed: int
) -> dict[str, torch.Tensor]:
    tensors = observation_tensors(view.public)
    if arm.get("path_shuffle", False):
        tensors = shuffled_path_tensors(
            tensors,
            seed=_stable_seed("path-shuffle", seed, _path_structure_digest(view)),
        )
    if arm["evidence"] == "closure":
        tensors = materialize_closure_evidence(tensors)
    return tensors


def _path_structure_digest(view: GraphView) -> str:
    """Hash only structure shared by paired corruption views, never observations."""
    arrays = {
        key: value
        for key, value in view.public.arrays().items()
        if key != "observed_log_ratio"
    }
    return _arrays_digest(arrays)


def _shuffle_diagnostics(views: list[GraphView], seed: int) -> dict[str, Any]:
    changed = total = valid_total = degenerate = identity_views = ineligible_views = 0
    for view in views:
        original = observation_tensors(view.public)
        shuffled = shuffled_path_tensors(
            original,
            seed=_stable_seed("path-shuffle", seed, _path_structure_digest(view)),
        )
        before, after = original["path_index"], shuffled["path_index"]
        eligible = bool(shuffled.get("path_shuffle_eligible", torch.tensor(False)))
        ineligible_views += int(not eligible)
        total += len(before)
        valid = original["path_valid"]
        valid_total += int(valid.sum())
        changed += (
            int(torch.any(before[valid] != after[valid], dim=1).sum())
            if torch.any(valid)
            else 0
        )
        identity_views += int(eligible and torch.equal(before, after))
        degenerate += (
            int(torch.any(after[:, 0, None] == after[:, 1:], dim=1).sum())
            if len(after)
            else 0
        )
    return {
        "paths": total,
        "valid_paths": valid_total,
        "changed_valid_paths": changed,
        "changed_fraction_among_valid": changed / valid_total if valid_total else None,
        "degenerate_paths": degenerate,
        "identity_views": identity_views,
        "ineligible_views": ineligible_views,
    }


def _loss_for_view(
    model: torch.nn.Module,
    view: GraphView,
    arm: dict[str, Any],
    seed: int,
    coefficients: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    tensors = _view_tensors(view, arm, seed)
    output = model(tensors)
    x_hat = differentiable_wls(
        view.public, output.corrected_log_ratio, output.reliability
    )
    x_true = torch.as_tensor(view.private.x_true, dtype=x_hat.dtype)
    clean = torch.as_tensor(view.private.clean_log_ratio, dtype=x_hat.dtype)
    quotient = torch.mean((x_hat - x_true) ** 2)
    relation = torch.mean((output.corrected_log_ratio - clean) ** 2)
    closure = local_closure_loss(view.public, output.corrected_log_ratio)
    loss = (
        float(coefficients["quotient_mse"]) * quotient
        + float(coefficients["relation_mse"]) * relation
        + float(coefficients["closure_l1"]) * closure
    )
    return loss, {
        "loss": float(loss.detach()),
        "quotient_rmse": float(torch.sqrt(quotient.detach())),
        "relation_rmse": float(torch.sqrt(relation.detach())),
        "closure_l1": float(closure.detach()),
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _evaluate_training_objective(
    model: torch.nn.Module,
    views: list[GraphView],
    arm: dict[str, Any],
    seed: int,
    coefficients: dict[str, float],
) -> dict[str, float]:
    model.eval()
    rows = []
    with torch.no_grad():
        for view in views:
            _, row = _loss_for_view(model, view, arm, seed, coefficients)
            rows.append(row)
    return _mean_metrics(rows)


def _train_arm(
    model: torch.nn.Module,
    train_views: list[GraphView],
    val_views: list[GraphView],
    arm: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    started: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    history = []
    batch_size = int(train_config["batch_size"])
    for epoch in range(int(train_config["epochs"])):
        model.train()
        order = np.random.default_rng(
            _stable_seed(seed, epoch, "train-order")
        ).permutation(len(train_views))
        train_rows = []
        for start in range(0, len(order), batch_size):
            optimizer.zero_grad(set_to_none=True)
            batch_rows = []
            losses = []
            for index in order[start : start + batch_size]:
                loss, row = _loss_for_view(
                    model, train_views[int(index)], arm, seed, config["loss"]
                )
                losses.append(loss)
                batch_rows.append(row)
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_config["grad_clip"])
            )
            optimizer.step()
            train_rows.extend(batch_rows)
            _enforce_budget(
                started, config, f"training {arm['name']} epoch {epoch + 1}"
            )
        val_metrics = _evaluate_training_objective(
            model, val_views, arm, seed, config["loss"]
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train": _mean_metrics(train_rows),
                "validation": val_metrics,
            }
        )
    return history, optimizer.state_dict()


def _profile_trainable_flops(
    model: torch.nn.Module,
    view: GraphView,
    arm: dict[str, Any],
    seed: int,
    config: dict[str, Any],
) -> int:
    """Count one single-view forward/backward training objective with Torch."""
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as counter:
        loss, _ = _loss_for_view(model, view, arm, seed, config["loss"])
        loss.backward()
    model.zero_grad(set_to_none=True)
    return int(counter.get_total_flops())


def _measure_forward_latency(
    model: torch.nn.Module,
    view: GraphView,
    arm: dict[str, Any],
    seed: int,
    repeats: int,
    process_started: float,
    config: dict[str, Any],
) -> dict[str, float | int]:
    """Measure single-view CPU forward latency after two untimed warmups."""
    tensors = _view_tensors(view, arm, seed)
    model.eval()
    with torch.inference_mode():
        model(tensors)
        model(tensors)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(tensors)
            samples.append(time.perf_counter() - started)
            _enforce_budget(process_started, config, f"latency {arm['name']}")
    return {
        "repeats": repeats,
        "median_seconds": float(np.median(samples)),
        "p95_seconds": float(np.percentile(samples, 95)),
        "n_nodes": view.public.n_nodes,
        "n_edges": len(view.public.edge_index),
        "n_paths": len(view.public.path_index),
    }


def _solver_metrics(
    view: GraphView,
    corrected: np.ndarray,
    reliability: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    graph = config["graph"]
    wls = score_solver(
        solve_weighted_least_squares(
            view.public,
            values=corrected,
            weights=reliability,
            weight_floor=float(graph["weight_floor"]),
        ),
        view.private,
    )
    irls = score_solver(
        solve_huber_irls(
            view.public,
            values=corrected,
            base_weights=reliability,
            delta=float(graph["huber_delta"]),
            max_iterations=int(graph["irls_iterations"]),
            damping=float(graph["irls_damping"]),
            weight_floor=float(graph["weight_floor"]),
        ),
        view.private,
    )
    irls_quotient_rmse = irls.quotient_rmse if irls.converged else float("nan")
    irls_relation_rmse = irls.relation_rmse if irls.converged else float("nan")
    direct_x = direct_centered_decoder(
        view.public,
        corrected,
        reliability,
        weight_floor=float(graph["weight_floor"]),
    )
    direct_quotient_rmse = float(
        np.sqrt(np.mean((direct_x - view.private.x_true) ** 2))
    )
    anomaly = 1.0 - reliability
    target = view.private.causal_corruption_mask.astype(np.int64)
    ap = (
        float(average_precision_score(target, anomaly))
        if len(np.unique(target)) == 2
        else float("nan")
    )
    brier = float(brier_score_loss(target, np.clip(anomaly, 0.0, 1.0)))
    relation_rmse = float(
        np.sqrt(np.mean((corrected - view.private.clean_log_ratio) ** 2))
    )
    metrics = {
        "view_id": view.private.view_id,
        "master_id": view.private.master_id,
        "split": view.private.split,
        "mechanism": view.private.corruption_mechanism,
        "relation_rmse": relation_rmse,
        "confidence_ap_diagnostic": ap,
        "confidence_brier_diagnostic": brier,
        "wls_quotient_rmse": wls.quotient_rmse,
        "wls_relation_rmse": wls.relation_rmse,
        "wls_laplacian_rank": wls.laplacian_rank,
        "wls_condition": wls.laplacian_condition,
        "direct_quotient_rmse": direct_quotient_rmse,
        "irls_quotient_rmse": irls_quotient_rmse,
        "irls_relation_rmse": irls_relation_rmse,
        "irls_laplacian_rank": irls.laplacian_rank,
        "irls_condition": irls.laplacian_condition,
        "irls_converged": irls.converged,
        "irls_iterations": irls.iterations,
    }
    arrays = {
        "x_hat_wls": wls.x_hat,
        "x_hat_direct": direct_x,
        "x_hat_irls": irls.x_hat,
        "irls_weights": irls.weights,
    }
    return metrics, arrays


def _evaluate_model(
    model: torch.nn.Module,
    views: list[GraphView],
    arm: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    started: float,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    model.eval()
    metrics, states = [], []
    with torch.no_grad():
        for view in views:
            tensors = _view_tensors(view, arm, seed)
            output = model(tensors)
            corrected = output.corrected_log_ratio.cpu().numpy().astype(np.float64)
            reliability = output.reliability.cpu().numpy().astype(np.float64)
            row, solver_arrays = _solver_metrics(view, corrected, reliability, config)
            row["path_shuffle_eligible"] = bool(
                tensors.get("path_shuffle_eligible", torch.tensor(True))
            )
            metrics.append(row)
            states.append(
                {
                    "observed_log_ratio": view.public.observed_log_ratio,
                    "clean_log_ratio": view.private.clean_log_ratio,
                    "corrected_log_ratio": corrected,
                    "reliability": reliability,
                    "causal_corruption_mask": view.private.causal_corruption_mask,
                    "x_true": view.private.x_true,
                    "edge_index": tensors["edge_index"].cpu().numpy(),
                    "edge_valid": tensors["edge_valid"].cpu().numpy(),
                    "edge_variance": tensors["edge_variance"].cpu().numpy(),
                    "path_index": tensors["path_index"].cpu().numpy(),
                    "path_sign": tensors["path_sign"].cpu().numpy(),
                    "path_valid": tensors["path_valid"].cpu().numpy(),
                    "attention": output.path_attention.cpu().numpy(),
                    "n_nodes": np.asarray(int(tensors["n_nodes"])),
                    **solver_arrays,
                }
            )
            _enforce_budget(started, config, f"evaluation {arm['name']}")
    return metrics, states


def _evaluate_control(
    name: str,
    views: list[GraphView],
    config: dict[str, Any],
    started: float,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    metrics, states = [], []
    for view in views:
        if name == "observed_unweighted":
            corrected = view.public.observed_log_ratio.copy()
            reliability = np.ones_like(corrected)
        elif name == "exact_closure_only":
            corrected, reliability = exact_closure_only(
                view.public, weight_floor=float(config["graph"]["weight_floor"])
            )
        elif name == "oracle_weights":
            corrected = view.public.observed_log_ratio.copy()
            reliability = np.where(
                view.private.causal_corruption_mask,
                float(config["graph"]["weight_floor"]),
                1.0,
            )
        else:
            raise ValueError(name)
        row, solver_arrays = _solver_metrics(view, corrected, reliability, config)
        row["path_shuffle_eligible"] = True
        metrics.append(row)
        states.append(
            {
                "observed_log_ratio": view.public.observed_log_ratio,
                "clean_log_ratio": view.private.clean_log_ratio,
                "corrected_log_ratio": corrected,
                "reliability": reliability,
                "causal_corruption_mask": view.private.causal_corruption_mask,
                "x_true": view.private.x_true,
                "edge_index": view.public.edge_index,
                "edge_valid": view.public.edge_valid,
                "edge_variance": view.public.edge_variance,
                "path_index": view.public.path_index,
                "path_sign": view.public.path_sign,
                "path_valid": view.public.path_valid,
                "attention": np.empty((0,), dtype=np.float64),
                "n_nodes": np.asarray(view.public.n_nodes),
                **solver_arrays,
            }
        )
        _enforce_budget(started, config, f"evaluation {name}")
    return metrics, states


def _save_ragged_states(
    path: Path,
    views: list[GraphView],
    metrics: list[dict[str, Any]],
    states: list[dict[str, np.ndarray]],
) -> None:
    edge_offsets = [0]
    node_offsets = [0]
    path_offsets = [0]
    attention_offsets = [0]
    edge_keys = (
        "observed_log_ratio",
        "clean_log_ratio",
        "corrected_log_ratio",
        "reliability",
        "causal_corruption_mask",
        "irls_weights",
    )
    node_keys = ("x_true", "x_hat_direct", "x_hat_wls", "x_hat_irls")
    arrays: dict[str, Any] = {}
    for state in states:
        edge_offsets.append(edge_offsets[-1] + len(state["observed_log_ratio"]))
        node_offsets.append(node_offsets[-1] + len(state["x_true"]))
        attention_offsets.append(attention_offsets[-1] + len(state["attention"]))
    for view in views:
        path_offsets.append(path_offsets[-1] + len(view.public.path_index))
    for key in edge_keys:
        arrays[key] = np.concatenate([state[key] for state in states])
    for key in node_keys:
        arrays[key] = np.concatenate([state[key] for state in states])
    arrays.update(
        {
            "edge_offsets": np.asarray(edge_offsets, dtype=np.int64),
            "node_offsets": np.asarray(node_offsets, dtype=np.int64),
            "path_offsets": np.asarray(path_offsets, dtype=np.int64),
            "attention_offsets": np.asarray(attention_offsets, dtype=np.int64),
            "attention": np.concatenate([state["attention"] for state in states]),
            "edge_index": np.concatenate([state["edge_index"] for state in states]),
            "edge_valid": np.concatenate([state["edge_valid"] for state in states]),
            "edge_variance": np.concatenate(
                [state["edge_variance"] for state in states]
            ),
            "path_index": np.concatenate([state["path_index"] for state in states]),
            "path_sign": np.concatenate([state["path_sign"] for state in states]),
            "path_valid": np.concatenate([state["path_valid"] for state in states]),
            "n_nodes": np.asarray(
                [state["n_nodes"] for state in states], dtype=np.int64
            ),
            "view_id": np.asarray([view.private.view_id for view in views]),
            "master_id": np.asarray([view.private.master_id for view in views]),
            "split": np.asarray([view.private.split for view in views]),
            "mechanism": np.asarray(
                [view.private.corruption_mechanism for view in views]
            ),
            "relation_rmse": np.asarray([row["relation_rmse"] for row in metrics]),
            "wls_quotient_rmse": np.asarray(
                [row["wls_quotient_rmse"] for row in metrics]
            ),
            "direct_quotient_rmse": np.asarray(
                [row["direct_quotient_rmse"] for row in metrics]
            ),
            "wls_laplacian_rank": np.asarray(
                [row["wls_laplacian_rank"] for row in metrics], dtype=np.int64
            ),
            "wls_condition": np.asarray([row["wls_condition"] for row in metrics]),
            "irls_quotient_rmse": np.asarray(
                [row["irls_quotient_rmse"] for row in metrics]
            ),
            "irls_laplacian_rank": np.asarray(
                [row["irls_laplacian_rank"] for row in metrics], dtype=np.int64
            ),
            "irls_condition": np.asarray([row["irls_condition"] for row in metrics]),
            "irls_converged": np.asarray([row["irls_converged"] for row in metrics]),
            "irls_iterations": np.asarray([row["irls_iterations"] for row in metrics]),
            "path_shuffle_eligible": np.asarray(
                [row["path_shuffle_eligible"] for row in metrics], dtype=bool
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer_state: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Persist a deterministic, framework-readable last-epoch checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray("proportional-neural-checkpoint-v1"),
        "metadata_json": np.asarray(_canonical_json(metadata)),
        "optimizer_param_groups_json": np.asarray(
            _canonical_json(optimizer_state["param_groups"])
        ),
    }
    for name, tensor in sorted(model.state_dict().items()):
        arrays[f"model::{name}"] = tensor.detach().cpu().numpy()
    optimizer_scalars: dict[str, Any] = {}
    for parameter_id, state in sorted(optimizer_state["state"].items()):
        for name, value in sorted(state.items()):
            key = f"optimizer::{parameter_id}::{name}"
            if torch.is_tensor(value):
                arrays[key] = value.detach().cpu().numpy()
            else:
                optimizer_scalars[key] = value
    arrays["optimizer_scalars_json"] = np.asarray(_canonical_json(optimizer_scalars))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Restore a deterministic checkpoint and return its metadata."""
    with np.load(path, allow_pickle=False) as saved:
        if saved["format_version"].item() != "proportional-neural-checkpoint-v1":
            raise ValueError("unsupported checkpoint format")
        model_state = {
            key.removeprefix("model::"): torch.from_numpy(saved[key].copy())
            for key in saved.files
            if key.startswith("model::")
        }
        model.load_state_dict(model_state)
        if optimizer is not None:
            state: dict[int, dict[str, Any]] = defaultdict(dict)
            for key in saved.files:
                if not key.startswith("optimizer::"):
                    continue
                _, parameter_id, name = key.split("::", 2)
                state[int(parameter_id)][name] = torch.from_numpy(saved[key].copy())
            scalars = json.loads(saved["optimizer_scalars_json"].item())
            for key, value in scalars.items():
                _, parameter_id, name = key.split("::", 2)
                state[int(parameter_id)][name] = value
            optimizer.load_state_dict(
                {
                    "state": dict(state),
                    "param_groups": json.loads(
                        saved["optimizer_param_groups_json"].item()
                    ),
                }
            )
        return json.loads(saved["metadata_json"].item())


def _summary(metrics_by_key: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    metric_names = (
        "relation_rmse",
        "confidence_ap_diagnostic",
        "confidence_brier_diagnostic",
        "wls_quotient_rmse",
        "wls_relation_rmse",
        "direct_quotient_rmse",
        "irls_quotient_rmse",
        "irls_relation_rmse",
    )
    for key, rows in sorted(metrics_by_key.items()):
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["split"], row["mechanism"])].append(row)
        output[key] = {}
        for (split, mechanism), group in sorted(grouped.items()):

            def finite_mean(name: str) -> float:
                values = np.asarray([row[name] for row in group], dtype=np.float64)
                return (
                    float(values[np.isfinite(values)].mean())
                    if np.isfinite(values).any()
                    else float("nan")
                )

            output[key][f"{split}|{mechanism}"] = {
                "views": len(group),
                **{name: finite_mean(name) for name in metric_names},
                "finite_counts": {
                    name: int(np.isfinite([row[name] for row in group]).sum())
                    for name in metric_names
                },
                "irls_failure_rate": float(
                    1.0 - np.mean([bool(row["irls_converged"]) for row in group])
                ),
                "irls_iterations_mean": float(
                    np.mean([int(row["irls_iterations"]) for row in group])
                ),
                "irls_iterations_max": int(
                    max(row["irls_iterations"] for row in group)
                ),
            }
        test = [row for row in rows if row["split"] == "test"]
        pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in test:
            pairs[row["master_id"]][row["mechanism"]] = row
        paired = [pair for pair in pairs.values() if set(pair) == {"iid", "grouped"}]
        output[key]["paired_ood_corruption"] = {
            "masters": len(paired),
            "delta_grouped_minus_iid": {
                name: _finite_mean(
                    [pair["grouped"][name] - pair["iid"][name] for pair in paired]
                )
                for name in ("relation_rmse", "wls_quotient_rmse", "irls_quotient_rmse")
            },
        }
    return output


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array[np.isfinite(array)].mean())
        if np.isfinite(array).any()
        else float("nan")
    )


CONTRASTS = {
    "typed_effect_raw": {"raw_typed": 1.0, "raw_generic": -1.0},
    "typed_effect_closure": {"closure_typed": 1.0, "closure_generic": -1.0},
    "closure_effect_generic": {"closure_generic": 1.0, "raw_generic": -1.0},
    "closure_effect_typed": {"closure_typed": 1.0, "raw_typed": -1.0},
    "factorial_interaction": {
        "closure_typed": 1.0,
        "closure_generic": -1.0,
        "raw_typed": -1.0,
        "raw_generic": 1.0,
    },
    "typed_closure_vs_path_shuffle": {
        "closure_typed": 1.0,
        "closure_typed_path_shuffle": -1.0,
    },
    "path_mixing_vs_pair_state": {"raw_generic": 1.0, "pair_state_no_mix": -1.0},
    "typed_path_vs_generic_message_passing": {
        "raw_typed": 1.0,
        "generic_message_passing": -1.0,
    },
}


def _seed_averaged_rows(
    metrics_by_key: dict[str, list[dict[str, Any]]],
    arm_names: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in arm_names:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for key, rows in metrics_by_key.items():
            if key.startswith(f"{arm}|seed="):
                for row in rows:
                    grouped[row["view_id"]].append(row)
        output[arm] = {}
        for view_id, rows in grouped.items():
            base = rows[0]
            output[arm][view_id] = {
                **{
                    key: base[key]
                    for key in ("view_id", "master_id", "split", "mechanism")
                },
                **{
                    metric: _strict_mean([row[metric] for row in rows])
                    for metric in (
                        "relation_rmse",
                        "direct_quotient_rmse",
                        "wls_quotient_rmse",
                        "irls_quotient_rmse",
                    )
                },
                "irls_failure_rate": float(
                    np.mean([not bool(row["irls_converged"]) for row in rows])
                ),
                "path_shuffle_eligible": all(
                    bool(row.get("path_shuffle_eligible", True)) for row in rows
                ),
            }
    return output


def _contrast_values(
    arm_rows: dict[str, dict[str, dict[str, Any]]],
    terms: dict[str, float],
    metric: str,
    mechanism: str,
    master_ids: list[str],
) -> np.ndarray:
    values = np.full(len(master_ids), np.nan, dtype=np.float64)
    for index, master_id in enumerate(master_ids):
        total = 0.0
        for arm, coefficient in terms.items():
            candidates = [
                row
                for row in arm_rows[arm].values()
                if row["master_id"] == master_id
                and row["split"] == "test"
                and row["mechanism"] == mechanism
            ]
            if (
                len(candidates) != 1
                or not candidates[0].get("path_shuffle_eligible", True)
                or not np.isfinite(candidates[0][metric])
            ):
                break
            total += coefficient * candidates[0][metric]
        else:
            values[index] = total
    return values


def _eligible_master_ids(
    arm_rows: dict[str, dict[str, dict[str, Any]]],
    terms: dict[str, float],
    master_ids: list[str],
) -> list[str]:
    """Freeze the paired estimand universe before inspecting metric finiteness."""
    eligible = []
    for master_id in master_ids:
        for arm in terms:
            candidates = [
                row
                for row in arm_rows[arm].values()
                if row["master_id"] == master_id
                and row["split"] == "test"
                and row["mechanism"] in {"iid", "grouped"}
            ]
            if len(candidates) != 2 or not all(
                row.get("path_shuffle_eligible", True) for row in candidates
            ):
                break
        else:
            eligible.append(master_id)
    return eligible


def _strict_mean(values: list[float]) -> float:
    """Average only when every paired value exists and is finite."""
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean())
        if len(array) and np.all(np.isfinite(array))
        else float("nan")
    )


def _bootstrap_result(
    values: np.ndarray,
    *,
    indices_for: Any,
    require_complete: bool,
) -> dict[str, Any]:
    valid = np.isfinite(values)
    base = {
        "n_total": int(len(values)),
        "n_complete": int(valid.sum()),
        "missing": int((~valid).sum()),
    }
    if require_complete and not np.all(valid):
        return {
            **base,
            "status": "NOT_EVALUABLE_SOLVER_FAILURE",
            "mean": None,
            "ci95": [None, None],
            "p_less_than_zero": None,
            "p_two_sided": None,
            "p_holm_secondary_family": None,
            "n": 0,
        }
    if not np.any(valid):
        return {
            **base,
            "status": "NOT_EVALUABLE_NO_FINITE_VALUES",
            "mean": None,
            "ci95": [None, None],
            "p_less_than_zero": None,
            "p_two_sided": None,
            "p_holm_secondary_family": None,
            "n": 0,
        }
    complete = values[valid]
    draws = complete[indices_for(len(complete))].mean(axis=1)
    p_less_than_zero = float(np.mean(draws < 0.0))
    return {
        **base,
        "status": "ESTIMATED",
        "mean": float(complete.mean()),
        "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
        "p_less_than_zero": p_less_than_zero,
        "p_two_sided": min(1.0, 2.0 * min(p_less_than_zero, 1.0 - p_less_than_zero)),
        "p_holm_secondary_family": None,
        "n": int(len(complete)),
        "bootstrap_replicates": int(len(draws)),
    }


def _bootstrap_contrasts(
    metrics_by_key: dict[str, list[dict[str, Any]]],
    arm_names: list[str],
    master_ids: list[str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    inference: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    arm_rows = _seed_averaged_rows(metrics_by_key, arm_names)
    output: dict[str, Any] = {}
    index_tables: dict[int, np.ndarray] = {}

    def indices_for(n: int) -> np.ndarray:
        if n not in index_tables:
            rng = np.random.default_rng(
                _stable_seed(bootstrap_seed, "complete-case", n)
            )
            index_tables[n] = rng.integers(
                0, n, size=(bootstrap_replicates, n), dtype=np.int32
            )
        return index_tables[n]

    for contrast_name, terms in CONTRASTS.items():
        eligible_ids = _eligible_master_ids(arm_rows, terms, master_ids)
        output[contrast_name] = {
            "terms": terms,
            "eligibility": {
                "predeclared_test_masters": len(master_ids),
                "included_test_masters": len(eligible_ids),
                "excluded_for_infeasible_path_shuffle": len(master_ids)
                - len(eligible_ids),
                "policy": "eligibility is fixed jointly across IID/grouped before metric values are inspected",
            },
            "metrics": {},
        }
        for metric in (
            "relation_rmse",
            "direct_quotient_rmse",
            "wls_quotient_rmse",
            "irls_quotient_rmse",
        ):
            iid = _contrast_values(arm_rows, terms, metric, "iid", eligible_ids)
            grouped = _contrast_values(arm_rows, terms, metric, "grouped", eligible_ids)
            slices = {
                "test|iid": iid,
                "test|grouped": grouped,
                "grouped_minus_iid": grouped - iid,
            }
            output[contrast_name]["metrics"][metric] = {}
            for slice_name, values in slices.items():
                result = _bootstrap_result(
                    values,
                    indices_for=indices_for,
                    require_complete=metric.startswith("irls_"),
                )
                output[contrast_name]["metrics"][metric][slice_name] = result
        wls_iid = _contrast_values(
            arm_rows, terms, "wls_quotient_rmse", "iid", eligible_ids
        )
        wls_grouped = _contrast_values(
            arm_rows, terms, "wls_quotient_rmse", "grouped", eligible_ids
        )
        irls_iid = _contrast_values(
            arm_rows, terms, "irls_quotient_rmse", "iid", eligible_ids
        )
        irls_grouped = _contrast_values(
            arm_rows, terms, "irls_quotient_rmse", "grouped", eligible_ids
        )
        output[contrast_name]["solver_interaction_irls_minus_wls"] = {
            slice_name: _bootstrap_result(
                values,
                indices_for=indices_for,
                require_complete=True,
            )
            for slice_name, values in {
                "test|iid": irls_iid - wls_iid,
                "test|grouped": irls_grouped - wls_grouped,
                "grouped_minus_iid": (irls_grouped - wls_grouped)
                - (irls_iid - wls_iid),
            }.items()
        }
    _apply_secondary_holm(output, inference)
    return output, index_tables


def _apply_secondary_holm(contrasts: dict[str, Any], inference: dict[str, Any]) -> None:
    primary = list(inference["primary_order"])
    secondary = list(inference["secondary_family_order"])
    if set(primary) | set(secondary) != set(contrasts) or set(primary) & set(secondary):
        raise AssertionError("inference families must partition the contrasts")
    records: list[dict[str, Any]] = []
    for name, payload in contrasts.items():
        payload["family_role"] = "primary" if name in primary else "secondary"
        payload["predeclared_order"] = (
            primary.index(name) if name in primary else secondary.index(name)
        )
        if name not in secondary:
            continue
        for slices in payload["metrics"].values():
            records.extend(
                result
                for result in slices.values()
                if result["p_two_sided"] is not None
            )
        records.extend(
            result
            for result in payload["solver_interaction_irls_minus_wls"].values()
            if result["p_two_sided"] is not None
        )
    ordered = sorted(records, key=lambda result: result["p_two_sided"])
    running = 0.0
    total = len(ordered)
    for rank, result in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * result["p_two_sided"])
        running = max(running, adjusted)
        result["p_holm_secondary_family"] = running


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(data), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _manifest(
    output: Path, config: dict[str, Any], source_records: dict[str, Any]
) -> dict[str, Any]:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {
            "manifest.json",
            "runtime_observation.json",
        }:
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
        "replay_scope": {
            "byte_exact_files": "all manifest-listed files",
            "excluded_nondeterministic_files": ["runtime_observation.json"],
        },
        "source_files": source_records,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "files": files,
    }


def _report(
    summary: dict[str, Any],
    parameter_counts: dict[str, int],
    contrasts: dict[str, Any],
    inference: dict[str, Any],
) -> str:
    lines = [
        "# Proportional graph neural smoke",
        "",
        "> Development smoke only. No architecture promotion and no GO/NO-GO decision.",
        "",
        "## Parameter counts",
        "",
        "| Arm | Parameters |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {count} |" for name, count in sorted(parameter_counts.items())
    )
    lines.extend(
        [
            "",
            "## Test summary",
            "",
            "| Arm/seed | Slice | Relation RMSE | Direct quotient RMSE | WLS quotient RMSE | IRLS quotient RMSE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for key, slices in sorted(summary.items()):
        for slice_name, metrics in sorted(slices.items()):
            if not slice_name.startswith("test|"):
                continue
            lines.append(
                f"| `{key}` | `{slice_name}` | {metrics['relation_rmse']:.4f} | "
                f"{metrics['direct_quotient_rmse']:.4f} | "
                f"{metrics['wls_quotient_rmse']:.4f} | {metrics['irls_quotient_rmse']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Predeclared contrasts",
            "",
            "Negative deltas favor the first-named arm because all errors are lower-is-better.",
            "The unit is a held-out master and each value first averages the two training seeds.",
            "",
            "Primary contrasts are reported first in their frozen order, followed by the Holm-adjusted secondary family.",
            "",
            "| Family | Contrast | Metric | Slice | Status | Mean delta | CI95 | n | Holm secondary |",
            "|---|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    contrast_order = list(inference["primary_order"]) + list(
        inference["secondary_family_order"]
    )
    if set(contrast_order) != set(contrasts):
        raise AssertionError(
            "report contrast order must cover the frozen inference family"
        )
    for name in contrast_order:
        payload = contrasts[name]
        family = payload["family_role"]
        for metric, slices in payload["metrics"].items():
            for slice_name, result in slices.items():
                mean = result["mean"]
                ci = result["ci95"]
                mean_text = "NA" if mean is None else f"{mean:+.4f}"
                ci_text = "NA" if ci[0] is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                holm = result["p_holm_secondary_family"]
                holm_text = "NA" if holm is None else f"{holm:.4f}"
                lines.append(
                    f"| `{family}` | `{name}` | `{metric}` | `{slice_name}` | `{result['status']}` | "
                    f"{mean_text} | {ci_text} | {result['n']} | {holm_text} |"
                )
        for slice_name, result in payload["solver_interaction_irls_minus_wls"].items():
            mean = result["mean"]
            ci = result["ci95"]
            mean_text = "NA" if mean is None else f"{mean:+.4f}"
            ci_text = "NA" if ci[0] is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
            holm = result["p_holm_secondary_family"]
            holm_text = "NA" if holm is None else f"{holm:.4f}"
            lines.append(
                f"| `{family}` | `{name}` | `solver_interaction_irls_minus_wls` | `{slice_name}` | "
                f"`{result['status']}` | {mean_text} | {ci_text} | {result['n']} | "
                f"{holm_text} |"
            )
    lines.extend(
        [
            "",
            "This development smoke checks mechanics and directional signal only. The intervals are",
            "diagnostic and do not authorize architecture promotion or a GO/NO-GO decision.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve(strict=True)
    config = _load_config(config_path)
    output = args.output.resolve()
    source_records = _source_records(config_path, args.development)
    _prepare_output(output, args.development)
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()

    graph_config = ProportionalGraphConfig.from_dict(config["graph"])
    generated_views = generate_graph_views(graph_config)
    shuffle_arm = next(
        arm for arm in config["arms"] if arm["name"] == "closure_typed_path_shuffle"
    )
    eligibility_seed = int(config["training"]["seeds"][0])
    analysis_eligibility = {
        view.private.view_id: bool(
            _view_tensors(view, shuffle_arm, eligibility_seed)["path_shuffle_eligible"]
        )
        for view in generated_views
    }
    eligibility_by_master: dict[str, set[bool]] = defaultdict(set)
    for view in generated_views:
        eligibility_by_master[view.private.master_id].add(
            analysis_eligibility[view.private.view_id]
        )
    if any(len(values) != 1 for values in eligibility_by_master.values()):
        raise AssertionError("path-shuffle eligibility must be paired by master")
    views = [
        view for view in generated_views if analysis_eligibility[view.private.view_id]
    ]
    by_split = defaultdict(list)
    for view in views:
        by_split[view.private.split].append(view)
    train_views = by_split["train"]
    val_views = by_split["validation"]
    eval_views = val_views + by_split["test"]
    scale = _input_scale(train_views)
    config["resolved_input_scale"] = scale
    config["generated_split_counts"] = {
        key: sum(view.private.split == key for view in generated_views)
        for key in sorted({view.private.split for view in generated_views})
    }
    config["analysis_split_counts"] = {
        key: len(value) for key, value in sorted(by_split.items())
    }
    _write_json(output / "resolved_config.json", config)
    _write_json(
        output / "dataset_manifest.json",
        [
            {
                **view.private.metadata(),
                "public_sha256": _arrays_digest(view.public.arrays()),
                "analysis_included": analysis_eligibility[view.private.view_id],
                "analysis_exclusion_reason": (
                    None
                    if analysis_eligibility[view.private.view_id]
                    else "no_feasible_balanced_path_derangement"
                ),
            }
            for view in generated_views
        ],
    )

    parameter_counts = {}
    metrics_by_key: dict[str, list[dict[str, Any]]] = {}
    histories = {}
    shuffle_diagnostics = {}
    initialization_digests: dict[str, dict[str, str]] = {}
    trainable_flops: dict[str, int] = {}
    runtime_by_key: dict[str, Any] = {}
    for seed in config["training"]["seeds"]:
        seed = int(seed)
        shuffle_diagnostics[str(seed)] = {
            split: _shuffle_diagnostics(split_views, seed)
            for split, split_views in (
                ("train", train_views),
                ("validation", val_views),
                ("test", by_split["test"]),
            )
        }
        if any(
            diagnostic["identity_views"] or diagnostic["degenerate_paths"]
            for diagnostic in shuffle_diagnostics[str(seed)].values()
        ):
            raise RuntimeError("path shuffle control is not a complete derangement")
        torch.manual_seed(seed)
        template = _model_for_arm(config["arms"][0], config, scale)
        initial_state = {
            key: value.detach().clone() for key, value in template.state_dict().items()
        }
        expected_initial_digest = _state_digest(initial_state)
        initialization_digests[str(seed)] = {}
        expected_count = parameter_count(template)
        for arm in config["arms"]:
            if arm["architecture"] == "edge_mlp":
                torch.manual_seed(_stable_seed(seed, arm["name"], "initialization"))
                model = _model_for_arm(arm, config, scale)
            else:
                model = _model_for_arm(arm, config, scale)
                model.load_state_dict(initial_state)
            actual_initial_digest = _state_digest(model.state_dict())
            if (
                arm["architecture"] != "edge_mlp"
                and actual_initial_digest != expected_initial_digest
            ):
                raise AssertionError(
                    "parameter/state-matched arms do not share identical initialization"
                )
            initialization_digests[str(seed)][arm["name"]] = actual_initial_digest
            count = parameter_count(model)
            if arm["architecture"] != "edge_mlp" and count != expected_count:
                raise AssertionError(
                    "parameter/state-matched arms differ in parameter count"
                )
            parameter_counts[arm["name"]] = count
            if arm["name"] not in trainable_flops:
                trainable_flops[arm["name"]] = 0
                for view in train_views:
                    trainable_flops[arm["name"]] += _profile_trainable_flops(
                        model, view, arm, seed, config
                    )
                    _enforce_budget(started, config, f"FLOP profile {arm['name']}")
            history, optimizer_state = _train_arm(
                model, train_views, val_views, arm, seed, config, started
            )
            key = f"{arm['name']}|seed={seed}"
            histories[key] = history
            metrics, states = _evaluate_model(
                model, eval_views, arm, seed, config, started
            )
            runtime_by_key[key] = {
                "forward_latency": _measure_forward_latency(
                    model,
                    eval_views[0],
                    arm,
                    seed,
                    int(config["training"]["latency_repeats"]),
                    started,
                    config,
                ),
                "cumulative": _resource_observation(started),
            }
            metrics_by_key[key] = metrics
            _save_ragged_states(
                output / "raw_eval" / f"{key}.npz", eval_views, metrics, states
            )
            _save_checkpoint(
                output / "checkpoints" / f"{key}.npz",
                model,
                optimizer_state,
                {
                    "arm": arm,
                    "seed": seed,
                    "input_scale": scale,
                    "last_epoch": int(config["training"]["epochs"]),
                    "parameter_count": count,
                },
            )

    for control in ("observed_unweighted", "exact_closure_only", "oracle_weights"):
        metrics, states = _evaluate_control(control, eval_views, config, started)
        metrics_by_key[control] = metrics
        _save_ragged_states(
            output / "raw_eval" / f"{control}.npz", eval_views, metrics, states
        )

    summary = _summary(metrics_by_key)
    test_master_ids = sorted({view.private.master_id for view in by_split["test"]})
    bootstrap_seed = _stable_seed(graph_config.seed, "neural-bootstrap")
    contrasts, bootstrap_tables = _bootstrap_contrasts(
        metrics_by_key,
        [arm["name"] for arm in config["arms"]],
        test_master_ids,
        int(graph_config.bootstrap_replicates),
        bootstrap_seed,
        config["inference"],
    )
    np.savez_compressed(
        output / "bootstrap_indices.npz",
        master_id=np.asarray(test_master_ids),
        bootstrap_seed=np.asarray(bootstrap_seed, dtype=np.int64),
        **{f"complete_n_{n}": table for n, table in sorted(bootstrap_tables.items())},
    )
    compute_contract = {
        "factorial_path_mlp_calls_per_valid_path": 4,
        "factorial_arms_equal_calls": True,
        "pair_state_no_mix_path_mlp_calls_per_valid_path": 4,
        "parameter_shape_state_matched_group": [
            arm["name"] for arm in config["arms"] if arm["architecture"] != "edge_mlp"
        ],
        "standalone_unmatched_controls": ["edge_mlp"],
        "parameters": parameter_counts,
        "initialization_sha256": initialization_digests,
        "trainable_flops": {
            "method": "torch FlopCounterMode over every training view's complete forward/backward objective",
            "per_epoch_by_arm": trainable_flops,
            "full_schedule_by_arm": {
                arm: count
                * int(config["training"]["epochs"])
                * len(config["training"]["seeds"])
                for arm, count in trainable_flops.items()
            },
            "scope": "model, differentiable WLS, losses and backward; excludes optimizer and validation",
        },
        "operation_contract": {
            "factorial": "four path-MLP calls per valid path (two operand orders times sign symmetrization)",
            "pair_state_no_mix": "same calls and parameters; operand states replaced by target state before mixing",
            "generic_message_passing": "same modules and parameters; unordered endpoint-neighborhood aggregation",
            "edge_mlp": "edge-local encoder and two heads; intentionally not capacity matched",
            "path_shuffle_exclusion": "the helper retains original paths only to return eligible=false; the runner excludes every such master-paired view from the common train, validation and test analysis universe before any arm or contrast",
            "generic_message_passing_interpretation": "matched in parameter count, tensor shapes and initialization, not in trainable FLOPs; FLOPs and latency are reported as descriptive covariates and no typage attribution may ignore them",
        },
        "solver_tuning": {
            "wls": {"weight_floor": float(config["graph"]["weight_floor"])},
            "huber_irls": {
                key: config["graph"][key]
                for key in (
                    "huber_delta",
                    "irls_damping",
                    "irls_iterations",
                    "weight_floor",
                )
            },
        },
        "timing_protocol": "monotonic wall clock for full process; checked after each training batch and evaluation view",
        "latency_protocol": {
            "unit": "single validation view forward on CPU",
            "warmups": 2,
            "repeats": int(config["training"]["latency_repeats"]),
            "summaries": ["median_seconds", "p95_seconds"],
            "observations_file": "runtime_observation.json",
        },
        "memory_protocol": "resource.getrusage(RUSAGE_SELF).ru_maxrss peak, converted from KiB to GiB",
        "budget": {
            "max_seconds": float(config["training"]["max_seconds"]),
            "max_rss_gib": float(config["training"]["max_rss_gib"]),
        },
    }
    _write_json(output / "history.json", histories)
    _write_json(
        output / "control_diagnostics.json", {"path_shuffle": shuffle_diagnostics}
    )
    _write_json(output / "per_view_metrics.json", metrics_by_key)
    _write_json(output / "summary.json", summary)
    _write_json(output / "contrasts.json", contrasts)
    _write_json(output / "compute_contract.json", compute_contract)
    (output / "SMOKE_REPORT.md").write_text(
        _report(summary, parameter_counts, contrasts, config["inference"]),
        encoding="utf-8",
    )
    replay = output / "replay.sh"
    replay_config = (
        config_path if args.development else DEFAULT_CONFIG.relative_to(REPO_ROOT)
    )
    development_flag = " --development" if args.development else ""
    replay.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "if [[ $# -ne 1 ]]; then echo 'usage: replay.sh OUTPUT_DIR' >&2; exit 2; fi\n"
        f"cd {shlex_quote(str(REPO_ROOT))}\n"
        "venv/bin/python experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py "
        f"--config {shlex_quote(str(replay_config))} "
        f'--output "$1"{development_flag}\n',
        encoding="utf-8",
    )
    replay.chmod(0o755)
    _enforce_budget(started, config, "artifact finalization")
    observed_resources = _resource_observation(started)
    _write_json(
        output / "runtime_observation.json",
        {
            **observed_resources,
            "by_model_run": runtime_by_key,
            "included_in_byte_exact_replay": False,
            "reason": "wall time and peak RSS are run-specific observations",
        },
    )
    _write_json(output / "manifest.json", _manifest(output, config, source_records))
    print(
        json.dumps(
            {
                "views": len(views),
                "train": len(train_views),
                "validation": len(val_views),
                "test": len(by_split["test"]),
                "neural_runs": len(config["arms"]) * len(config["training"]["seeds"]),
                **observed_resources,
            },
            indent=2,
        )
    )


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


if __name__ == "__main__":
    main()
