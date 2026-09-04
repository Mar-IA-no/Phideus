#!/usr/bin/env python3
"""Audit a fixed-depth differentiable IRLS surrogate against the CPU executor."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import resource
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/geometria_proporcional"))

import run_proportional_graph_neural_smoke as smoke  # noqa: E402
from geometria_proporcional.proportional_graph_contract import (  # noqa: E402
    ProportionalGraphConfig,
    generate_graph_views,
    incidence_matrix,
    solve_huber_irls,
)
from geometria_proporcional.proportional_graph_neural import (  # noqa: E402
    differentiable_huber_irls_fixed,
)

DEFAULT_CONFIG = (
    ROOT
    / "experiments/geometria_proporcional/configs/proportional_graph_irls_surrogate_fidelity_v1.json"
)
SOURCES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_IRLS_SURROGATE_FIDELITY_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_irls_surrogate_fidelity_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_irls_surrogate_fidelity.py",
    "experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py",
    "experiments/geometria_proporcional/configs/proportional_graph_neural_smoke_v1.json",
    "src/geometria_proporcional/proportional_graph_contract.py",
    "src/geometria_proporcional/proportional_graph_neural.py",
)


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    expected = {
        "schema_version",
        "source_smoke",
        "source_config",
        "arms",
        "seeds",
        "alphas",
        "depths",
        "gradient_depths",
        "gradient_alphas",
        "gradient_views",
        "value_path_views",
        "finite_difference_step",
        "kink_step_multiplier",
        "stability_coordinates",
        "conformity",
        "runtime",
    }
    if (
        set(data) != expected
        or data["schema_version"] != "proportional-graph-irls-surrogate-fidelity-v1"
    ):
        raise ValueError("invalid surrogate fidelity config schema")
    if data["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("primary arms must remain frozen")
    if data["seeds"] != [104729, 130363] or data["depths"] != [
        4,
        8,
        16,
        32,
        64,
        128,
        256,
    ]:
        raise ValueError("seeds and depth grid must remain frozen")
    if not set(data["gradient_depths"]).issubset(data["depths"]):
        raise ValueError("gradient depths must belong to the value grid")
    if data["runtime"]["torch_threads"] != 1 or data["finite_difference_step"] <= 0:
        raise ValueError("CPU and finite-difference contracts are invalid")
    return data


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, array in sorted(arrays.items()):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                payload.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def select_stratified(views: list[Any], count: int) -> list[Any]:
    """Deterministic round-robin over node cardinalities."""
    groups = {
        n: sorted(
            (v for v in views if v.public.n_nodes == n), key=lambda v: v.private.view_id
        )
        for n in sorted({v.public.n_nodes for v in views})
    }
    selected, offset = [], 0
    while len(selected) < count:
        advanced = False
        for n in groups:
            if offset < len(groups[n]) and len(selected) < count:
                selected.append(groups[n][offset])
                advanced = True
        if not advanced:
            break
        offset += 1
    if len(selected) != count:
        raise ValueError("insufficient views for requested stratified sample")
    return selected


def numpy_fixed_irls(
    observation: Any,
    values: np.ndarray,
    *,
    steps: int,
    delta: float,
    damping: float,
    weight_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Independent NumPy fixed-depth reference; no Torch surrogate helpers."""
    valid = np.asarray(observation.edge_valid, dtype=bool)
    matrix_all = incidence_matrix(observation.n_nodes, observation.edge_index).astype(
        np.float64
    )
    matrix = matrix_all[valid]
    y = np.asarray(values, dtype=np.float64)[valid]
    variance = np.asarray(observation.edge_variance, dtype=np.float64)[valid]
    threshold = float(delta) * max(float(np.sqrt(np.median(variance))), 1e-8)
    weights = np.ones(len(y), dtype=np.float64)
    margin = np.full(len(y), np.inf, dtype=np.float64)
    previous_x: np.ndarray | None = None
    solution_change = float("nan")

    def solve(local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized = np.clip(local, weight_floor, None)
        normalized = normalized / normalized.mean()
        laplacian = matrix.T @ (normalized[:, None] * matrix)
        rhs = matrix.T @ (normalized * y)
        ones = np.ones((observation.n_nodes, 1), dtype=np.float64)
        kkt = np.block([[laplacian, ones], [ones.T, np.zeros((1, 1))]])
        return np.linalg.solve(kkt, np.concatenate((rhs, np.zeros(1))))[:-1], normalized

    for _ in range(steps):
        x_hat, _ = solve(weights)
        if previous_x is not None:
            solution_change = float(np.max(np.abs(x_hat - previous_x)))
        residual = matrix @ x_hat - y
        margin = np.minimum(margin, np.abs(np.abs(residual) - threshold))
        candidate = np.ones_like(residual)
        large = np.abs(residual) > threshold
        candidate[large] = threshold / np.abs(residual[large])
        candidate = np.clip(candidate, weight_floor, 1.0)
        weights = (1.0 - damping) * weights + damping * candidate
        previous_x = x_hat.copy()
    x_hat, normalized = solve(weights)
    if previous_x is not None:
        solution_change = float(np.max(np.abs(x_hat - previous_x)))
    normalized_residual = (matrix @ x_hat - y) / max(
        float(np.sqrt(np.median(variance))), 1e-8
    )
    magnitude = np.abs(normalized_residual)
    huber_terms = np.where(
        magnitude <= delta,
        0.5 * normalized_residual * normalized_residual,
        delta * (magnitude - 0.5 * delta),
    )
    full_weights = np.zeros(len(values), dtype=np.float64)
    full_weights[valid] = normalized
    full_margin = np.full(len(values), np.inf, dtype=np.float64)
    full_margin[valid] = margin
    return x_hat, full_weights, full_margin, float(np.sum(huber_terms)), solution_change


def checkpoint_relations(source: Path, arm: str, seed: int) -> dict[str, np.ndarray]:
    with np.load(
        source / "raw_eval" / f"{arm}|seed={seed}.npz", allow_pickle=False
    ) as saved:
        offsets, values, ids = (
            saved["edge_offsets"],
            saved["corrected_log_ratio"],
            saved["view_id"],
        )
        return {
            str(view_id): values[offsets[i] : offsets[i + 1]].astype(np.float64)
            for i, view_id in enumerate(ids)
        }


def resource_sample(started: float, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }


def enforce(sample: dict[str, Any], cfg: dict[str, Any]) -> None:
    if sample["elapsed_seconds"] > cfg["runtime"]["max_seconds"]:
        raise RuntimeError("runtime budget exceeded")
    if sample["max_rss_gib"] > cfg["runtime"]["max_rss_gib"]:
        raise RuntimeError("RSS budget exceeded")


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
    torch.set_num_threads(cfg["runtime"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    samples = [resource_sample(started, "start")]

    source_cfg = smoke._load_config((ROOT / cfg["source_config"]).resolve())
    graph = source_cfg["graph"]
    views = generate_graph_views(ProportionalGraphConfig.from_dict(graph))
    shuffle_arm = next(
        a for a in source_cfg["arms"] if a["name"] == "closure_typed_path_shuffle"
    )
    eligible = {
        v.private.master_id: bool(
            smoke._view_tensors(v, shuffle_arm, cfg["seeds"][0])[
                "path_shuffle_eligible"
            ]
        )
        for v in views
    }
    validation = [
        v
        for v in views
        if v.private.split == "validation" and eligible[v.private.master_id]
    ]
    validation.sort(key=lambda v: v.private.view_id)
    path_views = select_stratified(validation, cfg["value_path_views"])
    probe_views = select_stratified(validation, cfg["gradient_views"])
    source = ROOT / cfg["source_smoke"]
    corrected = {
        (arm, seed): checkpoint_relations(source, arm, seed)
        for arm in cfg["arms"]
        for seed in cfg["seeds"]
    }

    states: list[tuple[Any, str, int | None, float, np.ndarray]] = []
    states.extend(
        (view, "observed", None, 0.0, view.public.observed_log_ratio.astype(np.float64))
        for view in validation
    )
    for view in path_views:
        observed = view.public.observed_log_ratio.astype(np.float64)
        for arm in cfg["arms"]:
            for seed in cfg["seeds"]:
                endpoint = corrected[(arm, seed)][view.private.view_id]
                for alpha in cfg["alphas"]:
                    states.append(
                        (
                            view,
                            arm,
                            seed,
                            alpha,
                            observed + alpha * (endpoint - observed),
                        )
                    )

    value_rows = []
    for state_index, (view, arm, seed, alpha, values) in enumerate(states):
        converged = solve_huber_irls(
            view.public,
            values=values,
            base_weights=None,
            delta=graph["huber_delta"],
            max_iterations=graph["irls_iterations"],
            damping=graph["irls_damping"],
            weight_floor=graph["weight_floor"],
        )
        for depth in cfg["depths"]:
            nx, nw, _, no, nc = numpy_fixed_irls(
                view.public,
                values,
                steps=depth,
                delta=graph["huber_delta"],
                damping=graph["irls_damping"],
                weight_floor=graph["weight_floor"],
            )
            tensor = torch.tensor(values, dtype=torch.float64)
            tout = differentiable_huber_irls_fixed(
                view.public,
                tensor,
                steps=depth,
                delta=graph["huber_delta"],
                damping=graph["irls_damping"],
                weight_floor=graph["weight_floor"],
            )
            tx, tw = (
                tout.x_hat.detach().numpy(),
                tout.normalized_weights.detach().numpy(),
            )
            row = {
                "view_id": view.private.view_id,
                "master_id": view.private.master_id,
                "n_nodes": view.public.n_nodes,
                "source": arm,
                "seed": seed,
                "alpha": alpha,
                "depth": depth,
                "executor_converged": converged.converged,
                "executor_iterations": converged.iterations,
                "fixed_x_rmse": float(np.sqrt(np.mean((tx - nx) ** 2))),
                "fixed_x_max_abs": float(np.max(np.abs(tx - nx))),
                "fixed_weight_rmse": float(np.sqrt(np.mean((tw - nw) ** 2))),
                "fixed_weight_max_abs": float(np.max(np.abs(tw - nw))),
                "fixed_objective_abs": float(abs(float(tout.huber_objective) - no)),
                "numpy_fixed_objective": no,
                "numpy_final_solution_change": nc,
                "torch_final_solution_change": float(tout.final_solution_change),
            }
            if converged.converged:
                row.update(
                    {
                        "converged_x_rmse": float(
                            np.sqrt(np.mean((tx - converged.x_hat) ** 2))
                        ),
                        "converged_x_max_abs": float(
                            np.max(np.abs(tx - converged.x_hat))
                        ),
                        "converged_weight_rmse": float(
                            np.sqrt(np.mean((tw - converged.weights) ** 2))
                        ),
                        "converged_weight_max_abs": float(
                            np.max(np.abs(tw - converged.weights))
                        ),
                    }
                )
            else:
                row.update(
                    {
                        k: None
                        for k in (
                            "converged_x_rmse",
                            "converged_x_max_abs",
                            "converged_weight_rmse",
                            "converged_weight_max_abs",
                        )
                    }
                )
            value_rows.append(row)
        if (state_index + 1) % cfg["runtime"]["sample_every_states"] == 0:
            sample = resource_sample(started, f"value_state_{state_index + 1}")
            samples.append(sample)
            enforce(sample, cfg)

    gradient_rows = []
    coordinate_rows = []
    arm_seeds = [(arm, seed) for arm in cfg["arms"] for seed in cfg["seeds"]]
    for probe_index, view in enumerate(probe_views):
        arm, seed = arm_seeds[probe_index % len(arm_seeds)]
        observed = view.public.observed_log_ratio.astype(np.float64)
        endpoint = corrected[(arm, seed)][view.private.view_id]
        valid = np.asarray(view.public.edge_valid, dtype=bool)
        for alpha in cfg["gradient_alphas"]:
            values = observed + alpha * (endpoint - observed)
            target = np.asarray(view.private.x_true, dtype=np.float64)
            for depth in cfg["gradient_depths"]:
                tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                tout = differentiable_huber_irls_fixed(
                    view.public,
                    tensor,
                    steps=depth,
                    delta=graph["huber_delta"],
                    damping=graph["irls_damping"],
                    weight_floor=graph["weight_floor"],
                )
                loss = torch.mean(
                    (tout.x_hat - torch.tensor(target, dtype=torch.float64)) ** 2
                )
                loss.backward()
                auto = tensor.grad.detach().numpy()
                numeric = np.full_like(values, np.nan)
                stable = valid & (
                    tout.min_huber_margin.detach().numpy()
                    > cfg["kink_step_multiplier"]
                    * cfg["finite_difference_step"]
                    * np.maximum(1.0, np.abs(values))
                )
                for edge in np.flatnonzero(stable):
                    h = cfg["finite_difference_step"] * max(1.0, abs(values[edge]))
                    losses = []
                    for sign in (-1.0, 1.0):
                        perturbed = values.copy()
                        perturbed[edge] += sign * h
                        x, _, _, _, _ = numpy_fixed_irls(
                            view.public,
                            perturbed,
                            steps=depth,
                            delta=graph["huber_delta"],
                            damping=graph["irls_damping"],
                            weight_floor=graph["weight_floor"],
                        )
                        losses.append(float(np.mean((x - target) ** 2)))
                    numeric[edge] = (losses[1] - losses[0]) / (2.0 * h)
                mask = np.isfinite(numeric)
                half_edges = np.flatnonzero(mask)[: cfg["stability_coordinates"]]
                half_numeric = []
                for edge in half_edges:
                    h = (
                        0.5
                        * cfg["finite_difference_step"]
                        * max(1.0, abs(values[edge]))
                    )
                    half_losses = []
                    for sign in (-1.0, 1.0):
                        perturbed = values.copy()
                        perturbed[edge] += sign * h
                        x, _, _, _, _ = numpy_fixed_irls(
                            view.public,
                            perturbed,
                            steps=depth,
                            delta=graph["huber_delta"],
                            damping=graph["irls_damping"],
                            weight_floor=graph["weight_floor"],
                        )
                        half_losses.append(float(np.mean((x - target) ** 2)))
                    half_numeric.append((half_losses[1] - half_losses[0]) / (2.0 * h))
                half_numeric = np.asarray(half_numeric)
                full_numeric = numeric[half_edges]
                half_by_edge = {
                    int(edge): float(value)
                    for edge, value in zip(half_edges, half_numeric)
                }
                denom = np.linalg.norm(auto[mask]) * np.linalg.norm(numeric[mask])
                cosine = (
                    float(np.dot(auto[mask], numeric[mask]) / denom) if denom else None
                )
                relative = (
                    float(
                        np.linalg.norm(auto[mask] - numeric[mask])
                        / max(np.linalg.norm(numeric[mask]), 1e-15)
                    )
                    if np.any(mask)
                    else None
                )
                sign_reversals = int(
                    np.sum(
                        (np.sign(auto[mask]) != np.sign(numeric[mask]))
                        & (
                            np.abs(numeric[mask])
                            >= cfg["conformity"]["stable_gradient_magnitude"]
                        )
                    )
                )
                gradient_rows.append(
                    {
                        "view_id": view.private.view_id,
                        "arm": arm,
                        "seed": seed,
                        "alpha": alpha,
                        "depth": depth,
                        "valid_coordinates": int(valid.sum()),
                        "stable_coordinates": int(mask.sum()),
                        "cosine": cosine,
                        "relative_l2": relative,
                        "max_abs": (
                            float(np.max(np.abs(auto[mask] - numeric[mask])))
                            if np.any(mask)
                            else None
                        ),
                        "half_step_coordinates": len(half_edges),
                        "half_step_relative_l2": (
                            float(
                                np.linalg.norm(full_numeric - half_numeric)
                                / max(np.linalg.norm(half_numeric), 1e-15)
                            )
                            if len(half_edges)
                            else None
                        ),
                        "half_step_max_abs": (
                            float(np.max(np.abs(full_numeric - half_numeric)))
                            if len(half_edges)
                            else None
                        ),
                        "sign_reversals": sign_reversals,
                    }
                )
                margins = tout.min_huber_margin.detach().numpy()
                for edge in np.flatnonzero(valid):
                    h = cfg["finite_difference_step"] * max(1.0, abs(values[edge]))
                    coordinate_rows.append(
                        {
                            "view_id": view.private.view_id,
                            "arm": arm,
                            "seed": seed,
                            "alpha": alpha,
                            "depth": depth,
                            "edge": int(edge),
                            "finite_difference_step": h,
                            "huber_margin": float(margins[edge]),
                            "stable": bool(mask[edge]),
                            "exclusion_reason": (
                                None if mask[edge] else "near_huber_kink"
                            ),
                            "autograd": float(auto[edge]),
                            "central_difference": (
                                float(numeric[edge]) if mask[edge] else None
                            ),
                            "half_step_difference": half_by_edge.get(int(edge)),
                        }
                    )
        sample = resource_sample(started, f"gradient_probe_{probe_index + 1}")
        samples.append(sample)
        enforce(sample, cfg)

    summary = summarize(value_rows, gradient_rows, cfg)
    write_json(output / "resolved_config.json", cfg)
    write_json(output / "value_fidelity.json", value_rows)
    write_json(output / "gradient_fidelity.json", gradient_rows)
    write_json(output / "gradient_coordinates.json", coordinate_rows)
    write_json(output / "summary.json", summary)
    samples.append(resource_sample(started, "complete"))
    write_json(output / "resource_samples.json", samples)
    save_npz(
        output / "selected_views.npz",
        {
            "value_path_view_id": np.asarray([v.private.view_id for v in path_views]),
            "gradient_view_id": np.asarray([v.private.view_id for v in probe_views]),
        },
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / p)}' \"$repo/{p}\" | sha256sum -c -"
        for p in SOURCES
    )
    replay = f"""#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || {{ echo "wrong git HEAD" >&2; exit 2; }}
[ -z "$(git -C "$repo" status --porcelain)" ] || {{ echo "dirty worktree" >&2; exit 2; }}
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR to a new path}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_irls_surrogate_fidelity.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_irls_surrogate_fidelity_v1.json" --output "$OUTPUT_DIR"
"""
    (output / "replay.sh").write_text(replay)
    (output / "replay.sh").chmod(0o755)
    deterministic = sorted(
        p
        for p in output.rglob("*")
        if p.is_file()
        and p.name
        not in {"manifest.json", "runtime_observation.json", "resource_samples.json"}
    )
    manifest = {
        "schema_version": cfg["schema_version"],
        "git_head": head,
        "source_hashes": {p: sha(ROOT / p) for p in SOURCES},
        "checkpoint_hashes": {
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted((source / "checkpoints").glob("*.npz"))
            if any(p.name.startswith(a + "|") for a in cfg["arms"])
        },
        "raw_eval_hashes": {
            str(
                (source / "raw_eval" / f"{arm}|seed={seed}.npz").relative_to(ROOT)
            ): sha(source / "raw_eval" / f"{arm}|seed={seed}.npz")
            for arm in cfg["arms"]
            for seed in cfg["seeds"]
        },
        "deterministic_files": {
            str(p.relative_to(output)): sha(p) for p in deterministic
        },
        "runtime_exclusions": ["resource_samples.json", "runtime_observation.json"],
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "runtime_observation.json", samples[-1])
    print(json.dumps({"summary": summary, "runtime": samples[-1]}, sort_keys=True))


def summarize(
    values: list[dict[str, Any]], gradients: list[dict[str, Any]], cfg: dict[str, Any]
) -> dict[str, Any]:
    result = {"by_depth": {}, "selected_depth": None, "status": "NO_CONFORMING_DEPTH"}
    limits = cfg["conformity"]
    for depth in cfg["depths"]:
        vr = [r for r in values if r["depth"] == depth]
        gr = [r for r in gradients if r["depth"] == depth and r["cosine"] is not None]
        converged = [r for r in vr if r["executor_converged"]]
        fixed_max = max(
            max(
                r["fixed_x_max_abs"],
                r["fixed_weight_max_abs"],
                r["fixed_objective_abs"],
            )
            for r in vr
        )
        x_rmse = np.asarray([r["converged_x_rmse"] for r in converged])
        summary = {
            "states": len(vr),
            "executor_converged_states": len(converged),
            "fixed_torch_numpy_max_error": fixed_max,
            "converged_x_rmse_p99": float(np.percentile(x_rmse, 99)),
            "converged_x_rmse_max": float(np.max(x_rmse)),
            "gradient_probes": len(gr),
            "gradient_cosine_median": (
                float(np.median([r["cosine"] for r in gr])) if gr else None
            ),
            "gradient_relative_l2_p95": (
                float(np.percentile([r["relative_l2"] for r in gr], 95)) if gr else None
            ),
            "gradient_sign_reversals": sum(r["sign_reversals"] for r in gr),
        }
        conforms = (
            depth in cfg["gradient_depths"]
            and fixed_max <= limits["fixed_torch_numpy_max_error"]
            and summary["converged_x_rmse_p99"] <= limits["converged_p99_x_rmse"]
            and summary["converged_x_rmse_max"] <= limits["converged_max_x_rmse"]
            and summary["gradient_cosine_median"] >= limits["gradient_median_cosine"]
            and summary["gradient_relative_l2_p95"]
            <= limits["gradient_p95_relative_l2"]
            and summary["gradient_sign_reversals"] == 0
        )
        summary["conforms"] = bool(conforms)
        result["by_depth"][str(depth)] = summary
        if conforms and result["selected_depth"] is None:
            result.update({"selected_depth": depth, "status": "CONFORMING_DEPTH_FOUND"})
    return result


if __name__ == "__main__":
    main()
