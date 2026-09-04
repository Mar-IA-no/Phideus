#!/usr/bin/env python3
"""Compare local-relation and post-IRLS losses on frozen graph trunks, CPU-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/geometria_proporcional"))

import run_proportional_graph_frozen_adapters as frozen  # noqa: E402
import run_proportional_graph_neural_smoke as smoke  # noqa: E402
from geometria_proporcional.proportional_graph_contract import (  # noqa: E402
    ProportionalGraphConfig,
    generate_graph_views,
)
from geometria_proporcional.proportional_graph_neural import (  # noqa: E402
    differentiable_huber_irls_fixed,
    local_closure_loss,
)

DEFAULT_CONFIG = (
    ROOT
    / "experiments/geometria_proporcional/configs/proportional_graph_irls_loss_contrast_v1.json"
)
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_IRLS_LOSS_CONTRAST_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_irls_loss_contrast_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_irls_loss_contrast.py",
    "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py",
    "experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py",
    "src/geometria_proporcional/proportional_graph_contract.py",
    "src/geometria_proporcional/proportional_graph_neural.py",
)


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    expected = {
        "schema_version",
        "source_smoke",
        "source_frozen_adapters",
        "source_config",
        "arms",
        "seeds",
        "surrogate",
        "training",
        "bootstrap_replicates",
        "bootstrap_seed",
    }
    if (
        set(cfg) != expected
        or cfg["schema_version"] != "proportional-graph-irls-loss-contrast-v1"
    ):
        raise ValueError("invalid IRLS loss contrast schema")
    if cfg["arms"] != ["raw_generic", "raw_typed", "closure_generic", "closure_typed"]:
        raise ValueError("arms must remain frozen")
    if cfg["seeds"] != [104729, 130363] or cfg["surrogate"]["steps"] != 64:
        raise ValueError("seed and surrogate contracts must remain frozen")
    training = cfg["training"]
    if (
        training["torch_threads"] != 1
        or training["epochs"] != 5
        or training["batch_size"] != 64
    ):
        raise ValueError("CPU schedule must remain frozen")
    if not 0 < training["scale_min"] < training["scale_max"]:
        raise ValueError("invalid gradient-scale bounds")
    return cfg


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sample_resource(started: float, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }


def enforce(sample: dict[str, Any], cfg: dict[str, Any]) -> None:
    training = cfg["training"]
    if sample["elapsed_seconds"] > training["max_seconds"]:
        raise RuntimeError("runtime budget exceeded")
    if sample["max_rss_gib"] > training["max_rss_gib"]:
        raise RuntimeError("RSS budget exceeded")


def configure_head(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("correction_head."))
        parameter.grad = None


def objective(
    model: torch.nn.Module,
    view: Any,
    arm: dict[str, Any],
    seed: int,
    cfg: dict[str, Any],
    mode: str,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    output = model(smoke._view_tensors(view, arm, seed))
    corrected = output.corrected_log_ratio
    clean = torch.as_tensor(view.private.clean_log_ratio, dtype=corrected.dtype)
    relation = torch.mean((corrected - clean) ** 2)
    closure = local_closure_loss(view.public, corrected)
    quotient: torch.Tensor | None = None
    if mode == "local_relation":
        total = relation + cfg["training"]["closure_l1"] * closure
    elif mode == "post_irls":
        graph = cfg["source"]["graph"]
        solved = differentiable_huber_irls_fixed(
            view.public,
            corrected,
            steps=cfg["surrogate"]["steps"],
            delta=graph["huber_delta"],
            damping=graph["irls_damping"],
            weight_floor=graph["weight_floor"],
        )
        target = torch.as_tensor(view.private.x_true, dtype=corrected.dtype)
        quotient = torch.mean((solved.x_hat - target) ** 2)
        total = quotient + cfg["training"]["closure_l1"] * closure
    else:
        raise ValueError(mode)
    return total, {
        "total": float(total.detach()),
        "relation_mse": float(relation.detach()),
        "closure_l1": float(closure.detach()),
        "quotient_mse": float(quotient.detach()) if quotient is not None else None,
    }


def gradient_norm(model: torch.nn.Module) -> float:
    squared = sum(
        float(torch.sum(parameter.grad.detach() ** 2))
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    return float(np.sqrt(squared))


def batch_order(n: int, arm: str, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(
        smoke._stable_seed(seed, arm, epoch, "irls_loss_shared_order")
    )
    return rng.permutation(n)


def probe_scale(
    models: dict[str, torch.nn.Module],
    views: list[Any],
    arm: dict[str, Any],
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    training = cfg["training"]
    order = batch_order(len(views), arm["name"], seed, 0)
    norms: dict[str, list[float]] = {mode: [] for mode in models}
    losses: dict[str, list[float]] = {mode: [] for mode in models}
    for mode, model in models.items():
        configure_head(model)
        model.train()
        for batch_index in range(training["scale_probe_batches"]):
            indices = order[
                batch_index
                * training["batch_size"] : (batch_index + 1)
                * training["batch_size"]
            ]
            model.zero_grad(set_to_none=True)
            local_losses = []
            for index in indices:
                loss, _ = objective(model, views[int(index)], arm, seed, cfg, mode)
                (loss / len(indices)).backward()
                local_losses.append(float(loss.detach()))
            norms[mode].append(gradient_norm(model))
            losses[mode].append(float(np.mean(local_losses)))
        model.zero_grad(set_to_none=True)
    local_median = float(np.median(norms["local_relation"]))
    post_median = float(np.median(norms["post_irls"]))
    if (
        not np.isfinite(local_median)
        or not np.isfinite(post_median)
        or post_median <= 0
    ):
        raise RuntimeError("non-finite initial gradient scale")
    multiplier = local_median / post_median
    if not training["scale_min"] <= multiplier <= training["scale_max"]:
        raise RuntimeError(f"gradient multiplier out of bounds: {multiplier}")
    return {
        "gradient_norms": norms,
        "mean_batch_losses": losses,
        "median_local_gradient_norm": local_median,
        "median_post_gradient_norm": post_median,
        "post_irls_multiplier": multiplier,
    }


def train(
    model: torch.nn.Module,
    views: list[Any],
    arm: dict[str, Any],
    seed: int,
    cfg: dict[str, Any],
    mode: str,
    multiplier: float,
    started: float,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    configure_head(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    training = cfg["training"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    history = []
    for epoch in range(training["epochs"]):
        order = batch_order(len(views), arm["name"], seed, epoch)
        records: dict[str, list[float]] = defaultdict(list)
        for start in range(0, len(order), training["batch_size"]):
            indices = order[start : start + training["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            batch_total = []
            for index in indices:
                loss, components = objective(
                    model, views[int(index)], arm, seed, cfg, mode
                )
                (multiplier * loss / len(indices)).backward()
                batch_total.append(components["total"])
                for key in ("relation_mse", "closure_l1", "quotient_mse"):
                    if components[key] is not None:
                        records[key].append(float(components[key]))
            records["raw_total"].append(float(np.mean(batch_total)))
            records["gradient_norm_preclip"].append(gradient_norm(model))
            torch.nn.utils.clip_grad_norm_(parameters, training["grad_clip"])
            optimizer.step()
        row = {"epoch": epoch + 1, "multiplier": multiplier}
        row.update(
            {f"mean_{key}": float(np.mean(values)) for key, values in records.items()}
        )
        history.append(row)
        sample = sample_resource(
            started, f"{arm['name']}|{seed}|{mode}|epoch={epoch + 1}"
        )
        samples.append(sample)
        enforce(sample, cfg)
    frozen_gradients = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if frozen_gradients:
        raise AssertionError(f"frozen gradients: {frozen_gradients}")
    return history


def historical_relations(source: Path, arm: str, seed: int) -> dict[str, np.ndarray]:
    with np.load(
        source / "raw_eval" / f"{arm}|seed={seed}.npz", allow_pickle=False
    ) as saved:
        return {
            str(view_id): saved["irls_corrected_relation"][
                saved["edge_offset"][i] : saved["edge_offset"][i + 1]
            ].astype(np.float64)
            for i, view_id in enumerate(saved["view_id"])
        }


def evaluate(
    models: dict[str, torch.nn.Module],
    historical: dict[str, np.ndarray],
    views: list[Any],
    arm: dict[str, Any],
    seed: int,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    for model in models.values():
        model.eval()
    rows, view_ids, counts = [], [], []
    corrected: dict[str, list[np.ndarray]] = {mode: [] for mode in models}
    with torch.no_grad():
        for view in views:
            tensors = smoke._view_tensors(view, arm, seed)
            outputs = {
                mode: model(tensors).corrected_log_ratio.cpu().numpy()
                for mode, model in models.items()
            }
            variants = {
                "observed_unit": np.asarray(view.public.observed_log_ratio),
                "local_relation": outputs["local_relation"],
                "post_irls": outputs["post_irls"],
                "historical_local": historical[view.private.view_id],
            }
            unit = np.ones(len(view.public.observed_log_ratio), dtype=np.float64)
            for variant, relation in variants.items():
                metrics, _ = smoke._solver_metrics(
                    view, np.asarray(relation, dtype=np.float64), unit, cfg["source"]
                )
                rows.append(
                    {
                        "arm": arm["name"],
                        "seed": seed,
                        "variant": variant,
                        **{
                            k: frozen.finite(v) if isinstance(v, float) else v
                            for k, v in metrics.items()
                        },
                    }
                )
            view_ids.append(view.private.view_id)
            counts.append(len(view.public.observed_log_ratio))
            for mode in models:
                corrected[mode].append(outputs[mode])
    packed = {
        "view_id": np.asarray(view_ids),
        "edge_count": np.asarray(counts, dtype=np.int64),
        "edge_offset": np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
    }
    for mode, arrays in corrected.items():
        packed[f"{mode}_corrected_relation"] = np.concatenate(arrays).astype(np.float32)
    return rows, packed


def effects(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    indices: np.ndarray,
    masters: np.ndarray,
) -> dict[str, Any]:
    comparisons = (
        ("post_irls", "local_relation"),
        ("post_irls", "observed_unit"),
        ("local_relation", "observed_unit"),
        ("historical_local", "observed_unit"),
    )
    metrics = ("irls_quotient_rmse", "relation_rmse")
    result: dict[str, Any] = {}
    master_ids = list(map(str, masters))
    for arm in cfg["arms"]:
        result[arm] = {}
        for mechanism in ("iid", "grouped"):
            subset = [
                r
                for r in rows
                if r["arm"] == arm
                and r["split"] == "test"
                and r["mechanism"] == mechanism
            ]
            result[arm][f"test_{mechanism}"] = {}
            for positive, baseline in comparisons:
                for metric in metrics:
                    by: dict[tuple[str, int], dict[str, float | None]] = defaultdict(
                        dict
                    )
                    for row in subset:
                        by[(row["master_id"], row["seed"])][row["variant"]] = row[
                            metric
                        ]
                    deltas, valid = [], True
                    for master in master_ids:
                        seed_deltas = []
                        for seed in cfg["seeds"]:
                            pair = by[(master, seed)]
                            if pair.get(positive) is None or pair.get(baseline) is None:
                                valid = False
                                break
                            seed_deltas.append(
                                float(pair[positive]) - float(pair[baseline])
                            )
                        if not valid:
                            break
                        deltas.append(float(np.mean(seed_deltas)))
                    key = f"{positive}_minus_{baseline}|{metric}"
                    if not valid:
                        result[arm][f"test_{mechanism}"][key] = {
                            "status": "NOT_EVALUABLE",
                            "n_masters": len(master_ids),
                        }
                    else:
                        data = np.asarray(deltas)
                        draws = data[indices].mean(axis=1)
                        result[arm][f"test_{mechanism}"][key] = {
                            "status": "EVALUABLE",
                            "n_masters": len(data),
                            "mean_delta": float(data.mean()),
                            "ci95": [
                                float(np.percentile(draws, 2.5)),
                                float(np.percentile(draws, 97.5)),
                            ],
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
    torch.set_num_threads(cfg["training"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    samples = [sample_resource(started, "start")]
    source_cfg = smoke._load_config((ROOT / cfg["source_config"]).resolve())
    cfg["source"] = source_cfg
    views = generate_graph_views(ProportionalGraphConfig.from_dict(source_cfg["graph"]))
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
    views = [v for v in views if eligible[v.private.master_id]]
    train_views = [v for v in views if v.private.split == "train"]
    eval_views = [v for v in views if v.private.split in {"validation", "test"}]
    scale = smoke._input_scale(train_views)
    with np.load(
        ROOT / cfg["source_smoke"] / "bootstrap_indices.npz", allow_pickle=False
    ) as saved:
        bootstrap_masters = saved["master_id"].copy()
        bootstrap_indices = saved["complete_n_252"].copy()
    frozen.save_npz(
        output / "bootstrap_indices.npz",
        {"master_id": bootstrap_masters, "complete_n_252": bootstrap_indices},
    )
    all_rows, histories, scales, freeze = [], {}, {}, {}
    source_historical = ROOT / cfg["source_frozen_adapters"]
    for arm_name in cfg["arms"]:
        arm = next(a for a in source_cfg["arms"] if a["name"] == arm_name)
        for seed in cfg["seeds"]:
            key = f"{arm_name}|seed={seed}"
            checkpoint = ROOT / cfg["source_smoke"] / "checkpoints" / f"{key}.npz"
            models = {}
            metadata = None
            for mode in ("local_relation", "post_irls"):
                model = smoke._model_for_arm(arm, source_cfg, scale)
                metadata = smoke.load_checkpoint(checkpoint, model)
                models[mode] = model
            if not np.isclose(metadata["input_scale"], scale, rtol=0.0, atol=1e-12):
                raise AssertionError("checkpoint input scale mismatch")
            for left, right in zip(
                models["local_relation"].parameters(), models["post_irls"].parameters()
            ):
                if not torch.equal(left, right):
                    raise AssertionError("loss arms do not share initialization")
            scale_record = probe_scale(models, train_views, arm, seed, cfg)
            scales[key] = scale_record
            histories[key] = {}
            for mode, model in models.items():
                multiplier = (
                    scale_record["post_irls_multiplier"] if mode == "post_irls" else 1.0
                )
                histories[key][mode] = train(
                    model,
                    train_views,
                    arm,
                    seed,
                    cfg,
                    mode,
                    multiplier,
                    started,
                    samples,
                )
                frozen.save_adapter(
                    output / "checkpoints" / f"{key}|{mode}.npz",
                    model,
                    {
                        "arm": arm_name,
                        "seed": seed,
                        "mode": mode,
                        "last_epoch": cfg["training"]["epochs"],
                        "loss_multiplier": multiplier,
                    },
                )
            freeze[key] = {
                "initial_state_equal": True,
                "modes": {
                    mode: {
                        "trainable_parameters": sum(
                            p.numel() for p in model.parameters() if p.requires_grad
                        ),
                        "frozen_parameters": sum(
                            p.numel() for p in model.parameters() if not p.requires_grad
                        ),
                        "trainable_names": [
                            name
                            for name, parameter in model.named_parameters()
                            if parameter.requires_grad
                        ],
                    }
                    for mode, model in models.items()
                },
            }
            historical = historical_relations(source_historical, arm_name, seed)
            rows, packed = evaluate(models, historical, eval_views, arm, seed, cfg)
            all_rows.extend(rows)
            frozen.save_npz(output / "raw_eval" / f"{key}.npz", packed)
            sample = sample_resource(started, f"evaluated|{key}")
            samples.append(sample)
            enforce(sample, cfg)
    write_json(output / "resolved_config.json", cfg)
    write_json(output / "scale_control.json", scales)
    write_json(output / "histories.json", histories)
    write_json(output / "freeze_contract.json", freeze)
    write_json(output / "metrics.json", all_rows)
    write_json(
        output / "effects.json",
        effects(all_rows, cfg, bootstrap_indices, bootstrap_masters),
    )
    write_json(
        output / "resource_samples.json",
        samples + [sample_resource(started, "complete")],
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    smoke_source = ROOT / cfg["source_smoke"]
    input_paths = [
        *(sorted((smoke_source / "checkpoints").glob("*.npz"))),
        *(sorted((source_historical / "raw_eval").glob("*.npz"))),
        smoke_source / "bootstrap_indices.npz",
    ]
    input_paths = [
        p
        for p in input_paths
        if p.name == "bootstrap_indices.npz"
        or any(p.name.startswith(arm + "|") for arm in cfg["arms"])
    ]
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT / p)}' \"$repo/{p}\" | sha256sum -c -"
        for p in SOURCE_FILES
    )
    checks += "\n" + "\n".join(
        f"printf '%s  %s\\n' '{sha(p)}' \"$repo/{p.relative_to(ROOT)}\" | sha256sum -c -"
        for p in input_paths
    )
    replay = f"""#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || {{ echo "wrong git HEAD" >&2; exit 2; }}
[ -z "$(git -C "$repo" status --porcelain)" ] || {{ echo "dirty worktree" >&2; exit 2; }}
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR to a new path}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_irls_loss_contrast.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_irls_loss_contrast_v1.json" --output "$OUTPUT_DIR"
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
        "source_hashes": {p: sha(ROOT / p) for p in SOURCE_FILES},
        "source_checkpoint_hashes": {
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted((smoke_source / "checkpoints").glob("*.npz"))
            if any(p.name.startswith(a + "|") for a in cfg["arms"])
        },
        "historical_raw_hashes": {
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted((source_historical / "raw_eval").glob("*.npz"))
            if any(p.name.startswith(a + "|") for a in cfg["arms"])
        },
        "bootstrap_hash": {
            str((smoke_source / "bootstrap_indices.npz").relative_to(ROOT)): sha(
                smoke_source / "bootstrap_indices.npz"
            )
        },
        "deterministic_files": {
            str(p.relative_to(output)): sha(p) for p in deterministic
        },
        "runtime_exclusions": ["resource_samples.json", "runtime_observation.json"],
    }
    write_json(output / "manifest.json", manifest)
    runtime = sample_resource(started, "complete")
    write_json(output / "runtime_observation.json", runtime)
    print(json.dumps(runtime, sort_keys=True))


if __name__ == "__main__":
    main()
