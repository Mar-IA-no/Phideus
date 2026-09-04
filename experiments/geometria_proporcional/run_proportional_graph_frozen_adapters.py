#!/usr/bin/env python3
"""Refit solver-typed output heads on frozen proportional graph trunks, CPU-only."""

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
from collections import defaultdict
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
)
from geometria_proporcional.proportional_graph_neural import (  # noqa: E402
    differentiable_wls,
    local_closure_loss,
)

DEFAULT_CONFIG = ROOT / "experiments/geometria_proporcional/configs/proportional_graph_frozen_adapters_v1.json"
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_FROZEN_ADAPTERS_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_frozen_adapters_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py",
    "experiments/geometria_proporcional/run_proportional_graph_neural_smoke.py",
    "src/geometria_proporcional/proportional_graph_neural.py",
)


def canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write compressed arrays with fixed ZIP metadata for byte-exact replay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in sorted(arrays.items()):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def save_adapter(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    arrays = {"format_version": np.asarray("proportional-frozen-adapter-v1"),
              "metadata_json": np.asarray(canonical(metadata))}
    arrays.update({f"model::{name}": tensor.detach().cpu().numpy()
                   for name, tensor in sorted(model.state_dict().items())})
    save_npz(path, arrays)


def finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def budget(started: float, cfg: dict[str, Any], stage: str) -> None:
    elapsed = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    if elapsed > cfg["training"]["max_seconds"] or rss > cfg["training"]["max_rss_gib"]:
        raise RuntimeError(f"budget exceeded at {stage}: {elapsed:.1f}s, {rss:.3f} GiB")


def train(model: torch.nn.Module, views: list[Any], arm: dict[str, Any], seed: int,
          cfg: dict[str, Any], kind: str, started: float) -> list[dict[str, float]]:
    prefix = "reliability_head." if kind == "wls_weight" else "correction_head."
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefix))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    history = []
    for epoch in range(cfg["training"]["epochs"]):
        order = np.random.default_rng(smoke._stable_seed(seed, epoch, kind, "order")).permutation(len(views))
        losses = []
        for start in range(0, len(order), cfg["training"]["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            batch = []
            for index in order[start:start + cfg["training"]["batch_size"]]:
                view = views[int(index)]
                output = model(smoke._view_tensors(view, arm, seed))
                if kind == "wls_weight":
                    observed = torch.as_tensor(view.public.observed_log_ratio, dtype=output.reliability.dtype)
                    x_hat = differentiable_wls(view.public, observed, output.reliability)
                    target = torch.as_tensor(view.private.x_true, dtype=x_hat.dtype)
                    loss = torch.mean((x_hat - target) ** 2)
                else:
                    clean = torch.as_tensor(view.private.clean_log_ratio, dtype=output.corrected_log_ratio.dtype)
                    loss = torch.mean((output.corrected_log_ratio - clean) ** 2)
                    loss = loss + cfg["irls_relation_loss"]["closure_l1"] * local_closure_loss(
                        view.public, output.corrected_log_ratio
                    )
                batch.append(loss)
            joined = torch.stack(batch).mean()
            joined.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["training"]["grad_clip"])
            optimizer.step()
            losses.append(float(joined.detach()))
            budget(started, cfg, f"{arm['name']} {seed} {kind} epoch {epoch + 1}")
        history.append({"epoch": epoch + 1, "mean_batch_loss": float(np.mean(losses))})
    frozen_with_grad = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    if frozen_with_grad:
        raise AssertionError(f"frozen gradients: {frozen_with_grad}")
    return history


def evaluate(models: dict[str, torch.nn.Module], inherited: torch.nn.Module,
             views: list[Any], arm: dict[str, Any], seed: int, cfg: dict[str, Any],
             started: float) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows, arrays = [], defaultdict(list)
    for model in [inherited, *models.values()]:
        model.eval()
    with torch.no_grad():
        for view in views:
            tensors = smoke._view_tensors(view, arm, seed)
            base = inherited(tensors)
            wout = models["wls_weight"](tensors)
            iout = models["irls_relation"](tensors)
            variants = {
                "wls_static": (view.public.observed_log_ratio, base.reliability.cpu().numpy()),
                "wls_adapter": (view.public.observed_log_ratio, wout.reliability.cpu().numpy()),
                "irls_static": (view.public.observed_log_ratio, np.ones(len(view.public.observed_log_ratio))),
                "irls_adapter": (iout.corrected_log_ratio.cpu().numpy(), np.ones(len(view.public.observed_log_ratio))),
                "delivered": (base.corrected_log_ratio.cpu().numpy(), base.reliability.cpu().numpy()),
            }
            for variant, (relation, weights) in variants.items():
                metrics, _ = smoke._solver_metrics(view, np.asarray(relation, dtype=np.float64),
                                                   np.asarray(weights, dtype=np.float64), cfg["source"])
                rows.append({"arm": arm["name"], "seed": seed, "variant": variant,
                             **{k: finite(v) if isinstance(v, float) else v for k, v in metrics.items()}})
            arrays["view_id"].append(view.private.view_id)
            arrays["edge_count"].append(len(view.public.observed_log_ratio))
            arrays["wls_reliability"].append(wout.reliability.cpu().numpy())
            arrays["irls_corrected_relation"].append(iout.corrected_log_ratio.cpu().numpy())
            budget(started, cfg, f"evaluate {arm['name']} {seed}")
    packed = {"view_id": np.asarray(arrays["view_id"]), "edge_count": np.asarray(arrays["edge_count"], dtype=np.int64)}
    for key in ("wls_reliability", "irls_corrected_relation"):
        packed[key] = np.concatenate(arrays[key]).astype(np.float32)
    packed["edge_offset"] = np.concatenate(([0], np.cumsum(packed["edge_count"]))).astype(np.int64)
    return rows, packed


def effects(rows: list[dict[str, Any]], cfg: dict[str, Any],
            bootstrap_indices: np.ndarray | None = None,
            bootstrap_master_ids: np.ndarray | None = None) -> dict[str, Any]:
    comparisons = (("wls_adapter", "wls_static", "wls_quotient_rmse"),
                   ("irls_adapter", "irls_static", "irls_quotient_rmse"))
    result = {}
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    for arm in cfg["arms"]:
        result[arm] = {}
        for split, mechanism in (("test_iid", "iid"), ("test_grouped", "grouped")):
            result[arm][split] = {}
            subset = [r for r in rows if r["arm"] == arm and r["split"] == "test" and r["mechanism"] == mechanism]
            for positive, baseline, metric in comparisons:
                by = defaultdict(dict)
                for row in subset:
                    by[(row["master_id"], row["seed"])][row["variant"]] = row[metric]
                masters = (list(map(str, bootstrap_master_ids)) if bootstrap_master_ids is not None
                           else sorted({m for m, _ in by}))
                values = []
                valid = True
                for master in masters:
                    seed_deltas = []
                    for seed in cfg["seeds"]:
                        pair = by[(master, seed)]
                        if pair.get(positive) is None or pair.get(baseline) is None:
                            valid = False
                            break
                        seed_deltas.append(pair[positive] - pair[baseline])
                    if not valid:
                        break
                    values.append(float(np.mean(seed_deltas)))
                key = f"{positive}_minus_{baseline}"
                if not valid or not values:
                    result[arm][split][key] = {"status": "NOT_EVALUABLE", "n_masters": len(masters)}
                else:
                    data = np.asarray(values)
                    indices = (bootstrap_indices if bootstrap_indices is not None
                               else rng.integers(0, len(data), size=(cfg["bootstrap_replicates"], len(data))))
                    if indices.shape[1] != len(data):
                        raise AssertionError("bootstrap/master cardinality mismatch")
                    draws = data[indices].mean(1)
                    result[arm][split][key] = {"status": "EVALUABLE", "n_masters": len(data),
                        "mean_delta": float(data.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not args.development and (subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout):
        raise RuntimeError("official run requires clean worktree")
    output.mkdir(parents=True)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    source_cfg = smoke._load_config((ROOT / cfg["source_config"]).resolve())
    cfg["source"] = source_cfg
    views = generate_graph_views(ProportionalGraphConfig.from_dict(source_cfg["graph"]))
    shuffle_arm = next(a for a in source_cfg["arms"] if a["name"] == "closure_typed_path_shuffle")
    eligible = {v.private.master_id: bool(smoke._view_tensors(v, shuffle_arm, cfg["seeds"][0])["path_shuffle_eligible"]) for v in views}
    views = [v for v in views if eligible[v.private.master_id]]
    train_views = [v for v in views if v.private.split == "train"]
    eval_views = [v for v in views if v.private.split in {"validation", "test"}]
    scale = smoke._input_scale(train_views)
    with np.load(ROOT / cfg["source_smoke"] / "bootstrap_indices.npz", allow_pickle=False) as saved:
        bootstrap_master_ids = saved["master_id"].copy()
        bootstrap_indices = saved["complete_n_252"].copy()
    save_npz(output / "bootstrap_indices.npz", {"master_id": bootstrap_master_ids,
                                                "complete_n_252": bootstrap_indices})
    all_rows, histories, freeze = [], {}, {}
    for arm_name in cfg["arms"]:
        arm = next(a for a in source_cfg["arms"] if a["name"] == arm_name)
        for seed in cfg["seeds"]:
            checkpoint = ROOT / cfg["source_smoke"] / "checkpoints" / f"{arm_name}|seed={seed}.npz"
            inherited = smoke._model_for_arm(arm, source_cfg, scale)
            meta = smoke.load_checkpoint(checkpoint, inherited)
            if not np.isclose(meta["input_scale"], scale, rtol=0.0, atol=1e-12):
                raise AssertionError("checkpoint input scale does not match regenerated train split")
            models, key_hist = {}, {}
            for kind in ("wls_weight", "irls_relation"):
                model = smoke._model_for_arm(arm, source_cfg, meta["input_scale"])
                smoke.load_checkpoint(checkpoint, model)
                key_hist[kind] = train(model, train_views, arm, seed, cfg, kind, started)
                models[kind] = model
                save_adapter(output / "checkpoints" / f"{arm_name}|seed={seed}|{kind}.npz", model,
                             {"arm": arm_name, "seed": seed, "kind": kind, "last_epoch": cfg["training"]["epochs"]})
            histories[f"{arm_name}|seed={seed}"] = key_hist
            freeze[f"{arm_name}|seed={seed}"] = {k: sum(p.numel() for p in m.parameters() if p.requires_grad) for k, m in models.items()}
            rows, packed = evaluate(models, inherited, eval_views, arm, seed, cfg, started)
            all_rows.extend(rows)
            (output / "raw_eval").mkdir(exist_ok=True)
            save_npz(output / "raw_eval" / f"{arm_name}|seed={seed}.npz", packed)
    write_json(output / "resolved_config.json", cfg)
    write_json(output / "histories.json", histories)
    write_json(output / "freeze_contract.json", freeze)
    write_json(output / "metrics.json", all_rows)
    write_json(output / "effects.json", effects(all_rows, cfg, bootstrap_indices, bootstrap_master_ids))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    checks = "\n".join(f"printf '%s  %s\\n' '{sha(ROOT / p)}' \"$repo/{p}\" | sha256sum -c -" for p in SOURCE_FILES)
    replay = f'''#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || {{ echo "wrong git HEAD" >&2; exit 2; }}
[ -z "$(git -C "$repo" status --porcelain)" ] || {{ echo "dirty worktree" >&2; exit 2; }}
{checks}
: "${{OUTPUT_DIR:?set OUTPUT_DIR to a new path}}"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_frozen_adapters_v1.json" --output "$OUTPUT_DIR"
'''
    (output / "replay.sh").write_text(replay)
    (output / "replay.sh").chmod(0o755)
    deterministic = sorted(p for p in output.rglob("*") if p.is_file() and p.name not in {"manifest.json", "runtime_observation.json"})
    manifest = {"schema_version": cfg["schema_version"], "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
                "source_hashes": {p: sha(ROOT / p) for p in SOURCE_FILES}, "source_checkpoint_hashes": {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / cfg["source_smoke"] / "checkpoints").glob("*.npz")) if any(p.name.startswith(a + "|") for a in cfg["arms"])},
                "deterministic_files": {str(p.relative_to(output)): sha(p) for p in deterministic}}
    write_json(output / "manifest.json", manifest)
    runtime = {"elapsed_seconds": time.monotonic() - started, "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    write_json(output / "runtime_observation.json", runtime)
    print(json.dumps(runtime, sort_keys=True))


if __name__ == "__main__":
    main()
