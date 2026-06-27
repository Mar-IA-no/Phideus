#!/usr/bin/env python3
"""Training del agrupamiento armónico — Fase 0 Atención Armónica.

Entrena {A-naive, A-rich, B, B-minus, B-local, B-shuffle} × seeds × runs {ID, OOD-poly, OOD-regime}.

Optimizer/schedule CONGELADO y COMPARTIDO por los 6 modelos (Codex r1 #1): AdamW lr=3e-4,
batch 256, 50 epochs, cosine + warmup, SIN early-stopping (corrida fija → no hay selección
de checkpoint por val). MODEL_CONFIGS por modelo (Codex r2 #3).

τ para ARI se elige por-modelo SOLO en val (mismo procedimiento para todos; nunca en test,
Codex). F1 primaria no usa τ (umbral 0.5). Métricas pairwise solo sobre upper-tri válido.

Outputs (en --output-dir):
    results.json                              # 1 record por (run, model, seed): params, τ, métricas
    test_pairs/<run>__<model>__seed<seed>.npz # pares válidos de test (flat) para bootstrap en report
    test_ari/<run>__<model>__seed<seed>.npz   # ARI por mezcla (τ de val) + polifonía

Run (tmux, GPU):
    python experiments/atencion_armonica/1_train_grouping.py \\
        --pool data/atencion_armonica/pool/mixtures.jsonl \\
        --output-dir data/atencion_armonica/fase0 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.atencion_armonica.grouping_dataset import (  # noqa: E402
    GroupingDataset, collate_grouping, load_pool, make_run_splits,
)
from src.atencion_armonica.pairformer import (  # noqa: E402
    MODEL_CONFIGS, MODEL_NAMES, build_model, count_params,
)
from experiments.atencion_armonica.harness import (  # noqa: E402
    ari_for_mixture, extract_valid_pairs, pairwise_metrics, random_baseline_f1,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEEDS = (42, 123, 456)
RUNS = ("ID", "OOD-poly", "OOD-regime")
TAU_GRID = np.round(np.arange(0.10, 0.91, 0.05), 3)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict, device) -> Dict:
    out = dict(batch)
    for k in ("tokens", "pair_cont", "ratio_class_id", "target", "pair_valid", "token_mask"):
        out[k] = batch[k].to(device, non_blocking=True)
    # mixture_id queda como lista de ints en CPU (B-shuffle lo usa para seed) — Codex r2 #5
    return out


def masked_bce(logit, target, pair_valid):
    pv = pair_valid.float()
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return (loss * pv).sum() / pv.sum().clamp(min=1.0)


def train_one(model, train_loader, device, epochs, lr, wd, warmup_frac=0.05):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = epochs * len(train_loader)
    warmup = max(1, int(warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    model.train()
    for _ep in range(epochs):
        for batch in train_loader:
            batch = move_batch(batch, device)
            logit = model(batch)
            loss = masked_bce(logit, batch["target"], batch["pair_valid"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return float(loss.item())


@torch.no_grad()
def collect_eval(model, loader, device):
    """Devuelve por-mezcla: pares válidos (logit,target) + matriz logit + true_source + poly/regime."""
    model.eval()
    per_mix_pairs: List[Dict] = []
    per_mix_full: List[Dict] = []
    for batch in loader:
        dev_batch = move_batch(batch, device)
        logit = model(dev_batch).cpu()
        pairs = extract_valid_pairs(logit, batch["target"], batch["pair_valid"], batch["token_mask"])
        for b, pr in enumerate(pairs):
            N = batch["n_peaks"][b]
            per_mix_pairs.append({
                "logit": pr["logit"], "target": pr["target"],
                "polyphony": batch["polyphony"][b], "regime": batch["regime"][b],
                "mixture_id": batch["mixture_id"][b],
            })
            per_mix_full.append({
                "logit_mat": logit[b].numpy(),
                "token_mask": batch["token_mask"][b].numpy(),
                "pair_valid": batch["pair_valid"][b].numpy(),
                "target_mat": batch["target"][b].numpy(),
                "polyphony": batch["polyphony"][b], "regime": batch["regime"][b],
                "mixture_id": batch["mixture_id"][b], "n_peaks": int(N),
            })
    return per_mix_pairs, per_mix_full


def _true_source_from_target(target_mat: np.ndarray, token_mask: np.ndarray) -> np.ndarray:
    """Recupera la partición verdadera (source id por token) desde la matriz target de equivalencia."""
    from scipy.sparse.csgraph import connected_components
    N = target_mat.shape[0]
    adj = (target_mat >= 0.5)
    np.fill_diagonal(adj, True)
    _, labels = connected_components(adj, directed=False)
    return labels


def select_tau_on_val(per_mix_full: List[Dict]) -> float:
    """Elige τ que maximiza ARI medio en VAL (nunca test). Mismo procedimiento para todos los modelos.

    Usa SOLO mezclas no-degeneradas poly>=2 (Codex Medio #1): poly-1 es todo-mismo-fuente y
    sesgaría τ hacia 'todo conectado'. Fallback a todo val si no hay poly>=2.
    """
    pool = [m for m in per_mix_full if m["polyphony"] >= 2] or per_mix_full
    best_tau, best_ari = float(TAU_GRID[0]), -2.0
    for tau in TAU_GRID:
        aris = []
        for m in pool:
            ts = _true_source_from_target(m["target_mat"], m["token_mask"])
            ari = ari_for_mixture(m["logit_mat"], m["token_mask"], ts, float(tau), m["pair_valid"])
            if np.isfinite(ari):
                aris.append(ari)
        mean_ari = float(np.mean(aris)) if aris else -2.0
        if mean_ari > best_ari:
            best_ari, best_tau = mean_ari, float(tau)
    return best_tau


def _cell_key(poly, regime) -> str:
    return f"poly{poly}_{regime}"


def eval_test(per_mix_pairs, per_mix_full, tau):
    """Métricas de test por celda polifonía×régimen (upper-tri válido) + overall; ARI por mezcla.

    Returns (summary_dict, per_mixture_ari_records).
    """
    # overall pooled
    all_lg = np.concatenate([m["logit"] for m in per_mix_pairs]) if per_mix_pairs else np.array([])
    all_tg = np.concatenate([m["target"] for m in per_mix_pairs]) if per_mix_pairs else np.array([])
    overall = pairwise_metrics(all_lg, all_tg)
    overall["random_f1"] = random_baseline_f1(all_tg)

    # por celda polifonía×régimen
    by_cell = {}
    cells = sorted({(m["polyphony"], m["regime"]) for m in per_mix_pairs})
    for (poly, regime) in cells:
        sub = [m for m in per_mix_pairs if m["polyphony"] == poly and m["regime"] == regime]
        lg = np.concatenate([m["logit"] for m in sub])
        tg = np.concatenate([m["target"] for m in sub])
        by_cell[_cell_key(poly, regime)] = pairwise_metrics(lg, tg)

    # ARI por mezcla con τ de val (excluye pares inválidos) + ARI@0.5 fijo (transparencia, Codex)
    ari_records = []
    ari_by_cell = {}
    ari05_by_cell = {}
    for m in per_mix_full:
        ts = _true_source_from_target(m["target_mat"], m["token_mask"])
        ari = ari_for_mixture(m["logit_mat"], m["token_mask"], ts, tau, m["pair_valid"])
        ari05 = ari_for_mixture(m["logit_mat"], m["token_mask"], ts, 0.5, m["pair_valid"])
        ari_records.append({
            "mixture_id": m["mixture_id"], "polyphony": m["polyphony"],
            "regime": m["regime"], "ari": ari, "ari_tau05": ari05, "n_peaks": m["n_peaks"],
        })
        key = _cell_key(m["polyphony"], m["regime"])
        ari_by_cell.setdefault(key, []).append(ari)
        ari05_by_cell.setdefault(key, []).append(ari05)
    ari_summary = {k: float(np.nanmean(v)) if len(v) else float("nan")
                   for k, v in ari_by_cell.items()}
    ari05_summary = {k: float(np.nanmean(v)) if len(v) else float("nan")
                     for k, v in ari05_by_cell.items()}

    summary = {"overall": overall, "by_cell": by_cell, "ari_by_cell": ari_summary,
               "ari05_by_cell": ari05_summary, "tau": tau}
    return summary, ari_records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--runs", nargs="+", default=list(RUNS))
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-mixtures", type=int, default=None, help="Smoke: limita mezclas por split")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "test_pairs").mkdir(parents=True, exist_ok=True)
    (out_dir / "test_ari").mkdir(parents=True, exist_ok=True)

    # Guard anti-CPU accidental (Codex Medio #2): si se pide cuda y no hay, abortar.
    # CPU solo permitido si se pide explícito (smoke) — directiva: no pilots en CPU.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA solicitado pero no disponible. El training real DEBE ir en GPU "
            "(directiva: NUNCA pilots en CPU). Usar --device cpu solo para smoke de shapes."
        )
    device = torch.device(args.device)
    logger.info("Device: %s", device)
    pool = load_pool(args.pool)
    logger.info("Pool: %d mezclas", len(pool))

    # run_meta.json — receta completa para reproducir (Codex Bajo #3)
    from datetime import datetime
    run_meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "pool_path": str(args.pool), "pool_size": len(pool),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "num_workers": args.num_workers, "device": str(device),
        "runs": args.runs, "models": args.models, "seeds": args.seeds,
        "limit_mixtures": args.limit_mixtures,
        "tau_grid": [float(t) for t in TAU_GRID],
        "split_sizes": {},
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    all_results = []
    t0 = time.time()

    for run in args.runs:
        train_recs, val_recs, test_recs = make_run_splits(pool, run)
        if args.limit_mixtures:
            # smoke: shuffle determinístico ANTES de limitar para estratificar sobre polifonías
            # (sin esto, [:N] toma solo la primera celda = poly-1, degenerada)
            rng = np.random.RandomState(0)
            def _lim(recs, n):
                idx = rng.permutation(len(recs))[:n]
                return [recs[i] for i in idx]
            train_recs = _lim(train_recs, args.limit_mixtures)
            val_recs = _lim(val_recs, max(8, args.limit_mixtures // 4))
            test_recs = _lim(test_recs, max(8, args.limit_mixtures // 4))
        logger.info("Run %s: train=%d val=%d test=%d", run, len(train_recs), len(val_recs), len(test_recs))
        run_meta["split_sizes"][run] = {
            "train": len(train_recs), "val": len(val_recs), "test": len(test_recs),
        }
        (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

        train_loader = DataLoader(GroupingDataset(train_recs), batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, collate_fn=collate_grouping)
        val_loader = DataLoader(GroupingDataset(val_recs), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, collate_fn=collate_grouping)
        test_loader = DataLoader(GroupingDataset(test_recs), batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, collate_fn=collate_grouping)

        for model_name in args.models:
            for seed in args.seeds:
                t_run = time.time()
                set_seed(seed)
                model = build_model(model_name).to(device)
                n_params = count_params(model)

                final_loss = train_one(model, train_loader, device, args.epochs,
                                       args.lr, args.weight_decay)

                _vp, val_full = collect_eval(model, val_loader, device)
                tau = select_tau_on_val(val_full)

                test_pairs, test_full = collect_eval(model, test_loader, device)
                metrics, ari_records = eval_test(test_pairs, test_full, tau)

                tag = f"{run}__{model_name}__seed{seed}"
                # guardar pares de test (flat) para bootstrap en report — con regime (Codex Alto #1)
                np.savez_compressed(
                    out_dir / "test_pairs" / f"{tag}.npz",
                    logit=np.concatenate([m["logit"] for m in test_pairs]) if test_pairs else np.array([]),
                    target=np.concatenate([m["target"] for m in test_pairs]) if test_pairs else np.array([]),
                    mix_id=np.concatenate([np.full(len(m["logit"]), m["mixture_id"]) for m in test_pairs]) if test_pairs else np.array([]),
                    polyphony=np.concatenate([np.full(len(m["logit"]), m["polyphony"]) for m in test_pairs]) if test_pairs else np.array([]),
                    regime=np.concatenate([np.full(len(m["logit"]), m["regime"]) for m in test_pairs]) if test_pairs else np.array([], dtype="<U4"),
                )
                # guardar ARI por mezcla (Codex Medio #3)
                np.savez_compressed(
                    out_dir / "test_ari" / f"{tag}.npz",
                    mixture_id=np.array([r["mixture_id"] for r in ari_records]),
                    polyphony=np.array([r["polyphony"] for r in ari_records]),
                    regime=np.array([r["regime"] for r in ari_records], dtype="<U4"),
                    ari=np.array([r["ari"] for r in ari_records], dtype=np.float64),
                    ari_tau05=np.array([r["ari_tau05"] for r in ari_records], dtype=np.float64),
                    n_peaks=np.array([r["n_peaks"] for r in ari_records]),
                    tau=np.float64(tau),
                )

                rec = {
                    "run": run, "model": model_name, "seed": int(seed),
                    "n_params": int(n_params), "config": MODEL_CONFIGS[model_name],
                    "tau": tau, "final_train_loss": final_loss,
                    "test": metrics,
                    "wall_seconds": time.time() - t_run,
                }
                all_results.append(rec)
                (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))

                ov = metrics["overall"]
                logger.info("  %-26s params=%d τ=%.2f F1=%.3f AP=%.3f AUC=%.3f (%.0fs)",
                            tag, n_params, tau, ov["f1"], ov["ap"], ov["roc_auc"],
                            rec["wall_seconds"])

    logger.info("All runs done in %.2f h", (time.time() - t0) / 3600)


if __name__ == "__main__":
    main()
