#!/usr/bin/env python3
"""Genera el POOL sintético de mezclas armónicas — Fase 0 Atención Armónica.

Produce ~36k mezclas (6k por celda polifonía×régimen) con ground truth exacto.
CPU, paralelo (ProcessPoolExecutor). Escribe:
    <output-dir>/mixtures.jsonl     # una mezcla por línea (auditable)
    <output-dir>/pool_meta.json     # config de la grilla congelada + counts

Uso:
    python experiments/atencion_armonica/0_generate.py \\
        --output-dir data/atencion_armonica/pool --workers 14
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.atencion_armonica.harmonic_synth import (  # noqa: E402
    EASY_MULTIPLIERS, EPS_CENTS, F0_MAX_HZ, F0_MIN_HZ, HARD_RATIOS, INHARM_BETA_WIDTH,
    JITTER_CENTS_MAX, JITTER_CENTS_MIN, K_HARMONICS, MIN_PARTIALS, POLYPHONIES, REGIMES,
    generate_mixture,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POOL_SEED = 20260626        # seed por DEFAULT; el final_pool usa seed DISTINTO al de calibración
N_PER_CELL = 6000           # mezclas por celda (polifonía × régimen)


def _worker(payload: dict) -> dict:
    g = payload["gen"]
    m = generate_mixture(
        mixture_id=payload["mixture_id"],
        polyphony=payload["polyphony"],
        regime=payload["regime"],
        master_seed=payload["seed"],
        beta_center=g["beta_center"], p_drop=g["p_drop"],
        alpha_range=tuple(g["alpha_range"]), sigma_amp=g["sigma_amp"],
    )
    return m.to_json()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--n-per-cell", type=int, default=N_PER_CELL)
    p.add_argument("--seed", type=int, default=POOL_SEED, help="master_seed del pool (final_pool DISTINTO de calibración)")
    # Params de generación CONGELADOS tras el sweep v2.1 (pasados explícitos, registrados en meta)
    p.add_argument("--beta-center", type=float, required=True)
    p.add_argument("--p-drop", type=float, required=True)
    p.add_argument("--alpha-lo", type=float, required=True)
    p.add_argument("--alpha-hi", type=float, required=True)
    p.add_argument("--sigma-amp", type=float, required=True)
    p.add_argument("--calibration-seed", type=int, default=None, help="seed del calibration_pool (para registro)")
    p.add_argument("--chosen-sweep-row", default=None, help="JSON de la fila elegida del sweep (para registro)")
    args = p.parse_args()

    # Guard (Codex r8 #2): final_pool DEBE usar seed distinto al de calibración.
    if args.calibration_seed is not None and args.seed == args.calibration_seed:
        raise SystemExit(
            f"ABORT: --seed ({args.seed}) == --calibration-seed. El final_pool debe usar un seed "
            "DISTINTO al calibration_pool (no ajustar dataset sobre las etiquetas de calibración)."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_cfg = {
        "beta_center": args.beta_center, "p_drop": args.p_drop,
        "alpha_range": [args.alpha_lo, args.alpha_hi], "sigma_amp": args.sigma_amp,
    }

    # Asignación de mixture_ids: celdas contiguas (polifonía × régimen)
    payloads = []
    mid = 0
    cell_ranges = {}
    for poly in POLYPHONIES:
        for regime in REGIMES:
            start = mid
            for _ in range(args.n_per_cell):
                payloads.append({"mixture_id": mid, "polyphony": poly, "regime": regime,
                                 "seed": args.seed, "gen": gen_cfg})
                mid += 1
            cell_ranges[f"poly{poly}_{regime}"] = [start, mid]

    n_total = len(payloads)
    logger.info("Generando %d mezclas (%d celdas × %d)", n_total,
                len(POLYPHONIES) * len(REGIMES), args.n_per_cell)

    results: dict[int, dict] = {}
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, pl): pl for pl in payloads}
        for fut in as_completed(futures):
            res = fut.result()
            results[res["mixture_id"]] = res
            n_done += 1
            if n_done % max(1, n_total // 40) == 0:
                rate = n_done / (time.time() - t0)
                eta = (n_total - n_done) / rate if rate > 0 else 0
                logger.info("Progress %d/%d (%.0f mix/s, ETA %.1f min)",
                            n_done, n_total, rate, eta / 60)

    # Escribir JSONL ordenado por mixture_id
    jsonl_path = out_dir / "mixtures.jsonl"
    with jsonl_path.open("w") as f:
        for i in range(n_total):
            f.write(json.dumps(results[i]) + "\n")

    # Estadísticas de auditoría
    n_peaks = [len(results[i]["peaks"]) for i in range(n_total)]
    n_masked = [len(results[i]["masked_pairs"]) for i in range(n_total)]
    mask_rate_by_cell = {}
    for cell, (start, end) in cell_ranges.items():
        total_pairs = 0
        total_masked = 0
        for i in range(start, end):
            npk = len(results[i]["peaks"])
            total_pairs += npk * (npk - 1) // 2
            total_masked += len(results[i]["masked_pairs"])
        mask_rate_by_cell[cell] = total_masked / max(1, total_pairs)

    chosen_row = json.loads(args.chosen_sweep_row) if args.chosen_sweep_row else None

    # Auditoría de restauración POR CELDA (Codex r8 #3/#4): medición real de was_restored,
    # distribución de armónicos sobrevivientes, P(fundamental ausente).
    def cell_of(mid_):
        for cell, (start, end) in cell_ranges.items():
            if start <= mid_ < end:
                return cell
        return "?"

    per_cell_restore = {c: {"n_sources": 0, "n_restored": 0, "n_fund_missing": 0,
                            "harm_counts": {}} for c in cell_ranges}
    surv_harm_counts = {}; n_with_fund = 0; n_sources = 0; n_restored = 0
    for i in range(n_total):
        cell = cell_of(i)
        for s in results[i]["sources"]:
            n_sources += 1
            pc = per_cell_restore[cell]; pc["n_sources"] += 1
            if s.get("was_restored"):
                n_restored += 1; pc["n_restored"] += 1     # medición REAL (no proxy)
            harms = [p["harmonic"] for p in s["partials"]]
            if 1 not in harms:
                pc["n_fund_missing"] += 1
            else:
                n_with_fund += 1
            for h in harms:
                surv_harm_counts[h] = surv_harm_counts.get(h, 0) + 1
                pc["harm_counts"][h] = pc["harm_counts"].get(h, 0) + 1
    restore_by_cell = {c: {
        "frac_restored": v["n_restored"] / max(1, v["n_sources"]),
        "p_fundamental_missing": v["n_fund_missing"] / max(1, v["n_sources"]),
        "harm_counts": v["harm_counts"],
    } for c, v in per_cell_restore.items()}

    meta = {
        "version": "v2.1",
        "pool_seed": args.seed,
        "calibration_seed": args.calibration_seed,
        "final_pool_seed": args.seed,
        "n_total": n_total,
        "n_per_cell": args.n_per_cell,
        "cell_ranges": cell_ranges,
        "frozen_params": {                       # CONGELADOS tras el sweep v2.1 (Codex r7 #4)
            "beta_center": args.beta_center,
            "beta_width": INHARM_BETA_WIDTH,
            "alpha_range": [args.alpha_lo, args.alpha_hi],
            "sigma_amp": args.sigma_amp,
            "p_drop": args.p_drop,
            "min_partials": MIN_PARTIALS,
        },
        "chosen_sweep_row": chosen_row,
        "grid": {
            "K_harmonics": K_HARMONICS,
            "f0_range_hz": [F0_MIN_HZ, F0_MAX_HZ],
            "polyphonies": list(POLYPHONIES),
            "regimes": list(REGIMES),
            "eps_cents": EPS_CENTS,
            "jitter_cents": [JITTER_CENTS_MIN, JITTER_CENTS_MAX],
            "hard_ratios": [list(r) for r in HARD_RATIOS],
            "easy_multipliers": list(EASY_MULTIPLIERS),
        },
        "audit": {
            "peaks_per_mixture_min": int(min(n_peaks)),
            "peaks_per_mixture_max": int(max(n_peaks)),
            "masked_pairs_total": int(sum(n_masked)),
            "mask_rate_by_cell": mask_rate_by_cell,
            "surviving_harmonic_counts": surv_harm_counts,
            "p_fundamental_present": n_with_fund / max(1, n_sources),
            "frac_restored_real": n_restored / max(1, n_sources),   # medición real (was_restored)
            "restore_by_cell": restore_by_cell,                      # por poly×regime (Codex r8 #4)
        },
    }
    (out_dir / "pool_meta.json").write_text(json.dumps(meta, indent=2))

    elapsed = time.time() - t0
    logger.info("Done in %.1f min. %d mezclas → %s", elapsed / 60, n_total, jsonl_path)
    logger.info("Peaks/mix: [%d, %d]. Masked pairs total: %d",
                min(n_peaks), max(n_peaks), sum(n_masked))
    logger.info("Mask rate by cell: %s",
                {k: f"{v:.4f}" for k, v in mask_rate_by_cell.items()})


if __name__ == "__main__":
    main()
