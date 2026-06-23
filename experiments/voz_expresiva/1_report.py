#!/usr/bin/env python3
"""Fase 1 report: agrega UAR, bootstrap CI sobre diferencias, CKA, plots, REPORTE_1.md.

Inputs:
    data/voz_expresiva/1/uar_results.json
    data/voz_expresiva/1/embeddings/*.npy
    data/voz_expresiva/1/predictions/*.json
    data/voz_expresiva/1/calib_manifest.json

Outputs:
    data/voz_expresiva/1/diff_bootstrap.json
    data/voz_expresiva/1/cka_per_run.json
    data/voz_expresiva/1/uar_comparison.png
    data/voz_expresiva/1/cka_comparison.png
    data/voz_expresiva/1/REPORTE_1.md
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


CONFIGS = ("none", "concat", "film", "xattn")
NORMS = ("strict", "adapt")
SEEDS = (42, 123, 456)
EN_SPEAKERS = tuple(f"{i:04d}" for i in range(11, 21))
MECHANISMS = ("concat", "film", "xattn")


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two embedding matrices, both [N, D]."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Row count mismatch: {X.shape[0]} vs {Y.shape[0]}")
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XtY_F2 = float(np.linalg.norm(X.T @ Y, "fro") ** 2)
    XtX_F2 = float(np.linalg.norm(X.T @ X, "fro") ** 2)
    YtY_F2 = float(np.linalg.norm(Y.T @ Y, "fro") ** 2)
    denom = np.sqrt(XtX_F2 * YtY_F2)
    if denom < 1e-12:
        return float("nan")
    return XtY_F2 / denom


# ---------------------------------------------------------------------------
# Aggregation: per-speaker mean over seeds → mean ± std over speakers
# ---------------------------------------------------------------------------

def aggregate_per_speaker(
    records: List[Dict], metric_key: str,
) -> Dict[Tuple[str, str], Dict]:
    """Build {(config, norm): {speaker: mean_over_seeds_value, ...}}."""
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in records:
        key = (r["config"], r["norm_condition"], r["test_speaker"])
        grouped[key].append(float(r[metric_key]))

    out: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for (cfg, norm, spk), vals in grouped.items():
        out[(cfg, norm)][spk] = float(np.mean(vals))
    return out


def per_speaker_array(spk_map: Dict[str, float]) -> np.ndarray:
    return np.array([spk_map.get(s, np.nan) for s in EN_SPEAKERS], dtype=np.float64)


# ---------------------------------------------------------------------------
# Bootstrap CI sobre diferencias per-speaker
# ---------------------------------------------------------------------------

def bootstrap_diff_ci(
    arr_a: np.ndarray, arr_b: np.ndarray, n_resamples: int = 1000, seed: int = 42,
) -> Tuple[float, float, float, float]:
    rng = np.random.RandomState(seed)
    diffs = arr_a - arr_b
    mask = np.isfinite(diffs)
    diffs = diffs[mask]
    if len(diffs) < 2:
        return (float("nan"),) * 4
    n = len(diffs)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        means[i] = diffs[idx].mean()
    ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])
    return (
        float(diffs.mean()),
        float(ci_lo),
        float(ci_hi),
        float((means > 0).mean()),
    )


# ---------------------------------------------------------------------------
# CKA computation per run
# ---------------------------------------------------------------------------

def compute_cka_all_runs(
    emb_dir: Path, records: List[Dict],
) -> Dict[Tuple[str, str, int, int], float]:
    """For each (config != 'none', norm, fold, seed): linear CKA vs WavLM-only embedding.

    Returns dict {(config, norm, fold_idx, seed): cka_value}.
    """
    out: Dict[Tuple[str, str, int, int], float] = {}
    for r in records:
        if r["config"] == "none":
            continue
        fold = r["fold_idx"]; cfg = r["config"]
        norm = r["norm_condition"]; seed = r["seed"]
        mech_path = emb_dir / f"fold{fold}_{cfg}_{norm}_seed{seed}.npy"
        base_path = emb_dir / f"fold{fold}_none_{norm}_seed{seed}.npy"
        if not mech_path.exists() or not base_path.exists():
            continue
        X = np.load(mech_path)
        Y = np.load(base_path)
        try:
            v = linear_cka(X, Y)
        except Exception:
            v = float("nan")
        out[(cfg, norm, fold, seed)] = v
    return out


def cka_per_speaker_array(
    cka_map: Dict[Tuple[str, str, int, int], float],
    config: str, norm: str,
) -> np.ndarray:
    """Per-speaker mean of CKA across 3 seeds."""
    arr = np.full(len(EN_SPEAKERS), np.nan)
    for fold_idx, spk in enumerate(EN_SPEAKERS):
        vals = [cka_map.get((config, norm, fold_idx, s)) for s in SEEDS]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals:
            arr[fold_idx] = float(np.mean(vals))
    return arr


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_uar_comparison(
    uar_per_speaker: Dict[Tuple[str, str], Dict[str, float]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, norm in zip(axes, NORMS):
        means, sems, labels = [], [], []
        for cfg in CONFIGS:
            arr = per_speaker_array(uar_per_speaker.get((cfg, norm), {}))
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                continue
            means.append(arr.mean())
            sems.append(arr.std() / np.sqrt(max(1, len(arr))))
            labels.append(cfg)
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=sems, capsize=4, color="#2C6E72")
        ax.axhline(0.2, color="grey", linestyle="--", linewidth=0.8, label="chance")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"N-{norm}", fontsize=11)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Mean UAR (± SEM over 10 speakers)")
    fig.suptitle("Fase 1 — UAR per config × norm", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_cka_comparison(
    cka_per_speaker: Dict[Tuple[str, str], np.ndarray],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, norm in zip(axes, NORMS):
        means, sems, labels = [], [], []
        for mech in MECHANISMS:
            arr = cka_per_speaker.get((mech, norm), np.full(len(EN_SPEAKERS), np.nan))
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                continue
            means.append(arr.mean())
            sems.append(arr.std() / np.sqrt(max(1, len(arr))))
            labels.append(mech)
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=sems, capsize=4, color="#8A5230")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"N-{norm} (CKA vs WavLM-only)", fontsize=11)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Linear CKA (± SEM over 10 speakers)")
    fig.suptitle("Fase 1 — Reorganización geométrica (CKA per mecanismo)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    out_path: Path,
    uar_per_speaker: Dict[Tuple[str, str], Dict[str, float]],
    diff_table: Dict[Tuple[str, str], dict],
    cka_per_speaker: Dict[Tuple[str, str], np.ndarray],
    n_records: int,
) -> None:
    lines: List[str] = []
    lines.append("# Reporte Fase 1 — Voz Expresiva Phideus\n")
    lines.append(
        f"> LOSO CV con 10 hablantes EN. 4 configs × 2 norm × 3 seeds × 10 folds. "
        f"Total {n_records} task results. Tono comparativo del piloto; con n=10 speakers los "
        "CI bootstrap son señal, no prueba fuerte. Generalización honesta requiere "
        "Fase 3 (MSP-Podcast).\n"
    )

    for norm in NORMS:
        lines.append(f"\n## {'N-strict (primaria, sin per-speaker en test)' if norm == 'strict' else 'N-adapt (secundaria, 1 calib repeat congelado seed=42)'}\n")
        lines.append("### UAR — mean ± std across 10 held-out speakers\n")
        lines.append("| Config | Mean UAR | Std |")
        lines.append("|---|---|---|")
        for cfg in CONFIGS:
            arr = per_speaker_array(uar_per_speaker.get((cfg, norm), {}))
            arr_f = arr[np.isfinite(arr)]
            if len(arr_f) == 0:
                lines.append(f"| **{cfg}** | — | — |")
            else:
                lines.append(f"| **{cfg}** | {arr_f.mean():.3f} | {arr_f.std():.3f} |")
        lines.append("")

        lines.append("### Diferencias principales — bootstrap CI95 sobre Δ per-speaker\n")
        lines.append("| Mecanismo vs WavLM-only | Δ mean | CI95 lo | CI95 hi | P(Δ>0) | Lectura |")
        lines.append("|---|---|---|---|---|---|")
        for mech in MECHANISMS:
            entry = diff_table.get((mech, norm))
            if not entry:
                lines.append(f"| {mech} | — | — | — | — | — |")
                continue
            m, lo, hi, frac = entry["mean"], entry["ci_lo"], entry["ci_hi"], entry["frac_pos"]
            if not np.isfinite(m):
                read = "—"
            elif lo > 0:
                read = "Δ > 0 robusto"
            elif hi < 0:
                read = "Δ < 0 robusto"
            else:
                read = "CI cruza 0"
            lines.append(f"| {mech} | {m:+.3f} | {lo:+.3f} | {hi:+.3f} | {frac:.2f} | {read} |")
        lines.append("")

        lines.append("### CKA per mecanismo vs WavLM-only (reorganización geométrica)\n")
        lines.append("| Mecanismo | Mean CKA | Std |")
        lines.append("|---|---|---|")
        for mech in MECHANISMS:
            arr = cka_per_speaker.get((mech, norm), np.full(len(EN_SPEAKERS), np.nan))
            arr_f = arr[np.isfinite(arr)]
            if len(arr_f) == 0:
                lines.append(f"| {mech} | — | — |")
            else:
                lines.append(f"| {mech} | {arr_f.mean():.3f} | {arr_f.std():.3f} |")
        lines.append("")

    lines.append("\n## Cierre direccional\n")
    lines.append("Cuatro escenarios prefigurados (decisión formal queda al usuario):\n")
    lines.append("- **WavLM-only > chance + algún mecanismo > WavLM-only en N-strict (CI95 excluye 0)**: target real. Phideus transfiere a SSL bajo generalización honesta.")
    lines.append("- **WavLM-only > chance, todos los mecanismos ≈ WavLM-only**: SSL resuelve, Phideus no agrega. Pivot a Carril B o Fase 1.2 con ajustes.")
    lines.append("- **Ninguno escapa N-strict, pero CKA muestra reorganización**: efecto geométrico sin funcional. Justifica Fase 1.2 con ajustes.")
    lines.append("- **Ninguno escapa N-strict y CKA no discrimina**: antes de Fase 3, considerar pooling alternativo / punto de inyección / baseline tuned / descriptor expandido.\n")

    lines.append("## Caveats metodológicos (declarados)\n")
    lines.append("- Single-speaker validation por fold (val_speaker = speakers[(k+1) % 10]). Tuning ruidoso, no selección estable.")
    lines.append("- Bootstrap n=10 speakers: señal comparativa, no prueba fuerte. Generalización honesta requiere Fase 3.")
    lines.append("- ESD actuado: cualquier resultado requiere validación en habla naturalística.")
    lines.append("- N-adapt con 1 calib repeat (vs 3 repeats en 0B): lectura secundaria menos estable que la N-adapt de 0B. Documentado.")
    lines.append("- N-strict: train con per-speaker z-score, val/test con train-pool stats. Mismatch declarado.")
    lines.append("- CKA mide reorganización geométrica pre-classifier-head; interpretación afectiva no es directa.")
    lines.append("- 3 seeds es piloto; 5 requeriría 5/3× cómputo.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, help="data/voz_expresiva/1")
    args = p.parse_args()

    res_dir = Path(args.results_dir)
    records = json.loads((res_dir / "uar_results.json").read_text())
    logger.info("Loaded %d records", len(records))

    # Per-speaker UAR
    uar_per_speaker = aggregate_per_speaker(records, "test_uar")

    # Diff bootstrap: each mechanism vs none, per norm
    diff_table: Dict[Tuple[str, str], dict] = {}
    diff_export = []
    for norm in NORMS:
        arr_base = per_speaker_array(uar_per_speaker.get(("none", norm), {}))
        for mech in MECHANISMS:
            arr_mech = per_speaker_array(uar_per_speaker.get((mech, norm), {}))
            m, lo, hi, frac = bootstrap_diff_ci(arr_mech, arr_base)
            diff_table[(mech, norm)] = {
                "mean": m, "ci_lo": lo, "ci_hi": hi, "frac_pos": frac,
            }
            diff_export.append({
                "norm_condition": norm, "mechanism": mech, "baseline": "none",
                "mean_diff": m, "ci95_lo": lo, "ci95_hi": hi, "p_diff_positive": frac,
            })

    (res_dir / "diff_bootstrap.json").write_text(json.dumps(diff_export, indent=2))
    logger.info("Saved diff_bootstrap.json")

    # CKA
    cka_map = compute_cka_all_runs(res_dir / "embeddings", records)
    cka_per_speaker: Dict[Tuple[str, str], np.ndarray] = {}
    cka_export = []
    for mech in MECHANISMS:
        for norm in NORMS:
            cka_arr = cka_per_speaker_array(cka_map, mech, norm)
            cka_per_speaker[(mech, norm)] = cka_arr
            for spk_idx, spk in enumerate(EN_SPEAKERS):
                cka_export.append({
                    "mechanism": mech, "norm_condition": norm,
                    "speaker": spk, "fold_idx": spk_idx,
                    "cka_per_speaker_mean_over_seeds": float(cka_arr[spk_idx]) if np.isfinite(cka_arr[spk_idx]) else None,
                })
    (res_dir / "cka_per_run.json").write_text(json.dumps(cka_export, indent=2))
    logger.info("Saved cka_per_run.json")

    # Plots
    plot_uar_comparison(uar_per_speaker, res_dir / "uar_comparison.png")
    logger.info("Saved uar_comparison.png")
    plot_cka_comparison(cka_per_speaker, res_dir / "cka_comparison.png")
    logger.info("Saved cka_comparison.png")

    # Report
    write_report(
        res_dir / "REPORTE_1.md",
        uar_per_speaker, diff_table, cka_per_speaker, len(records),
    )
    logger.info("Saved REPORTE_1.md")


if __name__ == "__main__":
    main()
