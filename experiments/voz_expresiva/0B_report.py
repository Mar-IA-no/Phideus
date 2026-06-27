#!/usr/bin/env python3
"""Fase 0B — analyze classifier results + generate REPORTE_0B.md.

Carga uar_results.json + predictions.npz, calcula bootstrap CI sobre diferencias
clave, genera plots y confusion matrices, escribe el reporte humano.

Run:
    python experiments/voz_expresiva/0B_report.py \\
        --input data/voz_expresiva/0B \\
        --output-dir data/voz_expresiva/0B
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Pairs of comparisons (config_a - config_b) the report should highlight
KEY_DIFFS = [
    ("A+D", "D-only"),
    ("C+D", "D-only"),
    ("A+D", "C+D"),
    ("A-only", "C-only"),
    ("A+B+D", "A+D"),
    ("A+B", "A-only"),
]

ALL_CONFIGS = ("D-only", "A-only", "B-only", "C-only", "A+B", "A+D", "C+D", "A+B+D")
NORM_CONDITIONS = ("strict", "adapt")
CLASSIFIERS = ("logreg", "svm_rbf")


def load_uar(uar_path: Path) -> List[dict]:
    return json.loads(uar_path.read_text())


def load_predictions(pred_path: Path) -> List[dict]:
    arr = np.load(pred_path, allow_pickle=True)
    return json.loads(str(arr["predictions"]))


# ---------------------------------------------------------------------------
# UAR per (norm, config, clf, speaker)
# ---------------------------------------------------------------------------

def build_uar_table(records: List[dict]) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """Returns {(norm, config, clf): {speaker: UAR}}."""
    out: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
    for r in records:
        key = (r["norm_condition"], r["config_name"], r["clf_name"])
        out[key][r["test_speaker"]] = r["test_uar"]
    return out


def per_speaker_array(uar_dict: Dict[str, float], speakers: List[str]) -> np.ndarray:
    return np.array([uar_dict.get(s, np.nan) for s in speakers], dtype=np.float64)


# ---------------------------------------------------------------------------
# Bootstrap difference CI95
# ---------------------------------------------------------------------------

def bootstrap_diff_ci(
    uar_a: np.ndarray, uar_b: np.ndarray, n_resamples: int = 1000, seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Bootstrap CI95 of mean(UAR_a - UAR_b) by resampling speakers with replacement.

    Returns (mean_diff, ci_lo, ci_hi, fraction_positive).
    """
    rng = np.random.RandomState(seed)
    diffs = uar_a - uar_b
    finite = np.isfinite(diffs)
    diffs = diffs[finite]
    if len(diffs) < 2:
        return (float("nan"),) * 4
    n = len(diffs)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        means[i] = diffs[idx].mean()
    ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])
    return float(diffs.mean()), float(ci_lo), float(ci_hi), float((means > 0).mean())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_uar_comparison(
    table: Dict[Tuple[str, str, str], Dict[str, float]],
    speakers: List[str],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for ax_norm, norm in zip(axes, NORM_CONDITIONS):
        for ax, clf in zip(ax_norm, CLASSIFIERS):
            means, stds, labels = [], [], []
            for cfg in ALL_CONFIGS:
                key = (norm, cfg, clf)
                if key not in table:
                    continue
                uars = per_speaker_array(table[key], speakers)
                uars = uars[np.isfinite(uars)]
                if len(uars) == 0:
                    continue
                means.append(uars.mean())
                stds.append(uars.std() / np.sqrt(max(1, len(uars))))  # SEM
                labels.append(cfg)
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=4, color="#2C6E72")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.axhline(0.2, color="grey", linestyle="--", linewidth=0.8, label="chance (1/5)")
            ax.set_title(f"{norm} · {clf}", fontsize=10)
            ax.set_ylim(0, 1)
            if clf == "logreg":
                ax.set_ylabel("Mean UAR (± SEM speakers)")
    fig.suptitle("Fase 0B — UAR per config × norm × classifier", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion(
    predictions: List[dict], out_dir: Path,
) -> None:
    """Generate confusion matrices per (norm, config, clf) aggregating across folds and (for adapt) reps."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    by_key: Dict[Tuple[str, str, str], Tuple[List, List]] = defaultdict(lambda: ([], []))
    for p in predictions:
        key = (p["norm_condition"], p["config_name"], p["clf_name"])
        y_true, y_pred = by_key[key]
        y_true.extend(p["y_true"])
        y_pred.extend(p["y_pred"])

    for (norm, cfg, clf), (y_true, y_pred) in by_key.items():
        if not y_true:
            continue
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{norm} · {cfg} · {clf}", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fname = f"{norm}_{cfg.replace('+','-')}_{clf}.png"
        fig.savefig(out_dir / fname, dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    out_path: Path,
    table: Dict[Tuple[str, str, str], Dict[str, float]],
    speakers: List[str],
    diff_table: Dict[Tuple[str, str, str], List[dict]],
    n_total_records: int,
) -> None:
    lines: List[str] = []
    lines.append("# Reporte Fase 0B — Voz Expresiva Phideus\n")
    lines.append(
        "> LOSO CV con 10 hablantes EN de ESD. 8 feature subsets × 2 norm × 2 clfs × 10 folds. "
        f"Total {n_total_records} task results. Tono comparativo del piloto; con n=10 speakers los "
        "CI bootstrap son señal, no prueba fuerte de causalidad. Generalización honesta requiere "
        "Fase 3 (MSP-Podcast).\n"
    )

    for norm in NORM_CONDITIONS:
        lines.append(f"\n## {'N-strict (primaria)' if norm == 'strict' else 'N-adapt (secundaria, label-agnóstica, 3 repeats agregados intra-speaker)'}\n")
        lines.append("### UAR — mean ± std across 10 held-out speakers\n")
        lines.append("| Config | LogReg UAR | SVM RBF UAR |")
        lines.append("|---|---|---|")
        for cfg in ALL_CONFIGS:
            row = f"| **{cfg}** |"
            for clf in CLASSIFIERS:
                key = (norm, cfg, clf)
                uars = per_speaker_array(table.get(key, {}), speakers)
                uars = uars[np.isfinite(uars)]
                if len(uars) == 0:
                    row += " — |"
                else:
                    row += f" {uars.mean():.3f} ± {uars.std():.3f} |"
            lines.append(row)
        lines.append("")

        lines.append("### Diferencias clave — bootstrap CI95 sobre Δ por speaker\n")
        lines.append("Para cada par y clasificador: mean Δ, CI95, fracción bootstrap > 0.\n")
        for cfg_a, cfg_b in KEY_DIFFS:
            lines.append(f"\n**{cfg_a} − {cfg_b}**\n")
            lines.append("| Classifier | Δ mean | CI95 lo | CI95 hi | P(Δ>0) | Lectura |")
            lines.append("|---|---|---|---|---|---|")
            for clf in CLASSIFIERS:
                key = (norm, cfg_a, cfg_b, clf)
                entry = diff_table.get(key)
                if not entry:
                    lines.append(f"| {clf} | — | — | — | — | — |")
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
                lines.append(f"| {clf} | {m:+.3f} | {lo:+.3f} | {hi:+.3f} | {frac:.2f} | {read} |")

    lines.append("\n## Cierre direccional\n")
    lines.append("Lectura propositiva basada en las diferencias clave bajo N-strict (la primaria):\n")
    lines.append("- Caso **target** (A+D > D-only sin C+D ≈ A+D): Phideus aporta sobre baseline. Plan mode Fase 1.")
    lines.append("- Caso **ambiguo** (A+D > D-only ≈ C+D > D-only): mejora no atribuible a ratios. Replantear A.")
    lines.append("- Caso **negativo** (A+D ≈ D-only): A no aporta sobre eGeMAPS. Reconsiderar antes de Fase 1.")
    lines.append("\nDecisión formal sobre escenario y GO/NO-GO queda en manos del usuario al cierre.\n")

    lines.append("\n## Caveats metodológicos (declarados)\n")
    lines.append("- **Single-speaker validation**: el grid search se evalúa con 1 held-out validation speaker por fold (val = `speakers[(k+1) % 10]`). Decisión de hiperparámetros por fold puede ser ruidosa — no es selección estable.")
    lines.append("- **N-adapt agrega intra-speaker**: para cada speaker se promedia UAR de las 3 repeticiones de calibración (seeds 42/43/44), después se agrega sobre los 10 speakers. La unidad de varianza final es speaker, no speaker×repeat.")
    lines.append("- **Bootstrap con n=10 speakers**: señal comparativa, no prueba fuerte. Generalización honesta requiere Fase 3.")
    lines.append("- **N-strict no per-speaker en test**: train tiene per-speaker z-score, test no — covariate shift es parte de la pregunta speaker-independent.")
    lines.append("- **N-adapt 25 utts label-agnostic**: calibración random, sin estratificar por emoción. Las 25 utts se excluyen de evaluación (no hay leakage).")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Directory with uar_results.json + predictions.npz")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output_dir)
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    records = load_uar(in_dir / "uar_results.json")
    logger.info("Loaded %d records", len(records))

    # Speaker list (sorted)
    speakers = sorted(set(r["test_speaker"] for r in records))
    table = build_uar_table(records)

    # Diff bootstrap
    diff_table: Dict[Tuple[str, str, str, str], dict] = {}
    diff_export = []
    for norm in NORM_CONDITIONS:
        for cfg_a, cfg_b in KEY_DIFFS:
            for clf in CLASSIFIERS:
                key_a = (norm, cfg_a, clf)
                key_b = (norm, cfg_b, clf)
                if key_a not in table or key_b not in table:
                    continue
                uars_a = per_speaker_array(table[key_a], speakers)
                uars_b = per_speaker_array(table[key_b], speakers)
                mean_d, lo, hi, frac = bootstrap_diff_ci(uars_a, uars_b)
                diff_table[(norm, cfg_a, cfg_b, clf)] = {
                    "mean": mean_d, "ci_lo": lo, "ci_hi": hi, "frac_pos": frac,
                }
                diff_export.append({
                    "norm_condition": norm, "cfg_a": cfg_a, "cfg_b": cfg_b, "clf": clf,
                    "mean_diff": mean_d, "ci95_lo": lo, "ci95_hi": hi, "p_diff_positive": frac,
                })

    (out_dir / "diff_bootstrap.json").write_text(json.dumps(diff_export, indent=2))
    logger.info("Saved diff_bootstrap.json")

    # Plots
    plot_uar_comparison(table, speakers, out_dir / "uar_comparison.png")
    logger.info("Saved uar_comparison.png")

    predictions = load_predictions(in_dir / "predictions.npz")
    plot_confusion(predictions, cm_dir)
    logger.info("Saved confusion matrices to %s", cm_dir)

    # Write report
    write_report(out_dir / "REPORTE_0B.md", table, speakers, diff_table, len(records))
    logger.info("Saved REPORTE_0B.md")


if __name__ == "__main__":
    main()
