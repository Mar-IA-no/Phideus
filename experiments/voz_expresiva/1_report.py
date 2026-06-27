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
MECHANISMS = ("concat", "film", "xattn")


def get_speaker_pool(records: List[Dict]) -> List[str]:
    """Return sorted list of unique test_speaker IDs present in the records."""
    return sorted({r["test_speaker"] for r in records})


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


def per_speaker_array(
    spk_map: Dict[str, float], speaker_pool: List[str],
) -> np.ndarray:
    return np.array([spk_map.get(s, np.nan) for s in speaker_pool], dtype=np.float64)


def deltas_per_speaker(
    uar_per_speaker: Dict[Tuple[str, str], Dict[str, float]],
    speaker_pool: List[str],
) -> Dict[Tuple[str, str], np.ndarray]:
    """Build {(mechanism, norm): per_speaker_delta_array} = arr_mech - arr_none."""
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for norm in NORMS:
        arr_base = per_speaker_array(uar_per_speaker.get(("none", norm), {}), speaker_pool)
        for mech in MECHANISMS:
            arr_mech = per_speaker_array(uar_per_speaker.get((mech, norm), {}), speaker_pool)
            out[(mech, norm)] = arr_mech - arr_base
    return out


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


def bootstrap_cross_language_shift(
    arr_en: np.ndarray, arr_zh: np.ndarray,
    n_resamples: int = 1000, seed: int = 42,
) -> Dict[str, float]:
    """Independent bootstrap of mean(arr_zh) - mean(arr_en).

    Each resample: sample with replacement from arr_en (size n_en) and arr_zh
    (size n_zh) independently. Compute mean(zh_boot) - mean(en_boot).

    Estatuto: exploratorio, no pareado (speakers EN y ZH son distintos).
    Devuelve dict con shift_mean, ci95_lo, ci95_hi, includes_zero.
    """
    rng = np.random.RandomState(seed)
    a = arr_en[np.isfinite(arr_en)]
    b = arr_zh[np.isfinite(arr_zh)]
    if len(a) < 2 or len(b) < 2:
        return {
            "shift_mean": float("nan"), "ci95_lo": float("nan"),
            "ci95_hi": float("nan"), "includes_zero": True,
            "n_en": int(len(a)), "n_zh": int(len(b)),
        }
    na = len(a); nb = len(b)
    shifts = np.empty(n_resamples)
    for i in range(n_resamples):
        idx_a = rng.randint(0, na, size=na)
        idx_b = rng.randint(0, nb, size=nb)
        shifts[i] = b[idx_b].mean() - a[idx_a].mean()
    ci_lo, ci_hi = np.percentile(shifts, [2.5, 97.5])
    return {
        "shift_mean": float(b.mean() - a.mean()),
        "ci95_lo": float(ci_lo),
        "ci95_hi": float(ci_hi),
        "includes_zero": bool(ci_lo <= 0 <= ci_hi),
        "n_en": int(na),
        "n_zh": int(nb),
    }


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
    config: str, norm: str, speaker_pool: List[str],
) -> np.ndarray:
    """Per-speaker mean of CKA across 3 seeds."""
    arr = np.full(len(speaker_pool), np.nan)
    for fold_idx, _spk in enumerate(speaker_pool):
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
    speaker_pool: List[str],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, norm in zip(axes, NORMS):
        means, sems, labels = [], [], []
        for cfg in CONFIGS:
            arr = per_speaker_array(uar_per_speaker.get((cfg, norm), {}), speaker_pool)
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
    speaker_pool: List[str],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, norm in zip(axes, NORMS):
        means, sems, labels = [], [], []
        for mech in MECHANISMS:
            arr = cka_per_speaker.get((mech, norm), np.full(len(speaker_pool), np.nan))
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
    speaker_pool: List[str],
    compare_info: Dict | None = None,
) -> None:
    """Write Fase 1 report. If compare_info is provided, augment with cross-language sections.

    compare_info structure:
        {
            "label_self": "ZH", "label_other": "EN",
            "diff_table_other": {(mech, norm): {mean, ci_lo, ci_hi, frac_pos}},
            "shift_bootstrap": {(mech, norm): {shift_mean, ci95_lo, ci95_hi, includes_zero, n_en, n_zh}},
            "uar_per_speaker_other": Dict[(config, norm), Dict[speaker, mean_uar]],
            "speaker_pool_other": [...],
        }
    """
    lines: List[str] = []
    label_self = compare_info["label_self"] if compare_info else "EN"
    n_speakers = len(speaker_pool)
    lines.append(f"# Reporte Fase 1 — Voz Expresiva Phideus ({label_self})\n")
    lines.append(
        f"> LOSO CV con {n_speakers} hablantes {label_self}. 4 configs × 2 norm × 3 seeds × "
        f"{n_speakers} folds. Total {n_records} task results. Tono comparativo del piloto; con "
        f"n={n_speakers} speakers los CI bootstrap son señal, no prueba fuerte. Generalización "
        "honesta requiere Fase 3 (MSP-Podcast).\n"
    )

    for norm in NORMS:
        lines.append(f"\n## {'N-strict (primaria, sin per-speaker en test)' if norm == 'strict' else 'N-adapt (secundaria, 1 calib repeat congelado seed=42)'}\n")
        lines.append(f"### UAR — mean ± std across {n_speakers} held-out speakers\n")
        lines.append("| Config | Mean UAR | Std |")
        lines.append("|---|---|---|")
        for cfg in CONFIGS:
            arr = per_speaker_array(uar_per_speaker.get((cfg, norm), {}), speaker_pool)
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
            arr = cka_per_speaker.get((mech, norm), np.full(n_speakers, np.nan))
            arr_f = arr[np.isfinite(arr)]
            if len(arr_f) == 0:
                lines.append(f"| {mech} | — | — |")
            else:
                lines.append(f"| {mech} | {arr_f.mean():.3f} | {arr_f.std():.3f} |")
        lines.append("")

    # Cross-language sections (only if compare_info provided)
    if compare_info:
        label_other = compare_info["label_other"]
        diff_other = compare_info["diff_table_other"]
        shift_boot = compare_info["shift_bootstrap"]

        lines.append(f"\n## Lectura cross-language {label_other} ↔ {label_self}\n")
        lines.append(
            f"Estas tablas comparan Δ (mecanismo vs WavLM-only) entre {label_other} y {label_self}. "
            "**Estatuto estadístico**: la tabla descriptiva NO es contraste formal — el solapamiento "
            "de CI95 separados es heurística visual, no test de equivalencia. El bootstrap "
            "exploratorio del shift sí da CI sobre la diferencia entre lenguas, pero como contraste "
            "independiente (no pareado — speakers distintos en cada lengua); reportado como secundario.\n"
        )

        for norm in NORMS:
            lines.append(f"### N-{norm}: tabla descriptiva Δ {label_other} ↔ Δ {label_self}\n")
            lines.append(
                f"| Mecanismo | Δ {label_other} | CI95 {label_other} | "
                f"Δ {label_self} | CI95 {label_self} |"
            )
            lines.append("|---|---|---|---|---|")
            for mech in MECHANISMS:
                e_other = diff_other.get((mech, norm), {})
                e_self = diff_table.get((mech, norm), {})
                if not e_other or not e_self:
                    lines.append(f"| {mech} | — | — | — | — |")
                    continue
                lines.append(
                    f"| {mech} | "
                    f"{e_other['mean']:+.3f} | [{e_other['ci_lo']:+.3f}, {e_other['ci_hi']:+.3f}] | "
                    f"{e_self['mean']:+.3f} | [{e_self['ci_lo']:+.3f}, {e_self['ci_hi']:+.3f}] |"
                )
            lines.append("")

            lines.append(
                f"### N-{norm}: shift exploratorio mean(Δ_{label_self}) - mean(Δ_{label_other}) "
                "(bootstrap independiente, 1000 resamples)\n"
            )
            lines.append("| Mecanismo | shift mean | CI95 lo | CI95 hi | Lectura |")
            lines.append("|---|---|---|---|---|")
            for mech in MECHANISMS:
                s = shift_boot.get((mech, norm))
                if not s or not np.isfinite(s["shift_mean"]):
                    lines.append(f"| {mech} | — | — | — | — |")
                    continue
                if s["includes_zero"]:
                    read = "CI incluye 0 — sin evidencia formal de shift"
                elif s["shift_mean"] > 0:
                    read = f"shift positivo: Δ_{label_self} > Δ_{label_other}"
                else:
                    read = f"shift negativo: Δ_{label_self} < Δ_{label_other}"
                lines.append(
                    f"| {mech} | {s['shift_mean']:+.3f} | {s['ci95_lo']:+.3f} | "
                    f"{s['ci95_hi']:+.3f} | {read} |"
                )
            lines.append("")

    lines.append("\n## Cierre direccional\n")
    lines.append("Cuatro escenarios prefigurados (decisión formal queda al usuario):\n")
    lines.append("- **WavLM-only > chance + algún mecanismo > WavLM-only en N-strict (CI95 excluye 0)**: target real. Phideus transfiere a SSL bajo generalización honesta.")
    lines.append("- **WavLM-only > chance, todos los mecanismos ≈ WavLM-only**: SSL resuelve, Phideus no agrega. Pivot a Carril B o Fase 1.2 con ajustes.")
    lines.append("- **Ninguno escapa N-strict, pero CKA muestra reorganización**: efecto geométrico sin funcional. Justifica Fase 1.2 con ajustes.")
    lines.append("- **Ninguno escapa N-strict y CKA no discrimina**: antes de Fase 3, considerar pooling alternativo / punto de inyección / baseline tuned / descriptor expandido.\n")

    lines.append("## Caveats metodológicos (declarados)\n")
    lines.append(f"- Single-speaker validation por fold (val_speaker = speakers[(k+1) % {n_speakers}]). Tuning ruidoso, no selección estable.")
    lines.append(f"- Bootstrap n={n_speakers} speakers: señal comparativa, no prueba fuerte. Generalización honesta requiere Fase 3.")
    lines.append("- ESD actuado: cualquier resultado requiere validación en habla naturalística.")
    lines.append("- N-adapt con 1 calib repeat (vs 3 repeats en 0B): lectura secundaria menos estable que la N-adapt de 0B. Documentado.")
    lines.append("- N-strict: train con per-speaker z-score, val/test con train-pool stats. Mismatch declarado.")
    lines.append("- CKA mide reorganización geométrica pre-classifier-head; interpretación afectiva no es directa.")
    lines.append("- 3 seeds es piloto; 5 requeriría 5/3× cómputo.")
    if compare_info:
        lines.append(
            f"- Cross-language {compare_info['label_other']} ↔ {label_self}: hablantes "
            "distintos en cada lengua; la atribución a 'idioma' no es estricta — puede tener "
            "componente cultural-actuativo difícil de aislar."
        )
        if label_self == "ZH":
            lines.append(
                "- Mandarín tonal: F0 codifica significado léxico. El descriptor armónico "
                "podría capturar varianza tonal además de afectiva."
            )
            lines.append(
                "- WavLM-large fue entrenado mayormente sobre habla inglesa. Su rendimiento "
                "sobre mandarín puede ser menor; los UAR absolutos no son comparables directamente "
                f"con {compare_info['label_other']}."
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _compute_diff_table(
    uar_per_speaker: Dict[Tuple[str, str], Dict[str, float]],
    speaker_pool: List[str],
) -> Tuple[Dict[Tuple[str, str], dict], List[dict]]:
    """Compute diff bootstrap intra-language (mechanism vs none, per norm)."""
    diff_table: Dict[Tuple[str, str], dict] = {}
    diff_export: List[dict] = []
    for norm in NORMS:
        arr_base = per_speaker_array(uar_per_speaker.get(("none", norm), {}), speaker_pool)
        for mech in MECHANISMS:
            arr_mech = per_speaker_array(
                uar_per_speaker.get((mech, norm), {}), speaker_pool,
            )
            m, lo, hi, frac = bootstrap_diff_ci(arr_mech, arr_base)
            diff_table[(mech, norm)] = {
                "mean": m, "ci_lo": lo, "ci_hi": hi, "frac_pos": frac,
            }
            diff_export.append({
                "norm_condition": norm, "mechanism": mech, "baseline": "none",
                "mean_diff": m, "ci95_lo": lo, "ci95_hi": hi, "p_diff_positive": frac,
            })
    return diff_table, diff_export


def _validate_completeness(records: List[Dict], dir_label: str) -> List[str]:
    """Guardrail: assert records cover all (config, norm, seed, speaker) combos exactly once.

    Aborts loudly if the results dir is partial (training interrupted, run in
    progress, missing combinations, or unexpected duplicates). Returns the
    detected speaker_pool.
    """
    speaker_pool = sorted({r["test_speaker"] for r in records})
    n_speakers = len(speaker_pool)
    expected = len(CONFIGS) * len(NORMS) * len(SEEDS) * n_speakers

    if len(records) != expected:
        raise RuntimeError(
            f"[{dir_label}] partial/inconsistent results: {len(records)} records "
            f"but expected {expected} = {len(CONFIGS)} configs × {len(NORMS)} norms × "
            f"{len(SEEDS)} seeds × {n_speakers} speakers. "
            f"Refusing to generate report over an incomplete experiment."
        )

    # Verify exact combinatorial coverage: each (config, norm, seed, speaker) exactly once
    seen = set()
    duplicates: List[Tuple[str, str, int, str]] = []
    for r in records:
        key = (r["config"], r["norm_condition"], r["seed"], r["test_speaker"])
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise RuntimeError(
            f"[{dir_label}] {len(duplicates)} duplicate (config, norm, seed, speaker) "
            f"combos detected. Sample: {duplicates[:3]}"
        )

    missing = []
    for cfg in CONFIGS:
        for norm in NORMS:
            for seed in SEEDS:
                for spk in speaker_pool:
                    if (cfg, norm, seed, spk) not in seen:
                        missing.append((cfg, norm, seed, spk))
    if missing:
        raise RuntimeError(
            f"[{dir_label}] {len(missing)} (config, norm, seed, speaker) combos missing. "
            f"Sample: {missing[:5]}"
        )

    logger.info(
        "[%s] completeness OK: %d records covering %d configs × %d norms × "
        "%d seeds × %d speakers", dir_label, len(records),
        len(CONFIGS), len(NORMS), len(SEEDS), n_speakers,
    )
    return speaker_pool


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, help="data/voz_expresiva/1 (self lang)")
    p.add_argument("--compare-against", default=None,
                   help="Optional: results-dir of another language for cross-language comparison")
    p.add_argument("--label-self", default=None,
                   help="Language label for self (default: inferred from results-dir name)")
    p.add_argument("--label-other", default=None,
                   help="Language label for the compare-against (default: inferred)")
    p.add_argument("--output-name", default="REPORTE_1.md",
                   help="Report markdown filename (default: REPORTE_1.md)")
    args = p.parse_args()

    res_dir = Path(args.results_dir)
    records = json.loads((res_dir / "uar_results.json").read_text())
    logger.info("Loaded %d records", len(records))

    speaker_pool = _validate_completeness(records, dir_label=res_dir.name)
    logger.info("Speaker pool (%d): %s", len(speaker_pool), speaker_pool)

    # Per-speaker UAR
    uar_per_speaker = aggregate_per_speaker(records, "test_uar")

    # Diff bootstrap: each mechanism vs none, per norm
    diff_table, diff_export = _compute_diff_table(uar_per_speaker, speaker_pool)

    (res_dir / "diff_bootstrap.json").write_text(json.dumps(diff_export, indent=2))
    logger.info("Saved diff_bootstrap.json")

    # CKA
    cka_map = compute_cka_all_runs(res_dir / "embeddings", records)
    cka_per_speaker: Dict[Tuple[str, str], np.ndarray] = {}
    cka_export: List[dict] = []
    for mech in MECHANISMS:
        for norm in NORMS:
            cka_arr = cka_per_speaker_array(cka_map, mech, norm, speaker_pool)
            cka_per_speaker[(mech, norm)] = cka_arr
            for spk_idx, spk in enumerate(speaker_pool):
                cka_export.append({
                    "mechanism": mech, "norm_condition": norm,
                    "speaker": spk, "fold_idx": spk_idx,
                    "cka_per_speaker_mean_over_seeds": float(cka_arr[spk_idx]) if np.isfinite(cka_arr[spk_idx]) else None,
                })
    (res_dir / "cka_per_run.json").write_text(json.dumps(cka_export, indent=2))
    logger.info("Saved cka_per_run.json")

    # Plots
    plot_uar_comparison(uar_per_speaker, speaker_pool, res_dir / "uar_comparison.png")
    logger.info("Saved uar_comparison.png")
    plot_cka_comparison(cka_per_speaker, speaker_pool, res_dir / "cka_comparison.png")
    logger.info("Saved cka_comparison.png")

    # Cross-language comparison (optional)
    compare_info = None
    if args.compare_against:
        other_dir = Path(args.compare_against)
        other_records = json.loads((other_dir / "uar_results.json").read_text())
        other_pool = _validate_completeness(other_records, dir_label=other_dir.name)
        logger.info(
            "Cross-language compare loaded: %d records from %s (pool=%d)",
            len(other_records), other_dir, len(other_pool),
        )

        other_uar = aggregate_per_speaker(other_records, "test_uar")
        other_diff_table, _ = _compute_diff_table(other_uar, other_pool)

        # Per-speaker deltas in each language
        deltas_self = deltas_per_speaker(uar_per_speaker, speaker_pool)
        deltas_other = deltas_per_speaker(other_uar, other_pool)

        # Cross-language shift bootstrap per (mech, norm)
        shift_bootstrap: Dict[Tuple[str, str], Dict[str, float]] = {}
        shift_export = {}
        for norm in NORMS:
            shift_export[norm] = {}
            for mech in MECHANISMS:
                # other = EN typically, self = ZH typically (but generic)
                arr_other = deltas_other.get((mech, norm), np.array([]))
                arr_self = deltas_self.get((mech, norm), np.array([]))
                shift = bootstrap_cross_language_shift(arr_other, arr_self)
                shift_bootstrap[(mech, norm)] = shift
                shift_export[norm][mech] = shift

        label_self = args.label_self or res_dir.name.split("_")[-1].upper() or "SELF"
        label_other = args.label_other or other_dir.name.split("_")[-1].upper() or "OTHER"
        # If labels collapse (e.g. both '1'), fall back to clearer defaults
        if label_self == label_other or label_self == "1":
            label_self = "ZH"
        if label_other == "1" or label_other == label_self:
            label_other = "EN"

        cross_lang_doc = {
            "estatuto": "exploratorio_independiente_no_pareado",
            "n_resamples": 1000,
            "label_self": label_self,
            "label_other": label_other,
            "n_speakers_self": len(speaker_pool),
            "n_speakers_other": len(other_pool),
            "shifts": shift_export,
            "ref_uar_self": str(res_dir / "uar_results.json"),
            "ref_uar_other": str(other_dir / "uar_results.json"),
            "note": (
                "shift_mean = mean(Δ_self) - mean(Δ_other). Bootstrap independiente "
                "(no pareado — speakers distintos en cada lengua). Estatuto secundario."
            ),
        }
        (res_dir / "cross_language_shift_bootstrap.json").write_text(
            json.dumps(cross_lang_doc, indent=2)
        )
        logger.info("Saved cross_language_shift_bootstrap.json")

        compare_info = {
            "label_self": label_self,
            "label_other": label_other,
            "diff_table_other": other_diff_table,
            "shift_bootstrap": shift_bootstrap,
            "uar_per_speaker_other": other_uar,
            "speaker_pool_other": other_pool,
        }

    # Report
    write_report(
        res_dir / args.output_name,
        uar_per_speaker, diff_table, cka_per_speaker, len(records),
        speaker_pool, compare_info,
    )
    logger.info("Saved %s", args.output_name)


if __name__ == "__main__":
    main()
