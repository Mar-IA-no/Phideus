#!/usr/bin/env python3
"""Fase 0A — analyze descriptors + generate plots + REPORTE_0A.md.

Loads the NPZ produced by `0A_extract.py`, applies z-score per-speaker normalization
(transductive over the whole corpus, declared as such), and produces:

  - PCA/UMAP plots per family + compound, coloured by emotion AND by speaker
  - Boxplots per dim (grouped by family) for emotion and speaker
  - Univariate ranking (eta², mutual information, Kruskal-Wallis) per dim
  - Silhouette per family and for the compound vector
  - Variance decomposition: intra-speaker (between emotions) vs inter-speaker (between persons)
  - REPORTE_0A.md — human synthesis closing with "¿qué familia justifica Fase 0B?"

Run:
    python experiments/voz_expresiva/0A_analyze.py \\
        --input data/esd/descriptors_0A_en.npz \\
        --output-dir data/visualizations/voz_expresiva/0A
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


# ---------------------------------------------------------------------------
# Loading + normalization
# ---------------------------------------------------------------------------

def load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


def zscore_per_speaker(X: np.ndarray, speakers: np.ndarray) -> np.ndarray:
    """Z-score normalization per speaker (transductive over the whole corpus).

    NaN-safe: uses nanmean/nanstd; missing values stay NaN after normalization.
    """
    X = X.astype(np.float64)
    out = np.empty_like(X)
    for spk in np.unique(speakers):
        mask = speakers == spk
        sub = X[mask]
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(sub, axis=0, keepdims=True)
            sd = np.nanstd(sub, axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        out[mask] = (sub - mu) / sd
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Univariate stats per dim against categorical label
# ---------------------------------------------------------------------------

def eta_squared(x: np.ndarray, labels: np.ndarray) -> float:
    """ANOVA eta² (variance explained by category).

    NaN-safe: skipped if too many NaNs or zero variance.
    """
    mask = np.isfinite(x)
    if mask.sum() < len(np.unique(labels)) * 2:
        return float("nan")
    x = x[mask]
    labs = labels[mask]
    grand_mean = x.mean()
    ss_total = ((x - grand_mean) ** 2).sum()
    if ss_total < 1e-12:
        return float("nan")
    ss_between = 0.0
    for c in np.unique(labs):
        sub = x[labs == c]
        if len(sub) == 0:
            continue
        ss_between += len(sub) * (sub.mean() - grand_mean) ** 2
    return float(ss_between / ss_total)


def mutual_information(x: np.ndarray, labels: np.ndarray, n_bins: int = 16) -> float:
    """Discrete MI between a binned scalar and categorical label."""
    mask = np.isfinite(x)
    if mask.sum() < 20:
        return float("nan")
    x = x[mask]
    labs = labels[mask]
    try:
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            return float("nan")
        x_bin = np.digitize(x, edges[1:-1])
    except Exception:
        return float("nan")
    n = len(x_bin)
    mi = 0.0
    for c in np.unique(labs):
        for b in np.unique(x_bin):
            p_cb = np.mean((labs == c) & (x_bin == b))
            p_c = np.mean(labs == c)
            p_b = np.mean(x_bin == b)
            if p_cb > 0 and p_c > 0 and p_b > 0:
                mi += p_cb * np.log(p_cb / (p_c * p_b))
    return float(mi)


def kruskal_wallis_p(x: np.ndarray, labels: np.ndarray) -> float:
    """Kruskal-Wallis H-test p-value."""
    from scipy.stats import kruskal
    mask = np.isfinite(x)
    if mask.sum() < len(np.unique(labels)) * 2:
        return float("nan")
    groups = [x[mask & (labels == c)] for c in np.unique(labels)]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan")
    try:
        _, p = kruskal(*groups)
        return float(p)
    except Exception:
        return float("nan")


def univariate_ranking(
    X: np.ndarray, labels: np.ndarray, names: List[str],
) -> List[dict]:
    """Per-column ranking against the categorical label."""
    out = []
    for i, n in enumerate(names):
        col = X[:, i]
        out.append({
            "name": n,
            "eta_squared": eta_squared(col, labels),
            "mutual_information": mutual_information(col, labels),
            "kruskal_wallis_p": kruskal_wallis_p(col, labels),
        })
    return out


# ---------------------------------------------------------------------------
# Silhouette per family / compound
# ---------------------------------------------------------------------------

def safe_silhouette(X: np.ndarray, labels: np.ndarray, sample_size: int = 4000) -> float:
    from sklearn.metrics import silhouette_score
    mask = np.all(np.isfinite(X), axis=1)
    if mask.sum() < 50:
        return float("nan")
    X = X[mask]
    labs = labels[mask]
    if len(np.unique(labs)) < 2:
        return float("nan")
    if X.shape[0] > sample_size:
        rng = np.random.RandomState(0)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X = X[idx]
        labs = labs[idx]
    try:
        return float(silhouette_score(X, labs, metric="euclidean"))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Variance decomposition: intra-speaker (between emotions) vs inter-speaker
# ---------------------------------------------------------------------------

def variance_decomposition(
    X: np.ndarray, speakers: np.ndarray, emotions: np.ndarray, names: List[str],
) -> Dict[str, dict]:
    """For each dim: intra-spk variance vs inter-spk variance.

    Intra-spk var = mean over speakers of var(emotion-means within that speaker).
    Inter-spk var = var(speaker-mean emotion-means over speakers).
    Ratio intra/(intra+inter) → close to 1 means emotion modulation dominates within-speaker.
    """
    out = {}
    for i, n in enumerate(names):
        col = X[:, i]
        # Per (speaker, emotion) mean
        cell_means = defaultdict(list)
        for v, s, e in zip(col, speakers, emotions):
            if np.isfinite(v):
                cell_means[(s, e)].append(v)
        cell_avg = {k: float(np.mean(v)) for k, v in cell_means.items() if v}
        if not cell_avg:
            out[n] = {"intra_spk_var": float("nan"), "inter_spk_var": float("nan"),
                      "intra_share": float("nan")}
            continue
        # intra-spk: var of emotion-means within each speaker, averaged across speakers
        intra_vars = []
        spk_emotion_means = defaultdict(dict)
        for (s, e), m in cell_avg.items():
            spk_emotion_means[s][e] = m
        for s, em_map in spk_emotion_means.items():
            if len(em_map) >= 2:
                intra_vars.append(float(np.var(list(em_map.values()))))
        intra = float(np.mean(intra_vars)) if intra_vars else float("nan")

        # inter-spk: var of (speaker-mean over emotions) across speakers
        spk_means = [float(np.mean(list(em.values()))) for em in spk_emotion_means.values()
                     if em]
        inter = float(np.var(spk_means)) if len(spk_means) >= 2 else float("nan")

        denom = (intra if np.isfinite(intra) else 0) + (inter if np.isfinite(inter) else 0)
        intra_share = (intra / denom) if denom > 0 else float("nan")
        out[n] = {"intra_spk_var": intra, "inter_spk_var": inter, "intra_share": intra_share}
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_pca(X: np.ndarray, labels: np.ndarray, title: str, out_path: Path,
              label_name: str = "label") -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    mask = np.all(np.isfinite(X), axis=1)
    if mask.sum() < 30:
        logger.warning("Skipping PCA %s — only %d finite rows", out_path.name, mask.sum())
        return
    Xc = X[mask]
    labs = labels[mask]
    pca = PCA(n_components=2, random_state=0).fit_transform(Xc)
    fig, ax = plt.subplots(figsize=(7, 5))
    cats = sorted(set(labs))
    for c in cats:
        idx = labs == c
        ax.scatter(pca[idx, 0], pca[idx, 1], s=4, alpha=0.5, label=str(c))
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if len(cats) <= 12:
        ax.legend(title=label_name, markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_umap(X: np.ndarray, labels: np.ndarray, title: str, out_path: Path,
               label_name: str = "label") -> None:
    import matplotlib.pyplot as plt
    import umap
    mask = np.all(np.isfinite(X), axis=1)
    if mask.sum() < 100:
        return
    Xc = X[mask]
    labs = labels[mask]
    try:
        reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=15, min_dist=0.1)
        emb = reducer.fit_transform(Xc)
    except Exception as exc:
        logger.warning("UMAP failed for %s: %s", out_path.name, exc)
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    cats = sorted(set(labs))
    for c in cats:
        idx = labs == c
        ax.scatter(emb[idx, 0], emb[idx, 1], s=4, alpha=0.5, label=str(c))
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    if len(cats) <= 12:
        ax.legend(title=label_name, markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_family_boxplots(
    X: np.ndarray, labels: np.ndarray, names: List[str],
    title: str, out_path: Path, max_dims: int = 24,
) -> None:
    import matplotlib.pyplot as plt
    n_dims = min(X.shape[1], max_dims)
    cols = 4
    rows = int(np.ceil(n_dims / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5), sharey=False)
    axes = np.array(axes).flatten()
    cats = sorted(set(labels))
    for i in range(n_dims):
        ax = axes[i]
        data = [X[(labels == c) & np.isfinite(X[:, i]), i] for c in cats]
        ax.boxplot(data, tick_labels=cats, showfliers=False)
        ax.set_title(names[i], fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=7)
    for j in range(n_dims, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="NPZ from 0A_extract.py")
    p.add_argument("--output-dir", required=True, help="Output directory for plots + JSONs")
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    (out_dir / "pca").mkdir(parents=True, exist_ok=True)
    (out_dir / "umap").mkdir(parents=True, exist_ok=True)
    (out_dir / "boxplots").mkdir(parents=True, exist_ok=True)

    data = load_npz(in_path)
    speakers = data["speaker_ids"]
    emotions = data["emotion_labels"]

    fams: Dict[str, Tuple[np.ndarray, List[str]]] = {
        "A": (data["family_A_pooled"], list(data["family_A_names"])),
        "B": (data["family_B"], list(data["family_B_names"])),
        "C": (data["family_C_pooled"], list(data["family_C_names"])),
        "D": (data["family_D_egemaps"], list(data["family_D_names"])),
    }
    compound_X = data["compound_pooled"]
    compound_names = list(data["family_A_names"]) + list(data["family_B_names"]) + \
        list(data["family_C_names"])

    # Z-score per speaker (transductive)
    logger.info("Normalizing per-speaker (transductive)…")
    fams_z = {k: (zscore_per_speaker(v[0], speakers), v[1]) for k, v in fams.items()}
    compound_X_z = zscore_per_speaker(compound_X, speakers)

    # Plots — PCA + UMAP per family + compound
    logger.info("Generating PCA/UMAP plots…")
    for fname, (X, _) in fams_z.items():
        _save_pca(X, emotions, f"Familia {fname} — PCA, emotion",
                  out_dir / "pca" / f"pca_familia_{fname}_by_emotion.png", "emotion")
        _save_pca(X, speakers, f"Familia {fname} — PCA, speaker (sanity)",
                  out_dir / "pca" / f"pca_familia_{fname}_by_speaker.png", "speaker")
        _save_umap(X, emotions, f"Familia {fname} — UMAP, emotion",
                   out_dir / "umap" / f"umap_familia_{fname}_by_emotion.png", "emotion")
    _save_pca(compound_X_z, emotions, "Compuesto A+B+C — PCA, emotion",
              out_dir / "pca" / "pca_compound_by_emotion.png", "emotion")
    _save_pca(compound_X_z, speakers, "Compuesto A+B+C — PCA, speaker (sanity)",
              out_dir / "pca" / "pca_compound_by_speaker.png", "speaker")
    _save_umap(compound_X_z, emotions, "Compuesto A+B+C — UMAP, emotion",
               out_dir / "umap" / "umap_compound_by_emotion.png", "emotion")

    # Boxplots per family (up to first 24 dims for legibility)
    logger.info("Generating boxplots…")
    for fname, (X, names) in fams_z.items():
        _save_family_boxplots(
            X, emotions, names,
            title=f"Familia {fname} — distribución por emoción (z-spk)",
            out_path=out_dir / "boxplots" / f"boxplots_familia_{fname}_by_emotion.png",
        )

    # Univariate ranking
    logger.info("Univariate ranking (eta², MI, KW)…")
    rankings = {}
    for fname, (X, names) in fams_z.items():
        rankings[fname] = univariate_ranking(X, emotions, names)
    (out_dir / "ranking_univariate.json").write_text(json.dumps(rankings, indent=2))

    # Silhouette per family + compound
    logger.info("Silhouette per family…")
    sil = {}
    for fname, (X, _) in fams_z.items():
        sil[fname] = safe_silhouette(X, emotions)
    sil["compound"] = safe_silhouette(compound_X_z, emotions)
    (out_dir / "silhouette_per_family.json").write_text(json.dumps(sil, indent=2))

    # Variance decomposition (raw + normalized)
    logger.info("Variance decomposition…")
    var_decomp = {"raw": {}, "zspk": {}}
    for fname, (X, names) in fams.items():
        var_decomp["raw"][fname] = variance_decomposition(X, speakers, emotions, names)
        var_decomp["zspk"][fname] = variance_decomposition(
            fams_z[fname][0], speakers, emotions, names
        )
    (out_dir / "variance_decomposition.json").write_text(json.dumps(var_decomp, indent=2))

    # Human report
    write_report(out_dir, sil, rankings, var_decomp, data["meta"])
    logger.info("Done. Outputs in %s", out_dir)


def write_report(out_dir: Path, sil: dict, rankings: dict, var_decomp: dict,
                 meta: dict) -> None:
    """Synthesize REPORTE_0A.md with human-readable conclusions."""
    lines = []
    lines.append(f"# Reporte Fase 0A — Voz Expresiva Phideus\n")
    lines.append(f"> Dataset: {meta.get('dataset')}, language={meta.get('language')}, "
                 f"N={meta.get('n_utterances')} utterances, "
                 f"speakers={meta.get('speakers')}.")
    lines.append(f"> Normalización: z-score por hablante intra-corpus, transductiva sobre todo {meta.get('language')}.\n")

    lines.append("## Silhouette por familia (emoción)\n")
    for fname in ("A", "B", "C", "D", "compound"):
        v = sil.get(fname, float("nan"))
        lines.append(f"- **{fname}**: {v:.3f}")
    lines.append("")

    lines.append("## Top-5 dimensiones por familia según eta² (variance explained por emoción)\n")
    for fname in ("A", "B", "C", "D"):
        ranks = sorted(
            rankings[fname], key=lambda r: r["eta_squared"] if np.isfinite(r["eta_squared"]) else -1,
            reverse=True,
        )[:5]
        lines.append(f"### Familia {fname}\n")
        lines.append("| dim | eta² | MI | KW p-value |")
        lines.append("|---|---|---|---|")
        for r in ranks:
            lines.append(f"| `{r['name']}` | {r['eta_squared']:.3f} | "
                         f"{r['mutual_information']:.3f} | {r['kruskal_wallis_p']:.2e} |")
        lines.append("")

    lines.append("## Descomposición de varianza (z-spk): intra-spk share por familia\n")
    lines.append("Promedio de `intra_spk_var / (intra_spk_var + inter_spk_var)` sobre dims con datos válidos. "
                 "Cerca de 1 → la emoción modula intra-hablante más que la identidad inter-hablante.\n")
    for fname in ("A", "B", "C", "D"):
        shares = [v["intra_share"] for v in var_decomp["zspk"][fname].values()
                  if np.isfinite(v["intra_share"])]
        mean_share = float(np.mean(shares)) if shares else float("nan")
        lines.append(f"- **{fname}**: intra-share medio = {mean_share:.3f}  (n_dims con datos: {len(shares)})")
    lines.append("")

    lines.append("## Notas metodológicas (fijas)\n")
    lines.append("- Familia B incluye 7 medidas directas (HNR, CPP, jitter, shimmer, F2/F1, F3/F1, alpha-ratio) "
                 "y 2 **proxies acústicos** (H1-H2_proxy, H1-A3_proxy) sin corrección formántica. Los proxies "
                 "no son medidas clínicas completas — su lectura debe leerse con cautela.")
    lines.append("- Normalización transductiva: todas las stats se computaron sobre todo el corpus. Fase 0B "
                 "deberá redefinir splits antes de re-normalizar.")
    lines.append("- Familia D (eGeMAPSv02) es baseline industrial; sus 88 functionals son utterance-level, sin "
                 "re-pooling.")
    lines.append("- Familias A y C son frame-level y fueron pooled con 4 estadísticos (mean, std, max, min).\n")

    lines.append("## Cierre operativo — ¿qué familia justifica Fase 0B?\n")
    lines.append("Lectura propuesta (no es GO/NO-GO formal; lo decide el usuario):\n")
    lines.append("- Si silhouette[A] o silhouette[B] supera silhouette[C] (control no-ratio) por margen visible "
                 "→ Familia ratio-based aporta señal específica. Recomendable seguir Fase 0B con esa familia.")
    lines.append("- Si silhouette[D] (eGeMAPS) lidera y A/B no se acercan → los descriptores Phideus no aportan "
                 "encima de features industriales; reconsiderar composición antes de Fase 0B.")
    lines.append("- Si ninguna familia discrimina → hallazgo válido; Fase 0B replantea scope o cierra Carril A.\n")
    lines.append("Decisión pendiente del usuario al cierre de la fase.\n")

    (out_dir / "REPORTE_0A.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
