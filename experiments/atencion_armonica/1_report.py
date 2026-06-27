#!/usr/bin/env python3
"""Reporte de Fase 0 Atención Armónica — agregación multi-seed + contrastes + REPORTE_0.md.

Lee results.json + test_pairs/*.npz + test_ari/*.npz de 1_train_grouping.py.

Agregación multi-seed (convención del proyecto): se PROMEDIAN los logits de los 3 seeds
por par de test (alineados por mixture_id), luego se hace bootstrap PAREADO sobre mezclas.
Esto captura incertidumbre de muestreo de mezclas; el spread entre seeds se reporta aparte
como mean±std de la F1 pooled por seed (transparencia).

Contrastes (por celda polifonía×régimen × run):
    PRIMARIO   B vs A-rich      (maquinaria pair-state+triangle con features igualadas)
    PRIMARIO   B vs B-local     (transitividad / suma sobre k, param-matched)
    SECUNDARIO B vs B-minus     (módulo triangle completo, params incluidos) — NO se eleva sobre B-local
    LATERAL    A-rich vs A-naive(aporte de las pair features solas)

Métricas: F1 pairwise (primaria, upper-tri válido), AP/AUC threshold-free (nanmean en celdas
degeneradas poly1), ARI (τ de val) + ARI@0.5.

Uso:
    python experiments/atencion_armonica/1_report.py --results-dir data/atencion_armonica/fase0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.atencion_armonica.harness import bootstrap_diff_ci, pairwise_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = ("A-naive", "A-rich", "B", "B-minus", "B-local", "B-shuffle")
RUNS = ("ID", "OOD-poly", "OOD-regime")
CONTRASTS = [
    ("B", "A-rich", "PRIMARIO"),
    ("B", "B-local", "PRIMARIO"),
    ("B", "B-minus", "secundario"),
    ("A-rich", "A-naive", "lateral"),
    ("B", "B-shuffle", "control-neg-parcial"),
]


def _cell_key(poly, regime) -> str:
    return f"poly{poly}_{regime}"


def load_seed_avg_pairs(results_dir: Path, run: str, model: str, seeds: List[int]) -> Dict[int, Dict]:
    """Carga test_pairs de los seeds, alinea por mixture_id, PROMEDIA logits sobre seeds.

    Returns: {mixture_id: {logit:[P], target:[P], polyphony, regime}}.
    Verifica alineación: mismos mixture_ids y mismo #pares por mezcla entre seeds.
    """
    per_seed = []
    for s in seeds:
        f = results_dir / "test_pairs" / f"{run}__{model}__seed{s}.npz"
        d = np.load(f, allow_pickle=True)
        by_mix: Dict[int, Dict] = defaultdict(lambda: {"logit": [], "target": [], "poly": None, "regime": None})
        mix_ids = d["mix_id"]; logit = d["logit"]; target = d["target"]
        poly = d["polyphony"]; regime = d["regime"]
        # agrupar pares por mezcla preservando orden (triu determinístico)
        order = defaultdict(list)
        for idx in range(len(mix_ids)):
            order[int(mix_ids[idx])].append(idx)
        grouped = {}
        for m, idxs in order.items():
            idxs = np.array(idxs)
            grouped[m] = {
                "logit": logit[idxs].astype(np.float64),
                "target": target[idxs].astype(np.float64),
                "poly": int(poly[idxs[0]]), "regime": str(regime[idxs[0]]),
            }
        per_seed.append(grouped)

    # alineación: mismos mixture_ids en todos los seeds
    base_ids = set(per_seed[0].keys())
    for g in per_seed[1:]:
        assert set(g.keys()) == base_ids, f"{run}/{model}: mixture_ids difieren entre seeds"

    out = {}
    for m in sorted(base_ids):
        # mismo #pares por mezcla entre seeds (Codex: alineación por mixture_id)
        lens = [len(per_seed[s][m]["logit"]) for s in range(len(seeds))]
        assert len(set(lens)) == 1, f"{run}/{model} mezcla {m}: #pares difiere entre seeds {lens}"
        # target IDÉNTICO entre seeds (Codex Medio #3): garantiza mismo orden de pares,
        # no solo mismo conteo, antes de promediar logits.
        t0 = per_seed[0][m]["target"]
        for s in range(1, len(seeds)):
            assert np.array_equal(per_seed[s][m]["target"], t0), \
                f"{run}/{model} mezcla {m}: target difiere entre seeds (orden de pares inconsistente)"
        logits = np.stack([per_seed[s][m]["logit"] for s in range(len(seeds))])  # [S,P]
        out[m] = {
            "logit": logits.mean(axis=0),                # logit-ensemble sobre seeds
            "target": t0,
            "polyphony": per_seed[0][m]["poly"],
            "regime": per_seed[0][m]["regime"],
        }
    return out


def per_mix_list(seed_avg: Dict[int, Dict], poly=None, regime=None) -> List[Dict]:
    """Lista alineada por mixture_id (ordenada), filtrada por celda si se pide."""
    out = []
    for m in sorted(seed_avg.keys()):
        e = seed_avg[m]
        if poly is not None and e["polyphony"] != poly:
            continue
        if regime is not None and e["regime"] != regime:
            continue
        out.append({"mixture_id": m, "logit": e["logit"], "target": e["target"]})
    return out


def cell_point_metrics(seed_avg: Dict[int, Dict], poly, regime) -> Dict:
    pm = per_mix_list(seed_avg, poly, regime)
    if not pm:
        return {}
    lg = np.concatenate([m["logit"] for m in pm])
    tg = np.concatenate([m["target"] for m in pm])
    return pairwise_metrics(lg, tg)


def _seed_vals(results, run, model, poly, regime, metric) -> List[float]:
    key = _cell_key(poly, regime)
    vals = [r["test"]["by_cell"].get(key, {}).get(metric)
            for r in results if r["run"] == run and r["model"] == model]
    return [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]


def seed_spread(results, run, model, poly, regime, metric="f1") -> str:
    """mean±std de una métrica pooled por seed en una celda (transparencia del spread entre seeds)."""
    vals = _seed_vals(results, run, model, poly, regime, metric)
    if not vals:
        return "—"
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"


def per_seed_contrast(results, run, a, b, poly, regime, metric="f1") -> str:
    """ΔF1 por-seed (F1(a,seed) - F1(b,seed) pooled) → mean±std sobre seeds. Codex Medio #2."""
    key = _cell_key(poly, regime)
    by_seed_a = {r["seed"]: r["test"]["by_cell"].get(key, {}).get(metric)
                 for r in results if r["run"] == run and r["model"] == a}
    by_seed_b = {r["seed"]: r["test"]["by_cell"].get(key, {}).get(metric)
                 for r in results if r["run"] == run and r["model"] == b}
    diffs = [by_seed_a[s] - by_seed_b[s] for s in by_seed_a
             if s in by_seed_b and by_seed_a[s] is not None and by_seed_b[s] is not None]
    if not diffs:
        return "—"
    return f"{np.mean(diffs):+.3f}±{np.std(diffs):.3f}"


def tau_spread(results, run, model) -> str:
    """mean±std de τ por (run, model) sobre seeds (Codex Bajo #5)."""
    vals = [r["tau"] for r in results if r["run"] == run and r["model"] == model]
    if not vals:
        return "—"
    return f"{np.mean(vals):.2f}±{np.std(vals):.2f}"


def load_ari(results_dir: Path, run, model, seeds) -> Dict:
    """ARI medio (τ de val) y ARI@0.5 por celda, nanmean sobre mezclas y seeds."""
    by_cell = defaultdict(list)
    by_cell05 = defaultdict(list)
    for s in seeds:
        f = results_dir / "test_ari" / f"{run}__{model}__seed{s}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        for i in range(len(d["ari"])):
            key = _cell_key(int(d["polyphony"][i]), str(d["regime"][i]))
            by_cell[key].append(float(d["ari"][i]))
            if "ari_tau05" in d:
                by_cell05[key].append(float(d["ari_tau05"][i]))
    ari = {k: float(np.nanmean(v)) if v else float("nan") for k, v in by_cell.items()}
    ari05 = {k: float(np.nanmean(v)) if v else float("nan") for k, v in by_cell05.items()}
    return {"ari": ari, "ari05": ari05}


def fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    rd = Path(args.results_dir)
    results = json.loads((rd / "results.json").read_text())
    runs = sorted({r["run"] for r in results}, key=lambda x: RUNS.index(x) if x in RUNS else 99)
    models = [m for m in MODELS if any(r["model"] == m for r in results)]
    logger.info("Runs: %s | Models: %s", runs, models)

    # seed-averaged pairs por (run, model)
    seed_avg = {}
    for run in runs:
        for model in models:
            seed_avg[(run, model)] = load_seed_avg_pairs(rd, run, model, args.seeds)

    # celdas presentes por run
    contrasts_out = []
    lines = ["# Reporte Fase 0 — Atención Armónica (Harmonic Pairformer)\n"]
    lines.append("> Agrupamiento armónico sobre mezclas sintéticas. Parciales exactos, acordes "
                 "estáticos, ground truth exacto. F1 pairwise primaria (upper-tri válido). "
                 "Multi-seed: logits promediados sobre 3 seeds → bootstrap pareado sobre mezclas. "
                 "poly1 es degenerada (todo-mismo-fuente): sanity, no evidencia.\n")
    lines.append("\n**Contrastes**: B vs A-rich y B vs B-local son PRIMARIOS. B vs B-minus es "
                 "secundario (módulo triangle con params incluidos) y NO se eleva por encima de "
                 "B vs B-local. A-rich vs A-naive es lateral (aporte de pair features solas).\n")

    for run in runs:
        lines.append(f"\n## Run {run}\n")
        cells = sorted({(e["polyphony"], e["regime"])
                        for sa in [seed_avg[(run, m)] for m in models] for e in sa.values()})

        cell_hdr = " | ".join(_cell_key(p, r) for (p, r) in cells)
        sep = "|" + "---|" * (len(cells) + 1)

        # tabla F1 por modelo × celda (mean±std sobre seeds)
        lines.append("### F1 pairwise por modelo × celda (mean±std sobre seeds)\n")
        lines.append(f"| Modelo | {cell_hdr} |")
        lines.append(sep)
        for model in models:
            row = [model] + [seed_spread(results, run, model, p, r, "f1") for (p, r) in cells]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # AP y ROC-AUC (threshold-free) por modelo × celda — Codex Medio #1
        for metric, label in (("ap", "AP/AUPRC"), ("roc_auc", "ROC-AUC")):
            lines.append(f"### {label} por modelo × celda (mean±std sobre seeds)\n")
            lines.append(f"| Modelo | {cell_hdr} |")
            lines.append(sep)
            for model in models:
                row = [model] + [seed_spread(results, run, model, p, r, metric) for (p, r) in cells]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        # contrastes: bootstrap (logit-ensemble) + ΔF1 per-seed (mean±std) — Codex Medio #2
        lines.append("### Contrastes Δ F1 — bootstrap pareado sobre mezclas (logit-ensemble 3 seeds) + ΔF1 per-seed\n")
        lines.append("> El CI95 es del bootstrap sobre mezclas con logits promediados (ensemble de 3 seeds). "
                     "ΔF1 per-seed (mean±std) acompaña como lectura single-seed, sin ensemble.\n")
        lines.append("| Contraste | Tipo | Celda | Δ F1 (ens) | CI95 | P(Δ>0) | ΔF1 per-seed |")
        lines.append("|---|---|---|---|---|---|---|")
        for (a, b, tipo) in CONTRASTS:
            if a not in models or b not in models:
                continue  # robusto a corridas parciales (smoke con subconjunto de modelos)
            for (poly, regime) in cells:
                if poly < 2:
                    continue  # poly1 degenerada, F1 trivial
                pm_a = per_mix_list(seed_avg[(run, a)], poly, regime)
                pm_b = per_mix_list(seed_avg[(run, b)], poly, regime)
                if not pm_a or not pm_b:
                    continue
                res = bootstrap_diff_ci(pm_a, pm_b, metric="f1", n_boot=args.n_boot)
                ci = f"[{res['ci95_lo']:+.3f}, {res['ci95_hi']:+.3f}]"
                psd = per_seed_contrast(results, run, a, b, poly, regime, "f1")
                lines.append(f"| {a} vs {b} | {tipo} | {_cell_key(poly,regime)} | "
                             f"{res['mean_diff']:+.3f} | {ci} | {res['frac_positive']:.2f} | {psd} |")
                contrasts_out.append({
                    "run": run, "contrast": f"{a} vs {b}", "tipo": tipo,
                    "cell": _cell_key(poly, regime), "per_seed_dF1": psd, **res,
                })
        lines.append("")

        # ARI (τ de val) y ARI@0.5 por modelo × celda — Codex Medio #1
        for ari_key, ari_label in (("ari", "ARI (τ de val)"), ("ari05", "ARI@0.5 (fijo)")):
            lines.append(f"### {ari_label} por modelo × celda\n")
            lines.append(f"| Modelo | {cell_hdr} |")
            lines.append(sep)
            for model in models:
                ad = load_ari(rd, run, model, args.seeds)[ari_key]
                row = [model] + [fmt(ad.get(_cell_key(p, r))) for (p, r) in cells]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        # τ por modelo (mean±std sobre seeds) — Codex Bajo #5
        lines.append("### τ elegido en val por modelo (mean±std sobre seeds)\n")
        lines.append("| Modelo | τ |")
        lines.append("|---|---|")
        for model in models:
            lines.append(f"| {model} | {tau_spread(results, run, model)} |")
        lines.append("")

    # cierre direccional
    lines.append("\n## Lectura direccional (GO/NO-GO lo decide el usuario)\n")
    lines.append("- **B>A-rich y B>B-local en difícil/OOD (CI95 excluye 0)** → pair-state+triangle y "
                 "la transitividad hacen trabajo real → GO al programa arquitectónico.")
    lines.append("- **B≈A-rich** → la maquinaria no aporta sobre features armónicas en readout simple "
                 "→ NO construir la cosa grande.")
    lines.append("- **B>A-rich pero B≈B-local** → el pair-state ayuda pero la transitividad específica no.")
    lines.append("- **B-shuffle gana mucho** → alarma de capacidad (confound). Si no gana, refuerza.")
    lines.append("\n## Caveats\n")
    lines.append("- Parciales exactos + acordes estáticos: Fase 0 aísla la pregunta arquitectónica. "
                 "Detección CQT (Fase 1) y estructura temporal (Fase 2) son siguientes.")
    lines.append("- B vs B-minus mezcla triangle con params; por eso B vs B-local (param-matched) es el "
                 "contraste de transitividad, NO B vs B-minus.")
    lines.append("- AP/AUC NaN en poly1 (una sola clase); agregado con nanmean. poly1 = sanity.")
    lines.append("- Multi-seed: 3 seeds. Bootstrap sobre mezclas (logits promediados sobre seeds).")

    (rd / "REPORTE_0.md").write_text("\n".join(lines), encoding="utf-8")
    (rd / "contrasts.json").write_text(json.dumps(contrasts_out, indent=2))
    logger.info("Escrito REPORTE_0.md y contrasts.json (%d contrastes)", len(contrasts_out))


if __name__ == "__main__":
    main()
