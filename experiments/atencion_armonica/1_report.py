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

Métricas: **ARI es primaria para el claim de transitividad** (partición inducida; contraste con
bootstrap pareado sobre mezclas, seed-averaged). F1 pairwise + AP/AUC threshold-free son
secundarias (la AUC de B satura ~0.99). ARI (τ de val) + ARI@0.5 por modelo también se tabulan.

Uso:
    python experiments/atencion_armonica/1_report.py --results-dir data/atencion_armonica/fase0
    # smoke (subconjunto de modelos/seeds): agregar --allow-partial
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


def per_seed_contrast_ari(results, run, a, b, poly, regime) -> str:
    """ΔARI por-seed (ari_by_cell[a,seed] − ari_by_cell[b,seed]) → mean±std sobre seeds.
    Lectura secundaria que expone variabilidad de inicialización (Codex r10 #6)."""
    key = _cell_key(poly, regime)
    sa = {r["seed"]: r["test"].get("ari_by_cell", {}).get(key)
          for r in results if r["run"] == run and r["model"] == a}
    sb = {r["seed"]: r["test"].get("ari_by_cell", {}).get(key)
          for r in results if r["run"] == run and r["model"] == b}
    diffs = [sa[s] - sb[s] for s in sa
             if s in sb and sa[s] is not None and sb[s] is not None]
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


def load_ari_per_mix(results_dir: Path, run, model, seeds) -> Dict[int, Dict]:
    """ARI por mezcla promediado sobre seeds. → {mix_id: {ari, poly, regime}}.

    ARI es métrica de partición (no se puede promediar logits→ARI); se promedia el ARI
    por mezcla sobre los 3 seeds y se hace bootstrap PAREADO sobre mezclas. Codex r9 #1.
    """
    acc = defaultdict(lambda: {"ari": [], "poly": None, "regime": None})
    for s in seeds:
        f = results_dir / "test_ari" / f"{run}__{model}__seed{s}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        for i in range(len(d["ari"])):
            mid = int(d["mixture_id"][i])
            acc[mid]["ari"].append(float(d["ari"][i]))
            acc[mid]["poly"] = int(d["polyphony"][i])
            acc[mid]["regime"] = str(d["regime"][i])
    return {mid: {"ari": float(np.nanmean(v["ari"])) if v["ari"] else float("nan"),
                  "poly": v["poly"], "regime": v["regime"]}
            for mid, v in acc.items()}


def bootstrap_ari_diff(ari_a: Dict, ari_b: Dict, poly, regime, n_boot=1000, seed=42):
    """Bootstrap PAREADO sobre mezclas del ΔARI (a−b) en una celda.

    ari_a/ari_b: {mix_id: {ari, poly, regime}} (per-mezcla, seed-averaged). Se resamplean
    las MISMAS mezclas para a y b (pareado por mixture_id). Devuelve mean_diff, ci95, P(Δ>0).
    """
    common = sorted(set(ari_a) & set(ari_b))
    mids = [m for m in common
            if ari_a[m]["poly"] == poly and ari_a[m]["regime"] == regime
            and not np.isnan(ari_a[m]["ari"]) and not np.isnan(ari_b[m]["ari"])]
    if len(mids) < 2:
        return None
    a = np.array([ari_a[m]["ari"] for m in mids])
    b = np.array([ari_b[m]["ari"] for m in mids])
    M = len(mids)
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for t in range(n_boot):
        idx = rng.randint(0, M, size=M)
        diffs[t] = a[idx].mean() - b[idx].mean()        # mismo idx para a y b → pareado
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(a.mean() - b.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi),
            "frac_positive": float((diffs > 0).mean()), "n_mixtures": M}


def load_permix_metrics(results_dir: Path, run, model, seeds) -> Dict[int, Dict]:
    """{mix_id: {f1, auc, ap, poly, reg}} — métricas POR MEZCLA, seed-averaged. Agrupación vectorizada.

    AUC/AP son threshold-free (primarias, Codex r11); F1 es a logit>=0 (prob>=0.5), secundaria.
    Reemplaza el bootstrap F1 pooled (O(n_boot × pares), inviable en OOD) por bootstrap sobre
    escalares por mezcla. Mezclas de una sola clase (poly1) se saltean para AUC/AP/F1.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    acc = defaultdict(lambda: {"f1": [], "auc": [], "ap": [], "poly": None, "reg": None})
    for s in seeds:
        f = results_dir / "test_pairs" / f"{run}__{model}__seed{s}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        mix = d["mix_id"]
        if len(mix) == 0:
            continue
        lg = d["logit"].astype(np.float64); tg = d["target"].astype(np.int64)
        poly = d["polyphony"]; reg = d["regime"]
        order = np.argsort(mix, kind="stable")             # agrupa por mezcla sin loop por par
        mix_s, lg_s, tg_s = mix[order], lg[order], tg[order]
        poly_s, reg_s = poly[order], reg[order]
        uniq, start = np.unique(mix_s, return_index=True)
        for k, sl in enumerate(np.split(np.arange(len(mix_s)), start[1:])):
            m = int(uniq[k]); y = tg_s[sl]; x = lg_s[sl]
            acc[m]["poly"] = int(poly_s[sl[0]]); acc[m]["reg"] = str(reg_s[sl[0]])
            if y.min() == y.max():
                continue                                   # una sola clase -> métricas indefinidas
            acc[m]["f1"].append(f1_score(y, (x >= 0.0).astype(np.int64), zero_division=0))
            acc[m]["auc"].append(roc_auc_score(y, x))
            acc[m]["ap"].append(average_precision_score(y, x))
    return {m: {"f1": float(np.mean(v["f1"])) if v["f1"] else np.nan,
                "auc": float(np.mean(v["auc"])) if v["auc"] else np.nan,
                "ap": float(np.mean(v["ap"])) if v["ap"] else np.nan,
                "poly": v["poly"], "reg": v["reg"]} for m, v in acc.items()}


def _permix_worker(task):
    """Worker picklable para ProcessPoolExecutor: (rd, run, model, seeds) -> ((run,model), dict)."""
    rd, run, model, seeds = task
    return (run, model), load_permix_metrics(rd, run, model, seeds)


def load_permix_all(rd: Path, runs, models, seeds, workers: int = 14) -> Dict:
    """Computa las métricas por-mezcla de TODOS los (run, model) en paralelo (Codex: usar 14 cores)."""
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(rd, run, m, seeds) for run in runs for m in models]
    n_workers = max(1, min(workers, len(tasks)))
    out = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for key, val in ex.map(_permix_worker, tasks):
            out[key] = val
    return out


def bootstrap_permix_diff(a: Dict, b: Dict, metric, poly, regime, n_boot=2000, seed=42):
    """Bootstrap PAREADO sobre mezclas del Δ(a−b) en una métrica por-mezcla, en una celda."""
    common = sorted(set(a) & set(b))
    mids = [m for m in common if a[m]["poly"] == poly and a[m]["reg"] == regime
            and not np.isnan(a[m][metric]) and not np.isnan(b[m][metric])]
    if len(mids) < 2:
        return None
    av = np.array([a[m][metric] for m in mids]); bv = np.array([b[m][metric] for m in mids])
    M = len(mids); rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for t in range(n_boot):
        idx = rng.randint(0, M, M)                          # mismo idx para a y b -> pareado
        diffs[t] = av[idx].mean() - bv[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_a": float(av.mean()), "mean_b": float(bv.mean()),
            "mean_diff": float(av.mean() - bv.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi),
            "frac_positive": float((diffs > 0).mean()), "n_mixtures": M}


def fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.3f}"


def validate_completeness(results, rd: Path, seeds, allow_partial: bool) -> None:
    """Para el cierre final: assertar 3 runs × 6 models × len(seeds) records + .npz de cada uno.

    Sin --allow-partial, aborta si el reporte sería parcial (Codex r9 #2). El smoke usa
    --allow-partial para reportar sobre un subconjunto.
    """
    problems = []
    have = {(r["run"], r["model"], int(r["seed"])) for r in results}
    expected = {(run, m, s) for run in RUNS for m in MODELS for s in seeds}
    missing = sorted(expected - have)
    if missing:
        problems.append(f"{len(missing)} records faltan en results.json (ej: {missing[:4]})")
    n_exp = len(RUNS) * len(MODELS) * len(seeds)
    if len(results) != n_exp:
        problems.append(f"{len(results)} records (esperados {n_exp} = 3 runs × 6 models × {len(seeds)} seeds)")
    n_files = 0
    for (run, m, s) in expected:
        for sub in ("test_pairs", "test_ari"):
            if not (rd / sub / f"{run}__{m}__seed{s}.npz").exists():
                n_files += 1
    if n_files:
        problems.append(f"{n_files} archivos .npz (test_pairs/test_ari) faltan")
    if problems:
        msg = "Reporte INCOMPLETO:\n  - " + "\n  - ".join(problems)
        if allow_partial:
            logger.warning("%s\n(--allow-partial: continúo con reporte parcial)", msg)
        else:
            raise SystemExit(msg + "\n\nUsá --allow-partial para reporte parcial (smoke).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--allow-partial", action="store_true",
                   help="permite reporte sobre subconjunto (smoke); sin esto exige 54 records completos")
    args = p.parse_args()

    rd = Path(args.results_dir)
    results = json.loads((rd / "results.json").read_text())
    validate_completeness(results, rd, args.seeds, args.allow_partial)
    runs = sorted({r["run"] for r in results}, key=lambda x: RUNS.index(x) if x in RUNS else 99)
    models = [m for m in MODELS if any(r["model"] == m for r in results)]
    logger.info("Runs: %s | Models: %s", runs, models)

    # contrastes por-mezcla (threshold-free), métrica primaria AUC/AP — Codex r11
    def _parse_cell(k):                                   # "poly{N}_{regime}" -> (N, regime)
        pn, reg = k[4:].split("_", 1)
        return int(pn), reg

    # métricas por-mezcla de todos los (run, model) en paralelo (14 cores) — antes del loop
    permix_all = load_permix_all(rd, runs, models, args.seeds, workers=14)

    contrasts_out = []
    lines = ["# Reporte Fase 0 — Atención Armónica (Harmonic Pairformer)\n"]
    lines.append("> Agrupamiento armónico sobre mezclas sintéticas. Parciales exactos, acordes "
                 "estáticos, ground truth exacto. **Métrica primaria: AUC/AP threshold-free** (mide la "
                 "representación, sin τ). ARI@τ_val es operating-point secundario (el τ de val NO "
                 "transfiere a OOD-poly para B). Multi-seed: métrica por-mezcla promediada sobre 3 "
                 "seeds → bootstrap pareado sobre mezclas. poly1 es degenerada: sanity, no evidencia.\n")
    lines.append("\n**Contraste central**: B vs B-local (param-matched) aísla la triangle update vs "
                 "mezcla local. B vs B-shuffle es el control de estructura. B-minus vs A-rich mide el "
                 "aporte del pair-state. Lectura: pair-state = salto grande; triángulo neutro IID, "
                 "mejor OOD-poly threshold-free.\n")

    for run in runs:
        lines.append(f"\n## Run {run}\n")
        cell_keys = set()
        for r in results:
            if r["run"] == run:
                cell_keys |= set(r["test"].get("by_cell", {}).keys())
        cells = sorted({_parse_cell(k) for k in cell_keys})
        # métricas por-mezcla (seed-averaged) por modelo (ya computadas en paralelo arriba)
        permix = {m: permix_all[(run, m)] for m in models}

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

        # contrastes threshold-free POR-MEZCLA (seed-avg) — PRIMARIOS (Codex r11): AUC/AP no dependen
        # de τ → miden si la REPRESENTACIÓN generaliza. F1@0.5 secundaria (operating-point fijo).
        lines.append("### Contrastes threshold-free Δ(AUC, AP) — bootstrap pareado por mezcla (PRIMARIO)\n")
        lines.append("> AUC/AP miden la representación sin τ. Bootstrap pareado sobre mezclas; métrica "
                     "por-mezcla promediada sobre seeds. F1@0.5 acompaña como operating-point fijo. "
                     "ARI@τ_val (abajo) es operating-point con τ transferido — para B NO transfiere OOD.\n")
        lines.append("| Contraste | Tipo | Celda | ΔAUC | CI95 | ΔAP | CI95 | ΔF1@.5 | P(ΔAUC>0) | n |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for (a, b, tipo) in CONTRASTS:
            if a not in models or b not in models:
                continue
            for (poly, regime) in cells:
                if poly < 2:
                    continue
                rA = bootstrap_permix_diff(permix[a], permix[b], "auc", poly, regime, n_boot=args.n_boot)
                if rA is None:
                    continue
                rP = bootstrap_permix_diff(permix[a], permix[b], "ap", poly, regime, n_boot=args.n_boot)
                rF = bootstrap_permix_diff(permix[a], permix[b], "f1", poly, regime, n_boot=args.n_boot)
                ciA = f"[{rA['ci95_lo']:+.3f},{rA['ci95_hi']:+.3f}]"
                ciP = f"[{rP['ci95_lo']:+.3f},{rP['ci95_hi']:+.3f}]" if rP else "—"
                lines.append(f"| {a} vs {b} | {tipo} | {_cell_key(poly,regime)} | "
                             f"{rA['mean_diff']:+.3f} | {ciA} | "
                             f"{(rP['mean_diff'] if rP else float('nan')):+.3f} | {ciP} | "
                             f"{(rF['mean_diff'] if rF else float('nan')):+.3f} | "
                             f"{rA['frac_positive']:.2f} | {rA['n_mixtures']} |")
                contrasts_out.append({
                    "run": run, "contrast": f"{a} vs {b}", "tipo": tipo,
                    "cell": _cell_key(poly, regime), "metric": "threshold_free",
                    "dAUC": rA["mean_diff"], "AUC_ci95": [rA["ci95_lo"], rA["ci95_hi"]],
                    "AUC_P": rA["frac_positive"],
                    "dAP": (rP["mean_diff"] if rP else None),
                    "AP_ci95": ([rP["ci95_lo"], rP["ci95_hi"]] if rP else None),
                    "dF1_at_0.5": (rF["mean_diff"] if rF else None),
                    "n_mixtures": rA["n_mixtures"],
                })
        lines.append("")

        # contrastes Δ ARI — PRIMARIO para el claim de transitividad (Codex r9 #1).
        # ARI mide la partición inducida, no solo ranking per-par; bootstrap pareado sobre mezclas
        # (ARI por mezcla promediado sobre 3 seeds).
        ari_pm = {m: load_ari_per_mix(rd, run, m, args.seeds) for m in models}
        lines.append("### Contrastes Δ ARI — bootstrap pareado sobre mezclas (PRIMARIO transitividad)\n")
        lines.append("> ARI es la métrica primaria del claim: mide el agrupamiento inducido. "
                     "ΔARI por mezcla (seed-averaged), CI95 bootstrap pareado. Criterio (congelado, Codex r9): "
                     "B−B-local con CI95 que excluye 0 en poly3_hard; material si ΔARI≥+0.05; "
                     "B-shuffle NO debe igualar a B dentro del CI.\n")
        lines.append("| Contraste | Tipo | Celda | Δ ARI (ens) | CI95 | P(Δ>0) | ΔARI per-seed | n_mix |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for (a, b, tipo) in CONTRASTS:
            if a not in models or b not in models:
                continue
            for (poly, regime) in cells:
                if poly < 2:
                    continue
                res = bootstrap_ari_diff(ari_pm[a], ari_pm[b], poly, regime, n_boot=args.n_boot)
                if res is None:
                    continue
                ci = f"[{res['ci95_lo']:+.3f}, {res['ci95_hi']:+.3f}]"
                psd = per_seed_contrast_ari(results, run, a, b, poly, regime)
                lines.append(f"| {a} vs {b} | {tipo} | {_cell_key(poly,regime)} | "
                             f"{res['mean_diff']:+.3f} | {ci} | {res['frac_positive']:.2f} | {psd} | {res['n_mixtures']} |")
                contrasts_out.append({
                    "run": run, "contrast": f"{a} vs {b}", "tipo": tipo, "metric": "ari",
                    "cell": _cell_key(poly, regime), **res,
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

    # cierre direccional — criterios congelados (Codex r9), ARI primaria
    lines.append("\n## Lectura direccional (GO/NO-GO lo decide el usuario)\n")
    lines.append("> Métrica primaria del claim de transitividad: **ΔARI** (partición inducida). "
                 "AUC/AP son secundarias (la AUC de B satura ~0.99 → poca dinámica). Orden de "
                 "lectura (Codex r9): **B vs B-local en ARI primero, B-shuffle segundo, B vs A-rich tercero**.\n")
    lines.append("- **B−B-local en ARI con CI95 que excluye 0 en poly3_hard** (material si ΔARI≥+0.05; "
                 "fuerte si se sostiene en OOD difícil) → la transitividad hace trabajo real → GO.")
    lines.append("- **B≈B-local en ARI** → el pair-state genérico ya captura la estructura; la "
                 "transitividad específica NO aporta (aunque B>A-rich).")
    lines.append("- **B−B-local en ARI con CI95 que excluye 0 EN NEGATIVO** (B-local > B) → la "
                 "triangle update implementada no solo no aporta sino que **perjudica** frente a una "
                 "mezcla local param-matched. NO es 'toda transitividad es inútil': es que ESTA receta "
                 "de triángulo, con este presupuesto, pierde contra mezcla local (Codex r10 #2/#7).")
    lines.append("- **B-shuffle iguala a B dentro del CI (ARI)** → se cae la atribución estructural "
                 "aunque B>A-rich (confound de capacidad). Si NO iguala, refuerza.")
    lines.append("- **B≈A-rich** → un baseline token-only fuerte ya capturó la estructura global; "
                 "NO que el dataset sea feature-trivial (el gate lo descartó). NO construir la cosa grande.")
    lines.append("- B vs A-rich es compatible con la hipótesis pero NO la identifica: la atribución "
                 "al triángulo sale de B vs B-local (param-matched), no de B vs A-rich.")
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
