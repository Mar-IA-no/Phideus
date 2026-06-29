#!/usr/bin/env python3
"""Fase 0.6 — clusterers deployables (Atención Armónica). Post-hoc, CPU, sobre matrices guardadas.

Fase 0.5 mostró que el colapso de B en OOD-poly NO es calibración de τ (gap_dist≈0) sino la fragilidad
transitiva de connected-components: B (mejor AUC) hace pocos edges cross-source de alta confianza que
encadenan clusters. Bajo agglo+k-verdadero (privilegiado) B es el mejor OOD-poly. Acá probamos reglas
de clustering DEPLOYABLES (sin k verdadero) que no tengan esa fragilidad, y medimos cuánto recuperan.

Sistema primario: ensemble de logits crudos (3 seeds), calibrador none (prob=sigmoid). Reusa matrices
val_mats/ + test_mats/. Reglas core (sklearn-only):
  cc_bridge_prune  : CC tras podar puentes (overlap de vecinos < θ_prune). τ_val FIJO = none|ari baseline.
  spectral_eigengap: SpectralClustering, k por eigengap (ε_gap). affinity=prob*pair_valid, aislados=singletons.
  agglo_estimated_k: AgglomerativeClustering, k=eigengap (ε_gap). dist=1-prob*pair_valid, diag 0.
  (modularity_louvain: OPCIONAL, fuera del core; skip si no hay networkx.)
Referencias recomputadas desde matrices: cc@τ_val (piso deployable), privileged_reference_k_known
(agglo+true-k) y oracle_tau_per_mixture (privilegiadas, NO techos duros).

Knobs: grilla cerrada, seleccionado SOLO en val por val-ARI, tie-break nanargmax (primer máximo),
por (run,model), congelado a test. k-estimadores reportan dist de k (alerta de colapso, no abort).
Salidas: REPORTE_0.6.md + deployable.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.special import expit
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = ("A-naive", "A-rich", "B", "B-minus", "B-local", "B-shuffle")
RUNS = ("ID", "OOD-poly", "OOD-regime")
TAU_GRID = np.round(np.arange(0.10, 0.91, 0.05), 3)
THETA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
EPS_GAP_GRID = [0.0, 0.02, 0.05, 0.10, 0.20]
RES_GRID = [0.5, 1.0, 1.5]
KMAX_CAP = 6
CONTRASTS = [("B", "B-local"), ("B", "B-shuffle"), ("B", "B-minus"), ("B-minus", "A-rich")]
CORE_RULES = ("cc_bridge_prune", "spectral_eigengap", "agglo_estimated_k")

try:
    import networkx as _nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False


# --------------------------------------------------------------------------- helpers (de 2_calibration_audit)
def fast_ari(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    cont = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(cont, (ai, bi), 1)
    cc = (cont * (cont - 1) // 2).sum()
    asum = cont.sum(1); bsum = cont.sum(0)
    ca = (asum * (asum - 1) // 2).sum(); cb = (bsum * (bsum - 1) // 2).sum()
    tot = n * (n - 1) // 2
    exp = ca * cb / tot if tot else 0.0
    mx = (ca + cb) / 2.0
    return 1.0 if mx == exp else float((cc - exp) / (mx - exp))


def load_ensemble(results_dir: Path, subdir, run, model, seeds):
    per_seed = []
    for s in seeds:
        f = results_dir / subdir / f"{run}__{model}__seed{s}.npz"
        if not f.exists():
            return None
        with np.load(f, allow_pickle=True) as d:
            mids = d["mixture_id"]; polys = d["polyphony"]; regs = d["regime"]; ns = d["n_peaks"]
            logits = d["logit_mat"]; pair_valids = d["pair_valid"]; targets = d["target_mat"]
            by = {int(mids[i]): {
                "logit_mat": logits[i].astype(np.float64), "pair_valid": pair_valids[i],
                "target_mat": targets[i], "n": int(ns[i]), "poly": int(polys[i]), "reg": str(regs[i]),
            } for i in range(len(mids))}
        per_seed.append(by)
    base = per_seed[0]
    out = {}
    for mid, e0 in base.items():
        if not all(mid in ps for ps in per_seed):
            continue
        for ps in per_seed[1:]:
            e = ps[mid]
            assert e["n"] == e0["n"] and e["poly"] == e0["poly"] and e["reg"] == e0["reg"]
            assert np.array_equal(e["target_mat"], e0["target_mat"])
            assert np.array_equal(e["pair_valid"], e0["pair_valid"])
        logits = np.stack([ps[mid]["logit_mat"] for ps in per_seed])
        out[mid] = {"logit_mat": logits.mean(axis=0), "pair_valid": e0["pair_valid"],
                    "target_mat": e0["target_mat"], "n": e0["n"], "poly": e0["poly"],
                    "reg": e0["reg"], "mid": mid}
    return out


def true_source(target_mat, n):
    adj = (target_mat[:n, :n] >= 0.5).copy()
    np.fill_diagonal(adj, True)
    _, lab = connected_components(adj, directed=False)
    return lab


def cell_mean(ari, meta, poly, reg):
    vals = [ari[mid] for mid, mm in meta.items()
            if mm["poly"] == poly and mm["reg"] == reg and mid in ari and not np.isnan(ari[mid])]
    return float(np.mean(vals)) if vals else float("nan")


def boot_ari(ari_a, meta, ari_b, poly, reg, n_boot=2000, seed=42):
    mids = [mid for mid, mm in meta.items()
            if mm["poly"] == poly and mm["reg"] == reg and mid in ari_a and mid in ari_b
            and not np.isnan(ari_a[mid]) and not np.isnan(ari_b[mid])]
    if len(mids) < 2:
        return None
    a = np.array([ari_a[mid] for mid in mids]); b = np.array([ari_b[mid] for mid in mids])
    rng = np.random.RandomState(seed); M = len(mids); diffs = np.empty(n_boot)
    for t in range(n_boot):
        idx = rng.randint(0, M, M); diffs[t] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(a.mean() - b.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi),
            "frac_positive": float((diffs > 0).mean()), "n_mixtures": M}


# --------------------------------------------------------------------------- clusterers
def cc_at_tau(prob, n, tau, pair_valid):
    adj = (prob[:n, :n] >= tau) & pair_valid[:n, :n]
    np.fill_diagonal(adj, True); adj = adj | adj.T
    _, lab = connected_components(adj, directed=False)
    return lab


def cc_bridge_prune(prob, n, tau, pair_valid, theta):
    """CC tras podar puentes: edge i–j sobrevive si overlap de vecinos >= theta."""
    E = (prob[:n, :n] >= tau) & pair_valid[:n, :n]
    np.fill_diagonal(E, False)
    E = E | E.T
    deg = E.sum(1)
    common = (E.astype(np.int64) @ E.astype(np.int64))           # |N(i)∩N(j)|
    min_deg = np.minimum(deg[:, None], deg[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = np.where(min_deg > 0, common / min_deg, 0.0)
    keep = E & (overlap >= theta)
    np.fill_diagonal(keep, True); keep = keep | keep.T
    _, lab = connected_components(keep, directed=False)
    return lab


def _estimate_k(prob, n, pair_valid, eps_gap):
    """k por eigengap del Laplaciano normalizado de A=prob*pair_valid; aislados = singletons.
    Devuelve (k_total, k_sub, non_iso_idx, iso_idx, A_sub)."""
    A = (prob[:n, :n] * pair_valid[:n, :n]).astype(np.float64)
    A = 0.5 * (A + A.T); np.fill_diagonal(A, 0.0)
    deg = A.sum(1)
    iso = np.where(deg <= 1e-12)[0]
    non = np.where(deg > 1e-12)[0]
    ns = len(non)
    if ns <= 1:
        return ns + len(iso), max(ns, 0), non, iso, None
    A_sub = A[np.ix_(non, non)]
    d = A_sub.sum(1)
    Dinv = 1.0 / np.sqrt(d + 1e-12)
    L = np.eye(ns) - (Dinv[:, None] * A_sub * Dinv[None, :])
    L = 0.5 * (L + L.T)
    ev = np.sort(np.linalg.eigvalsh(L))
    kmax = min(ns - 1, KMAX_CAP)
    gaps = ev[1:kmax + 1] - ev[:kmax]
    if len(gaps) == 0 or float(gaps.max()) < eps_gap:
        k_sub = 1
    else:
        k_sub = int(np.argmax(gaps)) + 1
    return k_sub + len(iso), k_sub, non, iso, A_sub


def cluster_spectral(prob, n, pair_valid, eps_gap):
    if n < 2:
        return np.zeros(max(n, 0), dtype=int), max(n, 0)
    k_total, k_sub, non, iso, A_sub = _estimate_k(prob, n, pair_valid, eps_gap)
    lab = np.full(n, -1, dtype=int); nxt = 0
    ns = len(non)
    if ns >= 2:
        if k_sub <= 1:
            sub = np.zeros(ns, dtype=int)
        elif k_sub >= ns:
            sub = np.arange(ns)
        else:
            try:
                sub = SpectralClustering(n_clusters=k_sub, affinity="precomputed",
                                         assign_labels="kmeans", random_state=0).fit_predict(A_sub)
            except Exception:
                sub = np.zeros(ns, dtype=int)
        lab[non] = sub + nxt; nxt += int(sub.max()) + 1
    elif ns == 1:
        lab[non[0]] = nxt; nxt += 1
    for j in iso:
        lab[j] = nxt; nxt += 1
    return lab, len(np.unique(lab))


def cluster_agglo_estk(prob, n, pair_valid, eps_gap):
    if n < 2:
        return np.zeros(max(n, 0), dtype=int), max(n, 0)
    k_total, _, _, _, _ = _estimate_k(prob, n, pair_valid, eps_gap)
    k = int(np.clip(k_total, 1, n))
    if k <= 1:
        return np.zeros(n, dtype=int), 1
    if k >= n:
        return np.arange(n), n
    dist = 1.0 - (prob[:n, :n] * pair_valid[:n, :n])
    dist = 0.5 * (dist + dist.T); np.fill_diagonal(dist, 0.0)        # precomputed exige diagonal 0
    lab = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                  linkage="average").fit_predict(dist)
    return lab, k


def cluster_louvain(prob, n, pair_valid, resolution):
    A = (prob[:n, :n] * pair_valid[:n, :n]).astype(np.float64)
    A = 0.5 * (A + A.T); np.fill_diagonal(A, 0.0)
    G = _nx.from_numpy_array(A)
    comms = _nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=0)
    lab = np.zeros(n, dtype=int)
    for ci, c in enumerate(comms):
        for v in c:
            lab[v] = ci
    return lab, len(comms)


def agglo_true_k(prob, n, pair_valid, true):
    """Referencia PRIVILEGIADA (k verdadero), no techo. Mismo agglo, k = #fuentes verdaderas."""
    k = len(set(true.tolist()))
    if k < 2:
        return np.zeros(n, dtype=int)
    if k >= n:
        return np.arange(n)
    dist = 1.0 - (prob[:n, :n] * pair_valid[:n, :n])
    dist = 0.5 * (dist + dist.T); np.fill_diagonal(dist, 0.0)
    return AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(dist)


# --------------------------------------------------------------------------- selección por val + eval test
def _ari(prob, n, pair_valid, true, labels):
    return fast_ari(true, labels)


def select_and_eval(val_pool, test_mixes, clust_fn, grid):
    """clust_fn(m, knob) -> (labels, k or None). Selecciona knob por val-ARI (nanargmax = primer máx),
    devuelve (best_knob, {mid: ari_test}, {mid: k_test or None}, val_ari_best)."""
    val_scores = []
    for knob in grid:
        aris = []
        for m in val_pool:
            if m["n"] < 2:
                continue
            lab, _ = clust_fn(m, knob)
            aris.append(fast_ari(m["_true"], lab))
        val_scores.append(float(np.nanmean(aris)) if aris else -2.0)
    bi = int(np.nanargmax(val_scores))                              # tie-break: primer máximo
    best_knob = grid[bi]
    test_ari, test_k = {}, {}
    for m in test_mixes:
        if m["n"] < 2:
            test_ari[m["mid"]] = float("nan"); test_k[m["mid"]] = None
            continue
        lab, k = clust_fn(m, best_knob)
        test_ari[m["mid"]] = fast_ari(m["_true"], lab); test_k[m["mid"]] = k
    return best_knob, test_ari, test_k, val_scores[bi]


def process_run_model(task):
    t0 = time.time()
    rd, run, model, seeds = task
    logger.info("[load] %s/%s val_mats...", run, model)
    val = load_ensemble(rd, "val_mats", run, model, seeds)
    logger.info("[load] %s/%s test_mats... (val %.0fs)", run, model, time.time() - t0)
    test = load_ensemble(rd, "test_mats", run, model, seeds)
    if val is None or test is None:
        logger.info("[skip] %s/%s", run, model)
        return (run, model), None
    logger.info("[start] %s/%s val=%d test=%d", run, model, len(val), len(test))
    for ens in (val, test):
        for m in ens.values():
            n = m["n"]
            m["_prob"] = expit(m["logit_mat"]) if n >= 1 else m["logit_mat"]
            m["_true"] = true_source(m["target_mat"], n) if n >= 2 else None
    val_pool = [m for m in val.values() if m["poly"] >= 2] or list(val.values())
    test_mixes = list(test.values())

    # --- referencias recomputadas desde matrices ---
    # cc@τ_val (piso deployable; da el τ_val que usa bridge_prune)
    tval_scores = []
    for tau in TAU_GRID:
        aris = [fast_ari(m["_true"], cc_at_tau(m["_prob"], m["n"], tau, m["pair_valid"]))
                for m in val_pool if m["n"] >= 2]
        tval_scores.append(float(np.nanmean(aris)) if aris else -2.0)
    tau_val = float(TAU_GRID[int(np.nanargmax(tval_scores))])
    cc_tauval = {m["mid"]: (fast_ari(m["_true"], cc_at_tau(m["_prob"], m["n"], tau_val, m["pair_valid"]))
                            if m["n"] >= 2 else np.nan) for m in test_mixes}
    # privileged_reference_k_known (agglo + true k) y oracle_tau_per_mixture
    ref_kknown, oracle_permix = {}, {}
    for m in test_mixes:
        if m["n"] < 2:
            ref_kknown[m["mid"]] = np.nan; oracle_permix[m["mid"]] = np.nan; continue
        ref_kknown[m["mid"]] = fast_ari(m["_true"], agglo_true_k(m["_prob"], m["n"], m["pair_valid"], m["_true"]))
        oracle_permix[m["mid"]] = max(fast_ari(m["_true"], cc_at_tau(m["_prob"], m["n"], t, m["pair_valid"]))
                                      for t in TAU_GRID)

    # --- reglas deployables (core) ---
    rules, kdist = {}, {}
    fns = {
        "cc_bridge_prune": (lambda m, th: (cc_bridge_prune(m["_prob"], m["n"], tau_val, m["pair_valid"], th), None), THETA_GRID),
        "spectral_eigengap": (lambda m, e: cluster_spectral(m["_prob"], m["n"], m["pair_valid"], e), EPS_GAP_GRID),
        "agglo_estimated_k": (lambda m, e: cluster_agglo_estk(m["_prob"], m["n"], m["pair_valid"], e), EPS_GAP_GRID),
    }
    if HAVE_NX:
        fns["modularity_louvain"] = (lambda m, r: cluster_louvain(m["_prob"], m["n"], m["pair_valid"], r), RES_GRID)
    for name, (fn, grid) in fns.items():
        knob, test_ari, test_k, val_ari = select_and_eval(val_pool, test_mixes, fn, grid)
        rules[name] = {"knob": knob, "val_ari": val_ari, "test_ari": test_ari}
        ks = [v for v in test_k.values() if v is not None]
        if ks:
            kdist[name] = {int(kk): int(np.sum(np.array(ks) == kk)) for kk in sorted(set(ks))}

    meta = {m["mid"]: {"poly": m["poly"], "reg": m["reg"]} for m in test_mixes}
    logger.info("[done] %s/%s en %.0fs (τ_val=%.2f)", run, model, time.time() - t0, tau_val)
    return (run, model), {"tau_val": tau_val, "cc_tauval": cc_tauval, "ref_kknown": ref_kknown,
                          "oracle_permix": oracle_permix, "rules": rules, "kdist": kdist, "meta": meta,
                          "cells": sorted({(m["poly"], m["reg"]) for m in test_mixes})}


# --------------------------------------------------------------------------- main / reporte
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--runs", nargs="+", default=list(RUNS))
    args = ap.parse_args()
    rd = Path(args.results_dir)
    tasks = [(rd, run, m, args.seeds) for run in args.runs for m in args.models]
    results = {}
    if args.workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)),
                                 mp_context=mp.get_context("spawn")) as ex:
            for key, val in ex.map(process_run_model, tasks):
                results[key] = val
    else:
        for tk in tasks:
            key, val = process_run_model(tk)
            results[key] = val
    results = {k: v for k, v in results.items() if v is not None}
    pruns = [r for r in args.runs if any((r, m) in results for m in args.models)]
    pmodels = [m for m in args.models if any((r, m) in results for r in args.runs)]
    core = list(CORE_RULES)                                  # louvain NO entra al veredicto core (Codex r-plan-06 #5)

    lines = ["# REPORTE_0.6 — Atención Armónica: clusterers deployables\n"]
    lines.append("> Post-Fase-0.5. ¿Una regla de clustering deployable (sin k verdadero) extrae la "
                 "representación de B sin la fragilidad de connected-components? Ensemble none, matrices "
                 "guardadas. Knobs seleccionados SOLO en val (nanargmax). Privilegiadas = referencias, NO techos.\n")
    if not HAVE_NX:
        lines.append("> (modularity_louvain omitido: networkx no disponible.)\n")
    export = {"per_run_model": {}, "contrasts": []}

    for run in pruns:
        lines.append(f"\n## Run {run}\n")
        cells = sorted({tuple(c) for m in pmodels if (run, m) in results
                        for c in map(tuple, results[(run, m)]["cells"]) if c[0] >= 2})
        hdr = " | ".join(f"poly{p}_{r}" for (p, r) in cells)
        sep = "|" + "---|" * (len(cells) + 1)

        lines.append("### ARI por celda × regla (deployables + piso cc@τ_val + referencias privilegiadas)\n")
        lines.append(f"| Modelo | regla | {hdr} |")
        lines.append(sep)
        verdict = {}
        for m in pmodels:
            if (run, m) not in results:
                continue
            R = results[(run, m)]
            rowmap = {"cc@τ_val (piso)": R["cc_tauval"]}
            for nm in core:
                rowmap[nm] = R["rules"][nm]["test_ari"]
            if HAVE_NX and "modularity_louvain" in R["rules"]:   # exploratorio, fuera del veredicto
                rowmap["modularity_louvain (exploratorio)"] = R["rules"]["modularity_louvain"]["test_ari"]
            rowmap["ref_k_known (priv)"] = R["ref_kknown"]
            rowmap["oracle_τ_permix (priv)"] = R["oracle_permix"]
            for nm, ari in rowmap.items():
                lines.append("| " + " | ".join([m, nm] + [f"{cell_mean(ari, R['meta'], p, r):.3f}"
                                                          for (p, r) in cells]) + " |")
            # mejor deployable por (run,model) por val-ARI
            best = max(core, key=lambda nm: (R["rules"][nm]["val_ari"]
                       if not np.isnan(R["rules"][nm]["val_ari"]) else -2.0))
            verdict[m] = best
            export["per_run_model"][f"{run}|{m}"] = {
                "tau_val": R["tau_val"], "best_deployable": best,
                "rules": {nm: {"knob": R["rules"][nm]["knob"], "val_ari": R["rules"][nm]["val_ari"]} for nm in core},
                "kdist": R["kdist"],
                # cell_ari por regla/celda (Codex r-06-audit #4: no depender del markdown)
                "cell_ari": {nm: {f"poly{p}_{r}": cell_mean(ari, R["meta"], p, r) for (p, r) in cells}
                             for nm, ari in rowmap.items()},
            }
        lines.append("")

        # distribución de k (alerta de colapso)
        lines.append("### Distribución de k estimado (k_total) — alerta si colapsa a 1 o a n\n")
        for m in pmodels:
            if (run, m) not in results:
                continue
            for nm in ("spectral_eigengap", "agglo_estimated_k"):
                kd = results[(run, m)]["kdist"].get(nm, {})
                if kd:
                    tot = sum(kd.values()); top = max(kd, key=kd.get)
                    alert = " ⚠️COLAPSO" if kd.get(top, 0) / tot > 0.95 and (top == 1) else ""
                    lines.append(f"- {m} {nm}: {dict(sorted(kd.items()))}{alert}")
        lines.append("")

        # contrastes: regla común para CADA regla core (Codex r-06-audit #2) + best deployable por modelo
        readings = [(f"común {nm}", (lambda _nm: (lambda m: _nm))(nm)) for nm in core]
        readings.append(("best deployable por modelo", lambda m: verdict[m]))
        for label, getter in readings:
            lines.append(f"### Contrastes ΔARI bajo {label} — bootstrap pareado\n")
            lines.append("| Contraste | Celda | ΔARI | CI95 | P(Δ>0) | n |")
            lines.append("|---|---|---|---|---|---|")
            for (a, b) in CONTRASTS:
                if (run, a) not in results or (run, b) not in results:
                    continue
                Ra, Rb = results[(run, a)], results[(run, b)]
                aa, ab = Ra["rules"][getter(a)]["test_ari"], Rb["rules"][getter(b)]["test_ari"]
                for (p, r) in cells:
                    res = boot_ari(aa, Ra["meta"], ab, p, r, n_boot=args.n_boot)
                    if res is None:
                        continue
                    excl = "" if (res["ci95_lo"] <= 0 <= res["ci95_hi"]) else " *"
                    lines.append(f"| {a} vs {b} | poly{p}_{r} | {res['mean_diff']:+.3f}{excl} | "
                                 f"[{res['ci95_lo']:+.3f},{res['ci95_hi']:+.3f}] | {res['frac_positive']:.2f} | {res['n_mixtures']} |")
                    export["contrasts"].append({"run": run, "contrast": f"{a} vs {b}", "rule": label,
                                                "cell": f"poly{p}_{r}", **res})
            lines.append("")

    lines.append("\n## Lectura (GO/NO-GO lo decide el usuario)\n")
    lines.append("Formulación acotada (sign-off Codex S66): el resultado es positivo pero condicionado, "
                 "y se sostiene en cuatro puntos.")
    lines.append("- **(1) Positivo**: en OOD-poly B SUPERA a B-local bajo **clusterers globales deployables** que "
                 "estiman estructura de partición — `spectral_eigengap` y `agglo_estimated_k` **comunes** (no solo "
                 "best-per-model), CI excluye 0. El resultado ya no depende de elegir una regla distinta por modelo.")
    lines.append("- **(2) Negativo/neutral**: B-local sigue ganando en **IID** y en **OOD-regime** (B vs B-local "
                 "negativo bajo todas las reglas, CI excluye 0). La ventaja de B es específica de generalizar a "
                 "polifonía nueva.")
    lines.append("- **(3) Diagnóstico — `cc_bridge_prune`**: la poda de puentes RECUPERA a B de forma fuerte "
                 "(OOD-poly poly3_hard 0.134 → 0.357) → **confirma que los puentes transitivos eran reales** (lo que "
                 "Fase 0.5 diagnosticó); pero aun así B sigue PERDIENDO vs B-local bajo esta regla (−0.035\\*) → "
                 "connected-components, incluso con poda, **no es el lector adecuado** para esta geometría. El cuello "
                 "era el algoritmo de partición, no el τ ni la representación.")
    lines.append("- **(4) Caveat — recupera, NO resuelve**: el estimador de k SUBESTIMA en OOD-poly (para B, k=2 en la "
                 "mayoría de mezclas poly3 → fusiona fuentes). `0.465` (mejor deployable) vs `0.607` (`ref_k_known`, k "
                 "verdadero) significa *recuperamos parte de la representación*, NO *resolvimos clustering deployable*.")
    lines.append("- `ref_k_known` y `oracle_τ_permix` son PRIVILEGIADAS (referencias, no techos). El gap a `ref_k_known` "
                 "mide cuánto cuesta no conocer k; sigue siendo grande → un k-predictor (Stage B) es el siguiente paso.")
    (rd / "REPORTE_0.6.md").write_text("\n".join(lines), encoding="utf-8")
    (rd / "deployable.json").write_text(json.dumps(export, indent=2, default=float))
    logger.info("Escrito REPORTE_0.6.md + deployable.json (%d run-model)", len(results))


if __name__ == "__main__":
    main()
