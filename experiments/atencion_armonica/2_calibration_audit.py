#!/usr/bin/env python3
"""Fase 0.5 — auditoría de calibración / decisión de clustering (Atención Armónica).

Separa REPRESENTACIÓN (AUC/AP, REPORTE_0) de DECISIÓN DE CLUSTERING / operating-point. Reusa los
artefactos de eval (val_mats/ + test_mats/) — NO re-forward.

Sistema primario (congelado): ENSEMBLE de logits CRUDOS por par (alineados por mixture_id, promediados
sobre 3 seeds) → UN calibrador entrenado en val. Calibradores: none (raw sigmoid), platt, isotonic
(fit pair-pooled en val poly≥2; cada uno su PROPIO τ). Clusterer deployable: connected-components a τ.

Performance (Codex r-plan-perf): para cada (run,model,calibrador,split) se computa UNA tabla
ari_table[mix, τ] (prob_mat + true_source una vez por mezcla) y de ahí salen select_tau, oráculos,
test_ari y bootstrap por indexado — sin recomputar clustering miles de veces. ARI = fast_ari inline
(idéntico a sklearn, sin su overhead). Grid τ = el del training (0.10..0.90).

Veredicto (doble lectura, val-seleccionadas por (run,model) sobre val poly≥2):
  baseline_deployable = none + cc@τ_val_ari ; best_val_deployable = argmax val-ARI sobre calibrador×τobj.
Diagnósticos privilegiados (calibrador none, gaps within-family vs baseline none):
  oracle_tau_global_test, oracle_tau_per_mixture_test, agglo_true_k(none).
Contrastes B vs B-local/B-shuffle/B-minus/A-rich en ARI: bajo regla común (none|ari) Y bajo best_val.

Salidas: REPORTE_0.5.md + calibration.json. CPU. GO/NO-GO lo decide el usuario.
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = ("A-naive", "A-rich", "B", "B-minus", "B-local", "B-shuffle")
RUNS = ("ID", "OOD-poly", "OOD-regime")
TAU_GRID = np.round(np.arange(0.10, 0.91, 0.05), 3)        # = grid del training (Codex r-plan-perf #3)
CONTRASTS = [("B", "B-local"), ("B", "B-shuffle"), ("B", "B-minus"), ("B-minus", "A-rich")]
CALIBS = ("none", "platt", "isotonic")
OBJECTIVES = ("ari", "f1")


def fast_ari(a, b):
    """Adjusted Rand Index inline (idéntico a sklearn, validado a 1e-16; sin su overhead)."""
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


# ---------------------------------------------------------------------------
# Carga + ensemble (logits crudos promediados sobre seeds) + asserts de fidelidad
# ---------------------------------------------------------------------------
def load_ensemble(results_dir: Path, subdir: str, run: str, model: str, seeds):
    """{mixture_id: {logit_mat(mean raw over seeds), pair_valid, target_mat, n, poly, reg, mid}}.

    Asserta igualdad de target_mat/pair_valid/n/poly/reg entre seeds antes de promediar logits
    (Codex r-plan-perf #4): el dataset es determinista, así que deben coincidir exactamente.
    """
    per_seed = []
    for s in seeds:
        f = results_dir / subdir / f"{run}__{model}__seed{s}.npz"
        if not f.exists():
            return None
        with np.load(f, allow_pickle=True) as d:
            # cargar CADA clave UNA sola vez (Codex): d["clave"] en un NpzFile re-lee/descomprime
            # el array entero del zip en cada acceso → indexar dentro del loop era O(N²).
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
        for ps in per_seed[1:]:                       # asserts de fidelidad entre seeds
            e = ps[mid]
            assert e["n"] == e0["n"] and e["poly"] == e0["poly"] and e["reg"] == e0["reg"], \
                f"{run}/{model} mid {mid}: meta difiere entre seeds"
            assert np.array_equal(e["target_mat"], e0["target_mat"]), f"{run}/{model} mid {mid}: target difiere"
            assert np.array_equal(e["pair_valid"], e0["pair_valid"]), f"{run}/{model} mid {mid}: pair_valid difiere"
        logits = np.stack([ps[mid]["logit_mat"] for ps in per_seed])
        out[mid] = {"logit_mat": logits.mean(axis=0), "pair_valid": e0["pair_valid"],
                    "target_mat": e0["target_mat"], "n": e0["n"], "poly": e0["poly"],
                    "reg": e0["reg"], "mid": mid}
    return out


def _true_source(target_mat, n):
    adj = (target_mat[:n, :n] >= 0.5).copy()
    np.fill_diagonal(adj, True)
    _, lab = connected_components(adj, directed=False)
    return lab


def pooled_valid_pairs(mixes):
    """(logits, targets) de pares válidos i<j de un iterable de mezclas (calibración / F1)."""
    lg, tg = [], []
    for m in mixes:
        n = m["n"]
        if n < 2:
            continue
        iu, ju = np.triu_indices(n, 1)
        v = m["pair_valid"][:n, :n][iu, ju]
        lg.append(m["logit_mat"][:n, :n][iu, ju][v])
        tg.append((m["target_mat"][:n, :n][iu, ju][v] >= 0.5).astype(int))
    if not lg:
        return np.array([]), np.array([])
    return np.concatenate(lg), np.concatenate(tg)


def make_calibrator(name, vlog, vtg):
    if name == "none" or len(vlog) == 0 or vtg.min() == vtg.max():
        return lambda x: expit(x)
    if name == "platt":
        clf = LogisticRegression(max_iter=2000).fit(vlog.reshape(-1, 1), vtg)
        return lambda x: clf.predict_proba(np.asarray(x).reshape(-1, 1))[:, 1].reshape(np.asarray(x).shape)
    if name == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(vlog, vtg)
        return lambda x: ir.predict(np.asarray(x).ravel()).reshape(np.asarray(x).shape)
    raise ValueError(name)


def calib_prob_mat(calib, logit_mat, n):
    sub = logit_mat[:n, :n]
    p = calib(sub.ravel()).reshape(n, n)
    return 0.5 * (p + p.T)


# ---------------------------------------------------------------------------
# Tabla ari[mix, tau] — computada UNA vez por (mixes, calibrador). Codex r-plan-perf #5.
# ---------------------------------------------------------------------------
def ari_table(mixes, calib, taus):
    """[n_mix, n_tau] ARI. prob_mat + true_source una vez por mezcla; clustering inline por τ."""
    out = np.full((len(mixes), len(taus)), np.nan)
    for i, m in enumerate(mixes):
        n = m["n"]
        if n < 2:
            continue
        pm = calib_prob_mat(calib, m["logit_mat"], n)
        pv = m["pair_valid"][:n, :n]
        true = m["_true"]
        for j, tau in enumerate(taus):
            adj = (pm >= tau) & pv
            np.fill_diagonal(adj, True)
            adj = adj | adj.T
            _, pred = connected_components(adj, directed=False)
            out[i, j] = fast_ari(true, pred)
    return out


def tau_f1(mixes, calib):
    """τ que maximiza F1 pooled en val (sobre prob calibrada). No clusteriza."""
    vlog, vtg = pooled_valid_pairs(mixes)
    if len(vlog) == 0 or vtg.min() == vtg.max():
        return float(TAU_GRID[0])
    prob = calib(vlog)
    best_tau, best = float(TAU_GRID[0]), -1.0
    for tau in TAU_GRID:
        s = f1_score(vtg, (prob >= tau).astype(int), zero_division=0)
        if s > best:
            best, best_tau = s, float(tau)
    return best_tau


# ---------------------------------------------------------------------------
# Procesamiento por (run, model) — unidad paralelizable
# ---------------------------------------------------------------------------
def process_run_model(task):
    t0 = time.time()
    rd, run, model, seeds = task
    logger.info("[load] %s/%s val_mats...", run, model)
    val = load_ensemble(rd, "val_mats", run, model, seeds)
    logger.info("[load] %s/%s test_mats... (val %.0fs)", run, model, time.time() - t0)
    test = load_ensemble(rd, "test_mats", run, model, seeds)
    if val is None or test is None:
        logger.info("[skip] %s/%s (faltan artefactos)", run, model)
        return (run, model), None
    logger.info("[start] %s/%s  val=%d test=%d", run, model, len(val), len(test))
    for ens in (val, test):
        for m in ens.values():
            m["_true"] = _true_source(m["target_mat"], m["n"]) if m["n"] >= 2 else None

    val_pool = [m for m in val.values() if m["poly"] >= 2] or list(val.values())
    test_mixes = list(test.values())
    test_mids = [m["mid"] for m in test_mixes]
    ti = np.array([(int(m["poly"]), m["reg"]) for m in test_mixes], dtype=object)
    cells = sorted({(m["poly"], m["reg"]) for m in test_mixes})

    calibs = {nm: make_calibrator(nm, *pooled_valid_pairs(val_pool)) for nm in CALIBS}

    # tabla ARI de val y test por calibrador (test solo para 'none' completa; platt/iso por τ elegido)
    rules = {}            # "cal|obj" -> {calib_name, objective, tau, val_ari}
    test_ari = {}         # "cal|obj" -> {mid: ari}
    none_test_tab = None  # [n_test, n_tau] para oráculos

    def tau_idx(tau):
        return int(np.argmin(np.abs(TAU_GRID - tau)))

    for cname, calib in calibs.items():
        val_tab = ari_table(val_pool, calib, TAU_GRID)           # [n_valpool, n_tau]
        mean_val = np.nanmean(val_tab, axis=0)
        t_ari = float(TAU_GRID[int(np.nanargmax(mean_val))])
        t_f1 = tau_f1(val_pool, calib)
        if cname == "none":
            none_test_tab = ari_table(test_mixes, calib, TAU_GRID)
            for obj, tau in (("ari", t_ari), ("f1", t_f1)):
                col = none_test_tab[:, tau_idx(tau)]
                test_ari[f"none|{obj}"] = {test_mids[i]: float(col[i]) for i in range(len(test_mids))}
                rules[f"none|{obj}"] = {"calib_name": "none", "objective": obj, "tau": tau,
                                        "val_ari": float(mean_val[tau_idx(tau)])}
        else:
            test_tab = ari_table(test_mixes, calib, np.array([t_ari, t_f1]))   # solo 2 columnas
            for k, (obj, tau) in enumerate((("ari", t_ari), ("f1", t_f1))):
                col = test_tab[:, k]
                test_ari[f"{cname}|{obj}"] = {test_mids[i]: float(col[i]) for i in range(len(test_mids))}
                rules[f"{cname}|{obj}"] = {"calib_name": cname, "objective": obj, "tau": tau,
                                           "val_ari": float(mean_val[tau_idx(tau)])}

    # oráculos por celda (calibrador none, desde none_test_tab) — within-family
    oracles = {}
    for (p, rg) in cells:
        mask = np.array([(m["poly"] == p and m["reg"] == rg) for m in test_mixes])
        sub = none_test_tab[mask]
        if sub.shape[0] == 0:
            continue
        oracle_global = float(np.nanmax(np.nanmean(sub, axis=0)))
        oracle_permix = float(np.nanmean(np.nanmax(sub, axis=1)))
        oracles[(p, rg)] = {"oracle_global": oracle_global, "oracle_permix": oracle_permix,
                            "agglo_true_k": _agglo_true_k(test_mixes, calibs["none"], p, rg)}

    meta = {m["mid"]: {"poly": m["poly"], "reg": m["reg"]} for m in test_mixes}
    logger.info("[done]  %s/%s en %.0fs", run, model, time.time() - t0)
    return (run, model), {"rules": rules, "test_ari": test_ari, "oracles": oracles, "meta": meta,
                          "cell_keys": [list(c) for c in cells]}


def _agglo_true_k(test_mixes, calib, poly, reg):
    vals = []
    for m in test_mixes:
        if m["poly"] != poly or m["reg"] != reg:
            continue
        n = m["n"]
        if n < 2:
            continue
        true = m["_true"]; k = len(set(true.tolist()))
        pm = calib_prob_mat(calib, m["logit_mat"], n)
        if k < 2:
            pred = np.zeros(n, dtype=int)
        elif k >= n:
            pred = np.arange(n)
        else:
            dist = 1.0 - pm; np.fill_diagonal(dist, 0.0); dist = 0.5 * (dist + dist.T)
            pred = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                           linkage="average").fit_predict(dist)
        vals.append(fast_ari(true, pred))
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Bootstrap pareado por mezcla + helpers de celda
# ---------------------------------------------------------------------------
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
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(a.mean() - b.mean()),
            "ci95_lo": float(lo), "ci95_hi": float(hi), "frac_positive": float((diffs > 0).mean()),
            "n_mixtures": M}


def cell_mean(ari, meta, poly, reg):
    vals = [ari[mid] for mid, mm in meta.items()
            if mm["poly"] == poly and mm["reg"] == reg and mid in ari and not np.isnan(ari[mid])]
    return float(np.mean(vals)) if vals else float("nan")


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
        for task in tasks:
            key, val = process_run_model(task)
            results[key] = val
    results = {k: v for k, v in results.items() if v is not None}
    present_runs = [r for r in args.runs if any((r, m) in results for m in args.models)]
    present_models = [m for m in args.models if any((r, m) in results for r in args.runs)]

    lines = ["# REPORTE_0.5 — Atención Armónica: auditoría de calibración / clustering\n"]
    lines.append("> Post-audit de Fase 0. Representación = AUC/AP (REPORTE_0). Acá: decisión de "
                 "clustering. Ensemble de logits crudos (3 seeds) → calibrador único en val. Grid τ = "
                 "training (0.10..0.90). Deployable = sin info de test; privilegiado = upper-bound.\n")
    export = {"per_run_model": {}, "contrasts": []}

    for run in present_runs:
        lines.append(f"\n## Run {run}\n")
        cells = sorted({tuple(c) for m in present_models if (run, m) in results
                        for c in map(tuple, results[(run, m)]["cell_keys"]) if c[0] >= 2})
        hdr = " | ".join(f"poly{p}_{r}" for (p, r) in cells)
        sep = "|" + "---|" * (len(cells) + 1)
        verdict_rule = {}

        lines.append("### ARI deployable por modelo (baseline=none|ari ; best_val=argmax val-ARI)\n")
        lines.append(f"| Modelo | regla | {hdr} |")
        lines.append(sep)
        for m in present_models:
            if (run, m) not in results:
                continue
            R = results[(run, m)]
            best_rname = max(R["rules"], key=lambda rn: (R["rules"][rn]["val_ari"]
                             if not np.isnan(R["rules"][rn]["val_ari"]) else -2.0))
            verdict_rule[m] = best_rname
            for tag, rn in (("baseline[none|ari]", "none|ari"), (f"best_val[{best_rname}]", best_rname)):
                ari = R["test_ari"][rn]
                lines.append("| " + " | ".join([m, tag] + [f"{cell_mean(ari, R['meta'], p, r):.3f}"
                                                            for (p, r) in cells]) + " |")
            export["per_run_model"][f"{run}|{m}"] = {"verdict_rule": best_rname, "rules": R["rules"]}
        lines.append("")

        lines.append("### Upper-bounds privilegiados (none) + gaps within-family (vs baseline none|ari)\n")
        lines.append(f"| Modelo | métrica | {hdr} |")
        lines.append(sep)
        for m in present_models:
            if (run, m) not in results:
                continue
            R = results[(run, m)]
            base = R["test_ari"]["none|ari"]
            basec = {c: cell_mean(base, R["meta"], c[0], c[1]) for c in cells}
            og = {c: R["oracles"].get(c, {}).get("oracle_global", np.nan) for c in cells}
            opm = {c: R["oracles"].get(c, {}).get("oracle_permix", np.nan) for c in cells}
            atk = {c: R["oracles"].get(c, {}).get("agglo_true_k", np.nan) for c in cells}
            bvc = {c: cell_mean(R["test_ari"][verdict_rule[m]], R["meta"], c[0], c[1]) for c in cells}
            def row(name, dd): return "| " + " | ".join([m, name] + [f"{dd.get(c, float('nan')):.3f}" for c in cells]) + " |"
            lines.append(row("baseline(none|ari)", basec))
            lines.append(row("oracle_global", og)); lines.append(row("oracle_permix", opm))
            lines.append(row("agglo_true_k", atk))
            lines.append(row("gap_dist", {c: og[c] - basec[c] for c in cells}))
            lines.append(row("gap_extreme", {c: opm[c] - basec[c] for c in cells}))
            lines.append(row("gap_k", {c: atk[c] - basec[c] for c in cells}))
            lines.append(row("calib_gain(best_val−base)", {c: bvc[c] - basec[c] for c in cells}))
        lines.append("")

        for rule_label, rule_fn in (("común none|ari", lambda m: "none|ari"),
                                    ("best_val por-modelo", lambda m: verdict_rule[m])):
            lines.append(f"### Contrastes ΔARI bajo {rule_label} — bootstrap pareado\n")
            lines.append("| Contraste | Celda | ΔARI | CI95 | P(Δ>0) | n |")
            lines.append("|---|---|---|---|---|---|")
            for (a, b) in CONTRASTS:
                if (run, a) not in results or (run, b) not in results:
                    continue
                Ra, Rb = results[(run, a)], results[(run, b)]
                aria, arib = Ra["test_ari"][rule_fn(a)], Rb["test_ari"][rule_fn(b)]
                for (p, r) in cells:
                    res = boot_ari(aria, Ra["meta"], arib, p, r, n_boot=args.n_boot)
                    if res is None:
                        continue
                    excl = "" if (res["ci95_lo"] <= 0 <= res["ci95_hi"]) else " *"
                    lines.append(f"| {a} vs {b} | poly{p}_{r} | {res['mean_diff']:+.3f}{excl} | "
                                 f"[{res['ci95_lo']:+.3f},{res['ci95_hi']:+.3f}] | {res['frac_positive']:.2f} | {res['n_mixtures']} |")
                    export["contrasts"].append({"run": run, "contrast": f"{a} vs {b}",
                                                "rule": rule_label, "cell": f"poly{p}_{r}", **res})
            lines.append("")

    lines.append("\n## Lectura (GO/NO-GO lo decide el usuario)\n")
    lines.append("- `gap_dist` grande = el τ no transfiere; con τ bien elegido por distribución se "
                 "recuperaría ARI. `gap_extreme` = techo no realista. `gap_k` = cuánto falta por no conocer k.")
    lines.append("- Si `calib_gain` (platt/isotonic/τ_val_f1) cierra el `gap_dist` sin tocar test → la "
                 "calibración deployable resuelve el caveat de Fase 0. Si no, queda como límite del sistema.")
    lines.append("- Contrastes con `*` excluyen 0. Se reportan bajo regla común (none|ari) y bajo best_val por-modelo.")

    (rd / "REPORTE_0.5.md").write_text("\n".join(lines), encoding="utf-8")
    (rd / "calibration.json").write_text(json.dumps(export, indent=2, default=float))
    logger.info("Escrito REPORTE_0.5.md + calibration.json (%d run-model)", len(results))


if __name__ == "__main__":
    main()
