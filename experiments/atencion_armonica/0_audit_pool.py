#!/usr/bin/env python3
"""Gate de feature-triviality + calibración (v2.1) — Fase 0 Atención Armónica.

Gate OBLIGATORIO antes de GPU. Audita VALIDEZ del pool (no código).

Headroom (techo per-par de A-rich): los probes usan TODO lo que recibe A-rich =
4 features continuas + `ratio_class_id` (one-hot en LogReg, nn.Embedding en PairMLP).
  - max(AUC,1-AUC) de cada single continua < 0.90 (referencia).
  - LogReg [4 cont + one-hot(ratio_class)] CV-por-mezcla < 0.90.
  - PairMLP [4 cont + emb(ratio_class)] CV-por-mezcla < 0.90  ← criterio duro.
Solvabilidad:
  - oracle_privileged_upper_bound (usa (f0,β) verdaderos sin labels) min-cell ARI > 0.80 ← gate.
  - oracle_unpriv_f0only (EM sin info privilegiada, f0-only, IGNORA β) → LOWER BOUND, diagnóstico.
Gate y desempate por PEOR celda decisiva (poly2/3 × easy/hard); métricas reportadas POR CELDA.

Modos:
  --sweep        : 16 combos (β-center × α-range × σ_amp × p_drop) sobre calibration_pool en
                   memoria; elegibilidad+desempate por peor celda; trigger de revisión manual. → AUDIT_SWEEP.md
  --pool <jsonl> : gate PASS/ABORT sobre un pool final. → AUDIT_POOL.md

Uso:
    python experiments/atencion_armonica/0_audit_pool.py --sweep --out-dir data/atencion_armonica
    python experiments/atencion_armonica/0_audit_pool.py --pool data/atencion_armonica/pool/mixtures.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.atencion_armonica.grouping_dataset import load_pool  # noqa: E402
from src.atencion_armonica.harmonic_synth import generate_mixture  # noqa: E402
from src.atencion_armonica.peak_tokens import mixture_to_arrays, n_ratio_classes  # noqa: E402

DECISIVE_CELLS = [(2, "easy"), (2, "hard"), (3, "easy"), (3, "hard")]
AUC_GATE = 0.90
ORACLE_ARI_GATE = 0.80
PAIRMLP_ELIG = 0.88
ORACLE_ELIG = 0.85
PAIRMLP_TARGET = 0.83
N_CV = 5
PAIR_CAP = 50000
PAIRMLP_EPOCHS = 30
PAIRMLP_LR = 1e-3
PAIRMLP_WD = 1e-4
PAIRMLP_BATCH = 256
PAIRMLP_SEED = 42
EMB_DIM = 8
# sweep v2.1
BETA_CENTERS = (1e-3, 3e-3)
ALPHA_RANGES = ((0.5, 1.5), (0.5, 2.5))
SIGMA_AMPS = (0.5, 1.0)
P_DROPS = (0.15, 0.30)
CALIB_N_PER_CELL = 400
CALIB_SEED = 99991

FEATURE_NAMES = ["dlogf", "ratio_residual", "common_f0_residual", "log_amp_diff"]


class PairMLP(nn.Module):
    """Espejo de A-rich: 4 continuas + embedding(ratio_class_id) → 64 → 64 → 1."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(n_ratio_classes(), EMB_DIM)
        self.net = nn.Sequential(
            nn.Linear(4 + EMB_DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x_cont, ratio_id):
        return self.net(torch.cat([x_cont, self.emb(ratio_id)], dim=-1)).squeeze(-1)


def cell_pairs(mixtures):
    """Pares válidos i<j → X_cont[P,4], ratio_id[P], y[P], groups[P] (mixture idx). Subsample a PAIR_CAP."""
    Xc, Rid, ys, gs = [], [], [], []
    for gi, m in enumerate(mixtures):
        arr = mixture_to_arrays(m)
        N = arr["n_peaks"]
        if N < 2:
            continue
        iu, ju = np.triu_indices(N, k=1)
        v = arr["pair_valid"][iu, ju]
        Xc.append(arr["pair_cont"][iu, ju][v])
        Rid.append(arr["ratio_class_id"][iu, ju][v])
        ys.append(arr["target"][iu, ju][v])
        gs.append(np.full(int(v.sum()), gi))
    if not Xc:
        return (np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), np.zeros(0))
    Xc = np.concatenate(Xc); Rid = np.concatenate(Rid).astype(int)
    y = np.concatenate(ys); g = np.concatenate(gs)
    if len(y) > PAIR_CAP:
        rng = np.random.RandomState(0)
        sel = rng.choice(len(y), PAIR_CAP, replace=False)
        Xc, Rid, y, g = Xc[sel], Rid[sel], y[sel], g[sel]
    return Xc, Rid, y, g


def single_aucs(Xc, y):
    if len(y) == 0 or y.min() == y.max():
        return {n: float("nan") for n in FEATURE_NAMES}
    return {nm: float(max(a, 1 - a)) for nm, a in
            ((FEATURE_NAMES[i], roc_auc_score(y, Xc[:, i])) for i in range(4))}


def _cv(y, g):
    ng = len(np.unique(g))
    if y.min() == y.max() or ng < 2:
        return None
    return GroupKFold(n_splits=min(N_CV, ng))


def logreg_auc(Xc, Rid, y, g):
    gkf = _cv(y, g)
    if gkf is None:
        return float("nan")
    onehot = np.eye(n_ratio_classes())[Rid]
    X = np.concatenate([Xc, onehot], axis=1)
    aucs = []
    for tr, te in gkf.split(X, y, g):
        if y[tr].min() == y[tr].max() or y[te].min() == y[te].max():
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def pairmlp_auc(Xc, Rid, y, g):
    gkf = _cv(y, g)
    if gkf is None:
        return float("nan")
    torch.manual_seed(PAIRMLP_SEED)
    aucs = []
    for tr, te in gkf.split(Xc, y, g):
        if y[tr].min() == y[tr].max() or y[te].min() == y[te].max():
            continue
        mu = Xc[tr].mean(0); sd = Xc[tr].std(0) + 1e-8
        Xtr = torch.tensor((Xc[tr] - mu) / sd, dtype=torch.float32)
        Xte = torch.tensor((Xc[te] - mu) / sd, dtype=torch.float32)
        Rtr = torch.tensor(Rid[tr], dtype=torch.long); Rte = torch.tensor(Rid[te], dtype=torch.long)
        ytr = torch.tensor(y[tr], dtype=torch.float32)
        m = PairMLP(); opt = torch.optim.Adam(m.parameters(), lr=PAIRMLP_LR, weight_decay=PAIRMLP_WD)
        m.train()
        for _ in range(PAIRMLP_EPOCHS):
            perm = torch.randperm(len(Xtr))
            for b in range(0, len(Xtr), PAIRMLP_BATCH):
                idx = perm[b:b + PAIRMLP_BATCH]
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(m(Xtr[idx], Rtr[idx]), ytr[idx])
                loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = torch.sigmoid(m(Xte, Rte)).numpy()
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)) if aucs else float("nan")


def oracle_privileged(mixtures):
    """Asigna cada pico a la fuente cuyo modelo VERDADERO (f0_s,β_s) mejor lo predice. ARI vs verdad."""
    aris = []
    for m in mixtures:
        src = np.array([pk["source_id"] for pk in m["peaks"]])
        k = len(set(src.tolist()))
        if k < 2 or len(src) < 2:
            continue
        models = [(s["f0"], s["beta"]) for s in m["sources"]]
        logf = np.log([pk["freq"] for pk in m["peaks"]])
        h = np.arange(1, 9)
        pred = np.zeros(len(logf), dtype=int)
        for i, lf in enumerate(logf):
            best = (1e9, 0)
            for sidx, (f0, beta) in enumerate(models):
                logmodel = np.log(h * f0 * np.sqrt(1.0 + beta * h ** 2))
                r = np.min(np.abs(lf - logmodel))
                if r < best[0]:
                    best = (r, sidx)
            pred[i] = best[1]
        aris.append(adjusted_rand_score(src, pred))
    return float(np.mean(aris)) if aris else float("nan")


def oracle_unpriv_f0only(mixtures, n_iter=6):
    """LOWER BOUND f0-only (Codex r8 #1): EM SIN info privilegiada, solo k verdadero. Init k-means
    en log-f; fit f0 (IGNORA β a propósito); reasignar. NO usa f0_s/β_s ni source_id PARA PREDECIR;
    source_id solo se usa para el scoring (ARI). Es un LOWER BOUND: al ignorar β sub-reporta
    solvabilidad. Diagnóstico, NO gate. Un gap grande vs oracle_privileged refleja cuánta de la
    estructura depende de β (no necesariamente 'no aprendible')."""
    aris = []
    for m in mixtures:
        src = np.array([pk["source_id"] for pk in m["peaks"]])
        k = len(set(src.tolist()))
        freqs = np.array([pk["freq"] for pk in m["peaks"]])
        if k < 2 or len(freqs) < 2:
            continue
        logf = np.log(freqs).reshape(-1, 1)
        try:
            assign = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(logf)
        except Exception:
            continue
        h = np.arange(1, 9)
        for _ in range(n_iter):
            f0s = []
            for c in range(k):
                fc = freqs[assign == c]
                if len(fc) == 0:
                    f0s.append(np.median(freqs)); continue
                f0_init = fc.min()
                hh = np.clip(np.round(fc / f0_init), 1, 8)
                f0s.append(float(np.median(fc / hh)))
            new = assign.copy()
            for i, f in enumerate(freqs):
                res = [np.min(np.abs(np.log(f) - np.log(h * f0))) for f0 in f0s]
                new[i] = int(np.argmin(res))
            if np.array_equal(new, assign):
                break
            assign = new
        aris.append(adjusted_rand_score(src, assign))
    return float(np.mean(aris)) if aris else float("nan")


def evaluate_cells(cells_mixtures):
    out = {}
    for cell, mixtures in cells_mixtures.items():
        Xc, Rid, y, g = cell_pairs(mixtures)
        sa = single_aucs(Xc, y)
        out[cell] = {
            "max_single_auc": float(max((v for v in sa.values() if v == v), default=float("nan"))),
            "single_aucs": sa,
            "logreg_auc": logreg_auc(Xc, Rid, y, g),
            "pairmlp_auc": pairmlp_auc(Xc, Rid, y, g),
            "oracle_priv_ari": oracle_privileged(mixtures),
            "oracle_unpriv_f0only_ari": oracle_unpriv_f0only(mixtures),
        }
    return out


def gen_decisive_cells(beta_center, alpha_range, sigma_amp, p_drop, n_per_cell, seed):
    out = {}; mid = 0
    for (poly, regime) in DECISIVE_CELLS:
        ms = []
        for _ in range(n_per_cell):
            ms.append(generate_mixture(mid, poly, regime, seed, beta_center=beta_center,
                                       p_drop=p_drop, alpha_range=alpha_range,
                                       sigma_amp=sigma_amp).to_json())
            mid += 1
        out[f"poly{poly}_{regime}"] = ms
    return out


def run_sweep(out_dir: Path):
    combos = [(bc, ar, sa, pd) for bc in BETA_CENTERS for ar in ALPHA_RANGES
              for sa in SIGMA_AMPS for pd in P_DROPS]
    lines = ["# Sweep de calibración v2.1 — Atención Armónica\n"]
    lines.append(f"> calibration_pool seed={CALIB_SEED}, {CALIB_N_PER_CELL} mezclas/celda. "
                 "Elegibilidad/desempate por PEOR celda decisiva. PairMLP/LogReg con ratio_class_id.\n")
    lines.append("| β-c | α-range | σ_amp | p_drop | maxcell single | LogReg | **PairMLP** | mincell oraclePriv | oracleUnpriv | elegible |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for (bc, ar, sa, pd) in combos:
        cells = gen_decisive_cells(bc, ar, sa, pd, CALIB_N_PER_CELL, CALIB_SEED)
        ev = evaluate_cells(cells)
        max_single = max(ev[c]["max_single_auc"] for c in ev)
        max_logreg = max(ev[c]["logreg_auc"] for c in ev)
        max_pairmlp = max(ev[c]["pairmlp_auc"] for c in ev)          # peor celda
        min_priv = min(ev[c]["oracle_priv_ari"] for c in ev)
        min_unpriv = min(ev[c]["oracle_unpriv_f0only_ari"] for c in ev)
        # Elegibilidad según plan (Codex r8 #1): PairMLP con margen <0.88; single/LogReg al
        # umbral del gate final <0.90 (no 0.88, eso era más estricto que el plan); oracle_priv >0.85.
        elig = (max_pairmlp < PAIRMLP_ELIG and max_single < AUC_GATE
                and max_logreg < AUC_GATE and min_priv > ORACLE_ELIG)
        rows.append({"beta_center": bc, "alpha_range": list(ar), "sigma_amp": sa, "p_drop": pd,
                     "max_single": max_single, "max_logreg": max_logreg, "max_pairmlp": max_pairmlp,
                     "min_oracle_priv": min_priv, "min_oracle_unpriv": min_unpriv,
                     "eligible": bool(elig), "per_cell": ev})
        lines.append(f"| {bc:.0e} | {ar} | {sa} | {pd} | {max_single:.3f} | {max_logreg:.3f} | "
                     f"{max_pairmlp:.3f} | {min_priv:.3f} | {min_unpriv:.3f} | {'sí' if elig else 'no'} |")

    elig = [r for r in rows if r["eligible"]]
    chosen = None
    if elig:
        elig.sort(key=lambda r: (abs(r["max_pairmlp"] - PAIRMLP_TARGET), -r["min_oracle_priv"],
                                 r["beta_center"], r["sigma_amp"], r["p_drop"]))
        chosen = elig[0]

    lines.append("\n## Combo recomendada\n")
    if chosen:
        lines.append(f"**β-center={chosen['beta_center']:.0e}, α-range={chosen['alpha_range']}, "
                     f"σ_amp={chosen['sigma_amp']}, p_drop={chosen['p_drop']}** "
                     f"(PairMLP={chosen['max_pairmlp']:.3f}, oraclePriv={chosen['min_oracle_priv']:.3f}, "
                     f"oracleUnpriv={chosen['min_oracle_unpriv']:.3f}).")
        # trigger de revisión manual (Codex r7 #2)
        if chosen["min_oracle_unpriv"] < 0.05:
            lines.append("\n⚠️ **MANUAL REVIEW REQUIRED**: oracle_priv pasa pero oracle_unpriv_f0only≈0 → "
                         "cuánto depende de β. Decisión humana antes de GPU.")
        # tabla POR CELDA de la combo elegida (Codex r8 #6 — poly3_hard visible)
        lines.append("\n### Métricas por celda — combo elegida\n")
        lines.append("| Celda | max single | LogReg | PairMLP | oraclePriv | oracleUnpriv(f0only) |")
        lines.append("|---|---|---|---|---|---|")
        for cell, e in chosen["per_cell"].items():
            lines.append(f"| {cell} | {e['max_single_auc']:.3f} | {e['logreg_auc']:.3f} | "
                         f"{e['pairmlp_auc']:.3f} | {e['oracle_priv_ari']:.3f} | "
                         f"{e['oracle_unpriv_f0only_ari']:.3f} |")
    else:
        lines.append("⚠️ **NINGUNA combo elegible** → volver a plan-mode (formante/spurious/rangos). NO relajar el gate.")

    (out_dir / "AUDIT_SWEEP.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "AUDIT_SWEEP.json").write_text(json.dumps({"rows": rows, "chosen": chosen}, indent=2))
    print("\n".join(l for l in lines if not l.startswith("|")))
    return chosen


def run_gate(pool_path: Path, out_path: Path):
    pool = load_pool(pool_path)
    cells = {f"poly{p}_{r}": [m for m in pool if m["polyphony"] == p and m["regime"] == r]
             for (p, r) in DECISIVE_CELLS}
    ev = evaluate_cells(cells)
    lines = ["# Gate de feature-triviality v2.1 (pool final)\n"]
    lines.append(f"> Pool: {pool_path}. PASS si en TODA celda decisiva: single/LogReg/PairMLP < {AUC_GATE} "
                 f"Y oracle_priv ARI > {ORACLE_ARI_GATE}.\n")
    lines.append("| Celda | max single | LogReg | PairMLP | oraclePriv ARI | oracleUnpriv ARI |")
    lines.append("|---|---|---|---|---|---|")
    passed = True
    for cell, e in ev.items():
        viol = (e["max_single_auc"] >= AUC_GATE or e["logreg_auc"] >= AUC_GATE
                or e["pairmlp_auc"] >= AUC_GATE or e["oracle_priv_ari"] <= ORACLE_ARI_GATE)
        passed = passed and not viol
        lines.append(f"| {cell} | {e['max_single_auc']:.3f} | {e['logreg_auc']:.3f} | "
                     f"{e['pairmlp_auc']:.3f} | {e['oracle_priv_ari']:.3f} | {e['oracle_unpriv_f0only_ari']:.3f} |")
    lines.append(f"\n## Veredicto: {'PASS ✅' if passed else 'ABORT ❌'}\n")
    # MANUAL REVIEW también en el gate final (Codex r8 #2)
    priv_ok = all(e["oracle_priv_ari"] > ORACLE_ARI_GATE for e in ev.values())
    unpriv_collapse = all(e["oracle_unpriv_f0only_ari"] < 0.05 for e in ev.values())
    if priv_ok and unpriv_collapse:
        lines.append("⚠️ **MANUAL REVIEW REQUIRED**: oracle_priv pasa pero oracle_unpriv_f0only≈0 en "
                     "todas las celdas → revisar cuánto depende de β antes de GPU.\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    (out_path.with_suffix(".json")).write_text(json.dumps(ev, indent=2))
    print("\n".join(lines))
    if not passed:
        sys.exit(2)
    return ev


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--out-dir", default="data/atencion_armonica")
    p.add_argument("--pool", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.sweep:
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        run_sweep(out_dir)
    elif args.pool:
        out = Path(args.out) if args.out else Path(args.pool).parent / "AUDIT_POOL.md"
        run_gate(Path(args.pool), out)
    else:
        p.error("usar --sweep o --pool")


if __name__ == "__main__":
    main()
