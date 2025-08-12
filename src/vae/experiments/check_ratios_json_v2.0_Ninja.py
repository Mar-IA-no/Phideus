#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# check_ratios_json.py · v 2.0   (circuito‑ninja edition)
#
# Lee el JSON del analizador v3.x y genera:
#   1) Tabla‑resumen:  archivo | Ratio objetivo | % de energía en bin objetivo
#   2) Tabla detallada con todas las columnas de intervalos ninja
#
# La lista TARGETS cubre **todos** los ratios sintetizados por generar_wavs_con_ratio_v2.0.py.
# Si duplicas o añades WAVs con otro ratio, basta con sumar la tupla (nombre, ratio)
# aquí mismo o pasar --extra "nombre:valor" por CLI.
# ─────────────────────────────────────────────────────────────────────────────

import argparse, json, math, csv, sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from tabulate import tabulate

# ╭──────────────────────────╮
# │  CONFIG AJUSTABLE        │
# ╰──────────────────────────╯
TOL_CENT  = 40
MAX_RATIO = 6.1  # un poco más del máx (bronze≈3.30, 3:1=3) pero deja margen

phi  = (1+math.sqrt(5))/2
phi2 = phi**2
sqrt2 = math.sqrt(2)

TARGETS: List[Tuple[str, float]] = [
    # serie armónica
    ("9/8", 9/8), ("8/7", 8/7), ("7/6", 7/6), ("6/5", 6/5),
    ("11/8", 11/8), ("13/8", 13/8),
    # subarmónicos duplican ratio (idéntico)
    ("81/80", 81/80), ("pyth_comm", 531441/524288), ("33/32", 33/32),
    # irracionales & medios
    ("√2", sqrt2), ("√3", math.sqrt(3)), ("√5", math.sqrt(5)),
    ("φ", phi), ("φ²", phi2), ("σ", 1+sqrt2), ("bronze", 3.302775637731995),
    # séptimas
    ("7/4", 7/4), ("9/5", 9/5),
    # casi duplicados & micro
    ("10/9", 10/9), ("16/15", 16/15),
    # octava fraccionada
    ("3/1", 3/1), ("5/2", 5/2), ("7/3", 7/3),
    # octave, tritone for legacy
    ("2/1", 2.0), ("45/32", 45/32),
]

# CLI can extend target list quickly: --extra "name:ratio,name2:ratio2"

def extend_targets(extra: str):
    for item in extra.split(','):
        if ':' not in item:
            continue
        name, val = item.split(':', 1)
        TARGETS.append((name.strip(), float(val)))

TOL_LOG = TOL_CENT / 1200

# ╭──────────────────────────╮
# │  ANALÍTICA               │
# ╰──────────────────────────╯

def analyze(info: Dict[str, Any]):
    hist = np.array(info["ratio_hist"])
    n_bins = len(hist)
    log_max = math.log2(MAX_RATIO)
    peak_pct = {}
    for name, ratio in TARGETS:
        center = math.log2(ratio)
        lo = max(0, int((center - TOL_LOG) / log_max * n_bins))
        hi = min(n_bins - 1, int((center + TOL_LOG) / log_max * n_bins))
        peak_pct[name] = hist[lo:hi+1].sum() * 100
    best_name = max(peak_pct, key=peak_pct.get)
    return best_name, peak_pct[best_name], peak_pct

# ╭──────────────────────────╮
# │  FORMATO TABLAS          │
# ╰──────────────────────────╯

def make_tables(js: Dict[str, Any], skip_zero=False):
    rows_detail, rows_summary = [], []
    header_targets = [n for n, _ in TARGETS]
    for rel, info in js.items():
        best_name, best_pct, pct_dict = analyze(info)
        rows_summary.append([rel, best_name, f"**{best_pct:.1f} %**"])
        row = {"archivo": rel}
        for k in header_targets:
            val = pct_dict.get(k, 0.0)
            row[k] = "" if (skip_zero and val < 1e-8) else val
        rows_detail.append(row)
    return rows_summary, rows_detail, header_targets

# ╭──────────────────────────╮
# │  CLI & SALIDA            │
# ╰──────────────────────────╯

def main():
    ap = argparse.ArgumentParser(description="Check ratios JSON – circuito ninja")
    ap.add_argument("json", type=Path)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--csv", action="store_true", help="CSV con ';'")
    ap.add_argument("--comma", action="store_true", help="CSV con ','")
    ap.add_argument("--extra", type=str, help="Añade pares name:ratio separados por coma")
    ap.add_argument("--no-zeros", action="store_true")
    args = ap.parse_args()

    if args.extra:
        extend_targets(args.extra)

    data = json.loads(Path(args.json).read_text())
    summary, detail, header_targets = make_tables(data, skip_zero=args.no_zeros)

    # imprime resumen
    fmt = "github" if args.markdown else "simple"
    print(tabulate(summary, headers=["archivo", "Ratio objetivo", "% de energía hallada en bin"], tablefmt=fmt))
    print()

    if args.csv or args.comma:
        delim = ',' if args.comma else ';'
        writer = csv.DictWriter(sys.stdout, fieldnames=["archivo"]+header_targets, delimiter=delim, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader(); writer.writerows(detail)
    else:
        print(tabulate([[d.get(col, "") for col in ["archivo"]+header_targets] for d in detail], headers=["archivo"]+header_targets, 
tablefmt=fmt, floatfmt=".1f"))

if __name__ == "__main__":
    main()

