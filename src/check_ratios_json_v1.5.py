#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# check_ratios_json.py · v 1.5            (compatible con analizador v3.x)
#
# ‣ Funciones:  • Tabla detallada (igual que v1.4)
#               • NUEVO: tabla‑resumen «archivo | Ratio objetivo | % energía…»
#                 donde «Ratio objetivo» es el intervalo con mayor energía de la
#                 lista TARGETS y el % correspondiente se muestra en negrita.
#
# Uso rápido:
#   python check_ratios_json_v1.5.py  ratios_dataset.json --markdown
#
# Flags extra:
#   --summary‑only      → imprime solo la tabla‑resumen
#   --no-summary        → omite la tabla‑resumen (solo tabla grande/CSV)
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
MAX_RATIO = 6.0
TARGETS: List[Tuple[str, float]] = [
    ("quinta_justa 3:2", 3/2),
    ("tercera_mayor 5:4", 5/4),
    ("cuarta_justa 4:3", 4/3),
    ("phi 1.618", 1.618),
    ("raiz_2", math.sqrt(2)),
    ("tritono 45:32", 45/32),
    ("octava 2:1", 2.0),
]
TOL_LOG = TOL_CENT / 1200

# ╭──────────────────────────╮
# │  ANALÍTICA               │
# ╰──────────────────────────╯

def analyze(entry: Dict[str, Any]):
    hist = np.array(entry["ratio_hist"])
    n_bins = len(hist)
    log_max = math.log2(MAX_RATIO)

    peaks = len(entry.get("peak_freqs", []))
    ratios_count = len(entry.get("ratios_lin", entry.get("ratios", [])))

    pct = {}
    for name, ratio in TARGETS:
        center = math.log2(ratio)
        lo = max(0, int((center - TOL_LOG) / log_max * n_bins))
        hi = min(n_bins - 1, int((center + TOL_LOG) / log_max * n_bins))
        pct[name] = hist[lo:hi+1].sum() * 100
    return peaks, ratios_count, pct

# ╭──────────────────────────╮
# │  FILAS TABLA PRINCIPAL   │
# ╰──────────────────────────╯

def build_rows(js: Dict[str, Any], max_bins: int, skip_zero: bool):
    header_cols = [n for n, _ in TARGETS]
    rows = []
    for wav_rel, info in js.items():
        peaks, ratios_count, pct = analyze(info)
        row = {"archivo": wav_rel, "#picos": peaks, "#ratios": ratios_count}
        row.update(pct)
        if skip_zero:
            for k in header_cols:
                if abs(row.get(k, 0)) < 1e-8:
                    row[k] = ""
        rows.append(row)
    return rows, header_cols

# ╭──────────────────────────╮
# │  TABLA RESUMEN           │
# ╰──────────────────────────╯

def build_summary(rows: List[Dict[str, Any]]) -> List[List[str]]:
    summary = []
    target_names = [n for n, _ in TARGETS]
    for r in rows:
        pct_vals = {k: r.get(k, 0.0) for k in target_names}
        best = max(pct_vals, key=pct_vals.get)
        best_pct = pct_vals[best]
        summary.append([
            r["archivo"],
            best,
            f"**{best_pct:.1f} %**" if best_pct else "0.0 %",
        ])
    return summary

# ╭──────────────────────────╮
# │  CLI                     │
# ╰──────────────────────────╯

def main() -> None:
    ap = argparse.ArgumentParser(description="Audita ratios_dataset.json y genera tabla resumen")
    ap.add_argument("json", type=Path, help="Archivo JSON de entrada")
    ap.add_argument("--csv", action="store_true", help="Salida CSV (tabla completa)")
    ap.add_argument("--comma", action="store_true", help="CSV con ','")
    ap.add_argument("--markdown", action="store_true", help="Tablas en formato GitHub Markdown")
    ap.add_argument("--summary-only", action="store_true", help="Mostrar solo la tabla resumen")
    ap.add_argument("--no-summary", action="store_true", help="Omitir tabla resumen")
    ap.add_argument("--no-zeros", action="store_true", help="Vaciar celdas con 0 % en tabla completa")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    rows, header_cols = build_rows(data, max_bins=10, skip_zero=args.no_zeros)

    # Tabla resumen
    if not args.no_summary:
        summary = build_summary(rows)
        if args.csv or args.comma:
            delim = ',' if args.comma else ';'
            w = csv.writer(sys.stdout, delimiter=delim, quoting=csv.QUOTE_MINIMAL)
            w.writerow(["archivo", "Ratio objetivo", "% de energía hallada en el bin del objetivo"])
            w.writerows([[*s] for s in summary])
            if not args.summary_only:
                print()  # línea vacía antes de la tabla grande
        else:
            fmt = "github" if args.markdown else "simple"
            print(tabulate(summary,
                           headers=["archivo", "Ratio objetivo", "% de energía hallada en el bin del objetivo"],
                           tablefmt=fmt, stralign="center", numalign="right"))
            if not args.summary_only:
                print()

    # Tabla completa
    if not args.summary_only:
        if args.csv or args.comma:
            delim = ',' if args.comma else ';'
            writer = csv.DictWriter(sys.stdout, fieldnames=["archivo", "#picos", "#ratios"] + header_cols, delimiter=delim, 
quoting=csv.QUOTE_MINIMAL)
            writer.writeheader(); writer.writerows(rows)
        else:
            fmt = "github" if args.markdown else "simple"
            print(tabulate([[r.get(f, "") for f in ["archivo", "#picos", "#ratios"] + header_cols] for r in rows],
                           headers=["archivo", "#picos", "#ratios"] + header_cols,
                           tablefmt=fmt, floatfmt=".1f"))

if __name__ == "__main__":
    main()

