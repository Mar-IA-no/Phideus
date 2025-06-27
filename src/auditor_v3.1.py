#!/usr/bin/env python3
# check_ratios_json_general_v3.0.py  (auditor general mejorado)
import argparse, json, math, csv, sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from tabulate import tabulate

# ╭──────────────────────────╮
# │  CONFIGURACIÓN          │
# ╰──────────────────────────╯
TOL_CENT_DEFAULT = 40      # tolerancia en cents para etiquetado
MIN_BIN_PCT      = 1.0     # porcentaje mínimo para considerar un bin activo

# Diccionario de intervalos semánticos: nombre -> ratio
SEMANTIC_RATIOS: List[Tuple[str, float]] = [
    ("Unísono",        1.0),
    ("Segunda menor",  16/15),
    ("Segunda mayor",  9/8),
    ("Tercera menor",  6/5),
    ("Tercera mayor",  5/4),
    ("Cuarta justa",   4/3),
    ("Tritono",        45/32),
    ("Quinta justa",   3/2),
    ("Sexta menor",    8/5),
    ("Sexta mayor",    5/3),
    ("Séptima menor",  9/5),
    ("Séptima mayor",  15/8),
    ("Octava",         2.0),
    ("√2",             math.sqrt(2)),
    ("√3",             math.sqrt(3)),
    ("φ (phi)",        (1+math.sqrt(5))/2),
]

# ╭──────────────────────────╮
# │  UTILIDADES             │
# ╰──────────────────────────╯
def cents_diff(r1: float, r2: float) -> float:
    """Diferencia absoluta en cents entre dos ratios."""
    return abs(1200 * math.log2(r1 / r2))

# ╭──────────────────────────╮
# │  ANÁLISIS POR ARCHIVO    │
# ╰──────────────────────────╯
def analyze_entry(info: Dict[str, Any], tol_cents: float) -> Tuple[int, float, float, str]:
    hist = np.array(info.get("ratio_hist", []))
    n_bins = len(hist)
    active_bins = int(np.sum(hist >= MIN_BIN_PCT/100))

    # Entropía de Shannon
    p = hist[hist > 0]
    H_bits = -float(np.sum(p * np.log2(p))) if p.size else 0.0

    # Flatness (geometric / arithmetic mean)
    gm = float(np.exp(np.mean(np.log(hist + 1e-12))))
    am = float(np.mean(hist))
    flatness = gm / am if am > 0 else 0.0

    # Ratio dominante (centro del bin mayor)
    max_idx = int(np.argmax(hist))
    log_max = math.log2(info.get("max_ratio", 6.0))
    ratio_dom = 2 ** (((max_idx + 0.5) / n_bins) * log_max)

    # Etiqueta semántica más cercana
    best_name, best_ratio, delta = None, None, None
    for name, r in SEMANTIC_RATIOS:
        d = cents_diff(ratio_dom, r)
        if best_name is None or d < delta:
            best_name, best_ratio, delta = name, r, d

    if delta > tol_cents:
        dom_label = f"{ratio_dom:.4f} (sugerido: {best_name}, Δ {delta:.1f} cents — muy lejos)"
    else:
        dom_label = best_name or f"{ratio_dom:.4f}"

    return active_bins, H_bits, flatness, dom_label

# ╭──────────────────────────╮
# │  TABLA RESUMEN          │
# ╰──────────────────────────╯
def build_summary_table(results: List[Tuple[str, int, float, float, str]]) -> List[List[Any]]:
    return [[fname, bins, f"{H:.2f}", f"{flat:.3f}", dom]
            for fname, bins, H, flat, dom in results]

# ╭──────────────────────────╮
# │  DETALLADO              │
# ╰──────────────────────────╯
def print_detailed(data: Dict[str, Any], tol: float):
    for fname, info in data.items():
        bins, H, flat, dom_label = analyze_entry(info, tol)

        # Texto descriptivo
        ent_desc = "baja perfil concentrado" if H < math.log2(bins + 1) else "alta perfil disperso"
        flat_desc = "baja (picos marcados)" if flat < 0.5 else "alta distribución uniforme"

        print(f"## Archivo: {fname}\n")
        print("-- Métricas globales:")
        print(f" • Bins activos (≥ {MIN_BIN_PCT:.1f}%): {bins}")
        print(f" • Entropía de Shannon: {H:.2f} bits → {ent_desc}")
        print(f" • Planitud (flatness): {flat:.3f} → {flat_desc}\n")

        # Top 5 ratios
        print("-- Ratios principales:")
        hist = np.array(info.get('ratio_hist', []))
        if hist.size:
            top_idx = np.argsort(hist)[-5:][::-1]
            for idx in top_idx:
                pct = hist[idx] * 100
                ratio = 2 ** (((idx + 0.5) / len(hist)) * math.log2(info.get('max_ratio', 6.0)))
                # etiquetado igual a dominante
                best_name, _, delta = None, None, None
                for name, r in SEMANTIC_RATIOS:
                    d = cents_diff(ratio, r)
                    if best_name is None or d < delta:
                        best_name, _, delta = name, r, d
                if delta > tol:
                    label = f"{ratio:.4f} (sugerido: {best_name}, Δ {delta:.1f} cents)"
                else:
                    label = best_name or f"{ratio:.4f}"
                print(f" • {label} | Energía: {pct:.1f}%")
        print("\n")

# ╭──────────────────────────╮
# │  MAIN                   │
# ╰──────────────────────────╯
def main():
    parser = argparse.ArgumentParser(description="Audita ratios_dataset.json con sugerencias y advertencias")
    parser.add_argument('json', type=Path, help='Archivo JSON de entrada')
    parser.add_argument('--tolerance', '-t', type=float, default=TOL_CENT_DEFAULT,
                        help='Tolerancia en cents para sugerir etiqueta sin aviso')
    parser.add_argument('--markdown', action='store_true', help='Salida de tabla resumen en Markdown')
    args = parser.parse_args()

    data = json.loads(args.json.read_text(encoding='utf-8'))

    # Resumen
    results = [(fname, *analyze_entry(info, args.tolerance)) for fname, info in data.items()]
    table = build_summary_table(results)
    fmt = 'github' if args.markdown else 'simple'
    print(tabulate(table, headers=["archivo", "bins_activos", "entropía_bits", "planitud", "ratio_dominante"], tablefmt=fmt))
    print("\n")

    # Detallado
    print_detailed(data, args.tolerance)

if __name__ == '__main__':
    main()

