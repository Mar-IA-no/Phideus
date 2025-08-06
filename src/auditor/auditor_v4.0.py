#!/usr/bin/env python3
"""
check_ratios_json_v4.0.py · Auditor combinado v4.0

Modos de análisis:
  --analisis armonico    → análisis perceptual/armónico (igual que v3.1).
  --analisis topologico  → análisis físico/topológico.
  --analisis comparativo → primero armónico, luego topológico.

Uso:
  python check_ratios_json_v4.0.py archivo.json --analisis armonico [--markdown] [-t TOL]
  python check_ratios_json_v4.0.py archivo.json --analisis topologico [-T UMBRAL]
  python check_ratios_json_v4.0.py archivo.json --analisis comparativo
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import correlate, find_peaks
from tabulate import tabulate

# Parámetros comunes
TOL_CENT_DEFAULT = 40.0    # tolerancia en cents (armónico)
MIN_BIN_PCT      = 1.0     # %% mínimo para bin activo (topológico)
SEMANTIC_RATIOS = [
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

# --- Funciones armónicas de v3.1 ---

def cents_diff(r1: float, r2: float) -> float:
    return abs(1200 * math.log2(r1 / r2))

def analyze_entry(info: dict, tol_cents: float):
    hist = np.array(info.get("ratio_hist", []))
    n_bins = len(hist)
    p = hist[hist > 0]
    H_bits = -float(np.sum(p * np.log2(p))) if p.size else 0.0
    gm = float(np.exp(np.mean(np.log(hist + 1e-12))))
    am = float(np.mean(hist))
    flat = gm / am if am > 0 else 0.0
    max_idx = int(np.argmax(hist))
    max_ratio_val = info.get("max_ratio", 6.0)
    ratio_dom = 2 ** (((max_idx + 0.5) / n_bins) * math.log2(max_ratio_val))
    best_name, best_ratio, delta = None, None, None
    for name, r in SEMANTIC_RATIOS:
        d = cents_diff(ratio_dom, r)
        if best_name is None or d < delta:
            best_name, best_ratio, delta = name, r, d
    if delta > tol_cents:
        dom_label = f"{ratio_dom:.4f} (sugerido: {best_name}, Δ{delta:.1f}c)"
    else:
        dom_label = best_name or f"{ratio_dom:.4f}"
    return int(np.sum(hist >= MIN_BIN_PCT/100)), H_bits, flat, dom_label

def build_summary_table(results):
    return [[fname, bins, f"{H:.2f}", f"{flat:.3f}", dom]
            for fname, bins, H, flat, dom in results]

def print_detailed(data: dict, tol_cents: float):
    for fname, info in data.items():
        bins, H, flat, dom_label = analyze_entry(info, tol_cents)
        ent_desc = "baja perfil concentrado" if H < math.log2(bins + 1) else "alta perfil disperso"
        flat_desc = "baja (picos marcados)" if flat < 0.5 else "alta distribución uniforme"
        print(f"## Archivo: {fname}\n")
        print("-- Métricas globales:")
        print(f" • Bins activos (≥ {MIN_BIN_PCT:.1f}%): {bins}")
        print(f" • Entropía de Shannon: {H:.2f} bits → {ent_desc}")
        print(f" • Planitud: {flat:.3f} → {flat_desc}\n")
        print("-- Ratios principales:")
        hist = np.array(info.get('ratio_hist', []))
        if hist.size:
            top_idx = np.argsort(hist)[-5:][::-1]
            for idx in top_idx:
                pct = hist[idx] * 100
                ratio = 2 ** (((idx + 0.5) / len(hist)) * math.log2(info.get('max_ratio', 6.0)))
                best_name, _, delta = None, None, None
                for name, r in SEMANTIC_RATIOS:
                    d = cents_diff(ratio, r)
                    if best_name is None or d < delta:
                        best_name, _, delta = name, r, d
                label = best_name if delta <= tol_cents else f"{ratio:.4f} (sugerido: {best_name}, Δ{delta:.1f}c)"
                print(f" • {label} | Energía: {pct:.1f}%")
        print("\n")

# --- Funciones topológicas ---

def shannon_entropy(hist: np.ndarray) -> float:
    p = hist[hist > 0]
    return -float(np.sum(p * np.log2(p))) if p.size else 0.0

def flatness_top(hist: np.ndarray) -> float:
    am = float(np.mean(hist))
    gm = float(np.exp(np.mean(np.log(hist + 1e-12))))
    return gm / am if am > 0 else 0.0

def spectral_centroid(hist: np.ndarray, min_r: float, max_r: float) -> float:
    bins = len(hist)
    ratios = min_r + (np.arange(bins) + 0.5)*(max_r - min_r)/bins
    return float(np.sum(ratios * hist)/(hist.sum()+1e-12))

def gini_coefficient(hist: np.ndarray) -> float:
    x = np.sort(hist)
    n = len(x)
    if x.sum() == 0: return 0.0
    idx = np.arange(1, n+1)
    cum = np.sum((2*idx - n - 1) * x)
    return float(1 - cum/(n*x.sum()+1e-12))

def first_autocorr_peak(hist: np.ndarray, prom: float=0.05) -> tuple:
    ac = correlate(hist, hist, mode='full')
    ac = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-12)
    peaks, props = find_peaks(ac, prominence=prom)
    if peaks.size:
        return int(peaks[0]), float(ac[peaks[0]])
    return None, 0.0

def analyze_topologico(info: dict, threshold: float, lin_key: str, max_ratio: float) -> dict:
    hist = np.array(info.get(lin_key, []), dtype=float)
    bins_active = int(np.sum(hist > threshold))
    H = shannon_entropy(hist)
    fl = flatness_top(hist)
    cent = spectral_centroid(hist, 1.0, max_ratio)
    gi = gini_coefficient(hist)
    lag, acv = first_autocorr_peak(hist)
    return {
        'bins_activos': bins_active,
        'entropia': round(H,3),
        'flatness': round(fl,3),
        'centroid': round(cent,4),
        'gini': round(gi,4),
        'autocorr_lag': lag,
        'autocorr_val': round(acv,3)
    }

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description='Auditor combinado v4.0')
    parser.add_argument('json', type=Path, help='JSON de entrada')
    parser.add_argument('--analisis', '-a', choices=['armonico','topologico','comparativo'],
                        default='armonico', help='Modo de análisis')
    parser.add_argument('--tolerance', '-t', type=float, default=TOL_CENT_DEFAULT,
                        help='Tolerancia en cents (armónico)')
    parser.add_argument('--markdown', action='store_true', help='Salida Markdown (armónico)')
    parser.add_argument('--threshold', '-T', type=float, default=0.0,
                        help='Umbral bins activos (topológico)')
    parser.add_argument('--lin-key', type=str, default='ratio_hist_lin',
                        help='Clave JSON para histograma lineal')
    parser.add_argument('--max-ratio', type=float, default=6.0,
                        help='Ratio máximo para centroides')
    args = parser.parse_args()
    data = json.loads(args.json.read_text(encoding='utf-8'))

    # Armónico
    if args.analisis in ['armonico','comparativo']:
        results = [(fname, *analyze_entry(info, args.tolerance)) for fname, info in data.items()]
        table = build_summary_table(results)
        fmt = 'github' if args.markdown else 'simple'
        print(tabulate(table, headers=["archivo","bins_activos","entropía_bits","planitud","ratio_dominante"], tablefmt=fmt))
        print()
        print_detailed(data, args.tolerance)

    # Topológico
    if args.analisis in ['topologico','comparativo']:
        top_res = []
        for fname, info in data.items():
            r = analyze_topologico(info, args.threshold, args.lin_key, args.max_ratio)
            top_res.append((fname, r))
        headers_top = ['archivo','bins_activos','entropia','flatness','centroid','gini','autocorr_lag','autocorr_val']
        table_top = []
        for fname, r in top_res:
            lag = r['autocorr_lag'] if r['autocorr_lag'] is not None else ''
            table_top.append([
                fname,
                r['bins_activos'],
                f"{r['entropia']:.3f}",
                f"{r['flatness']:.3f}",
                f"{r['centroid']:.4f}",
                f"{r['gini']:.4f}",
                lag,
                f"{r['autocorr_val']:.3f}"
            ])
        fmt_top = 'github' if args.markdown else 'simple'
        print(tabulate(table_top, headers=headers_top, tablefmt=fmt_top))

if __name__ == '__main__':
    main()

