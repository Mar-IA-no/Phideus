#!/usr/bin/env python3
"""
Analizador Phideus · v4.1 (histogramas enriquecidos)
====================================================

Cambios v4.1
------------
* Se agrega la función `compute_enriched_histogram` para generar
  un histograma de 3 canales (proporción, energía, entropía).
* `process_wav` ahora calcula `ratio_hist_enriched` (shape → (n_bins, 3)).
* Se expone el nuevo array en el JSON de salida manteniendo la
  compatibilidad con las claves anteriores.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import librosa
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

# ───────────────────────────────────
# 1. PARÁMETROS POR DEFECTO
# ───────────────────────────────────
DEFAULT_N_FFTS: Sequence[int] = (8192, 4096, 2048, 1024)
DEFAULT_HOP_LENGTH: int = 512
DEFAULT_PEAK_THRESHOLD_FACTOR: float = 1.25
DEFAULT_LOCAL_MEDIAN_WINDOW: int = 30
DEFAULT_CENT_TOL: float = 15  # cents
DEFAULT_MAX_BAND_HZ: float | None = None
DEFAULT_MIN_RATIO: float = 1.0
DEFAULT_MAX_RATIO: float = 6.0
DEFAULT_N_RATIO_BINS: int = 512  # ← 512 bins para resolución óptima (6.1 cents/bin)

# Rutas por defecto (pueden sobre‑escribirse por CLI)
INPUT_DIR = Path.cwd()
OUTPUT_JSON = Path("ratios_dataset_enriched.json")

# ───────────────────────────────────
# 2. FUNCIONES AUXILIARES
# ───────────────────────────────────

def parabolic_interpolation(arr: np.ndarray, idx: int) -> float:
    """Refinamiento sub‑bin de un máximo mediante interpolación parabólica."""
    if idx <= 0 or idx >= len(arr) - 1:
        return float(idx)
    a, b, c = arr[idx - 1], arr[idx], arr[idx + 1]
    return idx + 0.5 * (a - c) / (a - 2 * b + c)


def local_threshold(vec: np.ndarray, window: int, factor: float) -> np.ndarray:
    """Umbral local por ventana de tamaño *window* multiplicado por *factor*."""
    thr = np.empty_like(vec)
    for i in range(len(vec)):
        lo, hi = max(0, i - window), min(len(vec), i + window)
        thr[i] = np.median(vec[lo:hi]) * factor
    return thr


def compute_enriched_histogram(
    counts: np.ndarray, bin_edges: np.ndarray
) -> list[list[float]]:
    """Devuelve un histograma enriquecido (proporción, energía, entropía).

    * counts: array de conteos por bin (no normalizados)
    * bin_edges: edges de los bins (len = n_bins + 1) en escala log2
    """
    n_bins = len(counts)
    total = counts.sum() + 1e-12

    # Canal 0: proporción (PDF)
    prop = counts / total

    # Canal 1: energía (segundo momento en escala log2)
    log_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # centros en log2 space
    energy_raw = counts * (log_centers ** 2)
    energy = energy_raw / (energy_raw.sum() + 1e-12)

    # Canal 2: entropía local
    ent_raw = -prop * np.log(prop + 1e-12)
    ent = ent_raw / (ent_raw.sum() + 1e-12)

    enriched = np.stack([prop, energy, ent], axis=1)  # (n_bins, 3)
    return enriched.tolist()

# ───────────────────────────────────
# 3. PROCESADO DE UN WAV
# ───────────────────────────────────

def process_wav(
    path: Path,
    n_ffts: Sequence[int],
    hop_length: int,
    peak_thr_factor: float,
    local_window: int,
    cent_tol: float,
    max_band_hz: float | None,
    min_ratio: float,
    max_ratio: float,
    n_ratio_bins: int,
) -> Dict[str, Any]:
    y, sr = librosa.load(path, sr=None, mono=True)

    # 3.1 Espectro promedio multi‑escala
    mag_matrix, freq_ref = [], None
    for n in n_ffts:
        stft = librosa.stft(y, n_fft=n, hop_length=hop_length, center=False, window="hann")
        mag = np.abs(stft).mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n)
        if freq_ref is None:
            freq_ref = freqs
        mag_matrix.append(np.interp(freq_ref, freqs, mag))
    mag_per_bin = np.mean(mag_matrix, axis=0)

    # 3.2 Detección adaptativa de picos
    thr = local_threshold(mag_per_bin, local_window, peak_thr_factor)
    raw_peaks, _ = find_peaks(mag_per_bin, height=thr)
    if max_band_hz is not None:
        raw_peaks = raw_peaks[freq_ref[raw_peaks] <= max_band_hz]

    # 3.3 Refinamiento y deduplicado
    cand = sorted(((parabolic_interpolation(mag_per_bin, p), mag_per_bin[p]) for p in raw_peaks), key=lambda t: -t[1])
    bins_sel, amps_sel = [], []
    log_tol = cent_tol / 1200.0
    for b, a in cand:
        if all(abs(math.log2(b / prev)) > log_tol for prev in bins_sel):
            bins_sel.append(b)
            amps_sel.append(a)
    if not bins_sel:
        empty = [0.0] * n_ratio_bins
        empty_3ch = [[0.0, 0.0, 0.0] for _ in range(n_ratio_bins)]
        return {
            "sr": int(sr),
            "peak_freqs": [],
            "ratios_log": [],
            "ratios_lin": [],
            "ratio_hist_log": empty,
            "ratio_hist_lin": empty,
            "ratio_hist": empty,  # compat
            "ratio_hist_enriched": empty_3ch,
        }

    # 3.4 Conversión a Hz
    peak_freqs = np.interp(bins_sel, np.arange(len(freq_ref)), freq_ref)
    ord_idx = np.argsort(peak_freqs)
    peak_freqs = peak_freqs[ord_idx]
    amps_sel = np.array(amps_sel)[ord_idx]

    # 3.5 Ratios y pesos
    ratios_log, ratios_lin, weights = [], [], []
    for i, (fi, ai) in enumerate(zip(peak_freqs, amps_sel)):
        for fj, aj in zip(peak_freqs[i + 1 :], amps_sel[i + 1 :]):
            r = fj / fi
            if r < min_ratio or r > max_ratio:
                continue
            ratios_lin.append(r)
            ratios_log.append(math.log2(r))
            weights.append(math.sqrt(ai * aj))

    # 3.6 Histogramas (log y lin)
    if ratios_log:
        hist_log, edges_log = np.histogram(
            ratios_log,
            bins=n_ratio_bins,
            range=(0, math.log2(max_ratio)),
            weights=weights,
        )
        ratio_hist_log = (hist_log / (hist_log.sum() + 1e-12)).tolist()
    else:
        hist_log = np.zeros(n_ratio_bins)
        edges_log = np.linspace(0, math.log2(max_ratio), n_ratio_bins + 1)
        ratio_hist_log = hist_log.tolist()

    if ratios_lin:
        hist_lin, _ = np.histogram(
            ratios_lin,
            bins=n_ratio_bins,
            range=(min_ratio, max_ratio),
            weights=weights,
        )
        ratio_hist_lin = (hist_lin / (hist_lin.sum() + 1e-12)).tolist()
    else:
        hist_lin = np.zeros(n_ratio_bins)
        ratio_hist_lin = hist_lin.tolist()

    # 3.7 Histograma enriquecido (usa hist_log)
    ratio_hist_enriched = compute_enriched_histogram(hist_log.astype(float), edges_log)

    # 3.8 Salida
    return {
        "sr": int(sr),
        "peak_freqs": peak_freqs.tolist(),
        "ratios_log": ratios_log,
        "ratios_lin": ratios_lin,
        "ratio_hist_log": ratio_hist_log,
        "ratio_hist_lin": ratio_hist_lin,
        "ratio_hist": ratio_hist_log,  # compatibilidad
        "ratio_hist_enriched": ratio_hist_enriched,
    }

# ───────────────────────────────────
# 4. CLI & MAIN
# ───────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera ratios_dataset_enriched.json a partir de WAVs.")
    p.add_argument("--input-dir", "-i", type=Path, default=INPUT_DIR, help="Carpeta con WAVs (recursiva)")
    p.add_argument("--output", "-o", type=Path, default=OUTPUT_JSON, help="JSON de salida")
    p.add_argument("--n-ffts", nargs="*", type=int, default=list(DEFAULT_N_FFTS))
    p.add_argument("--hop", type=int, default=DEFAULT_HOP_LENGTH)
    p.add_argument("--thr", type=float, default=DEFAULT_PEAK_THRESHOLD_FACTOR)
    p.add_argument("--median-window", type=int, default=DEFAULT_LOCAL_MEDIAN_WINDOW)
    p.add_argument("--cent-tol", type=float, default=DEFAULT_CENT_TOL)
    p.add_argument("--max-band-hz", type=float, default=DEFAULT_MAX_BAND_HZ)
    p.add_argument("--min-ratio", type=float, default=DEFAULT_MIN_RATIO)
    p.add_argument("--max-ratio", type=float, default=DEFAULT_MAX_RATIO)
    p.add_argument("--bins", type=int, default=DEFAULT_N_RATIO_BINS)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    wav_paths = [p for p in args.input_dir.rglob("*") if p.suffix.lower() == ".wav"]
    if not wav_paths:
        raise SystemExit(f"[ERROR] No se encontraron archivos .wav en → {args.input_dir}")

    print(f"Procesando {len(wav_paths)} WAVs…")
    dataset: Dict[str, Any] = {}
    for wav in tqdm(wav_paths, unit="wav"):
        rel = wav.relative_to(args.input_dir).as_posix()
        dataset[rel] = process_wav(
            wav,
            n_ffts=args.n_ffts,
            hop_length=args.hop,
            peak_thr_factor=args.thr,
            local_window=args.median_window,
            cent_tol=args.cent_tol,
            max_band_hz=args.max_band_hz,
            min_ratio=args.min_ratio,
            max_ratio=args.max_ratio,
            n_ratio_bins=args.bins,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"\n✔ JSON generado en {args.output} | {len(dataset)} archivos procesados")


if __name__ == "__main__":
    main()

