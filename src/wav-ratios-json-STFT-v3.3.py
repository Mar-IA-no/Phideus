#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# wav_ratios_json_STFT.py · v3.3  (bug‑fix IndentationError + cierre JSON)
#
# Cambios v3.3
#   • Cuerpo del bucle for en main() completado y bien indentado.
#   • Escritura del JSON y mensaje final añadidos.
#   • Pequeña función helper para procesar un WAV.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Sequence

import librosa
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

# ╭──────────────────────────╮
# │ CONFIGURACIÓN  RÁPIDA    │
# ╰──────────────────────────╯
INPUT_DIR   = Path("/Users/mandinga/Python/Ratios/Phideus/wavs/wavs_sinteticos_2.0")
OUTPUT_JSON = Path(__file__).parent / "ratios_dataset.json"

# ╭──────────────────────────╮
# │ 1. PARÁMETROS POR DEFECTO │
# ╰──────────────────────────╯
DEFAULT_N_FFTS: Sequence[int] = (8192, 4096, 2048, 1024)
DEFAULT_HOP_LENGTH: int = 512
DEFAULT_PEAK_THRESHOLD_FACTOR: float = 1.25
DEFAULT_LOCAL_MEDIAN_WINDOW: int = 30
DEFAULT_CENT_TOL: float = 15
DEFAULT_MAX_BAND_HZ: float | None = None
DEFAULT_MIN_RATIO: float = 1.00
DEFAULT_MAX_RATIO: float = 6.0
DEFAULT_N_RATIO_BINS: int = 200

# ╭──────────────────────────╮
# │ 2. FUNCIONES AUXILIARES  │
# ╰──────────────────────────╯

def parabolic_interpolation(arr: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(arr) - 1:
        return float(idx)
    a, b, c = arr[idx - 1], arr[idx], arr[idx + 1]
    return idx + 0.5 * (a - c) / (a - 2 * b + c)


def local_threshold(vec: np.ndarray, window: int, factor: float) -> np.ndarray:
    thr = np.empty_like(vec)
    for i in range(len(vec)):
        lo, hi = max(0, i - window), min(len(vec), i + window)
        thr[i] = np.median(vec[lo:hi]) * factor
    return thr

# ╭──────────────────────────╮
# │ 3. PROCESADO DE UN WAV   │
# ╰──────────────────────────╯

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

    mag_matrix = []
    freq_ref = None
    for n in n_ffts:
        stft = librosa.stft(y, n_fft=n, hop_length=hop_length, center=False, window="hann")
        mag = np.abs(stft).mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n)
        if freq_ref is None:
            freq_ref = freqs
        mag_matrix.append(np.interp(freq_ref, freqs, mag))
    mag_per_bin = np.mean(mag_matrix, axis=0)

    thr = local_threshold(mag_per_bin, local_window, peak_thr_factor)
    raw_peaks, _ = find_peaks(mag_per_bin, height=thr)

    if max_band_hz is not None:
        raw_peaks = raw_peaks[freq_ref[raw_peaks] <= max_band_hz]

    cand = sorted(((parabolic_interpolation(mag_per_bin, p), mag_per_bin[p]) for p in raw_peaks),
                  key=lambda t: -t[1])
    bins_sel, amps_sel = [], []
    log_tol = cent_tol / 1200.0
    for b, a in cand:
        if all(abs(np.log2(b / prev)) > log_tol for prev in bins_sel):
            bins_sel.append(b)
            amps_sel.append(a)
    if not bins_sel:
        return {"sr": int(sr), "peak_freqs": [], "ratios_log": [], "ratios_lin": [], "ratio_hist": [0.0] * n_ratio_bins}

    peak_freqs = np.interp(bins_sel, np.arange(len(freq_ref)), freq_ref)

    ord_idx = np.argsort(peak_freqs)
    peak_freqs = peak_freqs[ord_idx]
    amps_sel = np.array(amps_sel)[ord_idx]

    ratios_log, ratios_lin, weights = [], [], []
    for i, (fi, ai) in enumerate(zip(peak_freqs, amps_sel)):
        for fj, aj in zip(peak_freqs[i+1:], amps_sel[i+1:]):
            r = fj / fi
            if r < min_ratio or r > max_ratio:
                continue
            ratios_lin.append(r)
            ratios_log.append(np.log2(r))
            weights.append(np.sqrt(ai * aj))

    if ratios_log:
        hist, _ = np.histogram(ratios_log, bins=n_ratio_bins, range=(0, np.log2(max_ratio)), weights=weights)
        ratio_hist = (hist / (hist.sum() + 1e-12)).tolist()
    else:
        ratio_hist = [0.0] * n_ratio_bins

    return {
        "sr": int(sr),
        "peak_freqs": peak_freqs.tolist(),
        "ratios_log": ratios_log,
        "ratios_lin": ratios_lin,
        "ratio_hist": ratio_hist,
    }

# ╭──────────────────────────╮
# │ 4. CLI & MAIN            │
# ╰──────────────────────────╯

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera ratios_dataset.json a partir de WAVs.")
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

    print(f"Procesando {len(wav_paths)} WAVs de '{args.input_dir}'…")
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

    print(f"\n✔ JSON generado: {args.output} | {len(dataset)} archivos procesados")


if __name__ == "__main__":
    main()

