#!/usr/bin/env python3
"""
Analizador Phideus · v5.0 (histogramas enriquecidos con dimensión temporal)
===========================================================================

Diferencias clave vs v4.1:
--------------------------
- v4.1 generaba un único histograma enriquecido GLOBAL por archivo WAV
  (espectro promedio en el tiempo).
- v5.0 genera un histograma enriquecido POR VENTANA (frame de STFT),
  produciendo así una SECUENCIA temporal de histogramas.

Diferencias conceptuales importantes:
-------------------------------------
- Se abandonan las escalas log2 y los "cents" (relacionados a percepción auditiva).
- Todas las proporciones se representan como RATIOS LINEALES f_j / f_i (adimensionales).
- La deduplicación de picos se hace por tolerancia relativa en frecuencia (ej. 1%),
  no por distancia en cents.

Uso típico:
-----------
python analizador_5_0_temporal.py \
    --input-dir data/wavs \
    --output ratios_temporal_enriched.json \
    --n-fft 2048 \
    --hop 512 \
    --min-ratio 1.0 \
    --max-ratio 6.0 \
    --bins 256

El JSON resultante tendrá, para cada WAV:
- "sr": sample rate
- "n_fft": tamaño de ventana
- "hop_length": hop entre frames
- "frame_times": lista de tiempos (segundos) por frame
- "n_frames": cantidad de frames
- "n_ratio_bins": cantidad de bins de ratio
- "ratio_hist_frames": lista de frames, cada uno (n_bins,) normalizado
- "ratio_hist_enriched_frames": lista de frames, cada uno (n_bins, 3)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import librosa
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

# ───────────────────────────────────
# 1. PARÁMETROS POR DEFECTO
# ───────────────────────────────────

DEFAULT_N_FFT: int = 2048
DEFAULT_HOP_LENGTH: int = 512          # ~75% overlap
DEFAULT_PEAK_THRESHOLD_FACTOR: float = 1.25
DEFAULT_LOCAL_MEDIAN_WINDOW: int = 30  # en bins de frecuencia
DEFAULT_REL_PEAK_TOL: float = 0.01     # 1% de diferencia relativa en frecuencia
DEFAULT_MAX_BAND_HZ: float | None = None
DEFAULT_MIN_RATIO: float = 1.0
DEFAULT_MAX_RATIO: float = 6.0
DEFAULT_N_RATIO_BINS: int = 256

INPUT_DIR = "."
OUTPUT_JSON = "ratios_temporal_enriched.json"


# ───────────────────────────────────
# 2. FUNCIONES AUXILIARES
# ───────────────────────────────────

def parabolic_interpolation(arr: np.ndarray, idx: int) -> float:
    """
    Refinamiento sub-bin de un máximo mediante interpolación parabólica.

    arr : vector de magnitudes (eje frecuencia)
    idx : índice entero del bin de pico
    Devuelve índice en coma flotante (sub-bin).
    """
    if idx <= 0 or idx >= len(arr) - 1:
        return float(idx)
    a, b, c = arr[idx - 1], arr[idx], arr[idx + 1]
    denom = (a - 2 * b + c)
    if abs(denom) < 1e-12:
        return float(idx)
    return idx + 0.5 * (a - c) / denom


def local_threshold(vec: np.ndarray, window: int, factor: float) -> np.ndarray:
    """
    Umbral local por ventana de tamaño *window* multiplicado por *factor*.

    Para cada índice i, calcula la mediana de vec[i-window:i+window]
    y la multiplica por factor.
    """
    thr = np.empty_like(vec)
    n = len(vec)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window)
        thr[i] = np.median(vec[lo:hi]) * factor
    return thr


def compute_enriched_histogram(
    counts: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """
    Devuelve un histograma enriquecido (proporción, "momento", entropía).

    counts    : array de pesos por bin (sumatoria de pesos de pares de picos).
    bin_edges : edges de los bins (len = n_bins + 1) en espacio de ratio LINEAL.

    Canales:
      - Canal 0: proporción (PDF normalizada)
      - Canal 1: "momento" sobre el ratio lineal (segundo momento sobre centers)
      - Canal 2: entropía normalizada de la distribución de proporciones
    """
    n_bins = len(counts)
    if n_bins + 1 != len(bin_edges):
        raise ValueError("bin_edges debe tener len(counts) + 1")

    total = counts.sum()
    if total <= 0:
        # Histograma vacío: devolvemos todo cero
        return np.zeros((n_bins, 3), dtype=float)

    # Canal 0: proporción (PDF)
    prop = counts / (total + 1e-12)

    # Canal 1: momento sobre ratio lineal
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # centros en r lineal
    moment_raw = counts * (centers ** 2)   # segundo momento (podés bajar a **1 si querés)
    msum = moment_raw.sum()
    if msum > 0:
        moment = moment_raw / (msum + 1e-12)
    else:
        moment = np.zeros_like(moment_raw)

    # Canal 2: entropía local normalizada
    ent_raw = -prop * np.log(prop + 1e-12)
    esum = ent_raw.sum()
    if esum > 0:
        ent = ent_raw / (esum + 1e-12)
    else:
        ent = np.zeros_like(ent_raw)

    enriched = np.stack([prop, moment, ent], axis=1)  # (n_bins, 3)
    return enriched


# ───────────────────────────────────
# 3. PROCESADO DE UN WAV (TEMPORAL)
# ───────────────────────────────────

def process_wav_temporal(
    path: Path,
    n_fft: int,
    hop_length: int,
    peak_thr_factor: float,
    local_window: int,
    rel_peak_tol: float,
    max_band_hz: float | None,
    min_ratio: float,
    max_ratio: float,
    n_ratio_bins: int,
) -> Dict[str, Any]:
    """
    Procesa un WAV (audio o vibración) y devuelve secuencias de histogramas enriquecidos por frame.

    - Trabaja con STFT 1D (ventana Hann).
    - Para cada frame:
        · detecta picos en el espectro,
        · calcula ratios lineales r = f_j / f_i,
        · construye un histograma de ratios,
        · lo enriquece con 3 canales.
    """
    # Carga del WAV (audio o vibración de acelerómetro)
    y, sr = librosa.load(path, sr=None, mono=True)

    # STFT 1D
    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        center=False,
        window="hann",
    )  # shape: (n_freq_bins, n_frames)
    mag = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_freq_bins, n_frames = mag.shape

    # Precomputo de edges lineales para los histogramas de ratios
    edges_ratio = np.linspace(min_ratio, max_ratio, n_ratio_bins + 1, dtype=float)

    # Secuencias resultado
    hist_ratio_frames: list[np.ndarray] = []
    enriched_frames: list[np.ndarray] = []

    # Procesado frame por frame
    for frame_idx in range(n_frames):
        mag_frame = mag[:, frame_idx]  # (n_freq_bins,)

        # 3.1 Umbral local y picos
        thr = local_threshold(mag_frame, local_window, peak_thr_factor)
        raw_peaks, _ = find_peaks(mag_frame, height=thr)

        if max_band_hz is not None and raw_peaks.size > 0:
            raw_peaks = raw_peaks[freqs[raw_peaks] <= max_band_hz]

        if raw_peaks.size == 0:
            # Frame sin picos significativos → histograma vacío
            hist_ratio = np.zeros(n_ratio_bins, dtype=float)
            enriched = np.zeros((n_ratio_bins, 3), dtype=float)
        else:
            # 3.2 Refinamiento parabólico de índices de pico
            cand_bins = []
            cand_amps = []
            for p in raw_peaks:
                p_int = int(p)
                b = parabolic_interpolation(mag_frame, p_int)
                cand_bins.append(b)
                cand_amps.append(mag_frame[p_int])

            cand_bins = np.array(cand_bins, dtype=float)
            cand_amps = np.array(cand_amps, dtype=float)

            # 3.3 Conversión de índices a frecuencias físicas (Hz)
            # f_k ≈ sr * k / n_fft ; usamos interpolación sobre "freqs" por claridad.
            peak_freqs_raw = np.interp(
                cand_bins,
                np.arange(n_freq_bins, dtype=float),
                freqs,
            )

            # 3.4 Ordenamos por amplitud descendente
            order = np.argsort(-cand_amps)
            peak_freqs_raw = peak_freqs_raw[order]
            cand_amps = cand_amps[order]

            # 3.5 Deduplicado por tolerancia RELATIVA en frecuencia (no cents)
            freqs_sel: list[float] = []
            amps_sel: list[float] = []
            for f, a in zip(peak_freqs_raw, cand_amps):
                if not freqs_sel:
                    freqs_sel.append(float(f))
                    amps_sel.append(float(a))
                    continue
                if all(abs(f - prev_f) / (prev_f + 1e-12) > rel_peak_tol for prev_f in freqs_sel):
                    freqs_sel.append(float(f))
                    amps_sel.append(float(a))

            if not freqs_sel:
                hist_ratio = np.zeros(n_ratio_bins, dtype=float)
                enriched = np.zeros((n_ratio_bins, 3), dtype=float)
            else:
                freqs_sel_arr = np.array(freqs_sel, dtype=float)
                amps_sel_arr = np.array(amps_sel, dtype=float)

                # 3.6 Ratios lineales y pesos
                ratios: list[float] = []
                weights: list[float] = []

                n_peaks = len(freqs_sel_arr)
                for i in range(n_peaks):
                    fi = freqs_sel_arr[i]
                    ai = amps_sel_arr[i]
                    for j in range(i + 1, n_peaks):
                        fj = freqs_sel_arr[j]
                        aj = amps_sel_arr[j]
                        if fi <= 0 or fj <= 0:
                            continue
                        r = fj / fi
                        if r < min_ratio or r > max_ratio:
                            continue
                        ratios.append(float(r))
                        weights.append(float(math.sqrt(ai * aj)))

                if ratios:
                    # 3.7 Histograma de ratios lineales
                    ratios_arr = np.array(ratios, dtype=float)
                    weights_arr = np.array(weights, dtype=float)
                    hist_ratio, _ = np.histogram(
                        ratios_arr,
                        bins=edges_ratio,
                        weights=weights_arr,
                    )
                else:
                    hist_ratio = np.zeros(n_ratio_bins, dtype=float)

                # 3.8 Histograma enriquecido (3 canales) a partir de hist_ratio
                enriched = compute_enriched_histogram(hist_ratio.astype(float), edges_ratio)

        hist_ratio_frames.append(hist_ratio)
        enriched_frames.append(enriched)

    # 3.9 Tiempos por frame (en segundos)
    frame_indices = np.arange(n_frames)
    frame_times = librosa.frames_to_time(
        frame_indices,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
    ).tolist()

    # Normalizamos hist_ratio por frame para "ratio_hist_frames"
    ratio_hist_frames = []
    for h in hist_ratio_frames:
        s = h.sum()
        if s > 0:
            ratio_hist_frames.append((h / (s + 1e-12)).tolist())
        else:
            ratio_hist_frames.append(h.tolist())

    ratio_hist_enriched_frames = [ef.tolist() for ef in enriched_frames]

    return {
        "sr": int(sr),
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "frame_times": frame_times,
        "n_frames": int(len(frame_times)),
        "n_ratio_bins": int(n_ratio_bins),
        "ratio_hist_frames": ratio_hist_frames,                   # [T, B]
        "ratio_hist_enriched_frames": ratio_hist_enriched_frames  # [T, B, 3]
    }


# ───────────────────────────────────
# 4. CLI & MAIN
# ───────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analizador Phideus v5.0 – Histogramas enriquecidos temporales a partir de WAVs."
    )
    p.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=INPUT_DIR,
        help="Carpeta con WAVs (recursiva). Default: directorio actual",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_JSON,
        help="JSON de salida. Default: ratios_temporal_enriched.json",
    )
    p.add_argument(
        "--n-fft",
        type=int,
        default=DEFAULT_N_FFT,
        help="Tamaño de ventana FFT (ej. 2048)",
    )
    p.add_argument(
        "--hop",
        type=int,
        default=DEFAULT_HOP_LENGTH,
        help="Hop entre frames (ej. 512 para ~75% overlap)",
    )
    p.add_argument(
        "--thr",
        type=float,
        default=DEFAULT_PEAK_THRESHOLD_FACTOR,
        help="Factor de umbral local para detección de picos",
    )
    p.add_argument(
        "--median-window",
        type=int,
        default=DEFAULT_LOCAL_MEDIAN_WINDOW,
        help="Tamaño de ventana (en bins de frecuencia) para la mediana local",
    )
    p.add_argument(
        "--rel-peak-tol",
        type=float,
        default=DEFAULT_REL_PEAK_TOL,
        help="Tolerancia relativa para deduplicar picos (ej. 0.01 = 1%)",
    )
    p.add_argument(
        "--max-band-hz",
        type=float,
        default=DEFAULT_MAX_BAND_HZ,
        help="Frecuencia máxima considerada (Hz), o None",
    )
    p.add_argument(
        "--min-ratio",
        type=float,
        default=DEFAULT_MIN_RATIO,
        help="Ratio mínimo considerado",
    )
    p.add_argument(
        "--max-ratio",
        type=float,
        default=DEFAULT_MAX_RATIO,
        help="Ratio máximo considerado",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_N_RATIO_BINS,
        help="Número de bins de ratio",
    )
    p.add_argument(
        "--format",
        type=str,
        choices=["json", "npz"],
        default="npz",
        help="Formato de salida: 'json' (legible, grande) o 'npz' (binario, compacto). Default: npz",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Número de workers paralelos. 0=auto (núcleos-2), 1=secuencial. Default: 0",
    )
    return p.parse_args()


def _process_single_wav(task: Tuple) -> Tuple[str, Dict[str, Any]]:
    """Worker function para multiprocessing."""
    wav_path, input_dir, params = task
    rel = wav_path.relative_to(input_dir).as_posix()
    result = process_wav_temporal(wav_path, **params)
    return (rel, result)


def main() -> None:
    import multiprocessing as mp
    from functools import partial

    args = parse_args()

    wav_paths = [p for p in args.input_dir.rglob("*") if p.suffix.lower() == ".wav"]
    if not wav_paths:
        raise SystemExit(f"[ERROR] No se encontraron archivos .wav en → {args.input_dir}")

    # Determinar número de workers
    if args.workers == 0:
        n_workers = max(1, mp.cpu_count() - 2)
    else:
        n_workers = args.workers

    params = {
        'n_fft': args.n_fft,
        'hop_length': args.hop,
        'peak_thr_factor': args.thr,
        'local_window': args.median_window,
        'rel_peak_tol': args.rel_peak_tol,
        'max_band_hz': args.max_band_hz,
        'min_ratio': args.min_ratio,
        'max_ratio': args.max_ratio,
        'n_ratio_bins': args.bins,
    }

    tasks = [(wav, args.input_dir, params) for wav in wav_paths]

    print(f"Procesando {len(wav_paths)} WAVs (analizador temporal 5.0)")
    print(f"Workers: {n_workers} | Formato: {args.format}")

    dataset: Dict[str, Any] = {}

    if n_workers == 1:
        # Modo secuencial
        for wav in tqdm(wav_paths, unit="wav"):
            rel = wav.relative_to(args.input_dir).as_posix()
            dataset[rel] = process_wav_temporal(wav, **params)
    else:
        # Modo paralelo
        with mp.Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_process_single_wav, tasks),
                total=len(tasks),
                unit="wav"
            ))
        dataset = dict(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "npz":
        # Formato binario compacto (recomendado)
        save_dataset_binary(dataset, args.output)
    else:
        # Formato JSON (legible pero grande)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        print(f"\n✔ JSON generado en {args.output} | {len(dataset)} archivos procesados")


def save_dataset_binary(dataset: Dict[str, Any], output_path: Path) -> None:
    """
    Guarda el dataset en formato numpy comprimido (.npz).

    Estructura del archivo:
    - 'filenames': array de nombres de archivo
    - 'metadata': dict con sr, n_fft, hop_length, n_frames por archivo
    - 'frames_<idx>': array float32 [T, B, 3] para cada archivo
    - 'frame_times_<idx>': array float32 [T] con tiempos de cada frame

    Ventajas vs JSON:
    - 12x más compacto
    - 10-50x más rápido de cargar
    - Directamente compatible con PyTorch/NumPy
    """
    filenames = list(dataset.keys())
    metadata = {}
    arrays = {}

    for idx, (filename, entry) in enumerate(dataset.items()):
        # Metadata
        metadata[filename] = {
            'idx': idx,
            'sr': entry['sr'],
            'n_fft': entry['n_fft'],
            'hop_length': entry['hop_length'],
            'n_frames': entry['n_frames'],
            'n_ratio_bins': entry['n_ratio_bins'],
        }

        # Arrays de datos
        arrays[f'frames_{idx}'] = np.array(
            entry['ratio_hist_enriched_frames'],
            dtype=np.float32
        )
        arrays[f'frame_times_{idx}'] = np.array(
            entry['frame_times'],
            dtype=np.float32
        )

    # Asegurar extensión .npz
    output_path = Path(output_path)
    if output_path.suffix != '.npz':
        output_path = output_path.with_suffix('.npz')

    # Guardar
    np.savez_compressed(
        output_path,
        filenames=np.array(filenames, dtype=object),
        metadata=metadata,
        n_files=len(filenames),
        **arrays
    )

    # Estadísticas
    total_frames = sum(m['n_frames'] for m in metadata.values())
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n✔ Dataset binario generado: {output_path}")
    print(f"  Archivos: {len(filenames)} | Frames totales: {total_frames:,} | Tamaño: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
