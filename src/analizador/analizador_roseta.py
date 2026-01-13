#!/usr/bin/env python3
"""
Analizador Phideus · Roseta (dual-domain: Audio + Vibración)
============================================================

Diseñado específicamente para el Experimento Roseta con dataset UOEMD.

Diferencias vs Analizador 5.0:
------------------------------
- Lee archivos CSV de UOEMD en lugar de WAV
- Procesa DOS canales simultáneamente:
  - Audio (Micrófono, columna 2)
  - Vibración (Acelerómetro 1, columna 1)
- Genera histogramas enriquecidos PAREADOS por frame temporal
- Incluye metadatos de condición (Healthy/Fault) y tipo de falla

Dataset UOEMD:
--------------
- Universidad de Ottawa Electric Motor Dataset
- 5 columnas: Accel1, Micrófono, Accel2, Accel3, Temperatura
- 42 kHz, 420,000 muestras por archivo (10 segundos)
- Múltiples condiciones de falla

Uso típico:
-----------
python analizador_roseta.py \
    --input-dir data/datasets/UOEMD/raw/2_CSV_Data_Files \
    --output data/datasets/roseta_dual.npz \
    --n-fft 4096 \
    --hop 1024 \
    --workers 14

Salida:
-------
NPZ con estructura:
- filenames: lista de archivos
- conditions: lista de condiciones (HH, RU, FB, etc.)
- audio_frames_<idx>: histogramas temporales del audio [T, B, 3]
- vibration_frames_<idx>: histogramas temporales de vibración [T, B, 3]
- frame_times_<idx>: timestamps de cada frame
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

# ───────────────────────────────────
# 1. PARÁMETROS POR DEFECTO (UOEMD-optimized)
# ───────────────────────────────────

DEFAULT_SAMPLE_RATE: int = 42000          # UOEMD fijo
DEFAULT_N_FFT: int = 4096                  # Δf ≈ 10.25 Hz (resuelve armónicos de 60Hz)
DEFAULT_HOP_LENGTH: int = 1024             # 75% overlap, ~40 frames/segundo
DEFAULT_PEAK_THRESHOLD_FACTOR: float = 1.25
DEFAULT_LOCAL_MEDIAN_WINDOW: int = 30
DEFAULT_REL_PEAK_TOL: float = 0.01
DEFAULT_MAX_BAND_HZ: float | None = None
DEFAULT_MIN_RATIO: float = 1.0
DEFAULT_MAX_RATIO: float = 6.0
DEFAULT_N_RATIO_BINS: int = 256

# Columnas del CSV de UOEMD (0-indexed)
UOEMD_COL_ACCEL1 = 0      # Vibración principal
UOEMD_COL_MICROPHONE = 1  # Audio
UOEMD_COL_ACCEL2 = 2      # Vibración Y (backup)
UOEMD_COL_ACCEL3 = 3      # Vibración Z (backup)
UOEMD_COL_TEMPERATURE = 4  # Temperatura (no usado)


# ───────────────────────────────────
# 2. FUNCIONES AUXILIARES
# ───────────────────────────────────

def parabolic_interpolation(arr: np.ndarray, idx: int) -> float:
    """Refinamiento sub-bin de un máximo mediante interpolación parabólica."""
    if idx <= 0 or idx >= len(arr) - 1:
        return float(idx)
    a, b, c = arr[idx - 1], arr[idx], arr[idx + 1]
    denom = (a - 2 * b + c)
    if abs(denom) < 1e-12:
        return float(idx)
    return idx + 0.5 * (a - c) / denom


def local_threshold(vec: np.ndarray, window: int, factor: float) -> np.ndarray:
    """Umbral local por ventana móvil de mediana."""
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
    Histograma enriquecido de 3 canales: proporción, momento, entropía.
    """
    n_bins = len(counts)
    if n_bins + 1 != len(bin_edges):
        raise ValueError("bin_edges debe tener len(counts) + 1")

    total = counts.sum()
    if total <= 0:
        return np.zeros((n_bins, 3), dtype=float)

    # Canal 0: proporción (PDF)
    prop = counts / (total + 1e-12)

    # Canal 1: momento sobre ratio lineal
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    moment_raw = counts * (centers ** 2)
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

    return np.stack([prop, moment, ent], axis=1)


def parse_uoemd_filename(filename: str) -> Dict[str, Any]:
    """
    Parsea nombre de archivo UOEMD para extraer condición.

    Ejemplos:
        H_H_4_0.csv → condition='HH', fault_type='H', specific='H', speed=4, load=0
        R_U_3_1.csv → condition='RU', fault_type='R', specific='U', speed=3, load=1
    """
    # Patrón: [FaultType]_[Specific]_[Speed]_[Load].csv
    match = re.match(r'([A-Z])_([A-Z])_(\d+)_(\d+)\.csv', filename, re.IGNORECASE)
    if match:
        fault_type = match.group(1).upper()
        specific = match.group(2).upper()
        speed = int(match.group(3))
        load = int(match.group(4))
        return {
            'condition': f"{fault_type}{specific}",
            'fault_type': fault_type,
            'specific': specific,
            'speed': speed,
            'load': load,
            'is_healthy': fault_type == 'H' and specific == 'H',
        }
    return {
        'condition': 'UNKNOWN',
        'fault_type': '?',
        'specific': '?',
        'speed': 0,
        'load': 0,
        'is_healthy': False,
    }


def stft_from_signal(
    signal: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    STFT manual (sin dependencia de librosa para señales CSV).

    Returns:
        magnitudes: array [n_freq_bins, n_frames]
        freqs: array [n_freq_bins] con frecuencias en Hz
    """
    # Ventana Hann
    window = np.hanning(n_fft)

    # Número de frames
    n_samples = len(signal)
    n_frames = max(0, (n_samples - n_fft) // hop_length + 1)

    if n_frames == 0:
        return np.array([]).reshape(n_fft // 2 + 1, 0), np.fft.rfftfreq(n_fft, 1/sr)

    # Calcular STFT
    n_freq_bins = n_fft // 2 + 1
    magnitudes = np.zeros((n_freq_bins, n_frames), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        magnitudes[:, i] = np.abs(spectrum)

    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    return magnitudes, freqs


def process_single_channel(
    signal: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    peak_thr_factor: float,
    local_window: int,
    rel_peak_tol: float,
    max_band_hz: float | None,
    min_ratio: float,
    max_ratio: float,
    n_ratio_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Procesa un canal (audio o vibración) y retorna histogramas temporales.

    Returns:
        enriched_frames: array [T, B, 3] con histogramas enriquecidos
        frame_times: array [T] con timestamps
    """
    # STFT
    mag, freqs = stft_from_signal(signal, sr, n_fft, hop_length)
    n_freq_bins, n_frames = mag.shape

    # Edges para histogramas
    edges_ratio = np.linspace(min_ratio, max_ratio, n_ratio_bins + 1, dtype=float)

    enriched_frames = []

    for frame_idx in range(n_frames):
        mag_frame = mag[:, frame_idx]

        # Umbral local y picos
        thr = local_threshold(mag_frame, local_window, peak_thr_factor)
        raw_peaks, _ = find_peaks(mag_frame, height=thr)

        if max_band_hz is not None and raw_peaks.size > 0:
            raw_peaks = raw_peaks[freqs[raw_peaks] <= max_band_hz]

        if raw_peaks.size == 0:
            enriched = np.zeros((n_ratio_bins, 3), dtype=float)
        else:
            # Refinamiento parabólico
            cand_bins = []
            cand_amps = []
            for p in raw_peaks:
                p_int = int(p)
                b = parabolic_interpolation(mag_frame, p_int)
                cand_bins.append(b)
                cand_amps.append(mag_frame[p_int])

            cand_bins = np.array(cand_bins, dtype=float)
            cand_amps = np.array(cand_amps, dtype=float)

            # Frecuencias físicas
            peak_freqs_raw = np.interp(
                cand_bins,
                np.arange(n_freq_bins, dtype=float),
                freqs,
            )

            # Ordenar por amplitud
            order = np.argsort(-cand_amps)
            peak_freqs_raw = peak_freqs_raw[order]
            cand_amps = cand_amps[order]

            # Deduplicado por tolerancia relativa
            freqs_sel = []
            amps_sel = []
            for f, a in zip(peak_freqs_raw, cand_amps):
                if not freqs_sel:
                    freqs_sel.append(float(f))
                    amps_sel.append(float(a))
                    continue
                if all(abs(f - prev_f) / (prev_f + 1e-12) > rel_peak_tol for prev_f in freqs_sel):
                    freqs_sel.append(float(f))
                    amps_sel.append(float(a))

            if not freqs_sel:
                enriched = np.zeros((n_ratio_bins, 3), dtype=float)
            else:
                freqs_sel_arr = np.array(freqs_sel, dtype=float)
                amps_sel_arr = np.array(amps_sel, dtype=float)

                # Ratios lineales
                ratios = []
                weights = []
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
                    ratios_arr = np.array(ratios, dtype=float)
                    weights_arr = np.array(weights, dtype=float)
                    hist_ratio, _ = np.histogram(
                        ratios_arr,
                        bins=edges_ratio,
                        weights=weights_arr,
                    )
                    enriched = compute_enriched_histogram(hist_ratio.astype(float), edges_ratio)
                else:
                    enriched = np.zeros((n_ratio_bins, 3), dtype=float)

        enriched_frames.append(enriched)

    # Tiempos de frame
    frame_times = np.arange(n_frames) * hop_length / sr

    return np.array(enriched_frames, dtype=np.float32), frame_times.astype(np.float32)


# ───────────────────────────────────
# 3. PROCESADO DE UN CSV UOEMD
# ───────────────────────────────────

def process_uoemd_csv(
    path: Path,
    sr: int,
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
    Procesa un archivo CSV de UOEMD y retorna histogramas PAREADOS para audio y vibración.
    """
    # Cargar CSV (skip header)
    data = np.loadtxt(path, delimiter=',', skiprows=1)

    # Extraer canales
    audio_signal = data[:, UOEMD_COL_MICROPHONE]       # Micrófono (Voltios)
    vibration_signal = data[:, UOEMD_COL_ACCEL1]       # Acelerómetro 1 (m/s²)

    # Normalizar para estabilidad numérica (z-score)
    audio_signal = (audio_signal - audio_signal.mean()) / (audio_signal.std() + 1e-12)
    vibration_signal = (vibration_signal - vibration_signal.mean()) / (vibration_signal.std() + 1e-12)

    # Procesar cada canal
    params = {
        'sr': sr,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'peak_thr_factor': peak_thr_factor,
        'local_window': local_window,
        'rel_peak_tol': rel_peak_tol,
        'max_band_hz': max_band_hz,
        'min_ratio': min_ratio,
        'max_ratio': max_ratio,
        'n_ratio_bins': n_ratio_bins,
    }

    audio_frames, frame_times = process_single_channel(audio_signal, **params)
    vibration_frames, _ = process_single_channel(vibration_signal, **params)

    # Metadata de condición
    condition_info = parse_uoemd_filename(path.name)

    return {
        'sr': sr,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'n_frames': len(frame_times),
        'n_ratio_bins': n_ratio_bins,
        'n_samples': len(data),
        'condition': condition_info['condition'],
        'fault_type': condition_info['fault_type'],
        'specific': condition_info['specific'],
        'speed': condition_info['speed'],
        'load': condition_info['load'],
        'is_healthy': condition_info['is_healthy'],
        'audio_frames': audio_frames,           # [T, B, 3]
        'vibration_frames': vibration_frames,   # [T, B, 3]
        'frame_times': frame_times,             # [T]
    }


def _process_single_csv(task: Tuple) -> Tuple[str, Dict[str, Any]]:
    """Worker function para multiprocessing."""
    csv_path, input_dir, params = task
    rel = csv_path.relative_to(input_dir).as_posix()
    result = process_uoemd_csv(csv_path, **params)
    return (rel, result)


# ───────────────────────────────────
# 4. CLI & MAIN
# ───────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analizador Roseta – Dual-domain (Audio + Vibración) para UOEMD"
    )
    p.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Carpeta con CSVs de UOEMD (ej. 2_CSV_Data_Files/1_Unloaded_Condition)",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default="roseta_dual.npz",
        help="NPZ de salida. Default: roseta_dual.npz",
    )
    p.add_argument(
        "--sr",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sample rate (UOEMD = 42000 Hz)",
    )
    p.add_argument(
        "--n-fft",
        type=int,
        default=DEFAULT_N_FFT,
        help="Tamaño de ventana FFT (ej. 4096 para Δf≈10Hz)",
    )
    p.add_argument(
        "--hop",
        type=int,
        default=DEFAULT_HOP_LENGTH,
        help="Hop entre frames (ej. 1024 para 75% overlap)",
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
        help="Tamaño de ventana para la mediana local",
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
        "--workers",
        type=int,
        default=0,
        help="Número de workers paralelos. 0=auto, 1=secuencial",
    )
    p.add_argument(
        "--healthy-only",
        action="store_true",
        help="Solo procesar archivos Healthy (HH)",
    )
    return p.parse_args()


def save_roseta_dataset(dataset: Dict[str, Dict], output_path: Path) -> None:
    """
    Guarda el dataset dual-domain en formato NPZ comprimido.

    Estructura:
    - filenames: array de nombres de archivo
    - conditions: array de condiciones (HH, RU, FB, etc.)
    - metadata: dict con info por archivo
    - audio_frames_<idx>: [T, B, 3] histogramas de audio
    - vibration_frames_<idx>: [T, B, 3] histogramas de vibración
    - frame_times_<idx>: [T] timestamps
    """
    filenames = list(dataset.keys())
    conditions = [dataset[f]['condition'] for f in filenames]
    metadata = {}
    arrays = {}

    for idx, (filename, entry) in enumerate(dataset.items()):
        metadata[filename] = {
            'idx': idx,
            'sr': entry['sr'],
            'n_fft': entry['n_fft'],
            'hop_length': entry['hop_length'],
            'n_frames': entry['n_frames'],
            'n_ratio_bins': entry['n_ratio_bins'],
            'n_samples': entry['n_samples'],
            'condition': entry['condition'],
            'fault_type': entry['fault_type'],
            'specific': entry['specific'],
            'speed': entry['speed'],
            'load': entry['load'],
            'is_healthy': entry['is_healthy'],
        }

        arrays[f'audio_frames_{idx}'] = entry['audio_frames']
        arrays[f'vibration_frames_{idx}'] = entry['vibration_frames']
        arrays[f'frame_times_{idx}'] = entry['frame_times']

    output_path = Path(output_path)
    if output_path.suffix != '.npz':
        output_path = output_path.with_suffix('.npz')

    np.savez_compressed(
        output_path,
        filenames=np.array(filenames, dtype=object),
        conditions=np.array(conditions, dtype=object),
        metadata=metadata,
        n_files=len(filenames),
        **arrays
    )

    # Estadísticas
    total_frames = sum(m['n_frames'] for m in metadata.values())
    healthy_count = sum(1 for m in metadata.values() if m['is_healthy'])
    fault_count = len(metadata) - healthy_count
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n✔ Dataset Roseta guardado: {output_path}")
    print(f"  Archivos: {len(filenames)} (Healthy: {healthy_count}, Fallas: {fault_count})")
    print(f"  Frames totales: {total_frames:,}")
    print(f"  Tamaño: {file_size_mb:.1f} MB")

    # Resumen de condiciones
    cond_counts = {}
    for c in conditions:
        cond_counts[c] = cond_counts.get(c, 0) + 1
    print(f"  Condiciones: {cond_counts}")


def main() -> None:
    import multiprocessing as mp

    args = parse_args()

    # Buscar CSVs
    csv_paths = [p for p in args.input_dir.rglob("*.csv")]

    if args.healthy_only:
        csv_paths = [p for p in csv_paths if p.name.upper().startswith("H_H")]
        print(f"Modo Healthy-only: {len(csv_paths)} archivos HH encontrados")

    if not csv_paths:
        raise SystemExit(f"[ERROR] No se encontraron archivos CSV en → {args.input_dir}")

    # Workers
    if args.workers == 0:
        n_workers = max(1, mp.cpu_count() - 2)
    else:
        n_workers = args.workers

    params = {
        'sr': args.sr,
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

    tasks = [(csv, args.input_dir, params) for csv in csv_paths]

    print(f"Procesando {len(csv_paths)} CSVs (Analizador Roseta dual-domain)")
    print(f"Workers: {n_workers} | STFT: n_fft={args.n_fft}, hop={args.hop}")
    print(f"Δf ≈ {args.sr / args.n_fft:.2f} Hz | ~{args.sr / args.hop:.0f} frames/segundo")

    dataset: Dict[str, Any] = {}

    if n_workers == 1:
        for csv in tqdm(csv_paths, unit="csv"):
            rel = csv.relative_to(args.input_dir).as_posix()
            dataset[rel] = process_uoemd_csv(csv, **params)
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_process_single_csv, tasks),
                total=len(tasks),
                unit="csv"
            ))
        dataset = dict(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_roseta_dataset(dataset, args.output)


if __name__ == "__main__":
    main()
