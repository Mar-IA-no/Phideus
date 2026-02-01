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
DEFAULT_PEAK_THRESHOLD_FACTOR: float = 2.5  # v2.2: más estricto (antes 1.25)
DEFAULT_LOCAL_MEDIAN_WINDOW: int = 30
DEFAULT_REL_PEAK_TOL: float = 0.01
DEFAULT_MAX_BAND_HZ: float | None = None
DEFAULT_MIN_RATIO: float = 1.0
DEFAULT_MAX_RATIO: float = 6.0
DEFAULT_N_RATIO_BINS: int = 256

# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTOR v2.2 - Nuevos parámetros para discriminabilidad
# ═══════════════════════════════════════════════════════════════════════════════

# Selección de picos (CORE)
DEFAULT_TOP_K_PEAKS: int = 12                    # Límite duro por frame
DEFAULT_MIN_PROMINENCE: float = 0.2              # Prominencia mínima normalizada
DEFAULT_MIN_PEAK_DISTANCE_HZ: float = 50.0       # Separación mínima entre picos

# Estabilidad temporal (CORE - NO OPCIONAL)
DEFAULT_TEMPORAL_WINDOW_FRAMES: int = 10         # ~0.5-1s dependiendo de hop
DEFAULT_TEMPORAL_STABILITY_THRESHOLD: float = 0.6  # 60% de frames
DEFAULT_TEMPORAL_FREQ_TOLERANCE_HZ: float = 20.0   # Tolerancia para matching
DEFAULT_MIN_PEAKS_FALLBACK: int = 3              # Mínimo si no hay estables

# Anti-ubiquidad (opcional)
DEFAULT_USE_TFIDF: bool = False                  # Desactivado por defecto para sweep

# Binning (opcional)
DEFAULT_USE_WARPED_BINS: bool = False            # Uniforme por defecto
DEFAULT_WARPED_GAMMA: float = 0.5                # gamma < 1 = más densidad cerca de 1.0

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


def calculate_prominence(mag_frame: np.ndarray, peak_idx: int, window: int = 10) -> float:
    """
    Calcula prominencia de un pico: altura relativa respecto a vecinos.

    Prominencia = amplitud_pico - max(min_left, min_right)
    donde min_left/min_right son los mínimos locales a cada lado.
    """
    n = len(mag_frame)
    peak_val = mag_frame[peak_idx]

    # Buscar mínimo a la izquierda
    left_start = max(0, peak_idx - window)
    left_min = np.min(mag_frame[left_start:peak_idx]) if peak_idx > left_start else peak_val

    # Buscar mínimo a la derecha
    right_end = min(n, peak_idx + window + 1)
    right_min = np.min(mag_frame[peak_idx+1:right_end]) if peak_idx + 1 < right_end else peak_val

    # Prominencia = altura sobre el "piso" más alto de ambos lados
    floor = max(left_min, right_min)
    prominence = peak_val - floor

    return max(0.0, prominence)


def warped_bin_edges(
    n_bins: int,
    ratio_min: float,
    ratio_max: float,
    gamma: float = 0.5,
    use_log: bool = False,
) -> np.ndarray:
    """
    Genera edges de bins con densidad variable (warped).

    Args:
        n_bins: Número de bins
        ratio_min: Ratio mínimo
        ratio_max: Ratio máximo
        gamma: Factor de warping (< 1 = más densidad cerca de ratio_min)
        use_log: Si True, usa escala logarítmica en lugar de potencia

    Returns:
        Array de n_bins + 1 edges
    """
    if use_log:
        # Escala logarítmica suave
        log_min = np.log(ratio_min)
        log_max = np.log(ratio_max)
        log_edges = np.linspace(log_min, log_max, n_bins + 1)
        return np.exp(log_edges)
    else:
        # Transformación tipo potencia
        t = np.linspace(0, 1, n_bins + 1)
        t_warped = t ** gamma
        return ratio_min + (ratio_max - ratio_min) * t_warped


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTOR v2.2 - Funciones de estabilidad temporal
# ═══════════════════════════════════════════════════════════════════════════════


def extract_peaks_with_prominence(
    mag_frame: np.ndarray,
    freqs: np.ndarray,
    local_window: int,
    peak_thr_factor: float,
    top_k: int,
    min_prominence: float,
    min_peak_distance_hz: float,
    max_band_hz: float | None,
) -> list:
    """
    Extrae picos con filtrado por prominencia y Top-K.

    Returns:
        Lista de dicts con 'freq', 'amp', 'prominence', 'score'
    """
    from scipy.signal import find_peaks

    # Umbral local
    thr = local_threshold(mag_frame, local_window, peak_thr_factor)

    # Detección inicial de picos
    raw_peaks, _ = find_peaks(mag_frame, height=thr)

    if raw_peaks.size == 0:
        return []

    # Filtrar por banda máxima
    if max_band_hz is not None:
        raw_peaks = raw_peaks[freqs[raw_peaks] <= max_band_hz]

    if raw_peaks.size == 0:
        return []

    # Calcular prominencia para cada pico
    peaks_data = []
    for p_idx in raw_peaks:
        amp = mag_frame[p_idx]
        freq = freqs[p_idx]
        prom = calculate_prominence(mag_frame, p_idx, window=local_window)

        # Normalizar prominencia por amplitud máxima del frame
        max_amp = mag_frame.max()
        prom_norm = prom / (max_amp + 1e-12)

        # Filtrar por prominencia mínima
        if prom_norm < min_prominence:
            continue

        # Score combinado: prominencia × amplitud
        score = prom_norm * amp

        peaks_data.append({
            'idx': p_idx,
            'freq': float(freq),
            'amp': float(amp),
            'prominence': float(prom_norm),
            'score': float(score),
        })

    if not peaks_data:
        return []

    # Ordenar por score descendente
    peaks_data.sort(key=lambda x: x['score'], reverse=True)

    # Filtrar por distancia mínima (evitar picos muy cercanos)
    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 10.0
    min_dist_bins = int(min_peak_distance_hz / freq_res)

    filtered_peaks = []
    for peak in peaks_data:
        # Verificar que no esté muy cerca de un pico ya seleccionado
        too_close = False
        for selected in filtered_peaks:
            if abs(peak['freq'] - selected['freq']) < min_peak_distance_hz:
                too_close = True
                break
        if not too_close:
            filtered_peaks.append(peak)

        # Límite Top-K
        if len(filtered_peaks) >= top_k:
            break

    return filtered_peaks


def filter_temporally_stable_peaks(
    peaks_per_frame: list,
    freqs: np.ndarray,
    window_size: int = 10,
    stability_threshold: float = 0.6,
    freq_tolerance_hz: float = 20.0,
    min_peaks_fallback: int = 3,
) -> list:
    """
    Filtra picos para mantener solo los temporalmente estables.

    Un pico en frame t es "estable" si aparece (con tolerancia frecuencial)
    en al menos stability_threshold × window_size frames de la ventana
    [t - W/2, t + W/2].

    Args:
        peaks_per_frame: Lista de listas de picos (dicts con 'freq', 'amp', etc.)
        freqs: Array de frecuencias para referencia
        window_size: Tamaño de ventana temporal (frames)
        stability_threshold: Fracción mínima de apariciones (0.0 - 1.0)
        freq_tolerance_hz: Tolerancia para considerar "mismo pico"
        min_peaks_fallback: Mínimo de picos si no hay estables suficientes

    Returns:
        Lista de listas de picos estables por frame
    """
    n_frames = len(peaks_per_frame)
    stable_peaks_per_frame = []

    for t in range(n_frames):
        frame_peaks = peaks_per_frame[t]

        if not frame_peaks:
            stable_peaks_per_frame.append([])
            continue

        # Definir ventana temporal
        t_start = max(0, t - window_size // 2)
        t_end = min(n_frames, t + window_size // 2 + 1)
        n_frames_in_window = t_end - t_start

        stable_peaks = []
        for peak in frame_peaks:
            peak_freq = peak['freq']

            # Contar apariciones en la ventana
            count = 0
            for t_neighbor in range(t_start, t_end):
                neighbor_peaks = peaks_per_frame[t_neighbor]
                # Buscar pico similar en frecuencia
                for np_peak in neighbor_peaks:
                    if abs(np_peak['freq'] - peak_freq) < freq_tolerance_hz:
                        count += 1
                        break  # Solo contar una vez por frame

            # Verificar estabilidad
            stability_ratio = count / n_frames_in_window
            if stability_ratio >= stability_threshold:
                # Agregar info de estabilidad al pico
                peak_copy = peak.copy()
                peak_copy['stability'] = stability_ratio
                stable_peaks.append(peak_copy)

        # Fallback si muy pocos picos estables
        if len(stable_peaks) < min_peaks_fallback and len(frame_peaks) > 0:
            # Tomar los top-K originales por score
            sorted_peaks = sorted(frame_peaks, key=lambda p: p['score'], reverse=True)
            stable_peaks = sorted_peaks[:min_peaks_fallback]
            # Marcar como fallback
            for p in stable_peaks:
                p['stability'] = 0.0  # No pasó filtro de estabilidad

        stable_peaks_per_frame.append(stable_peaks)

    return stable_peaks_per_frame


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTELLATION EXTRACTION (Fase 3A)
# ═══════════════════════════════════════════════════════════════════════════════

# Default parameters for constellation extraction
# v2 (Fase 3A-1b): Improved for discriminability
DEFAULT_MAX_ANCHORS: int = 16              # K anchors per frame (was 12)
DEFAULT_MAX_TARGETS_PER_ANCHOR: int = 4    # M targets per anchor (back to 4 for more tokens)
DEFAULT_TARGET_ZONE_FRAMES: int = 3        # Temporal window for targets (was 5, stricter)
DEFAULT_TARGET_ZONE_HZ: float = 1000.0     # Frequency window for targets (balanced)
DEFAULT_N_FREQUENCY_BANDS: int = 16        # Number of frequency bands (16 = 256 combos)
DEFAULT_MIN_ANCHOR_FREQ_HZ: float = 50.0   # Min frequency for anchors (filter DC/low noise)
DEFAULT_USE_LINEAR_BANDS: bool = True      # Linear bands for motor signals (vs log for audio)


def get_frequency_band_id(
    freq: float,
    n_bands: int = 16,
    max_freq: float = 8000.0,
    min_freq: float = 50.0,
    use_linear: bool = True,
) -> int:
    """
    Assign a frequency to a band ID (0 to n_bands-1).

    Args:
        freq: Frequency in Hz
        n_bands: Number of bands
        max_freq: Maximum frequency for band assignment
        min_freq: Minimum frequency (frequencies below go to band 0)
        use_linear: If True, use linear bands; if False, use log-scale

    Returns:
        Band ID from 0 to n_bands-1
    """
    if freq <= min_freq:
        return 0

    if use_linear:
        # Linear bands: equal Hz per band
        band = int((freq - min_freq) / (max_freq - min_freq) * n_bands)
    else:
        # Log-scale bands for perceptual relevance
        log_freq = np.log2(max(freq, min_freq))
        log_min = np.log2(min_freq)
        log_max = np.log2(max_freq)
        band = int((log_freq - log_min) / (log_max - log_min) * n_bands)

    return min(max(band, 0), n_bands - 1)


def extract_constellation(
    stable_peaks: list,
    frame_idx: int,
    stable_peaks_per_frame: list,
    max_anchors: int = DEFAULT_MAX_ANCHORS,
    max_targets_per_anchor: int = DEFAULT_MAX_TARGETS_PER_ANCHOR,
    target_zone_frames: int = DEFAULT_TARGET_ZONE_FRAMES,
    target_zone_hz: float = DEFAULT_TARGET_ZONE_HZ,
    n_frequency_bands: int = DEFAULT_N_FREQUENCY_BANDS,
    min_ratio: float = 1.0,
    max_ratio: float = 6.0,
    min_anchor_freq_hz: float = DEFAULT_MIN_ANCHOR_FREQ_HZ,
    use_linear_bands: bool = DEFAULT_USE_LINEAR_BANDS,
) -> np.ndarray:
    """
    Extract constellation tokens from stable peaks.

    For each anchor peak, find the M closest target peaks within a
    time-frequency zone and generate tokens.

    Token format: [log_ratio, delta_t, weight, anchor_band, target_band]

    Args:
        stable_peaks: List of peak dicts for current frame (anchor candidates)
        frame_idx: Current frame index
        stable_peaks_per_frame: All stable peaks for all frames (for targets)
        max_anchors: Maximum anchor peaks per frame (K)
        max_targets_per_anchor: Maximum targets per anchor (M)
        target_zone_frames: Temporal window for finding targets
        target_zone_hz: Frequency window for finding targets
        n_frequency_bands: Number of bands for band_id feature
        min_ratio: Minimum valid ratio
        max_ratio: Maximum valid ratio
        min_anchor_freq_hz: Minimum frequency for anchor peaks
        use_linear_bands: If True, use linear band assignment

    Returns:
        tokens: np.ndarray of shape [n_tokens, 5]
               where n_tokens <= max_anchors * max_targets_per_anchor
               Each token: [log_ratio, delta_t, weight, anchor_band, target_band]
    """
    tokens = []
    n_frames = len(stable_peaks_per_frame)

    # Filter anchors by minimum frequency
    valid_anchors = [p for p in stable_peaks if p['freq'] >= min_anchor_freq_hz]

    # Select top anchors from current frame (sorted by score/amp)
    anchors = sorted(valid_anchors, key=lambda p: p.get('score', p['amp']), reverse=True)
    anchors = anchors[:max_anchors]

    for anchor in anchors:
        anchor_freq = anchor['freq']
        anchor_amp = anchor['amp']
        anchor_band = get_frequency_band_id(
            anchor_freq, n_frequency_bands, use_linear=use_linear_bands
        )

        # Collect target candidates from nearby frames
        target_candidates = []
        for dt in range(-target_zone_frames, target_zone_frames + 1):
            target_frame_idx = frame_idx + dt
            if target_frame_idx < 0 or target_frame_idx >= n_frames:
                continue

            for target in stable_peaks_per_frame[target_frame_idx]:
                target_freq = target['freq']

                # Skip if same peak (same frame and close frequency)
                if dt == 0 and abs(target_freq - anchor_freq) < 10:
                    continue

                # Check frequency zone
                if abs(target_freq - anchor_freq) > target_zone_hz:
                    continue

                # Calculate ratio
                if target_freq > 0 and anchor_freq > 0:
                    ratio = max(target_freq, anchor_freq) / min(target_freq, anchor_freq)

                    # Check ratio bounds
                    if ratio < min_ratio or ratio > max_ratio:
                        continue

                    # Distance metric for sorting (prefer close targets)
                    freq_dist = abs(target_freq - anchor_freq) / target_zone_hz
                    time_dist = abs(dt) / max(target_zone_frames, 1)
                    distance = freq_dist + time_dist

                    target_candidates.append({
                        'freq': target_freq,
                        'amp': target['amp'],
                        'dt': dt,
                        'ratio': ratio,
                        'distance': distance,
                    })

        # Sort by distance and take top M targets
        target_candidates.sort(key=lambda t: t['distance'])
        selected_targets = target_candidates[:max_targets_per_anchor]

        # Generate tokens for each anchor-target pair
        for target in selected_targets:
            target_freq = target['freq']
            target_amp = target['amp']
            ratio = target['ratio']
            dt = target['dt']

            # Token features
            log_ratio = np.log2(ratio)  # [-2.58, 2.58] for ratios [1, 6]
            delta_t = float(dt)
            weight = np.sqrt(anchor_amp * target_amp)
            target_band = get_frequency_band_id(
                target_freq, n_frequency_bands, use_linear=use_linear_bands
            )

            tokens.append([log_ratio, delta_t, weight, float(anchor_band), float(target_band)])

    if not tokens:
        return np.zeros((0, 5), dtype=np.float32)

    return np.array(tokens, dtype=np.float32)


def process_single_channel_constellation(
    signal: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    peak_thr_factor: float,
    local_window: int,
    max_band_hz: float | None,
    # v2.2 parameters
    top_k_peaks: int = DEFAULT_TOP_K_PEAKS,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_peak_distance_hz: float = DEFAULT_MIN_PEAK_DISTANCE_HZ,
    temporal_window_frames: int = DEFAULT_TEMPORAL_WINDOW_FRAMES,
    temporal_stability_threshold: float = DEFAULT_TEMPORAL_STABILITY_THRESHOLD,
    temporal_freq_tolerance_hz: float = DEFAULT_TEMPORAL_FREQ_TOLERANCE_HZ,
    min_peaks_fallback: int = DEFAULT_MIN_PEAKS_FALLBACK,
    # Constellation parameters
    max_anchors: int = DEFAULT_MAX_ANCHORS,
    max_targets_per_anchor: int = DEFAULT_MAX_TARGETS_PER_ANCHOR,
    target_zone_frames: int = DEFAULT_TARGET_ZONE_FRAMES,
    target_zone_hz: float = DEFAULT_TARGET_ZONE_HZ,
    n_frequency_bands: int = DEFAULT_N_FREQUENCY_BANDS,
    min_ratio: float = 1.0,
    max_ratio: float = 6.0,
    min_anchor_freq_hz: float = DEFAULT_MIN_ANCHOR_FREQ_HZ,
    use_linear_bands: bool = DEFAULT_USE_LINEAR_BANDS,
) -> Tuple[list, np.ndarray]:
    """
    Process a channel and return constellation tokens instead of histograms.

    Returns:
        tokens_per_frame: list of np.ndarray, each [n_tokens, 5]
        frame_times: np.ndarray [T]
    """
    # STFT
    mag, freqs = stft_from_signal(signal, sr, n_fft, hop_length)
    n_freq_bins, n_frames = mag.shape

    # PASO 1: Extract peaks with prominence for all frames
    peaks_per_frame = []
    for frame_idx in range(n_frames):
        mag_frame = mag[:, frame_idx]
        peaks = extract_peaks_with_prominence(
            mag_frame=mag_frame,
            freqs=freqs,
            local_window=local_window,
            peak_thr_factor=peak_thr_factor,
            top_k=top_k_peaks,
            min_prominence=min_prominence,
            min_peak_distance_hz=min_peak_distance_hz,
            max_band_hz=max_band_hz,
        )
        peaks_per_frame.append(peaks)

    # PASO 2: Filter by temporal stability
    stable_peaks_per_frame = filter_temporally_stable_peaks(
        peaks_per_frame=peaks_per_frame,
        freqs=freqs,
        window_size=temporal_window_frames,
        stability_threshold=temporal_stability_threshold,
        freq_tolerance_hz=temporal_freq_tolerance_hz,
        min_peaks_fallback=min_peaks_fallback,
    )

    # PASO 3: Extract constellation tokens for each frame
    tokens_per_frame = []
    for frame_idx in range(n_frames):
        stable_peaks = stable_peaks_per_frame[frame_idx]
        tokens = extract_constellation(
            stable_peaks=stable_peaks,
            frame_idx=frame_idx,
            stable_peaks_per_frame=stable_peaks_per_frame,
            max_anchors=max_anchors,
            max_targets_per_anchor=max_targets_per_anchor,
            target_zone_frames=target_zone_frames,
            target_zone_hz=target_zone_hz,
            n_frequency_bands=n_frequency_bands,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            min_anchor_freq_hz=min_anchor_freq_hz,
            use_linear_bands=use_linear_bands,
        )
        tokens_per_frame.append(tokens)

    # Frame times
    frame_times = np.arange(n_frames) * hop_length / sr

    return tokens_per_frame, frame_times.astype(np.float32)


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
    # === v2.2 parameters ===
    top_k_peaks: int = DEFAULT_TOP_K_PEAKS,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_peak_distance_hz: float = DEFAULT_MIN_PEAK_DISTANCE_HZ,
    temporal_window_frames: int = DEFAULT_TEMPORAL_WINDOW_FRAMES,
    temporal_stability_threshold: float = DEFAULT_TEMPORAL_STABILITY_THRESHOLD,
    temporal_freq_tolerance_hz: float = DEFAULT_TEMPORAL_FREQ_TOLERANCE_HZ,
    min_peaks_fallback: int = DEFAULT_MIN_PEAKS_FALLBACK,
    use_warped_bins: bool = DEFAULT_USE_WARPED_BINS,
    warped_gamma: float = DEFAULT_WARPED_GAMMA,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Procesa un canal (audio o vibración) y retorna histogramas temporales.

    v2.2 Features:
    - Top-K peak selection with prominence filtering
    - Temporal stability filtering (CORE)
    - Optional warped bins

    Returns:
        enriched_frames: array [T, B, 3] con histogramas enriquecidos
        frame_times: array [T] con timestamps
    """
    # STFT
    mag, freqs = stft_from_signal(signal, sr, n_fft, hop_length)
    n_freq_bins, n_frames = mag.shape

    # Edges para histogramas (uniforme o warped)
    if use_warped_bins:
        edges_ratio = warped_bin_edges(n_ratio_bins, min_ratio, max_ratio, gamma=warped_gamma)
    else:
        edges_ratio = np.linspace(min_ratio, max_ratio, n_ratio_bins + 1, dtype=float)

    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 1: Extraer picos con prominencia para TODOS los frames
    # ═══════════════════════════════════════════════════════════════════════════
    peaks_per_frame = []
    for frame_idx in range(n_frames):
        mag_frame = mag[:, frame_idx]
        peaks = extract_peaks_with_prominence(
            mag_frame=mag_frame,
            freqs=freqs,
            local_window=local_window,
            peak_thr_factor=peak_thr_factor,
            top_k=top_k_peaks,
            min_prominence=min_prominence,
            min_peak_distance_hz=min_peak_distance_hz,
            max_band_hz=max_band_hz,
        )
        peaks_per_frame.append(peaks)

    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 2: Filtrar por estabilidad temporal (CORE v2.2)
    # ═══════════════════════════════════════════════════════════════════════════
    stable_peaks_per_frame = filter_temporally_stable_peaks(
        peaks_per_frame=peaks_per_frame,
        freqs=freqs,
        window_size=temporal_window_frames,
        stability_threshold=temporal_stability_threshold,
        freq_tolerance_hz=temporal_freq_tolerance_hz,
        min_peaks_fallback=min_peaks_fallback,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 3: Calcular ratios e histogramas para cada frame
    # ═══════════════════════════════════════════════════════════════════════════
    enriched_frames = []

    for frame_idx in range(n_frames):
        stable_peaks = stable_peaks_per_frame[frame_idx]

        if not stable_peaks:
            enriched = np.zeros((n_ratio_bins, 3), dtype=float)
        else:
            # Extraer frecuencias y amplitudes de picos estables
            freqs_sel = [p['freq'] for p in stable_peaks]
            amps_sel = [p['amp'] for p in stable_peaks]

            # Calcular ratios (solo entre picos estables)
            ratios = []
            weights = []
            n_peaks = len(freqs_sel)

            for i in range(n_peaks):
                fi = freqs_sel[i]
                ai = amps_sel[i]
                for j in range(i + 1, n_peaks):
                    fj = freqs_sel[j]
                    aj = amps_sel[j]

                    if fi <= 0 or fj <= 0:
                        continue

                    # Ratio siempre >= 1 (mayor / menor)
                    if fj >= fi:
                        r = fj / fi
                    else:
                        r = fi / fj

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
    # === v2.2 parameters ===
    top_k_peaks: int = DEFAULT_TOP_K_PEAKS,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_peak_distance_hz: float = DEFAULT_MIN_PEAK_DISTANCE_HZ,
    temporal_window_frames: int = DEFAULT_TEMPORAL_WINDOW_FRAMES,
    temporal_stability_threshold: float = DEFAULT_TEMPORAL_STABILITY_THRESHOLD,
    temporal_freq_tolerance_hz: float = DEFAULT_TEMPORAL_FREQ_TOLERANCE_HZ,
    min_peaks_fallback: int = DEFAULT_MIN_PEAKS_FALLBACK,
    use_warped_bins: bool = DEFAULT_USE_WARPED_BINS,
    warped_gamma: float = DEFAULT_WARPED_GAMMA,
    # === Constellation parameters (Fase 3A) ===
    output_format: str = 'histogram',  # 'histogram' or 'constellation'
    max_anchors: int = DEFAULT_MAX_ANCHORS,
    max_targets_per_anchor: int = DEFAULT_MAX_TARGETS_PER_ANCHOR,
    target_zone_frames: int = DEFAULT_TARGET_ZONE_FRAMES,
    target_zone_hz: float = DEFAULT_TARGET_ZONE_HZ,
    n_frequency_bands: int = DEFAULT_N_FREQUENCY_BANDS,
    min_anchor_freq_hz: float = DEFAULT_MIN_ANCHOR_FREQ_HZ,
    use_linear_bands: bool = DEFAULT_USE_LINEAR_BANDS,
) -> Dict[str, Any]:
    """
    Procesa un archivo CSV de UOEMD y retorna histogramas o tokens PAREADOS para audio y vibración.

    v2.2: Incluye parámetros de selección de picos, estabilidad temporal y warped bins.
    v3A: Soporte para output_format='constellation' (tokens sparse).
    """
    # Cargar CSV (skip header)
    data = np.loadtxt(path, delimiter=',', skiprows=1)

    # Extraer canales
    audio_signal = data[:, UOEMD_COL_MICROPHONE]       # Micrófono (Voltios)
    vibration_signal = data[:, UOEMD_COL_ACCEL1]       # Acelerómetro 1 (m/s²)

    # Normalizar para estabilidad numérica (z-score)
    audio_signal = (audio_signal - audio_signal.mean()) / (audio_signal.std() + 1e-12)
    vibration_signal = (vibration_signal - vibration_signal.mean()) / (vibration_signal.std() + 1e-12)

    # Metadata de condición
    condition_info = parse_uoemd_filename(path.name)

    if output_format == 'constellation':
        # Constellation mode: generate tokens instead of histograms
        params_const = {
            'sr': sr,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'peak_thr_factor': peak_thr_factor,
            'local_window': local_window,
            'max_band_hz': max_band_hz,
            'top_k_peaks': top_k_peaks,
            'min_prominence': min_prominence,
            'min_peak_distance_hz': min_peak_distance_hz,
            'temporal_window_frames': temporal_window_frames,
            'temporal_stability_threshold': temporal_stability_threshold,
            'temporal_freq_tolerance_hz': temporal_freq_tolerance_hz,
            'min_peaks_fallback': min_peaks_fallback,
            'max_anchors': max_anchors,
            'max_targets_per_anchor': max_targets_per_anchor,
            'target_zone_frames': target_zone_frames,
            'target_zone_hz': target_zone_hz,
            'n_frequency_bands': n_frequency_bands,
            'min_ratio': min_ratio,
            'max_ratio': max_ratio,
            'min_anchor_freq_hz': min_anchor_freq_hz,
            'use_linear_bands': use_linear_bands,
        }

        audio_tokens, frame_times = process_single_channel_constellation(audio_signal, **params_const)
        vibration_tokens, _ = process_single_channel_constellation(vibration_signal, **params_const)

        return {
            'sr': sr,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_frames': len(frame_times),
            'n_samples': len(data),
            'output_format': 'constellation',
            'max_tokens_per_frame': max_anchors * max_targets_per_anchor,
            'token_dim': 5,  # [log_ratio, delta_t, weight, anchor_band, target_band]
            'condition': condition_info['condition'],
            'fault_type': condition_info['fault_type'],
            'specific': condition_info['specific'],
            'speed': condition_info['speed'],
            'load': condition_info['load'],
            'is_healthy': condition_info['is_healthy'],
            'audio_tokens': audio_tokens,           # List of [n_tokens, 5]
            'vibration_tokens': vibration_tokens,   # List of [n_tokens, 5]
            'frame_times': frame_times,             # [T]
        }

    else:
        # Histogram mode (default, v2.2 behavior)
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
            'top_k_peaks': top_k_peaks,
            'min_prominence': min_prominence,
            'min_peak_distance_hz': min_peak_distance_hz,
            'temporal_window_frames': temporal_window_frames,
            'temporal_stability_threshold': temporal_stability_threshold,
            'temporal_freq_tolerance_hz': temporal_freq_tolerance_hz,
            'min_peaks_fallback': min_peaks_fallback,
            'use_warped_bins': use_warped_bins,
            'warped_gamma': warped_gamma,
        }

        audio_frames, frame_times = process_single_channel(audio_signal, **params)
        vibration_frames, _ = process_single_channel(vibration_signal, **params)

        return {
            'sr': sr,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_frames': len(frame_times),
            'n_ratio_bins': n_ratio_bins,
            'n_samples': len(data),
            'output_format': 'histogram',
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

    # ═══════════════════════════════════════════════════════════════════════════
    # EXTRACTOR v2.2 - Nuevos argumentos
    # ═══════════════════════════════════════════════════════════════════════════
    v22 = p.add_argument_group("Extractor v2.2 Options")

    v22.add_argument(
        "--top-k-peaks",
        type=int,
        default=DEFAULT_TOP_K_PEAKS,
        help=f"Máximo de picos por frame (default: {DEFAULT_TOP_K_PEAKS})",
    )
    v22.add_argument(
        "--min-prominence",
        type=float,
        default=DEFAULT_MIN_PROMINENCE,
        help=f"Prominencia mínima normalizada (default: {DEFAULT_MIN_PROMINENCE})",
    )
    v22.add_argument(
        "--min-peak-distance-hz",
        type=float,
        default=DEFAULT_MIN_PEAK_DISTANCE_HZ,
        help=f"Separación mínima entre picos en Hz (default: {DEFAULT_MIN_PEAK_DISTANCE_HZ})",
    )
    v22.add_argument(
        "--temporal-window",
        type=int,
        default=DEFAULT_TEMPORAL_WINDOW_FRAMES,
        help=f"Ventana temporal para estabilidad (frames) (default: {DEFAULT_TEMPORAL_WINDOW_FRAMES})",
    )
    v22.add_argument(
        "--temporal-stability",
        type=float,
        default=DEFAULT_TEMPORAL_STABILITY_THRESHOLD,
        help=f"Umbral de estabilidad temporal 0-1 (default: {DEFAULT_TEMPORAL_STABILITY_THRESHOLD})",
    )
    v22.add_argument(
        "--temporal-freq-tol",
        type=float,
        default=DEFAULT_TEMPORAL_FREQ_TOLERANCE_HZ,
        help=f"Tolerancia frecuencial para matching de picos (Hz) (default: {DEFAULT_TEMPORAL_FREQ_TOLERANCE_HZ})",
    )
    v22.add_argument(
        "--min-peaks-fallback",
        type=int,
        default=DEFAULT_MIN_PEAKS_FALLBACK,
        help=f"Mínimo de picos si no hay suficientes estables (default: {DEFAULT_MIN_PEAKS_FALLBACK})",
    )
    v22.add_argument(
        "--use-warped-bins",
        action="store_true",
        default=DEFAULT_USE_WARPED_BINS,
        help="Usar bins warped (más densidad cerca de ratio=1)",
    )
    v22.add_argument(
        "--warped-gamma",
        type=float,
        default=DEFAULT_WARPED_GAMMA,
        help=f"Gamma para warped bins (< 1 = más densidad en ratios bajos) (default: {DEFAULT_WARPED_GAMMA})",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # CONSTELLATION (Fase 3A) - Nuevos argumentos
    # ═══════════════════════════════════════════════════════════════════════════
    const = p.add_argument_group("Constellation Options (Fase 3A)")

    const.add_argument(
        "--output-format",
        type=str,
        choices=['histogram', 'constellation'],
        default='histogram',
        help="Output format: 'histogram' (default) or 'constellation' (sparse tokens)",
    )
    const.add_argument(
        "--max-anchors",
        type=int,
        default=DEFAULT_MAX_ANCHORS,
        help=f"Max anchor peaks per frame (K) (default: {DEFAULT_MAX_ANCHORS})",
    )
    const.add_argument(
        "--max-targets-per-anchor",
        type=int,
        default=DEFAULT_MAX_TARGETS_PER_ANCHOR,
        help=f"Max targets per anchor (M) (default: {DEFAULT_MAX_TARGETS_PER_ANCHOR})",
    )
    const.add_argument(
        "--target-zone-frames",
        type=int,
        default=DEFAULT_TARGET_ZONE_FRAMES,
        help=f"Temporal window for targets (frames) (default: {DEFAULT_TARGET_ZONE_FRAMES})",
    )
    const.add_argument(
        "--target-zone-hz",
        type=float,
        default=DEFAULT_TARGET_ZONE_HZ,
        help=f"Frequency window for targets (Hz) (default: {DEFAULT_TARGET_ZONE_HZ})",
    )
    const.add_argument(
        "--n-frequency-bands",
        type=int,
        default=DEFAULT_N_FREQUENCY_BANDS,
        help=f"Number of frequency bands for band_id feature (default: {DEFAULT_N_FREQUENCY_BANDS})",
    )
    const.add_argument(
        "--min-anchor-freq-hz",
        type=float,
        default=DEFAULT_MIN_ANCHOR_FREQ_HZ,
        help=f"Minimum frequency for anchor peaks in Hz (default: {DEFAULT_MIN_ANCHOR_FREQ_HZ})",
    )
    const.add_argument(
        "--use-linear-bands",
        action="store_true",
        default=DEFAULT_USE_LINEAR_BANDS,
        help="Use linear band assignment (default) vs log-scale",
    )
    const.add_argument(
        "--use-log-bands",
        action="store_true",
        help="Use log-scale band assignment (overrides --use-linear-bands)",
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


def save_roseta_constellation_dataset(dataset: Dict[str, Dict], output_path: Path) -> None:
    """
    Guarda el dataset constellation dual-domain en formato NPZ comprimido.

    Estructura (Fase 3A):
    - filenames: array de nombres de archivo
    - conditions: array de condiciones
    - metadata: dict con info por archivo
    - output_format: 'constellation'
    - max_tokens_per_frame: K * M
    - token_dim: 5
    - audio_tokens_<idx>: [T, max_tokens, 5] tokens de audio (padded)
    - audio_mask_<idx>: [T, max_tokens] máscara de tokens válidos
    - vibration_tokens_<idx>: [T, max_tokens, 5] tokens de vibración (padded)
    - vibration_mask_<idx>: [T, max_tokens] máscara de tokens válidos
    - frame_times_<idx>: [T] timestamps
    """
    filenames = list(dataset.keys())
    conditions = [dataset[f]['condition'] for f in filenames]
    metadata = {}
    arrays = {}

    # Get max_tokens from first entry
    first_entry = list(dataset.values())[0]
    max_tokens_per_frame = first_entry.get('max_tokens_per_frame', 48)
    token_dim = first_entry.get('token_dim', 5)

    total_tokens_audio = 0
    total_tokens_vib = 0

    for idx, (filename, entry) in enumerate(dataset.items()):
        metadata[filename] = {
            'idx': idx,
            'sr': entry['sr'],
            'n_fft': entry['n_fft'],
            'hop_length': entry['hop_length'],
            'n_frames': entry['n_frames'],
            'n_samples': entry['n_samples'],
            'condition': entry['condition'],
            'fault_type': entry['fault_type'],
            'specific': entry['specific'],
            'speed': entry['speed'],
            'load': entry['load'],
            'is_healthy': entry['is_healthy'],
        }

        n_frames = entry['n_frames']
        audio_tokens_list = entry['audio_tokens']
        vibration_tokens_list = entry['vibration_tokens']

        # Pad tokens to fixed size [T, max_tokens, token_dim]
        audio_tokens_padded = np.zeros((n_frames, max_tokens_per_frame, token_dim), dtype=np.float32)
        audio_mask = np.zeros((n_frames, max_tokens_per_frame), dtype=np.float32)
        vibration_tokens_padded = np.zeros((n_frames, max_tokens_per_frame, token_dim), dtype=np.float32)
        vibration_mask = np.zeros((n_frames, max_tokens_per_frame), dtype=np.float32)

        for t, tokens in enumerate(audio_tokens_list):
            n_tokens = min(len(tokens), max_tokens_per_frame)
            if n_tokens > 0:
                audio_tokens_padded[t, :n_tokens] = tokens[:n_tokens]
                audio_mask[t, :n_tokens] = 1.0
            total_tokens_audio += n_tokens

        for t, tokens in enumerate(vibration_tokens_list):
            n_tokens = min(len(tokens), max_tokens_per_frame)
            if n_tokens > 0:
                vibration_tokens_padded[t, :n_tokens] = tokens[:n_tokens]
                vibration_mask[t, :n_tokens] = 1.0
            total_tokens_vib += n_tokens

        arrays[f'audio_tokens_{idx}'] = audio_tokens_padded
        arrays[f'audio_mask_{idx}'] = audio_mask
        arrays[f'vibration_tokens_{idx}'] = vibration_tokens_padded
        arrays[f'vibration_mask_{idx}'] = vibration_mask
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
        output_format='constellation',
        max_tokens_per_frame=max_tokens_per_frame,
        token_dim=token_dim,
        **arrays
    )

    # Estadísticas
    total_frames = sum(m['n_frames'] for m in metadata.values())
    healthy_count = sum(1 for m in metadata.values() if m['is_healthy'])
    fault_count = len(metadata) - healthy_count
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    avg_tokens_audio = total_tokens_audio / max(total_frames, 1)
    avg_tokens_vib = total_tokens_vib / max(total_frames, 1)

    print(f"\n✔ Dataset Roseta Constellation guardado: {output_path}")
    print(f"  Archivos: {len(filenames)} (Healthy: {healthy_count}, Fallas: {fault_count})")
    print(f"  Frames totales: {total_frames:,}")
    print(f"  Tokens por frame: {max_tokens_per_frame} (dim={token_dim})")
    print(f"  Avg tokens/frame: Audio={avg_tokens_audio:.1f}, Vib={avg_tokens_vib:.1f}")
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
        # v2.2 params
        'top_k_peaks': args.top_k_peaks,
        'min_prominence': args.min_prominence,
        'min_peak_distance_hz': args.min_peak_distance_hz,
        'temporal_window_frames': args.temporal_window,
        'temporal_stability_threshold': args.temporal_stability,
        'temporal_freq_tolerance_hz': args.temporal_freq_tol,
        'min_peaks_fallback': args.min_peaks_fallback,
        'use_warped_bins': args.use_warped_bins,
        'warped_gamma': args.warped_gamma,
        # Constellation params (Fase 3A)
        'output_format': args.output_format,
        'max_anchors': args.max_anchors,
        'max_targets_per_anchor': args.max_targets_per_anchor,
        'target_zone_frames': args.target_zone_frames,
        'target_zone_hz': args.target_zone_hz,
        'n_frequency_bands': args.n_frequency_bands,
        'min_anchor_freq_hz': args.min_anchor_freq_hz,
        'use_linear_bands': not args.use_log_bands,
    }

    tasks = [(csv, args.input_dir, params) for csv in csv_paths]

    version_str = "Constellation (Fase 3A-1b)" if args.output_format == 'constellation' else "v2.2"
    print(f"Procesando {len(csv_paths)} CSVs (Analizador Roseta {version_str} dual-domain)")
    print(f"Workers: {n_workers} | STFT: n_fft={args.n_fft}, hop={args.hop}")
    print(f"Δf ≈ {args.sr / args.n_fft:.2f} Hz | ~{args.sr / args.hop:.0f} frames/segundo")
    print(f"\n[v2.2 Config]")
    print(f"  Top-K peaks: {args.top_k_peaks} | Min prominence: {args.min_prominence}")
    print(f"  Temporal window: {args.temporal_window} frames | Stability threshold: {args.temporal_stability}")
    if args.output_format == 'histogram':
        print(f"  Warped bins: {args.use_warped_bins} (gamma={args.warped_gamma})")
    else:
        print(f"\n[Constellation Config v2]")
        print(f"  Max anchors: {args.max_anchors} | Max targets/anchor: {args.max_targets_per_anchor}")
        print(f"  Target zone: {args.target_zone_frames} frames × {args.target_zone_hz} Hz")
        print(f"  Frequency bands: {args.n_frequency_bands} ({'linear' if not args.use_log_bands else 'log'})")
        print(f"  Min anchor freq: {args.min_anchor_freq_hz} Hz")

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

    # Use appropriate save function based on output format
    if args.output_format == 'constellation':
        save_roseta_constellation_dataset(dataset, args.output)
    else:
        save_roseta_dataset(dataset, args.output)


if __name__ == "__main__":
    main()
