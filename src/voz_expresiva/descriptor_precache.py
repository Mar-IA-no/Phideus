"""Familia A frame-level extraction + temporal alignment a 50 Hz para Fase 1.

Reutiliza V4-lin (4d) y H-series (8d) de `src/bias_control/vocal_descriptors.py`
(Escalón 2), llamados frame-level (NO poolear), y los downsampleamos de la grilla
nativa de Phideus (~100 Hz, hop=160 @ 16 kHz) a 50 Hz para alinear con WavLM
(hop=320 @ 16 kHz).

Política de downsampling (congelada en plan Fase 1):
    mean pool de 2 frames consecutivos a 100 Hz → 1 frame a 50 Hz.

Vector resultante: [T_50Hz, 12] (4 V4-lin + 8 H-series).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from src.bias_control.vocal_descriptors import (
    F0_HOP_LENGTH,
    F0_SR,
    compute_v4_linear,
    compute_h_series,
)

FAMILY_A_DIM = 12  # 4 V4-lin + 8 H-series
WAVLM_FRAME_RATE_HZ = 50  # WavLM hop=320 @ 16 kHz
DESCRIPTOR_FRAME_RATE_HZ = 100  # Phideus native (hop=160 @ 16 kHz)


def extract_f0_speech(
    wav: np.ndarray,
    sr: int = F0_SR,
    fmin: float = 75.0,
    fmax: float = 500.0,
    hop_length: int = F0_HOP_LENGTH,
    frame_length: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Speech F0 via librosa.pyin. Returns (f0, voiced) at 100 Hz."""
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced_flag, _ = librosa.pyin(
            wav, sr=sr, fmin=fmin, fmax=fmax,
            frame_length=frame_length, hop_length=hop_length, fill_na=0.0,
        )
    if f0 is None:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=bool)
    voiced = voiced_flag.astype(bool) if voiced_flag is not None else (f0 > 0)
    return f0.astype(np.float32), voiced


def extract_family_A_frame_level(
    wav_path: str | Path,
    sr_target: int = F0_SR,
    f0_floor: float = 75.0,
    f0_ceil: float = 500.0,
) -> np.ndarray:
    """Extract Familia A frame-level at 100 Hz, then downsample to 50 Hz.

    Returns:
        ndarray shape [T_50Hz, 12], dtype float32. NaN for failure.
    """
    import librosa

    wav, sr = librosa.load(str(wav_path), sr=sr_target, mono=True)
    if wav.size < 2048:
        return _nan_matrix(1, FAMILY_A_DIM)

    f0_np, voiced_np = extract_f0_speech(wav, sr=sr, fmin=f0_floor, fmax=f0_ceil)
    N = len(f0_np)
    if N < 2:
        return _nan_matrix(1, FAMILY_A_DIM)

    wav_t = torch.from_numpy(wav).float().unsqueeze(0)
    f0_t = torch.from_numpy(f0_np).float().unsqueeze(0)
    voiced_t = torch.from_numpy(voiced_np).bool().unsqueeze(0)

    # V4-lin: target_length = N → preserves 100 Hz
    try:
        v4 = compute_v4_linear(f0_t, voiced_t, target_length=N)  # [1, N, 4]
        v4_np = v4[0].cpu().numpy().astype(np.float32)
    except Exception:
        v4_np = _nan_matrix(N, 4)

    # H-series at 100 Hz, no norm_stats (z-scored later per-speaker)
    try:
        h = compute_h_series(wav_t, f0_t, voiced_t, target_length=N, norm_stats=None)
        h_np = h[0].cpu().numpy().astype(np.float32)
    except Exception:
        h_np = _nan_matrix(N, 8)

    # Concatenate to [N, 12] at 100 Hz
    desc_100 = np.concatenate([v4_np, h_np], axis=1)

    # Downsample to 50 Hz: mean pool of 2 consecutive frames
    desc_50 = _mean_pool_2(desc_100)
    return desc_50.astype(np.float32)


def _mean_pool_2(x: np.ndarray) -> np.ndarray:
    """Mean pool of 2 consecutive rows. If odd length, drop last row before pool.

    Args: x shape [N, D].
    Returns: [N//2, D].
    """
    N = x.shape[0]
    if N < 2:
        return x[:1]
    N_pair = (N // 2) * 2
    x_trimmed = x[:N_pair]
    return x_trimmed.reshape(N_pair // 2, 2, -1).mean(axis=1)


def _nan_matrix(rows: int, cols: int) -> np.ndarray:
    m = np.empty((rows, cols), dtype=np.float32)
    m[:] = np.nan
    return m


def align_to_wavlm_length(desc: np.ndarray, T_wavlm: int) -> np.ndarray:
    """Align descriptor length to WavLM frame count.

    If `desc` has more frames, truncate. If fewer, pad with last row's mean.
    Both modalities are at 50 Hz, so mismatches are usually 1-2 frames.

    Args:
        desc: [T_desc, 12]
        T_wavlm: target length

    Returns:
        [T_wavlm, 12]
    """
    T_desc = desc.shape[0]
    if T_desc == T_wavlm:
        return desc
    if T_desc > T_wavlm:
        return desc[:T_wavlm]
    # T_desc < T_wavlm: pad with mean of valid rows
    if T_desc > 0:
        finite_rows = np.isfinite(desc).all(axis=1)
        if finite_rows.any():
            mean_row = desc[finite_rows].mean(axis=0)
        else:
            mean_row = np.zeros(desc.shape[1], dtype=desc.dtype)
    else:
        mean_row = np.zeros(12, dtype=np.float32)
    pad = np.tile(mean_row, (T_wavlm - T_desc, 1))
    return np.vstack([desc, pad]).astype(np.float32)
