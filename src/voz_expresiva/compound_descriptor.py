"""Compound descriptor wrapper for Fase 0A.

Combina cuatro familias (A, B, C, D) sobre una utterance individual:

    Familia A — Phideus-ratio (frame-level, 4-stat pool → 48d)
        V4-lin (4d × 4 stats = 16d)
        H-series (8d × 4 stats = 32d)

    Familia B — Voice quality (utterance-level, 9d)
        7 directos: HNR, CPP, jitter, shimmer, F2/F1, F3/F1, alpha-ratio
        2 proxies:  H1-H2 proxy, H1-A3 proxy

    Familia C — Control no-ratio (frame-level, 4-stat pool → 32d)
        A4-16k (8d × 4 stats)

    Familia D — eGeMAPSv02 baseline (utterance-level, 88d) — reportado APARTE

Vector compuesto Phideus+VQ+Control = 48 + 9 + 32 = 89 dimensiones.
(El plan habla de "29d canónicos"; éste es el conteo post-pooling — cada dim frame-level
expande a 4 stats.)

Familia D NO se concatena al compuesto: se reporta como vector independiente.

Política de pooling (per plan 0A § "Política de alineación temporal y pooling"):
    - Frame-level (A y C): pool 4-estadísticos (mean, std, max, min).
    - Utterance-level (B y D): NO re-poolear.

F0 se extrae con librosa.pyin sobre speech (cf. comentario en
`src/bias_control/vocal_descriptors.py`: "Speech: PYIN").

Para Familia A (V4-lin, H-series), `norm_stats=None` — no se aplican stats heredados
de Lombard; la normalización en 0A es z-score por hablante intra-corpus, posterior a
esta extracción.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.bias_control.vocal_descriptors import (
    F0_HOP_LENGTH,
    F0_SR,
    compute_v4_linear,
    compute_h_series,
    compute_a4_16k,
)
from src.voz_expresiva.voice_quality import (
    VOICE_QUALITY_ALL_KEYS,
    VOICE_QUALITY_KIND,
    compute_voice_quality,
    compute_egemaps_functionals,
)


# Frame-level descriptor dims (before pooling)
FAMILY_A_DIMS = {"V4_lin": 4, "H_series": 8}   # total 12 frame-level
FAMILY_C_DIMS = {"A4_16k": 8}                  # total 8 frame-level

POOL_STATS = ("mean", "std", "max", "min")     # 4 stats per frame-level dim


# ---------------------------------------------------------------------------
# Speech F0 (librosa.pyin)
# ---------------------------------------------------------------------------

def extract_f0_speech(
    wav: np.ndarray,
    sr: int = F0_SR,
    fmin: float = 75.0,
    fmax: float = 500.0,
    hop_length: int = F0_HOP_LENGTH,
    frame_length: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract F0 from speech using librosa.pyin.

    Returns:
        f0: (N_frames,) F0 in Hz, 0.0 for unvoiced.
        voiced: (N_frames,) boolean voicing mask.
    """
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced_flag, _ = librosa.pyin(
            wav, sr=sr, fmin=fmin, fmax=fmax,
            frame_length=frame_length, hop_length=hop_length,
            fill_na=0.0,
        )
    if f0 is None:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=bool)
    voiced = voiced_flag.astype(bool) if voiced_flag is not None else (f0 > 0)
    return f0.astype(np.float32), voiced


# ---------------------------------------------------------------------------
# 4-stat pooling for frame-level descriptors
# ---------------------------------------------------------------------------

def pool_frame_level(desc: torch.Tensor) -> np.ndarray:
    """Pool a frame-level descriptor with mean/std/max/min.

    Args:
        desc: shape [1, T, D] (batched single utterance).

    Returns:
        np.ndarray shape [D * 4] in order: all means, all stds, all maxes, all mins.
    """
    if desc.dim() != 3 or desc.shape[0] != 1:
        raise ValueError(f"Expected [1, T, D], got {tuple(desc.shape)}")
    d = desc[0]  # [T, D]
    pooled = torch.stack([
        d.mean(dim=0),
        d.std(dim=0, unbiased=False),
        d.max(dim=0).values,
        d.min(dim=0).values,
    ], dim=0)  # [4, D]
    return pooled.flatten().cpu().numpy().astype(np.float32)


def pool_stat_names(prefix: str, n_dims: int) -> List[str]:
    """Build column names for a pooled frame-level descriptor."""
    return [f"{prefix}_d{d}_{stat}" for stat in POOL_STATS for d in range(n_dims)]


# ---------------------------------------------------------------------------
# Per-utterance extraction
# ---------------------------------------------------------------------------

@dataclass
class UtteranceVector:
    """Result of extracting all descriptors for one utterance."""
    family_A: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    family_B: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    family_C: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    family_D: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    @property
    def compound(self) -> np.ndarray:
        """Vector compuesto (Familias A + B + C; SIN D)."""
        return np.concatenate([self.family_A, self.family_B, self.family_C])


def compute_all_descriptors(
    wav_path: Path | str,
    sr_target: int = F0_SR,
    f0_floor: float = 75.0,
    f0_ceil: float = 500.0,
    include_egemaps: bool = True,
) -> UtteranceVector:
    """Compute all 4 descriptor families for one utterance.

    Returns a UtteranceVector. Any failed family returns NaN-filled.
    """
    import librosa

    # 1) Load audio
    wav, sr = librosa.load(str(wav_path), sr=sr_target, mono=True)
    if wav.size == 0:
        return UtteranceVector(
            family_A=_nan_vector(48),
            family_B=_nan_vector(9),
            family_C=_nan_vector(32),
            family_D=_nan_vector(88) if include_egemaps else np.array([], dtype=np.float32),
        )

    # 2) F0 for Familia A
    f0_np, voiced_np = extract_f0_speech(
        wav, sr=sr, fmin=f0_floor, fmax=f0_ceil,
        hop_length=F0_HOP_LENGTH, frame_length=2048,
    )

    # Match the H-series expected STFT frame count when possible
    # We pass target_length equal to N (the F0 frame count) so the descriptors keep
    # their native resolution and we pool after.
    N = len(f0_np)
    target_length = max(N, 1)

    wav_t = torch.from_numpy(wav).float().unsqueeze(0)            # [1, T_samples]
    f0_t = torch.from_numpy(f0_np).float().unsqueeze(0)           # [1, N]
    voiced_t = torch.from_numpy(voiced_np).bool().unsqueeze(0)    # [1, N]

    # 3) Familia A — V4-lin + H-series (frame-level)
    try:
        v4 = compute_v4_linear(f0_t, voiced_t, target_length=target_length)  # [1, T, 4]
        v4_pooled = pool_frame_level(v4)
    except Exception:
        v4_pooled = _nan_vector(4 * len(POOL_STATS))

    try:
        h = compute_h_series(
            wav_t, f0_t, voiced_t,
            target_length=target_length,
            norm_stats=None,        # raw scale; z-score per-speaker happens later
        )                                                          # [1, T, 8]
        h_pooled = pool_frame_level(h)
    except Exception:
        h_pooled = _nan_vector(8 * len(POOL_STATS))

    family_A = np.concatenate([v4_pooled, h_pooled])  # 16 + 32 = 48

    # 4) Familia C — A4-16k (frame-level, no F0 needed)
    try:
        a4 = compute_a4_16k(wav_t, target_length=target_length)    # [1, T, 8]
        a4_pooled = pool_frame_level(a4)
    except Exception:
        a4_pooled = _nan_vector(8 * len(POOL_STATS))
    family_C = a4_pooled  # 32

    # 5) Familia B — Voice quality (utterance-level)
    try:
        vq = compute_voice_quality(wav_path, sr_target=sr_target)
        family_B = np.array([vq[k] for k in VOICE_QUALITY_ALL_KEYS], dtype=np.float32)
    except Exception:
        family_B = _nan_vector(9)

    # 6) Familia D — eGeMAPSv02 (utterance-level)
    if include_egemaps:
        try:
            ege = compute_egemaps_functionals(wav_path)
            family_D = np.array(list(ege.values()), dtype=np.float32)
        except Exception:
            family_D = _nan_vector(88)
    else:
        family_D = np.array([], dtype=np.float32)

    return UtteranceVector(family_A=family_A, family_B=family_B,
                           family_C=family_C, family_D=family_D)


def _nan_vector(n: int) -> np.ndarray:
    v = np.empty(n, dtype=np.float32)
    v[:] = np.nan
    return v


# ---------------------------------------------------------------------------
# Column metadata
# ---------------------------------------------------------------------------

def build_feature_names() -> Dict[str, List[str]]:
    """Build feature name lists for each family.

    Returns dict with keys 'A', 'B', 'C', 'D' → list of column names.
    """
    names_A = (
        pool_stat_names("V4lin", 4)
        + pool_stat_names("Hseries", 8)
    )
    names_B = list(VOICE_QUALITY_ALL_KEYS)
    names_C = pool_stat_names("A416k", 8)

    # D names require an opensmile instance; we delay this to runtime.
    try:
        from src.voz_expresiva.voice_quality import egemaps_feature_names
        names_D = list(egemaps_feature_names())
    except Exception:
        names_D = [f"egemaps_d{i}" for i in range(88)]

    return {"A": names_A, "B": names_B, "C": names_C, "D": names_D}


def family_index_for_compound() -> Dict[str, Tuple[int, int]]:
    """Slice indices (start, end_exclusive) of each family inside the compound vector A+B+C."""
    return {
        "A": (0, 48),
        "B": (48, 48 + 9),
        "C": (48 + 9, 48 + 9 + 32),  # 89 total
    }


def voice_quality_kind_array() -> List[str]:
    """Per-position kind ('direct' or 'proxy') for Familia B columns."""
    return [VOICE_QUALITY_KIND[k] for k in VOICE_QUALITY_ALL_KEYS]
