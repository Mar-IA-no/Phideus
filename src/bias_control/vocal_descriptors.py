"""
Vocal descriptors for Escalon 2 S2-P2-main: Speech <-> EGG with natural harmony.

Four descriptor families, all pure signal processing (no learnable parameters):
  - V4-lin: Temporal F0 dynamics with LINEAR ratios (natural scale)
  - V4-log: Temporal F0 dynamics with log2 ratios (perceptual control)
  - H-series: Intra-frame harmonic structure (H2/H1..H6/H1, natural)
  - A4-16k: Spectral band energy deltas (non-ratio control)

F0 is extracted per-modality (no cross-modal leakage):
  - Speech: PYIN (via precomputed cache)
  - EGG: Autocorrelation (via precomputed cache)

All descriptors:
  - Are deterministic per-segment (no batch dependence)
  - Include temporal alignment via F.interpolate to T_cnn
  - Use center=True for STFT consistency
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Frozen constants (versioned in F0 cache metadata)
F0_HOP_LENGTH = 160       # samples @ 16kHz
F0_HOP_SEC = 0.01         # 160 / 16000
F0_SR = 16000
F0_FMIN = 50.0
F0_FMAX = 500.0
F0_FRAME_LENGTH = 2048
SEGMENT_LEN = 2.0
FRAMES_PER_SEGMENT = 201  # round(2.0 / 0.01) + 1


# ---------------------------------------------------------------------------
# F0 extraction: autocorrelation for EGG
# ---------------------------------------------------------------------------

def extract_f0_egg(
    egg: np.ndarray,
    sr: int = F0_SR,
    fmin: float = F0_FMIN,
    fmax: float = F0_FMAX,
    hop_length: int = F0_HOP_LENGTH,
    frame_length: int = F0_FRAME_LENGTH,
    voicing_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract F0 from EGG signal using autocorrelation.

    EGG is quasi-periodic with impedance baseline — PYIN doesn't work well.
    Uses autocorrelation with period search in [T_min, T_max] range.

    Args:
        egg: 1D EGG waveform
        sr: sample rate
        fmin, fmax: F0 range in Hz
        hop_length: hop between frames
        frame_length: analysis window size
        voicing_threshold: periodicity threshold for voicing decision

    Returns:
        f0: [N_frames] F0 in Hz (0.0 for unvoiced)
        voiced: [N_frames] boolean voicing flag
    """
    T_min = int(sr / fmax)   # min period in samples (500 Hz -> 32 @ 16kHz)
    T_max = int(sr / fmin)   # max period in samples (50 Hz -> 320 @ 16kHz)

    n_frames = 1 + (len(egg) - frame_length) // hop_length
    if n_frames <= 0:
        n_frames = 1 + len(egg) // hop_length

    f0 = np.zeros(n_frames, dtype=np.float32)
    voiced = np.zeros(n_frames, dtype=bool)

    # Pad signal for center=True behavior
    pad = frame_length // 2
    egg_padded = np.pad(egg, (pad, pad), mode='reflect')

    for i in range(n_frames):
        center = i * hop_length + pad
        start = center - frame_length // 2
        end = start + frame_length
        if end > len(egg_padded):
            break

        frame = egg_padded[start:end].astype(np.float64)

        # Remove DC
        frame = frame - frame.mean()

        # Skip silent frames
        if np.max(np.abs(frame)) < 1e-6:
            continue

        # Normalized autocorrelation (full, keep lags >= 0)
        energy = np.sum(frame ** 2)
        if energy < 1e-12:
            continue

        acf_full = np.correlate(frame, frame, mode='full')
        acf = acf_full[frame_length - 1:]  # lags 0, 1, 2, ...
        acf = acf / energy

        # Search for peak in valid period range [T_min, T_max]
        search_end = min(len(acf), T_max + 1)
        if T_min >= search_end:
            continue

        search_region = acf[T_min:search_end]
        if len(search_region) == 0:
            continue

        peak_idx = np.argmax(search_region)
        peak_val = search_region[peak_idx]
        period = T_min + peak_idx

        if peak_val >= voicing_threshold and period > 0:
            f0[i] = sr / period
            voiced[i] = True

    return f0, voiced


# ---------------------------------------------------------------------------
# V4-lin: Temporal F0 dynamics (LINEAR ratios, NATURAL scale)
# ---------------------------------------------------------------------------

def compute_v4_linear(
    f0: torch.Tensor,
    voiced: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """V4-lin descriptor: 4D temporal F0 dynamics with linear ratios.

    Dimensions:
      0: ratio_prev = F0[t]/F0[t-1], clipped [0.5, 2.0], scaled (ratio-1.0)
      1: ratio_next = F0[t+1]/F0[t], same
      2: voicing_strength [0,1], smoothed 3-frame avg
      3: period_regularity = 1 - std_local(ratios), clipped [0,1]

    Unvoiced: ratios=1.0 (neutral -> 0.0 after scaling), voicing=0.

    Args:
        f0: [B, N] F0 in Hz (0 for unvoiced)
        voiced: [B, N] boolean voicing mask
        target_length: T_cnn to interpolate to

    Returns:
        [B, target_length, 4]
    """
    B, N = f0.shape
    device = f0.device

    # Voicing as float
    v = voiced.float()  # [B, N]

    # Smoothed voicing strength (3-frame average)
    v_smooth = F.avg_pool1d(v.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)

    # Replace unvoiced F0 with neighbor or 1.0 for safe ratio computation
    f0_safe = f0.clone()
    f0_safe[~voiced] = 0.0

    # Compute ratios only where both frames are voiced
    # ratio_prev = F0[t] / F0[t-1]
    f0_prev = torch.cat([f0_safe[:, :1], f0_safe[:, :-1]], dim=1)
    v_prev = torch.cat([voiced[:, :1], voiced[:, :-1]], dim=1)
    both_prev = voiced & v_prev

    ratio_prev = torch.ones(B, N, device=device)
    safe_denom = f0_prev.clamp(min=1.0)
    ratio_prev[both_prev] = (f0_safe[both_prev] / safe_denom[both_prev])

    # ratio_next = F0[t+1] / F0[t]
    f0_next = torch.cat([f0_safe[:, 1:], f0_safe[:, -1:]], dim=1)
    v_next = torch.cat([voiced[:, 1:], voiced[:, -1:]], dim=1)
    both_next = voiced & v_next

    ratio_next = torch.ones(B, N, device=device)
    safe_denom_curr = f0_safe.clamp(min=1.0)
    ratio_next[both_next] = (f0_next[both_next] / safe_denom_curr[both_next])

    # Clip to [0.5, 2.0] and scale to [-0.5, 1.0]
    ratio_prev = ratio_prev.clamp(0.5, 2.0) - 1.0
    ratio_next = ratio_next.clamp(0.5, 2.0) - 1.0

    # Period regularity: 1 - std of local ratios (5-frame window)
    all_ratios = torch.stack([ratio_prev, ratio_next], dim=-1)  # [B, N, 2]
    # Use avg of abs ratios in local window as regularity proxy
    ratio_mag = all_ratios.abs().mean(dim=-1)  # [B, N]
    # Local std via pooling
    ratio_sq = F.avg_pool1d(
        (ratio_mag ** 2).unsqueeze(1), kernel_size=5, stride=1, padding=2
    ).squeeze(1)
    ratio_mean = F.avg_pool1d(
        ratio_mag.unsqueeze(1), kernel_size=5, stride=1, padding=2
    ).squeeze(1)
    local_var = (ratio_sq - ratio_mean ** 2).clamp(min=0.0)
    local_std = local_var.sqrt()
    regularity = (1.0 - local_std).clamp(0.0, 1.0)

    # Stack [B, N, 4]
    desc = torch.stack([ratio_prev, ratio_next, v_smooth, regularity], dim=-1)

    # Interpolate to target_length
    desc = desc.transpose(1, 2)  # [B, 4, N]
    desc = F.interpolate(desc, size=target_length, mode='linear', align_corners=False)
    return desc.transpose(1, 2)  # [B, target_length, 4]


# ---------------------------------------------------------------------------
# V4-log: Temporal F0 dynamics (log2 ratios, perceptual control)
# ---------------------------------------------------------------------------

def compute_v4_log(
    f0: torch.Tensor,
    voiced: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """V4-log descriptor: 4D temporal F0 dynamics with log2 ratios.

    Same as V4-lin but uses log2(ratio), clipped [-1, 1].
    Unvoiced: 0.0 (neutral in log space).

    Args:
        f0: [B, N] F0 in Hz (0 for unvoiced)
        voiced: [B, N] boolean voicing mask
        target_length: T_cnn to interpolate to

    Returns:
        [B, target_length, 4]
    """
    B, N = f0.shape
    device = f0.device

    v = voiced.float()
    v_smooth = F.avg_pool1d(v.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)

    f0_safe = f0.clone()
    f0_safe[~voiced] = 0.0

    # ratio_prev
    f0_prev = torch.cat([f0_safe[:, :1], f0_safe[:, :-1]], dim=1)
    v_prev = torch.cat([voiced[:, :1], voiced[:, :-1]], dim=1)
    both_prev = voiced & v_prev

    log_ratio_prev = torch.zeros(B, N, device=device)
    safe_denom = f0_prev.clamp(min=1.0)
    raw_ratio = f0_safe / safe_denom
    log_ratio_prev[both_prev] = torch.log2(raw_ratio[both_prev].clamp(min=0.01))
    log_ratio_prev = log_ratio_prev.clamp(-1.0, 1.0)

    # ratio_next
    f0_next = torch.cat([f0_safe[:, 1:], f0_safe[:, -1:]], dim=1)
    v_next = torch.cat([voiced[:, 1:], voiced[:, -1:]], dim=1)
    both_next = voiced & v_next

    log_ratio_next = torch.zeros(B, N, device=device)
    safe_denom_curr = f0_safe.clamp(min=1.0)
    raw_ratio_next = f0_next / safe_denom_curr
    log_ratio_next[both_next] = torch.log2(raw_ratio_next[both_next].clamp(min=0.01))
    log_ratio_next = log_ratio_next.clamp(-1.0, 1.0)

    # Period regularity (same as V4-lin)
    ratio_mag = torch.stack([log_ratio_prev.abs(), log_ratio_next.abs()], dim=-1).mean(dim=-1)
    ratio_sq = F.avg_pool1d(
        (ratio_mag ** 2).unsqueeze(1), kernel_size=5, stride=1, padding=2
    ).squeeze(1)
    ratio_mean = F.avg_pool1d(
        ratio_mag.unsqueeze(1), kernel_size=5, stride=1, padding=2
    ).squeeze(1)
    local_var = (ratio_sq - ratio_mean ** 2).clamp(min=0.0)
    regularity = (1.0 - local_var.sqrt()).clamp(0.0, 1.0)

    desc = torch.stack([log_ratio_prev, log_ratio_next, v_smooth, regularity], dim=-1)
    desc = desc.transpose(1, 2)
    desc = F.interpolate(desc, size=target_length, mode='linear', align_corners=False)
    return desc.transpose(1, 2)


# ---------------------------------------------------------------------------
# H-series: Intra-frame harmonic structure (NATURAL)
# ---------------------------------------------------------------------------

def compute_h_series(
    waveform: torch.Tensor,
    f0: torch.Tensor,
    voiced: torch.Tensor,
    target_length: int,
    norm_stats: Optional[Dict] = None,
    n_fft: int = 2048,
    hop_length: int = F0_HOP_LENGTH,
    sr: int = F0_SR,
    peak_search: int = 2,
) -> torch.Tensor:
    """H-series descriptor: 8D harmonic amplitude structure.

    Dimensions 0-4: log(H_{n+1}/H_1 + 1e-3) for n=1..5 (normalized with frozen stats)
    Dimension 5: harmonic_concentration = sum(H1..H6) / total_energy
    Dimension 6: harmonic_deviation = std(log(Hn/H1))
    Dimension 7: voicing_strength

    Uses n_fft=2048 + local peak search (+-2 bins) around expected harmonic bin.
    Stats are frozen per-modality (speech vs EGG have different harmonic profiles).

    Args:
        waveform: [B, T_samples] raw audio
        f0: [B, N_frames] F0 in Hz (0 for unvoiced)
        voiced: [B, N_frames] boolean voicing mask
        target_length: T_cnn to interpolate to
        norm_stats: dict with 'mean' and 'std' tensors [8] for z-normalization.
                    If None, no normalization applied (used for precomputing stats).
        n_fft: STFT window size (2048 for ~7.8 Hz/bin @ 16kHz)
        hop_length: STFT hop (160, aligned with F0 frames)
        sr: sample rate
        peak_search: +- bins for local peak search

    Returns:
        [B, target_length, 8]
    """
    B = waveform.shape[0]
    device = waveform.device
    freq_res = sr / n_fft  # 16000/2048 = 7.8125 Hz/bin

    # STFT
    window = torch.hann_window(n_fft, device=device)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    stft_out = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=window,
        center=True, return_complex=True,
    )
    magnitude = stft_out.abs()  # [B, n_fft//2+1, T_stft]
    T_stft = magnitude.size(2)
    n_bins = magnitude.size(1)

    # Total energy per frame for harmonic_concentration
    total_energy = magnitude.sum(dim=1)  # [B, T_stft]

    # Align F0 frames with STFT frames (should be same resolution)
    N_f0 = f0.shape[1]
    N = min(T_stft, N_f0)

    f0_aligned = f0[:, :N]         # [B, N]
    v_aligned = voiced[:, :N]      # [B, N]
    mag_aligned = magnitude[:, :, :N]  # [B, n_bins, N]
    total_aligned = total_energy[:, :N]  # [B, N]

    # Extract harmonics H1..H6 with local peak search
    n_harmonics = 6
    harmonics = torch.zeros(B, n_harmonics, N, device=device)

    for h in range(1, n_harmonics + 1):
        # Expected bin for harmonic h
        expected_freq = f0_aligned * h  # [B, N]
        expected_bin = (expected_freq / freq_res).round().long()  # [B, N]

        # Search range
        for offset in range(-peak_search, peak_search + 1):
            bin_idx = (expected_bin + offset).clamp(0, n_bins - 1)  # [B, N]
            # Gather magnitude at this bin
            bin_mag = mag_aligned.gather(1, bin_idx.unsqueeze(1)).squeeze(1)  # [B, N]
            harmonics[:, h - 1, :] = torch.max(harmonics[:, h - 1, :], bin_mag)

    # Zero out unvoiced frames
    v_mask = v_aligned.float().unsqueeze(1)  # [B, 1, N]
    harmonics = harmonics * v_mask  # [B, 6, N]

    H1 = harmonics[:, 0, :]  # [B, N]

    # Dims 0-4: log(H_{n+1}/H_1 + 1e-3) for n=1..5
    h_ratios = []
    for n in range(1, 6):
        ratio = torch.log(harmonics[:, n, :] / (H1 + 1e-6) + 1e-3)
        ratio = ratio * v_aligned.float()  # zero unvoiced
        h_ratios.append(ratio)

    # Dim 5: harmonic_concentration = sum(H1..H6) / total_energy
    h_sum = harmonics.sum(dim=1)  # [B, N]
    h_conc = h_sum / (total_aligned + 1e-8)
    h_conc = h_conc.clamp(0.0, 1.0) * v_aligned.float()

    # Dim 6: harmonic_deviation = std(log(Hn/H1)) over n=2..6
    h_ratio_stack = torch.stack(h_ratios, dim=-1)  # [B, N, 5]
    h_dev = h_ratio_stack.std(dim=-1)  # [B, N]
    h_dev = h_dev * v_aligned.float()

    # Dim 7: voicing_strength (smoothed 3-frame average)
    v_float = v_aligned.float()  # [B, N]
    if N >= 3:
        v_smooth = F.avg_pool1d(v_float.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    else:
        v_smooth = v_float

    # Stack all 8 dims
    desc = torch.stack(h_ratios + [h_conc, h_dev, v_smooth], dim=-1)  # [B, N, 8]

    # Apply frozen normalization stats if provided
    if norm_stats is not None:
        mean = norm_stats['mean'].to(device)  # [8]
        std = norm_stats['std'].to(device)    # [8]
        # Only normalize voiced frames to avoid pulling zeros toward nonzero mean
        v_expand = v_aligned.float().unsqueeze(-1)  # [B, N, 1]
        normalized = (desc - mean) / (std + 1e-8)
        desc = normalized * v_expand  # zero stays zero for unvoiced

    # Pad if STFT produced fewer frames than expected
    if N < FRAMES_PER_SEGMENT:
        pad_n = FRAMES_PER_SEGMENT - N
        desc = F.pad(desc, (0, 0, 0, pad_n))

    # Interpolate to target_length
    desc = desc.transpose(1, 2)  # [B, 8, N]
    desc = F.interpolate(desc, size=target_length, mode='linear', align_corners=False)
    return desc.transpose(1, 2)  # [B, target_length, 8]


def load_h_series_norm_stats(path: str, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Load frozen H-series normalization stats from JSON."""
    with open(path) as f:
        data = json.load(f)
    return {
        'mean': torch.tensor(data['mean'], dtype=torch.float32, device=device),
        'std': torch.tensor(data['std'], dtype=torch.float32, device=device),
    }


# ---------------------------------------------------------------------------
# A4-16k: Spectral band energy deltas (non-ratio control)
# ---------------------------------------------------------------------------

# 8 log-spaced bands for sr=16kHz, n_fft=1024 (freq_res=15.625 Hz/bin)
A4_16K_BAND_EDGES = [
    (3, 6),      # ~47-94 Hz
    (6, 12),     # ~94-188 Hz
    (12, 24),    # ~188-375 Hz
    (24, 48),    # ~375-750 Hz
    (48, 96),    # ~750-1500 Hz
    (96, 192),   # ~1500-3000 Hz
    (192, 384),  # ~3000-6000 Hz
    (384, 513),  # ~6000-8000 Hz (Nyquist at 8kHz)
]


def compute_a4_16k(
    waveform: torch.Tensor,
    target_length: int,
    n_fft: int = 1024,
    hop_length: int = F0_HOP_LENGTH,
) -> torch.Tensor:
    """A4-16k descriptor: 8D spectral band energy deltas (control).

    Adapted from compute_audio_descriptor_a4 for sr=16kHz.
    Does NOT require F0. Deterministic per-segment (z-score per-segment per-band).

    Args:
        waveform: [B, T_samples] raw audio at 16kHz
        target_length: T_cnn to interpolate to

    Returns:
        [B, target_length, 8]
    """
    B = waveform.shape[0]
    device = waveform.device

    window = torch.hann_window(n_fft, device=device)
    stft_out = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=window,
        center=True, return_complex=True,
    )
    log_mag = torch.log1p(stft_out.abs())  # [B, n_fft//2+1, T_stft]

    T_stft = log_mag.size(2)
    bands = []
    for lo, hi in A4_16K_BAND_EDGES:
        hi_clamped = min(hi, log_mag.size(1))
        band_mean = log_mag[:, lo:hi_clamped, :].mean(dim=1)
        bands.append(band_mean)
    banded = torch.stack(bands, dim=1)  # [B, 8, T_stft]

    # Temporal delta
    delta = banded[:, :, 1:] - banded[:, :, :-1]
    zero_pad = torch.zeros(B, 8, 1, device=device)
    delta = torch.cat([zero_pad, delta], dim=2)  # [B, 8, T_stft]

    # Z-score per-segment per-band (deterministic)
    mean = delta.mean(dim=2, keepdim=True)
    std = delta.std(dim=2, keepdim=True).clamp(min=1e-8)
    delta = (delta - mean) / std

    # Interpolate to target_length
    delta = F.interpolate(delta, size=target_length, mode='linear', align_corners=False)
    return delta.transpose(1, 2)  # [B, target_length, 8]


# ---------------------------------------------------------------------------
# Descriptor factory
# ---------------------------------------------------------------------------

DESCRIPTOR_DIMS = {
    'v4_lin': 4,
    'v4_log': 4,
    'h_series': 8,
    'a4_16k': 8,
}


def get_descriptor_dim(name: str) -> int:
    """Return the output dimensionality for a descriptor name."""
    if name not in DESCRIPTOR_DIMS:
        raise ValueError(f"Unknown descriptor: {name}. Options: {list(DESCRIPTOR_DIMS.keys())}")
    return DESCRIPTOR_DIMS[name]
