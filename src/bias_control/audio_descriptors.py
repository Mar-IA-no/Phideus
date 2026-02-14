"""
Audio-domain ratio descriptors for Gate 4.3.

Two descriptors computed from raw audio waveform via STFT:
  - A4: Local log-frequency deltas (analog of D4 for audio)
  - A7: Rational Attractor (tests Phideus hypothesis directly)

Both are pure signal-processing — no learnable parameters.
All operations use torch for GPU compatibility.
"""

import torch
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# A4: Local Log-Frequency Deltas
# ---------------------------------------------------------------------------

# 8 log-spaced frequency bands (STFT bin ranges, n_fft=2048, sr=24000)
# freq_res = 24000/2048 ≈ 11.72 Hz/bin
A4_BAND_EDGES = [
    (4, 8),       # ~47-94 Hz
    (8, 16),      # ~94-188 Hz
    (16, 32),     # ~188-375 Hz
    (32, 64),     # ~375-750 Hz
    (64, 128),    # ~750-1500 Hz
    (128, 256),   # ~1500-3000 Hz
    (256, 512),   # ~3000-6000 Hz
    (512, 1025),  # ~6000-12000 Hz
]


def compute_audio_descriptor_a4(audio: torch.Tensor, target_length: int = None,
                                 n_fft: int = 2048, hop_length: int = 512
                                 ) -> torch.Tensor:
    """
    Local log-frequency deltas — analog of D4 for audio side.

    Args:
        audio: [B, 96000] raw waveform
        target_length: T' (CNN output temporal dim, typically 2400).
                       If None, returns at native STFT resolution (~188 frames).
        n_fft: STFT window size
        hop_length: STFT hop

    Returns:
        [B, target_length, 8] — normalized temporal deltas per freq band
    """
    B = audio.size(0)
    device = audio.device

    # STFT with explicit params for reproducibility
    window = torch.hann_window(n_fft, device=device)
    stft_out = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=window,
        center=True, return_complex=True,
    )
    magnitude = stft_out.abs()  # [B, n_fft//2+1, T_stft]

    # Log magnitude
    log_mag = torch.log1p(magnitude)  # [B, 1025, T_stft]

    # Group into 8 log-freq bands by averaging bins
    T_stft = log_mag.size(2)
    bands = []
    for lo, hi in A4_BAND_EDGES:
        band_mean = log_mag[:, lo:hi, :].mean(dim=1)  # [B, T_stft]
        bands.append(band_mean)
    banded = torch.stack(bands, dim=1)  # [B, 8, T_stft]

    # Temporal delta: diff along time axis, pad first frame with zeros
    delta = banded[:, :, 1:] - banded[:, :, :-1]  # [B, 8, T_stft-1]
    zero_pad = torch.zeros(B, 8, 1, device=device)
    delta = torch.cat([zero_pad, delta], dim=2)  # [B, 8, T_stft]

    # Normalize per band per sample (zero mean, unit std)
    mean = delta.mean(dim=2, keepdim=True)
    std = delta.std(dim=2, keepdim=True).clamp(min=1e-8)
    delta = (delta - mean) / std  # [B, 8, T_stft]

    # Interpolate to target_length (or keep native STFT resolution)
    if target_length is not None:
        delta = F.interpolate(
            delta, size=target_length, mode='linear', align_corners=False,
        )  # [B, 8, target_length]

    return delta.transpose(1, 2)  # [B, T, 8]


# ---------------------------------------------------------------------------
# A7: Rational Attractor
# ---------------------------------------------------------------------------

# 12 just-intonation attractors (log2, octave-folded to [0, 1))
A7_ATTRACTORS = torch.tensor([
    math.log2(1 / 1),      # 0.000  unison/octave
    math.log2(16 / 15),    # 0.093  minor 2nd
    math.log2(9 / 8),      # 0.170  major 2nd
    math.log2(6 / 5),      # 0.263  minor 3rd
    math.log2(5 / 4),      # 0.322  major 3rd
    math.log2(4 / 3),      # 0.415  perfect 4th
    math.log2(7 / 5),      # 0.485  tritone
    math.log2(3 / 2),      # 0.585  perfect 5th
    math.log2(8 / 5),      # 0.678  minor 6th
    math.log2(5 / 3),      # 0.737  major 6th
    math.log2(7 / 4),      # 0.807  harmonic 7th
    math.log2(15 / 8),     # 0.907  major 7th
], dtype=torch.float32)

A7_SIGMA = 0.02  # Gaussian width for soft assignment
A7_NUM_PEAKS = 8
A7_MIN_FREQ_HZ = 50.0


def compute_audio_descriptor_a7(audio: torch.Tensor, target_length: int = None,
                                 n_fft: int = 2048, hop_length: int = 512,
                                 sample_rate: int = 24000,
                                 ) -> torch.Tensor:
    """
    Rational Attractor — tests Phideus hypothesis directly.

    Picks spectral peaks, computes pairwise log2 frequency ratios,
    and measures proximity to just-intonation attractors.

    Args:
        audio: [B, 96000] raw waveform
        target_length: T' (CNN output temporal dim, typically 2400).
                       If None, returns at native STFT resolution (~188 frames).
        n_fft: STFT window size
        hop_length: STFT hop
        sample_rate: audio sample rate

    Returns:
        [B, target_length, 12] — attractor activations per frame
    """
    B = audio.size(0)
    device = audio.device
    freq_res = sample_rate / n_fft  # Hz per bin

    # STFT
    window = torch.hann_window(n_fft, device=device)
    stft_out = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=window,
        center=True, return_complex=True,
    )
    magnitude = stft_out.abs()  # [B, n_fft//2+1, T_stft]
    T_stft = magnitude.size(2)

    # Top-k peaks by magnitude
    k = A7_NUM_PEAKS
    peaks_mag, peaks_idx = torch.topk(magnitude, k=k, dim=1)  # [B, k, T_stft]

    # Convert bin indices to Hz
    freq_hz = peaks_idx.float() * freq_res  # [B, k, T_stft]

    # Sort peaks by frequency (topk returns by magnitude — we need freq order)
    sorted_idx = freq_hz.argsort(dim=1)  # [B, k, T_stft]
    freq_hz = freq_hz.gather(1, sorted_idx)
    peaks_mag = peaks_mag.gather(1, sorted_idx)

    # Mask peaks below minimum frequency (DC artifacts)
    freq_mask = (freq_hz >= A7_MIN_FREQ_HZ).float()  # [B, k, T_stft]
    peaks_mag = peaks_mag * freq_mask

    # Clamp frequencies for safe log2 (masked ones won't contribute due to mag=0)
    freq_hz = freq_hz.clamp(min=A7_MIN_FREQ_HZ)

    # Pairwise log2 ratios (upper triangle: i < j)
    # freq_hz: [B, k, T_stft]
    # We want all pairs (i, j) where i < j
    idx_i, idx_j = torch.triu_indices(k, k, offset=1)  # C(8,2) = 28 pairs
    n_pairs = idx_i.size(0)

    freq_i = freq_hz[:, idx_i, :]  # [B, 28, T_stft]
    freq_j = freq_hz[:, idx_j, :]  # [B, 28, T_stft]
    mag_i = peaks_mag[:, idx_i, :]
    mag_j = peaks_mag[:, idx_j, :]

    # Log2 ratio, octave-folded to [0, 1)
    log2_ratio = torch.log2(freq_j / (freq_i + 1e-8))  # [B, 28, T_stft]
    log2_ratio = log2_ratio % 1.0  # octave fold

    # Magnitude weighting: geometric mean of pair magnitudes
    pair_weight = torch.sqrt(mag_i * mag_j + 1e-8)  # [B, 28, T_stft]

    # Soft Gaussian assignment to 12 attractors
    attractors = A7_ATTRACTORS.to(device)  # [12]
    # Reshape for broadcasting: ratio [B, 28, T, 1] vs attractor [12]
    log2_ratio = log2_ratio.unsqueeze(-1)   # [B, 28, T_stft, 1]
    pair_weight = pair_weight.unsqueeze(-1)  # [B, 28, T_stft, 1]

    # Distance to each attractor
    dist = log2_ratio - attractors  # [B, 28, T_stft, 12]
    # Handle wraparound at 0/1 boundary (e.g., ratio=0.99 is close to attractor=0.0)
    dist = torch.min(dist.abs(), (1.0 - dist.abs()))

    activation = torch.exp(-0.5 * (dist / A7_SIGMA) ** 2)  # [B, 28, T_stft, 12]
    weighted = activation * pair_weight  # [B, 28, T_stft, 12]

    # Sum over all 28 pairs → [B, T_stft, 12]
    result = weighted.sum(dim=1)  # [B, T_stft, 12]

    # Normalize per frame (sum=1), handle silent frames
    frame_sum = result.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    result = result / frame_sum  # [B, T_stft, 12]

    # Interpolate to target_length (or keep native STFT resolution)
    if target_length is not None:
        result = result.transpose(1, 2)  # [B, 12, T_stft]
        result = F.interpolate(
            result, size=target_length, mode='linear', align_corners=False,
        )  # [B, 12, target_length]
        result = result.transpose(1, 2)  # [B, target_length, 12]

    return result  # [B, T, 12]
