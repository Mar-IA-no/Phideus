"""
Gate 6 — A4 Descriptor Wrapper for 44.1kHz (Transkun regime)

Adapts the existing A4 descriptor (24kHz, n_fft=2048, hop=512)
to work at 44.1kHz with proportionally adjusted STFT parameters.

The frequency band edges don't change (they're defined in Hz),
but the STFT bin indices need recalculation for the new sample rate.

At 44.1kHz with n_fft=4096: freq_res = 44100/4096 ≈ 10.77 Hz/bin
"""

import torch
import torch.nn.functional as F

# ── Band edges at 44.1kHz, n_fft=4096 ────────────────────────────────────
# Same physical frequency ranges as A4, but different bin indices
# freq_res = 44100 / 4096 ≈ 10.77 Hz/bin

# Original bands (Hz):
#   0: 47-94,  1: 94-188,  2: 188-375,  3: 375-750
#   4: 750-1500, 5: 1500-3000, 6: 3000-6000, 7: 6000-12000

A4_44K_BAND_EDGES = [
    (4, 9),        # ~43-97 Hz
    (9, 17),       # ~97-183 Hz
    (17, 35),      # ~183-377 Hz
    (35, 70),      # ~377-754 Hz
    (70, 139),     # ~754-1498 Hz
    (139, 279),    # ~1498-3005 Hz
    (279, 557),    # ~3005-6001 Hz
    (557, 1114),   # ~6001-12003 Hz
]


def compute_a4_44k(
    audio: torch.Tensor,
    target_length: int = None,
    n_fft: int = 4096,
    hop_length: int = 1024,
) -> torch.Tensor:
    """Compute A4 descriptor from 44.1kHz audio.

    Same algorithm as compute_audio_descriptor_a4 but with
    STFT parameters adjusted for 44.1kHz:
      n_fft: 2048 → 4096 (same frequency resolution)
      hop: 512 → 1024 (same temporal resolution ~23ms/frame)

    Args:
        audio: [B, T_samples] raw waveform at 44.1kHz
        target_length: If set, interpolate output to this temporal length.
                       For Transkun: floor(segment_duration * 44100 / 1024).
        n_fft: STFT window size (4096 for ~10.8 Hz/bin)
        hop_length: STFT hop (1024 for ~23.2ms/frame)

    Returns:
        [B, T', 8] normalized temporal deltas per frequency band
    """
    B = audio.size(0)
    device = audio.device

    # Handle mono/stereo: ensure [B, T]
    if audio.dim() == 3:
        audio = audio.mean(dim=1)  # stereo → mono

    window = torch.hann_window(n_fft, device=device)
    stft_out = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window=window,
        center=True, return_complex=True,
    )
    magnitude = stft_out.abs()  # [B, n_fft//2+1, T_stft]

    log_mag = torch.log1p(magnitude)

    T_stft = log_mag.size(2)
    bands = []
    for lo, hi in A4_44K_BAND_EDGES:
        hi = min(hi, log_mag.size(1))
        band_mean = log_mag[:, lo:hi, :].mean(dim=1)
        bands.append(band_mean)
    banded = torch.stack(bands, dim=1)  # [B, 8, T_stft]

    # Temporal delta
    delta = banded[:, :, 1:] - banded[:, :, :-1]
    zero_pad = torch.zeros(B, 8, 1, device=device)
    delta = torch.cat([zero_pad, delta], dim=2)

    # Normalize per band per sample
    mean = delta.mean(dim=2, keepdim=True)
    std = delta.std(dim=2, keepdim=True).clamp(min=1e-8)
    delta = (delta - mean) / std

    # Interpolate to target length if specified
    if target_length is not None:
        delta = F.interpolate(
            delta, size=target_length, mode='linear', align_corners=False,
        )

    return delta.transpose(1, 2)  # [B, T', 8]


def compute_a4_for_transkun_frames(
    audio: torch.Tensor,
    n_transkun_frames: int,
) -> torch.Tensor:
    """Compute A4 aligned with Transkun's temporal frames.

    Transkun uses hop=1024 at 44.1kHz, giving ~23.2ms/frame.
    This function computes A4 and aligns it to the same frame grid.

    Args:
        audio: [B, T_samples] at 44.1kHz
        n_transkun_frames: Number of Transkun temporal frames

    Returns:
        [B, n_frames, 8]
    """
    return compute_a4_44k(audio, target_length=n_transkun_frames)
