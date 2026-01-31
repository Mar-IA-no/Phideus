"""
Pytest fixtures and utilities for testing the ratio extraction pipeline.

This module provides:
- Functions to generate synthetic signals with known ratios
- Functions to extract ratios from signals (wrapping analizador_roseta)
- Metrics for validating extraction quality (precision, recall, entropy)
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analizador.analizador_roseta import (
    process_single_channel,
    compute_enriched_histogram,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_N_FFT,
    DEFAULT_HOP_LENGTH,
    DEFAULT_PEAK_THRESHOLD_FACTOR,
    DEFAULT_LOCAL_MEDIAN_WINDOW,
    DEFAULT_REL_PEAK_TOL,
    DEFAULT_MIN_RATIO,
    DEFAULT_MAX_RATIO,
    DEFAULT_N_RATIO_BINS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def generate_harmonic_signal(
    fundamental: float,
    harmonics: List[int],
    sr: int = DEFAULT_SAMPLE_RATE,
    duration: float = 1.0,
    amplitudes: List[float] | None = None,
) -> np.ndarray:
    """
    Generate a signal with a fundamental frequency and specified harmonics.

    Args:
        fundamental: Fundamental frequency in Hz
        harmonics: List of harmonic numbers (e.g., [1, 2, 3, 4, 5] for 1st through 5th)
        sr: Sample rate
        duration: Duration in seconds
        amplitudes: Amplitude for each harmonic (default: equal amplitudes)

    Returns:
        Signal as numpy array

    Example:
        # Generate 200 Hz with harmonics at 400, 600, 800, 1000 Hz
        signal = generate_harmonic_signal(200, [1, 2, 3, 4, 5])
    """
    t = np.arange(int(sr * duration)) / sr

    if amplitudes is None:
        amplitudes = [1.0] * len(harmonics)

    signal = np.zeros_like(t)
    for h, amp in zip(harmonics, amplitudes):
        freq = fundamental * h
        signal += amp * np.sin(2 * np.pi * freq * t)

    # Normalize to [-1, 1]
    signal = signal / (np.abs(signal).max() + 1e-12)
    return signal.astype(np.float32)


def generate_multitone_signal(
    frequencies: List[float],
    sr: int = DEFAULT_SAMPLE_RATE,
    duration: float = 1.0,
    amplitudes: List[float] | None = None,
) -> np.ndarray:
    """
    Generate a signal with arbitrary frequencies (not necessarily harmonic).

    Args:
        frequencies: List of frequencies in Hz
        sr: Sample rate
        duration: Duration in seconds
        amplitudes: Amplitude for each frequency (default: equal)

    Returns:
        Signal as numpy array
    """
    t = np.arange(int(sr * duration)) / sr

    if amplitudes is None:
        amplitudes = [1.0] * len(frequencies)

    signal = np.zeros_like(t)
    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)

    signal = signal / (np.abs(signal).max() + 1e-12)
    return signal.astype(np.float32)


def generate_noise_signal(
    sr: int = DEFAULT_SAMPLE_RATE,
    duration: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate white Gaussian noise.

    Args:
        sr: Sample rate
        duration: Duration in seconds
        seed: Random seed for reproducibility

    Returns:
        Noise signal as numpy array
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = int(sr * duration)
    noise = np.random.randn(n_samples)
    noise = noise / (np.abs(noise).max() + 1e-12)
    return noise.astype(np.float32)


def add_noise_to_signal(
    signal: np.ndarray,
    snr_db: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add white Gaussian noise to a signal at specified SNR.

    Args:
        signal: Clean signal
        snr_db: Signal-to-noise ratio in dB
        seed: Random seed for reproducibility

    Returns:
        Noisy signal
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate signal power
    signal_power = np.mean(signal ** 2)

    # Calculate required noise power for desired SNR
    # SNR_dB = 10 * log10(signal_power / noise_power)
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Generate noise with required power
    noise = np.random.randn(len(signal)) * np.sqrt(noise_power)

    noisy_signal = signal + noise
    noisy_signal = noisy_signal / (np.abs(noisy_signal).max() + 1e-12)
    return noisy_signal.astype(np.float32)


def generate_intermittent_signal(
    frequencies: List[float],
    sr: int = DEFAULT_SAMPLE_RATE,
    duration: float = 1.0,
    on_probability: float = 0.5,
    segment_duration: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a signal where tones appear/disappear randomly.

    Args:
        frequencies: List of frequencies in Hz
        sr: Sample rate
        duration: Total duration in seconds
        on_probability: Probability that a tone is "on" in each segment
        segment_duration: Duration of each on/off segment
        seed: Random seed

    Returns:
        Signal with intermittent tones
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = int(sr * duration)
    segment_samples = int(sr * segment_duration)
    n_segments = n_samples // segment_samples

    signal = np.zeros(n_samples, dtype=np.float32)
    t = np.arange(n_samples) / sr

    for freq in frequencies:
        tone = np.sin(2 * np.pi * freq * t)

        # Create on/off mask
        for seg_idx in range(n_segments):
            if np.random.random() > on_probability:
                start = seg_idx * segment_samples
                end = min(start + segment_samples, n_samples)
                tone[start:end] = 0

        signal += tone

    signal = signal / (np.abs(signal).max() + 1e-12)
    return signal


# ═══════════════════════════════════════════════════════════════════════════════
# RATIO EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def extract_ratios_from_signal(
    signal: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    peak_thr_factor: float = DEFAULT_PEAK_THRESHOLD_FACTOR,
    local_window: int = DEFAULT_LOCAL_MEDIAN_WINDOW,
    rel_peak_tol: float = DEFAULT_REL_PEAK_TOL,
    max_band_hz: float | None = None,
    min_ratio: float = DEFAULT_MIN_RATIO,
    max_ratio: float = DEFAULT_MAX_RATIO,
    n_ratio_bins: int = DEFAULT_N_RATIO_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ratio histograms from a numpy signal.

    This is a wrapper around process_single_channel for testing.

    Args:
        signal: Input signal as numpy array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop between frames
        peak_thr_factor: Peak detection threshold factor
        local_window: Window for local median
        rel_peak_tol: Relative tolerance for peak deduplication
        max_band_hz: Maximum frequency to consider
        min_ratio: Minimum ratio
        max_ratio: Maximum ratio
        n_ratio_bins: Number of bins in ratio histogram

    Returns:
        Tuple of:
        - enriched_frames: [n_frames, n_bins, 3] histograms
        - frame_times: [n_frames] timestamps
    """
    # Normalize signal (same as analizador_roseta does for CSV data)
    signal = (signal - signal.mean()) / (signal.std() + 1e-12)

    return process_single_channel(
        signal=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        peak_thr_factor=peak_thr_factor,
        local_window=local_window,
        rel_peak_tol=rel_peak_tol,
        max_band_hz=max_band_hz,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        n_ratio_bins=n_ratio_bins,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════


def ratio_to_bin(
    ratio: float,
    min_ratio: float = DEFAULT_MIN_RATIO,
    max_ratio: float = DEFAULT_MAX_RATIO,
    n_bins: int = DEFAULT_N_RATIO_BINS,
) -> int:
    """Convert a ratio value to its bin index."""
    if ratio < min_ratio or ratio > max_ratio:
        return -1
    bin_width = (max_ratio - min_ratio) / n_bins
    return int((ratio - min_ratio) / bin_width)


def calculate_histogram_entropy(histogram: np.ndarray) -> float:
    """
    Calculate normalized entropy of a histogram.

    Args:
        histogram: Histogram array (can be [n_bins] or [n_frames, n_bins, 3])

    Returns:
        Normalized entropy (0 = all mass in one bin, 1 = uniform)
    """
    # If multi-dimensional, use channel 0 (proportion) and average over frames
    if histogram.ndim == 3:
        # [n_frames, n_bins, 3] -> use channel 0
        hist = histogram[:, :, 0].mean(axis=0)
    elif histogram.ndim == 2:
        # [n_bins, 3] -> use channel 0
        hist = histogram[:, 0]
    else:
        hist = histogram

    # Normalize to probability distribution
    total = hist.sum()
    if total <= 0:
        return 1.0  # Empty histogram = maximum entropy (undefined)

    p = hist / total
    p = p[p > 0]  # Remove zeros for log

    # Calculate entropy
    entropy = -np.sum(p * np.log(p))

    # Maximum entropy (uniform distribution)
    n_bins = len(hist)
    max_entropy = np.log(n_bins)

    return entropy / max_entropy if max_entropy > 0 else 1.0


def calculate_ratio_recall(
    histogram: np.ndarray,
    expected_ratios: List[float],
    tolerance_bins: int = 3,
    threshold: float = 0.01,
    min_ratio: float = DEFAULT_MIN_RATIO,
    max_ratio: float = DEFAULT_MAX_RATIO,
) -> float:
    """
    Calculate recall: proportion of expected ratios that were detected.

    Args:
        histogram: Ratio histogram [n_frames, n_bins, 3] or [n_bins, 3]
        expected_ratios: List of ratios that should be present
        tolerance_bins: Number of bins tolerance for matching
        threshold: Minimum mass in bin to count as "detected"
        min_ratio, max_ratio: Ratio range

    Returns:
        Recall (0 to 1)
    """
    # Average over frames if needed
    if histogram.ndim == 3:
        hist = histogram[:, :, 0].mean(axis=0)
    else:
        hist = histogram[:, 0] if histogram.ndim == 2 else histogram

    n_bins = len(hist)
    detected = 0

    for ratio in expected_ratios:
        target_bin = ratio_to_bin(ratio, min_ratio, max_ratio, n_bins)
        if target_bin < 0:
            continue

        # Check target bin and neighbors
        lo = max(0, target_bin - tolerance_bins)
        hi = min(n_bins, target_bin + tolerance_bins + 1)

        if hist[lo:hi].max() >= threshold:
            detected += 1

    return detected / len(expected_ratios) if expected_ratios else 0.0


def calculate_ratio_precision(
    histogram: np.ndarray,
    expected_ratios: List[float],
    tolerance_bins: int = 3,
    min_ratio: float = DEFAULT_MIN_RATIO,
    max_ratio: float = DEFAULT_MAX_RATIO,
) -> float:
    """
    Calculate precision: proportion of histogram mass in expected ratio bins.

    Args:
        histogram: Ratio histogram
        expected_ratios: List of expected ratios
        tolerance_bins: Number of bins tolerance
        min_ratio, max_ratio: Ratio range

    Returns:
        Precision (0 to 1)
    """
    if histogram.ndim == 3:
        hist = histogram[:, :, 0].mean(axis=0)
    else:
        hist = histogram[:, 0] if histogram.ndim == 2 else histogram

    n_bins = len(hist)
    total_mass = hist.sum()
    if total_mass <= 0:
        return 0.0

    # Create mask of expected bins
    expected_mask = np.zeros(n_bins, dtype=bool)
    for ratio in expected_ratios:
        target_bin = ratio_to_bin(ratio, min_ratio, max_ratio, n_bins)
        if target_bin < 0:
            continue
        lo = max(0, target_bin - tolerance_bins)
        hi = min(n_bins, target_bin + tolerance_bins + 1)
        expected_mask[lo:hi] = True

    mass_in_expected = hist[expected_mask].sum()
    return mass_in_expected / total_mass


def calculate_frame_variance(histogram: np.ndarray) -> float:
    """
    Calculate variance across frames (measures temporal stability).

    Args:
        histogram: [n_frames, n_bins, 3]

    Returns:
        Mean variance across bins
    """
    if histogram.ndim != 3:
        return 0.0

    # Use channel 0 (proportion)
    frames = histogram[:, :, 0]
    variance_per_bin = np.var(frames, axis=0)
    return float(variance_per_bin.mean())


# ═══════════════════════════════════════════════════════════════════════════════
# PYTEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def default_sr():
    """Default sample rate for tests."""
    return DEFAULT_SAMPLE_RATE


@pytest.fixture
def harmonic_signal():
    """Generate a clean harmonic signal for testing."""
    return generate_harmonic_signal(
        fundamental=200.0,
        harmonics=[1, 2, 3, 4, 5],
        sr=DEFAULT_SAMPLE_RATE,
        duration=1.0,
    )


@pytest.fixture
def noise_signal():
    """Generate pure noise for testing."""
    return generate_noise_signal(
        sr=DEFAULT_SAMPLE_RATE,
        duration=1.0,
        seed=42,
    )
