#!/usr/bin/env python3
"""
Ruta B: Improved TF-Constellations Extractor.

Mejoras sobre el extractor original:
1. Harmonic folding (octave-invariant): mapear frecuencias a pitch class
2. Anchor condicionado a onsets: solo anchors cerca de spectral flux peaks
3. IDF más agresivo: stoplist threshold más bajo
4. Hash más específico: pitch_class_anchor + octave_coarse

Based on GPT5.2Think specification.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
from collections import defaultdict, Counter


# Configuration
SR = 22050
HOP_LENGTH = 512
N_CQT_BINS = 84  # 7 octaves * 12 bins
BINS_PER_OCTAVE = 12
FMIN = 27.5  # A0

# Improved hash config
DT_BIN_SIZE = 2           # frames
LOG_RATIO_BIN_SIZE = 1/24  # ~50 cents (octave invariant)
PAIR_WINDOW_FRAMES = 86   # ~2.0s
FAN_OUT = 4
PEAKS_PER_FRAME_MAX = 8


def hz_to_midi(freq: float) -> int:
    """Convert frequency to MIDI pitch."""
    if freq <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def cqt_bin_to_hz(bin_idx: int, fmin: float = FMIN, bins_per_octave: int = BINS_PER_OCTAVE) -> float:
    """Convert CQT bin index to frequency."""
    return fmin * (2 ** (bin_idx / bins_per_octave))


def cqt_bin_to_pitch_class(bin_idx: int) -> int:
    """Convert CQT bin to pitch class (0-11), assuming fmin=A0."""
    # A0 = pitch class 9 (A), so bin 0 = pitch class 9
    return (9 + bin_idx) % 12


def fold_to_octave(freq: float, fmin: float = FMIN) -> float:
    """Fold frequency to base octave [fmin, 2*fmin)."""
    if freq <= 0:
        return fmin
    while freq >= 2 * fmin:
        freq /= 2
    while freq < fmin:
        freq *= 2
    return freq


@dataclass
class TFPeak:
    """A peak in the TF representation."""
    t_frame: int
    bin_idx: int
    freq: float
    amplitude: float
    pitch_class: int
    octave: int
    is_onset: bool = False  # True if near onset


@dataclass
class TFToken:
    """A constellation token from TF."""
    dt: int           # Delta time in frames
    log_ratio: float  # Log2(f_target / f_anchor)
    log_ratio_folded: float  # Folded to [-0.5, 0.5] (within octave)
    pc_anchor: int    # Pitch class of anchor (0-11)
    octave_anchor: int  # Octave of anchor
    weight: float
    anchor_is_onset: bool


def detect_onsets_spectral_flux(
    cqt_db: np.ndarray,
    threshold_percentile: float = 90,
) -> np.ndarray:
    """
    Detect onsets using spectral flux.

    Returns boolean array indicating onset frames.
    """
    # Compute spectral flux (positive half-wave rectified difference)
    diff = np.diff(cqt_db, axis=1)
    flux = np.maximum(0, diff).sum(axis=0)

    # Pad to match original length
    flux = np.concatenate([[0], flux])

    # Normalize
    if flux.max() > 0:
        flux = flux / flux.max()

    # Threshold
    threshold = np.percentile(flux, threshold_percentile)

    # Local maxima above threshold
    is_onset = np.zeros(len(flux), dtype=bool)
    for i in range(1, len(flux) - 1):
        if flux[i] > threshold and flux[i] > flux[i-1] and flux[i] > flux[i+1]:
            is_onset[i] = True

    return is_onset


def extract_peaks_with_onset_info(
    cqt: np.ndarray,
    is_onset: np.ndarray,
    peaks_per_frame_max: int = PEAKS_PER_FRAME_MAX,
    peak_threshold_rel: float = 0.2,
    onset_window: int = 3,  # frames around onset to mark as onset
) -> List[TFPeak]:
    """
    Extract peaks from CQT with onset information.

    Args:
        cqt: CQT magnitude array [n_bins, n_frames]
        is_onset: Boolean array of onset frames
        peaks_per_frame_max: Maximum peaks per frame
        peak_threshold_rel: Relative threshold (fraction of frame max)
        onset_window: Frames around onset to mark as onset

    Returns:
        List of TFPeak
    """
    n_bins, n_frames = cqt.shape
    peaks = []

    # Create onset mask (dilated)
    onset_mask = np.zeros(n_frames, dtype=bool)
    onset_indices = np.where(is_onset)[0]
    for idx in onset_indices:
        start = max(0, idx - onset_window)
        end = min(n_frames, idx + onset_window + 1)
        onset_mask[start:end] = True

    for t in range(n_frames):
        frame = cqt[:, t]
        frame_max = frame.max()

        if frame_max <= 0:
            continue

        threshold = frame_max * peak_threshold_rel

        # Find local maxima
        frame_peaks = []
        for b in range(1, n_bins - 1):
            if frame[b] >= threshold:
                if frame[b] > frame[b-1] and frame[b] > frame[b+1]:
                    freq = cqt_bin_to_hz(b)
                    pc = cqt_bin_to_pitch_class(b)
                    octave = b // BINS_PER_OCTAVE

                    frame_peaks.append(TFPeak(
                        t_frame=t,
                        bin_idx=b,
                        freq=freq,
                        amplitude=frame[b] / frame_max,
                        pitch_class=pc,
                        octave=octave,
                        is_onset=onset_mask[t]
                    ))

        # Sort by amplitude and keep top
        frame_peaks.sort(key=lambda p: -p.amplitude)
        peaks.extend(frame_peaks[:peaks_per_frame_max])

    return peaks


def build_tokens_improved(
    peaks: List[TFPeak],
    pair_window: int = PAIR_WINDOW_FRAMES,
    fan_out: int = FAN_OUT,
    require_onset_anchor: bool = True,  # Key improvement: only onset anchors
    diversity_split: bool = True,  # 50% close, 50% far
) -> List[TFToken]:
    """
    Build constellation tokens with improvements.

    Args:
        peaks: List of TFPeak
        pair_window: Maximum dt for pairs
        fan_out: Number of targets per anchor
        require_onset_anchor: If True, only use onset frames as anchors
        diversity_split: If True, split targets into close/far

    Returns:
        List of TFToken
    """
    if len(peaks) < 2:
        return []

    # Index peaks by frame
    by_frame = defaultdict(list)
    for p in peaks:
        by_frame[p.t_frame].append(p)

    tokens = []

    for anchor in peaks:
        # Skip non-onset anchors if required
        if require_onset_anchor and not anchor.is_onset:
            continue

        # Find targets in window
        candidates = []
        for dt in range(1, pair_window + 1):
            t_target = anchor.t_frame + dt
            if t_target in by_frame:
                for target in by_frame[t_target]:
                    candidates.append((dt, target))

        if not candidates:
            continue

        # Diversity: split into close and far (by pitch class distance)
        if diversity_split:
            close = []  # Same or adjacent pitch classes
            far = []    # Distant pitch classes

            for dt, target in candidates:
                pc_dist = min(
                    abs(target.pitch_class - anchor.pitch_class),
                    12 - abs(target.pitch_class - anchor.pitch_class)
                )
                if pc_dist <= 2:
                    close.append((dt, target))
                else:
                    far.append((dt, target))

            # Sort by amplitude
            close.sort(key=lambda x: -x[1].amplitude)
            far.sort(key=lambda x: -x[1].amplitude)

            # Select targets
            selected = close[:fan_out // 2] + far[:fan_out // 2]
        else:
            # Original: just sort by amplitude
            candidates.sort(key=lambda x: -x[1].amplitude)
            selected = candidates[:fan_out]

        # Build tokens
        for dt, target in selected:
            # Standard log ratio
            log_ratio = np.log2(target.freq / anchor.freq) if anchor.freq > 0 else 0

            # Folded log ratio (octave invariant): map to [-0.5, 0.5]
            log_ratio_folded = log_ratio - round(log_ratio)

            tokens.append(TFToken(
                dt=dt,
                log_ratio=log_ratio,
                log_ratio_folded=log_ratio_folded,
                pc_anchor=anchor.pitch_class,
                octave_anchor=anchor.octave,
                weight=np.sqrt(anchor.amplitude * target.amplitude),
                anchor_is_onset=anchor.is_onset
            ))

    return tokens


def token_to_hash_improved(token: TFToken, use_folded: bool = True) -> int:
    """
    Convert token to hash with improvements.

    Hash structure (16 bits):
    - dt_bin (6 bits): 0-63
    - log_ratio_bin (6 bits): 0-63 (folded or standard)
    - pc_anchor (4 bits): 0-15 (pitch class)

    Using folded ratio makes it octave-invariant.
    """
    dt_bin = min(token.dt // DT_BIN_SIZE, 63)

    if use_folded:
        # Folded: map [-0.5, 0.5] to [0, 24] bins (50 cents each)
        lr = token.log_ratio_folded
        lr_bin = int((lr + 0.5) / LOG_RATIO_BIN_SIZE) % 64
    else:
        # Standard: map to bins
        lr_bin = int(token.log_ratio / LOG_RATIO_BIN_SIZE) % 64

    h = (dt_bin << 10) | (lr_bin << 4) | (token.pc_anchor & 0xF)
    return h


def token_to_hash_extended(token: TFToken) -> int:
    """
    Extended hash with octave information (less octave-invariant but more specific).

    Hash structure (20 bits):
    - dt_bin (6 bits)
    - log_ratio_bin (6 bits)
    - pc_anchor (4 bits)
    - octave_anchor_coarse (4 bits): 0-7
    """
    dt_bin = min(token.dt // DT_BIN_SIZE, 63)
    lr_bin = int((token.log_ratio_folded + 0.5) / LOG_RATIO_BIN_SIZE) % 64
    octave = min(token.octave_anchor, 7)

    h = (dt_bin << 14) | (lr_bin << 8) | (token.pc_anchor << 4) | octave
    return h


# ============================================================================
# Full extraction pipeline
# ============================================================================

def extract_improved_tokens_from_audio(
    audio: np.ndarray,
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
    require_onset_anchor: bool = True,
) -> Tuple[List[TFToken], dict]:
    """
    Extract improved tokens from audio.

    Returns (tokens, metadata).
    """
    import librosa

    # Compute CQT
    cqt = np.abs(librosa.cqt(
        audio, sr=sr, hop_length=hop_length,
        fmin=FMIN, n_bins=N_CQT_BINS, bins_per_octave=BINS_PER_OCTAVE
    ))

    # Convert to dB for onset detection
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)

    # Detect onsets
    is_onset = detect_onsets_spectral_flux(cqt_db)

    # Extract peaks
    peaks = extract_peaks_with_onset_info(cqt, is_onset)

    # Build tokens
    tokens = build_tokens_improved(
        peaks,
        require_onset_anchor=require_onset_anchor
    )

    metadata = {
        'n_frames': cqt.shape[1],
        'n_peaks': len(peaks),
        'n_onsets': is_onset.sum(),
        'n_tokens': len(tokens),
        'tokens_per_frame': len(tokens) / max(cqt.shape[1], 1),
    }

    return tokens, metadata


def extract_improved_tokens_from_midi_tf(
    notes: List,
    duration: float,
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
    require_onset_anchor: bool = True,
) -> Tuple[List[TFToken], dict]:
    """
    Extract improved tokens from MIDI via pseudo-TF.

    Uses same pipeline as audio for comparability.
    """
    # Build pseudo-TF from MIDI notes
    frame_dt = hop_length / sr
    n_frames = int(duration / frame_dt) + 1

    # Synthesize pseudo-spectrogram
    pseudo_tf = np.zeros((N_CQT_BINS, n_frames), dtype=np.float32)

    # ADSR parameters
    attack_frames = int(0.01 / frame_dt)
    decay_frames = int(0.15 / frame_dt)
    sustain_level = 0.25
    release_frames = int(0.25 / frame_dt)

    n_harmonics = 6
    harmonic_decay = 1.3

    for note in notes:
        pitch = note.pitch
        onset_frame = int(note.onset / frame_dt)
        offset_frame = int(note.offset / frame_dt)
        velocity = note.velocity / 127.0

        # Fundamental frequency
        f0 = 440.0 * (2 ** ((pitch - 69) / 12.0))

        # Add harmonics
        for h in range(1, n_harmonics + 1):
            freq = f0 * h
            if freq > cqt_bin_to_hz(N_CQT_BINS - 1):
                break

            # Map to CQT bin
            bin_idx = int(round(BINS_PER_OCTAVE * np.log2(freq / FMIN)))
            if bin_idx < 0 or bin_idx >= N_CQT_BINS:
                continue

            amp = velocity / (h ** harmonic_decay)

            # Apply ADSR envelope
            for t in range(onset_frame, min(offset_frame + release_frames, n_frames)):
                dt_from_onset = t - onset_frame
                dt_from_offset = t - offset_frame

                if dt_from_onset < attack_frames:
                    env = dt_from_onset / max(attack_frames, 1)
                elif dt_from_onset < attack_frames + decay_frames:
                    env = 1.0 - (1.0 - sustain_level) * (dt_from_onset - attack_frames) / max(decay_frames, 1)
                elif t < offset_frame:
                    env = sustain_level
                else:
                    env = sustain_level * (1.0 - dt_from_offset / max(release_frames, 1))

                env = max(0, min(1, env))
                pseudo_tf[bin_idx, t] += amp * env

    # Apply frequency smoothing
    from scipy.ndimage import gaussian_filter1d
    pseudo_tf = gaussian_filter1d(pseudo_tf, sigma=1.0, axis=0)

    # Convert to dB
    pseudo_tf_db = 20 * np.log10(pseudo_tf + 1e-10)

    # Detect onsets (MIDI has clear onsets)
    is_onset = detect_onsets_spectral_flux(pseudo_tf_db)

    # Extract peaks
    peaks = extract_peaks_with_onset_info(pseudo_tf, is_onset)

    # Build tokens
    tokens = build_tokens_improved(
        peaks,
        require_onset_anchor=require_onset_anchor
    )

    metadata = {
        'n_frames': n_frames,
        'n_peaks': len(peaks),
        'n_onsets': is_onset.sum(),
        'n_tokens': len(tokens),
        'tokens_per_frame': len(tokens) / max(n_frames, 1),
    }

    return tokens, metadata


# ============================================================================
# Hash set operations for comparison
# ============================================================================

def tokens_to_hash_set(
    tokens: List[TFToken],
    use_folded: bool = True,
    use_extended: bool = False,
) -> Set[int]:
    """Convert tokens to set of hashes."""
    if use_extended:
        return {token_to_hash_extended(t) for t in tokens}
    else:
        return {token_to_hash_improved(t, use_folded=use_folded) for t in tokens}


def compute_overlap(set_a: Set[int], set_b: Set[int]) -> float:
    """Compute Jaccard-like overlap."""
    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def compute_idf(hash_sets: List[Set[int]]) -> dict:
    """Compute IDF weights for hashes."""
    n_docs = len(hash_sets)
    df = Counter()

    for hs in hash_sets:
        for h in hs:
            df[h] += 1

    idf = {}
    for h, doc_freq in df.items():
        idf[h] = np.log((n_docs + 1) / (doc_freq + 1)) + 1

    return idf


def filter_stoplist(
    hash_set: Set[int],
    df: Counter,
    n_docs: int,
    threshold: float = 0.3,  # Reduced from 0.5
) -> Set[int]:
    """Remove hashes that appear in too many documents."""
    max_df = int(n_docs * threshold)
    return {h for h in hash_set if df.get(h, 0) <= max_df}
