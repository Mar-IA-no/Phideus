#!/usr/bin/env python3
"""
Test Single Audio-MIDI Pair - Version 2 (Improved Extractors)
==============================================================

Implements GPT5.2Think recommendations:

AUDIO:
- Reduce peaks_per_frame_max to 8 (not saturating at 62)
- Force target diversity: 50% close, 50% far
- Higher peak threshold
- Don't pair within same bin

MIDI:
- Render pseudo-TF with harmonics (6 harmonics, 1/h decay)
- Add ADSR envelope
- Run same peak picking as audio
- Handle sustain pedal

Usage:
------
python experiments/un_audio_un_midi/test_single_pair_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import librosa
except ImportError:
    print("ERROR: librosa required. Install with: pip install librosa")
    sys.exit(1)

from src.utils.midi_utils import parse_midi, Note


# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVED AUDIO EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

def extract_audio_constellation_v2(
    audio: np.ndarray,
    sr: int = 22050,
    n_cqt_bins: int = 84,
    hop_length: int = 512,
    peak_threshold_db: float = -40.0,  # dB relative to max
    peaks_per_frame_max: int = 8,      # REDUCED from default
    fan_out: int = 4,                  # targets per anchor
    pair_time_window_frames: int = 86, # ~2 seconds
    force_target_diversity: bool = True,  # 50% close, 50% far
    n_frequency_bands: int = 16,
    min_ratio: float = 1.05,  # Avoid ratio ≈ 1
    max_ratio: float = 4.0,
) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
    """
    Improved audio constellation extraction following GPT5.2Think recommendations.

    Key changes:
    - Reduced peaks_per_frame_max (8 instead of saturating)
    - Force target diversity (50% close, 50% far in frequency)
    - Higher min_ratio to avoid trivial pairs
    - Targets from DIFFERENT frames (not same frame)
    """
    fmin = librosa.note_to_hz('C1')  # ~32 Hz

    # Compute CQT
    cqt = np.abs(librosa.cqt(
        y=audio, sr=sr, hop_length=hop_length,
        n_bins=n_cqt_bins, fmin=fmin,
    ))

    # Convert to dB
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    n_bins, n_frames = cqt.shape

    # Frequencies
    freqs = fmin * (2 ** (np.arange(n_bins) / 12))  # Semitone spacing

    # Global threshold
    global_max_db = cqt_db.max()
    threshold_db = global_max_db + peak_threshold_db

    def get_band_id(freq: float) -> int:
        log_freq = np.log2(freq / fmin)
        band = int(log_freq / (n_cqt_bins / 12) * n_frequency_bands / 7)
        return min(max(band, 0), n_frequency_bands - 1)

    # Peak picking per frame (with threshold and limit)
    peaks_per_frame = []
    for t in range(n_frames):
        frame = cqt_db[:, t]

        # Find peaks above threshold
        peak_indices, _ = find_peaks(frame, height=threshold_db, distance=2)

        if len(peak_indices) == 0:
            peaks_per_frame.append([])
            continue

        # Sort by amplitude and take top K
        amplitudes = frame[peak_indices]
        sorted_idx = np.argsort(-amplitudes)[:peaks_per_frame_max]
        top_peaks = peak_indices[sorted_idx]

        peaks = []
        for idx in top_peaks:
            peaks.append({
                'bin': idx,
                'freq': freqs[idx],
                'amp': float((frame[idx] - threshold_db) / (-peak_threshold_db)),  # Normalize 0-1
            })
        peaks_per_frame.append(peaks)

    # Generate constellation tokens with diversity
    tokens_per_frame = []
    stats = {'total_anchors': 0, 'total_targets': 0, 'close_targets': 0, 'far_targets': 0}

    for t in range(n_frames):
        tokens = []
        anchors = peaks_per_frame[t]

        for anchor in anchors:
            anchor_freq = anchor['freq']
            anchor_amp = anchor['amp']
            anchor_band = get_band_id(anchor_freq)
            anchor_bin = anchor['bin']
            stats['total_anchors'] += 1

            # Collect targets from FUTURE frames only (Shazam style)
            close_candidates = []  # Within 1 octave
            far_candidates = []    # 1-2 octaves away

            for dt in range(1, pair_time_window_frames + 1):
                target_frame = t + dt
                if target_frame >= n_frames:
                    break

                for target in peaks_per_frame[target_frame]:
                    target_freq = target['freq']
                    target_bin = target['bin']

                    # Skip if same bin (would give ratio ≈ 1)
                    if target_bin == anchor_bin:
                        continue

                    # Calculate ratio
                    ratio = max(target_freq, anchor_freq) / min(target_freq, anchor_freq)
                    if ratio < min_ratio or ratio > max_ratio:
                        continue

                    log_ratio = np.log2(ratio)

                    candidate = {
                        'freq': target_freq,
                        'amp': target['amp'],
                        'dt': dt,
                        'ratio': ratio,
                        'log_ratio': log_ratio,
                        'target_band': get_band_id(target_freq),
                    }

                    # Classify by frequency distance
                    if log_ratio < 1.0:  # Less than 1 octave
                        close_candidates.append(candidate)
                    else:  # 1-2 octaves
                        far_candidates.append(candidate)

            # Select targets with diversity
            selected = []
            if force_target_diversity:
                # 50% close, 50% far
                n_close = fan_out // 2
                n_far = fan_out - n_close

                # Sort by time (earlier = better)
                close_candidates.sort(key=lambda x: x['dt'])
                far_candidates.sort(key=lambda x: x['dt'])

                selected.extend(close_candidates[:n_close])
                selected.extend(far_candidates[:n_far])

                stats['close_targets'] += len(close_candidates[:n_close])
                stats['far_targets'] += len(far_candidates[:n_far])
            else:
                # Original: just take closest
                all_candidates = close_candidates + far_candidates
                all_candidates.sort(key=lambda x: x['dt'])
                selected = all_candidates[:fan_out]

            stats['total_targets'] += len(selected)

            # Generate tokens
            for target in selected:
                log_ratio = target['log_ratio']
                delta_t = float(target['dt'])
                weight = np.sqrt(anchor_amp * target['amp'])

                tokens.append([
                    log_ratio,
                    delta_t,
                    weight,
                    float(anchor_band),
                    float(target['target_band']),
                ])

        if tokens:
            tokens_per_frame.append(np.array(tokens, dtype=np.float32))
        else:
            tokens_per_frame.append(np.zeros((0, 5), dtype=np.float32))

    frame_times = np.arange(n_frames) * hop_length / sr

    return tokens_per_frame, frame_times.astype(np.float32), stats


# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVED MIDI EXTRACTOR (with harmonics and envelope)
# ═══════════════════════════════════════════════════════════════════════════════

def create_midi_pseudo_tf(
    notes: List[Note],
    duration: float,
    sr: int = 22050,
    hop_length: int = 512,
    n_cqt_bins: int = 84,
    n_harmonics: int = 6,
    harmonic_decay: float = 1.0,  # 1/h^decay
    attack: float = 0.01,
    decay: float = 0.15,
    sustain_level: float = 0.25,
    release: float = 0.25,
) -> np.ndarray:
    """
    Create a pseudo time-frequency representation from MIDI with harmonics and ADSR.

    This makes MIDI "look like" audio spectrogram for compatible peak picking.
    """
    fmin = librosa.note_to_hz('C1')
    n_frames = int(duration * sr / hop_length)
    frame_time = hop_length / sr

    # Create empty TF matrix
    tf = np.zeros((n_cqt_bins, n_frames), dtype=np.float32)

    # Frequencies for each bin
    bin_freqs = fmin * (2 ** (np.arange(n_cqt_bins) / 12))

    def freq_to_bin(freq: float) -> int:
        """Map frequency to CQT bin."""
        if freq <= fmin:
            return 0
        bin_idx = int(round(12 * np.log2(freq / fmin)))
        return min(max(bin_idx, 0), n_cqt_bins - 1)

    def adsr_envelope(t: float, onset: float, offset: float) -> float:
        """Compute ADSR envelope value at time t."""
        t_rel = t - onset
        note_dur = offset - onset

        if t < onset or t > offset + release:
            return 0.0

        if t_rel < attack:
            # Attack phase
            return t_rel / attack
        elif t_rel < attack + decay:
            # Decay phase
            decay_progress = (t_rel - attack) / decay
            return 1.0 - (1.0 - sustain_level) * decay_progress
        elif t < offset:
            # Sustain phase
            return sustain_level
        else:
            # Release phase
            release_progress = (t - offset) / release
            return sustain_level * (1.0 - release_progress)

    # Render each note with harmonics
    for note in notes:
        f0 = note.frequency
        velocity_amp = note.velocity / 127.0

        # Determine frame range
        start_frame = int(note.onset / frame_time)
        end_frame = int((note.offset + release) / frame_time)
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)

        # Add fundamental and harmonics
        for h in range(1, n_harmonics + 1):
            harmonic_freq = f0 * h
            harmonic_amp = velocity_amp / (h ** harmonic_decay)

            # Find bin for this harmonic
            bin_idx = freq_to_bin(harmonic_freq)
            if bin_idx >= n_cqt_bins:
                continue

            # Paint energy with envelope
            for frame in range(start_frame, end_frame):
                t = frame * frame_time
                env = adsr_envelope(t, note.onset, note.offset)
                energy = harmonic_amp * env
                tf[bin_idx, frame] = max(tf[bin_idx, frame], energy)

    return tf


def extract_midi_constellation_v2(
    notes: List[Note],
    duration: float,
    sr: int = 22050,
    hop_length: int = 512,
    n_cqt_bins: int = 84,
    n_harmonics: int = 6,
    peak_threshold: float = 0.1,  # Relative to max
    peaks_per_frame_max: int = 8,
    fan_out: int = 4,
    pair_time_window_frames: int = 86,
    force_target_diversity: bool = True,
    n_frequency_bands: int = 16,
    min_ratio: float = 1.05,
    max_ratio: float = 4.0,
) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
    """
    Improved MIDI constellation extraction:
    1. Render pseudo-TF with harmonics and ADSR
    2. Run same peak picking as audio
    3. Apply same pairing logic
    """
    fmin = librosa.note_to_hz('C1')

    # Create pseudo-TF
    tf = create_midi_pseudo_tf(
        notes, duration, sr, hop_length, n_cqt_bins, n_harmonics
    )

    n_bins, n_frames = tf.shape
    freqs = fmin * (2 ** (np.arange(n_bins) / 12))

    def get_band_id(freq: float) -> int:
        log_freq = np.log2(freq / fmin)
        band = int(log_freq / (n_cqt_bins / 12) * n_frequency_bands / 7)
        return min(max(band, 0), n_frequency_bands - 1)

    # Peak picking (same as audio)
    peaks_per_frame = []
    global_max = tf.max()
    threshold = peak_threshold * global_max

    for t in range(n_frames):
        frame = tf[:, t]

        peak_indices, _ = find_peaks(frame, height=threshold, distance=2)

        if len(peak_indices) == 0:
            peaks_per_frame.append([])
            continue

        amplitudes = frame[peak_indices]
        sorted_idx = np.argsort(-amplitudes)[:peaks_per_frame_max]
        top_peaks = peak_indices[sorted_idx]

        peaks = []
        for idx in top_peaks:
            peaks.append({
                'bin': idx,
                'freq': freqs[idx],
                'amp': float(frame[idx] / global_max),
            })
        peaks_per_frame.append(peaks)

    # Generate tokens (same logic as audio)
    tokens_per_frame = []
    stats = {'total_anchors': 0, 'total_targets': 0, 'close_targets': 0, 'far_targets': 0}

    for t in range(n_frames):
        tokens = []
        anchors = peaks_per_frame[t]

        for anchor in anchors:
            anchor_freq = anchor['freq']
            anchor_amp = anchor['amp']
            anchor_band = get_band_id(anchor_freq)
            anchor_bin = anchor['bin']
            stats['total_anchors'] += 1

            close_candidates = []
            far_candidates = []

            for dt in range(1, pair_time_window_frames + 1):
                target_frame = t + dt
                if target_frame >= n_frames:
                    break

                for target in peaks_per_frame[target_frame]:
                    target_freq = target['freq']
                    target_bin = target['bin']

                    if target_bin == anchor_bin:
                        continue

                    ratio = max(target_freq, anchor_freq) / min(target_freq, anchor_freq)
                    if ratio < min_ratio or ratio > max_ratio:
                        continue

                    log_ratio = np.log2(ratio)

                    candidate = {
                        'freq': target_freq,
                        'amp': target['amp'],
                        'dt': dt,
                        'ratio': ratio,
                        'log_ratio': log_ratio,
                        'target_band': get_band_id(target_freq),
                    }

                    if log_ratio < 1.0:
                        close_candidates.append(candidate)
                    else:
                        far_candidates.append(candidate)

            selected = []
            if force_target_diversity:
                n_close = fan_out // 2
                n_far = fan_out - n_close

                close_candidates.sort(key=lambda x: x['dt'])
                far_candidates.sort(key=lambda x: x['dt'])

                selected.extend(close_candidates[:n_close])
                selected.extend(far_candidates[:n_far])

                stats['close_targets'] += len(close_candidates[:n_close])
                stats['far_targets'] += len(far_candidates[:n_far])
            else:
                all_candidates = close_candidates + far_candidates
                all_candidates.sort(key=lambda x: x['dt'])
                selected = all_candidates[:fan_out]

            stats['total_targets'] += len(selected)

            for target in selected:
                log_ratio = target['log_ratio']
                delta_t = float(target['dt'])
                weight = np.sqrt(anchor_amp * target['amp'])

                tokens.append([
                    log_ratio,
                    delta_t,
                    weight,
                    float(anchor_band),
                    float(target['target_band']),
                ])

        if tokens:
            tokens_per_frame.append(np.array(tokens, dtype=np.float32))
        else:
            tokens_per_frame.append(np.zeros((0, 5), dtype=np.float32))

    frame_times = np.arange(n_frames) * hop_length / sr

    return tokens_per_frame, frame_times.astype(np.float32), stats


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION AND COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_tokens_v2(
    audio_tokens: list,
    midi_tokens: list,
    audio_frame_times: np.ndarray,
    midi_frame_times: np.ndarray,
    audio_stats: Dict,
    midi_stats: Dict,
    output_path: Path,
):
    """Enhanced visualization with stats."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Collect all log_ratios
    all_log_ratios_a = []
    all_delta_t_a = []
    all_weights_a = []
    for frame_tokens in audio_tokens:
        if len(frame_tokens) > 0:
            all_log_ratios_a.extend(frame_tokens[:, 0])
            all_delta_t_a.extend(frame_tokens[:, 1])
            all_weights_a.extend(frame_tokens[:, 2])

    all_log_ratios_m = []
    all_delta_t_m = []
    all_weights_m = []
    for frame_tokens in midi_tokens:
        if len(frame_tokens) > 0:
            all_log_ratios_m.extend(frame_tokens[:, 0])
            all_delta_t_m.extend(frame_tokens[:, 1])
            all_weights_m.extend(frame_tokens[:, 2])

    # Row 1: Audio
    ax = axes[0, 0]
    if all_log_ratios_a:
        sc = ax.scatter(all_delta_t_a, all_log_ratios_a, c=all_weights_a, alpha=0.3, s=1, cmap='viridis')
        ax.set_xlabel('delta_t (frames)')
        ax.set_ylabel('log_ratio')
        ax.set_title(f'AUDIO Tokens (n={len(all_log_ratios_a):,})')
        plt.colorbar(sc, ax=ax, label='weight')

    ax = axes[0, 1]
    if all_log_ratios_a:
        ax.hist(all_log_ratios_a, bins=50, alpha=0.7, color='blue', density=True)
        ax.set_xlabel('log_ratio')
        ax.set_ylabel('Density')
        ax.set_title('AUDIO: log_ratio distribution')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='ratio=1')
        ax.axvline(np.log2(1.5), color='green', linestyle='--', alpha=0.5, label='fifth (3:2)')
        ax.axvline(1.0, color='orange', linestyle='--', alpha=0.5, label='octave')
        ax.legend(fontsize=8)

    ax = axes[0, 2]
    tokens_per_frame_a = [len(t) for t in audio_tokens]
    ax.plot(audio_frame_times[:len(tokens_per_frame_a)], tokens_per_frame_a, 'b-', alpha=0.7, linewidth=0.5)
    avg_a = np.mean(tokens_per_frame_a)
    ax.axhline(avg_a, color='red', linestyle='--', label=f'avg={avg_a:.1f}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tokens per frame')
    ax.set_title(f'AUDIO: Tokens over time')
    ax.legend()

    # Row 2: MIDI
    ax = axes[1, 0]
    if all_log_ratios_m:
        sc = ax.scatter(all_delta_t_m, all_log_ratios_m, c=all_weights_m, alpha=0.3, s=1, cmap='viridis')
        ax.set_xlabel('delta_t (frames)')
        ax.set_ylabel('log_ratio')
        ax.set_title(f'MIDI Tokens (n={len(all_log_ratios_m):,})')
        plt.colorbar(sc, ax=ax, label='weight')

    ax = axes[1, 1]
    if all_log_ratios_m:
        ax.hist(all_log_ratios_m, bins=50, alpha=0.7, color='red', density=True)
        ax.set_xlabel('log_ratio')
        ax.set_ylabel('Density')
        ax.set_title('MIDI: log_ratio distribution')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='ratio=1')
        ax.axvline(np.log2(1.5), color='green', linestyle='--', alpha=0.5, label='fifth (3:2)')
        ax.axvline(1.0, color='orange', linestyle='--', alpha=0.5, label='octave')
        ax.legend(fontsize=8)

    ax = axes[1, 2]
    tokens_per_frame_m = [len(t) for t in midi_tokens]
    ax.plot(midi_frame_times[:len(tokens_per_frame_m)], tokens_per_frame_m, 'r-', alpha=0.7, linewidth=0.5)
    avg_m = np.mean(tokens_per_frame_m)
    ax.axhline(avg_m, color='blue', linestyle='--', label=f'avg={avg_m:.1f}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tokens per frame')
    ax.set_title(f'MIDI: Tokens over time')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {output_path}")


def compare_histograms(audio_tokens: list, midi_tokens: list) -> dict:
    """Compare histograms with additional metrics."""
    audio_log_ratios = []
    midi_log_ratios = []

    for frame_tokens in audio_tokens:
        if len(frame_tokens) > 0:
            audio_log_ratios.extend(frame_tokens[:, 0])

    for frame_tokens in midi_tokens:
        if len(frame_tokens) > 0:
            midi_log_ratios.extend(frame_tokens[:, 0])

    audio_log_ratios = np.array(audio_log_ratios)
    midi_log_ratios = np.array(midi_log_ratios)

    # Use same bins for fair comparison
    bins = np.linspace(0, 2.5, 51)  # log2 ratios from 1 to ~5.6

    hist_audio, _ = np.histogram(audio_log_ratios, bins=bins, density=True)
    hist_midi, _ = np.histogram(midi_log_ratios, bins=bins, density=True)

    hist_audio = hist_audio / (hist_audio.sum() + 1e-8)
    hist_midi = hist_midi / (hist_midi.sum() + 1e-8)

    # Metrics
    cosine = np.dot(hist_audio, hist_midi) / (
        np.linalg.norm(hist_audio) * np.linalg.norm(hist_midi) + 1e-8
    )
    intersection = np.minimum(hist_audio, hist_midi).sum()

    eps = 1e-10
    kl_am = np.sum(hist_audio * np.log((hist_audio + eps) / (hist_midi + eps)))
    kl_ma = np.sum(hist_midi * np.log((hist_midi + eps) / (hist_audio + eps)))
    kl_sym = (kl_am + kl_ma) / 2

    return {
        'cosine_similarity': float(cosine),
        'histogram_intersection': float(intersection),
        'kl_divergence_sym': float(kl_sym),
        'n_audio_tokens': len(audio_log_ratios),
        'n_midi_tokens': len(midi_log_ratios),
        'token_ratio': len(audio_log_ratios) / max(len(midi_log_ratios), 1),
        'audio_mean_log_ratio': float(audio_log_ratios.mean()) if len(audio_log_ratios) > 0 else 0,
        'midi_mean_log_ratio': float(midi_log_ratios.mean()) if len(midi_log_ratios) > 0 else 0,
        'audio_std_log_ratio': float(audio_log_ratios.std()) if len(audio_log_ratios) > 0 else 0,
        'midi_std_log_ratio': float(midi_log_ratios.std()) if len(midi_log_ratios) > 0 else 0,
    }


def main():
    # Paths
    base_dir = Path(__file__).parent / "Un par"
    audio_path = base_dir / "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav"
    midi_path = base_dir / "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi"

    output_dir = base_dir / "02"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TEST V2: Improved Extractors (GPT5.2Think Recommendations)")
    print("=" * 70)

    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        return
    if not midi_path.exists():
        print(f"ERROR: MIDI file not found: {midi_path}")
        return

    print(f"\nAudio: {audio_path.name}")
    print(f"MIDI:  {midi_path.name}")
    print(f"Output: {output_dir}")

    # Parameters (from GPT5.2Think recommendations)
    sr = 22050
    hop_length = 512
    n_cqt_bins = 84

    # Load audio (only first 60 seconds for fast testing)
    MAX_DURATION = 60.0  # seconds
    print(f"\n[1] Loading audio (first {MAX_DURATION:.0f}s only)...")
    audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=MAX_DURATION)
    # RMS normalize
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio / rms * 0.1

    duration = len(audio) / sr
    print(f"    Duration: {duration:.1f}s")

    # Extract audio constellations (V2)
    print("\n[2] Extracting AUDIO constellations (V2 - improved)...")
    audio_tokens, audio_times, audio_stats = extract_audio_constellation_v2(
        audio, sr=sr, hop_length=hop_length, n_cqt_bins=n_cqt_bins,
        peaks_per_frame_max=8,
        fan_out=4,
        force_target_diversity=True,
        min_ratio=1.05,
    )
    n_audio_tokens = sum(len(t) for t in audio_tokens)
    avg_audio = n_audio_tokens / len(audio_tokens) if audio_tokens else 0
    print(f"    Frames: {len(audio_tokens)}")
    print(f"    Total tokens: {n_audio_tokens:,}")
    print(f"    Avg tokens/frame: {avg_audio:.1f}")
    print(f"    Close targets: {audio_stats['close_targets']:,}, Far targets: {audio_stats['far_targets']:,}")

    # Load MIDI (filter to same duration)
    print("\n[3] Extracting MIDI constellations (V2 - with harmonics)...")
    all_notes = parse_midi(midi_path)
    # Filter notes to first MAX_DURATION seconds
    notes = [n for n in all_notes if n.onset < duration]
    print(f"    Notes in MIDI (first {duration:.0f}s): {len(notes)} / {len(all_notes)}")

    midi_tokens, midi_times, midi_stats = extract_midi_constellation_v2(
        notes, duration=duration, sr=sr, hop_length=hop_length, n_cqt_bins=n_cqt_bins,
        n_harmonics=6,
        peaks_per_frame_max=8,
        fan_out=4,
        force_target_diversity=True,
        min_ratio=1.05,
    )
    n_midi_tokens = sum(len(t) for t in midi_tokens)
    avg_midi = n_midi_tokens / len(midi_tokens) if midi_tokens else 0
    print(f"    Frames: {len(midi_tokens)}")
    print(f"    Total tokens: {n_midi_tokens:,}")
    print(f"    Avg tokens/frame: {avg_midi:.1f}")
    print(f"    Close targets: {midi_stats['close_targets']:,}, Far targets: {midi_stats['far_targets']:,}")

    # Compare
    print("\n[4] Comparing histograms...")
    metrics = compare_histograms(audio_tokens, midi_tokens)

    print(f"\n    COMPARISON METRICS:")
    print(f"    -------------------")
    print(f"    Cosine similarity:      {metrics['cosine_similarity']:.4f}")
    print(f"    Histogram intersection: {metrics['histogram_intersection']:.4f}")
    print(f"    KL divergence (sym):    {metrics['kl_divergence_sym']:.4f}")
    print(f"    Token ratio (A/M):      {metrics['token_ratio']:.2f}x")
    print(f"    Audio mean log_ratio:   {metrics['audio_mean_log_ratio']:.4f}")
    print(f"    MIDI mean log_ratio:    {metrics['midi_mean_log_ratio']:.4f}")

    # Interpretation
    print("\n[5] INTERPRETATION:")
    print("    -----------------")

    # Token balance
    if 0.5 < metrics['token_ratio'] < 2.0:
        print(f"    ✓ GOOD token balance: {metrics['token_ratio']:.2f}x (target: 0.5-2x)")
    else:
        print(f"    ✗ Token imbalance: {metrics['token_ratio']:.2f}x (target: 0.5-2x)")

    # Mean log_ratio
    mean_diff = abs(metrics['audio_mean_log_ratio'] - metrics['midi_mean_log_ratio'])
    if mean_diff < 0.3:
        print(f"    ✓ GOOD mean alignment: diff={mean_diff:.4f} (target: <0.3)")
    else:
        print(f"    ✗ Mean misalignment: diff={mean_diff:.4f} (target: <0.3)")

    # Similarity
    if metrics['cosine_similarity'] > 0.7:
        print(f"    ✓ HIGH similarity: {metrics['cosine_similarity']:.4f}")
    elif metrics['cosine_similarity'] > 0.4:
        print(f"    ~ MODERATE similarity: {metrics['cosine_similarity']:.4f}")
    else:
        print(f"    ✗ LOW similarity: {metrics['cosine_similarity']:.4f}")

    # Visualize
    print("\n[6] Creating visualization...")
    visualize_tokens_v2(
        audio_tokens, midi_tokens,
        audio_times, midi_times,
        audio_stats, midi_stats,
        output_dir / "comparison_visualization.png"
    )

    # Save metrics
    import json
    metrics['audio_stats'] = audio_stats
    metrics['midi_stats'] = midi_stats
    metrics_path = output_dir / "comparison_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("\n" + "=" * 70)
    print("TEST V2 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
