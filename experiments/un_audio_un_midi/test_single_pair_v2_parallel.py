#!/usr/bin/env python3
"""
Test Single Audio-MIDI Pair - Version 2 PARALLEL
=================================================

Same as V2 but with parallel processing to use all CPU cores.

Parallelization strategy:
- CQT computation: librosa (already optimized)
- Peak picking: parallel across frame chunks
- Token generation: parallel across frame chunks
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import librosa
except ImportError:
    print("ERROR: librosa required")
    sys.exit(1)

from src.utils.midi_utils import parse_midi, Note

# Number of workers (leave 2 cores for system)
N_WORKERS = max(1, mp.cpu_count() - 2)


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL AUDIO EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

def _pick_peaks_chunk(args):
    """Worker function for parallel peak picking."""
    chunk_data, threshold_db, peaks_per_frame_max, freqs = args

    peaks_chunk = []
    for frame in chunk_data:
        peak_indices, _ = find_peaks(frame, height=threshold_db, distance=2)

        if len(peak_indices) == 0:
            peaks_chunk.append([])
            continue

        amplitudes = frame[peak_indices]
        sorted_idx = np.argsort(-amplitudes)[:peaks_per_frame_max]
        top_peaks = peak_indices[sorted_idx]

        peaks = []
        for idx in top_peaks:
            peaks.append({
                'bin': idx,
                'freq': freqs[idx],
                'amp': float((frame[idx] - threshold_db) / 40.0),  # Normalize
            })
        peaks_chunk.append(peaks)

    return peaks_chunk


def _generate_tokens_chunk(args):
    """Worker function for parallel token generation."""
    (start_frame, end_frame, peaks_per_frame, n_frames,
     fan_out, pair_time_window_frames, force_diversity,
     min_ratio, max_ratio, n_frequency_bands, n_cqt_bins, fmin) = args

    def get_band_id(freq):
        log_freq = np.log2(freq / fmin)
        band = int(log_freq / (n_cqt_bins / 12) * n_frequency_bands / 7)
        return min(max(band, 0), n_frequency_bands - 1)

    tokens_chunk = []
    stats = {'anchors': 0, 'targets': 0, 'close': 0, 'far': 0}

    for t in range(start_frame, end_frame):
        tokens = []
        anchors = peaks_per_frame[t]

        for anchor in anchors:
            anchor_freq = anchor['freq']
            anchor_amp = anchor['amp']
            anchor_band = get_band_id(anchor_freq)
            anchor_bin = anchor['bin']
            stats['anchors'] += 1

            close_candidates = []
            far_candidates = []

            for dt in range(1, min(pair_time_window_frames + 1, n_frames - t)):
                target_frame = t + dt

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
                        'log_ratio': log_ratio,
                        'target_band': get_band_id(target_freq),
                    }

                    if log_ratio < 1.0:
                        close_candidates.append(candidate)
                    else:
                        far_candidates.append(candidate)

            selected = []
            if force_diversity:
                n_close = fan_out // 2
                n_far = fan_out - n_close
                close_candidates.sort(key=lambda x: x['dt'])
                far_candidates.sort(key=lambda x: x['dt'])
                selected.extend(close_candidates[:n_close])
                selected.extend(far_candidates[:n_far])
                stats['close'] += len(close_candidates[:n_close])
                stats['far'] += len(far_candidates[:n_far])
            else:
                all_cands = close_candidates + far_candidates
                all_cands.sort(key=lambda x: x['dt'])
                selected = all_cands[:fan_out]

            stats['targets'] += len(selected)

            for target in selected:
                tokens.append([
                    target['log_ratio'],
                    float(target['dt']),
                    np.sqrt(anchor_amp * target['amp']),
                    float(anchor_band),
                    float(target['target_band']),
                ])

        if tokens:
            tokens_chunk.append(np.array(tokens, dtype=np.float32))
        else:
            tokens_chunk.append(np.zeros((0, 5), dtype=np.float32))

    return tokens_chunk, stats


def extract_audio_constellation_parallel(
    audio: np.ndarray,
    sr: int = 22050,
    n_cqt_bins: int = 84,
    hop_length: int = 512,
    peak_threshold_db: float = -40.0,
    peaks_per_frame_max: int = 8,
    fan_out: int = 4,
    pair_time_window_frames: int = 86,
    force_target_diversity: bool = True,
    n_frequency_bands: int = 16,
    min_ratio: float = 1.05,
    max_ratio: float = 4.0,
    n_workers: int = N_WORKERS,
) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
    """Parallel audio constellation extraction."""

    print(f"      Using {n_workers} workers")

    fmin = librosa.note_to_hz('C1')

    # CQT (already optimized by librosa)
    cqt = np.abs(librosa.cqt(
        y=audio, sr=sr, hop_length=hop_length,
        n_bins=n_cqt_bins, fmin=fmin,
    ))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    n_bins, n_frames = cqt.shape
    freqs = fmin * (2 ** (np.arange(n_bins) / 12))

    global_max_db = cqt_db.max()
    threshold_db = global_max_db + peak_threshold_db

    # Parallel peak picking
    chunk_size = max(100, n_frames // n_workers)
    chunks = []
    for i in range(0, n_frames, chunk_size):
        end = min(i + chunk_size, n_frames)
        chunks.append((cqt_db[:, i:end].T, threshold_db, peaks_per_frame_max, freqs))

    peaks_per_frame = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_pick_peaks_chunk, chunks))
        for chunk_peaks in results:
            peaks_per_frame.extend(chunk_peaks)

    # Parallel token generation (need shared peaks, so use threads)
    chunk_size = max(100, n_frames // n_workers)
    tasks = []
    for i in range(0, n_frames, chunk_size):
        end = min(i + chunk_size, n_frames)
        tasks.append((
            i, end, peaks_per_frame, n_frames,
            fan_out, pair_time_window_frames, force_target_diversity,
            min_ratio, max_ratio, n_frequency_bands, n_cqt_bins, fmin
        ))

    tokens_per_frame = []
    total_stats = {'total_anchors': 0, 'total_targets': 0, 'close_targets': 0, 'far_targets': 0}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_generate_tokens_chunk, tasks))
        for chunk_tokens, stats in results:
            tokens_per_frame.extend(chunk_tokens)
            total_stats['total_anchors'] += stats['anchors']
            total_stats['total_targets'] += stats['targets']
            total_stats['close_targets'] += stats['close']
            total_stats['far_targets'] += stats['far']

    frame_times = np.arange(n_frames) * hop_length / sr
    return tokens_per_frame, frame_times.astype(np.float32), total_stats


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL MIDI EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

def create_midi_pseudo_tf_fast(
    notes: List[Note],
    duration: float,
    sr: int = 22050,
    hop_length: int = 512,
    n_cqt_bins: int = 84,
    n_harmonics: int = 6,
) -> np.ndarray:
    """Fast pseudo-TF creation using vectorized operations."""

    fmin = librosa.note_to_hz('C1')
    n_frames = int(duration * sr / hop_length)
    frame_time = hop_length / sr

    tf = np.zeros((n_cqt_bins, n_frames), dtype=np.float32)

    def freq_to_bin(freq):
        if freq <= fmin:
            return 0
        bin_idx = int(round(12 * np.log2(freq / fmin)))
        return min(max(bin_idx, 0), n_cqt_bins - 1)

    # Simple envelope (faster than full ADSR)
    for note in notes:
        f0 = note.frequency
        velocity_amp = note.velocity / 127.0

        start_frame = max(0, int(note.onset / frame_time))
        end_frame = min(n_frames, int((note.offset + 0.1) / frame_time))

        if start_frame >= end_frame:
            continue

        # Add harmonics
        for h in range(1, n_harmonics + 1):
            harmonic_freq = f0 * h
            harmonic_amp = velocity_amp / h

            bin_idx = freq_to_bin(harmonic_freq)
            if bin_idx >= n_cqt_bins:
                continue

            # Simple decay envelope
            n_note_frames = end_frame - start_frame
            envelope = np.exp(-np.arange(n_note_frames) * frame_time * 2) * harmonic_amp

            tf[bin_idx, start_frame:end_frame] = np.maximum(
                tf[bin_idx, start_frame:end_frame],
                envelope
            )

    return tf


def extract_midi_constellation_parallel(
    notes: List[Note],
    duration: float,
    sr: int = 22050,
    hop_length: int = 512,
    n_cqt_bins: int = 84,
    n_harmonics: int = 6,
    peak_threshold: float = 0.1,
    peaks_per_frame_max: int = 8,
    fan_out: int = 4,
    pair_time_window_frames: int = 86,
    force_target_diversity: bool = True,
    n_frequency_bands: int = 16,
    min_ratio: float = 1.05,
    max_ratio: float = 4.0,
    n_workers: int = N_WORKERS,
) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
    """Parallel MIDI constellation extraction."""

    print(f"      Using {n_workers} workers")

    fmin = librosa.note_to_hz('C1')
    freqs = fmin * (2 ** (np.arange(n_cqt_bins) / 12))

    # Create pseudo-TF
    tf = create_midi_pseudo_tf_fast(notes, duration, sr, hop_length, n_cqt_bins, n_harmonics)
    n_bins, n_frames = tf.shape

    global_max = tf.max()
    if global_max == 0:
        return [np.zeros((0, 5), dtype=np.float32) for _ in range(n_frames)], \
               np.arange(n_frames) * hop_length / sr, \
               {'total_anchors': 0, 'total_targets': 0, 'close_targets': 0, 'far_targets': 0}

    threshold = peak_threshold * global_max

    # Parallel peak picking
    chunk_size = max(100, n_frames // n_workers)
    chunks = []
    for i in range(0, n_frames, chunk_size):
        end = min(i + chunk_size, n_frames)
        chunks.append((tf[:, i:end].T, threshold, peaks_per_frame_max, freqs))

    peaks_per_frame = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_pick_peaks_chunk, chunks))
        for chunk_peaks in results:
            peaks_per_frame.extend(chunk_peaks)

    # Parallel token generation
    chunk_size = max(100, n_frames // n_workers)
    tasks = []
    for i in range(0, n_frames, chunk_size):
        end = min(i + chunk_size, n_frames)
        tasks.append((
            i, end, peaks_per_frame, n_frames,
            fan_out, pair_time_window_frames, force_target_diversity,
            min_ratio, max_ratio, n_frequency_bands, n_cqt_bins, fmin
        ))

    tokens_per_frame = []
    total_stats = {'total_anchors': 0, 'total_targets': 0, 'close_targets': 0, 'far_targets': 0}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_generate_tokens_chunk, tasks))
        for chunk_tokens, stats in results:
            tokens_per_frame.extend(chunk_tokens)
            total_stats['total_anchors'] += stats['anchors']
            total_stats['total_targets'] += stats['targets']
            total_stats['close_targets'] += stats['close']
            total_stats['far_targets'] += stats['far']

    frame_times = np.arange(n_frames) * hop_length / sr
    return tokens_per_frame, frame_times.astype(np.float32), total_stats


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION AND COMPARISON (same as v2)
# ═══════════════════════════════════════════════════════════════════════════════

def compare_histograms(audio_tokens: list, midi_tokens: list) -> dict:
    audio_log_ratios = np.concatenate([t[:, 0] for t in audio_tokens if len(t) > 0])
    midi_log_ratios = np.concatenate([t[:, 0] for t in midi_tokens if len(t) > 0])

    bins = np.linspace(0, 2.5, 51)
    hist_audio, _ = np.histogram(audio_log_ratios, bins=bins, density=True)
    hist_midi, _ = np.histogram(midi_log_ratios, bins=bins, density=True)

    hist_audio = hist_audio / (hist_audio.sum() + 1e-8)
    hist_midi = hist_midi / (hist_midi.sum() + 1e-8)

    cosine = np.dot(hist_audio, hist_midi) / (
        np.linalg.norm(hist_audio) * np.linalg.norm(hist_midi) + 1e-8
    )
    intersection = np.minimum(hist_audio, hist_midi).sum()

    return {
        'cosine_similarity': float(cosine),
        'histogram_intersection': float(intersection),
        'n_audio_tokens': len(audio_log_ratios),
        'n_midi_tokens': len(midi_log_ratios),
        'token_ratio': len(audio_log_ratios) / max(len(midi_log_ratios), 1),
        'audio_mean_log_ratio': float(audio_log_ratios.mean()),
        'midi_mean_log_ratio': float(midi_log_ratios.mean()),
    }


def visualize_results(audio_tokens, midi_tokens, audio_times, midi_times, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    audio_lr = np.concatenate([t[:, 0] for t in audio_tokens if len(t) > 0])
    midi_lr = np.concatenate([t[:, 0] for t in midi_tokens if len(t) > 0])
    audio_dt = np.concatenate([t[:, 1] for t in audio_tokens if len(t) > 0])
    midi_dt = np.concatenate([t[:, 1] for t in midi_tokens if len(t) > 0])
    audio_w = np.concatenate([t[:, 2] for t in audio_tokens if len(t) > 0])
    midi_w = np.concatenate([t[:, 2] for t in midi_tokens if len(t) > 0])

    # Audio scatter
    ax = axes[0, 0]
    sc = ax.scatter(audio_dt, audio_lr, c=audio_w, alpha=0.3, s=1, cmap='viridis')
    ax.set_xlabel('delta_t'); ax.set_ylabel('log_ratio')
    ax.set_title(f'AUDIO (n={len(audio_lr):,})')
    plt.colorbar(sc, ax=ax)

    # Audio histogram
    ax = axes[0, 1]
    ax.hist(audio_lr, bins=50, alpha=0.7, color='blue', density=True)
    ax.axvline(np.log2(1.5), color='green', ls='--', alpha=0.5, label='fifth')
    ax.axvline(1.0, color='orange', ls='--', alpha=0.5, label='octave')
    ax.set_title('AUDIO log_ratio dist'); ax.legend()

    # Audio tokens/frame
    ax = axes[0, 2]
    tpf_a = [len(t) for t in audio_tokens]
    ax.plot(audio_times[:len(tpf_a)], tpf_a, 'b-', lw=0.5, alpha=0.7)
    ax.axhline(np.mean(tpf_a), color='red', ls='--')
    ax.set_title(f'AUDIO tokens/frame (avg={np.mean(tpf_a):.1f})')

    # MIDI scatter
    ax = axes[1, 0]
    sc = ax.scatter(midi_dt, midi_lr, c=midi_w, alpha=0.3, s=1, cmap='viridis')
    ax.set_xlabel('delta_t'); ax.set_ylabel('log_ratio')
    ax.set_title(f'MIDI (n={len(midi_lr):,})')
    plt.colorbar(sc, ax=ax)

    # MIDI histogram
    ax = axes[1, 1]
    ax.hist(midi_lr, bins=50, alpha=0.7, color='red', density=True)
    ax.axvline(np.log2(1.5), color='green', ls='--', alpha=0.5, label='fifth')
    ax.axvline(1.0, color='orange', ls='--', alpha=0.5, label='octave')
    ax.set_title('MIDI log_ratio dist'); ax.legend()

    # MIDI tokens/frame
    ax = axes[1, 2]
    tpf_m = [len(t) for t in midi_tokens]
    ax.plot(midi_times[:len(tpf_m)], tpf_m, 'r-', lw=0.5, alpha=0.7)
    ax.axhline(np.mean(tpf_m), color='blue', ls='--')
    ax.set_title(f'MIDI tokens/frame (avg={np.mean(tpf_m):.1f})')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    import time

    base_dir = Path(__file__).parent / "Un par"
    audio_path = base_dir / "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav"
    midi_path = base_dir / "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi"

    output_dir = base_dir / "02"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TEST V2 PARALLEL: Using all CPU cores")
    print(f"Workers: {N_WORKERS}")
    print("=" * 70)

    sr = 22050
    hop_length = 512
    n_cqt_bins = 84
    MAX_DURATION = 60.0

    # Load audio
    print(f"\n[1] Loading audio (first {MAX_DURATION:.0f}s)...")
    t0 = time.time()
    audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=MAX_DURATION)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio / rms * 0.1
    duration = len(audio) / sr
    print(f"    Duration: {duration:.1f}s, loaded in {time.time()-t0:.1f}s")

    # Audio extraction
    print("\n[2] Extracting AUDIO constellations (parallel)...")
    t0 = time.time()
    audio_tokens, audio_times, audio_stats = extract_audio_constellation_parallel(
        audio, sr=sr, hop_length=hop_length, n_cqt_bins=n_cqt_bins,
        n_workers=N_WORKERS,
    )
    n_audio = sum(len(t) for t in audio_tokens)
    print(f"    Tokens: {n_audio:,}, time: {time.time()-t0:.1f}s")

    # MIDI extraction
    print("\n[3] Extracting MIDI constellations (parallel)...")
    t0 = time.time()
    all_notes = parse_midi(midi_path)
    notes = [n for n in all_notes if n.onset < duration]
    print(f"    Notes: {len(notes)}")

    midi_tokens, midi_times, midi_stats = extract_midi_constellation_parallel(
        notes, duration=duration, sr=sr, hop_length=hop_length, n_cqt_bins=n_cqt_bins,
        n_workers=N_WORKERS,
    )
    n_midi = sum(len(t) for t in midi_tokens)
    print(f"    Tokens: {n_midi:,}, time: {time.time()-t0:.1f}s")

    # Compare
    print("\n[4] Comparing...")
    metrics = compare_histograms(audio_tokens, midi_tokens)

    print(f"\n    Cosine similarity:      {metrics['cosine_similarity']:.4f}")
    print(f"    Histogram intersection: {metrics['histogram_intersection']:.4f}")
    print(f"    Token ratio (A/M):      {metrics['token_ratio']:.2f}x")
    print(f"    Audio mean log_ratio:   {metrics['audio_mean_log_ratio']:.4f}")
    print(f"    MIDI mean log_ratio:    {metrics['midi_mean_log_ratio']:.4f}")

    # Visualize
    print("\n[5] Creating visualization...")
    visualize_results(audio_tokens, midi_tokens, audio_times, midi_times,
                     output_dir / "comparison_visualization.png")

    # Save
    import json
    metrics['audio_stats'] = audio_stats
    metrics['midi_stats'] = midi_stats
    with open(output_dir / "comparison_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
