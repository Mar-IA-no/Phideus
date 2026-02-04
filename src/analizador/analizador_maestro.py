#!/usr/bin/env python3
"""
Analizador MAESTRO - Constellation Extraction for Audio and MIDI
=================================================================

Extracts ratio constellation tokens from MAESTRO audio and MIDI pairs.
Adapted from analizador_roseta.py for the MAESTRO cross-modal experiment.

Features:
- Audio: CQT-based peak picking → constellation tokens
- MIDI: Note onsets as peaks → constellation tokens
- Aligned frame times for paired training

Token format: [log_ratio, delta_t, weight, anchor_band, target_band]

Usage:
------
python src/analizador/analizador_maestro.py \
    --data data/maestro_v3/processed \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/maestro_v3/constellations/tokens.npz \
    --workers 14
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import librosa
except ImportError:
    librosa = None

from src.utils.midi_utils import parse_midi, midi_to_constellation_tokens, filter_notes_by_time


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AUDIO CONSTELLATION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

# Default parameters (tuned for music)
DEFAULT_SR = 22050
DEFAULT_N_CQT_BINS = 84  # 7 octaves
DEFAULT_HOP_LENGTH = 512
DEFAULT_PEAK_THRESHOLD = 0.3
DEFAULT_MAX_ANCHORS = 16
DEFAULT_MAX_TARGETS_PER_ANCHOR = 4
DEFAULT_TARGET_ZONE_FRAMES = 5
DEFAULT_TARGET_ZONE_SEMITONES = 24  # 2 octaves
DEFAULT_N_FREQUENCY_BANDS = 16
DEFAULT_MIN_RATIO = 1.0
DEFAULT_MAX_RATIO = 4.0  # Narrower for music (mostly octaves and fifths)


def extract_audio_constellation(
    audio: np.ndarray,
    sr: int = DEFAULT_SR,
    n_cqt_bins: int = DEFAULT_N_CQT_BINS,
    hop_length: int = DEFAULT_HOP_LENGTH,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
    max_anchors: int = DEFAULT_MAX_ANCHORS,
    max_targets_per_anchor: int = DEFAULT_MAX_TARGETS_PER_ANCHOR,
    target_zone_frames: int = DEFAULT_TARGET_ZONE_FRAMES,
    target_zone_semitones: int = DEFAULT_TARGET_ZONE_SEMITONES,
    n_frequency_bands: int = DEFAULT_N_FREQUENCY_BANDS,
    min_ratio: float = DEFAULT_MIN_RATIO,
    max_ratio: float = DEFAULT_MAX_RATIO,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Extract constellation tokens from audio using CQT.

    Args:
        audio: Audio waveform
        sr: Sample rate
        n_cqt_bins: Number of CQT frequency bins
        hop_length: Hop length for CQT
        peak_threshold: Threshold for peak detection (relative to max)
        max_anchors: Maximum anchor peaks per frame
        max_targets_per_anchor: Maximum targets per anchor
        target_zone_frames: Temporal window for targets
        target_zone_semitones: Frequency window (in semitones)
        n_frequency_bands: Number of bands for band_id feature
        min_ratio, max_ratio: Ratio bounds

    Returns:
        tokens_per_frame: List of [n_tokens, 5] arrays
        frame_times: [T] time of each frame
    """
    if librosa is None:
        raise ImportError("librosa required for audio processing")

    # Compute CQT (better frequency resolution for music)
    fmin = librosa.note_to_hz('C1')  # ~32 Hz
    cqt = np.abs(librosa.cqt(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_cqt_bins,
        fmin=fmin,
    ))

    # Convert to log-amplitude
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    cqt_db = (cqt_db - cqt_db.min()) / (cqt_db.max() - cqt_db.min() + 1e-8)

    n_bins, n_frames = cqt.shape

    # CQT frequencies (log-spaced)
    freqs = fmin * (2 ** (np.arange(n_bins) / 12))  # Semitone spacing

    # Peak picking per frame
    peaks_per_frame = []
    for t in range(n_frames):
        frame = cqt_db[:, t]
        threshold = peak_threshold * frame.max()

        # Find peaks
        peak_indices, properties = find_peaks(frame, height=threshold, distance=2)

        if len(peak_indices) == 0:
            peaks_per_frame.append([])
            continue

        # Sort by amplitude and take top K
        amplitudes = frame[peak_indices]
        sorted_idx = np.argsort(-amplitudes)[:max_anchors]
        top_peaks = peak_indices[sorted_idx]

        peaks = []
        for idx in top_peaks:
            peaks.append({
                'bin': idx,
                'freq': freqs[idx],
                'amp': float(frame[idx]),
            })
        peaks_per_frame.append(peaks)

    # Generate constellation tokens
    tokens_per_frame = []
    target_zone_hz = fmin * (2 ** (target_zone_semitones / 12)) - fmin  # Convert semitones to Hz

    def get_band_id(freq: float) -> int:
        """Log-scale band assignment for music."""
        log_freq = np.log2(freq / fmin)
        band = int(log_freq / (n_cqt_bins / 12) * n_frequency_bands / 7)
        return min(max(band, 0), n_frequency_bands - 1)

    for t in range(n_frames):
        tokens = []
        anchors = peaks_per_frame[t]

        for anchor in anchors[:max_anchors]:
            anchor_freq = anchor['freq']
            anchor_amp = anchor['amp']
            anchor_band = get_band_id(anchor_freq)

            # Find targets in nearby frames
            target_candidates = []
            for dt in range(-target_zone_frames, target_zone_frames + 1):
                target_frame = t + dt
                if target_frame < 0 or target_frame >= n_frames:
                    continue

                for target in peaks_per_frame[target_frame]:
                    target_freq = target['freq']

                    # Skip if same peak
                    if dt == 0 and abs(target_freq - anchor_freq) < 1:
                        continue

                    # Check frequency zone
                    if abs(target_freq - anchor_freq) > target_zone_hz:
                        continue

                    # Calculate ratio
                    ratio = max(target_freq, anchor_freq) / min(target_freq, anchor_freq)
                    if ratio < min_ratio or ratio > max_ratio:
                        continue

                    # Distance metric
                    freq_dist = abs(np.log2(target_freq / anchor_freq))
                    time_dist = abs(dt) / max(target_zone_frames, 1)
                    distance = freq_dist + time_dist

                    target_candidates.append({
                        'freq': target_freq,
                        'amp': target['amp'],
                        'dt': dt,
                        'ratio': ratio,
                        'distance': distance,
                    })

            # Sort and select top M targets
            target_candidates.sort(key=lambda x: x['distance'])
            selected = target_candidates[:max_targets_per_anchor]

            # Generate tokens
            for target in selected:
                log_ratio = np.log2(target['ratio'])
                delta_t = float(target['dt'])
                weight = np.sqrt(anchor_amp * target['amp'])
                target_band = get_band_id(target['freq'])

                tokens.append([log_ratio, delta_t, weight, float(anchor_band), float(target_band)])

        if tokens:
            tokens_per_frame.append(np.array(tokens, dtype=np.float32))
        else:
            tokens_per_frame.append(np.zeros((0, 5), dtype=np.float32))

    frame_times = np.arange(n_frames) * hop_length / sr

    return tokens_per_frame, frame_times.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SEGMENT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_segment(args: Tuple) -> Optional[Dict]:
    """
    Process a single segment (worker function).

    Args:
        args: (segment_metadata, processed_dir, maestro_dir, params)

    Returns:
        Processed data dict or None
    """
    seg, processed_dir, maestro_dir, params = args
    processed_dir = Path(processed_dir)
    maestro_dir = Path(maestro_dir)

    try:
        # Load audio
        audio_path = processed_dir / seg['audio_path']
        audio = np.load(audio_path)

        sr = params.get('sr', DEFAULT_SR)
        hop_length = params.get('hop_length', DEFAULT_HOP_LENGTH)

        # Extract audio constellations
        audio_tokens, frame_times = extract_audio_constellation(
            audio=audio,
            sr=sr,
            hop_length=hop_length,
            n_cqt_bins=params.get('n_cqt_bins', DEFAULT_N_CQT_BINS),
            peak_threshold=params.get('peak_threshold', DEFAULT_PEAK_THRESHOLD),
            max_anchors=params.get('max_anchors', DEFAULT_MAX_ANCHORS),
            max_targets_per_anchor=params.get('max_targets_per_anchor', DEFAULT_MAX_TARGETS_PER_ANCHOR),
            target_zone_frames=params.get('target_zone_frames', DEFAULT_TARGET_ZONE_FRAMES),
            n_frequency_bands=params.get('n_frequency_bands', DEFAULT_N_FREQUENCY_BANDS),
            min_ratio=params.get('min_ratio', DEFAULT_MIN_RATIO),
            max_ratio=params.get('max_ratio', DEFAULT_MAX_RATIO),
        )

        # Extract MIDI constellations
        midi_path = maestro_dir / seg['midi_path']
        notes = parse_midi(midi_path)

        # Filter to segment time range
        start_time = seg['start_time']
        end_time = seg['end_time']
        seg_notes = filter_notes_by_time(notes, start_time, end_time)

        # Shift times
        for n in seg_notes:
            n.onset = max(0, n.onset - start_time)
            n.offset = max(n.onset + 0.01, n.offset - start_time)

        duration = seg.get('duration', end_time - start_time)
        frame_rate = sr / hop_length

        midi_tokens, _ = midi_to_constellation_tokens(
            notes=seg_notes,
            duration=duration,
            frame_rate=frame_rate,
            max_anchors=params.get('max_anchors', DEFAULT_MAX_ANCHORS),
            max_targets_per_anchor=params.get('max_targets_per_anchor', DEFAULT_MAX_TARGETS_PER_ANCHOR),
            target_zone_frames=params.get('target_zone_frames', DEFAULT_TARGET_ZONE_FRAMES),
            n_frequency_bands=params.get('n_frequency_bands', DEFAULT_N_FREQUENCY_BANDS),
            min_ratio=params.get('min_ratio', DEFAULT_MIN_RATIO),
            max_ratio=params.get('max_ratio', DEFAULT_MAX_RATIO),
        )

        # Ensure same length
        n_frames = min(len(audio_tokens), len(midi_tokens))
        audio_tokens = audio_tokens[:n_frames]
        midi_tokens = midi_tokens[:n_frames]
        frame_times = frame_times[:n_frames]

        return {
            'segment_id': seg['segment_id'],
            'piece_id': seg['piece_id'],
            'composer': seg['composer'],
            'split': seg['split'],
            'n_frames': n_frames,
            'audio_tokens': audio_tokens,
            'midi_tokens': midi_tokens,
            'frame_times': frame_times,
        }

    except Exception as e:
        print(f"Error processing {seg['segment_id']}: {e}")
        return None


def process_maestro_constellations(
    processed_dir: Path,
    maestro_dir: Path,
    output_path: Path,
    sr: int = DEFAULT_SR,
    hop_length: int = DEFAULT_HOP_LENGTH,
    max_anchors: int = DEFAULT_MAX_ANCHORS,
    max_targets_per_anchor: int = DEFAULT_MAX_TARGETS_PER_ANCHOR,
    workers: int = 14,
    max_segments: Optional[int] = None,
) -> Dict:
    """
    Process entire MAESTRO dataset for constellation extraction.

    Args:
        processed_dir: Path to processed segments
        maestro_dir: Path to original MAESTRO
        output_path: Output NPZ path
        workers: Number of parallel workers
        max_segments: Limit for testing

    Returns:
        Dataset metadata
    """
    processed_dir = Path(processed_dir)
    maestro_dir = Path(maestro_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting MAESTRO constellation tokens")
    print(f"  Source: {processed_dir}")
    print(f"  Output: {output_path}")
    print(f"  Workers: {workers}")

    # Load metadata
    with open(processed_dir / 'dataset_metadata.json') as f:
        metadata = json.load(f)

    segments = metadata['segments']
    if max_segments is not None:
        segments = segments[:max_segments]

    print(f"  Segments: {len(segments)}")

    # Parameters
    params = {
        'sr': sr,
        'hop_length': hop_length,
        'max_anchors': max_anchors,
        'max_targets_per_anchor': max_targets_per_anchor,
        'n_cqt_bins': DEFAULT_N_CQT_BINS,
        'peak_threshold': DEFAULT_PEAK_THRESHOLD,
        'target_zone_frames': DEFAULT_TARGET_ZONE_FRAMES,
        'n_frequency_bands': DEFAULT_N_FREQUENCY_BANDS,
        'min_ratio': DEFAULT_MIN_RATIO,
        'max_ratio': DEFAULT_MAX_RATIO,
    }

    max_tokens_per_frame = max_anchors * max_targets_per_anchor
    token_dim = 5

    # Process
    tasks = [(s, processed_dir, maestro_dir, params) for s in segments]

    results = []
    if workers == 1:
        for task in tqdm(tasks, desc="Processing"):
            result = process_segment(task)
            if result:
                results.append(result)
    else:
        with mp.Pool(workers) as pool:
            for result in tqdm(pool.imap(process_segment, tasks), total=len(tasks), desc="Processing"):
                if result:
                    results.append(result)

    # Save to NPZ
    segment_ids = []
    piece_ids = []
    composers = []
    splits = []
    arrays = {}

    for idx, r in enumerate(results):
        segment_ids.append(r['segment_id'])
        piece_ids.append(r['piece_id'])
        composers.append(r['composer'])
        splits.append(r['split'])

        n_frames = r['n_frames']
        audio_tokens = r['audio_tokens']
        midi_tokens = r['midi_tokens']

        # Pad to fixed size
        audio_padded = np.zeros((n_frames, max_tokens_per_frame, token_dim), dtype=np.float32)
        audio_mask = np.zeros((n_frames, max_tokens_per_frame), dtype=np.float32)
        midi_padded = np.zeros((n_frames, max_tokens_per_frame, token_dim), dtype=np.float32)
        midi_mask = np.zeros((n_frames, max_tokens_per_frame), dtype=np.float32)

        for t in range(n_frames):
            n_audio = min(len(audio_tokens[t]), max_tokens_per_frame)
            if n_audio > 0:
                audio_padded[t, :n_audio] = audio_tokens[t][:n_audio]
                audio_mask[t, :n_audio] = 1.0

            n_midi = min(len(midi_tokens[t]), max_tokens_per_frame)
            if n_midi > 0:
                midi_padded[t, :n_midi] = midi_tokens[t][:n_midi]
                midi_mask[t, :n_midi] = 1.0

        arrays[f'audio_tokens_{idx}'] = audio_padded
        arrays[f'audio_mask_{idx}'] = audio_mask
        arrays[f'midi_tokens_{idx}'] = midi_padded
        arrays[f'midi_mask_{idx}'] = midi_mask
        arrays[f'frame_times_{idx}'] = r['frame_times']

    # Dataset metadata
    dataset_meta = {
        'n_segments': len(results),
        'max_tokens_per_frame': max_tokens_per_frame,
        'token_dim': token_dim,
        'sr': sr,
        'hop_length': hop_length,
        'splits': {
            'train': sum(1 for s in splits if s == 'train'),
            'validation': sum(1 for s in splits if s == 'validation'),
            'test': sum(1 for s in splits if s == 'test'),
        },
    }

    np.savez_compressed(
        output_path,
        segment_ids=np.array(segment_ids, dtype=object),
        piece_ids=np.array(piece_ids, dtype=object),
        composers=np.array(composers, dtype=object),
        splits=np.array(splits, dtype=object),
        metadata=dataset_meta,
        n_segments=len(results),
        output_format='constellation',
        max_tokens_per_frame=max_tokens_per_frame,
        token_dim=token_dim,
        **arrays,
    )

    # Stats
    total_frames = sum(r['n_frames'] for r in results)
    total_audio_tokens = sum(sum(len(t) for t in r['audio_tokens']) for r in results)
    total_midi_tokens = sum(sum(len(t) for t in r['midi_tokens']) for r in results)

    print(f"\n✓ Constellation dataset saved: {output_path}")
    print(f"  Segments: {len(results)}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Avg tokens/frame: audio={total_audio_tokens/total_frames:.1f}, midi={total_midi_tokens/total_frames:.1f}")
    print(f"  Size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    return dataset_meta


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='MAESTRO Constellation Extraction')
    parser.add_argument('--data', type=Path, required=True,
                        help='Path to processed MAESTRO segments')
    parser.add_argument('--maestro-dir', type=Path, required=True,
                        help='Path to original MAESTRO')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output NPZ path')
    parser.add_argument('--sr', type=int, default=DEFAULT_SR)
    parser.add_argument('--hop-length', type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument('--max-anchors', type=int, default=DEFAULT_MAX_ANCHORS)
    parser.add_argument('--max-targets', type=int, default=DEFAULT_MAX_TARGETS_PER_ANCHOR)
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--max-segments', type=int, default=None,
                        help='Limit segments (for testing)')
    return parser.parse_args()


def main():
    args = parse_args()

    process_maestro_constellations(
        processed_dir=args.data,
        maestro_dir=args.maestro_dir,
        output_path=args.output,
        sr=args.sr,
        hop_length=args.hop_length,
        max_anchors=args.max_anchors,
        max_targets_per_anchor=args.max_targets,
        workers=args.workers,
        max_segments=args.max_segments,
    )


if __name__ == '__main__':
    main()
