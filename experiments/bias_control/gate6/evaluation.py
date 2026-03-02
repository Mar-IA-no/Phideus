"""
Gate 6 — Common Evaluation Framework for AMT experiments.

Wraps mir_eval with fixed conventions for consistent evaluation across
Exp 0 (baseline), Exp A (Transkun+A4), Exp B (degraded), and Exp C (VICReg decoder).

Conventions (fixed before measuring):
  - Onset tolerance: 50ms
  - Offset tolerance: 50ms or 20% of duration (whichever is greater)
  - Pedal extension: No Ext (default Transkun convention)
  - Note clipping: Notes crossing segment borders are clipped
  - Velocity bins: 128 (standard MIDI)
  - Evaluation: Over MIDI events decoded by Semi-CRF, NOT piano roll frame-level
"""

import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mir_eval
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────

ONSET_TOLERANCE = 0.05          # 50ms
OFFSET_RATIO = 0.2              # 20% of note duration
OFFSET_MIN_TOLERANCE = 0.05     # 50ms minimum

# Frame-level evaluation
FRAME_HOP_SEC = 0.01            # 10ms frame hop (mir_eval standard)
MIN_MIDI_PITCH = 21             # A0
MAX_MIDI_PITCH = 108            # C8
N_PITCHES = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1  # 88


# ── Note-level metrics (mir_eval) ──────────────────────────────────────────

def evaluate_note_level(
    est_intervals: np.ndarray,
    est_pitches: np.ndarray,
    est_velocities: np.ndarray,
    ref_intervals: np.ndarray,
    ref_pitches: np.ndarray,
    ref_velocities: np.ndarray,
) -> Dict[str, float]:
    """Evaluate note-level AMT metrics using mir_eval.

    Args:
        est_intervals: [N, 2] estimated note (onset, offset) in seconds
        est_pitches: [N] estimated pitches in Hz
        est_velocities: [N] estimated velocities (0-127)
        ref_intervals: [M, 2] reference note (onset, offset)
        ref_pitches: [M] reference pitches in Hz
        ref_velocities: [M] reference velocities (0-127)

    Returns:
        Dict with P/R/F1 for note, note+offset, note+offset+velocity
    """
    results = {}

    # Handle empty predictions or references
    if len(est_intervals) == 0:
        est_intervals = np.zeros((0, 2))
        est_pitches = np.zeros(0)
        est_velocities = np.zeros(0)
    if len(ref_intervals) == 0:
        ref_intervals = np.zeros((0, 2))
        ref_pitches = np.zeros(0)
        ref_velocities = np.zeros(0)

    # Note Onset only
    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=ONSET_TOLERANCE,
        offset_ratio=None,
    )
    results['note_onset_P'] = p
    results['note_onset_R'] = r
    results['note_onset_F1'] = f1

    # Note + Offset
    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=ONSET_TOLERANCE,
        offset_ratio=OFFSET_RATIO,
        offset_min_tolerance=OFFSET_MIN_TOLERANCE,
    )
    results['note_offset_P'] = p
    results['note_offset_R'] = r
    results['note_offset_F1'] = f1

    # Note + Offset + Velocity
    p, r, f1, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, ref_velocities,
        est_intervals, est_pitches, est_velocities,
        onset_tolerance=ONSET_TOLERANCE,
        offset_ratio=OFFSET_RATIO,
        offset_min_tolerance=OFFSET_MIN_TOLERANCE,
    )
    results['note_offset_vel_P'] = p
    results['note_offset_vel_R'] = r
    results['note_offset_vel_F1'] = f1

    results['n_ref'] = len(ref_intervals)
    results['n_est'] = len(est_intervals)

    return results


# ── Frame-level metrics ────────────────────────────────────────────────────

def notes_to_piano_roll(
    notes: List,
    duration_sec: float,
    hop_sec: float = FRAME_HOP_SEC,
    min_pitch: int = MIN_MIDI_PITCH,
    max_pitch: int = MAX_MIDI_PITCH,
) -> np.ndarray:
    """Convert note list to binary piano roll [T, 88].

    Args:
        notes: List of (onset_sec, offset_sec, midi_pitch, velocity)
              or objects with .start, .end, .pitch attributes
        duration_sec: total duration in seconds
        hop_sec: frame hop size in seconds
        min_pitch: lowest MIDI pitch
        max_pitch: highest MIDI pitch

    Returns:
        Binary piano roll [T, n_pitches]
    """
    n_frames = int(np.ceil(duration_sec / hop_sec))
    n_pitches = max_pitch - min_pitch + 1
    pr = np.zeros((n_frames, n_pitches), dtype=np.float32)

    for note in notes:
        if hasattr(note, 'start'):
            onset, offset, pitch = note.start, note.end, note.pitch
        else:
            onset, offset, pitch = note[0], note[1], note[2]

        if pitch < min_pitch or pitch > max_pitch:
            continue

        frame_start = int(np.floor(onset / hop_sec))
        frame_end = int(np.ceil(offset / hop_sec))
        frame_start = max(0, frame_start)
        frame_end = min(n_frames, frame_end)

        if frame_start < frame_end:
            pr[frame_start:frame_end, pitch - min_pitch] = 1.0

    return pr


def evaluate_frame_level(
    est_pr: np.ndarray,
    ref_pr: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate frame-level piano roll metrics.

    Args:
        est_pr: [T, 88] estimated piano roll (or probabilities)
        ref_pr: [T, 88] reference binary piano roll
        threshold: binarization threshold for est_pr

    Returns:
        Dict with frame_P, frame_R, frame_F1
    """
    est_binary = (est_pr >= threshold).astype(np.float32)
    ref_binary = ref_pr.astype(np.float32)

    # Ensure same length
    min_len = min(len(est_binary), len(ref_binary))
    est_binary = est_binary[:min_len]
    ref_binary = ref_binary[:min_len]

    tp = (est_binary * ref_binary).sum()
    fp = (est_binary * (1 - ref_binary)).sum()
    fn = ((1 - est_binary) * ref_binary).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'frame_P': float(precision),
        'frame_R': float(recall),
        'frame_F1': float(f1),
        'frame_TP': int(tp),
        'frame_FP': int(fp),
        'frame_FN': int(fn),
    }


def evaluate_onset_level(
    est_pr: np.ndarray,
    ref_pr: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate onset-level accuracy from piano rolls.

    Onsets are detected as positive-going edges in the piano roll.

    Args:
        est_pr: [T, 88] estimated piano roll
        ref_pr: [T, 88] reference piano roll

    Returns:
        Dict with onset_P, onset_R, onset_F1
    """
    est_binary = (est_pr >= threshold).astype(np.float32)
    ref_binary = ref_pr.astype(np.float32)

    min_len = min(len(est_binary), len(ref_binary))
    est_binary = est_binary[:min_len]
    ref_binary = ref_binary[:min_len]

    # Detect onsets: positive edge
    est_onset = np.zeros_like(est_binary)
    ref_onset = np.zeros_like(ref_binary)
    est_onset[0] = est_binary[0]
    ref_onset[0] = ref_binary[0]
    est_onset[1:] = np.maximum(0, est_binary[1:] - est_binary[:-1])
    ref_onset[1:] = np.maximum(0, ref_binary[1:] - ref_binary[:-1])

    tp = (est_onset * ref_onset).sum()
    fp = (est_onset * (1 - ref_onset)).sum()
    fn = ((1 - est_onset) * ref_onset).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'onset_P': float(precision),
        'onset_R': float(recall),
        'onset_F1': float(f1),
    }


# ── Transkun output → mir_eval format conversion ──────────────────────────

def transkun_notes_to_mir_eval(
    notes: List,
    segment_start: float = 0.0,
    segment_end: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Transkun Note objects to mir_eval format.

    Clips notes to segment boundaries.

    Returns:
        (intervals [N, 2], pitches_hz [N], velocities [N])
    """
    intervals = []
    pitches_hz = []
    velocities = []

    for note in notes:
        # Skip pedal events (negative pitch)
        if note.pitch < 0:
            continue

        onset = note.start
        offset = note.end

        # Clip to segment boundaries
        if segment_end is not None:
            onset = max(onset, segment_start)
            offset = min(offset, segment_end)

        if offset <= onset:
            continue

        intervals.append([onset - segment_start, offset - segment_start])
        pitches_hz.append(mir_eval.util.midi_to_hz(note.pitch))
        velocities.append(note.velocity)

    if len(intervals) == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)

    return (
        np.array(intervals),
        np.array(pitches_hz),
        np.array(velocities),
    )


def midi_to_mir_eval(
    midi_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a MIDI file and extract note intervals/pitches/velocities.

    Clips notes to [start_time, end_time].

    Returns:
        (intervals [N, 2], pitches_hz [N], velocities [N])
    """
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(midi_path))

    intervals = []
    pitches_hz = []
    velocities = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            onset = note.start
            offset = note.end

            # Clip to segment boundaries
            if end_time is not None:
                if onset >= end_time or offset <= start_time:
                    continue
                onset = max(onset, start_time)
                offset = min(offset, end_time)

            if offset <= onset:
                continue

            intervals.append([onset - start_time, offset - start_time])
            pitches_hz.append(mir_eval.util.midi_to_hz(note.pitch))
            velocities.append(note.velocity)

    if len(intervals) == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)

    return (
        np.array(intervals),
        np.array(pitches_hz),
        np.array(velocities),
    )


# ── Aggregate metrics ──────────────────────────────────────────────────────

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics from multiple segments.

    Computes micro-averaged P/R/F1 using TP/FP/FN counts where available,
    otherwise uses macro averaging.
    """
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    agg = {}

    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if not values:
            continue
        agg[f'{key}_mean'] = float(np.mean(values))
        agg[f'{key}_std'] = float(np.std(values))

    return agg


# ── Result serialization ──────────────────────────────────────────────────

def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd='/mnt/m2-1TB/Phideus',
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def save_results(
    results: Dict,
    output_path: Path,
    experiment: str,
    config: Optional[Dict] = None,
    seed: int = 42,
) -> Path:
    """Save experiment results as JSON with metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'results': results,
        '_metadata': {
            'experiment': experiment,
            'config': config or {},
            'seed': seed,
            'git_hash': _get_git_hash(),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        },
    }

    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x))

    return output_path
