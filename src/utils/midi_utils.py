#!/usr/bin/env python3
"""
MIDI Utilities for MAESTRO Dataset
====================================

Functions for parsing MIDI files and extracting note information.
Designed for MAESTRO v3.0.0 piano recordings.

Usage:
------
from src.utils.midi_utils import parse_midi, notes_to_piano_roll, Note

notes = parse_midi('path/to/file.midi')
piano_roll = notes_to_piano_roll(notes, duration=4.0, frame_rate=50)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import pretty_midi
except ImportError:
    pretty_midi = None
    warnings.warn("pretty_midi not installed. Install with: pip install pretty_midi")


@dataclass
class Note:
    """Represents a single MIDI note event."""
    pitch: int           # MIDI pitch (0-127, piano: 21-108)
    onset: float         # Start time in seconds
    offset: float        # End time in seconds
    velocity: int        # MIDI velocity (0-127)

    @property
    def duration(self) -> float:
        """Note duration in seconds."""
        return self.offset - self.onset

    @property
    def frequency(self) -> float:
        """Frequency in Hz (A4 = 440 Hz, MIDI 69)."""
        return 440.0 * (2.0 ** ((self.pitch - 69) / 12.0))

    @property
    def pitch_class(self) -> int:
        """Pitch class (0-11, C=0, C#=1, ..., B=11)."""
        return self.pitch % 12


def parse_midi(path: str | Path) -> List[Note]:
    """
    Parse a MIDI file and extract all note events.

    Args:
        path: Path to MIDI file

    Returns:
        List of Note objects sorted by onset time
    """
    if pretty_midi is None:
        raise ImportError("pretty_midi required. Install with: pip install pretty_midi")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    midi = pretty_midi.PrettyMIDI(str(path))
    notes = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append(Note(
                pitch=note.pitch,
                onset=note.start,
                offset=note.end,
                velocity=note.velocity,
            ))

    # Sort by onset time
    notes.sort(key=lambda n: (n.onset, n.pitch))

    return notes


def notes_to_piano_roll(
    notes: List[Note],
    duration: float,
    frame_rate: float = 50.0,
    velocity_weighted: bool = True,
    piano_range: bool = True,
) -> np.ndarray:
    """
    Convert note list to piano roll representation.

    Args:
        notes: List of Note objects
        duration: Total duration in seconds
        frame_rate: Frames per second
        velocity_weighted: If True, use velocity as intensity; else binary
        piano_range: If True, use 88-key piano range (21-108); else full 128

    Returns:
        piano_roll: [T, P] where T = frames, P = pitches (88 or 128)
    """
    n_frames = int(duration * frame_rate)

    if piano_range:
        min_pitch, max_pitch = 21, 108  # A0 to C8
        n_pitches = max_pitch - min_pitch + 1  # 88 keys
    else:
        min_pitch, max_pitch = 0, 127
        n_pitches = 128

    piano_roll = np.zeros((n_frames, n_pitches), dtype=np.float32)

    for note in notes:
        if note.pitch < min_pitch or note.pitch > max_pitch:
            continue

        pitch_idx = note.pitch - min_pitch
        start_frame = int(note.onset * frame_rate)
        end_frame = int(note.offset * frame_rate)

        # Clamp to valid range
        start_frame = max(0, min(start_frame, n_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, n_frames))

        if velocity_weighted:
            intensity = note.velocity / 127.0
        else:
            intensity = 1.0

        piano_roll[start_frame:end_frame, pitch_idx] = np.maximum(
            piano_roll[start_frame:end_frame, pitch_idx],
            intensity
        )

    return piano_roll


def notes_to_pitch_histogram(
    notes: List[Note],
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    weighted: bool = True,
) -> np.ndarray:
    """
    Compute pitch class histogram from notes in a time window.

    Args:
        notes: List of Note objects
        start_time: Window start in seconds
        end_time: Window end in seconds (None = all notes)
        weighted: If True, weight by duration * velocity

    Returns:
        histogram: [12] pitch class histogram (normalized)
    """
    histogram = np.zeros(12, dtype=np.float32)

    for note in notes:
        # Check if note overlaps with window
        if end_time is not None:
            if note.offset <= start_time or note.onset >= end_time:
                continue

        pitch_class = note.pitch_class

        if weighted:
            # Weight by note duration and velocity
            if end_time is not None:
                overlap_start = max(note.onset, start_time)
                overlap_end = min(note.offset, end_time)
                weight = (overlap_end - overlap_start) * (note.velocity / 127.0)
            else:
                weight = note.duration * (note.velocity / 127.0)
        else:
            weight = 1.0

        histogram[pitch_class] += weight

    # Normalize
    total = histogram.sum()
    if total > 0:
        histogram /= total

    return histogram


def get_note_density(
    notes: List[Note],
    duration: float,
    frame_rate: float = 50.0,
) -> np.ndarray:
    """
    Compute note density (onset count) per frame.

    Args:
        notes: List of Note objects
        duration: Total duration in seconds
        frame_rate: Frames per second

    Returns:
        density: [T] note density per frame
    """
    n_frames = int(duration * frame_rate)
    density = np.zeros(n_frames, dtype=np.float32)

    for note in notes:
        frame_idx = int(note.onset * frame_rate)
        if 0 <= frame_idx < n_frames:
            density[frame_idx] += 1

    return density


def midi_to_constellation_tokens(
    notes: List[Note],
    duration: float,
    frame_rate: float = 50.0,
    max_anchors: int = 16,
    max_targets_per_anchor: int = 4,
    target_zone_frames: int = 3,
    target_zone_hz: float = 2000.0,
    n_frequency_bands: int = 16,
    min_ratio: float = 1.0,
    max_ratio: float = 6.0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Extract constellation tokens from MIDI notes.

    Treats each note onset as a "peak" at (time, frequency).
    Generates anchor-target pairs similar to audio constellation.

    Token format: [log_ratio, delta_t, weight, anchor_band, target_band]

    Args:
        notes: List of Note objects
        duration: Total duration in seconds
        frame_rate: Frames per second
        max_anchors: Max anchor notes per frame
        max_targets_per_anchor: Max targets per anchor
        target_zone_frames: Temporal window for targets
        target_zone_hz: Frequency window for targets
        n_frequency_bands: Number of frequency bands
        min_ratio, max_ratio: Ratio bounds

    Returns:
        tokens_per_frame: List of [n_tokens, 5] arrays
        frame_times: [T] time of each frame
    """
    n_frames = int(duration * frame_rate)
    frame_time = 1.0 / frame_rate

    # Group notes by frame
    notes_by_frame = [[] for _ in range(n_frames)]
    for note in notes:
        frame_idx = int(note.onset * frame_rate)
        if 0 <= frame_idx < n_frames:
            notes_by_frame[frame_idx].append(note)

    # Sort notes in each frame by velocity (descending)
    for frame_notes in notes_by_frame:
        frame_notes.sort(key=lambda n: n.velocity, reverse=True)

    def get_band_id(freq: float) -> int:
        """Assign frequency to band ID using log-scale (better for music)."""
        min_freq = 27.5  # A0
        max_freq = 4186.0  # C8
        if freq <= min_freq:
            return 0
        log_freq = np.log2(max(freq, min_freq))
        log_min = np.log2(min_freq)
        log_max = np.log2(max_freq)
        band = int((log_freq - log_min) / (log_max - log_min) * n_frequency_bands)
        return min(max(band, 0), n_frequency_bands - 1)

    tokens_per_frame = []

    for t, frame_notes in enumerate(notes_by_frame):
        tokens = []

        # Select anchors (top by velocity)
        anchors = frame_notes[:max_anchors]

        for anchor in anchors:
            anchor_freq = anchor.frequency
            anchor_weight = anchor.velocity / 127.0
            anchor_band = get_band_id(anchor_freq)

            # Find target candidates in nearby frames
            target_candidates = []
            for dt in range(-target_zone_frames, target_zone_frames + 1):
                target_frame = t + dt
                if target_frame < 0 or target_frame >= n_frames:
                    continue

                for target in notes_by_frame[target_frame]:
                    target_freq = target.frequency

                    # Skip same note
                    if dt == 0 and abs(target_freq - anchor_freq) < 1.0:
                        continue

                    # Check frequency zone
                    if abs(target_freq - anchor_freq) > target_zone_hz:
                        continue

                    # Calculate ratio
                    if target_freq > 0 and anchor_freq > 0:
                        ratio = max(target_freq, anchor_freq) / min(target_freq, anchor_freq)

                        if ratio < min_ratio or ratio > max_ratio:
                            continue

                        # Distance metric
                        freq_dist = abs(target_freq - anchor_freq) / target_zone_hz
                        time_dist = abs(dt) / max(target_zone_frames, 1)
                        distance = freq_dist + time_dist

                        target_candidates.append({
                            'freq': target_freq,
                            'velocity': target.velocity,
                            'dt': dt,
                            'ratio': ratio,
                            'distance': distance,
                        })

            # Sort by distance and take top M
            target_candidates.sort(key=lambda x: x['distance'])
            selected_targets = target_candidates[:max_targets_per_anchor]

            # Generate tokens
            for target in selected_targets:
                log_ratio = np.log2(target['ratio'])
                delta_t = float(target['dt'])
                weight = np.sqrt(anchor_weight * (target['velocity'] / 127.0))
                target_band = get_band_id(target['freq'])

                tokens.append([log_ratio, delta_t, weight, float(anchor_band), float(target_band)])

        if tokens:
            tokens_per_frame.append(np.array(tokens, dtype=np.float32))
        else:
            tokens_per_frame.append(np.zeros((0, 5), dtype=np.float32))

    frame_times = np.arange(n_frames) / frame_rate

    return tokens_per_frame, frame_times.astype(np.float32)


def get_midi_duration(path: str | Path) -> float:
    """Get total duration of a MIDI file in seconds."""
    if pretty_midi is None:
        raise ImportError("pretty_midi required")

    midi = pretty_midi.PrettyMIDI(str(path))
    return midi.get_end_time()


def filter_notes_by_time(
    notes: List[Note],
    start_time: float,
    end_time: float,
) -> List[Note]:
    """
    Filter notes that overlap with a time window.

    Args:
        notes: List of Note objects
        start_time: Window start in seconds
        end_time: Window end in seconds

    Returns:
        Filtered list of notes
    """
    return [
        note for note in notes
        if note.offset > start_time and note.onset < end_time
    ]


if __name__ == '__main__':
    # Quick test
    print("MIDI Utils Test")
    print("=" * 50)

    # Create test notes (C major chord)
    test_notes = [
        Note(pitch=60, onset=0.0, offset=1.0, velocity=100),  # C4
        Note(pitch=64, onset=0.0, offset=1.0, velocity=80),   # E4
        Note(pitch=67, onset=0.0, offset=1.0, velocity=90),   # G4
        Note(pitch=72, onset=0.5, offset=1.5, velocity=70),   # C5
    ]

    print(f"Test notes: {len(test_notes)}")
    for n in test_notes:
        print(f"  Pitch {n.pitch} ({n.pitch_class}): {n.frequency:.1f} Hz, "
              f"onset={n.onset:.2f}s, vel={n.velocity}")

    # Test piano roll
    pr = notes_to_piano_roll(test_notes, duration=2.0, frame_rate=50)
    print(f"\nPiano roll shape: {pr.shape}")
    print(f"Non-zero frames: {(pr.sum(axis=1) > 0).sum()}")

    # Test pitch histogram
    hist = notes_to_pitch_histogram(test_notes)
    print(f"\nPitch class histogram: {hist}")
    print(f"Peak at: {np.argmax(hist)} (C=0, E=4, G=7)")

    # Test constellation tokens
    tokens, times = midi_to_constellation_tokens(test_notes, duration=2.0)
    total_tokens = sum(len(t) for t in tokens)
    print(f"\nConstellation tokens: {total_tokens} total")
    print(f"Frames with tokens: {sum(1 for t in tokens if len(t) > 0)}")

    print("\n" + "=" * 50)
    print("MIDI Utils test passed!")
