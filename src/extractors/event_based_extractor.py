#!/usr/bin/env python3
"""
Ruta A: Event-Based Ratio Language Extractor.

Converts both Audio and MIDI to musical events (onset + pitch + amplitude),
then builds ratio language tokens from intervals and temporal ratios.

Based on GPT5.2Think specification in Documents/ESCALON_1/Extractor_nuevos_enfoques_GPT5.2Think.md
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, replace
from typing import List, Tuple, Optional
from collections import defaultdict

# Event representation
@dataclass
class MusicEvent:
    """A musical event (note onset)."""
    t_frame: int      # Time in frames (100 Hz)
    pitch: int        # MIDI pitch (0-127)
    amplitude: float  # Normalized amplitude [0, 1]
    duration: int     # Duration in frames (optional)


# Configuration defaults
EVENT_FRAME_RATE = 100  # Hz
DT_BIN_SIZE = 2         # frames (20ms)
DP_BIN_SIZE = 1         # semitones
CHORD_ONSET_TOL = 3     # frames (30ms)
PAIR_WINDOW = 200       # frames (2.0s)
FAN_OUT = 4             # targets per anchor


def hz_to_midi(freq: float) -> int:
    """Convert frequency to MIDI pitch."""
    if freq <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midi_to_hz(pitch: int) -> float:
    """Convert MIDI pitch to frequency."""
    return 440.0 * (2 ** ((pitch - 69) / 12.0))


# ============================================================================
# A.1 MIDI → Events (ground truth)
# ============================================================================

def extract_events_from_midi(
    notes: List,  # List of Note namedtuples from midi_utils
    frame_rate: int = EVENT_FRAME_RATE,
    min_note_dur: float = 0.05,
    merge_gap: float = 0.03,
) -> List[MusicEvent]:
    """
    Extract musical events from MIDI notes.

    Args:
        notes: List of Note(pitch, onset, offset, velocity) from parse_midi
        frame_rate: Output frame rate (default 100 Hz)
        min_note_dur: Minimum note duration in seconds
        merge_gap: Gap threshold for merging same-pitch notes

    Returns:
        List of MusicEvent sorted by time
    """
    if not notes:
        return []

    # Filter short notes
    filtered = [n for n in notes if (n.offset - n.onset) >= min_note_dur]

    # Sort by onset
    filtered = sorted(filtered, key=lambda n: n.onset)

    # Merge close same-pitch notes
    merged = []
    for note in filtered:
        if merged and note.pitch == merged[-1].pitch:
            gap = note.onset - merged[-1].offset
            if gap < merge_gap:
                # Extend previous note - use replace() for dataclass
                merged[-1] = replace(
                    merged[-1],
                    offset=note.offset,
                    velocity=max(merged[-1].velocity, note.velocity)
                )
                continue
        merged.append(note)

    # Convert to events
    events = []
    for note in merged:
        t_frame = int(round(note.onset * frame_rate))
        duration = int(round((note.offset - note.onset) * frame_rate))
        amplitude = note.velocity / 127.0

        events.append(MusicEvent(
            t_frame=t_frame,
            pitch=note.pitch,
            amplitude=amplitude,
            duration=duration
        ))

    return sorted(events, key=lambda e: (e.t_frame, -e.pitch))


# ============================================================================
# A.2 Audio → Events (onset detection + pitch estimation)
# ============================================================================

def extract_events_from_audio(
    audio: np.ndarray,
    sr: int = 22050,
    frame_rate: int = EVENT_FRAME_RATE,
    min_note_dur: float = 0.05,
    onset_conf_th: float = 0.5,
    max_polyphony: int = 10,
) -> List[MusicEvent]:
    """
    Extract musical events from audio using CQT + onset detection.

    This is a heuristic approach - not perfect but consistent.

    Args:
        audio: Audio waveform (mono)
        sr: Sample rate
        frame_rate: Output frame rate (default 100 Hz)
        min_note_dur: Minimum note duration
        onset_conf_th: Onset confidence threshold
        max_polyphony: Maximum simultaneous notes per frame

    Returns:
        List of MusicEvent sorted by time
    """
    import librosa

    # Compute CQT
    hop_length = sr // frame_rate
    n_bins = 84  # 7 octaves * 12 bins
    fmin = librosa.note_to_hz('C1')  # ~32.7 Hz

    cqt = np.abs(librosa.cqt(
        audio, sr=sr, hop_length=hop_length,
        fmin=fmin, n_bins=n_bins, bins_per_octave=12
    ))

    # Convert to dB
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)

    # Onset detection using spectral flux
    onset_env = librosa.onset.onset_strength(
        S=cqt_db, sr=sr, hop_length=hop_length
    )

    # Find onset frames
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length,
        backtrack=False, units='frames'
    )

    # For each onset, find active pitches
    events = []
    n_frames = cqt.shape[1]

    for onset_frame in onset_frames:
        if onset_frame >= n_frames:
            continue

        # Get spectrum at onset
        spectrum = cqt[:, onset_frame]

        # Normalize
        if spectrum.max() > 0:
            spectrum_norm = spectrum / spectrum.max()
        else:
            continue

        # Find peaks (local maxima above threshold)
        peaks = []
        for bin_idx in range(1, n_bins - 1):
            if spectrum_norm[bin_idx] >= onset_conf_th:
                if (spectrum_norm[bin_idx] > spectrum_norm[bin_idx - 1] and
                    spectrum_norm[bin_idx] > spectrum_norm[bin_idx + 1]):
                    # Convert bin to MIDI pitch
                    # CQT bin 0 = C1 = MIDI 24
                    pitch = 24 + bin_idx
                    peaks.append((pitch, spectrum_norm[bin_idx]))

        # Sort by amplitude and keep top max_polyphony
        peaks = sorted(peaks, key=lambda x: -x[1])[:max_polyphony]

        # Estimate duration by looking at decay
        # Simple: count frames until amplitude drops below threshold
        for pitch, amp in peaks:
            bin_idx = pitch - 24
            duration_frames = 1
            for t in range(onset_frame + 1, min(onset_frame + 500, n_frames)):  # Max 5s
                if cqt[bin_idx, t] / cqt[bin_idx, onset_frame] < 0.1:
                    break
                duration_frames += 1

            if duration_frames * (1 / frame_rate) >= min_note_dur:
                events.append(MusicEvent(
                    t_frame=onset_frame,
                    pitch=pitch,
                    amplitude=amp,
                    duration=duration_frames
                ))

    return sorted(events, key=lambda e: (e.t_frame, -e.pitch))


def align_audio_events_to_midi(
    audio_events: List[MusicEvent],
    midi_events: List[MusicEvent],
    max_shift_frames: int = 20,  # ±200ms
) -> List[MusicEvent]:
    """
    Align audio events to MIDI events by finding optimal time shift.

    Uses cross-correlation of onset binary series to find shift.
    """
    if not audio_events or not midi_events:
        return audio_events

    # Find time range
    max_frame = max(
        max(e.t_frame for e in audio_events),
        max(e.t_frame for e in midi_events)
    )

    # Create onset binary series
    audio_onsets = np.zeros(max_frame + 1)
    midi_onsets = np.zeros(max_frame + 1)

    for e in audio_events:
        if e.t_frame < len(audio_onsets):
            audio_onsets[e.t_frame] = 1
    for e in midi_events:
        if e.t_frame < len(midi_onsets):
            midi_onsets[e.t_frame] = 1

    # Find best shift
    best_shift = 0
    best_corr = -1

    for shift in range(-max_shift_frames, max_shift_frames + 1):
        if shift >= 0:
            corr = np.correlate(midi_onsets[shift:], audio_onsets[:len(midi_onsets) - shift])
        else:
            corr = np.correlate(midi_onsets[:len(midi_onsets) + shift], audio_onsets[-shift:])

        if len(corr) > 0 and corr[0] > best_corr:
            best_corr = corr[0]
            best_shift = shift

    # Apply shift to audio events
    aligned = []
    for e in audio_events:
        new_t = e.t_frame + best_shift
        if new_t >= 0:
            aligned.append(MusicEvent(
                t_frame=new_t,
                pitch=e.pitch,
                amplitude=e.amplitude,
                duration=e.duration
            ))

    return sorted(aligned, key=lambda e: (e.t_frame, -e.pitch))


# ============================================================================
# A.3 Token Construction
# ============================================================================

@dataclass
class EventToken:
    """A ratio language token from events."""
    token_type: int   # 1=chord, 2=sequential, 3=constellation
    dt: int           # Delta time in frames
    dp: int           # Delta pitch in semitones
    pc_anchor: int    # Pitch class of anchor (0-11)
    weight: float     # Token weight
    octave_anchor: int = 0  # Octave of anchor (for type 3)


def extract_chord_tokens(
    events: List[MusicEvent],
    chord_onset_tol: int = CHORD_ONSET_TOL,
) -> List[EventToken]:
    """
    Extract chord tokens (simultaneous intervals).

    Token Type 1: intervals within chords (dt=0).
    """
    if len(events) < 2:
        return []

    tokens = []

    # Group events by onset time
    groups = defaultdict(list)
    for e in events:
        # Round to tolerance
        group_key = e.t_frame // chord_onset_tol
        groups[group_key].append(e)

    for events_in_chord in groups.values():
        if len(events_in_chord) < 2:
            continue

        # Sort by pitch (lowest = anchor)
        chord = sorted(events_in_chord, key=lambda e: e.pitch)
        anchor = chord[0]

        for target in chord[1:]:
            dp = target.pitch - anchor.pitch
            pc_anchor = anchor.pitch % 12
            weight = np.sqrt(anchor.amplitude * target.amplitude)

            tokens.append(EventToken(
                token_type=1,
                dt=0,
                dp=dp,
                pc_anchor=pc_anchor,
                weight=weight
            ))

    return tokens


def extract_sequential_tokens(
    events: List[MusicEvent],
) -> List[EventToken]:
    """
    Extract sequential tokens (melodic intervals).

    Token Type 2: intervals between consecutive highest-amplitude notes.
    """
    if len(events) < 2:
        return []

    # Get one note per frame (highest amplitude)
    by_frame = defaultdict(list)
    for e in events:
        by_frame[e.t_frame].append(e)

    melody = []
    for frame in sorted(by_frame.keys()):
        events_at_frame = by_frame[frame]
        best = max(events_at_frame, key=lambda e: e.amplitude)
        melody.append(best)

    # Build tokens from consecutive pairs
    tokens = []
    for i in range(len(melody) - 1):
        e1 = melody[i]
        e2 = melody[i + 1]

        dt = e2.t_frame - e1.t_frame
        dp = e2.pitch - e1.pitch
        pc_anchor = e1.pitch % 12
        weight = np.sqrt(e1.amplitude * e2.amplitude)

        tokens.append(EventToken(
            token_type=2,
            dt=dt,
            dp=dp,
            pc_anchor=pc_anchor,
            weight=weight
        ))

    return tokens


def extract_constellation_tokens(
    events: List[MusicEvent],
    pair_window: int = PAIR_WINDOW,
    fan_out: int = FAN_OUT,
    k_anchors: int = 60,
) -> List[EventToken]:
    """
    Extract constellation tokens (Shazam-style in time-pitch plane).

    Token Type 3: pairs within window, with diversity (close + far targets).
    """
    if len(events) < 2:
        return []

    # Select top-K anchors by amplitude
    sorted_by_amp = sorted(events, key=lambda e: -e.amplitude)
    anchors = sorted_by_amp[:min(k_anchors, len(events))]

    tokens = []

    for anchor in anchors:
        # Find targets in window
        candidates = [
            e for e in events
            if 0 < (e.t_frame - anchor.t_frame) <= pair_window
        ]

        if not candidates:
            continue

        # Split into close (|dp| <= 12) and far (12 < |dp| <= 36)
        close = [e for e in candidates if abs(e.pitch - anchor.pitch) <= 12]
        far = [e for e in candidates if 12 < abs(e.pitch - anchor.pitch) <= 36]

        # Select targets with diversity
        targets = []

        # 2 close targets (by amplitude)
        close_sorted = sorted(close, key=lambda e: -e.amplitude)
        targets.extend(close_sorted[:2])

        # 2 far targets (by amplitude)
        far_sorted = sorted(far, key=lambda e: -e.amplitude)
        targets.extend(far_sorted[:2])

        for target in targets:
            dt = target.t_frame - anchor.t_frame
            dp = target.pitch - anchor.pitch
            pc_anchor = anchor.pitch % 12
            octave_anchor = anchor.pitch // 12
            weight = np.sqrt(anchor.amplitude * target.amplitude)

            tokens.append(EventToken(
                token_type=3,
                dt=dt,
                dp=dp,
                pc_anchor=pc_anchor,
                weight=weight,
                octave_anchor=octave_anchor
            ))

    return tokens


def extract_all_tokens(events: List[MusicEvent]) -> List[EventToken]:
    """Extract all token types from events."""
    tokens = []
    tokens.extend(extract_chord_tokens(events))
    tokens.extend(extract_sequential_tokens(events))
    tokens.extend(extract_constellation_tokens(events))
    return tokens


# ============================================================================
# A.4 Hash Construction
# ============================================================================

def token_to_hash(token: EventToken) -> int:
    """
    Convert token to hash for Shazam-style matching.

    Hash structure (20 bits):
    - type (2 bits): token type 1-3
    - dt_bin (6 bits): delta time bin
    - dp_bin (7 bits): delta pitch bin (-36 to +36)
    - pc_anchor (4 bits): pitch class
    """
    # Quantize
    dt_bin = min(token.dt // DT_BIN_SIZE, 63)  # 0-63
    dp_bin = max(-36, min(36, token.dp)) + 36  # 0-72

    # Build hash
    h = (token.token_type & 0x3) << 17  # bits 17-18
    h |= (dt_bin & 0x3F) << 11          # bits 11-16
    h |= (dp_bin & 0x7F) << 4           # bits 4-10
    h |= (token.pc_anchor & 0xF)         # bits 0-3

    return h


def hash_to_components(h: int) -> Tuple[int, int, int, int]:
    """Decode hash to (type, dt_bin, dp_bin, pc_anchor)."""
    token_type = (h >> 17) & 0x3
    dt_bin = (h >> 11) & 0x3F
    dp_bin = (h >> 4) & 0x7F
    pc_anchor = h & 0xF
    return token_type, dt_bin, dp_bin - 36, pc_anchor


def tokens_to_hash_set(tokens: List[EventToken]) -> set:
    """Convert list of tokens to set of hashes."""
    return {token_to_hash(t) for t in tokens}


def tokens_to_weighted_hashes(tokens: List[EventToken]) -> List[Tuple[int, float]]:
    """Convert tokens to list of (hash, weight) tuples."""
    return [(token_to_hash(t), t.weight) for t in tokens]


# ============================================================================
# A.5 Histogram Representation (alternative to hashes)
# ============================================================================

def build_histogram_2d(
    tokens: List[EventToken],
    dt_max: int = 100,  # bins
    dp_range: int = 36,  # ±36 semitones
) -> np.ndarray:
    """
    Build 2D histogram (dt_bin × dp_bin) from tokens.

    Returns normalized histogram [dt_bins, dp_bins].
    """
    dt_bins = dt_max // DT_BIN_SIZE + 1
    dp_bins = 2 * dp_range + 1  # -36 to +36

    hist = np.zeros((dt_bins, dp_bins), dtype=np.float32)

    for token in tokens:
        dt_bin = min(token.dt // DT_BIN_SIZE, dt_bins - 1)
        dp_bin = max(-dp_range, min(dp_range, token.dp)) + dp_range

        hist[dt_bin, dp_bin] += token.weight

    # Normalize
    total = hist.sum()
    if total > 0:
        hist /= total

    return hist


# ============================================================================
# Convenience functions for extraction pipeline
# ============================================================================

def process_midi_segment(
    notes: List,
    start_sec: float,
    end_sec: float,
    frame_rate: int = EVENT_FRAME_RATE,
) -> Tuple[List[MusicEvent], List[EventToken]]:
    """
    Process a MIDI segment: extract events and tokens.

    Returns (events, tokens).
    """
    # Filter notes in range
    segment_notes = [
        n for n in notes
        if n.onset >= start_sec and n.onset < end_sec
    ]

    # Adjust times relative to segment start
    adjusted_notes = []
    for n in segment_notes:
        adjusted_notes.append(n._replace(
            onset=n.onset - start_sec,
            offset=n.offset - start_sec
        ))

    events = extract_events_from_midi(adjusted_notes, frame_rate=frame_rate)
    tokens = extract_all_tokens(events)

    return events, tokens


def process_audio_segment(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    frame_rate: int = EVENT_FRAME_RATE,
) -> Tuple[List[MusicEvent], List[EventToken]]:
    """
    Process an audio segment: extract events and tokens.

    Returns (events, tokens).
    """
    # Extract segment
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    segment = audio[start_sample:end_sample]

    events = extract_events_from_audio(segment, sr=sr, frame_rate=frame_rate)
    tokens = extract_all_tokens(events)

    return events, tokens
