#!/usr/bin/env python3
"""
Deep analysis of cross-modal overlap to understand why it's so low (13-29%).

Questions:
1. Are audio/MIDI events aligned temporally?
2. Are pitch estimations matching?
3. What types of tokens overlap vs don't overlap?
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import librosa
from src.utils.midi_utils import parse_midi
from src.extractors.event_based_extractor import (
    extract_events_from_midi,
    extract_events_from_audio,
    align_audio_events_to_midi,
    extract_all_tokens as extract_event_tokens,
    token_to_hash as event_token_to_hash,
    MusicEvent, EventToken,
    EVENT_FRAME_RATE,
)
from src.extractors.improved_tf_extractor import SR


def analyze_event_alignment(audio_events, midi_events, piece_name):
    """Compare event timing between audio and MIDI."""
    print(f"\n{'='*60}")
    print(f"EVENT ALIGNMENT: {piece_name}")
    print(f"{'='*60}")

    # MusicEvent uses t_frame (in frames at 100 Hz)
    audio_times = np.array([e.t_frame / EVENT_FRAME_RATE for e in audio_events])
    midi_times = np.array([e.t_frame / EVENT_FRAME_RATE for e in midi_events])

    print(f"\n📊 Event Counts:")
    print(f"  Audio events: {len(audio_events)}")
    print(f"  MIDI events:  {len(midi_events)}")

    # Find closest MIDI event for each audio event
    if len(audio_times) == 0 or len(midi_times) == 0:
        print("  No events to compare!")
        return

    time_diffs = []
    for at in audio_times:
        closest_idx = np.argmin(np.abs(midi_times - at))
        diff = at - midi_times[closest_idx]
        time_diffs.append(diff)

    time_diffs = np.array(time_diffs)

    print(f"\n⏱️ Time Alignment (audio - closest MIDI):")
    print(f"  Mean diff:   {np.mean(time_diffs)*1000:.1f} ms")
    print(f"  Std diff:    {np.std(time_diffs)*1000:.1f} ms")
    print(f"  Median diff: {np.median(time_diffs)*1000:.1f} ms")
    print(f"  Within 50ms: {(np.abs(time_diffs) < 0.05).mean()*100:.1f}%")
    print(f"  Within 100ms: {(np.abs(time_diffs) < 0.1).mean()*100:.1f}%")

    return time_diffs


def analyze_pitch_alignment(audio_events, midi_events, piece_name):
    """Compare pitch between audio and MIDI events."""
    print(f"\n🎵 PITCH ALIGNMENT: {piece_name}")

    audio_pitches = [e.pitch for e in audio_events if e.pitch > 0]
    midi_pitches = [e.pitch for e in midi_events if e.pitch > 0]

    if not audio_pitches or not midi_pitches:
        print("  No pitch data!")
        return

    # Pitch class distribution
    audio_pc = Counter([p % 12 for p in audio_pitches])
    midi_pc = Counter([p % 12 for p in midi_pitches])

    print(f"\n📊 Pitch Class Distribution:")
    print(f"  {'PC':<4} {'Audio':>8} {'MIDI':>8} {'Match':>8}")
    for pc in range(12):
        a = audio_pc.get(pc, 0) / len(audio_pitches) * 100
        m = midi_pc.get(pc, 0) / len(midi_pitches) * 100
        match = "✓" if abs(a - m) < 5 else ""
        note = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][pc]
        print(f"  {note:<4} {a:>7.1f}% {m:>7.1f}% {match:>8}")

    # Pitch range
    print(f"\n🎹 Pitch Range:")
    print(f"  Audio: {min(audio_pitches)} - {max(audio_pitches)} (range {max(audio_pitches)-min(audio_pitches)})")
    print(f"  MIDI:  {min(midi_pitches)} - {max(midi_pitches)} (range {max(midi_pitches)-min(midi_pitches)})")


def analyze_token_types(audio_tokens, midi_tokens, piece_name):
    """Analyze which token types overlap."""
    print(f"\n🔤 TOKEN TYPE ANALYSIS: {piece_name}")

    # Group by type
    audio_by_type = defaultdict(list)
    midi_by_type = defaultdict(list)

    for t in audio_tokens:
        audio_by_type[t.token_type].append(t)
    for t in midi_tokens:
        midi_by_type[t.token_type].append(t)

    type_names = {1: 'chord', 2: 'sequential', 3: 'constellation'}

    print(f"\n📊 Token Counts by Type:")
    print(f"  {'Type':<15} {'Audio':>8} {'MIDI':>8}")
    for t_type in [1, 2, 3]:
        a = len(audio_by_type[t_type])
        m = len(midi_by_type[t_type])
        print(f"  {type_names[t_type]:<15} {a:>8} {m:>8}")

    # Hash overlap by type
    print(f"\n🔗 Hash Overlap by Type:")
    for t_type in [1, 2, 3]:
        audio_hashes = set(event_token_to_hash(t) for t in audio_by_type[t_type])
        midi_hashes = set(event_token_to_hash(t) for t in midi_by_type[t_type])
        overlap = audio_hashes & midi_hashes
        if midi_hashes:
            overlap_pct = len(overlap) / len(midi_hashes) * 100
            print(f"  {type_names[t_type]:<15}: {len(overlap):>4} / {len(midi_hashes):>4} ({overlap_pct:.1f}%)")


def analyze_hash_components(audio_tokens, midi_tokens, piece_name):
    """Analyze individual components of hashes."""
    print(f"\n🔧 HASH COMPONENT ANALYSIS: {piece_name}")

    # For type 3 (constellation) tokens, analyze dt and dp distributions
    audio_type3 = [t for t in audio_tokens if t.token_type == 3]
    midi_type3 = [t for t in midi_tokens if t.token_type == 3]

    if not audio_type3 or not midi_type3:
        print("  No type-3 tokens!")
        return

    # Delta time distribution
    audio_dt = Counter([t.dt for t in audio_type3])
    midi_dt = Counter([t.dt for t in midi_type3])

    print(f"\n⏱️ Delta Time Distribution (type 3):")
    print(f"  {'DT':>4} {'Audio':>8} {'MIDI':>8}")
    all_dts = sorted(set(audio_dt.keys()) | set(midi_dt.keys()))[:15]
    for dt in all_dts:
        a = audio_dt.get(dt, 0)
        m = midi_dt.get(dt, 0)
        print(f"  {dt:>4} {a:>8} {m:>8}")

    # Delta pitch distribution
    audio_dp = Counter([t.dp for t in audio_type3])
    midi_dp = Counter([t.dp for t in midi_type3])

    print(f"\n🎵 Delta Pitch Distribution (type 3):")
    print(f"  {'DP':>4} {'Audio':>8} {'MIDI':>8}")
    all_dps = sorted(set(audio_dp.keys()) | set(midi_dp.keys()))
    for dp in all_dps[:20]:
        a = audio_dp.get(dp, 0)
        m = midi_dp.get(dp, 0)
        print(f"  {dp:>4} {a:>8} {m:>8}")

    # Pitch class anchor distribution
    audio_pc = Counter([t.pc_anchor for t in audio_type3])
    midi_pc = Counter([t.pc_anchor for t in midi_type3])

    print(f"\n🎹 Anchor Pitch Class Distribution (type 3):")
    for pc in range(12):
        a = audio_pc.get(pc, 0)
        m = midi_pc.get(pc, 0)
        note = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][pc]
        print(f"  {note:<4} {a:>8} {m:>8}")


def main():
    input_dir = Path(__file__).parent / "muestra_replicacion"
    audio_files = sorted(input_dir.glob("*.wav"))[:3]  # Analyze first 3 pieces

    print("="*70)
    print("DEEP OVERLAP ANALYSIS")
    print("="*70)

    for audio_path in audio_files:
        midi_path = audio_path.with_suffix('.midi')
        if not midi_path.exists():
            midi_path = audio_path.with_suffix('.mid')
        if not midi_path.exists():
            continue

        piece_name = audio_path.stem[:50]
        print(f"\n\n{'#'*70}")
        print(f"# PIECE: {piece_name}")
        print(f"{'#'*70}")

        # Load
        audio, _ = librosa.load(audio_path, sr=SR, mono=True, duration=120.0)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms * 0.1
        duration = len(audio) / SR

        notes = parse_midi(midi_path)
        notes = [n for n in notes if n.onset < duration]

        # Extract events
        audio_events = extract_events_from_audio(audio, sr=SR)
        midi_events = extract_events_from_midi(notes)

        # Align (this modifies audio_events in place)
        audio_events_aligned = align_audio_events_to_midi(audio_events, midi_events)

        # Analyze alignment
        analyze_event_alignment(audio_events_aligned, midi_events, piece_name)
        analyze_pitch_alignment(audio_events_aligned, midi_events, piece_name)

        # Extract tokens
        audio_tokens = extract_event_tokens(audio_events_aligned)
        midi_tokens = extract_event_tokens(midi_events)

        analyze_token_types(audio_tokens, midi_tokens, piece_name)
        analyze_hash_components(audio_tokens, midi_tokens, piece_name)

    print("\n\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
