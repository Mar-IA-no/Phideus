#!/usr/bin/env python3
"""Utilities to render generated MIDI events to .mid and .wav files."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pretty_midi
import soundfile as sf

from experiments.bias_control.gate5b.midi_event_codec import MidiEventCodec


def notes_to_pretty_midi(
    notes: Sequence[Tuple[int, float, float, int]],
    tempo: float = 120.0,
    program: int = 0,
) -> pretty_midi.PrettyMIDI:
    """Build a PrettyMIDI object from (pitch, onset, offset, velocity) notes."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    piano = pretty_midi.Instrument(program=int(program), is_drum=False)

    for pitch, onset, offset, velocity in notes:
        p = int(np.clip(pitch, 0, 127))
        s = float(max(0.0, onset))
        e = float(max(s + 1e-3, offset))
        v = int(np.clip(round(velocity), 1, 127))
        piano.notes.append(pretty_midi.Note(velocity=v, pitch=p, start=s, end=e))

    piano.notes.sort(key=lambda n: (n.start, n.pitch))
    pm.instruments.append(piano)
    return pm


def write_notes_to_midi(
    notes: Sequence[Tuple[int, float, float, int]],
    output_path: Path,
    tempo: float = 120.0,
    program: int = 0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm = notes_to_pretty_midi(notes, tempo=tempo, program=program)
    pm.write(str(output_path))
    return output_path


def tokens_to_midi(
    codec: MidiEventCodec,
    tokens: Sequence[int],
    output_path: Path,
    tempo: float = 120.0,
) -> Path:
    """Decode tokens with codec and write MIDI file."""
    notes = codec.decode_tokens(tokens)
    return write_notes_to_midi(notes, output_path=output_path, tempo=tempo, program=0)


def _render_with_fluidsynth_binary(
    midi_path: Path,
    wav_path: Path,
    soundfont: Path,
    sample_rate: int,
) -> bool:
    if shutil.which('fluidsynth') is None:
        return False
    if not soundfont.exists():
        return False

    cmd = [
        'fluidsynth',
        '-ni',
        str(soundfont),
        str(midi_path),
        '-F',
        str(wav_path),
        '-r',
        str(sample_rate),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_path.exists() and wav_path.stat().st_size > 0
    except Exception:
        return False


def _render_with_pretty_midi(
    midi_path: Path,
    wav_path: Path,
    sample_rate: int,
    soundfont: Path | None = None,
) -> bool:
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        audio = None

        # Try high-quality fluidsynth backend first when available.
        if soundfont is not None and soundfont.exists():
            try:
                audio = pm.fluidsynth(fs=sample_rate, sf2_path=str(soundfont))
            except Exception:
                audio = None

        if audio is None:
            audio = pm.synthesize(fs=sample_rate)

        if audio is None or len(audio) == 0:
            return False

        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 1e-8:
            audio = audio / peak * 0.95

        sf.write(str(wav_path), audio.astype(np.float32), sample_rate)
        return wav_path.exists() and wav_path.stat().st_size > 0
    except Exception:
        return False


def render_midi_to_audio(
    midi_path: Path,
    wav_path: Path,
    renderer: str = 'auto',
    soundfont: str | None = None,
    sample_rate: int = 24000,
) -> str:
    """
    Render MIDI to WAV.

    Returns:
        Renderer actually used: "fluidsynth", "prettymidi", or "failed".
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf_path = Path(soundfont) if soundfont else None

    renderer = renderer.lower().strip()
    if renderer not in {'auto', 'fluidsynth', 'prettymidi'}:
        raise ValueError(f'Unknown renderer: {renderer}')

    if renderer in {'auto', 'fluidsynth'}:
        if sf_path is not None and _render_with_fluidsynth_binary(midi_path, wav_path, sf_path, sample_rate):
            return 'fluidsynth'
        if renderer == 'fluidsynth':
            return 'failed'

    if _render_with_pretty_midi(midi_path, wav_path, sample_rate=sample_rate, soundfont=sf_path):
        return 'prettymidi'

    return 'failed'
