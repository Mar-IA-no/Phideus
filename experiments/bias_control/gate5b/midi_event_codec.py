#!/usr/bin/env python3
"""
MIDI event tokenization codec for perceptual Test11 pipeline.

Token vocabulary (fixed):
- PAD, BOS, EOS
- TIME_SHIFT (20ms steps, up to 4s)
- VELOCITY bins
- NOTE_ON / NOTE_OFF for piano range 21..108
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.midi_utils import parse_midi


@dataclass(frozen=True)
class MidiEventCodecConfig:
    step_ms: int = 20
    segment_seconds: float = 4.0
    min_pitch: int = 21
    max_pitch: int = 108
    velocity_bins: int = 32
    max_seq_len: int = 512


class MidiEventCodec:
    """REMI-like codec specialized for MAESTRO piano segments."""

    def __init__(self, config: Optional[MidiEventCodecConfig] = None):
        self.cfg = config or MidiEventCodecConfig()

        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2

        self.pitch_count = self.cfg.max_pitch - self.cfg.min_pitch + 1
        self.max_time_shifts = int(round((self.cfg.segment_seconds * 1000) / self.cfg.step_ms))

        self.time_shift_start = 3
        self.velocity_start = self.time_shift_start + self.max_time_shifts
        self.note_on_start = self.velocity_start + self.cfg.velocity_bins
        self.note_off_start = self.note_on_start + self.pitch_count
        self.vocab_size = self.note_off_start + self.pitch_count

        self.step_seconds = self.cfg.step_ms / 1000.0

    def token_type(self, token_id: int) -> str:
        if token_id == self.pad_id:
            return 'pad'
        if token_id == self.bos_id:
            return 'bos'
        if token_id == self.eos_id:
            return 'eos'
        if self.time_shift_start <= token_id < self.velocity_start:
            return 'time_shift'
        if self.velocity_start <= token_id < self.note_on_start:
            return 'velocity'
        if self.note_on_start <= token_id < self.note_off_start:
            return 'note_on'
        if self.note_off_start <= token_id < self.vocab_size:
            return 'note_off'
        return 'unknown'

    def _velocity_to_bin(self, velocity: int) -> int:
        velocity = int(np.clip(velocity, 1, 127))
        # 1..127 -> 1..velocity_bins
        bin_idx = int(np.ceil((velocity / 127.0) * self.cfg.velocity_bins))
        return int(np.clip(bin_idx, 1, self.cfg.velocity_bins))

    def _bin_to_velocity(self, bin_idx: int) -> int:
        bin_idx = int(np.clip(bin_idx, 1, self.cfg.velocity_bins))
        center = (bin_idx - 0.5) / self.cfg.velocity_bins
        return int(np.clip(round(center * 127.0), 1, 127))

    def _pitch_to_id(self, pitch: int, is_on: bool) -> int:
        pitch_idx = pitch - self.cfg.min_pitch
        if is_on:
            return self.note_on_start + pitch_idx
        return self.note_off_start + pitch_idx

    def _id_to_pitch(self, token_id: int) -> int:
        ttype = self.token_type(token_id)
        if ttype == 'note_on':
            return self.cfg.min_pitch + (token_id - self.note_on_start)
        if ttype == 'note_off':
            return self.cfg.min_pitch + (token_id - self.note_off_start)
        raise ValueError(f'Not a note token id: {token_id}')

    def _time_shift_to_id(self, n_steps: int) -> int:
        n_steps = int(np.clip(n_steps, 1, self.max_time_shifts))
        return self.time_shift_start + (n_steps - 1)

    def _id_to_time_shift(self, token_id: int) -> int:
        return (token_id - self.time_shift_start) + 1

    def _velocity_to_id(self, vel_bin: int) -> int:
        vel_bin = int(np.clip(vel_bin, 1, self.cfg.velocity_bins))
        return self.velocity_start + (vel_bin - 1)

    def _id_to_velocity_bin(self, token_id: int) -> int:
        return (token_id - self.velocity_start) + 1

    def _notes_from_piece_window(
        self,
        midi_path: Path,
        start_time: float,
        end_time: Optional[float] = None,
    ) -> List[Tuple[int, float, float, int]]:
        """Extract clipped segment notes as (pitch, onset_rel, offset_rel, velocity)."""
        if end_time is None:
            end_time = start_time + self.cfg.segment_seconds

        notes = []
        for note in parse_midi(midi_path):
            if note.offset <= start_time or note.onset >= end_time:
                continue

            onset_rel = max(0.0, note.onset - start_time)
            offset_rel = min(self.cfg.segment_seconds, note.offset - start_time)
            if offset_rel <= onset_rel:
                continue

            if note.pitch < self.cfg.min_pitch or note.pitch > self.cfg.max_pitch:
                continue

            notes.append((note.pitch, onset_rel, offset_rel, int(note.velocity)))

        return notes

    def encode_segment(
        self,
        midi_path: Path,
        start_time: float,
        end_time: Optional[float] = None,
    ) -> np.ndarray:
        notes = self._notes_from_piece_window(midi_path, start_time, end_time)
        return self.encode_notes(notes)

    def encode_notes(self, notes: Sequence[Tuple[int, float, float, int]]) -> np.ndarray:
        """
        Encode notes to fixed-length token sequence.

        Input notes: (pitch, onset_rel, offset_rel, velocity)
        """
        events: List[Tuple[float, int, int, int]] = []
        # (time, priority, pitch, velocity)
        # priority: off=0 before on=1 at same time
        for pitch, onset, offset, velocity in notes:
            onset = float(np.clip(onset, 0.0, self.cfg.segment_seconds))
            offset = float(np.clip(offset, 0.0, self.cfg.segment_seconds))
            if offset <= onset:
                continue
            events.append((offset, 0, int(pitch), int(velocity)))
            events.append((onset, 1, int(pitch), int(velocity)))

        events.sort(key=lambda x: (x[0], x[1], x[2]))

        tokens = [self.bos_id]
        cur_t = 0.0
        cur_vel_bin = None

        for t, priority, pitch, velocity in events:
            delta = max(0.0, t - cur_t)
            n_steps = int(np.round(delta / self.step_seconds))
            while n_steps > 0:
                chunk = min(n_steps, self.max_time_shifts)
                tokens.append(self._time_shift_to_id(chunk))
                cur_t += chunk * self.step_seconds
                n_steps -= chunk

            if priority == 1:  # NOTE_ON
                vel_bin = self._velocity_to_bin(velocity)
                if vel_bin != cur_vel_bin:
                    tokens.append(self._velocity_to_id(vel_bin))
                    cur_vel_bin = vel_bin
                tokens.append(self._pitch_to_id(pitch, is_on=True))
            else:  # NOTE_OFF
                tokens.append(self._pitch_to_id(pitch, is_on=False))

            if len(tokens) >= self.cfg.max_seq_len - 1:
                break

        tokens.append(self.eos_id)

        if len(tokens) > self.cfg.max_seq_len:
            tokens = tokens[: self.cfg.max_seq_len]
            tokens[-1] = self.eos_id

        out = np.full((self.cfg.max_seq_len,), self.pad_id, dtype=np.int64)
        out[: len(tokens)] = np.asarray(tokens, dtype=np.int64)
        return out

    def decode_tokens(self, tokens: Sequence[int]) -> List[Tuple[int, float, float, int]]:
        """
        Decode tokens back to note tuples: (pitch, onset, offset, velocity).
        """
        t = 0.0
        cur_velocity = 80
        active: Dict[int, List[Tuple[float, int]]] = {}
        notes: List[Tuple[int, float, float, int]] = []

        for tok in tokens:
            tok = int(tok)
            ttype = self.token_type(tok)

            if ttype in ('pad', 'bos'):
                continue
            if ttype == 'eos':
                break
            if ttype == 'time_shift':
                t += self._id_to_time_shift(tok) * self.step_seconds
                t = min(t, self.cfg.segment_seconds)
                continue
            if ttype == 'velocity':
                cur_velocity = self._bin_to_velocity(self._id_to_velocity_bin(tok))
                continue
            if ttype == 'note_on':
                pitch = self._id_to_pitch(tok)
                active.setdefault(pitch, []).append((t, cur_velocity))
                continue
            if ttype == 'note_off':
                pitch = self._id_to_pitch(tok)
                stack = active.get(pitch)
                if stack:
                    onset, vel = stack.pop(0)
                    offset = max(t, onset + self.step_seconds)
                    notes.append((pitch, onset, min(offset, self.cfg.segment_seconds), vel))

        # Close dangling notes at segment end
        for pitch, stack in active.items():
            for onset, vel in stack:
                offset = max(onset + self.step_seconds, self.cfg.segment_seconds)
                notes.append((pitch, onset, min(offset, self.cfg.segment_seconds), vel))

        notes.sort(key=lambda x: (x[1], x[0]))
        return notes

    def non_pad_length(self, tokens: Sequence[int]) -> int:
        arr = np.asarray(tokens)
        nz = np.where(arr == self.pad_id)[0]
        return int(nz[0]) if len(nz) > 0 else int(len(arr))


def build_codec(config: Optional[MidiEventCodecConfig] = None) -> MidiEventCodec:
    return MidiEventCodec(config=config)
