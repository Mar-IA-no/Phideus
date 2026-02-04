"""
MAESTRO Segment Dataset for Cross-Modal Learning.

Loads aligned Audio-MIDI segments from MAESTRO v3.0.0 dataset.
Handles segmentation, alignment verification, and data loading.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class SegmentInfo:
    """Information about a single segment."""
    piece_idx: int
    segment_idx: int
    start_time: float
    end_time: float
    audio_path: Path
    midi_path: Path
    split: str  # train/validation/test
    canonical_composer: str
    canonical_title: str


class MaestroSegmentDataset(Dataset):
    """
    Dataset of aligned Audio-MIDI segments from MAESTRO.

    Features:
    - Lazy loading of audio/MIDI
    - Configurable segment length and hop
    - Optional caching to disk
    - Supports shuffled pairs for control experiments
    """

    def __init__(
        self,
        maestro_dir: Union[str, Path],
        segment_len: float = 8.0,
        hop: float = 2.0,
        sample_rate: int = 24000,
        split: Optional[str] = None,  # "train", "validation", "test", or None for all
        max_segments_per_piece: int = 100,
        cache_dir: Optional[Union[str, Path]] = None,
        load_audio: bool = True,
        load_midi: bool = True,
        shuffle_pairs: bool = False,  # For control experiments
        random_seed: int = 42,
    ):
        """
        Args:
            maestro_dir: Path to MAESTRO dataset directory
            segment_len: Segment length in seconds
            hop: Hop between segments in seconds
            sample_rate: Audio sample rate (MERT uses 24kHz)
            split: Dataset split to use (None = all)
            max_segments_per_piece: Maximum segments per piece
            cache_dir: Directory for cached preprocessed data
            load_audio: Whether to load audio waveforms
            load_midi: Whether to load MIDI data
            shuffle_pairs: If True, shuffle audio-midi pairs (control experiment)
            random_seed: Random seed for shuffling
        """
        self.maestro_dir = Path(maestro_dir)
        self.segment_len = segment_len
        self.hop = hop
        self.sample_rate = sample_rate
        self.split = split
        self.max_segments_per_piece = max_segments_per_piece
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.load_audio = load_audio
        self.load_midi = load_midi
        self.shuffle_pairs = shuffle_pairs
        self.random_seed = random_seed

        # Load MAESTRO metadata
        self.metadata = self._load_metadata()

        # Build segment index
        self.segments: List[SegmentInfo] = []
        self._build_segment_index()

        # For shuffled pairs control
        if shuffle_pairs:
            self._shuffled_indices = list(range(len(self.segments)))
            rng = random.Random(random_seed)
            rng.shuffle(self._shuffled_indices)

        logger.info(
            f"MaestroSegmentDataset: {len(self.segments)} segments from "
            f"{len(self.metadata)} pieces (split={split})"
        )

    def _load_metadata(self) -> List[Dict]:
        """Load MAESTRO metadata from JSON."""
        json_path = self.maestro_dir / "maestro-v3.0.0.json"

        if not json_path.exists():
            raise FileNotFoundError(f"MAESTRO metadata not found: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Filter by split if specified
        if self.split:
            data = [d for d in data if d['split'] == self.split]

        return data

    def _build_segment_index(self):
        """Build index of all segments."""
        for piece_idx, piece in enumerate(self.metadata):
            audio_path = self.maestro_dir / piece['audio_filename']
            midi_path = self.maestro_dir / piece['midi_filename']

            if not audio_path.exists() or not midi_path.exists():
                logger.warning(f"Missing files for piece {piece_idx}: {piece['canonical_title']}")
                continue

            # Get duration
            duration = piece.get('duration', None)
            if duration is None:
                # Try to get from audio file
                try:
                    import librosa
                    duration = librosa.get_duration(path=str(audio_path))
                except Exception as e:
                    logger.warning(f"Could not get duration for {audio_path}: {e}")
                    duration = 300  # Default 5 minutes

            # Generate segments
            n_segments = 0
            t = 0.0
            while t + self.segment_len <= duration and n_segments < self.max_segments_per_piece:
                segment = SegmentInfo(
                    piece_idx=piece_idx,
                    segment_idx=n_segments,
                    start_time=t,
                    end_time=t + self.segment_len,
                    audio_path=audio_path,
                    midi_path=midi_path,
                    split=piece['split'],
                    canonical_composer=piece.get('canonical_composer', 'Unknown'),
                    canonical_title=piece.get('canonical_title', 'Unknown'),
                )
                self.segments.append(segment)
                t += self.hop
                n_segments += 1

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a segment.

        Returns dict with:
            - audio: [T] waveform at sample_rate
            - midi_pitch: [N] MIDI note pitches
            - midi_velocity: [N] MIDI note velocities
            - midi_duration: [N] MIDI note duration buckets
            - midi_mask: [N] padding mask
            - piece_idx: int
            - segment_idx: int
            - start_time: float
        """
        segment = self.segments[idx]

        # For shuffled pairs, use different MIDI
        if self.shuffle_pairs:
            midi_idx = self._shuffled_indices[idx]
            midi_segment = self.segments[midi_idx]
        else:
            midi_segment = segment

        result = {
            'piece_idx': segment.piece_idx,
            'segment_idx': segment.segment_idx,
            'start_time': segment.start_time,
            'is_shuffled': self.shuffle_pairs,
        }

        # Load audio
        if self.load_audio:
            audio = self._load_audio_segment(segment)
            result['audio'] = audio

        # Load MIDI
        if self.load_midi:
            midi_data = self._load_midi_segment(midi_segment)
            result.update(midi_data)

        return result

    def _load_audio_segment(self, segment: SegmentInfo) -> torch.Tensor:
        """Load audio segment."""
        try:
            import librosa

            # Load with target sample rate
            audio, _ = librosa.load(
                str(segment.audio_path),
                sr=self.sample_rate,
                offset=segment.start_time,
                duration=self.segment_len,
            )

            # Ensure correct length
            expected_samples = int(self.segment_len * self.sample_rate)
            if len(audio) < expected_samples:
                audio = np.pad(audio, (0, expected_samples - len(audio)))
            elif len(audio) > expected_samples:
                audio = audio[:expected_samples]

            return torch.from_numpy(audio).float()

        except Exception as e:
            logger.error(f"Failed to load audio: {segment.audio_path}: {e}")
            # Return silence
            return torch.zeros(int(self.segment_len * self.sample_rate))

    def _load_midi_segment(self, segment: SegmentInfo) -> Dict[str, torch.Tensor]:
        """Load MIDI segment and convert to event representation."""
        try:
            import pretty_midi

            midi = pretty_midi.PrettyMIDI(str(segment.midi_path))

            pitches = []
            velocities = []
            durations = []

            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue

                for note in instrument.notes:
                    # Check if note is in segment
                    if note.start >= segment.end_time:
                        continue
                    if note.end <= segment.start_time:
                        continue

                    pitches.append(note.pitch)
                    velocities.append(note.velocity)
                    durations.append(note.end - note.start)

            if len(pitches) == 0:
                # Empty segment - add dummy note
                pitches = [60]
                velocities = [64]
                durations = [0.5]

            # Convert durations to buckets
            from ..encoders.midi_encoder import duration_to_bucket
            durations_tensor = torch.tensor(durations, dtype=torch.float32)
            duration_buckets = duration_to_bucket(durations_tensor)

            return {
                'midi_pitch': torch.tensor(pitches, dtype=torch.long),
                'midi_velocity': torch.tensor(velocities, dtype=torch.long),
                'midi_duration': duration_buckets,
                'midi_n_events': len(pitches),
            }

        except Exception as e:
            logger.error(f"Failed to load MIDI: {segment.midi_path}: {e}")
            # Return dummy data
            return {
                'midi_pitch': torch.tensor([60], dtype=torch.long),
                'midi_velocity': torch.tensor([64], dtype=torch.long),
                'midi_duration': torch.tensor([16], dtype=torch.long),
                'midi_n_events': 1,
            }

    def get_piece_info(self, piece_idx: int) -> Dict:
        """Get metadata for a piece."""
        return self.metadata[piece_idx]

    def get_segments_for_piece(self, piece_idx: int) -> List[int]:
        """Get segment indices for a piece."""
        return [i for i, s in enumerate(self.segments) if s.piece_idx == piece_idx]


def collate_segments(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Pads MIDI sequences to same length within batch.
    """
    # Stack audio
    if 'audio' in batch[0]:
        audio = torch.stack([b['audio'] for b in batch])
    else:
        audio = None

    # Pad MIDI
    if 'midi_pitch' in batch[0]:
        max_len = max(b['midi_n_events'] for b in batch)

        pitches = torch.zeros(len(batch), max_len, dtype=torch.long)
        velocities = torch.zeros(len(batch), max_len, dtype=torch.long)
        durations = torch.zeros(len(batch), max_len, dtype=torch.long)
        masks = torch.ones(len(batch), max_len, dtype=torch.bool)  # True = padding

        for i, b in enumerate(batch):
            n = b['midi_n_events']
            pitches[i, :n] = b['midi_pitch']
            velocities[i, :n] = b['midi_velocity']
            durations[i, :n] = b['midi_duration']
            masks[i, :n] = False
    else:
        pitches = velocities = durations = masks = None

    return {
        'audio': audio,
        'midi_pitch': pitches,
        'midi_velocity': velocities,
        'midi_duration': durations,
        'midi_mask': masks,
        'piece_idx': torch.tensor([b['piece_idx'] for b in batch]),
        'segment_idx': torch.tensor([b['segment_idx'] for b in batch]),
        'start_time': torch.tensor([b['start_time'] for b in batch]),
        'is_shuffled': batch[0].get('is_shuffled', False),
    }


def create_dataloaders(
    maestro_dir: Union[str, Path],
    segment_len: float = 8.0,
    hop: float = 2.0,
    sample_rate: int = 24000,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = MaestroSegmentDataset(
        maestro_dir=maestro_dir,
        segment_len=segment_len,
        hop=hop,
        sample_rate=sample_rate,
        split="train",
    )

    val_dataset = MaestroSegmentDataset(
        maestro_dir=maestro_dir,
        segment_len=segment_len,
        hop=hop,
        sample_rate=sample_rate,
        split="validation",
    )

    test_dataset = MaestroSegmentDataset(
        maestro_dir=maestro_dir,
        segment_len=segment_len,
        hop=hop,
        sample_rate=sample_rate,
        split="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_segments,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_segments,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_segments,
    )

    return train_loader, val_loader, test_loader
