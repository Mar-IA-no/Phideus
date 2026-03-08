"""
Lombard Segment Dataset for Speech <-> EGG Cross-Modal Learning.

Loads aligned Speech-EGG segments from French Lombard dataset.
Uses the frozen segment_index.json from S2-P0 for deterministic windows.

Protocol (frozen in P0):
  sr=16kHz, segment=2.0s (32000 samples), hop=0.5s
  Pilot: noise0 only. Multi-condition after P2.5.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SEGMENT_SAMPLES = 32000  # 2.0s @ 16kHz


class LombardSegmentDataset(Dataset):
    """
    Dataset of aligned Speech-EGG segments from French Lombard.

    Each item returns speech and EGG waveforms from the same temporal window.
    Segment windows are defined by segment_index.json (deterministic, from P0).
    """

    def __init__(
        self,
        lombard_dir: Union[str, Path],
        segment_index_path: Union[str, Path],
        split: Optional[str] = None,
        noise_condition: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.lombard_dir = Path(lombard_dir)
        self.sample_rate = sample_rate

        with open(segment_index_path) as f:
            si = json.load(f)

        segments = si['segments']

        # Filter
        if split:
            segments = [s for s in segments if s['split'] == split]
        if noise_condition:
            segments = [s for s in segments if s['noise_condition'] == noise_condition]

        self.segments = segments

        logger.info(
            f"LombardSegmentDataset: {len(self.segments)} segments "
            f"(split={split}, condition={noise_condition})"
        )

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seg = self.segments[idx]
        clip_id = seg['clip_id']
        speaker_id = seg['speaker_id']
        start_sample = int(seg['start_sec'] * self.sample_rate)
        n_samples = SEGMENT_SAMPLES

        # Load speech
        speech_path = self.lombard_dir / 'process' / 'wav' / speaker_id / f"{clip_id}.wav"
        speech, sr = sf.read(str(speech_path), start=start_sample,
                             stop=start_sample + n_samples, dtype='float32')
        assert sr == self.sample_rate

        # Load EGG
        egg_path = self.lombard_dir / 'process' / 'egg' / speaker_id / f"{clip_id}.wav"
        egg, sr = sf.read(str(egg_path), start=start_sample,
                          stop=start_sample + n_samples, dtype='float32')
        assert sr == self.sample_rate

        # Pad if needed (shouldn't happen with valid segments)
        if len(speech) < n_samples:
            speech = np.pad(speech, (0, n_samples - len(speech)))
        if len(egg) < n_samples:
            egg = np.pad(egg, (0, n_samples - len(egg)))

        return {
            'speech': torch.from_numpy(speech).float(),
            'egg': torch.from_numpy(egg).float(),
            'clip_id': clip_id,
            'segment_idx': seg['segment_idx'],
            'speaker_id': speaker_id,
            'sentence_id': seg['sentence_id'],
            'noise_condition': seg['noise_condition'],
        }


def collate_lombard(batch: List[Dict]) -> Dict:
    """Collate function for LombardSegmentDataset."""
    return {
        'speech': torch.stack([b['speech'] for b in batch]),
        'egg': torch.stack([b['egg'] for b in batch]),
        'clip_id': [b['clip_id'] for b in batch],
        'segment_idx': [b['segment_idx'] for b in batch],
        'speaker_id': [b['speaker_id'] for b in batch],
        'sentence_id': [b['sentence_id'] for b in batch],
        'noise_condition': [b['noise_condition'] for b in batch],
    }


def create_lombard_dataloaders(
    lombard_dir: Union[str, Path],
    segment_index_path: Union[str, Path],
    noise_condition: str = 'noise0',
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders for Lombard."""

    train_ds = LombardSegmentDataset(
        lombard_dir, segment_index_path,
        split='train', noise_condition=noise_condition,
    )
    val_ds = LombardSegmentDataset(
        lombard_dir, segment_index_path,
        split='validation', noise_condition=noise_condition,
    )
    test_ds = LombardSegmentDataset(
        lombard_dir, segment_index_path,
        split='test', noise_condition=noise_condition,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_lombard, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_lombard,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_lombard,
    )

    return train_loader, val_loader, test_loader
