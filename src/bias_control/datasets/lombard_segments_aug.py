"""
Augmented Lombard Segment Dataset for S2-P2-main descriptor arms.

Extends LombardSegmentDataset to include F0 cache data per-modality.
Each item returns speech/EGG waveforms + precomputed F0/voiced arrays
needed by V4-lin, V4-log, and H-series descriptors.

The F0 cache is loaded once at init and sliced per-segment using the
frozen formula: frame_start = round(start_sec / 0.01).

A4-16k does NOT need F0 — it works on raw waveform only.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .lombard_segments import (
    LombardSegmentDataset,
    collate_lombard,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
)
from ..vocal_descriptors import F0_HOP_SEC, FRAMES_PER_SEGMENT

logger = logging.getLogger(__name__)


class LombardSegmentDatasetAug(LombardSegmentDataset):
    """Lombard dataset with F0 cache for descriptor computation.

    Adds f0_speech, voiced_speech, f0_egg, voiced_egg to each item.
    Falls back gracefully if a clip is missing from cache (zeros + unvoiced).
    """

    def __init__(
        self,
        lombard_dir: Union[str, Path],
        segment_index_path: Union[str, Path],
        f0_cache_path: Union[str, Path],
        split: Optional[str] = None,
        noise_condition: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(
            lombard_dir=lombard_dir,
            segment_index_path=segment_index_path,
            split=split,
            noise_condition=noise_condition,
            sample_rate=sample_rate,
        )

        # Load F0 cache
        logger.info(f"Loading F0 cache from {f0_cache_path}")
        self.f0_cache = np.load(str(f0_cache_path), allow_pickle=True)
        n_keys = len([k for k in self.f0_cache.files if k.startswith('f0_')])
        logger.info(f"F0 cache loaded: {n_keys} F0 arrays")

    def _get_f0_slice(self, clip_id: str, modality: str, start_sec: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get F0/voiced slice for a segment from cache.

        Uses frozen formula:
          frame_start = round(start_sec / F0_HOP_SEC)
          frame_end = frame_start + FRAMES_PER_SEGMENT

        Args:
            clip_id: clip identifier
            modality: 'speech' or 'egg'
            start_sec: segment start in seconds

        Returns:
            f0: [FRAMES_PER_SEGMENT] F0 in Hz
            voiced: [FRAMES_PER_SEGMENT] boolean
        """
        f0_key = f'f0_{modality}_{clip_id}'
        voiced_key = f'voiced_{modality}_{clip_id}'

        if f0_key not in self.f0_cache or voiced_key not in self.f0_cache:
            # Missing clip — return zeros (unvoiced)
            return (
                np.zeros(FRAMES_PER_SEGMENT, dtype=np.float32),
                np.zeros(FRAMES_PER_SEGMENT, dtype=bool),
            )

        f0_full = self.f0_cache[f0_key]
        voiced_full = self.f0_cache[voiced_key]

        frame_start = round(start_sec / F0_HOP_SEC)
        frame_end = frame_start + FRAMES_PER_SEGMENT

        # Slice with safe bounds
        n_frames = len(f0_full)
        if frame_start >= n_frames:
            return (
                np.zeros(FRAMES_PER_SEGMENT, dtype=np.float32),
                np.zeros(FRAMES_PER_SEGMENT, dtype=bool),
            )

        f0_slice = f0_full[frame_start:frame_end]
        voiced_slice = voiced_full[frame_start:frame_end]

        # Pad if needed
        if len(f0_slice) < FRAMES_PER_SEGMENT:
            pad = FRAMES_PER_SEGMENT - len(f0_slice)
            f0_slice = np.pad(f0_slice, (0, pad))
            voiced_slice = np.pad(voiced_slice, (0, pad))

        return f0_slice.astype(np.float32), voiced_slice.astype(bool)

    def __getitem__(self, idx: int) -> Dict:
        # Get base item (speech, egg, metadata)
        item = super().__getitem__(idx)

        seg = self.segments[idx]
        clip_id = seg['clip_id']
        start_sec = seg['start_sec']

        # Get F0 slices per-modality
        f0_speech, voiced_speech = self._get_f0_slice(clip_id, 'speech', start_sec)
        f0_egg, voiced_egg = self._get_f0_slice(clip_id, 'egg', start_sec)

        item['f0_speech'] = torch.from_numpy(f0_speech).float()
        item['voiced_speech'] = torch.from_numpy(voiced_speech).bool()
        item['f0_egg'] = torch.from_numpy(f0_egg).float()
        item['voiced_egg'] = torch.from_numpy(voiced_egg).bool()

        return item


def collate_lombard_aug(batch: List[Dict]) -> Dict:
    """Collate function for LombardSegmentDatasetAug."""
    base = collate_lombard(batch)
    base['f0_speech'] = torch.stack([b['f0_speech'] for b in batch])
    base['voiced_speech'] = torch.stack([b['voiced_speech'] for b in batch])
    base['f0_egg'] = torch.stack([b['f0_egg'] for b in batch])
    base['voiced_egg'] = torch.stack([b['voiced_egg'] for b in batch])
    return base


def create_lombard_dataloaders_aug(
    lombard_dir: Union[str, Path],
    segment_index_path: Union[str, Path],
    f0_cache_path: Union[str, Path],
    noise_condition: str = 'noise0',
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with F0 cache."""

    train_ds = LombardSegmentDatasetAug(
        lombard_dir, segment_index_path, f0_cache_path,
        split='train', noise_condition=noise_condition,
    )
    val_ds = LombardSegmentDatasetAug(
        lombard_dir, segment_index_path, f0_cache_path,
        split='validation', noise_condition=noise_condition,
    )
    test_ds = LombardSegmentDatasetAug(
        lombard_dir, segment_index_path, f0_cache_path,
        split='test', noise_condition=noise_condition,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_lombard_aug, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_lombard_aug,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_lombard_aug,
    )

    return train_loader, val_loader, test_loader
