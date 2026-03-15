"""
Precomputed WavLM Dataset for S2-P3.

Extends LombardSegmentDatasetAug to add precomputed WavLM speech features.
Each item returns:
  - speech_feat: [1024] precomputed WavLM mean-pooled embedding
  - speech: [32000] raw speech waveform (for descriptor computation only)
  - egg: [32000] raw EGG waveform (for trainable encoder + descriptor computation)
  - f0_speech, voiced_speech, f0_egg, voiced_egg (from F0 cache)
  - metadata: clip_id, segment_idx, speaker_id, sentence_id

The raw speech waveform is NOT passed through any encoder during training.
It is loaded only because some descriptors (A4-16k, H-series) need it.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .lombard_segments_aug import (
    LombardSegmentDatasetAug,
    collate_lombard_aug,
)
from .lombard_segments import SAMPLE_RATE

logger = logging.getLogger(__name__)


class LombardPrecomputedDataset(LombardSegmentDatasetAug):
    """Lombard dataset with precomputed WavLM features + F0 cache.

    Adds speech_feat to each item alongside the raw waveforms and F0 data.
    """

    def __init__(
        self,
        lombard_dir: Union[str, Path],
        segment_index_path: Union[str, Path],
        f0_cache_path: Union[str, Path],
        wavlm_features_path: Union[str, Path],
        split: Optional[str] = None,
        noise_condition: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(
            lombard_dir=lombard_dir,
            segment_index_path=segment_index_path,
            f0_cache_path=f0_cache_path,
            split=split,
            noise_condition=noise_condition,
            sample_rate=sample_rate,
        )

        # Load WavLM features into memory
        logger.info(f"Loading WavLM features from {wavlm_features_path}")
        npz = np.load(str(wavlm_features_path), allow_pickle=True)
        self.wavlm_cache = {k: npz[k] for k in npz.files if k != '__metadata__'}

        if '__metadata__' in npz.files:
            meta = json.loads(str(npz['__metadata__']))
            logger.info(f"WavLM features: model={meta.get('model_name')}, "
                        f"dim={meta.get('output_dim')}, n={meta.get('n_segments')}")

        npz.close()
        n_feats = len(self.wavlm_cache)
        logger.info(f"WavLM features loaded: {n_feats} vectors (in memory)")

    def __getitem__(self, idx: int) -> Dict:
        # Get base item (speech, egg, f0, metadata)
        item = super().__getitem__(idx)

        seg = self.segments[idx]
        clip_id = seg['clip_id']
        segment_idx = seg['segment_idx']

        # Lookup precomputed WavLM feature
        key = f"wavlm_{clip_id}_{segment_idx}"
        if key in self.wavlm_cache:
            feat = self.wavlm_cache[key]
        else:
            logger.warning(f"WavLM feature missing for {key}, using zeros")
            feat = np.zeros(1024, dtype=np.float32)

        item['speech_feat'] = torch.from_numpy(feat).float()

        return item


def collate_lombard_precomputed(batch: List[Dict]) -> Dict:
    """Collate function for LombardPrecomputedDataset."""
    base = collate_lombard_aug(batch)
    base['speech_feat'] = torch.stack([b['speech_feat'] for b in batch])
    return base


def create_p3_dataloaders(
    lombard_dir: Union[str, Path],
    segment_index_path: Union[str, Path],
    f0_cache_path: Union[str, Path],
    wavlm_features_path: Union[str, Path],
    noise_condition: str = 'noise0',
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with precomputed WavLM features."""

    train_ds = LombardPrecomputedDataset(
        lombard_dir, segment_index_path, f0_cache_path, wavlm_features_path,
        split='train', noise_condition=noise_condition,
    )
    val_ds = LombardPrecomputedDataset(
        lombard_dir, segment_index_path, f0_cache_path, wavlm_features_path,
        split='validation', noise_condition=noise_condition,
    )
    test_ds = LombardPrecomputedDataset(
        lombard_dir, segment_index_path, f0_cache_path, wavlm_features_path,
        split='test', noise_condition=noise_condition,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_lombard_precomputed, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_lombard_precomputed,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_lombard_precomputed,
    )

    return train_loader, val_loader, test_loader
