#!/usr/bin/env python3
"""
MAESTRO Dataset Loader - Constellation Tokens (Audio + MIDI)
=============================================================

PyTorch Dataset for loading MAESTRO constellation tokens.
Provides paired (audio_tokens, midi_tokens) for cross-modal training.

Features:
- Loads NPZ from analizador_maestro.py
- Returns aligned audio-MIDI token sequences
- Supports train/val/test splits
- Automatic format detection (constellation or dense)

Usage:
------
from src.datasets.maestro_dataset import MAESTROConstellationDataset, create_maestro_dataloaders

train_loader, val_loader, test_loader = create_maestro_dataloaders(
    npz_path='data/maestro_v3/constellations/tokens.npz',
    batch_size=64,
    max_frames=100,
)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MAESTROConstellationDataset(Dataset):
    """
    PyTorch Dataset for MAESTRO constellation tokens.

    Each sample returns:
    - audio_tokens: Tensor [T, K, 5]
    - audio_mask: Tensor [T, K]
    - midi_tokens: Tensor [T, K, 5]
    - midi_mask: Tensor [T, K]
    - metadata: dict
    """

    def __init__(
        self,
        npz_path: str | Path,
        split: str = 'train',
        indices: Optional[List[int]] = None,
        max_frames: Optional[int] = None,
        frame_sample_mode: str = 'random',
        normalize_weights: bool = True,
    ):
        """
        Initialize MAESTRO Constellation dataset.

        Args:
            npz_path: Path to tokens NPZ file
            split: 'train', 'validation', or 'test'
            indices: Specific indices to use (overrides split)
            max_frames: Maximum frames per sequence
            frame_sample_mode: 'random', 'start', or 'end'
            normalize_weights: Normalize token weights
        """
        self.npz_path = Path(npz_path)
        self.split = split
        self.max_frames = max_frames
        self.frame_sample_mode = frame_sample_mode
        self.normalize_weights = normalize_weights

        self._load_data(indices)

    def _load_data(self, indices: Optional[List[int]]):
        """Load NPZ and build index mapping."""
        data = np.load(self.npz_path, allow_pickle=True)

        self.segment_ids = list(data['segment_ids'])
        self.piece_ids = list(data['piece_ids'])
        self.composers = list(data['composers'])
        self.splits = list(data['splits'])
        self.n_segments = int(data['n_segments'])
        self.max_tokens = int(data['max_tokens_per_frame'])
        self.token_dim = int(data['token_dim'])

        # Filter by split or indices
        if indices is not None:
            self.valid_indices = indices
        else:
            self.valid_indices = [
                i for i in range(self.n_segments)
                if self.splits[i] == self.split
            ]

        # Load data for valid indices
        self.audio_tokens = {}
        self.audio_masks = {}
        self.midi_tokens = {}
        self.midi_masks = {}
        self.frame_times = {}

        all_weights = []

        for idx in self.valid_indices:
            self.audio_tokens[idx] = data[f'audio_tokens_{idx}']
            self.audio_masks[idx] = data[f'audio_mask_{idx}']
            self.midi_tokens[idx] = data[f'midi_tokens_{idx}']
            self.midi_masks[idx] = data[f'midi_mask_{idx}']
            self.frame_times[idx] = data[f'frame_times_{idx}']

            # Collect weights for normalization
            if self.normalize_weights:
                audio_w = self.audio_tokens[idx][:, :, 2]
                midi_w = self.midi_tokens[idx][:, :, 2]
                all_weights.extend(audio_w[self.audio_masks[idx] > 0].flatten())
                all_weights.extend(midi_w[self.midi_masks[idx] > 0].flatten())

        # Normalization params
        if self.normalize_weights and all_weights:
            all_weights = np.array(all_weights)
            self.weight_mean = float(np.mean(all_weights))
            self.weight_std = float(np.std(all_weights)) + 1e-8
        else:
            self.weight_mean = 0.0
            self.weight_std = 1.0

    def _normalize_tokens(self, tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Normalize weight channel."""
        tokens = tokens.copy()
        if self.normalize_weights:
            tokens[:, :, 2] = (tokens[:, :, 2] - self.weight_mean) / self.weight_std
            tokens[mask == 0] = 0.0
        return tokens

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        file_idx = self.valid_indices[idx]

        audio_tok = self.audio_tokens[file_idx]
        audio_msk = self.audio_masks[file_idx]
        midi_tok = self.midi_tokens[file_idx]
        midi_msk = self.midi_masks[file_idx]

        T = audio_tok.shape[0]

        # Truncate if needed
        if self.max_frames is not None and T > self.max_frames:
            if self.frame_sample_mode == 'random':
                start = random.randint(0, T - self.max_frames)
            elif self.frame_sample_mode == 'start':
                start = 0
            else:
                start = T - self.max_frames

            audio_tok = audio_tok[start:start + self.max_frames]
            audio_msk = audio_msk[start:start + self.max_frames]
            midi_tok = midi_tok[start:start + self.max_frames]
            midi_msk = midi_msk[start:start + self.max_frames]

        # Normalize
        audio_tok = self._normalize_tokens(audio_tok, audio_msk)
        midi_tok = self._normalize_tokens(midi_tok, midi_msk)

        metadata = {
            'segment_id': self.segment_ids[file_idx],
            'piece_id': self.piece_ids[file_idx],
            'composer': self.composers[file_idx],
            'split': self.splits[file_idx],
        }

        return (
            torch.tensor(audio_tok, dtype=torch.float32),
            torch.tensor(audio_msk, dtype=torch.float32),
            torch.tensor(midi_tok, dtype=torch.float32),
            torch.tensor(midi_msk, dtype=torch.float32),
            metadata,
        )

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        total_frames = 0
        total_audio_tokens = 0
        total_midi_tokens = 0

        for idx in self.valid_indices:
            n_frames = len(self.frame_times[idx])
            total_frames += n_frames
            total_audio_tokens += self.audio_masks[idx].sum()
            total_midi_tokens += self.midi_masks[idx].sum()

        return {
            'n_segments': len(self.valid_indices),
            'total_frames': total_frames,
            'avg_audio_tokens': total_audio_tokens / max(total_frames, 1),
            'avg_midi_tokens': total_midi_tokens / max(total_frames, 1),
            'max_tokens': self.max_tokens,
            'token_dim': self.token_dim,
        }


def collate_maestro_constellation(batch):
    """
    Collate function for MAESTRO constellation dataset.

    Pads sequences to max length in batch.

    Returns:
        audio_tokens: [B, T_max, K, 5]
        audio_masks: [B, T_max, K]
        midi_tokens: [B, T_max, K, 5]
        midi_masks: [B, T_max, K]
        metas: list of dicts
        lengths: [B]
    """
    audio_tokens, audio_masks, midi_tokens, midi_masks, metas = zip(*batch)

    max_len = max(a.shape[0] for a in audio_tokens)
    K = audio_tokens[0].shape[1]
    D = audio_tokens[0].shape[2]

    B = len(batch)
    audio_tok_batch = torch.zeros(B, max_len, K, D)
    audio_msk_batch = torch.zeros(B, max_len, K)
    midi_tok_batch = torch.zeros(B, max_len, K, D)
    midi_msk_batch = torch.zeros(B, max_len, K)
    lengths = []

    for i, (at, am, mt, mm) in enumerate(zip(audio_tokens, audio_masks, midi_tokens, midi_masks)):
        T = at.shape[0]
        audio_tok_batch[i, :T] = at
        audio_msk_batch[i, :T] = am
        midi_tok_batch[i, :T] = mt
        midi_msk_batch[i, :T] = mm
        lengths.append(T)

    return (
        audio_tok_batch,
        audio_msk_batch,
        midi_tok_batch,
        midi_msk_batch,
        list(metas),
        torch.tensor(lengths),
    )


def create_maestro_dataloaders(
    npz_path: str | Path,
    batch_size: int = 64,
    max_frames: Optional[int] = 100,
    shuffle: bool = True,
    num_workers: int = 8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for MAESTRO.

    Args:
        npz_path: Path to tokens NPZ
        batch_size: Batch size
        max_frames: Max frames per sequence
        shuffle: Shuffle training data
        num_workers: DataLoader workers
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    random.seed(seed)

    train_dataset = MAESTROConstellationDataset(
        npz_path=npz_path,
        split='train',
        max_frames=max_frames,
        frame_sample_mode='random',
    )

    val_dataset = MAESTROConstellationDataset(
        npz_path=npz_path,
        split='validation',
        max_frames=max_frames,
        frame_sample_mode='start',
    )

    test_dataset = MAESTROConstellationDataset(
        npz_path=npz_path,
        split='test',
        max_frames=max_frames,
        frame_sample_mode='start',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_maestro_constellation,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_maestro_constellation,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_maestro_constellation,
        pin_memory=True,
    )

    print(f"MAESTRO DataLoaders created:")
    print(f"  Train: {len(train_dataset)} segments")
    print(f"  Val: {len(val_dataset)} segments")
    print(f"  Test: {len(test_dataset)} segments")
    print(f"  Max frames: {max_frames}, Batch size: {batch_size}")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-frames', type=int, default=50)
    args = parser.parse_args()

    print("Testing MAESTRO Constellation Dataset...")
    print("=" * 50)

    train_loader, val_loader, test_loader = create_maestro_dataloaders(
        npz_path=args.npz,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        num_workers=0,
    )

    print("\nTesting batch iteration...")
    for batch in train_loader:
        audio_tok, audio_msk, midi_tok, midi_msk, metas, lengths = batch
        print(f"Audio tokens: {audio_tok.shape}")
        print(f"Audio masks: {audio_msk.shape}")
        print(f"MIDI tokens: {midi_tok.shape}")
        print(f"MIDI masks: {midi_msk.shape}")
        print(f"Lengths: {lengths}")
        print(f"Valid audio tokens (sample 0, frame 0): {audio_msk[0, 0].sum():.0f}")
        print(f"Sample metadata: {metas[0]}")
        break

    print("\n" + "=" * 50)
    print("Dataset test passed!")
