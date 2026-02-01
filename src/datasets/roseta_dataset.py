#!/usr/bin/env python3
"""
Roseta Dataset Loader - Dual-Domain (Audio + Vibration)
========================================================

Dataset loader for the Roseta experiment with UOEMD data.
Provides paired (audio, vibration) samples for InfoNCE training.

Features:
- Loads NPZ from analizador_roseta.py
- Returns paired temporal sequences
- Supports train/val split
- Filtering by condition (healthy-only or all)
- Multiple output strategies (sequence, frames, average)

Usage:
------
from src.datasets.roseta_dataset import RosetaDataset, create_roseta_dataloaders

# Healthy-only training
train_loader, val_loader = create_roseta_dataloaders(
    npz_path='data/datasets/roseta_full.npz',
    healthy_only=True,
    batch_size=32,
    max_frames=100,
)

# Full dataset with fault evaluation
train_loader, val_loader = create_roseta_dataloaders(
    npz_path='data/datasets/roseta_full.npz',
    healthy_only=False,
    batch_size=32,
)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RosetaDataset(Dataset):
    """
    PyTorch Dataset for Roseta dual-domain data.

    Each sample returns:
    - audio: Tensor [T, B, 3] or [B, 3] depending on strategy
    - vibration: Tensor [T, B, 3] or [B, 3] (paired with audio)
    - metadata: dict with condition, is_healthy, etc.
    """

    def __init__(
        self,
        npz_path: str | Path,
        indices: Optional[List[int]] = None,
        healthy_only: bool = False,
        strategy: str = 'sequence',
        max_frames: Optional[int] = None,
        frame_sample_mode: str = 'random',
    ):
        """
        Initialize Roseta dataset.

        Args:
            npz_path: Path to roseta_*.npz file
            indices: Specific file indices to use (for train/val split)
            healthy_only: If True, only include HH (healthy) files
            strategy: Output strategy:
                - 'sequence': Return full temporal sequence [T, B, 3]
                - 'frames': Return individual frames [B, 3] (expands dataset)
                - 'average': Return time-averaged histogram [B, 3]
            max_frames: Maximum frames per sequence (for padding/truncation)
            frame_sample_mode: How to sample frames if max_frames < actual:
                - 'random': Random contiguous segment
                - 'start': First max_frames
                - 'end': Last max_frames
        """
        self.npz_path = Path(npz_path)
        self.strategy = strategy
        self.max_frames = max_frames
        self.frame_sample_mode = frame_sample_mode

        # Load data
        self._load_data(indices, healthy_only)

    def _load_data(self, indices: Optional[List[int]], healthy_only: bool):
        """Load NPZ and build index mapping."""
        data = np.load(self.npz_path, allow_pickle=True)

        self.filenames = list(data['filenames'])
        self.conditions = list(data['conditions'])
        self.metadata = data['metadata'].item()
        self.n_files = int(data['n_files'])

        # Filter by indices or condition
        if indices is not None:
            valid_indices = indices
        else:
            valid_indices = list(range(self.n_files))

        if healthy_only:
            valid_indices = [
                i for i in valid_indices
                if self.conditions[i] == 'HH'
            ]

        self.valid_indices = valid_indices

        # Preload all arrays for efficiency
        self.audio_data = {}
        self.vibration_data = {}
        self.frame_times = {}

        for idx in valid_indices:
            self.audio_data[idx] = data[f'audio_frames_{idx}']
            self.vibration_data[idx] = data[f'vibration_frames_{idx}']
            self.frame_times[idx] = data[f'frame_times_{idx}']

        # Build sample index based on strategy
        if self.strategy == 'frames':
            # Each frame is a sample
            self.samples = []
            for idx in valid_indices:
                n_frames = len(self.frame_times[idx])
                for frame_idx in range(n_frames):
                    self.samples.append((idx, frame_idx))
        else:
            # Each file is a sample
            self.samples = [(idx, None) for idx in valid_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        file_idx, frame_idx = self.samples[idx]

        audio = self.audio_data[file_idx]      # [T, B, 3]
        vibration = self.vibration_data[file_idx]  # [T, B, 3]

        # Get metadata
        filename = self.filenames[file_idx]
        meta = self.metadata[filename].copy()

        if self.strategy == 'frames':
            # Return single frame
            audio_out = torch.tensor(audio[frame_idx], dtype=torch.float32)
            vibration_out = torch.tensor(vibration[frame_idx], dtype=torch.float32)

        elif self.strategy == 'average':
            # Time-averaged histogram
            audio_out = torch.tensor(audio.mean(axis=0), dtype=torch.float32)
            vibration_out = torch.tensor(vibration.mean(axis=0), dtype=torch.float32)

        else:  # 'sequence'
            # Full or truncated sequence
            T = audio.shape[0]

            if self.max_frames is not None and T > self.max_frames:
                # Need to truncate
                if self.frame_sample_mode == 'random':
                    start = random.randint(0, T - self.max_frames)
                elif self.frame_sample_mode == 'start':
                    start = 0
                else:  # 'end'
                    start = T - self.max_frames

                audio = audio[start:start + self.max_frames]
                vibration = vibration[start:start + self.max_frames]

            audio_out = torch.tensor(audio, dtype=torch.float32)
            vibration_out = torch.tensor(vibration, dtype=torch.float32)

        return audio_out, vibration_out, meta

    def get_condition_indices(self, condition: str) -> List[int]:
        """Get sample indices for a specific condition."""
        return [
            i for i, (file_idx, _) in enumerate(self.samples)
            if self.conditions[file_idx] == condition
        ]

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        conditions = {}
        total_frames = 0

        for idx in self.valid_indices:
            cond = self.conditions[idx]
            conditions[cond] = conditions.get(cond, 0) + 1
            total_frames += len(self.frame_times[idx])

        return {
            'n_files': len(self.valid_indices),
            'n_samples': len(self.samples),
            'total_frames': total_frames,
            'conditions': conditions,
            'strategy': self.strategy,
        }


class RosetaConstellationDataset(Dataset):
    """
    PyTorch Dataset for Roseta constellation tokens (Fase 3A).

    Each sample returns:
    - audio_tokens: Tensor [T, K, 5] sparse tokens
    - audio_mask: Tensor [T, K] validity mask
    - vibration_tokens: Tensor [T, K, 5] sparse tokens
    - vibration_mask: Tensor [T, K] validity mask
    - metadata: dict with condition, is_healthy, etc.

    Token format: [log_ratio, delta_t, weight, anchor_band, target_band]
    """

    def __init__(
        self,
        npz_path: str | Path,
        indices: Optional[List[int]] = None,
        healthy_only: bool = False,
        strategy: str = 'sequence',
        max_frames: Optional[int] = None,
        frame_sample_mode: str = 'random',
        normalize_weights: bool = True,
    ):
        """
        Initialize Roseta Constellation dataset.

        Args:
            npz_path: Path to roseta_constellation.npz file
            indices: Specific file indices to use (for train/val split)
            healthy_only: If True, only include HH (healthy) files
            strategy: Output strategy:
                - 'sequence': Return full temporal sequence [T, K, 5]
                - 'frames': Return individual frames [K, 5] (expands dataset)
            max_frames: Maximum frames per sequence (for padding/truncation)
            frame_sample_mode: How to sample frames if max_frames < actual
            normalize_weights: If True, normalize token weights to [0, 1]
        """
        self.npz_path = Path(npz_path)
        self.strategy = strategy
        self.max_frames = max_frames
        self.frame_sample_mode = frame_sample_mode
        self.normalize_weights = normalize_weights

        self._load_data(indices, healthy_only)

    def _load_data(self, indices: Optional[List[int]], healthy_only: bool):
        """Load NPZ constellation data and build index mapping."""
        data = np.load(self.npz_path, allow_pickle=True)

        self.filenames = list(data['filenames'])
        self.conditions = list(data['conditions'])
        self.metadata = data['metadata'].item()
        self.n_files = int(data['n_files'])
        self.max_tokens = int(data.get('max_tokens_per_frame', 48))
        self.token_dim = int(data.get('token_dim', 5))

        # Filter by indices or condition
        if indices is not None:
            valid_indices = indices
        else:
            valid_indices = list(range(self.n_files))

        if healthy_only:
            valid_indices = [
                i for i in valid_indices
                if self.conditions[i] == 'HH'
            ]

        self.valid_indices = valid_indices

        # Preload all arrays
        self.audio_tokens = {}
        self.audio_masks = {}
        self.vibration_tokens = {}
        self.vibration_masks = {}
        self.frame_times = {}

        # Track weight statistics for normalization
        all_weights = []

        for idx in valid_indices:
            self.audio_tokens[idx] = data[f'audio_tokens_{idx}']
            self.audio_masks[idx] = data[f'audio_mask_{idx}']
            self.vibration_tokens[idx] = data[f'vibration_tokens_{idx}']
            self.vibration_masks[idx] = data[f'vibration_mask_{idx}']
            self.frame_times[idx] = data[f'frame_times_{idx}']

            # Collect weights for normalization
            if self.normalize_weights:
                audio_w = self.audio_tokens[idx][:, :, 2]  # weight is index 2
                vib_w = self.vibration_tokens[idx][:, :, 2]
                all_weights.extend(audio_w[self.audio_masks[idx] > 0].flatten())
                all_weights.extend(vib_w[self.vibration_masks[idx] > 0].flatten())

        # Compute normalization parameters
        if self.normalize_weights and all_weights:
            all_weights = np.array(all_weights)
            self.weight_mean = float(np.mean(all_weights))
            self.weight_std = float(np.std(all_weights)) + 1e-8
        else:
            self.weight_mean = 0.0
            self.weight_std = 1.0

        # Build sample index based on strategy
        if self.strategy == 'frames':
            self.samples = []
            for idx in valid_indices:
                n_frames = len(self.frame_times[idx])
                for frame_idx in range(n_frames):
                    self.samples.append((idx, frame_idx))
        else:
            self.samples = [(idx, None) for idx in valid_indices]

    def _normalize_tokens(self, tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Normalize token weights to zero mean, unit variance."""
        tokens = tokens.copy()
        if self.normalize_weights:
            # Only normalize weight channel (index 2)
            tokens[:, :, 2] = (tokens[:, :, 2] - self.weight_mean) / self.weight_std
            # Zero out padded positions
            tokens[mask == 0] = 0.0
        return tokens

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        file_idx, frame_idx = self.samples[idx]

        audio_tok = self.audio_tokens[file_idx]      # [T, K, 5]
        audio_msk = self.audio_masks[file_idx]       # [T, K]
        vib_tok = self.vibration_tokens[file_idx]    # [T, K, 5]
        vib_msk = self.vibration_masks[file_idx]     # [T, K]

        filename = self.filenames[file_idx]
        meta = self.metadata[filename].copy()

        if self.strategy == 'frames':
            # Single frame
            audio_tok_out = self._normalize_tokens(
                audio_tok[frame_idx:frame_idx+1], audio_msk[frame_idx:frame_idx+1]
            )[0]
            audio_msk_out = audio_msk[frame_idx]
            vib_tok_out = self._normalize_tokens(
                vib_tok[frame_idx:frame_idx+1], vib_msk[frame_idx:frame_idx+1]
            )[0]
            vib_msk_out = vib_msk[frame_idx]

        else:  # 'sequence'
            T = audio_tok.shape[0]

            if self.max_frames is not None and T > self.max_frames:
                if self.frame_sample_mode == 'random':
                    start = random.randint(0, T - self.max_frames)
                elif self.frame_sample_mode == 'start':
                    start = 0
                else:  # 'end'
                    start = T - self.max_frames

                audio_tok = audio_tok[start:start + self.max_frames]
                audio_msk = audio_msk[start:start + self.max_frames]
                vib_tok = vib_tok[start:start + self.max_frames]
                vib_msk = vib_msk[start:start + self.max_frames]

            audio_tok_out = self._normalize_tokens(audio_tok, audio_msk)
            audio_msk_out = audio_msk
            vib_tok_out = self._normalize_tokens(vib_tok, vib_msk)
            vib_msk_out = vib_msk

        return (
            torch.tensor(audio_tok_out, dtype=torch.float32),
            torch.tensor(audio_msk_out, dtype=torch.float32),
            torch.tensor(vib_tok_out, dtype=torch.float32),
            torch.tensor(vib_msk_out, dtype=torch.float32),
            meta
        )

    def get_condition_indices(self, condition: str) -> List[int]:
        """Get sample indices for a specific condition."""
        return [
            i for i, (file_idx, _) in enumerate(self.samples)
            if self.conditions[file_idx] == condition
        ]

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        conditions = {}
        total_frames = 0
        total_tokens_audio = 0
        total_tokens_vib = 0

        for idx in self.valid_indices:
            cond = self.conditions[idx]
            conditions[cond] = conditions.get(cond, 0) + 1
            n_frames = len(self.frame_times[idx])
            total_frames += n_frames
            total_tokens_audio += self.audio_masks[idx].sum()
            total_tokens_vib += self.vibration_masks[idx].sum()

        return {
            'n_files': len(self.valid_indices),
            'n_samples': len(self.samples),
            'total_frames': total_frames,
            'conditions': conditions,
            'strategy': self.strategy,
            'format': 'constellation',
            'max_tokens': self.max_tokens,
            'token_dim': self.token_dim,
            'avg_tokens_audio': total_tokens_audio / max(total_frames, 1),
            'avg_tokens_vib': total_tokens_vib / max(total_frames, 1),
        }


def collate_constellation_sequences(batch):
    """
    Custom collate function for constellation token sequences.

    Pads sequences to max length in batch.

    Returns:
        audio_tokens: [B, T_max, K, 5]
        audio_masks: [B, T_max, K]
        vib_tokens: [B, T_max, K, 5]
        vib_masks: [B, T_max, K]
        metas: list of metadata dicts
        lengths: [B] actual sequence lengths
    """
    audio_tokens, audio_masks, vib_tokens, vib_masks, metas = zip(*batch)

    # Get max sequence length in batch
    max_len = max(a.shape[0] for a in audio_tokens)
    K = audio_tokens[0].shape[1]  # max tokens per frame
    D = audio_tokens[0].shape[2]  # token dim

    # Initialize padded tensors
    B = len(batch)
    audio_tok_batch = torch.zeros(B, max_len, K, D)
    audio_msk_batch = torch.zeros(B, max_len, K)
    vib_tok_batch = torch.zeros(B, max_len, K, D)
    vib_msk_batch = torch.zeros(B, max_len, K)
    lengths = []

    for i, (at, am, vt, vm) in enumerate(zip(audio_tokens, audio_masks, vib_tokens, vib_masks)):
        T = at.shape[0]
        audio_tok_batch[i, :T] = at
        audio_msk_batch[i, :T] = am
        vib_tok_batch[i, :T] = vt
        vib_msk_batch[i, :T] = vm
        lengths.append(T)

    return (
        audio_tok_batch,
        audio_msk_batch,
        vib_tok_batch,
        vib_msk_batch,
        list(metas),
        torch.tensor(lengths)
    )


def collate_constellation_frames(batch):
    """Simple collate for constellation frame-level samples."""
    audio_tokens, audio_masks, vib_tokens, vib_masks, metas = zip(*batch)
    return (
        torch.stack(audio_tokens),
        torch.stack(audio_masks),
        torch.stack(vib_tokens),
        torch.stack(vib_masks),
        list(metas)
    )


def collate_roseta_sequences(batch):
    """
    Custom collate function for variable-length sequences.

    Pads sequences to max length in batch.
    """
    audios, vibrations, metas = zip(*batch)

    # Get max sequence length in batch
    max_len = max(a.shape[0] for a in audios)

    # Pad sequences
    B = audios[0].shape[1]  # bins
    C = audios[0].shape[2]  # channels

    audio_batch = torch.zeros(len(batch), max_len, B, C)
    vibration_batch = torch.zeros(len(batch), max_len, B, C)
    lengths = []

    for i, (a, v) in enumerate(zip(audios, vibrations)):
        T = a.shape[0]
        audio_batch[i, :T] = a
        vibration_batch[i, :T] = v
        lengths.append(T)

    return audio_batch, vibration_batch, list(metas), torch.tensor(lengths)


def collate_roseta_frames(batch):
    """Simple collate for frame-level samples."""
    audios, vibrations, metas = zip(*batch)
    return (
        torch.stack(audios),
        torch.stack(vibrations),
        list(metas)
    )


def detect_npz_format(npz_path: str | Path) -> str:
    """
    Detect the format of a Roseta NPZ file.

    Returns:
        'constellation' if the file contains constellation tokens
        'histogram' if the file contains histogram frames
    """
    data = np.load(npz_path, allow_pickle=True)
    output_format = str(data.get('output_format', 'histogram'))
    return output_format


def create_roseta_dataloaders(
    npz_path: str | Path,
    batch_size: int = 32,
    healthy_only: bool = False,
    strategy: str = 'sequence',
    max_frames: Optional[int] = None,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    shuffle: bool = True,
    num_workers: int = 8,
    seed: int = 42,
    return_test: bool = False,
    normalize_weights: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and optionally test dataloaders for Roseta dataset.

    IMPORTANT: Split is done BY FILE to prevent data leakage. No frames from
    the same file appear in multiple splits.

    Args:
        npz_path: Path to roseta_*.npz file
        batch_size: Batch size
        healthy_only: Only use healthy (HH) files
        strategy: 'sequence', 'frames', or 'average'
        max_frames: Max frames per sequence (for padding)
        train_split: Fraction for training (default 0.70)
        val_split: Fraction for validation (default 0.15)
        test_split: Fraction for test (default 0.15)
        shuffle: Shuffle training data
        num_workers: DataLoader workers
        seed: Random seed for reproducibility
        return_test: If True, return 3 loaders; if False, return 2 (legacy)

    Returns:
        If return_test=True: (train_loader, val_loader, test_loader)
        If return_test=False: (train_loader, val_loader) - legacy compatibility
    """
    # Validate splits
    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 0.01:
        raise ValueError(f"Splits must sum to 1.0, got {total_split}")

    # Load to get total count
    data = np.load(npz_path, allow_pickle=True)
    n_files = int(data['n_files'])
    conditions = list(data['conditions'])

    # Get valid indices
    if healthy_only:
        all_indices = [i for i in range(n_files) if conditions[i] == 'HH']
    else:
        all_indices = list(range(n_files))

    # Shuffle and split BY FILE (anti-leakage)
    random.seed(seed)
    shuffled_indices = all_indices.copy()
    random.shuffle(shuffled_indices)

    total = len(shuffled_indices)
    n_train = int(total * train_split)
    n_val = int(total * val_split)
    # n_test = total - n_train - n_val (remainder goes to test)

    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:n_train + n_val]
    test_indices = shuffled_indices[n_train + n_val:]

    # CRITICAL: Verify no leakage between splits
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)

    assert len(train_set & val_set) == 0, "Leakage detected: train/val overlap"
    assert len(val_set & test_set) == 0, "Leakage detected: val/test overlap"
    assert len(train_set & test_set) == 0, "Leakage detected: train/test overlap"

    # Auto-detect format
    data_format = detect_npz_format(npz_path)
    is_constellation = (data_format == 'constellation')

    # Create datasets based on format
    if is_constellation:
        DatasetClass = RosetaConstellationDataset
        extra_kwargs = {'normalize_weights': normalize_weights}
    else:
        DatasetClass = RosetaDataset
        extra_kwargs = {}

    train_dataset = DatasetClass(
        npz_path=npz_path,
        indices=train_indices,
        healthy_only=False,  # Already filtered
        strategy=strategy,
        max_frames=max_frames,
        **extra_kwargs,
    )

    val_dataset = DatasetClass(
        npz_path=npz_path,
        indices=val_indices,
        healthy_only=False,
        strategy=strategy,
        max_frames=max_frames,
        frame_sample_mode='start',  # Deterministic for validation
        **extra_kwargs,
    )

    test_dataset = DatasetClass(
        npz_path=npz_path,
        indices=test_indices,
        healthy_only=False,
        strategy=strategy,
        max_frames=max_frames,
        frame_sample_mode='start',  # Deterministic for test
        **extra_kwargs,
    ) if test_indices else None

    # Select collate function based on format and strategy
    if is_constellation:
        if strategy == 'sequence':
            collate_fn = collate_constellation_sequences
        else:
            collate_fn = collate_constellation_frames
    else:
        if strategy == 'sequence':
            collate_fn = collate_roseta_sequences
        else:
            collate_fn = collate_roseta_frames

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    format_str = "constellation" if is_constellation else "histogram"
    print(f"Roseta DataLoaders created (split by file, no leakage):")
    print(f"  Format: {format_str}")
    print(f"  Train: {len(train_dataset)} samples ({len(train_indices)} files)")
    print(f"  Val: {len(val_dataset)} samples ({len(val_indices)} files)")
    if test_dataset:
        print(f"  Test: {len(test_dataset)} samples ({len(test_indices)} files)")
    print(f"  Strategy: {strategy}, Max frames: {max_frames}")

    if return_test:
        return train_loader, val_loader, test_loader
    else:
        # Legacy compatibility: return only train and val
        return train_loader, val_loader


def create_evaluation_loader(
    npz_path: str | Path,
    condition: Optional[str] = None,
    batch_size: int = 32,
    strategy: str = 'sequence',
    max_frames: Optional[int] = None,
    num_workers: int = 8,
    normalize_weights: bool = True,
) -> DataLoader:
    """
    Create evaluation dataloader for specific condition.

    Auto-detects format (histogram or constellation) from the NPZ file.

    Args:
        npz_path: Path to roseta_*.npz
        condition: Specific condition (e.g., 'HH', 'RU', 'FB')
                   If None, includes all conditions
        batch_size: Batch size
        strategy: Output strategy
        max_frames: Max frames per sequence
        num_workers: DataLoader workers
        normalize_weights: Normalize token weights (constellation only)

    Returns:
        eval_loader
    """
    # Load to filter by condition and detect format
    data = np.load(npz_path, allow_pickle=True)
    n_files = int(data['n_files'])
    conditions = list(data['conditions'])

    if condition is not None:
        indices = [i for i in range(n_files) if conditions[i] == condition]
    else:
        indices = list(range(n_files))

    # Auto-detect format
    data_format = detect_npz_format(npz_path)
    is_constellation = (data_format == 'constellation')

    if is_constellation:
        dataset = RosetaConstellationDataset(
            npz_path=npz_path,
            indices=indices,
            healthy_only=False,
            strategy=strategy,
            max_frames=max_frames,
            frame_sample_mode='start',
            normalize_weights=normalize_weights,
        )
        collate_fn = collate_constellation_sequences if strategy == 'sequence' else collate_constellation_frames
    else:
        dataset = RosetaDataset(
            npz_path=npz_path,
            indices=indices,
            healthy_only=False,
            strategy=strategy,
            max_frames=max_frames,
            frame_sample_mode='start',
        )
        collate_fn = collate_roseta_sequences if strategy == 'sequence' else collate_roseta_frames

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    cond_str = condition if condition else 'ALL'
    format_str = "constellation" if is_constellation else "histogram"
    print(f"Evaluation loader ({cond_str}, {format_str}): {len(dataset)} samples")

    return loader


if __name__ == "__main__":
    # Quick test
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, default='data/datasets/roseta_full.npz')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--strategy', type=str, default='sequence')
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    print("Testing Roseta Dataset Loader...")
    print("=" * 50)

    # Auto-detect format
    data_format = detect_npz_format(args.npz)
    print(f"Detected format: {data_format}")

    train_loader, val_loader = create_roseta_dataloaders(
        npz_path=args.npz,
        batch_size=args.batch_size,
        healthy_only=False,  # Test with all data
        strategy=args.strategy,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
    )

    print("\nTesting batch iteration...")
    for batch in train_loader:
        if data_format == 'constellation':
            # Constellation format: (audio_tokens, audio_masks, vib_tokens, vib_masks, metas, lengths)
            audio_tokens, audio_masks, vib_tokens, vib_masks, metas, lengths = batch
            print(f"Audio tokens shape: {audio_tokens.shape}")
            print(f"Audio masks shape: {audio_masks.shape}")
            print(f"Vib tokens shape: {vib_tokens.shape}")
            print(f"Vib masks shape: {vib_masks.shape}")
            print(f"Lengths: {lengths}")
            print(f"Valid audio tokens (sample 0, frame 0): {audio_masks[0, 0].sum():.0f}")
        elif args.strategy == 'sequence':
            audio, vibration, metas, lengths = batch
            print(f"Audio batch shape: {audio.shape}")
            print(f"Vibration batch shape: {vibration.shape}")
            print(f"Lengths: {lengths}")
        else:
            audio, vibration, metas = batch
            print(f"Audio batch shape: {audio.shape}")
            print(f"Vibration batch shape: {vibration.shape}")

        print(f"First sample condition: {metas[0]['condition']}")
        break

    print("\n✔ Dataset loader test passed!")
