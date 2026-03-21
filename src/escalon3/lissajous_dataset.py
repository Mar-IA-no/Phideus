"""Escalón 3: LissajousDataset + collate + dataloader factory.

P0-P2: Always loads figure_clean from bundled .npy (memmapped).
No render augmentation. Render variants accessed only by eval for render-OOD.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LissajousDataset(Dataset):
    """PyTorch Dataset for bundled Lissajous scenes.

    Loads audio.npy (memmapped) and image.npy (memmapped) from a split directory.
    Metadata loaded from meta.npz into RAM (small).
    """

    def __init__(self, split_dir: Union[str, Path]):
        split_dir = Path(split_dir)
        assert (split_dir / 'audio.npy').exists(), f"Missing audio.npy in {split_dir}"

        self.audio = np.load(str(split_dir / 'audio.npy'), mmap_mode='r')
        self.image = np.load(str(split_dir / 'image.npy'), mmap_mode='r')

        # Optional: precomputed CQT (for cqtconv/cqtshift/cqthybrid encoders)
        cqt_path = split_dir / 'audio_cqt.npy'
        self.audio_cqt = np.load(str(cqt_path), mmap_mode='r') if cqt_path.exists() else None

        meta = np.load(str(split_dir / 'meta.npz'))

        self.ratio_id = meta['ratio_id']
        self.equiv_id = meta['equiv_id']
        self.phase_idx = meta['phase_idx']
        self.amp_ratio = meta['amp_ratio']
        self.base_freq = meta['base_freq']
        self.ratio_float = meta['ratio_float']
        self.harmonic_order = meta['harmonic_order']
        self.scene_id = meta['scene_id']

    def __len__(self) -> int:
        return len(self.audio)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'audio': torch.from_numpy(np.array(self.audio[idx])),
            'image': torch.from_numpy(np.array(self.image[idx])).unsqueeze(0).float() / 255.0,
            'ratio_id': int(self.ratio_id[idx]),
            'equiv_id': int(self.equiv_id[idx]),
            'phase_idx': int(self.phase_idx[idx]),
            'amp_ratio': float(self.amp_ratio[idx]),
            'base_freq': float(self.base_freq[idx]),
            'ratio_float': float(self.ratio_float[idx]),
            'harmonic_order': int(self.harmonic_order[idx]),
            'scene_id': str(self.scene_id[idx]),
        }
        if self.audio_cqt is not None:
            item['audio_cqt'] = torch.from_numpy(np.array(self.audio_cqt[idx])).float()
        return item


def collate_lissajous(batch: List[Dict]) -> Dict:
    """Collate function: stack tensors, list strings."""
    result = {
        'audio': torch.stack([b['audio'] for b in batch]),
        'image': torch.stack([b['image'] for b in batch]),
        'ratio_id': torch.tensor([b['ratio_id'] for b in batch], dtype=torch.long),
        'equiv_id': torch.tensor([b['equiv_id'] for b in batch], dtype=torch.long),
        'phase_idx': torch.tensor([b['phase_idx'] for b in batch], dtype=torch.long),
        'amp_ratio': torch.tensor([b['amp_ratio'] for b in batch], dtype=torch.float32),
        'base_freq': torch.tensor([b['base_freq'] for b in batch], dtype=torch.float32),
        'ratio_float': torch.tensor([b['ratio_float'] for b in batch], dtype=torch.float32),
        'harmonic_order': torch.tensor([b['harmonic_order'] for b in batch], dtype=torch.long),
        'scene_id': [b['scene_id'] for b in batch],
    }
    if 'audio_cqt' in batch[0]:
        result['audio_cqt'] = torch.stack([b['audio_cqt'] for b in batch])
    return result


def create_lissajous_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from bundled directory.

    Args:
        data_dir: Path to data/escalon3/bundled/
    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    train_ds = LissajousDataset(data_dir / 'train')
    val_ds = LissajousDataset(data_dir / 'val')
    test_ds = LissajousDataset(data_dir / 'test')

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_lissajous,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def load_ood_dataset(data_dir: Union[str, Path], split_name: str) -> LissajousDataset:
    """Load an OOD split dataset.

    Args:
        data_dir: Path to data/escalon3/bundled/
        split_name: 'ratio_ood', 'scale_ood', or 'equiv_ood'
    Returns:
        LissajousDataset for the specified OOD split
    """
    return LissajousDataset(Path(data_dir) / split_name)


class ConcatLissajousDataset(Dataset):
    """Concatenation of multiple LissajousDatasets.

    Exposes the same interface as LissajousDataset (audio, image, ratio_id, etc.)
    by concatenating metadata arrays from child datasets. Audio/image access is
    delegated to the child dataset that owns each index.
    """

    def __init__(self, datasets: List[LissajousDataset]):
        self.datasets = datasets
        self.cum_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cum_sizes.append(total)

        # Concatenate small metadata arrays for index-building compatibility
        self.ratio_id = np.concatenate([ds.ratio_id for ds in datasets])
        self.equiv_id = np.concatenate([ds.equiv_id for ds in datasets])
        self.phase_idx = np.concatenate([ds.phase_idx for ds in datasets])
        self.amp_ratio = np.concatenate([ds.amp_ratio for ds in datasets])
        self.base_freq = np.concatenate([ds.base_freq for ds in datasets])
        self.ratio_float = np.concatenate([ds.ratio_float for ds in datasets])
        self.harmonic_order = np.concatenate([ds.harmonic_order for ds in datasets])
        self.scene_id = np.concatenate([ds.scene_id for ds in datasets])

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        """Map global index to (dataset_idx, local_idx)."""
        for ds_idx, cum in enumerate(self.cum_sizes):
            if idx < cum:
                local = idx - (self.cum_sizes[ds_idx - 1] if ds_idx > 0 else 0)
                return ds_idx, local
        raise IndexError(f"Index {idx} out of range for {len(self)} items")

    def __len__(self) -> int:
        return self.cum_sizes[-1] if self.cum_sizes else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_idx, local_idx = self._resolve_index(idx)
        return self.datasets[ds_idx][local_idx]


def load_reduced_gallery_dataset(
    data_dir: Union[str, Path],
    splits: Tuple[str, ...] = ('train', 'val', 'test'),
) -> ConcatLissajousDataset:
    """Load a reduced reference atlas by concatenating multiple IID splits.

    This gallery contains ALL reduced-ratio scenes across train+val+test,
    providing complete coverage for OOD evaluation (no missing positives).

    Args:
        data_dir: Path to data/escalon3/bundled/
        splits: Which splits to include in the gallery
    Returns:
        ConcatLissajousDataset with all specified splits
    """
    data_dir = Path(data_dir)
    datasets = [LissajousDataset(data_dir / s) for s in splits]
    return ConcatLissajousDataset(datasets)
