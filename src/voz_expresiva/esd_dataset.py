"""PyTorch Dataset wrappers que leen caches WavLM + descriptor para Fase 1.

Asume que `1_precache_wavlm.py` y `1_precache_descriptors.py` ya generaron:
    data/voz_expresiva/wavlm_cache/wavlm_features.npy  (memmap [N_utts, T_max, 1024])
    data/voz_expresiva/wavlm_cache/wavlm_lengths.npy   (array [N_utts])
    data/voz_expresiva/wavlm_cache/wavlm_index.json    (per-utt metadata)
    data/voz_expresiva/descriptors_cache/family_A.npy  (memmap [N_utts, T_max, 12])

Estructura compartida del index:
    {
      "utterances": [
        {"speaker_id": "0011", "emotion": "Angry", "sentence_id": "000351",
         "language": "EN", "wav_path": "...", "row_idx": 0, "T_50Hz": 245}
      ],
      "T_max": 350,
      "sample_rate": 16000,
      "frame_rate": 50
    }

El Dataset NO carga audio crudo — sólo features y descriptores cacheados.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

EMOTION_TO_LABEL = {
    "Angry": 0, "Happy": 1, "Neutral": 2, "Sad": 3, "Surprise": 4,
}
LABEL_TO_EMOTION = {v: k for k, v in EMOTION_TO_LABEL.items()}


class ESDCachedDataset(Dataset):
    """Lee WavLM features + Familia A descriptor desde memmaps.

    Args:
        wavlm_features: memmap [N_utts, T_max, 1024] (mmap_mode='r').
        wavlm_lengths: ndarray [N_utts] con la longitud real (T_50Hz) por utt.
        descriptors: memmap [N_utts, T_max, 12].
        utterances: lista de dicts con metadata por utt (orden alineado con row_idx).
        indices: subset opcional de row_idx a usar (None = todos).
        descriptor_zscore: tupla (mean, std) por dim para normalizar el descriptor,
            o None para usar crudo. Mean/std de shape [12].
    """

    def __init__(
        self,
        wavlm_features: np.ndarray,
        wavlm_lengths: np.ndarray,
        descriptors: np.ndarray,
        utterances: List[Dict],
        indices: Optional[List[int]] = None,
        per_row_zscore: Optional[tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]] = None,
    ):
        """
        Args:
            per_row_zscore: tupla (mean_map, std_map) donde cada uno es
                {row_idx: ndarray[12]} con stats por utterance. Si la utt
                no está en el dict, NO se normaliza ese utterance.
        """
        self.wavlm_features = wavlm_features
        self.wavlm_lengths = wavlm_lengths
        self.descriptors = descriptors
        self.utterances = utterances
        self.indices = indices if indices is not None else list(range(len(utterances)))
        if per_row_zscore is not None:
            mu_map, sd_map = per_row_zscore
            self._desc_mu_map = mu_map
            self._desc_sd_map = sd_map
        else:
            self._desc_mu_map = None
            self._desc_sd_map = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        row = self.indices[idx]
        utt = self.utterances[row]
        T = int(self.wavlm_lengths[row])

        # Slice from memmap (only the valid portion)
        features = np.array(self.wavlm_features[row, :T, :], dtype=np.float32)  # [T, 1024]
        descriptor = np.array(self.descriptors[row, :T, :], dtype=np.float32)   # [T, 12]

        # Impute NaN descriptors with 0.0 (safe: features handle the variability)
        nan_mask = ~np.isfinite(descriptor)
        if nan_mask.any():
            descriptor = np.where(nan_mask, 0.0, descriptor)

        # Apply per-row z-score if this utt has stats assigned
        if self._desc_mu_map is not None and row in self._desc_mu_map:
            mu = self._desc_mu_map[row]
            sd = self._desc_sd_map[row]
            sd_safe = np.where(sd < 1e-8, 1.0, sd)
            descriptor = (descriptor - mu) / sd_safe

        return {
            "features": torch.from_numpy(features),
            "descriptor": torch.from_numpy(descriptor.astype(np.float32)),
            "length": T,
            "label": EMOTION_TO_LABEL[utt["emotion"]],
            "speaker_id": utt["speaker_id"],
            "emotion": utt["emotion"],
            "sentence_id": utt["sentence_id"],
            "row_idx": row,
        }


def collate_padded(batch: List[Dict]) -> Dict:
    """Pad variable-length sequences to max in batch."""
    B = len(batch)
    T_max = max(item["length"] for item in batch)
    F = batch[0]["features"].shape[1]
    D = batch[0]["descriptor"].shape[1]

    features = torch.zeros(B, T_max, F, dtype=torch.float32)
    descriptors = torch.zeros(B, T_max, D, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        T = item["length"]
        features[i, :T] = item["features"]
        descriptors[i, :T] = item["descriptor"]
        mask[i, :T] = True
        labels[i] = item["label"]
        lengths[i] = T

    return {
        "features": features,
        "descriptor": descriptors,
        "mask": mask,
        "label": labels,
        "length": lengths,
        "speaker_id": [item["speaker_id"] for item in batch],
        "row_idx": [item["row_idx"] for item in batch],
    }


def load_cache(
    cache_root: str | Path,
    wavlm_subdir: str = "wavlm_cache",
    desc_subdir: str = "descriptors_cache",
) -> Dict:
    """Load all caches lazily (memmap) + index. Returns dict ready for ESDCachedDataset.

    Args:
        cache_root: parent dir holding the cache subdirs.
        wavlm_subdir: WavLM cache subdir name (default 'wavlm_cache'). For the ZH
            replication of Fase 1, use 'wavlm_cache_zh'.
        desc_subdir: descriptor cache subdir name (default 'descriptors_cache').
    """
    root = Path(cache_root)
    wavlm_dir = root / wavlm_subdir
    desc_dir = root / desc_subdir

    index_path = wavlm_dir / "wavlm_index.json"
    index = json.loads(index_path.read_text())
    T_max = index["T_max"]
    utterances = index["utterances"]
    N = len(utterances)

    wavlm_features = np.memmap(
        wavlm_dir / "wavlm_features.npy", dtype=np.float32, mode="r",
        shape=(N, T_max, 1024),
    )
    wavlm_lengths = np.load(wavlm_dir / "wavlm_lengths.npy")

    descriptors = np.memmap(
        desc_dir / "family_A.npy", dtype=np.float32, mode="r",
        shape=(N, T_max, 12),
    )

    return {
        "wavlm_features": wavlm_features,
        "wavlm_lengths": wavlm_lengths,
        "descriptors": descriptors,
        "utterances": utterances,
        "index": index,
    }


def compute_descriptor_zscore(
    descriptors: np.ndarray, lengths: np.ndarray, indices: List[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/std per-dim of descriptor across the given utterance indices.

    Used for per-speaker z-score in N-strict (train speakers) or for the
    calib set in N-adapt.

    Args:
        descriptors: memmap [N_utts, T_max, 12]
        lengths: [N_utts]
        indices: list of row_idx to include

    Returns:
        (mean, std) each shape [12].
    """
    all_frames = []
    for idx in indices:
        T = int(lengths[idx])
        chunk = np.array(descriptors[idx, :T, :], dtype=np.float32)
        # Drop NaN frames
        finite = np.isfinite(chunk).all(axis=1)
        if finite.any():
            all_frames.append(chunk[finite])
    if not all_frames:
        return np.zeros(12, dtype=np.float32), np.ones(12, dtype=np.float32)
    stack = np.concatenate(all_frames, axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    return mean.astype(np.float32), std.astype(np.float32)
