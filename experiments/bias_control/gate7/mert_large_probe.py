#!/usr/bin/env python3
"""
Gate 7 — MERT-large Linear Probe (Exp 7.0 + 7.0b).

Asks: how much of the A4 descriptor is linearly accessible in each encoder?

Encoders compared:
  MERTLite (D0)   — our trained audio encoder (~60M params, VICReg MAESTRO)
  MERT-v1-95M     — HuggingFace, ~95M params, audio foundation model
  MERT-v1-330M    — HuggingFace, ~330M params, audio foundation model (main test)

Primary endpoint: LinearProbe segment-level (Ridge closed-form, 5 group splits by piece).
Secondary endpoint: LinearProbe frame-level (within each encoder, no cross-encoder comparison).
Exploratory: Exp 7.0b per-layer analysis of MERT-330M.

IMPORTANT reading of results (from plan):
  - A high R² for MERT-large vs MERTLite suggests encoder capacity was a relevant limit.
  - A low R² does not prove complementarity — could be probe/target issue.
  - Results reduce ambiguity on the audio side only; full cross-modal picture requires Exp 7.1.

Usage:
    python experiments/bias_control/gate7/mert_large_probe.py \
        --maestro-dir data/maestro_v3/maestro-v3.0.0 \
        --d0-checkpoint models/gate5b/D0/best_model.pt \
        --output data/gate7_results \
        --encoders MERTLite MERT-v1-95M MERT-v1-330M \
        --n-splits 5 --seed 42

    # Frame-level secondary analysis (slower):
    python ... --frame-level

    # Per-layer analysis of MERT-330M (Exp 7.0b):
    python ... --per-layer
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_SEGS_PER_PIECE = 5
TRAIN_FRAC = 0.80
RIDGE_ALPHA = 1e-3   # Ridge regularization (features normalized → this is appropriate)
A4_N_BANDS = 8
A4_BAND_NAMES = [
    'band0_47Hz', 'band1_94Hz', 'band2_188Hz', 'band3_375Hz',
    'band4_750Hz', 'band5_1500Hz', 'band6_3000Hz', 'band7_6000Hz',
]


# ── Ridge Regression (closed form) ─────────────────────────────────────────────

def ridge_r2_per_band(
    X_train: np.ndarray,  # [N_train, D]
    Y_train: np.ndarray,  # [N_train, 8]
    X_test: np.ndarray,   # [N_test, D]
    Y_test: np.ndarray,   # [N_test, 8]
    alpha: float = RIDGE_ALPHA,
) -> np.ndarray:
    """
    Closed-form Ridge regression, R² per A4 band.

    Solves W = (X'X + αI)^{-1} X'Y for all 8 bands simultaneously.
    Returns R² per band [8].
    """
    D = X_train.shape[1]
    A = X_train.T @ X_train + alpha * np.eye(D)
    B = X_train.T @ Y_train  # [D, 8]
    W = np.linalg.solve(A, B)  # [D, 8]

    Y_pred = X_test @ W  # [N_test, 8]
    ss_res = ((Y_test - Y_pred) ** 2).sum(axis=0)  # [8]
    ss_tot = ((Y_test - Y_test.mean(axis=0)) ** 2).sum(axis=0)  # [8]
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)  # [8]

    return r2


def normalize_split(
    X_tr: np.ndarray, X_te: np.ndarray,
    Y_tr: np.ndarray, Y_te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalization fit on train, applied to test.
    No leakage between train and test.
    """
    X_mean = X_tr.mean(axis=0)
    X_std  = X_tr.std(axis=0) + 1e-8
    X_tr_n = (X_tr - X_mean) / X_std
    X_te_n = (X_te - X_mean) / X_std

    Y_mean = Y_tr.mean(axis=0)
    Y_std  = Y_tr.std(axis=0) + 1e-8
    Y_tr_n = (Y_tr - Y_mean) / Y_std
    Y_te_n = (Y_te - Y_mean) / Y_std

    return X_tr_n, X_te_n, Y_tr_n, Y_te_n


# ── Group split by piece ────────────────────────────────────────────────────────

def group_split_by_piece(
    piece_ids: np.ndarray,  # [N] — piece_idx for each segment
    train_frac: float = TRAIN_FRAC,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split segment indices into train/test, keeping all segments of a piece together.
    Returns (train_mask, test_mask) as boolean arrays of shape [N].
    """
    rng = np.random.default_rng(seed)
    unique_pieces = np.unique(piece_ids)
    rng.shuffle(unique_pieces)

    n_train_pieces = max(1, int(len(unique_pieces) * train_frac))
    train_pieces = set(unique_pieces[:n_train_pieces].tolist())

    train_mask = np.array([p in train_pieces for p in piece_ids])
    test_mask  = ~train_mask

    return train_mask, test_mask


# ── Data loading ───────────────────────────────────────────────────────────────

def load_validation_segments(
    maestro_dir: str,
    max_per_piece: int = MAX_SEGS_PER_PIECE,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Load audio waveforms and piece_ids from MAESTRO validation set.
    Caps at max_per_piece segments per piece.

    Returns:
        audio_np:  [N, 96000]  float32 waveforms
        piece_ids: [N]         int piece indices
        segments:  list of SegmentInfo
    """
    from src.bias_control.datasets.maestro_segments import MaestroSegmentDataset

    logger.info("Loading MAESTRO validation segments...")
    dataset = MaestroSegmentDataset(
        maestro_dir=maestro_dir,
        segment_len=4.0,
        hop=2.0,
        split='validation',
        max_segments_per_piece=max_per_piece,
        load_audio=True,
        load_midi=False,
    )
    logger.info(f"  Total segments: {len(dataset)}")

    # Collect all segments
    audio_list = []
    piece_list = []

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        collate_fn=lambda batch: {
            'audio': torch.stack([b['audio'] for b in batch]),
            'piece_idx': torch.tensor([b['piece_idx'] for b in batch]),
        },
    )

    for batch in tqdm(loader, desc="Loading audio"):
        audio_list.append(batch['audio'].numpy())
        piece_list.append(batch['piece_idx'].numpy())

    audio_np = np.concatenate(audio_list, axis=0)   # [N, 96000]
    piece_ids = np.concatenate(piece_list, axis=0)  # [N]

    logger.info(
        f"  Collected {audio_np.shape[0]} segments from "
        f"{len(np.unique(piece_ids))} pieces"
    )

    return audio_np, piece_ids, dataset.segments


# ── A4 target computation ──────────────────────────────────────────────────────

def compute_a4_targets(audio_np: np.ndarray, device: str, batch_size: int = 64) -> np.ndarray:
    """
    Compute mean-pooled A4 descriptor for each segment.

    Args:
        audio_np: [N, 96000]
    Returns:
        a4_np: [N, 8]  — segment-level mean of A4 over time
    """
    from src.bias_control.audio_descriptors import compute_audio_descriptor_a4

    N = audio_np.shape[0]
    a4_list = []

    for i in tqdm(range(0, N, batch_size), desc="Computing A4 targets"):
        batch = torch.from_numpy(audio_np[i:i+batch_size]).to(device)
        # Native STFT resolution (~188 frames for 4s @ hop=512)
        a4 = compute_audio_descriptor_a4(batch)  # [B, T_stft, 8]
        a4_mean = a4.mean(dim=1).cpu().numpy()   # [B, 8]
        a4_list.append(a4_mean)

    return np.concatenate(a4_list, axis=0)  # [N, 8]


def compute_a4_frame_targets(
    audio_np: np.ndarray,
    T_target: int,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute A4 descriptor downsampled to T_target frames (for frame-level probe).

    Args:
        audio_np: [N, 96000]
        T_target: target temporal length (e.g. T_mert ~ 300 for MERT on 4s)
    Returns:
        a4_np: [N, T_target, 8]
    """
    from src.bias_control.audio_descriptors import compute_audio_descriptor_a4
    from experiments.bias_control.gate7.mert_large_feature_extractor import MERTLargeExtractor

    N = audio_np.shape[0]
    a4_list = []

    for i in tqdm(range(0, N, batch_size), desc="Computing frame-level A4"):
        batch = torch.from_numpy(audio_np[i:i+batch_size]).to(device)
        # Compute at native resolution
        a4 = compute_audio_descriptor_a4(batch)  # [B, T_stft, 8]
        # Downsample to T_target using adaptive avg pool
        a4_ds = MERTLargeExtractor.align_a4_to_mert(a4, T_target)  # [B, T_target, 8]
        a4_list.append(a4_ds.cpu().numpy())

    return np.concatenate(a4_list, axis=0)  # [N, T_target, 8]


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_mertlite_features(
    audio_np: np.ndarray,
    d0_checkpoint: str,
    device: str,
    batch_size: int = 64,
) -> Dict:
    """
    Extract MERTLite (D0) audio encoder features.
    Returns pooled [N, 1024] and per-frame [N, T', 1024].
    """
    from experiments.bias_control.gate5b.checkpoint_loader import load_model_from_checkpoint

    logger.info(f"Loading D0 checkpoint: {d0_checkpoint}")
    model, meta = load_model_from_checkpoint(d0_checkpoint, device=torch.device(device))
    audio_encoder = model.audio_encoder
    audio_encoder.eval()

    N = audio_np.shape[0]
    pooled_list = []
    frame_list = []
    hidden_size = audio_encoder.get_output_dim()  # 1024

    for i in tqdm(range(0, N, batch_size), desc="Extracting MERTLite features"):
        batch = torch.from_numpy(audio_np[i:i+batch_size]).to(device)
        with torch.no_grad():
            # MERTEncoderLite.forward() returns pooled [B, D]
            pooled = audio_encoder(batch)  # [B, 1024]

            # For frame-level: extract pre-pool features via forward hooks
            frames = _extract_mertlite_frames(audio_encoder, batch)  # [B, T', 1024]

        pooled_list.append(pooled.cpu().numpy())
        frame_list.append(frames.cpu().numpy())

    pooled_np = np.concatenate(pooled_list, axis=0)  # [N, 1024]
    # frame features may have variable T' — pad if needed
    frame_np = _pad_frames(frame_list)               # [N, T', 1024]

    return {
        'pooled': pooled_np,
        'frames': frame_np,
        'hidden_size': hidden_size,
        'encoder_name': 'MERTLite-D0',
    }


def _extract_mertlite_frames(audio_encoder, waveform: torch.Tensor) -> torch.Tensor:
    """
    Extract pre-pool transformer output from MERTEncoderLite using a forward hook.
    Returns [B, T', hidden_size].
    """
    captured = {}

    def hook(module, input, output):
        # TransformerEncoder output: [B, T', D]
        captured['encoded'] = output

    handle = audio_encoder.transformer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = audio_encoder(waveform)
    finally:
        handle.remove()

    return captured['encoded']


def _pad_frames(frame_list: list) -> np.ndarray:
    """Pad frame-level outputs to uniform T if needed, then stack."""
    max_T = max(f.shape[1] for f in frame_list)
    padded = []
    for f in frame_list:
        T = f.shape[1]
        if T < max_T:
            pad = np.zeros((f.shape[0], max_T - T, f.shape[2]), dtype=f.dtype)
            f = np.concatenate([f, pad], axis=1)
        padded.append(f)
    return np.concatenate(padded, axis=0)


def extract_hf_mert_features(
    audio_np: np.ndarray,
    model_name: str,
    device: str,
    batch_size: int = 16,
) -> Dict:
    """
    Extract HuggingFace MERT features.
    Returns pooled [N, H] and last_layer [N, T_mert, H].
    """
    from experiments.bias_control.gate7.mert_large_feature_extractor import MERTLargeExtractor

    extractor = MERTLargeExtractor(model_name=model_name, device=device)

    N = audio_np.shape[0]
    pooled_list = []
    frame_list = []
    T_mert_val = None

    for i in tqdm(range(0, N, batch_size), desc=f"Extracting {model_name}"):
        batch = torch.from_numpy(audio_np[i:i+batch_size]).to(device)
        result = extractor.extract(batch, return_all_layers=False)
        pooled_list.append(result['pooled'].cpu().numpy())
        frame_list.append(result['last_layer'].cpu().numpy())
        if T_mert_val is None:
            T_mert_val = result['T_mert']

    pooled_np = np.concatenate(pooled_list, axis=0)  # [N, H]
    frame_np  = np.concatenate(frame_list,  axis=0)  # [N, T_mert, H]

    return {
        'pooled': pooled_np,
        'frames': frame_np,
        'hidden_size': extractor.hidden_size,
        'T_mert': T_mert_val,
        'encoder_name': model_name,
    }


def extract_hf_mert_per_layer(
    audio_np: np.ndarray,
    model_name: str,
    device: str,
    batch_size: int = 8,
) -> Dict:
    """
    Extract per-layer pooled features from HF MERT (Exp 7.0b).
    Returns per_layer_pooled: [n_layers+1, N, H] (includes embedding layer).
    """
    from experiments.bias_control.gate7.mert_large_feature_extractor import MERTLargeExtractor

    extractor = MERTLargeExtractor(model_name=model_name, device=device)
    N = audio_np.shape[0]
    n_layers = extractor.model.config.num_hidden_layers
    H = extractor.hidden_size

    # Pre-allocate [n_layers+1, N, H]
    per_layer = np.zeros((n_layers + 1, N, H), dtype=np.float32)

    for i in tqdm(range(0, N, batch_size), desc=f"Per-layer {model_name}"):
        b_slice = slice(i, min(i + batch_size, N))
        batch = torch.from_numpy(audio_np[b_slice]).to(device)
        result = extractor.extract(batch, return_all_layers=True)
        B_actual = batch.shape[0]
        for layer_idx, hidden in enumerate(result['all_layers']):
            # hidden: [B, T_mert, H] → mean pool → [B, H]
            pooled = hidden.mean(dim=1).cpu().numpy()  # [B, H]
            per_layer[layer_idx, i:i+B_actual] = pooled

    return {
        'per_layer_pooled': per_layer,  # [n_layers+1, N, H]
        'n_layers': n_layers,
        'hidden_size': H,
        'encoder_name': model_name,
    }


# ── Probe runner ───────────────────────────────────────────────────────────────

def run_segment_probe(
    features_np: np.ndarray,  # [N, D]
    a4_np: np.ndarray,        # [N, 8]
    piece_ids: np.ndarray,    # [N]
    n_splits: int,
    base_seed: int,
    encoder_name: str,
) -> Dict:
    """
    Run segment-level LinearProbe (Ridge closed-form) with n_splits group splits.
    Returns means ± std over splits for each band + global.
    """
    r2_per_split = []  # [n_splits, 8]

    for k in range(n_splits):
        split_seed = base_seed + k * 1000
        train_mask, test_mask = group_split_by_piece(piece_ids, TRAIN_FRAC, seed=split_seed)

        X_tr, X_te = features_np[train_mask], features_np[test_mask]
        Y_tr, Y_te = a4_np[train_mask], a4_np[test_mask]

        X_tr_n, X_te_n, Y_tr_n, Y_te_n = normalize_split(X_tr, X_te, Y_tr, Y_te)
        r2_bands = ridge_r2_per_band(X_tr_n, Y_tr_n, X_te_n, Y_te_n)  # [8]
        r2_per_split.append(r2_bands)

        n_train_pieces = len(np.unique(piece_ids[train_mask]))
        n_test_pieces  = len(np.unique(piece_ids[test_mask]))
        logger.info(
            f"  {encoder_name} split {k+1}/{n_splits}: "
            f"train={train_mask.sum()} segs ({n_train_pieces} pieces), "
            f"test={test_mask.sum()} segs ({n_test_pieces} pieces) | "
            f"R²_global={r2_bands.mean():.3f}"
        )

    r2_arr = np.array(r2_per_split)  # [n_splits, 8]

    return {
        'r2_per_band_mean': r2_arr.mean(axis=0).tolist(),
        'r2_per_band_std':  r2_arr.std(axis=0).tolist(),
        'r2_global_mean': float(r2_arr.mean()),
        'r2_global_std':  float(r2_arr.std()),
        'n_splits': n_splits,
        'n_segments': int(features_np.shape[0]),
        'n_pieces': int(len(np.unique(piece_ids))),
    }


def run_null_shuffled_between(
    features_np: np.ndarray,  # [N, D]
    a4_np: np.ndarray,        # [N, 8]
    piece_ids: np.ndarray,    # [N]
    n_splits: int,
    base_seed: int,
) -> Dict:
    """
    Null: shuffle A4 targets between segments.
    Breaks feature-target correspondence. Expected R² ≈ 0.
    Verification: if R² > 0.05, there's a bug in the protocol.
    """
    r2_per_split = []

    for k in range(n_splits):
        split_seed = base_seed + k * 1000
        train_mask, test_mask = group_split_by_piece(piece_ids, TRAIN_FRAC, seed=split_seed)

        X_tr, X_te = features_np[train_mask], features_np[test_mask]
        Y_tr_orig = a4_np[train_mask]
        Y_te_orig = a4_np[test_mask]

        # Shuffle targets within each split independently
        rng = np.random.default_rng(split_seed + 99999)
        Y_tr = Y_tr_orig[rng.permutation(len(Y_tr_orig))]
        Y_te = Y_te_orig[rng.permutation(len(Y_te_orig))]

        X_tr_n, X_te_n, Y_tr_n, Y_te_n = normalize_split(X_tr, X_te, Y_tr, Y_te)
        r2_bands = ridge_r2_per_band(X_tr_n, Y_tr_n, X_te_n, Y_te_n)
        r2_per_split.append(r2_bands)

    r2_arr = np.array(r2_per_split)
    return {
        'r2_global_mean': float(r2_arr.mean()),
        'r2_global_std':  float(r2_arr.std()),
        'r2_per_band_mean': r2_arr.mean(axis=0).tolist(),
    }


def run_null_dummy(
    a4_np: np.ndarray,   # [N, 8]
    piece_ids: np.ndarray,
    n_splits: int,
    base_seed: int,
) -> Dict:
    """
    Null: predict train mean for all test samples.
    Expected R² ≈ 0 (or slightly negative due to normalization).
    """
    r2_per_split = []

    for k in range(n_splits):
        split_seed = base_seed + k * 1000
        train_mask, test_mask = group_split_by_piece(piece_ids, TRAIN_FRAC, seed=split_seed)

        Y_tr = a4_np[train_mask]
        Y_te = a4_np[test_mask]

        Y_mean = Y_tr.mean(axis=0)  # [8]
        Y_pred = np.tile(Y_mean, (len(Y_te), 1))  # [N_test, 8]

        ss_res = ((Y_te - Y_pred) ** 2).sum(axis=0)
        ss_tot = ((Y_te - Y_te.mean(axis=0)) ** 2).sum(axis=0)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        r2_per_split.append(r2)

    r2_arr = np.array(r2_per_split)
    return {
        'r2_global_mean': float(r2_arr.mean()),
        'r2_global_std':  float(r2_arr.std()),
    }


def run_frame_probe(
    features_np: np.ndarray,   # [N, T, D] — frame-level features
    a4_frame_np: np.ndarray,   # [N, T, 8]
    piece_ids: np.ndarray,     # [N]
    n_splits: int,
    base_seed: int,
    encoder_name: str,
) -> Dict:
    """
    Frame-level LinearProbe (secondary endpoint).
    Flattens across frames: treat [N*T, D] → probe [N*T, 8].
    Also runs time-shuffle within segment to test if probe captures dynamics.
    """
    N, T, D = features_np.shape
    N2, T2, K = a4_frame_np.shape
    assert N == N2 and T == T2, f"Shape mismatch: features {features_np.shape}, a4 {a4_frame_np.shape}"

    r2_real_splits = []
    r2_timeshuffle_splits = []

    for k in range(n_splits):
        split_seed = base_seed + k * 1000
        train_mask, test_mask = group_split_by_piece(piece_ids, TRAIN_FRAC, seed=split_seed)

        # Flatten: [N_piece * T, D]
        X_tr = features_np[train_mask].reshape(-1, D)
        X_te = features_np[test_mask].reshape(-1, D)
        Y_tr = a4_frame_np[train_mask].reshape(-1, K)
        Y_te = a4_frame_np[test_mask].reshape(-1, K)

        X_tr_n, X_te_n, Y_tr_n, Y_te_n = normalize_split(X_tr, X_te, Y_tr, Y_te)
        r2_real = ridge_r2_per_band(X_tr_n, Y_tr_n, X_te_n, Y_te_n)
        r2_real_splits.append(r2_real)

        # Time-shuffle within segment (shuffle T dimension within each test segment)
        rng = np.random.default_rng(split_seed + 88888)
        test_segs = features_np[test_mask]  # [N_test, T, D]
        a4_test_segs = a4_frame_np[test_mask]  # [N_test, T, K]
        a4_shuffled = np.stack([
            a4_test_segs[i, rng.permutation(T), :]
            for i in range(test_segs.shape[0])
        ])
        Y_te_shuffled = a4_shuffled.reshape(-1, K)
        Y_te_shuf_n = (Y_te_shuffled - Y_tr.mean(axis=0)) / (Y_tr.std(axis=0) + 1e-8)
        r2_ts = ridge_r2_per_band(X_tr_n, Y_tr_n, X_te_n, Y_te_shuf_n)
        r2_timeshuffle_splits.append(r2_ts)

    r2_real_arr = np.array(r2_real_splits)
    r2_ts_arr   = np.array(r2_timeshuffle_splits)

    return {
        'frame_level_real': {
            'r2_per_band_mean': r2_real_arr.mean(axis=0).tolist(),
            'r2_per_band_std':  r2_real_arr.std(axis=0).tolist(),
            'r2_global_mean': float(r2_real_arr.mean()),
        },
        'frame_level_time_shuffle': {
            'r2_per_band_mean': r2_ts_arr.mean(axis=0).tolist(),
            'r2_per_band_std':  r2_ts_arr.std(axis=0).tolist(),
            'r2_global_mean': float(r2_ts_arr.mean()),
        },
    }


def run_per_layer_probe(
    per_layer_pooled: np.ndarray,  # [n_layers+1, N, H]
    a4_np: np.ndarray,             # [N, 8]
    piece_ids: np.ndarray,
    n_splits: int,
    base_seed: int,
) -> Dict:
    """Exp 7.0b: probe each layer of MERT-330M separately."""
    n_layers = per_layer_pooled.shape[0]
    layer_r2 = []

    for layer_idx in range(n_layers):
        features = per_layer_pooled[layer_idx]  # [N, H]
        result = run_segment_probe(
            features, a4_np, piece_ids, n_splits=n_splits,
            base_seed=base_seed, encoder_name=f'layer_{layer_idx}',
        )
        layer_r2.append(result['r2_global_mean'])
        logger.info(f"  Layer {layer_idx:02d}: R²_global={result['r2_global_mean']:.3f}")

    return {
        'layer_r2_global': layer_r2,  # [n_layers+1]
        'n_layers': n_layers,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def make_plots(results: Dict, output_dir: Path):
    """Create comparison charts and save as PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    encoders = list(results['segment_level'].keys())
    colors = {'MERTLite-D0': '#2196F3', 'MERT-v1-95M': '#FF9800', 'MERT-v1-330M': '#4CAF50'}
    default_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']

    # ── Plot 1: R²_seg comparison (primary) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(encoders))
    w = 0.6

    r2_means = [results['segment_level'][enc]['r2_global_mean'] for enc in encoders]
    r2_stds  = [results['segment_level'][enc]['r2_global_std']  for enc in encoders]
    null_mean = results['nulls']['shuffled_between']['r2_global_mean']

    bar_colors = [colors.get(enc, default_colors[i % len(default_colors)])
                  for i, enc in enumerate(encoders)]
    bars = ax.bar(x, r2_means, width=w, yerr=r2_stds, capsize=5,
                  color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.axhline(null_mean, color='red', linestyle='--', linewidth=1.5,
               label=f'Null (shuffled): {null_mean:.3f}')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace('m-a-p/', '') for e in encoders], fontsize=11)
    ax.set_ylabel('R² (segment-level, global mean over 8 bands)', fontsize=11)
    ax.set_title('Gate 7 — A4 Linear Probe: Segment-Level R² by Encoder', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.1, 1.05)
    for bar, mean, std in zip(bars, r2_means, r2_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'comparison_r2_seg.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot 2: Per-band breakdown ──
    n_enc = len(encoders)
    fig, axes = plt.subplots(1, n_enc, figsize=(5 * n_enc, 5), sharey=True)
    if n_enc == 1:
        axes = [axes]

    for ax, enc in zip(axes, encoders):
        r2_bands = np.array(results['segment_level'][enc]['r2_per_band_mean'])
        r2_stds  = np.array(results['segment_level'][enc]['r2_per_band_std'])
        y = np.arange(A4_N_BANDS)
        color = colors.get(enc, '#607D8B')
        ax.barh(y, r2_bands, xerr=r2_stds, capsize=4,
                color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        ax.axvline(null_mean, color='red', linestyle='--', linewidth=1.2)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(A4_BAND_NAMES, fontsize=9)
        ax.set_xlabel('R²', fontsize=10)
        ax.set_title(enc.replace('m-a-p/', ''), fontsize=11, fontweight='bold')
        ax.set_xlim(-0.05, max(0.6, max(r2_bands) + 0.1))

    axes[0].set_ylabel('A4 band', fontsize=10)
    plt.suptitle('Gate 7 — A4 Probe: Per-Band R² (segment-level, mean ± std)', fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / 'per_band_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot 3: Per-layer curve (if available) ──
    if 'per_layer' in results:
        layer_r2 = results['per_layer']['layer_r2_global']
        n_layers = len(layer_r2)
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(n_layers)
        ax.plot(x, layer_r2, marker='o', linewidth=2, color='#4CAF50', markersize=5)
        ax.axhline(null_mean, color='red', linestyle='--', linewidth=1.5,
                   label=f'Null (shuffled): {null_mean:.3f}')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('MERT-330M Layer Index', fontsize=11)
        ax.set_ylabel('R²_global', fontsize=11)
        ax.set_title('Gate 7 Exp 7.0b — A4 Probe R² vs MERT-330M Layer Depth', fontsize=12)
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig.savefig(output_dir / 'per_layer_r2.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    logger.info(f"Plots saved to {output_dir}")


# ── Feature cache helpers ──────────────────────────────────────────────────────

def save_features(cache_path: Path, features_dict: Dict):
    """Save extracted features to .npz cache."""
    np.savez_compressed(
        str(cache_path),
        pooled=features_dict['pooled'],
        frames=features_dict['frames'],
        hidden_size=np.array(features_dict['hidden_size']),
    )
    logger.info(f"  Features cached: {cache_path}")


def load_features(cache_path: Path) -> Dict:
    """Load features from .npz cache."""
    data = np.load(str(cache_path))
    return {
        'pooled': data['pooled'],
        'frames': data['frames'],
        'hidden_size': int(data['hidden_size']),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Gate 7 — MERT-large Linear Probe (Exp 7.0 + 7.0b)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--d0-checkpoint', type=str, default='models/gate5b/D0/best_model.pt',
                        help='D0 checkpoint for MERTLite baseline encoder')
    parser.add_argument('--output', type=str, default='data/gate7_results',
                        help='Output directory for results and feature cache')
    parser.add_argument('--encoders', nargs='+',
                        default=['MERTLite', 'MERT-v1-95M', 'MERT-v1-330M'],
                        help='Encoders to include. Use "MERTLite", "MERT-v1-95M", "MERT-v1-330M"')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--frame-level', action='store_true',
                        help='Also run frame-level secondary probe')
    parser.add_argument('--per-layer', action='store_true',
                        help='Exp 7.0b: per-layer probe on MERT-v1-330M')
    parser.add_argument('--skip-cache', action='store_true',
                        help='Re-extract features even if cache exists')

    args = parser.parse_args()

    output_dir = Path(args.output)
    features_dir = output_dir / 'features'
    results_dir  = output_dir / 'probe_results'
    features_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    t_start = time.time()

    # ── Phase 1: Load validation segments ─────────────────────────────────────
    audio_np, piece_ids, segments = load_validation_segments(
        maestro_dir=args.maestro_dir,
        max_per_piece=MAX_SEGS_PER_PIECE,
        seed=args.seed,
    )
    N = audio_np.shape[0]
    logger.info(f"N={N} segments, {len(np.unique(piece_ids))} unique pieces")

    # ── Phase 2: Compute A4 targets ────────────────────────────────────────────
    a4_cache = features_dir / 'a4_targets.npz'
    if a4_cache.exists() and not args.skip_cache:
        logger.info("Loading cached A4 targets...")
        a4_np = np.load(str(a4_cache))['a4']
    else:
        logger.info("Computing A4 targets (segment-level mean)...")
        a4_np = compute_a4_targets(audio_np, device=device, batch_size=args.batch_size)
        np.savez_compressed(str(a4_cache), a4=a4_np, piece_ids=piece_ids)
        logger.info(f"  A4 targets shape: {a4_np.shape}, range [{a4_np.min():.2f}, {a4_np.max():.2f}]")

    # ── Phase 3: Extract features for each encoder ────────────────────────────
    encoder_features = {}

    for enc_name in args.encoders:
        safe_name = enc_name.replace('/', '_').replace('-', '_')
        cache_path = features_dir / f'{safe_name}_features.npz'

        if cache_path.exists() and not args.skip_cache:
            logger.info(f"Loading cached features: {enc_name}")
            encoder_features[enc_name] = load_features(cache_path)
            encoder_features[enc_name]['encoder_name'] = enc_name
        else:
            logger.info(f"Extracting features: {enc_name}")
            if enc_name == 'MERTLite':
                feat = extract_mertlite_features(
                    audio_np, d0_checkpoint=args.d0_checkpoint,
                    device=device, batch_size=args.batch_size,
                )
            elif enc_name in ('MERT-v1-95M', 'MERT-v1-330M'):
                hf_id = f'm-a-p/{enc_name}'
                feat = extract_hf_mert_features(
                    audio_np, model_name=hf_id,
                    device=device, batch_size=args.batch_size,
                )
            else:
                # Assume it's a full HF model ID
                feat = extract_hf_mert_features(
                    audio_np, model_name=enc_name,
                    device=device, batch_size=args.batch_size,
                )

            save_features(cache_path, feat)
            encoder_features[enc_name] = feat

    # ── Phase 4: Segment-level probe (primary endpoint) ───────────────────────
    logger.info("\n── Segment-Level LinearProbe (primary) ──")
    seg_results = {}

    for enc_name, feat in encoder_features.items():
        logger.info(f"\n{enc_name} (hidden_size={feat.get('hidden_size', '?')}):")
        seg_results[enc_name] = run_segment_probe(
            features_np=feat['pooled'],
            a4_np=a4_np,
            piece_ids=piece_ids,
            n_splits=args.n_splits,
            base_seed=args.seed,
            encoder_name=enc_name,
        )
        logger.info(
            f"  R²_global = {seg_results[enc_name]['r2_global_mean']:.3f} "
            f"± {seg_results[enc_name]['r2_global_std']:.3f}"
        )

    # ── Phase 5: Null controls ────────────────────────────────────────────────
    logger.info("\n── Null controls ──")
    # Use first encoder's features for null (doesn't matter which)
    first_enc = list(encoder_features.keys())[0]
    first_feat = encoder_features[first_enc]['pooled']

    null_shuffled = run_null_shuffled_between(
        features_np=first_feat,
        a4_np=a4_np,
        piece_ids=piece_ids,
        n_splits=args.n_splits,
        base_seed=args.seed,
    )
    logger.info(
        f"  Null shuffled_between: R²_global = {null_shuffled['r2_global_mean']:.3f} "
        f"± {null_shuffled['r2_global_std']:.3f}"
    )
    if null_shuffled['r2_global_mean'] > 0.05:
        logger.warning("  ⚠ NULL R² > 0.05 — possible protocol bug!")

    null_dummy = run_null_dummy(
        a4_np=a4_np,
        piece_ids=piece_ids,
        n_splits=args.n_splits,
        base_seed=args.seed,
    )
    logger.info(f"  Null dummy: R²_global = {null_dummy['r2_global_mean']:.3f}")

    # ── Phase 6: Frame-level probe (secondary, optional) ─────────────────────
    frame_results = {}
    if args.frame_level:
        logger.info("\n── Frame-Level LinearProbe (secondary) ──")
        for enc_name, feat in encoder_features.items():
            frames = feat.get('frames')
            if frames is None or frames.ndim != 3:
                logger.warning(f"  {enc_name}: no frame features, skipping")
                continue
            T_mert = frames.shape[1]
            logger.info(f"  {enc_name}: T_mert={T_mert}")

            # Compute A4 at MERT resolution for this encoder
            a4_frame_np = compute_a4_frame_targets(
                audio_np, T_target=T_mert, device=device, batch_size=16,
            )
            frame_results[enc_name] = run_frame_probe(
                features_np=frames,
                a4_frame_np=a4_frame_np,
                piece_ids=piece_ids,
                n_splits=args.n_splits,
                base_seed=args.seed,
                encoder_name=enc_name,
            )
            logger.info(
                f"  {enc_name} frame R²_global = "
                f"{frame_results[enc_name]['frame_level_real']['r2_global_mean']:.3f} | "
                f"time_shuffle = "
                f"{frame_results[enc_name]['frame_level_time_shuffle']['r2_global_mean']:.3f}"
            )

    # ── Phase 7: Per-layer probe (Exp 7.0b, optional) ─────────────────────────
    per_layer_result = {}
    if args.per_layer:
        logger.info("\n── Exp 7.0b: Per-Layer Probe (MERT-v1-330M) ──")
        cache_path = features_dir / 'MERT_v1_330M_per_layer.npz'
        if cache_path.exists() and not args.skip_cache:
            logger.info("Loading cached per-layer features...")
            data = np.load(str(cache_path))
            pld = data['per_layer_pooled']
            n_layers = int(data['n_layers'])
            H = int(data['hidden_size'])
        else:
            pl_feat = extract_hf_mert_per_layer(
                audio_np, model_name='m-a-p/MERT-v1-330M',
                device=device, batch_size=8,
            )
            pld = pl_feat['per_layer_pooled']
            n_layers = pl_feat['n_layers']
            H = pl_feat['hidden_size']
            np.savez_compressed(
                str(cache_path),
                per_layer_pooled=pld, n_layers=n_layers, hidden_size=H,
            )

        per_layer_result = run_per_layer_probe(
            per_layer_pooled=pld,
            a4_np=a4_np,
            piece_ids=piece_ids,
            n_splits=args.n_splits,
            base_seed=args.seed,
        )

    # ── Assemble final results ─────────────────────────────────────────────────
    total_time = time.time() - t_start

    final = {
        'metadata': {
            'N_segments': N,
            'N_pieces': int(len(np.unique(piece_ids))),
            'n_splits': args.n_splits,
            'seed': args.seed,
            'ridge_alpha': RIDGE_ALPHA,
            'train_frac': TRAIN_FRAC,
            'max_segs_per_piece': MAX_SEGS_PER_PIECE,
            'encoders': args.encoders,
            'total_time_minutes': round(total_time / 60, 2),
            'a4_band_names': A4_BAND_NAMES,
        },
        'segment_level': seg_results,
        'nulls': {
            'shuffled_between': null_shuffled,
            'dummy': null_dummy,
        },
    }
    if frame_results:
        final['frame_level'] = frame_results
    if per_layer_result:
        final['per_layer'] = per_layer_result

    # ── Save JSON ──────────────────────────────────────────────────────────────
    json_path = results_dir / 'probe_results.json'
    with open(json_path, 'w') as f:
        json.dump(final, f, indent=2)
    logger.info(f"\nResults saved: {json_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    make_plots(final, results_dir)

    # ── Summary table ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("Gate 7 — SUMMARY (segment-level LinearProbe)")
    logger.info("=" * 65)
    logger.info(f"{'Encoder':<22} {'R²_global':>12} {'± std':>8} {'H':>6}")
    logger.info("-" * 55)
    for enc_name in args.encoders:
        if enc_name not in seg_results:
            continue
        r = seg_results[enc_name]
        H = encoder_features[enc_name].get('hidden_size', '?')
        logger.info(
            f"{enc_name:<22} {r['r2_global_mean']:>12.3f} {r['r2_global_std']:>8.3f} {H:>6}"
        )
    logger.info("-" * 55)
    logger.info(
        f"{'Null (shuffled)':<22} {null_shuffled['r2_global_mean']:>12.3f} "
        f"{null_shuffled['r2_global_std']:>8.3f}   —"
    )
    logger.info(
        f"{'Null (dummy)':<22} {null_dummy['r2_global_mean']:>12.3f} "
        f"{null_dummy['r2_global_std']:>8.3f}   —"
    )
    logger.info("=" * 65)
    logger.info(f"\nTotal time: {total_time/60:.1f} min")
    logger.info(f"Results: {json_path}")


if __name__ == '__main__':
    main()
