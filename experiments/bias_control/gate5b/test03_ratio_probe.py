#!/usr/bin/env python3
"""
Gate 5B — Test 3: RatioProbeDecoder.

"Smoking gun" test: trains a linear probe on FROZEN embeddings to predict
cross-modal features. If the model truly learned cross-modal ratio structure,
z_audio should predict MIDI features and vice versa.

Method:
  1. Freeze best model, extract embeddings for all validation segments
  2. Compute ground-truth features:
     - MIDI features: pitch histogram (128-bin), interval distribution (25-bin)
     - Audio features: spectral centroid, chroma (12-bin)
  3. Train probes:
     - z_audio [256] → MIDI features (cross-decoding: does audio embed know about MIDI?)
     - z_midi [256] → Audio features (cross-decoding: does MIDI embed know about audio?)
     - z_audio [256] → Audio features (self-decoding: sanity check)
     - z_midi [256] → MIDI features (self-decoding: sanity check)
  4. Report R² for each probe, compare D0 vs d4a4/a4r/d4-a4r

NOTE: This test requires training linear probes. Can run on LOCAL (fast, few minutes)
or UNC for multi-seed.

Usage:
    python experiments/bias_control/gate5b/test03_ratio_probe.py \
        --model models/gate5b/d4a4/best_model.pt

    python experiments/bias_control/gate5b/test03_ratio_probe.py --all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS = {
    'D0': 'models/gate5b/D0/best_model.pt',
    'd4a4': 'models/gate5b/d4a4/best_model.pt',
    'a4r': 'models/gate5b/a4r/best_model.pt',
    'd4-a4r': 'models/gate5b/d4-a4r/best_model.pt',
}


class LinearProbe(nn.Module):
    """Simple linear probe: embedding → target features."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.probe(x)


@torch.no_grad()
def compute_midi_features(dataset, n_pitch_bins=128, n_interval_bins=25, max_samples=5000):
    """Compute MIDI ground-truth features for all segments (vectorized)."""
    from src.bias_control.datasets.maestro_segments import collate_segments

    loader = DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=8,
        collate_fn=collate_segments, pin_memory=True,
    )

    pitch_hists = []
    interval_hists = []
    collected = 0

    for batch in tqdm(loader, desc="Computing MIDI features"):
        midi_pitch = batch['midi_pitch']  # [B, T]
        midi_mask = batch['midi_mask']    # [B, T]
        B = midi_pitch.size(0)

        # Vectorized: process entire batch at once
        for b in range(B):
            valid = ~midi_mask[b]
            pitches = midi_pitch[b][valid].long()

            # Pitch histogram via bincount (vectorized)
            if pitches.numel() > 0:
                clamped = pitches.clamp(0, n_pitch_bins - 1)
                hist = torch.bincount(clamped, minlength=n_pitch_bins).float()
                hist = hist / (hist.sum() + 1e-8)
            else:
                hist = torch.zeros(n_pitch_bins)
            pitch_hists.append(hist)

            # Interval histogram via bincount (vectorized)
            if pitches.numel() > 1:
                intervals = pitches[1:] - pitches[:-1]
                bin_idx = (intervals + 12).clamp(0, n_interval_bins - 1)
                int_hist = torch.bincount(bin_idx, minlength=n_interval_bins).float()
                int_hist = int_hist / (int_hist.sum() + 1e-8)
            else:
                int_hist = torch.zeros(n_interval_bins)
            interval_hists.append(int_hist)

        collected += B
        if collected >= max_samples:
            break

    return {
        'pitch_hist': torch.stack(pitch_hists[:max_samples]),
        'interval_hist': torch.stack(interval_hists[:max_samples]),
    }


@torch.no_grad()
def compute_audio_features(dataset, max_samples=5000):
    """Compute audio ground-truth features (spectral centroid, chroma) — vectorized."""
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8,
                        collate_fn=lambda b: {'audio': torch.stack([x['audio'] for x in b])},
                        pin_memory=True)

    centroids = []
    chromas = []
    collected = 0

    for batch in tqdm(loader, desc="Computing audio features"):
        audio = batch['audio']  # [B, 64000]
        B = audio.size(0)

        # Batch FFT (vectorized over entire batch)
        specs = torch.fft.rfft(audio)  # [B, F]
        mags = specs.abs()  # [B, F]
        F = mags.size(1)

        freqs = torch.arange(F, dtype=torch.float32)  # [F]

        # Spectral centroid — fully vectorized
        cents = (freqs.unsqueeze(0) * mags).sum(dim=1) / (mags.sum(dim=1) + 1e-8)  # [B]
        centroids.append(cents)

        # Chroma — vectorized scatter_add
        n_fft = (F - 1) * 2
        sr = 16000
        freq_hz = freqs * sr / n_fft  # [F]
        valid_mask = freq_hz > 20  # [F]
        valid_freqs = freq_hz[valid_mask]  # [F']
        midi_notes = 69 + 12 * torch.log2(valid_freqs / 440.0 + 1e-8)
        chroma_idx = (midi_notes.long() % 12).clamp(0, 11)  # [F']

        valid_mags = mags[:, valid_mask]  # [B, F']
        batch_chroma = torch.zeros(B, 12)
        batch_chroma.scatter_add_(1, chroma_idx.unsqueeze(0).expand(B, -1), valid_mags)
        batch_chroma = batch_chroma / (batch_chroma.sum(dim=1, keepdim=True) + 1e-8)
        chromas.append(batch_chroma)

        collected += B
        if collected >= max_samples:
            break

    return {
        'spectral_centroid': torch.cat(centroids)[:max_samples],
        'chroma': torch.cat(chromas)[:max_samples],
    }


def train_probe(embeddings, targets, n_epochs=50, lr=1e-3, test_frac=0.2, device='cpu'):
    """Train a linear probe on GPU and return R² on held-out test set."""
    N = embeddings.size(0)
    n_test = max(1, int(N * test_frac))
    n_train = N - n_test

    # Split
    perm = torch.randperm(N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    Y_train, Y_test = targets[train_idx], targets[test_idx]

    # Normalize
    X_mean, X_std = X_train.mean(0), X_train.std(0).clamp(min=1e-8)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    Y_mean, Y_std = Y_train.mean(0), Y_train.std(0).clamp(min=1e-8)
    Y_train_norm = (Y_train - Y_mean) / Y_std
    Y_test_norm = (Y_test - Y_mean) / Y_std

    # Move to device
    dev = torch.device(device)
    X_train, X_test = X_train.to(dev), X_test.to(dev)
    Y_train_norm, Y_test_norm = Y_train_norm.to(dev), Y_test_norm.to(dev)

    # Train
    input_dim = X_train.size(1)
    output_dim = Y_train.size(1) if Y_train.dim() > 1 else 1
    probe = LinearProbe(input_dim, output_dim).to(dev)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    dataset = TensorDataset(X_train, Y_train_norm)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    probe.train()
    for epoch in range(n_epochs):
        for xb, yb in loader:
            pred = probe(xb)
            if yb.dim() == 1:
                pred = pred.squeeze(-1)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate R²
    probe.eval()
    with torch.no_grad():
        pred_test = probe(X_test)
        if Y_test_norm.dim() == 1:
            pred_test = pred_test.squeeze(-1)

        ss_res = ((Y_test_norm - pred_test) ** 2).sum().item()
        ss_tot = ((Y_test_norm - Y_test_norm.mean(0)) ** 2).sum().item()
        r2 = 1 - ss_res / (ss_tot + 1e-8)

    return float(r2)


def evaluate_single(model_path, maestro_dir, device, num_workers, seed,
                    cached_features=None):
    """Run ratio probe analysis on one checkpoint.

    If cached_features is provided, skip ground-truth feature computation
    (features are identical across models — same dataset, same order).
    """
    from experiments.bias_control.gate5b.harness import (
        setup_gate5b_test, get_normal_embeddings, get_output_dir, save_test_result,
    )

    class Args:
        pass
    args = Args()
    args.model = model_path
    args.maestro_dir = maestro_dir
    args.device = device
    args.seed = seed
    args.num_workers = num_workers
    args.output = None

    torch.manual_seed(seed)
    np.random.seed(seed)

    model, meta, dataset, index = setup_gate5b_test(args)
    descriptor = meta['descriptor']
    output_dir = get_output_dir(args, meta)

    # Extract embeddings (uses cache if available)
    logger.info("Loading/extracting embeddings...")
    audio_embs, midi_embs = get_normal_embeddings(
        model, dataset, device, descriptor, output_dir,
        num_workers=num_workers,
    )

    # Compute or reuse ground-truth features
    max_samples = min(audio_embs.size(0), 5000)
    if cached_features is not None:
        logger.info(f"Reusing cached ground-truth features (N={max_samples})")
        midi_features = cached_features['midi']
        audio_features = cached_features['audio']
    else:
        logger.info(f"Computing ground-truth features (N={max_samples})...")
        midi_features = compute_midi_features(dataset, max_samples=max_samples)
        audio_features = compute_audio_features(dataset, max_samples=max_samples)

    # Ensure same size
    N = min(audio_embs.size(0), midi_features['pitch_hist'].size(0),
            audio_features['chroma'].size(0))
    audio_embs = audio_embs[:N]
    midi_embs = midi_embs[:N]

    # Train probes (on GPU)
    probes = {}

    # Cross-decoding: audio_emb → MIDI features
    logger.info("Training probe: z_audio → pitch_hist")
    probes['audio_to_pitch_hist'] = train_probe(
        audio_embs, midi_features['pitch_hist'][:N], device=device)

    logger.info("Training probe: z_audio → interval_hist")
    probes['audio_to_interval_hist'] = train_probe(
        audio_embs, midi_features['interval_hist'][:N], device=device)

    # Cross-decoding: midi_emb → Audio features
    logger.info("Training probe: z_midi → chroma")
    probes['midi_to_chroma'] = train_probe(
        midi_embs, audio_features['chroma'][:N], device=device)

    logger.info("Training probe: z_midi → spectral_centroid")
    probes['midi_to_centroid'] = train_probe(
        midi_embs, audio_features['spectral_centroid'][:N].unsqueeze(1), device=device)

    # Self-decoding (sanity check)
    logger.info("Training probe: z_audio → chroma (self)")
    probes['audio_to_chroma_self'] = train_probe(
        audio_embs, audio_features['chroma'][:N], device=device)

    logger.info("Training probe: z_midi → pitch_hist (self)")
    probes['midi_to_pitch_hist_self'] = train_probe(
        midi_embs, midi_features['pitch_hist'][:N], device=device)

    result = {
        'probes': probes,
        'n_samples': N,
        'cross_decoding': {
            'audio_to_midi': {
                'pitch_hist_r2': probes['audio_to_pitch_hist'],
                'interval_hist_r2': probes['audio_to_interval_hist'],
            },
            'midi_to_audio': {
                'chroma_r2': probes['midi_to_chroma'],
                'centroid_r2': probes['midi_to_centroid'],
            },
        },
        'self_decoding': {
            'audio_to_chroma_r2': probes['audio_to_chroma_self'],
            'midi_to_pitch_hist_r2': probes['midi_to_pitch_hist_self'],
        },
    }

    output_dir = get_output_dir(args, meta)
    save_test_result(result, output_dir, 'test03_ratio_probe', meta, seed=seed)

    features_cache = {'midi': midi_features, 'audio': audio_features}
    return result, meta, features_cache


def main():
    parser = argparse.ArgumentParser(description='Gate 5B — Test 3: RatioProbeDecoder')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Provide --model PATH or --all")

    models = {}
    if args.all:
        models = dict(DEFAULT_CHECKPOINTS)
    else:
        models = {'custom': args.model}

    all_results = {}
    cached_features = None  # Compute once, reuse across models

    for arm, path in models.items():
        if not Path(path).exists():
            logger.warning(f"Checkpoint not found: {path} — skipping {arm}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"RATIO PROBE: {arm}")
        logger.info(f"{'='*60}")

        start = time.time()
        result, meta, features_cache = evaluate_single(
            path, args.maestro_dir, args.device, args.num_workers, args.seed,
            cached_features=cached_features,
        )
        elapsed = time.time() - start

        # Cache features from first model for reuse
        if cached_features is None:
            cached_features = features_cache
            logger.info("Ground-truth features cached — will reuse for remaining models")

        all_results[arm] = {
            'descriptor': meta['descriptor'],
            'probes': result['probes'],
            'time_s': elapsed,
        }

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("RATIO PROBE SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Arm':<10} {'a→pitch':>10} {'a→intv':>10} {'m→chroma':>10} {'m→cent':>10} {'a→chr(s)':>10} {'m→pit(s)':>10}")
    logger.info("-" * 70)

    for arm, data in all_results.items():
        p = data['probes']
        logger.info(
            f"{arm:<10} "
            f"{p.get('audio_to_pitch_hist', 0):>9.3f} "
            f"{p.get('audio_to_interval_hist', 0):>9.3f} "
            f"{p.get('midi_to_chroma', 0):>9.3f} "
            f"{p.get('midi_to_centroid', 0):>9.3f} "
            f"{p.get('audio_to_chroma_self', 0):>9.3f} "
            f"{p.get('midi_to_pitch_hist_self', 0):>9.3f}"
        )


if __name__ == '__main__':
    main()
