#!/usr/bin/env python3
"""
Gate 5B - Test 11: Decoder Suite (Cross-Modal + Intra-Domain Generation).

A powerful Transformer decoder reconstructs full temporal sequences
(mel spectrogram [188,128], piano roll [188,88]) from frozen 256-dim embeddings.

4 decoder tasks per arm:
  - audio2mel: z_audio -> mel      (intra-domain)
  - midi2pr:   z_midi  -> PR       (intra-domain)
  - audio2pr:  z_audio -> PR       (cross-modal)
  - midi2mel:  z_midi  -> mel      (cross-modal)

2 random baselines (shared across arms):
  - random2mel: z~N(0,1) -> mel
  - random2pr:  z~N(0,1) -> PR

Controls (eval-only, no separate training):
  - shuffle: same decoder with z from different segment (derangement)
  - mean_z:  z = mean(z_train)
  - zero_z:  z = 0

Execution order: D0 -> a4r -> d4a4. d4-a4r optional.

Usage:
    # Precompute targets only
    python experiments/bias_control/gate5b/test11_decoder_suite.py --precompute-only

    # Single arm
    python experiments/bias_control/gate5b/test11_decoder_suite.py \
        --model models/gate5b/D0/best_model.pt

    # All arms (D0 -> a4r -> d4a4)
    python experiments/bias_control/gate5b/test11_decoder_suite.py --all

    # Skip precompute if caches exist
    python experiments/bias_control/gate5b/test11_decoder_suite.py \
        --model models/gate5b/a4r/best_model.pt --skip-precompute
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

GATE5B_RESULTS_DIR = Path('data/gate5b_results')
N_TRAIN_SUBSAMPLE = 20_000
N_FRAMES = 188
N_MELS = 128
N_PITCHES = 88
SR = 24000
HOP_LENGTH = 512
N_FFT = 2048
FMIN = 20
FMAX = 12000

DEFAULT_CHECKPOINTS = {
    'D0': 'models/gate5b/D0/best_model.pt',
    'd4a4': 'models/gate5b/d4a4/best_model.pt',
    'a4r': 'models/gate5b/a4r/best_model.pt',
    'd4-a4r': 'models/gate5b/d4-a4r/best_model.pt',
}

# Training config
TRAIN_CONFIG = {
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 60,
    'batch_size': 64,
    'patience': 10,
    'min_delta': 1e-5,
    'grad_clip': 1.0,
    'mel_l1_weight': 0.1,
    'pr_pos_weight': 50.0,
    'pr_threshold': 0.1,
    'onset_tolerance': 2,  # frames
}


# ============================================================
# A. Precompute Targets
# ============================================================

def _parse_midi_for_piece(midi_path: Path) -> list:
    """Parse MIDI file, return list of (pitch, onset, offset, velocity)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append((n.pitch, n.start, n.end, n.velocity))
    notes.sort(key=lambda x: (x[1], x[0]))
    return notes


def _build_piano_roll_on_mel_grid(notes: list, start_time: float, segment_len: float = 4.0) -> np.ndarray:
    """
    Build piano roll on mel temporal grid (T=188).

    Uses floor for onset, ceil for offset to avoid systematic note shortening.
    Segment-relative time: note times are clipped to [start_time, start_time+segment_len].
    """
    pr = np.zeros((N_FRAMES, N_PITCHES), dtype=np.float32)
    for pitch, onset, offset, velocity in notes:
        pitch_idx = pitch - 21  # A0=21
        if pitch_idx < 0 or pitch_idx >= N_PITCHES:
            continue

        # Convert to segment-relative time + clip edges
        onset_rel = max(0.0, onset - start_time)
        offset_rel = min(segment_len, offset - start_time)
        if offset_rel <= onset_rel:
            continue

        frame_start = int(np.floor(onset_rel * SR / HOP_LENGTH))
        frame_end = int(np.ceil(offset_rel * SR / HOP_LENGTH))
        frame_start = max(0, min(frame_start, N_FRAMES - 1))
        frame_end = max(frame_start + 1, min(frame_end, N_FRAMES))

        intensity = velocity / 127.0
        pr[frame_start:frame_end, pitch_idx] = np.maximum(
            pr[frame_start:frame_end, pitch_idx], intensity
        )
    return pr


def precompute_targets(
    maestro_dir: str,
    output_dir: Path,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute mel spectrograms and piano rolls for train (subsampled) + val.

    Saves:
        targets_mel_train.npz, targets_mel_val.npz
        targets_pr_train.npz, targets_pr_val.npz
        train_indices.npy

    Returns:
        (train_indices, val_size)
    """
    import librosa
    from src.bias_control.datasets.maestro_segments import MaestroSegmentDataset

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already computed
    all_exist = all(
        (output_dir / f).exists()
        for f in ['targets_mel_train.npz', 'targets_mel_val.npz',
                   'targets_pr_train.npz', 'targets_pr_val.npz',
                   'train_indices.npy']
    )
    if all_exist:
        train_indices = np.load(output_dir / 'train_indices.npy')
        logger.info(f"Target caches already exist, skipping precompute. "
                    f"Train indices: {len(train_indices)}")
        return train_indices, None

    logger.info("Precomputing targets (mel spectrograms + piano rolls)...")

    # --- Subsample train indices ---
    rng = np.random.RandomState(seed)
    train_ds = MaestroSegmentDataset(
        maestro_dir=Path(maestro_dir), segment_len=4.0, hop=1.0, split='train',
    )
    n_train = len(train_ds)
    if n_train > N_TRAIN_SUBSAMPLE:
        train_indices = np.sort(rng.choice(n_train, N_TRAIN_SUBSAMPLE, replace=False))
    else:
        train_indices = np.arange(n_train)
    np.save(output_dir / 'train_indices.npy', train_indices)
    logger.info(f"Train: {n_train} total, subsampled {len(train_indices)}")

    val_ds = MaestroSegmentDataset(
        maestro_dir=Path(maestro_dir), segment_len=4.0, hop=1.0, split='validation',
    )
    n_val = len(val_ds)
    logger.info(f"Val: {n_val} segments")

    # --- Cache MIDI parses per piece ---
    def _cache_midi_parses(dataset):
        """Parse MIDI once per piece, cache results."""
        cache = {}
        for seg in dataset.segments:
            if seg.piece_idx not in cache:
                cache[seg.piece_idx] = _parse_midi_for_piece(seg.midi_path)
        return cache

    # Process each split
    for split_name, dataset, indices in [
        ('train', train_ds, train_indices),
        ('val', val_ds, np.arange(n_val)),
    ]:
        n = len(indices)
        logger.info(f"Processing {split_name}: {n} segments...")
        t0 = time.time()

        midi_cache = _cache_midi_parses(dataset)
        logger.info(f"  MIDI cache: {len(midi_cache)} pieces parsed")

        mel_data = np.zeros((n, N_FRAMES, N_MELS), dtype=np.float16)
        pr_data = np.zeros((n, N_FRAMES, N_PITCHES), dtype=np.float16)

        for out_i, ds_idx in enumerate(indices):
            seg = dataset.segments[ds_idx]

            # --- Mel spectrogram ---
            try:
                audio, _ = librosa.load(
                    str(seg.audio_path), sr=SR,
                    offset=seg.start_time, duration=4.0,
                )
                expected = int(4.0 * SR)
                if len(audio) < expected:
                    audio = np.pad(audio, (0, expected - len(audio)))
                elif len(audio) > expected:
                    audio = audio[:expected]

                S = librosa.feature.melspectrogram(
                    y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                    n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
                )
                log_mel = np.log1p(S).T  # [128, T] -> [T, 128]

                # Ensure T=188 (may be 188 or 189 depending on librosa)
                if log_mel.shape[0] > N_FRAMES:
                    log_mel = log_mel[:N_FRAMES]
                elif log_mel.shape[0] < N_FRAMES:
                    pad = np.zeros((N_FRAMES - log_mel.shape[0], N_MELS), dtype=np.float32)
                    log_mel = np.concatenate([log_mel, pad], axis=0)

                mel_data[out_i] = log_mel.astype(np.float16)
            except Exception as e:
                logger.warning(f"  Mel failed idx={ds_idx}: {e}")

            # --- Piano roll on mel grid ---
            try:
                piece_notes = midi_cache[seg.piece_idx]
                # Filter notes overlapping with segment
                seg_notes = [
                    (p, on, off, v) for p, on, off, v in piece_notes
                    if off > seg.start_time and on < seg.start_time + 4.0
                ]
                pr = _build_piano_roll_on_mel_grid(seg_notes, seg.start_time, 4.0)
                pr_data[out_i] = pr.astype(np.float16)
            except Exception as e:
                logger.warning(f"  PR failed idx={ds_idx}: {e}")

            if (out_i + 1) % 2000 == 0 or out_i == n - 1:
                elapsed = time.time() - t0
                rate = (out_i + 1) / elapsed
                eta = (n - out_i - 1) / rate if rate > 0 else 0
                logger.info(f"  {split_name}: {out_i+1}/{n} ({rate:.1f} seg/s, ETA {eta:.0f}s)")

        # Save compressed
        mel_path = output_dir / f'targets_mel_{split_name}.npz'
        pr_path = output_dir / f'targets_pr_{split_name}.npz'
        np.savez_compressed(mel_path, data=mel_data)
        np.savez_compressed(pr_path, data=pr_data)

        elapsed = time.time() - t0
        logger.info(f"  {split_name} done in {elapsed:.0f}s. "
                    f"Mel: {mel_path} ({mel_path.stat().st_size/1e9:.2f} GB), "
                    f"PR: {pr_path} ({pr_path.stat().st_size/1e9:.2f} GB)")

    return train_indices, n_val


def load_targets(output_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load precomputed mel and PR targets for a split."""
    mel = np.load(output_dir / f'targets_mel_{split}.npz')['data']
    pr = np.load(output_dir / f'targets_pr_{split}.npz')['data']
    return mel.astype(np.float32), pr.astype(np.float32)


# ============================================================
# B. Extract Train Embeddings
# ============================================================

def extract_train_embeddings(
    model_path: str,
    maestro_dir: str,
    train_indices: np.ndarray,
    output_dir: Path,
    device: str = 'cuda',
    num_workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract audio + midi embeddings for train subsample.

    Uses Subset(sorted indices) + shuffle=False for deterministic order
    matching the target cache.

    Returns:
        (audio_embs, midi_embs): both [N, 256] numpy arrays
    """
    cache_path = output_dir / 'embeddings_train.npz'
    if cache_path.exists():
        data = np.load(cache_path)
        cached_indices = data['indices']
        if np.array_equal(cached_indices, train_indices):
            logger.info(f"Train embeddings cache hit: {cache_path}")
            return data['audio_embs'], data['midi_embs']
        else:
            logger.warning("Train indices mismatch, re-extracting...")

    from experiments.bias_control.gate5b.checkpoint_loader import (
        load_model_from_checkpoint, get_eval_batch_size,
    )
    from src.bias_control.datasets.maestro_segments import (
        MaestroSegmentDataset, collate_segments,
    )

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model, meta = load_model_from_checkpoint(model_path, device=dev)
    descriptor = meta['descriptor']
    batch_size = get_eval_batch_size(descriptor)

    train_ds = MaestroSegmentDataset(
        maestro_dir=Path(maestro_dir), segment_len=4.0, hop=1.0, split='train',
    )

    # Subset with sorted indices — deterministic order
    subset = Subset(train_ds, sorted(train_indices.tolist()))
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_segments,
        pin_memory=True, prefetch_factor=2,
    )

    logger.info(f"Extracting train embeddings: {len(subset)} segments, "
                f"batch_size={batch_size}, descriptor={descriptor}")

    audio_embs_list = []
    midi_embs_list = []
    t0 = time.time()

    with torch.no_grad():
        for batch in loader:
            audio = batch['audio'].to(dev, non_blocking=True)
            midi_pitch = batch['midi_pitch'].to(dev, non_blocking=True)
            midi_velocity = batch['midi_velocity'].to(dev, non_blocking=True)
            midi_duration = batch['midi_duration'].to(dev, non_blocking=True)
            midi_mask = batch['midi_mask'].to(dev, non_blocking=True)

            z_audio, z_midi = model(
                audio, midi_pitch, midi_velocity, midi_duration, midi_mask,
            )
            audio_embs_list.append(z_audio.cpu().numpy())
            midi_embs_list.append(z_midi.cpu().numpy())

    audio_embs = np.concatenate(audio_embs_list, axis=0)
    midi_embs = np.concatenate(midi_embs_list, axis=0)

    elapsed = time.time() - t0
    logger.info(f"Train extraction done in {elapsed:.0f}s: "
                f"audio={audio_embs.shape}, midi={midi_embs.shape}")

    # Save with indices for alignment verification
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        audio_embs=audio_embs, midi_embs=midi_embs,
        indices=train_indices,
    )
    logger.info(f"Saved: {cache_path}")

    # Free model VRAM
    del model
    torch.cuda.empty_cache()

    return audio_embs, midi_embs


# ============================================================
# C. Train Decoder
# ============================================================

def _mel_loss_fn(pred, target):
    """MSE + 0.1*L1 on log(1+mel)."""
    return nn.functional.mse_loss(pred, target) + TRAIN_CONFIG['mel_l1_weight'] * nn.functional.l1_loss(pred, target)


def _pr_loss_fn(pred_logits, target):
    """BCEWithLogits with pos_weight for sparse piano roll."""
    pos_weight = torch.tensor([TRAIN_CONFIG['pr_pos_weight']], device=pred_logits.device)
    return nn.functional.binary_cross_entropy_with_logits(
        pred_logits, target, pos_weight=pos_weight,
    )


def train_decoder(
    task_name: str,
    train_z: np.ndarray,
    train_targets: np.ndarray,
    val_z: np.ndarray,
    val_targets: np.ndarray,
    output_dir: Path,
    device: str = 'cuda',
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a decoder for one task.

    Returns dict with training stats and best val metrics.
    """
    from experiments.bias_control.gate5b.decoder_model import create_decoder

    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    is_pr = 'pr' in task_name
    loss_fn = _pr_loss_fn if is_pr else _mel_loss_fn

    # Create decoder
    decoder = create_decoder(task_name).to(dev)
    logger.info(f"[{task_name}] Decoder: {decoder.n_params/1e6:.1f}M params, "
                f"output={'PR 88' if is_pr else 'mel 128'}")

    # Create dataloaders
    train_z_t = torch.from_numpy(train_z).float()
    train_tgt_t = torch.from_numpy(train_targets).float()
    val_z_t = torch.from_numpy(val_z).float()
    val_tgt_t = torch.from_numpy(val_targets).float()

    train_loader = DataLoader(
        TensorDataset(train_z_t, train_tgt_t),
        batch_size=TRAIN_CONFIG['batch_size'], shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_z_t, val_tgt_t),
        batch_size=TRAIN_CONFIG['batch_size'], shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=TRAIN_CONFIG['lr'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_CONFIG['epochs'],
    )

    # Training loop
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    model_dir = output_dir / 'test11_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / f'{task_name}_best.pt'

    t0 = time.time()

    for epoch in range(TRAIN_CONFIG['epochs']):
        # --- Train ---
        decoder.train()
        train_loss_sum = 0.0
        n_train_batches = 0

        for z_batch, tgt_batch in train_loader:
            z_batch = z_batch.to(dev, non_blocking=True)
            tgt_batch = tgt_batch.to(dev, non_blocking=True)

            pred = decoder(z_batch)
            loss = loss_fn(pred, tgt_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), TRAIN_CONFIG['grad_clip'])
            optimizer.step()

            train_loss_sum += loss.item()
            n_train_batches += 1

        scheduler.step()

        # --- Validate ---
        decoder.eval()
        val_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for z_batch, tgt_batch in val_loader:
                z_batch = z_batch.to(dev, non_blocking=True)
                tgt_batch = tgt_batch.to(dev, non_blocking=True)
                pred = decoder(z_batch)
                val_loss_sum += loss_fn(pred, tgt_batch).item()
                n_val_batches += 1

        val_loss = val_loss_sum / n_val_batches
        train_loss = train_loss_sum / n_train_batches

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  [{task_name}] e{epoch+1}: train={train_loss:.4f}, "
                        f"val={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        # Early stopping
        if val_loss < best_val_loss - TRAIN_CONFIG['min_delta']:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'model_state_dict': decoder.state_dict(),
                'epoch': best_epoch,
                'val_loss': best_val_loss,
                'task_name': task_name,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['patience']:
                logger.info(f"  [{task_name}] Early stopping at epoch {epoch+1} "
                            f"(best={best_epoch}, val_loss={best_val_loss:.4f})")
                break

    elapsed = time.time() - t0

    # Load best checkpoint
    ckpt = torch.load(save_path, map_location=dev)
    decoder.load_state_dict(ckpt['model_state_dict'])
    decoder.eval()

    logger.info(f"  [{task_name}] Training done in {elapsed:.0f}s, "
                f"best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")

    return {
        'decoder': decoder,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_time_sec': elapsed,
    }


# ============================================================
# D. Evaluate Decoder
# ============================================================

def _compute_onset_f1(pred_pr: np.ndarray, true_pr: np.ndarray, threshold: float = 0.1, tolerance: int = 2) -> Dict[str, float]:
    """
    Compute onset F1 with +-tolerance frame matching.

    Greedy closest-first, pitch-specific, 1-to-1 constraint.
    Tie-break: frame distance first, then GT index, then pred index.
    """
    def _detect_onsets(roll: np.ndarray, thresh: float) -> List[Tuple[int, int]]:
        """Detect onsets: (pitch, frame) for first frame of each active region."""
        onsets = []
        T, P = roll.shape
        for p in range(P):
            active = roll[:, p] > thresh
            for t in range(T):
                if active[t] and (t == 0 or not active[t - 1]):
                    onsets.append((p, t))
        return onsets

    pred_onsets = _detect_onsets(pred_pr, threshold)
    true_onsets = _detect_onsets(true_pr, threshold)

    if len(true_onsets) == 0 and len(pred_onsets) == 0:
        return {'onset_precision': 1.0, 'onset_recall': 1.0, 'onset_f1': 1.0}
    if len(true_onsets) == 0:
        return {'onset_precision': 0.0, 'onset_recall': 1.0, 'onset_f1': 0.0}
    if len(pred_onsets) == 0:
        return {'onset_precision': 1.0, 'onset_recall': 0.0, 'onset_f1': 0.0}

    # Build candidate matches: (distance, gt_idx, pred_idx) for same-pitch within tolerance
    candidates = []
    for gi, (gp, gt) in enumerate(true_onsets):
        for pi, (pp, pt) in enumerate(pred_onsets):
            if gp == pp and abs(gt - pt) <= tolerance:
                candidates.append((abs(gt - pt), gi, pi))

    # Sort by distance, then gt_idx, then pred_idx (deterministic)
    candidates.sort()

    matched_gt = set()
    matched_pred = set()
    tp = 0

    for dist, gi, pi in candidates:
        if gi not in matched_gt and pi not in matched_pred:
            tp += 1
            matched_gt.add(gi)
            matched_pred.add(pi)

    fp = len(pred_onsets) - tp
    fn = len(true_onsets) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {'onset_precision': precision, 'onset_recall': recall, 'onset_f1': f1}


@torch.no_grad()
def evaluate_decoder(
    decoder: nn.Module,
    val_z: np.ndarray,
    val_targets: np.ndarray,
    task_type: str,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate decoder on val set.

    Args:
        task_type: 'mel' or 'pr'

    Returns:
        Dict with metrics
    """
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    decoder.eval()

    z_t = torch.from_numpy(val_z).float()
    tgt_t = torch.from_numpy(val_targets).float()

    loader = DataLoader(
        TensorDataset(z_t, tgt_t),
        batch_size=TRAIN_CONFIG['batch_size'], shuffle=False,
    )

    if task_type == 'mel':
        mse_sum, l1_sum, cos_sum = 0.0, 0.0, 0.0
        n = 0
        for z_b, tgt_b in loader:
            z_b = z_b.to(dev, non_blocking=True)
            tgt_b = tgt_b.to(dev, non_blocking=True)
            pred = decoder(z_b)
            B = pred.shape[0]
            mse_sum += nn.functional.mse_loss(pred, tgt_b, reduction='sum').item()
            l1_sum += nn.functional.l1_loss(pred, tgt_b, reduction='sum').item()
            # Cosine similarity per frame, averaged
            cos = nn.functional.cosine_similarity(
                pred.reshape(B * N_FRAMES, N_MELS),
                tgt_b.reshape(B * N_FRAMES, N_MELS),
                dim=1,
            )
            cos_sum += cos.sum().item()
            n += B

        n_elements = n * N_FRAMES * N_MELS
        return {
            'mse': mse_sum / n_elements,
            'l1': l1_sum / n_elements,
            'cosine_sim': cos_sum / (n * N_FRAMES),
        }

    else:  # pr
        bce_sum = 0.0
        all_pred_pr = []
        all_true_pr = []
        n = 0

        pos_weight = torch.tensor([TRAIN_CONFIG['pr_pos_weight']], device=dev)
        for z_b, tgt_b in loader:
            z_b = z_b.to(dev, non_blocking=True)
            tgt_b = tgt_b.to(dev, non_blocking=True)
            pred_logits = decoder(z_b)
            bce_sum += nn.functional.binary_cross_entropy_with_logits(
                pred_logits, tgt_b, pos_weight=pos_weight, reduction='sum',
            ).item()
            pred_prob = torch.sigmoid(pred_logits).cpu().numpy()
            all_pred_pr.append(pred_prob)
            all_true_pr.append(tgt_b.cpu().numpy())
            n += z_b.shape[0]

        n_elements = n * N_FRAMES * N_PITCHES
        bce = bce_sum / n_elements

        all_pred_pr = np.concatenate(all_pred_pr, axis=0)
        all_true_pr = np.concatenate(all_true_pr, axis=0)

        # F1 at threshold
        pred_binary = (all_pred_pr > TRAIN_CONFIG['pr_threshold']).astype(np.float32)
        true_binary = (all_true_pr > TRAIN_CONFIG['pr_threshold']).astype(np.float32)
        tp = (pred_binary * true_binary).sum()
        fp = (pred_binary * (1 - true_binary)).sum()
        fn = ((1 - pred_binary) * true_binary).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Onset F1 on subset (first 500 for speed)
        n_onset_eval = min(500, len(all_pred_pr))
        onset_metrics_list = []
        for i in range(n_onset_eval):
            m = _compute_onset_f1(all_pred_pr[i], all_true_pr[i])
            onset_metrics_list.append(m)

        avg_onset = {
            k: np.mean([m[k] for m in onset_metrics_list])
            for k in onset_metrics_list[0]
        }

        return {
            'bce': bce,
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'onset_f1': avg_onset['onset_f1'],
            'onset_precision': avg_onset['onset_precision'],
            'onset_recall': avg_onset['onset_recall'],
        }


@torch.no_grad()
def evaluate_with_z(
    decoder: nn.Module,
    z_input: np.ndarray,
    val_targets: np.ndarray,
    task_type: str,
    device: str = 'cuda',
) -> float:
    """Evaluate decoder with arbitrary z, return primary loss (MSE or BCE)."""
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    decoder.eval()

    z_t = torch.from_numpy(z_input).float()
    tgt_t = torch.from_numpy(val_targets).float()
    loader = DataLoader(TensorDataset(z_t, tgt_t), batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)

    loss_sum = 0.0
    n = 0
    for z_b, tgt_b in loader:
        z_b = z_b.to(dev, non_blocking=True)
        tgt_b = tgt_b.to(dev, non_blocking=True)
        pred = decoder(z_b)
        if task_type == 'mel':
            loss_sum += nn.functional.mse_loss(pred, tgt_b, reduction='sum').item()
        else:
            pos_weight = torch.tensor([TRAIN_CONFIG['pr_pos_weight']], device=dev)
            loss_sum += nn.functional.binary_cross_entropy_with_logits(
                pred, tgt_b, pos_weight=pos_weight, reduction='sum',
            ).item()
        n += z_b.shape[0]

    n_elements = n * N_FRAMES * (N_MELS if task_type == 'mel' else N_PITCHES)
    return loss_sum / n_elements


def generate_derangement(n: int, seed: int = 42) -> np.ndarray:
    """Generate a derangement (permutation with no fixed points)."""
    rng = np.random.RandomState(seed)
    perm = np.arange(n)
    for _ in range(1000):  # retry until valid derangement
        rng.shuffle(perm)
        if not np.any(perm == np.arange(n)):
            return perm
    # Fallback: shift by 1 (always a derangement for n>1)
    return np.roll(np.arange(n), 1)


# ============================================================
# E. Generate Samples
# ============================================================

@torch.no_grad()
def generate_samples(
    decoder: nn.Module,
    val_z: np.ndarray,
    val_targets: np.ndarray,
    task_type: str,
    task_name: str,
    output_dir: Path,
    device: str = 'cuda',
    n_samples: int = 10,
):
    """
    Generate sample reconstructions (3 best, 4 median, 3 worst by loss).

    For mel: save .wav via Griffin-Lim
    For PR: save .mid via pretty_midi
    Also save ground truth for comparison.
    """
    import soundfile as sf
    import librosa

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    decoder.eval()
    samples_dir = output_dir / 'test11_samples'
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Compute per-sample loss to select diverse samples
    z_t = torch.from_numpy(val_z).float()
    tgt_t = torch.from_numpy(val_targets).float()

    losses = []
    preds = []
    bs = TRAIN_CONFIG['batch_size']
    for i in range(0, len(z_t), bs):
        z_b = z_t[i:i+bs].to(dev)
        tgt_b = tgt_t[i:i+bs].to(dev)
        pred = decoder(z_b)
        for j in range(pred.shape[0]):
            if task_type == 'mel':
                l = nn.functional.mse_loss(pred[j], tgt_b[j]).item()
            else:
                l = nn.functional.binary_cross_entropy_with_logits(
                    pred[j], tgt_b[j],
                    pos_weight=torch.tensor([TRAIN_CONFIG['pr_pos_weight']], device=dev),
                ).item()
            losses.append(l)
        preds.append(pred.cpu().numpy())

    losses = np.array(losses)
    preds = np.concatenate(preds, axis=0)
    sorted_idx = np.argsort(losses)

    # Select: 3 best, 4 median, 3 worst
    n_total = len(sorted_idx)
    mid = n_total // 2
    selected = list(sorted_idx[:3]) + list(sorted_idx[mid-2:mid+2]) + list(sorted_idx[-3:])
    selected = selected[:n_samples]

    for si, idx in enumerate(selected):
        pred_out = preds[idx]
        truth_out = val_targets[idx]

        if task_type == 'mel':
            # Mel -> wav via Griffin-Lim
            for label, data in [('sample', pred_out), ('truth', truth_out)]:
                mel_power = np.expm1(np.clip(data, 0, None))  # undo log1p
                mel_power = np.maximum(mel_power, 0).T  # [128, 188]
                audio = librosa.feature.inverse.mel_to_audio(
                    mel_power, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                    fmin=FMIN, fmax=FMAX, n_iter=64,
                )
                # Normalize
                peak = np.abs(audio).max() + 1e-8
                audio = audio / peak
                path = samples_dir / f'{task_name}_{label}_{si:02d}.wav'
                sf.write(str(path), audio, SR)

        else:  # pr
            import pretty_midi as pm_lib
            for label, data in [('sample', pred_out), ('truth', truth_out)]:
                # For predictions, data is logits -> sigmoid
                if label == 'sample':
                    probs = 1.0 / (1.0 + np.exp(-data))  # sigmoid
                else:
                    probs = data

                midi = pm_lib.PrettyMIDI(initial_tempo=120)
                piano = pm_lib.Instrument(program=0, name='Piano')

                # Threshold and extract note regions
                active = probs > TRAIN_CONFIG['pr_threshold']
                for p in range(N_PITCHES):
                    in_note = False
                    note_start = 0
                    for t in range(N_FRAMES):
                        if active[t, p] and not in_note:
                            in_note = True
                            note_start = t
                        elif not active[t, p] and in_note:
                            in_note = False
                            onset_sec = note_start * HOP_LENGTH / SR
                            offset_sec = t * HOP_LENGTH / SR
                            velocity = int(np.clip(probs[note_start:t, p].mean() * 127, 1, 127))
                            piano.notes.append(pm_lib.Note(
                                velocity=velocity,
                                pitch=p + 21,
                                start=onset_sec,
                                end=offset_sec,
                            ))
                    # Handle note active at end
                    if in_note:
                        onset_sec = note_start * HOP_LENGTH / SR
                        offset_sec = N_FRAMES * HOP_LENGTH / SR
                        velocity = int(np.clip(probs[note_start:N_FRAMES, p].mean() * 127, 1, 127))
                        piano.notes.append(pm_lib.Note(
                            velocity=velocity,
                            pitch=p + 21,
                            start=onset_sec,
                            end=offset_sec,
                        ))

                midi.instruments.append(piano)
                path = samples_dir / f'{task_name}_{label}_{si:02d}.mid'
                midi.write(str(path))

    logger.info(f"  [{task_name}] Generated {len(selected)} samples in {samples_dir}")


# ============================================================
# F. Run All Tasks for One Arm
# ============================================================

def run_arm(
    model_path: str,
    maestro_dir: str,
    results_dir: Path,
    train_indices: np.ndarray,
    device: str = 'cuda',
    seed: int = 42,
    num_workers: int = 8,
    skip_train_embs: bool = False,
) -> Dict[str, Any]:
    """
    Run all 4 decoder tasks for one arm.

    Returns the full result dict for JSON.
    """
    from experiments.bias_control.gate5b.checkpoint_loader import load_model_from_checkpoint
    from experiments.bias_control.gate5b.harness import (
        get_normal_embeddings, save_test_result,
    )

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Get arm name from checkpoint
    temp_model, meta = load_model_from_checkpoint(model_path, device=dev)
    descriptor = meta['descriptor']
    arm_dir = 'D0' if descriptor == 'd0' else descriptor
    output_dir = results_dir / arm_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    del temp_model
    torch.cuda.empty_cache()

    logger.info(f"\n{'='*60}")
    logger.info(f"ARM: {arm_dir} ({model_path})")
    logger.info(f"{'='*60}")

    # --- Load targets ---
    mel_train, pr_train = load_targets(results_dir, 'train')
    mel_val, pr_val = load_targets(results_dir, 'val')
    logger.info(f"Targets loaded: mel_train={mel_train.shape}, pr_train={pr_train.shape}")
    logger.info(f"                mel_val={mel_val.shape}, pr_val={pr_val.shape}")

    # --- Extract/load train embeddings ---
    cache_path = output_dir / 'embeddings_train.npz'
    if skip_train_embs and cache_path.exists():
        cache = np.load(cache_path)
        cached_indices = cache['indices']
        if np.array_equal(cached_indices, train_indices):
            logger.info(f"Using cached train embeddings (--skip-train-embs): {cache_path}")
            audio_train_z = cache['audio_embs']
            midi_train_z = cache['midi_embs']
        else:
            logger.warning("--skip-train-embs requested, but cache indices mismatch; re-extracting.")
            audio_train_z, midi_train_z = extract_train_embeddings(
                model_path, maestro_dir, train_indices, output_dir,
                device=device, num_workers=num_workers,
            )
    elif skip_train_embs and not cache_path.exists():
        logger.warning("--skip-train-embs requested, but cache not found; extracting.")
        audio_train_z, midi_train_z = extract_train_embeddings(
            model_path, maestro_dir, train_indices, output_dir,
            device=device, num_workers=num_workers,
        )
    else:
        audio_train_z, midi_train_z = extract_train_embeddings(
            model_path, maestro_dir, train_indices, output_dir,
            device=device, num_workers=num_workers,
        )

    # --- Get val embeddings (from existing cache or extract) ---
    # Reload model for val embeddings
    model, meta = load_model_from_checkpoint(model_path, device=dev)
    from src.bias_control.datasets.maestro_segments import MaestroSegmentDataset
    val_ds = MaestroSegmentDataset(
        maestro_dir=Path(maestro_dir), segment_len=4.0, hop=1.0, split='validation',
    )
    audio_val_z, midi_val_z = get_normal_embeddings(
        model, val_ds, device, descriptor, output_dir, num_workers=num_workers,
    )
    audio_val_z = audio_val_z.numpy()
    midi_val_z = midi_val_z.numpy()

    del model
    torch.cuda.empty_cache()

    logger.info(f"Embeddings: train audio={audio_train_z.shape}, midi={midi_train_z.shape}")
    logger.info(f"            val audio={audio_val_z.shape}, midi={midi_val_z.shape}")

    # --- Train 4 decoders ---
    tasks = {
        'audio2mel': ('mel', audio_train_z, mel_train, audio_val_z, mel_val),
        'midi2pr':   ('pr',  midi_train_z,  pr_train,  midi_val_z,  pr_val),
        'audio2pr':  ('pr',  audio_train_z, pr_train,  audio_val_z, pr_val),
        'midi2mel':  ('mel', midi_train_z,  mel_train, midi_val_z,  mel_val),
    }

    result = {}
    decoders = {}

    for task_name, (task_type, trn_z, trn_tgt, vl_z, vl_tgt) in tasks.items():
        logger.info(f"\n--- Training {task_name} ---")
        train_result = train_decoder(
            task_name, trn_z, trn_tgt, vl_z, vl_tgt,
            output_dir, device=device, seed=seed,
        )
        decoder = train_result['decoder']
        decoders[task_name] = (decoder, task_type, vl_z, vl_tgt)

        # Full evaluation
        metrics = evaluate_decoder(decoder, vl_z, vl_tgt, task_type, device=device)

        result[task_name] = {
            **{f'val_{k}': v for k, v in metrics.items()},
            'best_epoch': train_result['best_epoch'],
            'train_time_sec': train_result['train_time_sec'],
        }
        logger.info(f"  [{task_name}] Metrics: {metrics}")

    # --- Controls ---
    logger.info("\n--- Evaluating controls ---")
    controls = {}
    derangement = generate_derangement(len(audio_val_z), seed=seed)

    # Compute mean_z and zero_z
    audio_mean_z = np.mean(audio_val_z, axis=0, keepdims=True).repeat(len(audio_val_z), axis=0)
    midi_mean_z = np.mean(midi_val_z, axis=0, keepdims=True).repeat(len(midi_val_z), axis=0)
    zero_z = np.zeros_like(audio_val_z)

    control_configs = {
        'audio2mel': {
            'decoder': decoders['audio2mel'][0],
            'type': 'mel',
            'targets': mel_val,
            'shuffle_z': audio_val_z[derangement],
            'mean_z': audio_mean_z,
            'zero_z': zero_z,
        },
        'midi2pr': {
            'decoder': decoders['midi2pr'][0],
            'type': 'pr',
            'targets': pr_val,
            'shuffle_z': midi_val_z[derangement],
            'mean_z': midi_mean_z,
            'zero_z': zero_z,
        },
        'audio2pr': {
            'decoder': decoders['audio2pr'][0],
            'type': 'pr',
            'targets': pr_val,
            'shuffle_z': audio_val_z[derangement],
            'mean_z': audio_mean_z,
            'zero_z': zero_z,
        },
        'midi2mel': {
            'decoder': decoders['midi2mel'][0],
            'type': 'mel',
            'targets': mel_val,
            'shuffle_z': midi_val_z[derangement],
            'mean_z': midi_mean_z,
            'zero_z': zero_z,
        },
    }

    loss_key = {'mel': 'mse', 'pr': 'bce'}

    for task_name, cfg in control_configs.items():
        tt = cfg['type']
        lk = loss_key[tt]

        shuffle_loss = evaluate_with_z(cfg['decoder'], cfg['shuffle_z'], cfg['targets'], tt, device)
        mean_loss = evaluate_with_z(cfg['decoder'], cfg['mean_z'], cfg['targets'], tt, device)
        zero_loss = evaluate_with_z(cfg['decoder'], cfg['zero_z'], cfg['targets'], tt, device)

        controls[f'{task_name}_shuffle_{lk}'] = shuffle_loss
        controls[f'{task_name}_mean_z_{lk}'] = mean_loss
        controls[f'{task_name}_zero_z_{lk}'] = zero_loss
        logger.info(f"  [{task_name}] shuffle={shuffle_loss:.4f}, "
                    f"mean_z={mean_loss:.4f}, zero_z={zero_loss:.4f}")

    result['controls'] = controls

    # --- Info retention ratios ---
    logger.info("\n--- Info retention ratios ---")
    raw_losses = {}

    # Intra losses (aligned)
    raw_losses['intra_mel'] = result['audio2mel']['val_mse']
    raw_losses['cross_mel'] = result['midi2mel']['val_mse']
    # Use shuffle from the SAME decoder as cross_* to avoid mixing decoder-specific biases.
    raw_losses['shuffle_mel'] = controls['midi2mel_shuffle_mse']
    raw_losses['intra_pr'] = result['midi2pr']['val_bce']
    raw_losses['cross_pr'] = result['audio2pr']['val_bce']
    raw_losses['shuffle_pr'] = controls['audio2pr_shuffle_bce']

    # Load random baselines if available
    baselines_path = results_dir / 'baselines' / 'test11_decoder_suite.json'
    if baselines_path.exists():
        with open(baselines_path) as f:
            bl = json.load(f)
        raw_losses['random_mel'] = bl.get('random2mel', {}).get('val_mse', None)
        raw_losses['random_pr'] = bl.get('random2pr', {}).get('val_bce', None)
    else:
        raw_losses['random_mel'] = None
        raw_losses['random_pr'] = None

    def _safe_ratio(shuffle_l, cross_l, intra_l):
        denom = shuffle_l - intra_l
        if abs(denom) < 1e-6:
            return None
        return (shuffle_l - cross_l) / denom

    mel_ratio = _safe_ratio(raw_losses['shuffle_mel'], raw_losses['cross_mel'], raw_losses['intra_mel'])
    pr_ratio = _safe_ratio(raw_losses['shuffle_pr'], raw_losses['cross_pr'], raw_losses['intra_pr'])

    result['info_retention'] = {
        'mel_ratio': mel_ratio,
        'mel_ratio_note': '(shuffle_mel - cross_mel) / (shuffle_mel - intra_mel)',
        'pr_ratio': pr_ratio,
        'pr_ratio_note': '(shuffle_pr - cross_pr) / (shuffle_pr - intra_pr)',
        'raw_losses': raw_losses,
    }
    logger.info(f"  mel_ratio={mel_ratio}, pr_ratio={pr_ratio}")

    # --- Generate samples ---
    logger.info("\n--- Generating samples ---")
    for task_name, (decoder, task_type, vl_z, vl_tgt) in decoders.items():
        generate_samples(
            decoder, vl_z, vl_tgt, task_type, task_name,
            output_dir, device=device, n_samples=10,
        )

    # --- Save result ---
    save_test_result(result, output_dir, 'test11_decoder_suite', meta, seed=seed)
    logger.info(f"\nResults saved to {output_dir / 'test11_decoder_suite.json'}")

    # Cleanup
    for name, (dec, _, _, _) in decoders.items():
        del dec
    torch.cuda.empty_cache()

    return result


# ============================================================
# Random Baselines (model-independent)
# ============================================================

def run_random_baselines(
    results_dir: Path,
    device: str = 'cuda',
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train random2mel and random2pr baselines.
    z ~ N(0,1), independent of any model.
    """
    from experiments.bias_control.gate5b.harness import save_test_result

    output_dir = results_dir / 'baselines'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    json_path = output_dir / 'test11_decoder_suite.json'
    if json_path.exists():
        logger.info(f"Random baselines already exist: {json_path}")
        with open(json_path) as f:
            return json.load(f)

    logger.info("\n" + "=" * 60)
    logger.info("RANDOM BASELINES")
    logger.info("=" * 60)

    # Load targets
    mel_train, pr_train = load_targets(results_dir, 'train')
    mel_val, pr_val = load_targets(results_dir, 'val')
    n_train = mel_train.shape[0]
    n_val = mel_val.shape[0]

    # Generate random z
    rng = np.random.RandomState(seed)
    train_z = rng.randn(n_train, 256).astype(np.float32)
    val_z = rng.randn(n_val, 256).astype(np.float32)

    result = {}

    for task_name, task_type, targets_train, targets_val in [
        ('random2mel', 'mel', mel_train, mel_val),
        ('random2pr', 'pr', pr_train, pr_val),
    ]:
        logger.info(f"\n--- Training {task_name} ---")
        train_result = train_decoder(
            task_name, train_z, targets_train, val_z, targets_val,
            output_dir, device=device, seed=seed,
        )
        decoder = train_result['decoder']
        metrics = evaluate_decoder(decoder, val_z, targets_val, task_type, device=device)

        result[task_name] = {
            **{f'val_{k}': v for k, v in metrics.items()},
            'best_epoch': train_result['best_epoch'],
            'train_time_sec': train_result['train_time_sec'],
        }
        logger.info(f"  [{task_name}] Metrics: {metrics}")

        # Cleanup
        del decoder
        torch.cuda.empty_cache()

    # Save
    meta = {'descriptor': 'random_baseline'}
    save_test_result(result, output_dir, 'test11_decoder_suite', meta, seed=seed)
    logger.info(f"Baselines saved to {json_path}")

    return result


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Gate 5B - Test 11: Decoder Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--all', action='store_true', help='Run all arms (D0 -> a4r -> d4a4)')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--output', type=str, default=None, help='Results directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--skip-precompute', action='store_true',
                        help='Skip target precomputation if caches exist')
    parser.add_argument('--skip-train-embs', action='store_true',
                        help='Skip train embedding extraction if caches exist')
    parser.add_argument('--precompute-only', action='store_true',
                        help='Only precompute targets, then exit')
    parser.add_argument('--baselines-only', action='store_true',
                        help='Only run random baselines')

    args = parser.parse_args()

    results_dir = Path(args.output) if args.output else GATE5B_RESULTS_DIR

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Step 1: Precompute targets
    if not args.skip_precompute:
        train_indices, _ = precompute_targets(
            args.maestro_dir, results_dir,
            seed=args.seed,
        )
    else:
        idx_path = results_dir / 'train_indices.npy'
        if not idx_path.exists():
            logger.error(f"No train_indices.npy found at {idx_path}. "
                         f"Run without --skip-precompute first.")
            sys.exit(1)
        train_indices = np.load(idx_path)
        logger.info(f"Loaded existing train_indices: {len(train_indices)}")

    if args.precompute_only:
        logger.info("Precompute done. Exiting.")
        return

    # Step 2: Random baselines
    if args.baselines_only or args.all:
        run_random_baselines(results_dir, device=args.device, seed=args.seed)
        if args.baselines_only:
            return

    # Step 3: Run arms
    if args.all:
        arm_order = ['D0', 'a4r', 'd4a4']
        for arm_name in arm_order:
            ckpt = DEFAULT_CHECKPOINTS.get(arm_name)
            if not ckpt or not Path(ckpt).exists():
                logger.warning(f"Checkpoint not found for {arm_name}: {ckpt}, skipping")
                continue
            run_arm(
                ckpt, args.maestro_dir, results_dir, train_indices,
                device=args.device, seed=args.seed, num_workers=args.num_workers,
                skip_train_embs=args.skip_train_embs,
            )
    elif args.model:
        run_arm(
            args.model, args.maestro_dir, results_dir, train_indices,
            device=args.device, seed=args.seed, num_workers=args.num_workers,
            skip_train_embs=args.skip_train_embs,
        )
    else:
        parser.print_help()
        logger.error("\nSpecify --model or --all")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("TEST 11 COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
