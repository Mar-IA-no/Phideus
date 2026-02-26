#!/usr/bin/env python3
"""
Gate 5B - Test11 Perceptual Suite.

Perceptual-first pipeline:
- Decode MIDI event sequences from frozen embeddings (midi2events + audio2events)
- Generate .mid and rendered .wav samples for human evaluation

This does not replace Test11 decoder-suite metrics; it complements them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.bias_control.gate5b.checkpoint_loader import (
    get_eval_batch_size,
    load_model_from_checkpoint,
)
from experiments.bias_control.gate5b.event_decoder_model import (
    ConditionedEventTransformerDecoder,
    EventDecoderConfig,
    build_note_weight_vector,
)
from experiments.bias_control.gate5b.harness import (
    get_normal_embeddings,
    get_output_dir,
    save_test_result,
)
from experiments.bias_control.gate5b.midi_event_codec import MidiEventCodec, MidiEventCodecConfig
from experiments.bias_control.gate5b.render_midi_audio import render_midi_to_audio, tokens_to_midi
from src.bias_control.datasets.maestro_segments import MaestroSegmentDataset, collate_segments
from src.utils.midi_utils import parse_midi

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

GATE5B_RESULTS_DIR = Path('data/gate5b_results')
N_TRAIN_SUBSAMPLE = 20_000

CODEC_CFG = MidiEventCodecConfig(
    step_ms=20,
    segment_seconds=4.0,
    min_pitch=21,
    max_pitch=108,
    velocity_bins=32,
    max_seq_len=512,
)

MODEL_CFG = EventDecoderConfig(
    z_dim=256,
    d_model=512,
    n_heads=8,
    n_layers=8,
    d_ff=2048,
    dropout=0.1,
    n_cond_tokens=16,
    max_seq_len=CODEC_CFG.max_seq_len,
    vocab_size=512,  # set at runtime with codec.vocab_size
    pad_id=0,
)

TRAIN_CFG = {
    'epochs': 120,
    'batch_size': 48,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'patience': 15,
    'min_delta': 1e-5,
    'grad_clip': 1.0,
    'label_smoothing': 0.1,
    'note_weight': 2.0,
    'top_k': 48,
    'top_p': 0.95,
    'temperature': 0.9,
    'eval_gen_count': 512,
}

DEFAULT_CHECKPOINTS = {
    'D0': 'models/gate5b/D0/best_model.pt',
    'a4r': 'models/gate5b/a4r/best_model.pt',
    'd4a4': 'models/gate5b/d4a4/best_model.pt',
    'd4-a4r': 'models/gate5b/d4-a4r/best_model.pt',
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_derangement(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    while True:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p


def _segment_relative_notes(
    piece_notes: Sequence,
    start_time: float,
    segment_seconds: float,
    min_pitch: int,
    max_pitch: int,
) -> List[Tuple[int, float, float, int]]:
    out: List[Tuple[int, float, float, int]] = []
    end_time = start_time + segment_seconds
    for note in piece_notes:
        if note.offset <= start_time or note.onset >= end_time:
            continue
        if note.pitch < min_pitch or note.pitch > max_pitch:
            continue

        onset_rel = max(0.0, note.onset - start_time)
        offset_rel = min(segment_seconds, note.offset - start_time)
        if offset_rel <= onset_rel:
            continue

        out.append((int(note.pitch), float(onset_rel), float(offset_rel), int(note.velocity)))

    return out


def precompute_event_targets(
    maestro_dir: Path,
    output_dir: Path,
    codec: MidiEventCodec,
    seed: int,
) -> Tuple[np.ndarray, int]:
    """Precompute event token targets for train subsample + full validation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        output_dir / 'targets_event_train.npz',
        output_dir / 'targets_event_val.npz',
        output_dir / 'train_indices.npy',
    ]
    if all(p.exists() for p in required):
        train_indices = np.load(output_dir / 'train_indices.npy')
        n_val = int(np.load(output_dir / 'targets_event_val.npz')['data'].shape[0])
        logger.info('Event target caches found; skipping precompute.')
        return train_indices, n_val

    logger.info('Precomputing event targets...')

    train_ds = MaestroSegmentDataset(maestro_dir=maestro_dir, segment_len=4.0, hop=1.0, split='train')
    val_ds = MaestroSegmentDataset(maestro_dir=maestro_dir, segment_len=4.0, hop=1.0, split='validation')

    rng = np.random.RandomState(seed)
    if len(train_ds) > N_TRAIN_SUBSAMPLE:
        train_indices = np.sort(rng.choice(len(train_ds), size=N_TRAIN_SUBSAMPLE, replace=False))
    else:
        train_indices = np.arange(len(train_ds), dtype=np.int64)

    np.save(output_dir / 'train_indices.npy', train_indices)

    def process_split(ds: MaestroSegmentDataset, indices: np.ndarray, split: str) -> np.ndarray:
        tokens = np.full((len(indices), codec.cfg.max_seq_len), codec.pad_id, dtype=np.int32)

        piece_cache: Dict[int, Sequence] = {}
        t0 = time.time()

        for i, ds_idx in enumerate(indices):
            seg = ds.segments[int(ds_idx)]

            if seg.piece_idx not in piece_cache:
                piece_cache[seg.piece_idx] = parse_midi(seg.midi_path)

            rel_notes = _segment_relative_notes(
                piece_notes=piece_cache[seg.piece_idx],
                start_time=float(seg.start_time),
                segment_seconds=codec.cfg.segment_seconds,
                min_pitch=codec.cfg.min_pitch,
                max_pitch=codec.cfg.max_pitch,
            )
            tokens[i] = codec.encode_notes(rel_notes)

            if (i + 1) % 2000 == 0 or i == len(indices) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(indices) - i - 1) / max(rate, 1e-6)
                logger.info(f'  {split}: {i+1}/{len(indices)} ({rate:.1f} seg/s, ETA {eta:.0f}s)')

        np.savez_compressed(output_dir / f'targets_event_{split}.npz', data=tokens.astype(np.int16))
        return tokens

    process_split(train_ds, train_indices, 'train')
    val_indices = np.arange(len(val_ds), dtype=np.int64)
    val_tokens = process_split(val_ds, val_indices, 'val')

    logger.info(
        'Event target precompute done: train=%d, val=%d',
        len(train_indices),
        len(val_tokens),
    )
    return train_indices, len(val_tokens)


def load_event_targets(output_dir: Path, split: str) -> np.ndarray:
    data = np.load(output_dir / f'targets_event_{split}.npz')['data']
    return data.astype(np.int64)


def extract_train_embeddings(
    model_path: str,
    maestro_dir: Path,
    train_indices: np.ndarray,
    output_dir: Path,
    device: str,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract train embeddings in deterministic index order."""
    cache_path = output_dir / 'embeddings_train.npz'
    if cache_path.exists():
        cached = np.load(cache_path)
        if np.array_equal(cached['indices'], train_indices):
            logger.info('Train embeddings cache hit: %s', cache_path)
            return cached['audio_embs'], cached['midi_embs']

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model, meta = load_model_from_checkpoint(model_path, device=dev)
    descriptor = meta['descriptor']
    batch_size = get_eval_batch_size(descriptor)

    train_ds = MaestroSegmentDataset(maestro_dir=maestro_dir, segment_len=4.0, hop=1.0, split='train')
    subset = Subset(train_ds, sorted(train_indices.tolist()))

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_segments,
        pin_memory=True,
        prefetch_factor=2,
    )

    audio_out: List[np.ndarray] = []
    midi_out: List[np.ndarray] = []
    t0 = time.time()
    model.eval()

    with torch.no_grad():
        for batch in loader:
            audio = batch['audio'].to(dev, non_blocking=True)
            midi_pitch = batch['midi_pitch'].to(dev, non_blocking=True)
            midi_velocity = batch['midi_velocity'].to(dev, non_blocking=True)
            midi_duration = batch['midi_duration'].to(dev, non_blocking=True)
            midi_mask = batch['midi_mask'].to(dev, non_blocking=True)

            z_audio, z_midi = model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
            audio_out.append(z_audio.cpu().numpy())
            midi_out.append(z_midi.cpu().numpy())

    audio_embs = np.concatenate(audio_out, axis=0)
    midi_embs = np.concatenate(midi_out, axis=0)

    np.savez_compressed(
        cache_path,
        audio_embs=audio_embs.astype(np.float32),
        midi_embs=midi_embs.astype(np.float32),
        indices=train_indices.astype(np.int64),
    )
    logger.info('Train extraction done in %.1fs: audio=%s midi=%s', time.time() - t0, audio_embs.shape, midi_embs.shape)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return audio_embs, midi_embs


def load_train_embeddings_cache(output_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = output_dir / 'embeddings_train.npz'
    if not cache_path.exists():
        raise FileNotFoundError(f'Missing train embedding cache: {cache_path}')
    data = np.load(cache_path)
    return data['audio_embs'], data['midi_embs'], data['indices']


def _make_event_dataloaders(
    z_train: np.ndarray,
    tok_train: np.ndarray,
    z_val: np.ndarray,
    tok_val: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    zt = torch.from_numpy(z_train).float()
    tt = torch.from_numpy(tok_train).long()
    zv = torch.from_numpy(z_val).float()
    tv = torch.from_numpy(tok_val).long()

    train_loader = DataLoader(
        TensorDataset(zt, tt),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(zv, tv),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader


def _seq_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    pad_id: int,
    label_smoothing: float,
) -> torch.Tensor:
    vocab = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        weight=class_weights,
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )


def _token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> float:
    pred = logits.argmax(dim=-1)
    mask = targets.ne(pad_id)
    if not bool(mask.any()):
        return 0.0
    return float(pred[mask].eq(targets[mask]).float().mean().item())


def _build_decoder(codec: MidiEventCodec, device: torch.device) -> ConditionedEventTransformerDecoder:
    cfg = EventDecoderConfig(**{**asdict(MODEL_CFG), 'vocab_size': codec.vocab_size, 'pad_id': codec.pad_id})
    return ConditionedEventTransformerDecoder(config=cfg).to(device)


def train_event_decoder(
    task_name: str,
    train_z: np.ndarray,
    train_tokens: np.ndarray,
    val_z: np.ndarray,
    val_tokens: np.ndarray,
    codec: MidiEventCodec,
    model_out_dir: Path,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """Train one autoregressive event decoder."""
    set_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    decoder = _build_decoder(codec, dev)
    logger.info('[%s] Decoder params: %.1fM', task_name, decoder.n_params / 1e6)

    train_loader, val_loader = _make_event_dataloaders(
        z_train=train_z,
        tok_train=train_tokens,
        z_val=val_z,
        tok_val=val_tokens,
        batch_size=TRAIN_CFG['batch_size'],
    )

    class_weights = build_note_weight_vector(
        vocab_size=codec.vocab_size,
        note_on_start=codec.note_on_start,
        note_off_start=codec.note_off_start,
        pitch_count=codec.pitch_count,
        note_weight=TRAIN_CFG['note_weight'],
    ).to(dev)

    opt = torch.optim.AdamW(decoder.parameters(), lr=TRAIN_CFG['lr'], weight_decay=TRAIN_CFG['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_CFG['epochs'])

    best_val = float('inf')
    best_epoch = 0
    no_improve = 0
    ckpt_path = model_out_dir / f'{task_name}_best.pt'
    model_out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    for epoch in range(1, TRAIN_CFG['epochs'] + 1):
        decoder.train()
        train_losses: List[float] = []

        for z_batch, tok_batch in train_loader:
            z_batch = z_batch.to(dev, non_blocking=True)
            tok_batch = tok_batch.to(dev, non_blocking=True)
            inp = tok_batch[:, :-1]
            tgt = tok_batch[:, 1:]

            logits = decoder(z_batch, inp)
            loss = _seq_ce_loss(
                logits=logits,
                targets=tgt,
                class_weights=class_weights,
                pad_id=codec.pad_id,
                label_smoothing=TRAIN_CFG['label_smoothing'],
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), TRAIN_CFG['grad_clip'])
            opt.step()
            train_losses.append(float(loss.item()))

        decoder.eval()
        val_losses: List[float] = []
        val_accs: List[float] = []
        with torch.no_grad():
            for z_batch, tok_batch in val_loader:
                z_batch = z_batch.to(dev, non_blocking=True)
                tok_batch = tok_batch.to(dev, non_blocking=True)
                inp = tok_batch[:, :-1]
                tgt = tok_batch[:, 1:]

                logits = decoder(z_batch, inp)
                vloss = _seq_ce_loss(
                    logits=logits,
                    targets=tgt,
                    class_weights=class_weights,
                    pad_id=codec.pad_id,
                    label_smoothing=TRAIN_CFG['label_smoothing'],
                )
                val_losses.append(float(vloss.item()))
                val_accs.append(_token_accuracy(logits, tgt, codec.pad_id))

        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
        val_acc = float(np.mean(val_accs)) if val_accs else float('nan')

        if epoch == 1 or epoch % 5 == 0:
            logger.info(
                '  [%s] e%d train=%.4f val=%.4f tok_acc=%.4f lr=%.2e',
                task_name,
                epoch,
                train_loss,
                val_loss,
                val_acc,
                sched.get_last_lr()[0],
            )

        if val_loss < best_val - TRAIN_CFG['min_delta']:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    'task_name': task_name,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'codec_cfg': asdict(codec.cfg),
                    'model_cfg': asdict(decoder.cfg),
                    'state_dict': decoder.state_dict(),
                },
                ckpt_path,
            )
        else:
            no_improve += 1

        sched.step()

        if no_improve >= TRAIN_CFG['patience']:
            logger.info('[%s] Early stopping at epoch %d (best=%d, val=%.4f)', task_name, epoch, best_epoch, best_val)
            break

    train_time = time.time() - t0

    ckpt = torch.load(ckpt_path, map_location=dev)
    decoder.load_state_dict(ckpt['state_dict'])

    return {
        'task_name': task_name,
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val),
        'train_time_sec': float(train_time),
        'checkpoint': str(ckpt_path),
        'decoder': decoder,
    }


def _notes_to_roll(
    notes: Sequence[Tuple[int, float, float, int]],
    n_frames: int = 188,
    sr: int = 24000,
    hop: int = 512,
) -> np.ndarray:
    pr = np.zeros((n_frames, 88), dtype=np.float32)
    for pitch, onset, offset, velocity in notes:
        if pitch < 21 or pitch > 108:
            continue
        start = int(np.floor(onset * sr / hop))
        end = int(np.ceil(offset * sr / hop))
        start = max(0, min(start, n_frames - 1))
        end = max(start + 1, min(end, n_frames))
        pr[start:end, pitch - 21] = max(pr[start:end, pitch - 21].max(), velocity / 127.0)
    return pr


def _frame_f1(pred_roll: np.ndarray, true_roll: np.ndarray, thr: float = 0.1) -> Tuple[float, float, float]:
    pred = pred_roll > thr
    truth = true_roll > thr

    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, np.logical_not(truth)).sum())
    fn = float(np.logical_and(np.logical_not(pred), truth).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return f1, precision, recall


def evaluate_event_decoder(
    decoder: ConditionedEventTransformerDecoder,
    z: np.ndarray,
    targets: np.ndarray,
    codec: MidiEventCodec,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """Teacher-forcing CE/acc and generation F1 on a validation subset."""
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    class_weights = build_note_weight_vector(
        vocab_size=codec.vocab_size,
        note_on_start=codec.note_on_start,
        note_off_start=codec.note_off_start,
        pitch_count=codec.pitch_count,
        note_weight=TRAIN_CFG['note_weight'],
    ).to(dev)

    z_t = torch.from_numpy(z).float()
    tok_t = torch.from_numpy(targets).long()

    loader = DataLoader(TensorDataset(z_t, tok_t), batch_size=TRAIN_CFG['batch_size'], shuffle=False)

    decoder.eval()
    ce_vals: List[float] = []
    acc_vals: List[float] = []
    with torch.no_grad():
        for zb, tb in loader:
            zb = zb.to(dev)
            tb = tb.to(dev)
            inp = tb[:, :-1]
            tgt = tb[:, 1:]
            logits = decoder(zb, inp)
            loss = _seq_ce_loss(
                logits=logits,
                targets=tgt,
                class_weights=class_weights,
                pad_id=codec.pad_id,
                label_smoothing=TRAIN_CFG['label_smoothing'],
            )
            ce_vals.append(float(loss.item()))
            acc_vals.append(_token_accuracy(logits, tgt, codec.pad_id))

    # generation-based frame-F1 on subset
    rng = np.random.RandomState(seed)
    n_eval = min(TRAIN_CFG['eval_gen_count'], len(z))
    eval_indices = rng.choice(len(z), size=n_eval, replace=False)

    f1_vals: List[float] = []
    prec_vals: List[float] = []
    rec_vals: List[float] = []

    with torch.no_grad():
        for idx in eval_indices:
            z_i = torch.from_numpy(z[idx: idx + 1]).float().to(dev)
            gen = decoder.generate(
                z=z_i,
                bos_id=codec.bos_id,
                eos_id=codec.eos_id,
                max_len=codec.cfg.max_seq_len,
                temperature=TRAIN_CFG['temperature'],
                top_k=TRAIN_CFG['top_k'],
                top_p=TRAIN_CFG['top_p'],
                deterministic=False,
            )[0].cpu().numpy()

            pred_notes = codec.decode_tokens(gen)
            true_notes = codec.decode_tokens(targets[idx])

            pred_roll = _notes_to_roll(pred_notes)
            true_roll = _notes_to_roll(true_notes)
            f1, p, r = _frame_f1(pred_roll, true_roll, thr=0.1)
            f1_vals.append(f1)
            prec_vals.append(p)
            rec_vals.append(r)

    return {
        'val_ce': float(np.mean(ce_vals)) if ce_vals else float('nan'),
        'val_token_acc': float(np.mean(acc_vals)) if acc_vals else float('nan'),
        'val_frame_f1': float(np.mean(f1_vals)) if f1_vals else float('nan'),
        'val_frame_precision': float(np.mean(prec_vals)) if prec_vals else float('nan'),
        'val_frame_recall': float(np.mean(rec_vals)) if rec_vals else float('nan'),
        'n_eval_generation': int(n_eval),
    }


def evaluate_controls(
    decoder: ConditionedEventTransformerDecoder,
    z_aligned: np.ndarray,
    z_train: np.ndarray,
    targets: np.ndarray,
    codec: MidiEventCodec,
    device: str,
    seed: int,
) -> Dict[str, float]:
    """Evaluate CE under aligned/shuffle/mean/zero z inputs."""
    controls: Dict[str, float] = {}

    der = generate_derangement(len(z_aligned), seed=seed)
    z_shuffle = z_aligned[der]
    z_mean = np.tile(z_train.mean(axis=0, keepdims=True), (len(z_aligned), 1))
    z_zero = np.zeros_like(z_aligned)

    for name, z_in in [
        ('aligned_ce', z_aligned),
        ('shuffle_ce', z_shuffle),
        ('mean_z_ce', z_mean),
        ('zero_z_ce', z_zero),
    ]:
        m = evaluate_event_decoder(
            decoder=decoder,
            z=z_in,
            targets=targets,
            codec=codec,
            device=device,
            seed=seed,
        )
        controls[name] = float(m['val_ce'])

    return controls


def _select_sample_indices(n_total: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n = min(n_total, n_samples)
    return np.sort(rng.choice(n_total, size=n, replace=False))


def generate_samples(
    decoder: ConditionedEventTransformerDecoder,
    task_name: str,
    z_val: np.ndarray,
    targets: np.ndarray,
    codec: MidiEventCodec,
    samples_dir: Path,
    renderer: str,
    soundfont: Optional[str],
    n_samples: int,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """Generate paired predicted+truth MIDI/WAV samples."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    idxs = _select_sample_indices(len(z_val), n_samples=n_samples, seed=seed)

    renderer_stats = {'fluidsynth': 0, 'prettymidi': 0, 'failed': 0}

    decoder.eval()
    with torch.no_grad():
        for out_i, idx in enumerate(idxs):
            z_i = torch.from_numpy(z_val[idx: idx + 1]).float().to(dev)
            gen_tokens = decoder.generate(
                z=z_i,
                bos_id=codec.bos_id,
                eos_id=codec.eos_id,
                max_len=codec.cfg.max_seq_len,
                temperature=TRAIN_CFG['temperature'],
                top_k=TRAIN_CFG['top_k'],
                top_p=TRAIN_CFG['top_p'],
                deterministic=False,
            )[0].cpu().numpy()

            pred_mid = samples_dir / f'{task_name}_pred_{out_i:02d}.mid'
            truth_mid = samples_dir / f'{task_name}_truth_{out_i:02d}.mid'
            pred_wav = samples_dir / f'{task_name}_pred_{out_i:02d}.wav'
            truth_wav = samples_dir / f'{task_name}_truth_{out_i:02d}.wav'

            tokens_to_midi(codec, gen_tokens, pred_mid)
            tokens_to_midi(codec, targets[idx], truth_mid)

            used_pred = render_midi_to_audio(
                midi_path=pred_mid,
                wav_path=pred_wav,
                renderer=renderer,
                soundfont=soundfont,
                sample_rate=24000,
            )
            used_truth = render_midi_to_audio(
                midi_path=truth_mid,
                wav_path=truth_wav,
                renderer=renderer,
                soundfont=soundfont,
                sample_rate=24000,
            )

            renderer_stats[used_pred] = renderer_stats.get(used_pred, 0) + 1
            renderer_stats[used_truth] = renderer_stats.get(used_truth, 0) + 1

    return {
        'n_generated': int(len(idxs)),
        'renderer_usage': renderer_stats,
        'samples_dir': str(samples_dir),
    }


def run_arm(
    model_path: str,
    maestro_dir: Path,
    output_root: Path,
    codec: MidiEventCodec,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run perceptual pipeline for one arm."""
    set_seed(args.seed)

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Read metadata early to resolve arm output dir.
    model, meta = load_model_from_checkpoint(model_path, device=dev)
    out_dir = get_output_dir(args=SimpleNamespace(output=None), meta=meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / 'test11_perceptual_models'
    samples_dir = out_dir / 'test11_perceptual_samples'

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info('============================================================')
    logger.info('PERCEPTUAL ARM: %s (%s)', meta['descriptor'], model_path)
    logger.info('============================================================')

    train_indices, _ = precompute_event_targets(
        maestro_dir=maestro_dir,
        output_dir=output_root,
        codec=codec,
        seed=args.seed,
    )

    tok_train = load_event_targets(output_root, 'train')
    tok_val = load_event_targets(output_root, 'val')

    if len(tok_train) != len(train_indices):
        raise RuntimeError('train token cache length mismatch with train_indices')

    # Train embeddings
    if args.skip_train_embs:
        audio_train_z, midi_train_z, cached_indices = load_train_embeddings_cache(out_dir)
        if not np.array_equal(cached_indices, train_indices):
            raise RuntimeError(
                'embeddings_train.npz indices do not match train_indices.npy; '
                'rerun without --skip-train-embs to regenerate cache.'
            )
        logger.info('Using cached train embeddings (--skip-train-embs): %s', out_dir / 'embeddings_train.npz')
    else:
        audio_train_z, midi_train_z = extract_train_embeddings(
            model_path=model_path,
            maestro_dir=maestro_dir,
            train_indices=train_indices,
            output_dir=out_dir,
            device=args.device,
            num_workers=args.num_workers,
        )

    # Validation embeddings (cache-aware via harness)
    model, meta = load_model_from_checkpoint(model_path, device=dev)
    val_ds = MaestroSegmentDataset(maestro_dir=maestro_dir, segment_len=4.0, hop=1.0, split='validation')
    audio_val_t, midi_val_t = get_normal_embeddings(
        model=model,
        dataset=val_ds,
        device=args.device,
        descriptor=meta['descriptor'],
        output_dir=out_dir,
        num_workers=args.num_workers,
        force_extract=False,
    )
    audio_val_z = audio_val_t.cpu().numpy().astype(np.float32)
    midi_val_z = midi_val_t.cpu().numpy().astype(np.float32)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(audio_val_z) != len(tok_val):
        raise RuntimeError(f'val alignment mismatch: z={len(audio_val_z)} tokens={len(tok_val)}')

    results: Dict[str, Any] = {
        'codec': {
            'step_ms': codec.cfg.step_ms,
            'segment_seconds': codec.cfg.segment_seconds,
            'max_seq_len': codec.cfg.max_seq_len,
            'vocab_size': codec.vocab_size,
            'velocity_bins': codec.cfg.velocity_bins,
        },
        'training_cfg': TRAIN_CFG,
        'tasks': {},
        'controls': {},
        'human_eval': {
            'required_per_arm': {'intra': 15, 'cross': 15},
            'acceptance_gate': {
                'recognizable_rate': 0.80,
                'related_to_truth_rate': 0.60,
            },
        },
    }

    tasks = [
        ('midi2events', midi_train_z, midi_val_z),
        ('audio2events', audio_train_z, audio_val_z),
    ]

    trained_decoders: Dict[str, ConditionedEventTransformerDecoder] = {}

    for task_name, z_train, z_val in tasks:
        ckpt_path = models_dir / f'{task_name}_best.pt'
        train_stats: Dict[str, Any]

        if args.skip_train and ckpt_path.exists():
            logger.info('[%s] --skip-train active: loading checkpoint %s', task_name, ckpt_path)
            dec = _build_decoder(codec, dev)
            ckpt = torch.load(ckpt_path, map_location=dev)
            dec.load_state_dict(ckpt['state_dict'])
            train_stats = {
                'task_name': task_name,
                'best_epoch': int(ckpt.get('epoch', 0)),
                'best_val_loss': float(ckpt.get('val_loss', float('nan'))),
                'train_time_sec': float('nan'),
                'checkpoint': str(ckpt_path),
                'decoder': dec,
            }
        else:
            train_stats = train_event_decoder(
                task_name=task_name,
                train_z=z_train,
                train_tokens=tok_train,
                val_z=z_val,
                val_tokens=tok_val,
                codec=codec,
                model_out_dir=models_dir,
                device=args.device,
                seed=args.seed,
            )

        dec = train_stats.pop('decoder')
        trained_decoders[task_name] = dec

        eval_metrics = evaluate_event_decoder(
            decoder=dec,
            z=z_val,
            targets=tok_val,
            codec=codec,
            device=args.device,
            seed=args.seed,
        )

        ctrl = evaluate_controls(
            decoder=dec,
            z_aligned=z_val,
            z_train=z_train,
            targets=tok_val,
            codec=codec,
            device=args.device,
            seed=args.seed,
        )

        sample_stats = generate_samples(
            decoder=dec,
            task_name=task_name,
            z_val=z_val,
            targets=tok_val,
            codec=codec,
            samples_dir=samples_dir,
            renderer=args.renderer,
            soundfont=args.soundfont,
            n_samples=args.n_samples,
            device=args.device,
            seed=args.seed,
        )

        results['tasks'][task_name] = {
            **train_stats,
            **eval_metrics,
            'samples': sample_stats,
        }
        results['controls'][task_name] = ctrl

    intra_ce = results['controls']['midi2events']['aligned_ce']
    cross_ce = results['controls']['audio2events']['aligned_ce']
    shuffle_ce = results['controls']['audio2events']['shuffle_ce']

    denom = shuffle_ce - intra_ce
    if abs(denom) < 1e-6:
        info_ratio = float('nan')
    else:
        info_ratio = (shuffle_ce - cross_ce) / denom

    results['info_retention_events'] = {
        'ratio': float(info_ratio),
        'formula': '(shuffle_ce - cross_ce) / (shuffle_ce - intra_ce)',
        'raw': {
            'intra_ce': float(intra_ce),
            'cross_ce': float(cross_ce),
            'shuffle_ce': float(shuffle_ce),
        },
    }

    save_test_result(
        result=results,
        output_dir=out_dir,
        test_name='test11_perceptual',
        meta=meta,
        seed=args.seed,
    )

    logger.info('Perceptual arm done: %s', out_dir)
    return results


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    order = ['D0', 'a4r', 'd4a4']
    if args.include_d4a4r:
        order.append('d4-a4r')

    all_results: Dict[str, Any] = {}
    codec = MidiEventCodec(CODEC_CFG)
    output_root = GATE5B_RESULTS_DIR

    for arm in order:
        model_path = DEFAULT_CHECKPOINTS[arm]
        all_results[arm] = run_arm(
            model_path=model_path,
            maestro_dir=Path(args.maestro_dir),
            output_root=output_root,
            codec=codec,
            args=args,
        )

    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Gate5B Test11 perceptual suite')
    parser.add_argument('--model', type=str, default=None, help='Single checkpoint path')
    parser.add_argument('--all', action='store_true', help='Run D0 -> a4r -> d4a4 (optional d4-a4r)')
    parser.add_argument('--include-d4a4r', action='store_true', help='Include d4-a4r in --all')

    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--renderer', type=str, default='auto', choices=['auto', 'fluidsynth', 'prettymidi'])
    parser.add_argument('--soundfont', type=str, default=None)

    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-precompute', action='store_true')
    parser.add_argument('--skip-train-embs', action='store_true')
    parser.add_argument('--n-samples', type=int, default=10)
    parser.add_argument('--n-samples-human', type=int, default=30)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model and not args.all:
        raise SystemExit('Use --model <path> or --all')

    set_seed(args.seed)

    # Optional precompute stage (shared caches)
    codec = MidiEventCodec(CODEC_CFG)
    if not args.skip_precompute:
        precompute_event_targets(
            maestro_dir=Path(args.maestro_dir),
            output_dir=GATE5B_RESULTS_DIR,
            codec=codec,
            seed=args.seed,
        )

    if args.all:
        all_results = run_all(args)
        summary_path = GATE5B_RESULTS_DIR / 'test11_perceptual_summary.json'
        with summary_path.open('w') as f:
            json.dump(all_results, f, indent=2)
        logger.info('Saved summary: %s', summary_path)
        return

    # single arm
    run_arm(
        model_path=args.model,
        maestro_dir=Path(args.maestro_dir),
        output_root=GATE5B_RESULTS_DIR,
        codec=codec,
        args=args,
    )


if __name__ == '__main__':
    main()
