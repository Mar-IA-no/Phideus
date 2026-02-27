#!/usr/bin/env python3
"""
Gate 5B — Test11 Pre-Projection A/B Test.

Diagnoses whether the z→256 projection bottleneck limits decoder quality.

Conditions:
  A) z_dim=256  (post-projection, existing baseline from test11_perceptual)
  B) z_dim=1024 (audio pre-projection) / z_dim=512 (midi pre-projection)

Method: Forward hooks on audio_projection / midi_projection capture
pre-projection features for ANY model variant (D0, a4r, d4a4, etc.).

Usage:
    python experiments/bias_control/gate5b/test11_preproj_ab_test.py \\
        --arms D0 a4r --device cuda --skip-precompute
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

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
from experiments.bias_control.gate5b.harness import GATE5B_RESULTS_DIR
from experiments.bias_control.gate5b.midi_event_codec import MidiEventCodec
from experiments.bias_control.gate5b.test11_perceptual_suite import (
    CODEC_CFG,
    MODEL_CFG,
    TRAIN_CFG,
    N_TRAIN_SUBSAMPLE,
    DEFAULT_CHECKPOINTS,
    set_seed,
    generate_derangement,
    precompute_event_targets,
    load_event_targets,
    evaluate_event_decoder,
    evaluate_controls,
)
from src.bias_control.datasets.maestro_segments import (
    MaestroSegmentDataset,
    collate_segments,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ─── Pre-Projection Extraction ───────────────────────────────────────────────


def _get_base_model(model: nn.Module) -> nn.Module:
    """Traverse wrappers to find the CrossModalModel base."""
    base = model
    while hasattr(base, 'base_model'):
        base = base.base_model
    return base


@torch.no_grad()
def extract_preproj_embeddings(
    model: nn.Module,
    dataset,
    device: str,
    descriptor: str,
    num_workers: int = 8,
    indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pre-projection embeddings using forward hooks.

    Hooks on base_model.audio_projection and base_model.midi_projection
    capture the INPUT tensors (before projection to 256d).

    Returns:
        (audio_preproj, midi_preproj) as float32 numpy arrays.
        audio_preproj: [N, 1024], midi_preproj: [N, 512]
    """
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    batch_size = get_eval_batch_size(descriptor)

    base = _get_base_model(model)

    audio_pre: List[torch.Tensor] = []
    midi_pre: List[torch.Tensor] = []

    def hook_audio(module, inp, out):
        audio_pre.append(inp[0].detach().cpu())

    def hook_midi(module, inp, out):
        midi_pre.append(inp[0].detach().cpu())

    h_a = base.audio_projection.register_forward_hook(hook_audio)
    h_m = base.midi_projection.register_forward_hook(hook_midi)

    try:
        ds = Subset(dataset, sorted(indices.tolist())) if indices is not None else dataset

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_segments,
            pin_memory=True,
            prefetch_factor=2,
        )

        model.eval()
        n_segs = len(indices) if indices is not None else len(dataset)

        pbar = tqdm(
            loader,
            desc='  Extracting pre-proj',
            unit='batch',
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] segs={postfix}',
        )
        pbar.set_postfix_str(f'0/{n_segs}')

        for batch in pbar:
            audio = batch['audio'].to(dev, non_blocking=True)
            midi_pitch = batch['midi_pitch'].to(dev, non_blocking=True)
            midi_velocity = batch['midi_velocity'].to(dev, non_blocking=True)
            midi_duration = batch['midi_duration'].to(dev, non_blocking=True)
            midi_mask = batch['midi_mask'].to(dev, non_blocking=True)

            # Normal forward — hooks capture pre-projection features
            model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)
            done = sum(t.shape[0] for t in audio_pre)
            pbar.set_postfix_str(f'{done}/{n_segs}')

        audio_embs = torch.cat(audio_pre, dim=0).numpy()
        midi_embs = torch.cat(midi_pre, dim=0).numpy()

        logger.info(
            'Pre-proj extraction done: audio=%s midi=%s',
            audio_embs.shape, midi_embs.shape,
        )
    finally:
        h_a.remove()
        h_m.remove()

    return audio_embs.astype(np.float32), midi_embs.astype(np.float32)


def extract_and_cache_preproj(
    model_path: str,
    maestro_dir: Path,
    output_dir: Path,
    device: str,
    num_workers: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and cache pre-projection embeddings for train + val.

    Returns:
        (audio_train, midi_train, audio_val, midi_val)
    """
    cache_train = output_dir / 'embeddings_preproj_train.npz'
    cache_val = output_dir / 'embeddings_preproj_normal.npz'

    if cache_train.exists() and cache_val.exists():
        logger.info('Pre-proj caches found, loading...')
        dt = np.load(cache_train)
        dv = np.load(cache_val)
        logger.info(
            '  train: audio=%s midi=%s | val: audio=%s midi=%s',
            dt['audio_embs'].shape, dt['midi_embs'].shape,
            dv['audio_embs'].shape, dv['midi_embs'].shape,
        )
        return dt['audio_embs'], dt['midi_embs'], dv['audio_embs'], dv['midi_embs']

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model, meta = load_model_from_checkpoint(model_path, device=dev)
    descriptor = meta['descriptor']

    # --- Resolve train indices (same as original perceptual suite) ---
    indices_path = GATE5B_RESULTS_DIR / 'train_indices.npy'
    existing_emb_cache = output_dir / 'embeddings_train.npz'

    if indices_path.exists():
        train_indices = np.load(indices_path)
        logger.info('Train indices from %s (%d segments)', indices_path, len(train_indices))
    elif existing_emb_cache.exists():
        train_indices = np.load(existing_emb_cache)['indices']
        logger.info('Train indices from embeddings cache (%d segments)', len(train_indices))
    else:
        rng = np.random.RandomState(seed)
        train_ds_tmp = MaestroSegmentDataset(
            maestro_dir=maestro_dir, segment_len=4.0, hop=1.0, split='train',
        )
        train_indices = np.sort(rng.choice(len(train_ds_tmp), size=N_TRAIN_SUBSAMPLE, replace=False))
        del train_ds_tmp
        logger.info('Generated new train indices (%d segments)', len(train_indices))

    # --- Extract train pre-proj ---
    train_ds = MaestroSegmentDataset(maestro_dir=maestro_dir, segment_len=4.0, hop=1.0, split='train')

    logger.info('Extracting pre-proj TRAIN embeddings (%d segments)...', len(train_indices))
    audio_train, midi_train = extract_preproj_embeddings(
        model, train_ds, device, descriptor, num_workers, indices=train_indices,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_train,
        audio_embs=audio_train,
        midi_embs=midi_train,
        indices=train_indices,
    )
    logger.info('Saved: %s (audio=%s, midi=%s)', cache_train, audio_train.shape, midi_train.shape)

    # --- Extract val pre-proj ---
    val_ds = MaestroSegmentDataset(maestro_dir=maestro_dir, segment_len=4.0, hop=1.0, split='validation')

    logger.info('Extracting pre-proj VAL embeddings (%d segments)...', len(val_ds))
    audio_val, midi_val = extract_preproj_embeddings(
        model, val_ds, device, descriptor, num_workers,
    )

    np.savez_compressed(
        cache_val,
        audio_embs=audio_val,
        midi_embs=midi_val,
    )
    logger.info('Saved: %s (audio=%s, midi=%s)', cache_val, audio_val.shape, midi_val.shape)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return audio_train, midi_train, audio_val, midi_val


# ─── Decoder Training (custom z_dim) ─────────────────────────────────────────


def _build_decoder(
    codec: MidiEventCodec,
    device: torch.device,
    z_dim: int,
) -> ConditionedEventTransformerDecoder:
    """Build event decoder with custom z_dim."""
    cfg = EventDecoderConfig(
        z_dim=z_dim,
        d_model=MODEL_CFG.d_model,
        n_heads=MODEL_CFG.n_heads,
        n_layers=MODEL_CFG.n_layers,
        d_ff=MODEL_CFG.d_ff,
        dropout=MODEL_CFG.dropout,
        n_cond_tokens=MODEL_CFG.n_cond_tokens,
        max_seq_len=MODEL_CFG.max_seq_len,
        vocab_size=codec.vocab_size,
        pad_id=codec.pad_id,
    )
    return ConditionedEventTransformerDecoder(config=cfg).to(device)


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


def train_preproj_decoder(
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
    """Train event decoder with z_dim inferred from training data shape."""
    set_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    z_dim = train_z.shape[1]
    decoder = _build_decoder(codec, dev, z_dim=z_dim)
    logger.info('[%s] Pre-proj decoder: z_dim=%d, params=%.1fM', task_name, z_dim, decoder.n_params / 1e6)

    # Dataloaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_z).float(), torch.from_numpy(train_tokens).long()),
        batch_size=TRAIN_CFG['batch_size'], shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_z).float(), torch.from_numpy(val_tokens).long()),
        batch_size=TRAIN_CFG['batch_size'], shuffle=False, num_workers=0, pin_memory=True,
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
    max_epochs = TRAIN_CFG['epochs']

    epoch_pbar = tqdm(
        range(1, max_epochs + 1),
        desc=f'  [{task_name}] Training',
        unit='ep',
        dynamic_ncols=True,
    )

    for epoch in epoch_pbar:
        decoder.train()
        train_losses: List[float] = []

        for z_batch, tok_batch in train_loader:
            z_batch = z_batch.to(dev, non_blocking=True)
            tok_batch = tok_batch.to(dev, non_blocking=True)
            inp = tok_batch[:, :-1]
            tgt = tok_batch[:, 1:]

            logits = decoder(z_batch, inp)
            loss = _seq_ce_loss(logits, tgt, class_weights, codec.pad_id, TRAIN_CFG['label_smoothing'])

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
                vloss = _seq_ce_loss(logits, tgt, class_weights, codec.pad_id, TRAIN_CFG['label_smoothing'])
                val_losses.append(float(vloss.item()))
                val_accs.append(_token_accuracy(logits, tgt, codec.pad_id))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_acc = float(np.mean(val_accs))

        improved = ''
        if val_loss < best_val - TRAIN_CFG['min_delta']:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            improved = ' *BEST*'
            torch.save({
                'task_name': task_name,
                'epoch': epoch,
                'val_loss': val_loss,
                'z_dim': z_dim,
                'codec_cfg': asdict(codec.cfg),
                'model_cfg': asdict(decoder.cfg),
                'state_dict': decoder.state_dict(),
            }, ckpt_path)
        else:
            no_improve += 1

        sched.step()

        epoch_pbar.set_postfix_str(
            f'train={train_loss:.3f} val={val_loss:.3f} acc={val_acc:.3f} '
            f'best=e{best_epoch}({best_val:.3f}) wait={no_improve}/{TRAIN_CFG["patience"]}{improved}'
        )

        if no_improve >= TRAIN_CFG['patience']:
            logger.info(
                '[%s] Early stop e%d (best=e%d, val=%.4f)',
                task_name, epoch, best_epoch, best_val,
            )
            break

    epoch_pbar.close()
    train_time = time.time() - t0
    logger.info('[%s] Training done in %.0fs (best=e%d, val=%.4f)', task_name, train_time, best_epoch, best_val)

    # Reload best checkpoint
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    decoder.load_state_dict(ckpt['state_dict'])

    return {
        'task_name': task_name,
        'z_dim': int(z_dim),
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val),
        'train_time_sec': float(train_time),
        'n_params': int(decoder.n_params),
        'checkpoint': str(ckpt_path),
        'decoder': decoder,
    }


# ─── Main Pipeline ────────────────────────────────────────────────────────────


def run_arm(
    arm: str,
    model_path: str,
    maestro_dir: Path,
    codec: MidiEventCodec,
    device: str,
    num_workers: int,
    seed: int,
    skip_train: bool = False,
) -> Dict[str, Any]:
    """Run pre-projection A/B test for one arm."""
    set_seed(seed)

    arm_dir = 'D0' if arm == 'D0' else arm
    out_dir = GATE5B_RESULTS_DIR / arm_dir
    models_dir = out_dir / 'test11_preproj_models'

    logger.info('=' * 60)
    logger.info('PRE-PROJ A/B TEST: %s (%s)', arm, model_path)
    logger.info('=' * 60)
    logger.info('Pipeline: (1) Extract pre-proj → (2) Train midi2events(z=%dd) '
                '→ (3) Train audio2events(z=%dd) → (4) Evaluate + controls',
                512, 1024)

    # 1. Extract pre-projection embeddings
    audio_train, midi_train, audio_val, midi_val = extract_and_cache_preproj(
        model_path=model_path,
        maestro_dir=maestro_dir,
        output_dir=out_dir,
        device=device,
        num_workers=num_workers,
        seed=seed,
    )

    logger.info(
        'Pre-proj dims: audio_train=%s midi_train=%s audio_val=%s midi_val=%s',
        audio_train.shape, midi_train.shape, audio_val.shape, midi_val.shape,
    )

    # 2. Load event targets (shared, already precomputed)
    logger.info('[%s] Loading event targets...', arm)
    tok_train = load_event_targets(GATE5B_RESULTS_DIR, 'train')
    tok_val = load_event_targets(GATE5B_RESULTS_DIR, 'val')
    logger.info('[%s] Targets: train=%d val=%d', arm, len(tok_train), len(tok_val))

    assert len(tok_train) == len(audio_train), (
        f'Train mismatch: tok={len(tok_train)} emb={len(audio_train)}'
    )
    assert len(tok_val) == len(audio_val), (
        f'Val mismatch: tok={len(tok_val)} emb={len(audio_val)}'
    )

    results: Dict[str, Any] = {
        'experiment': 'test11_preproj_ab',
        'arm': arm,
        'model_path': model_path,
        'pre_proj_dims': {
            'audio': int(audio_train.shape[1]),
            'midi': int(midi_train.shape[1]),
        },
        'post_proj_dim': 256,
        'training_cfg': TRAIN_CFG,
        'tasks': {},
        'controls': {},
    }

    # 3. Train/load decoders and evaluate
    tasks = [
        ('preproj_midi2events', midi_train, midi_val),
        ('preproj_audio2events', audio_train, audio_val),
    ]

    for task_name, z_train, z_val in tasks:
        ckpt_path = models_dir / f'{task_name}_best.pt'

        if skip_train and ckpt_path.exists():
            logger.info('[%s] Loading existing checkpoint: %s', task_name, ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            z_dim = ckpt['model_cfg']['z_dim']
            dec = _build_decoder(codec, torch.device(device), z_dim)
            dec.load_state_dict(ckpt['state_dict'])
            train_stats = {
                'task_name': task_name,
                'z_dim': int(z_dim),
                'best_epoch': int(ckpt.get('epoch', 0)),
                'best_val_loss': float(ckpt.get('val_loss', float('nan'))),
                'train_time_sec': float('nan'),
                'n_params': int(dec.n_params),
                'checkpoint': str(ckpt_path),
            }
        else:
            res = train_preproj_decoder(
                task_name=task_name,
                train_z=z_train,
                train_tokens=tok_train,
                val_z=z_val,
                val_tokens=tok_val,
                codec=codec,
                model_out_dir=models_dir,
                device=device,
                seed=seed,
            )
            dec = res.pop('decoder')
            train_stats = res

        # Evaluate
        logger.info('[%s] Evaluating (CE + generation F1 on %d samples)...', task_name, TRAIN_CFG['eval_gen_count'])
        t_eval = time.time()
        eval_metrics = evaluate_event_decoder(
            decoder=dec,
            z=z_val,
            targets=tok_val,
            codec=codec,
            device=device,
            seed=seed,
        )
        logger.info('[%s] Eval done in %.0fs: CE=%.4f f1=%.4f', task_name, time.time() - t_eval,
                     eval_metrics['val_ce'], eval_metrics['val_frame_f1'])

        logger.info('[%s] Running controls (shuffle/mean/zero — 4 conditions)...', task_name)
        t_ctrl = time.time()
        ctrl = evaluate_controls(
            decoder=dec,
            z_aligned=z_val,
            z_train=z_train,
            targets=tok_val,
            codec=codec,
            device=device,
            seed=seed,
        )
        logger.info('[%s] Controls done in %.0fs', task_name, time.time() - t_ctrl)

        results['tasks'][task_name] = {**train_stats, **eval_metrics}
        results['controls'][task_name] = ctrl

        shuffle_gap = ctrl['shuffle_ce'] - ctrl['aligned_ce']
        logger.info(
            '[%s] CE=%.4f tok_acc=%.4f frame_f1=%.4f shuffle_gap=%.4f',
            task_name,
            eval_metrics['val_ce'],
            eval_metrics['val_token_acc'],
            eval_metrics['val_frame_f1'],
            shuffle_gap,
        )

        del dec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Info retention
    if all(k in results['controls'] for k in ('preproj_midi2events', 'preproj_audio2events')):
        intra_ce = results['controls']['preproj_midi2events']['aligned_ce']
        cross_ce = results['controls']['preproj_audio2events']['aligned_ce']
        shuffle_ce = results['controls']['preproj_audio2events']['shuffle_ce']
        denom = shuffle_ce - intra_ce
        info_ratio = (shuffle_ce - cross_ce) / denom if abs(denom) > 1e-6 else float('nan')
        results['info_retention_events'] = {
            'ratio': float(info_ratio),
            'formula': '(shuffle_ce - cross_ce) / (shuffle_ce - intra_ce)',
            'raw': {
                'intra_ce': float(intra_ce),
                'cross_ce': float(cross_ce),
                'shuffle_ce': float(shuffle_ce),
            },
        }

    # 5. Load baseline results for comparison
    baseline_path = out_dir / 'test11_perceptual.json'
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        results['comparison'] = _build_comparison(results, baseline)

    # 6. Save
    result_path = out_dir / 'test11_preproj_ab.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info('Results saved: %s', result_path)

    return results


def _build_comparison(preproj: Dict, baseline: Dict) -> Dict[str, Any]:
    """Build side-by-side comparison between pre-proj and baseline (post-proj)."""
    comp: Dict[str, Any] = {}

    task_pairs = [
        ('preproj_midi2events', 'midi2events'),
        ('preproj_audio2events', 'audio2events'),
    ]

    for pp_task, bl_task in task_pairs:
        pp = preproj['tasks'].get(pp_task, {})
        bl = baseline.get('tasks', {}).get(bl_task, {})
        pp_ctrl = preproj['controls'].get(pp_task, {})
        bl_ctrl = baseline.get('controls', {}).get(bl_task, {})

        if not pp or not bl:
            continue

        bl_gap = (bl_ctrl.get('shuffle_ce', 0) - bl_ctrl.get('aligned_ce', 0)) if bl_ctrl else None
        pp_gap = (pp_ctrl.get('shuffle_ce', 0) - pp_ctrl.get('aligned_ce', 0)) if pp_ctrl else None

        comp[bl_task] = {
            'z_dim': {'baseline': 256, 'preproj': pp.get('z_dim', '?')},
            'n_params': {'baseline': '~36.4M', 'preproj': pp.get('n_params')},
            'best_epoch': {'baseline': bl.get('best_epoch'), 'preproj': pp.get('best_epoch')},
            'val_ce': {
                'baseline': bl.get('val_ce'),
                'preproj': pp.get('val_ce'),
            },
            'token_acc': {
                'baseline': bl.get('val_token_acc'),
                'preproj': pp.get('val_token_acc'),
            },
            'frame_f1': {
                'baseline': bl.get('val_frame_f1'),
                'preproj': pp.get('val_frame_f1'),
            },
            'shuffle_gap': {
                'baseline': bl_gap,
                'preproj': pp_gap,
            },
        }

    return comp


def _print_comparison_table(all_results: Dict[str, Dict]) -> None:
    """Print formatted comparison table to stdout."""
    print('\n' + '=' * 80)
    print('PRE-PROJECTION A/B TEST — RESULTS')
    print('=' * 80)

    for arm, res in all_results.items():
        comp = res.get('comparison', {})
        if not comp:
            print(f'\n--- {arm} (no baseline for comparison) ---')
            for task_name, task_data in res.get('tasks', {}).items():
                ctrl = res.get('controls', {}).get(task_name, {})
                gap = ctrl.get('shuffle_ce', 0) - ctrl.get('aligned_ce', 0)
                print(f'  {task_name}: z_dim={task_data.get("z_dim")} '
                      f'CE={task_data.get("val_ce", "?"):.4f} '
                      f'tok_acc={task_data.get("val_token_acc", "?"):.4f} '
                      f'f1={task_data.get("val_frame_f1", "?"):.4f} '
                      f'gap={gap:.4f}')
            continue

        print(f'\n--- {arm} ---')
        for task, data in comp.items():
            print(f'\n  {task}:')
            print(f'    {"metric":15s}  {"baseline(256d)":>15s}  {"preproj":>15s}  {"delta":>10s}')
            print(f'    {"─" * 60}')

            for metric in ('val_ce', 'token_acc', 'frame_f1', 'shuffle_gap'):
                vals = data.get(metric, {})
                bl = vals.get('baseline')
                pp = vals.get('preproj')

                bl_s = f'{bl:.4f}' if isinstance(bl, (int, float)) and bl is not None else '—'
                pp_s = f'{pp:.4f}' if isinstance(pp, (int, float)) and pp is not None else '—'

                if isinstance(bl, (int, float)) and isinstance(pp, (int, float)):
                    delta = pp - bl
                    d_s = f'{delta:+.4f}'
                else:
                    d_s = '—'

                print(f'    {metric:15s}  {bl_s:>15s}  {pp_s:>15s}  {d_s:>10s}')

    print('\n' + '=' * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description='Gate5B Test11 Pre-Projection A/B Test')
    parser.add_argument('--arms', type=str, nargs='+', default=['D0', 'a4r'],
                        help='Arms to test (default: D0 a4r)')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-precompute', action='store_true',
                        help='Skip event target precomputation (use existing caches)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Load existing decoder checkpoints instead of training')

    args = parser.parse_args()
    set_seed(args.seed)

    codec = MidiEventCodec(CODEC_CFG)

    if not args.skip_precompute:
        precompute_event_targets(
            maestro_dir=Path(args.maestro_dir),
            output_dir=GATE5B_RESULTS_DIR,
            codec=codec,
            seed=args.seed,
        )

    all_results: Dict[str, Any] = {}

    for arm in args.arms:
        model_path = DEFAULT_CHECKPOINTS.get(arm)
        if not model_path:
            logger.warning('No default checkpoint for arm %s, skipping', arm)
            continue

        all_results[arm] = run_arm(
            arm=arm,
            model_path=model_path,
            maestro_dir=Path(args.maestro_dir),
            codec=codec,
            device=args.device,
            num_workers=args.num_workers,
            seed=args.seed,
            skip_train=args.skip_train,
        )

    # Summary
    summary_path = GATE5B_RESULTS_DIR / 'test11_preproj_ab_summary.json'
    with summary_path.open('w') as f:
        json.dump(all_results, f, indent=2)
    logger.info('Summary saved: %s', summary_path)

    _print_comparison_table(all_results)


if __name__ == '__main__':
    main()
