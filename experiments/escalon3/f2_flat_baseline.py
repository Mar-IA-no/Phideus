#!/usr/bin/env python3
"""E3-P2: Flat Cross-Modal Baseline (VICReg on R^256).

Dual encoder (audio XY + image) with ProjectionHead and standard VICReg loss.
L0 baseline. best_model selected on VAL, final eval on TEST (no contamination).

OOD evaluation uses cross-set retrieval (query OOD → reduced reference atlas).

Usage:
    # Train
    python experiments/escalon3/f2_flat_baseline.py \
        --data data/escalon3/bundled \
        --output data/escalon3/p2_flat_seed42 \
        --epochs 50 --seed 42

    # Re-evaluate only (no training)
    python experiments/escalon3/f2_flat_baseline.py \
        --data data/escalon3/bundled \
        --output data/escalon3/p2_flat_seed42 \
        --eval-only --checkpoint data/escalon3/p2_flat_seed42/best_model.pt
"""

import argparse
import json
import math
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.escalon3.lissajous_dataset import (
    create_lissajous_dataloaders, load_ood_dataset, collate_lissajous,
    LissajousDataset, load_reduced_gallery_dataset,
)
from src.escalon3.encoders import build_lissajous_audio_encoder, build_lissajous_image_encoder
from src.RNA.vicreg import VICRegLoss
from src.bias_control.encoders.projection import ProjectionHead
from experiments.escalon3.eval_escalon3 import (
    extract_embeddings, run_full_eval,
    evaluate_retrieval_crossset, diagnose_ratio_ood,
    evaluate_render_ood,
    find_positives_scale_ood, find_positives_equiv_ood,
    summarize_positive_coverage,
    _build_key_index, _build_equiv_key_index,
)


# ============================================================
# Utilities (inlined, no cross-experiment dependency)
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LinearWarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def state_dict(self):
        return {'step_count': self.step_count, 'base_lrs': self.base_lrs}

    def load_state_dict(self, d):
        self.step_count = d['step_count']
        self.base_lrs = d['base_lrs']


# ============================================================
# Training
# ============================================================

def _audio_key(enc):
    """Resolve batch key for audio encoder."""
    return 'audio_cqt' if getattr(enc, 'input_kind', 'waveform') == 'cqt' else 'audio'


def train_one_epoch(audio_enc, image_enc, proj_audio, proj_image,
                    vicreg_loss, optimizer, scheduler, loader, device):
    audio_enc.train(); image_enc.train()
    proj_audio.train(); proj_image.train()

    total_loss = 0
    comps = {'invariance': 0, 'variance': 0, 'covariance': 0}
    n = 0
    akey = _audio_key(audio_enc)

    for batch in tqdm(loader, desc="Train", leave=False):
        audio = batch[akey].to(device)
        image = batch['image'].to(device)

        z_a = proj_audio(audio_enc(audio))
        z_i = proj_image(image_enc(image))
        loss_dict = vicreg_loss(z_a, z_i)

        optimizer.zero_grad()
        loss_dict['total'].backward()
        nn.utils.clip_grad_norm_(
            list(audio_enc.parameters()) + list(image_enc.parameters()) +
            list(proj_audio.parameters()) + list(proj_image.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss_dict['total'].item()
        for k in comps:
            comps[k] += loss_dict[k].item()
        n += 1

    return {'total': total_loss / max(n, 1),
            **{k: v / max(n, 1) for k, v in comps.items()}}


@torch.no_grad()
def quick_val_loss(audio_enc, image_enc, proj_audio, proj_image,
                   vicreg_loss, loader, device) -> float:
    audio_enc.eval(); image_enc.eval()
    proj_audio.eval(); proj_image.eval()
    total, n = 0, 0
    for batch in loader:
        z_a = proj_audio(audio_enc(batch[_audio_key(audio_enc)].to(device)))
        z_i = proj_image(image_enc(batch['image'].to(device)))
        total += vicreg_loss(z_a, z_i)['total'].item()
        n += 1
    return total / max(n, 1)


# ============================================================
# Final evaluation suite
# ============================================================

def run_final_eval(
    audio_enc, image_enc, proj_audio, proj_image,
    data_dir: str, scene_dir: str, device: torch.device,
) -> dict:
    """Comprehensive final evaluation: IID + all OOD splits.

    IID: same-set retrieval on test.
    OOD: cross-set retrieval against gallery_reduced_full (train+val+test atlas).
    """
    data_dir = Path(data_dir)
    results = {}

    # === IID test (same-set) ===
    print("\n--- IID Test (same-set) ---")
    test_ds = LissajousDataset(data_dir / 'test')
    test_a, test_i = extract_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                         test_ds, device)
    iid = run_full_eval(test_a, test_i, test_ds, n_queries=min(480, len(test_ds)))
    results['iid_test'] = iid
    print(f"  S={iid['S']:.3f} | A2I@10={iid['a2i']['recall_at_10']:.3f} "
          f"I2A@10={iid['i2a']['recall_at_10']:.3f} | sil={iid['silhouette_combined']:.3f}")

    # === Build reduced reference atlas for OOD ===
    print("\n--- Building gallery_reduced_full (train+val+test atlas) ---")
    gallery_ds = load_reduced_gallery_dataset(data_dir)
    gallery_a, gallery_i = extract_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                               gallery_ds, device)
    print(f"  Gallery size: {len(gallery_ds)} scenes")
    results['ood_gallery'] = {
        'type': 'reduced_full_atlas',
        'splits': ['train', 'val', 'test'],
        'n_gallery': len(gallery_ds),
    }

    gallery_key_idx = _build_key_index(gallery_ds)
    gallery_equiv_idx = _build_equiv_key_index(gallery_ds)

    # === scale-OOD (cross-set: query → atlas) ===
    print("\n--- scale-OOD (cross-set → atlas) ---")
    try:
        scale_ds = load_ood_dataset(data_dir, 'scale_ood')
        scale_a, scale_i = extract_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                               scale_ds, device)

        finder = partial(find_positives_scale_ood,
                         query_ds=scale_ds, gallery_key_index=gallery_key_idx)
        coverage = summarize_positive_coverage(scale_ds, finder)
        results['scale_ood_coverage'] = coverage
        print(f"  Coverage: {coverage['n_with_positive']}/{coverage['n_total_queries']} "
              f"(no_positive={coverage['n_no_positive']}, "
              f"set_size={coverage['positive_set_size_min']}-{coverage['positive_set_size_max']})")

        for direction in ['a2i', 'i2a']:
            r = evaluate_retrieval_crossset(
                scale_a, scale_i, scale_ds,
                gallery_a, gallery_i, gallery_ds,
                direction=direction,
                positive_finder=finder,
                n_queries=min(500, len(scale_ds)))
            results[f'scale_ood_{direction}'] = r
            print(f"  {direction}: R@10={r.get('recall_at_10', 'N/A'):.3f} "
                  f"n_queries={r.get('n_queries', 0)} n_no_pos={r.get('n_no_positive', 0)}")
    except Exception as e:
        print(f"  Skipped: {e}")

    # === equiv-OOD (cross-set: query → atlas) ===
    print("\n--- equiv-OOD (cross-set → atlas) ---")
    try:
        equiv_ds = load_ood_dataset(data_dir, 'equiv_ood')
        equiv_a, equiv_i = extract_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                               equiv_ds, device)

        finder = partial(find_positives_equiv_ood,
                         query_ds=equiv_ds, gallery_equiv_index=gallery_equiv_idx)
        coverage = summarize_positive_coverage(equiv_ds, finder)
        results['equiv_ood_coverage'] = coverage
        print(f"  Coverage: {coverage['n_with_positive']}/{coverage['n_total_queries']} "
              f"(no_positive={coverage['n_no_positive']}, "
              f"set_size={coverage['positive_set_size_min']}-{coverage['positive_set_size_max']})")

        for direction in ['a2i', 'i2a']:
            r = evaluate_retrieval_crossset(
                equiv_a, equiv_i, equiv_ds,
                gallery_a, gallery_i, gallery_ds,
                direction=direction,
                positive_finder=finder,
                n_queries=min(500, len(equiv_ds)))
            results[f'equiv_ood_{direction}'] = r
            print(f"  {direction}: R@10={r.get('recall_at_10', 'N/A'):.3f} "
                  f"n_queries={r.get('n_queries', 0)} n_no_pos={r.get('n_no_positive', 0)}")
    except Exception as e:
        print(f"  Skipped: {e}")

    # === ratio-OOD (no positive — structural diagnostics vs atlas) ===
    print("\n--- ratio-OOD (diagnostics vs atlas, no S) ---")
    try:
        ratio_ds = load_ood_dataset(data_dir, 'ratio_ood')
        ratio_a, ratio_i = extract_embeddings(audio_enc, image_enc, proj_audio, proj_image,
                                               ratio_ds, device)
        diag = diagnose_ratio_ood(ratio_a, ratio_i, ratio_ds,
                                   gallery_a, gallery_i, gallery_ds)
        results['ratio_ood_diagnostics'] = diag
        for rid, info in diag.items():
            top = info['nearest_gallery_ratios']
            top_str = ', '.join(f"rid={r}: {p:.0%}" for r, p in top[:3])
            print(f"  {rid} (n={info['n_queries']}): {top_str}")
    except Exception as e:
        print(f"  Skipped: {e}")

    # === render-OOD (noisy/thick image query → clean audio gallery) ===
    if scene_dir and Path(scene_dir).exists():
        for style in ['noisy', 'thick']:
            print(f"\n--- render-OOD ({style}) ---")
            try:
                r = evaluate_render_ood(
                    audio_enc, image_enc, proj_audio, proj_image,
                    test_ds, scene_dir, device,
                    render_style=style, n_queries=min(480, len(test_ds)))
                results[f'render_ood_{style}'] = r
                print(f"  R@10={r.get('recall_at_10', 'N/A'):.3f}")
            except Exception as e:
                print(f"  Skipped: {e}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="E3-P2: Flat VICReg baseline")
    parser.add_argument("--data", default="data/escalon3/bundled")
    parser.add_argument("--scenes", default="data/escalon3/scenes",
                        help="Scene directory for render-OOD evaluation")
    parser.add_argument("--audio-encoder", default="baseline",
                        choices=["baseline", "attpool", "resatt", "cnnxfmr",
                                 "cqtconv", "cqtshift", "cqthybrid"])
    parser.add_argument("--image-encoder", default="baseline",
                        choices=["baseline"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-enc", type=float, default=5e-4)
    parser.add_argument("--lr-proj", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--lambda-inv", type=float, default=25.0)
    parser.add_argument("--lambda-var", type=float, default=25.0)
    parser.add_argument("--lambda-cov", type=float, default=1.0)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-epochs", nargs='+', type=int,
                        default=[5, 10, 20, 30, 40, 50])
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, run final eval only")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to load for --eval-only")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/escalon3/p2_{args.audio_encoder}_seed{args.seed}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Models
    audio_enc = build_lissajous_audio_encoder(args.audio_encoder, output_dim=256).to(device)
    image_enc = build_lissajous_image_encoder(args.image_encoder, output_dim=256).to(device)
    print(f"Audio encoder: {args.audio_encoder} ({sum(p.numel() for p in audio_enc.parameters()):,} params)")
    print(f"Image encoder: {args.image_encoder} ({sum(p.numel() for p in image_enc.parameters()):,} params)")
    proj_audio = ProjectionHead(input_dim=256, hidden_dim=512,
                                output_dim=args.proj_dim, n_layers=3).to(device)
    proj_image = ProjectionHead(input_dim=256, hidden_dim=512,
                                output_dim=args.proj_dim, n_layers=3).to(device)

    n_params = (sum(p.numel() for p in audio_enc.parameters()) +
                sum(p.numel() for p in image_enc.parameters()) +
                sum(p.numel() for p in proj_audio.parameters()) +
                sum(p.numel() for p in proj_image.parameters()))
    print(f"Total params: {n_params:,}")

    # ========= EVAL-ONLY MODE =========
    if args.eval_only:
        ckpt_path = args.checkpoint or str(output_dir / 'best_model.pt')
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        audio_enc.load_state_dict(ckpt['audio_enc'])
        image_enc.load_state_dict(ckpt['image_enc'])
        proj_audio.load_state_dict(ckpt['proj_audio'])
        proj_image.load_state_dict(ckpt['proj_image'])

        results = run_final_eval(audio_enc, image_enc, proj_audio, proj_image,
                                  args.data, args.scenes, device)
        with open(output_dir / 'final_results_v2.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_dir / 'final_results_v2.json'}")
        return

    # ========= TRAINING MODE =========
    vicreg_loss = VICRegLoss(
        lambda_inv=args.lambda_inv, lambda_var=args.lambda_var, lambda_cov=args.lambda_cov)

    train_loader, val_loader, test_loader = create_lissajous_dataloaders(
        args.data, batch_size=args.batch_size, num_workers=8)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    optimizer = AdamW([
        {'params': audio_enc.parameters(), 'lr': args.lr_enc},
        {'params': image_enc.parameters(), 'lr': args.lr_enc},
        {'params': proj_audio.parameters(), 'lr': args.lr_proj},
        {'params': proj_image.parameters(), 'lr': args.lr_proj},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = LinearWarmupCosineScheduler(optimizer, args.warmup_steps, total_steps)

    history = []
    best_val_S = 0.0

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(audio_enc, image_enc, proj_audio, proj_image,
                                   vicreg_loss, optimizer, scheduler, train_loader, device)
        val_loss = quick_val_loss(audio_enc, image_enc, proj_audio, proj_image,
                                  vicreg_loss, val_loader, device)

        log = {'epoch': epoch, **train_m, 'val_loss': val_loss}

        print(f"E{epoch:02d} | loss={train_m['total']:.4f} "
              f"inv={train_m['invariance']:.4f} var={train_m['variance']:.4f} "
              f"cov={train_m['covariance']:.4f} | val={val_loss:.4f}")

        # Checkpoint every epoch
        ckpt = {
            'epoch': epoch,
            'audio_enc': audio_enc.state_dict(),
            'image_enc': image_enc.state_dict(),
            'proj_audio': proj_audio.state_dict(),
            'proj_image': proj_image.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'config': vars(args),
        }
        torch.save(ckpt, output_dir / f'checkpoint_e{epoch:02d}.pt')

        # Periodic eval on VAL (not test) for monitoring + model selection
        if epoch in args.eval_epochs:
            print(f"  Running val eval at epoch {epoch}...")
            val_ds = val_loader.dataset
            a_embs, i_embs = extract_embeddings(
                audio_enc, image_enc, proj_audio, proj_image, val_ds, device)
            val_eval = run_full_eval(a_embs, i_embs, val_ds,
                                     n_queries=min(448, len(val_ds)))
            log['val_S'] = val_eval['S']
            log['val_sil'] = val_eval['silhouette_combined']
            print(f"  val_S={val_eval['S']:.3f} | val_sil={val_eval['silhouette_combined']:.3f}")

            # Best model by val_S (VICReg loss is NOT a good proxy for retrieval)
            if val_eval['S'] > best_val_S:
                best_val_S = val_eval['S']
                torch.save(ckpt, output_dir / 'best_model.pt')
                print(f"  New best val_S={best_val_S:.3f} (epoch {epoch})")

        history.append(log)

    # ========= FINAL EVAL ON TEST (using best val model) =========
    print("\n=== FINAL EVALUATION (best val model → test) ===")
    best_ckpt = torch.load(output_dir / 'best_model.pt', map_location=device)
    audio_enc.load_state_dict(best_ckpt['audio_enc'])
    image_enc.load_state_dict(best_ckpt['image_enc'])
    proj_audio.load_state_dict(best_ckpt['proj_audio'])
    proj_image.load_state_dict(best_ckpt['proj_image'])

    results = run_final_eval(audio_enc, image_enc, proj_audio, proj_image,
                              args.data, args.scenes, device)
    results['history'] = history
    results['config'] = vars(args)
    results['best_val_S'] = best_val_S
    results['best_epoch'] = best_ckpt['epoch']

    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # GO summary (aligned with CRITERIOS_GO_NO_GO_ESCALON_3.md)
    S = results['iid_test']['S']
    sil = results['iid_test']['silhouette_combined']
    print(f"\n=== E3-P2 GO/NO-GO ===")
    print("  (See CRITERIOS_GO_NO_GO_ESCALON_3.md for canonical vs heuristic distinction)")

    # Canonical: retrieval signal + structured latent + OOD coverage
    retrieval_clear = S > 0.40  # any non-trivial retrieval
    latent_organized = sil > 0.30
    scale_cov = results.get('scale_ood_coverage', {}).get('n_no_positive', -1)
    equiv_cov = results.get('equiv_ood_coverage', {}).get('n_no_positive', -1)
    ood_clean = (scale_cov == 0 and equiv_cov == 0)

    print(f"\n  [CANONICAL] Retrieval signal: S={S:.3f} {'YES' if retrieval_clear else 'WEAK'}")
    print(f"  [CANONICAL] Latent organized: sil={sil:.3f} {'GO' if latent_organized else 'FAIL'}")
    print(f"  [CANONICAL] OOD coverage clean: {'YES' if ood_clean else 'NO'} "
          f"(scale n_no_pos={scale_cov}, equiv n_no_pos={equiv_cov})")

    # Heuristic thresholds (useful for monitoring, not for closing the front)
    print(f"\n  [HEURISTIC] S > 0.60: {S:.3f} {'PASS' if S > 0.60 else 'BELOW'}")
    print(f"  [HEURISTIC] sil > 0.30: {sil:.3f} {'PASS' if sil > 0.30 else 'BELOW'}")

    # Render-OOD as diagnostic
    for style in ['noisy', 'thick']:
        k = f'render_ood_{style}'
        if k in results:
            r10 = results[k].get('recall_at_10', 0)
            print(f"  [DIAGNOSTIC] render-OOD {style}: R@10={r10:.3f}")

    print(f"\n  Best epoch: {best_ckpt['epoch']} (selected on val_S={best_val_S:.3f})")

    canonical_go = retrieval_clear and latent_organized and ood_clean
    print(f"  Canonical reading: {'Baseline usable for P4' if canonical_go else 'NEEDS REVIEW'}")


if __name__ == "__main__":
    main()
