#!/usr/bin/env python3
"""
S2-P3 Statistical Interpretation — Paired Grouped Bootstrap CI Delta

Reevaluates all P3 arms on best checkpoints, computes paired bootstrap
CI for each Δ(arm - P3-D0), and produces the formal closure table.

Operative rule: A > B iff Δ >= 2pp AND CI excludes 0.

Usage:
  python experiments/bias_control/escalon2/interpret_p3.py \
      --lombard-dir data/lombard/FLombard \
      --segment-index data/lombard/segment_index.json \
      --f0-cache data/lombard/f0_cache_noise0.npz \
      --wavlm-features data/lombard/wavlm_features_noise0.npz \
      --output data/lombard/p3_interpretation
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.bias_control.encoders.speech_egg_encoder import SpeechEGGEncoder
from src.bias_control.encoders.projection import ProjectionHead, ConditionedProjectionHead
from src.bias_control.datasets.lombard_precomputed import create_p3_dataloaders
from src.bias_control.vocal_descriptors import get_descriptor_dim, load_h_series_norm_stats

from experiments.bias_control.escalon2.train_escalon2_descriptors import DescriptorComputer
from experiments.bias_control.escalon2.train_escalon2_p3 import extract_embeddings_p3
from experiments.bias_control.escalon2.eval_escalon2 import (
    evaluate_retrieval_lombard,
    paired_grouped_bootstrap_ci_delta,
    grouped_bootstrap_ci,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

WAVLM_DIM = 1024

# ── Arm definitions ──────────────────────────────────────────────────────────

ARMS = {
    'p3_d0': {
        'dir': 'data/lombard/p3_d0_seed42',
        'descriptor': None,
        'descriptor_dim': 0,
        'injection': None,
    },
    'p3_v4lin_pca': {
        'dir': 'data/lombard/p3_v4lin_pca_seed42',
        'descriptor': 'v4_lin',
        'descriptor_dim': 4,
        'injection': 'pca',
    },
    'p3_hseries_pca': {
        'dir': 'data/lombard/p3_hseries_pca_seed42',
        'descriptor': 'h_series',
        'descriptor_dim': 8,
        'injection': 'pca',
        'has_norm_stats': True,
    },
    'p3_a4_16k_pca': {
        'dir': 'data/lombard/p3_a4_16k_pca_seed42',
        'descriptor': 'a4_16k',
        'descriptor_dim': 8,
        'injection': 'pca',
    },
}


def load_p3_model(arm_name, arm_cfg, device):
    """Load P3 model from checkpoint.

    P3 architecture: WavLM (precomputed, 1024d) + SpeechEGGEncoder (EGG side).
    Speech side has no encoder — just a projection from 1024d.
    """
    ckpt_path = os.path.join(arm_cfg['dir'], 'best_model.pt')
    logger.info(f"Loading {arm_name} from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # EGG encoder (always the same)
    egg_enc = SpeechEGGEncoder(output_dim=512, n_layers=4, n_heads=8)

    # Projections: conditioned if descriptor present, plain otherwise
    if arm_cfg['injection'] == 'pca':
        desc_dim = arm_cfg['descriptor_dim']
        proj_speech = ConditionedProjectionHead(
            input_dim=WAVLM_DIM, hidden_dim=512, output_dim=256, cond_dim=desc_dim,
        )
        proj_egg = ConditionedProjectionHead(
            input_dim=512, hidden_dim=512, output_dim=256, cond_dim=desc_dim,
        )
    else:
        proj_speech = ProjectionHead(input_dim=WAVLM_DIM, hidden_dim=512, output_dim=256)
        proj_egg = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=256)

    egg_enc.load_state_dict(ckpt['egg_enc'])
    proj_speech.load_state_dict(ckpt['proj_speech'])
    proj_egg.load_state_dict(ckpt['proj_egg'])

    egg_enc.to(device).eval()
    proj_speech.to(device).eval()
    proj_egg.to(device).eval()

    # Build descriptor_fn if needed
    descriptor_fn = None
    if arm_cfg['descriptor'] is not None:
        h_series_norm_stats = None
        if arm_cfg.get('has_norm_stats'):
            h_series_norm_stats = {}
            for mod in ['speech', 'egg']:
                stats_path = os.path.join(arm_cfg['dir'], f'h_series_norm_{mod}.json')
                h_series_norm_stats[mod] = load_h_series_norm_stats(stats_path, device=str(device))
        descriptor_fn = DescriptorComputer(
            descriptor_type=arm_cfg['descriptor'],
            h_series_norm_stats=h_series_norm_stats,
        )

    return egg_enc, proj_speech, proj_egg, descriptor_fn, ckpt


def main():
    parser = argparse.ArgumentParser(description='S2-P3 Statistical Interpretation')
    parser.add_argument('--lombard-dir', type=str, required=True)
    parser.add_argument('--segment-index', type=str, required=True)
    parser.add_argument('--f0-cache', type=str, required=True)
    parser.add_argument('--wavlm-features', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Number of bootstrap iterations (default: 10000 for publication)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    os.makedirs(args.output, exist_ok=True)

    # ── Create test dataloader (P3: precomputed WavLM features) ──────────
    logger.info("Creating P3 dataloaders...")
    _, _, test_loader = create_p3_dataloaders(
        lombard_dir=args.lombard_dir,
        segment_index_path=args.segment_index,
        f0_cache_path=args.f0_cache,
        wavlm_features_path=args.wavlm_features,
        noise_condition='noise0',
        batch_size=args.batch_size,
        num_workers=8,
    )

    # Load segment metadata for retrieval eval
    with open(args.segment_index) as f:
        seg_data = json.load(f)
    all_segments = seg_data['segments']
    test_segments = [s for s in all_segments if s['split'] == 'test' and s['noise_condition'] == 'noise0']
    logger.info(f"Test segments: {len(test_segments)}")

    # ── Extract embeddings for each arm ──────────────────────────────────
    per_query_results = {}

    for arm_name, arm_cfg in ARMS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing arm: {arm_name}")
        logger.info(f"{'='*60}")

        egg_enc, proj_speech, proj_egg, descriptor_fn, ckpt = \
            load_p3_model(arm_name, arm_cfg, device)

        logger.info("Extracting embeddings on test set...")
        speech_embs, egg_embs = extract_embeddings_p3(
            egg_enc, proj_speech, proj_egg,
            test_loader, device=str(device),
            descriptor_computer=descriptor_fn,
        )
        logger.info(f"Extracted {len(speech_embs)} speech, {len(egg_embs)} egg embeddings")

        logger.info("Running retrieval evaluation...")
        eval_result = evaluate_retrieval_lombard(
            speech_embs, egg_embs, test_segments,
            pool_size=128, n_queries=500, seed=42, k=10,
        )

        S = eval_result['S']
        S2E = eval_result['S2E_at_k']
        E2S = eval_result['E2S_at_k']
        logger.info(f"  S2E@10={S2E*100:.1f}%, E2S@10={E2S*100:.1f}%, S={S*100:.1f}%")

        # Bootstrap CI for this arm
        ci = grouped_bootstrap_ci(eval_result['per_query'], n_bootstrap=args.n_bootstrap, seed=42, k=10)
        logger.info(f"  CI: [{ci['S_ci_lo']*100:.1f}%, {ci['S_ci_hi']*100:.1f}%] (n_speakers={ci['n_speakers']})")

        per_query_results[arm_name] = {
            'per_query': eval_result['per_query'],
            'S': S,
            'S2E': S2E,
            'E2S': E2S,
            'ci': ci,
            'best_epoch': ckpt.get('epoch', '?'),
        }

        # Save per-arm result
        arm_result = {
            'arm': arm_name,
            'S': S, 'S2E_at_10': S2E, 'E2S_at_10': E2S,
            'ci_lo': ci['S_ci_lo'], 'ci_hi': ci['S_ci_hi'],
            'n_queries': eval_result['n_queries'],
            'best_epoch': ckpt.get('epoch', '?'),
        }
        with open(os.path.join(args.output, f'{arm_name}_eval.json'), 'w') as f:
            json.dump(arm_result, f, indent=2)

        # Free GPU memory
        del egg_enc, proj_speech, proj_egg
        torch.cuda.empty_cache()

    # ── Pairwise delta comparisons vs P3-D0 ──────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("PAIRWISE DELTA COMPARISONS vs P3-D0")
    logger.info(f"{'='*60}")

    d0_pq = per_query_results['p3_d0']['per_query']
    delta_results = {}

    for arm_name in ARMS:
        if arm_name == 'p3_d0':
            continue

        arm_pq = per_query_results[arm_name]['per_query']
        delta = paired_grouped_bootstrap_ci_delta(
            arm_pq, d0_pq,
            alpha=0.05, n_bootstrap=args.n_bootstrap, seed=42, k=10,
        )

        delta_results[arm_name] = delta
        logger.info(
            f"  {arm_name} vs P3-D0: Δ={delta['delta_point_pp']:+.1f}pp "
            f"CI=[{delta['ci_lo_pp']:+.1f}, {delta['ci_hi_pp']:+.1f}] "
            f"→ {delta['declaration']}"
        )

    # ── Also compute delta vs P2-D0 (78.8% is P3-D0, P2-D0 = 77.8%) ────
    logger.info(f"\n{'='*60}")
    logger.info("P3-D0 vs P2-D0 REFERENCE")
    logger.info(f"{'='*60}")
    p3d0_S = per_query_results['p3_d0']['S']
    logger.info(f"  P3-D0 = {p3d0_S*100:.1f}%, P2-D0 = 77.8% (from training log)")
    logger.info(f"  Regime shift Δ = {(p3d0_S - 0.778)*100:+.1f}pp")

    # ── Summary table ────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY TABLE — P3 CLOSURE")
    logger.info(f"{'='*60}")
    logger.info(f"{'Arm':<22} {'BestEp':>6} {'S':>6} {'Δ vs P3-D0':>10} {'CI_Δ':>18} {'Declaration':>12}")
    logger.info("-" * 78)

    d0_S = per_query_results['p3_d0']['S']
    d0_ep = per_query_results['p3_d0']['best_epoch']
    logger.info(f"{'p3_d0 (baseline)':<22} {d0_ep:>6} {d0_S*100:>5.1f}% {'---':>10} {'---':>18} {'baseline':>12}")

    for arm_name in ['p3_v4lin_pca', 'p3_hseries_pca', 'p3_a4_16k_pca']:
        S = per_query_results[arm_name]['S']
        ep = per_query_results[arm_name]['best_epoch']
        d = delta_results[arm_name]
        ci_str = f"[{d['ci_lo_pp']:+.1f}, {d['ci_hi_pp']:+.1f}]"
        logger.info(
            f"{arm_name:<22} {ep:>6} {S*100:>5.1f}% {d['delta_point_pp']:>+9.1f}pp {ci_str:>18} {d['declaration']:>12}"
        )

    # ── Save all results ─────────────────────────────────────────────────
    all_results = {
        'regime': 'P3 (WavLM-Large frozen speech + trainable EGG)',
        'n_bootstrap': args.n_bootstrap,
        'p2_d0_reference': 0.778,
        'arms': {},
        'delta_vs_p3_d0': {},
    }
    for arm_name in ARMS:
        pqr = per_query_results[arm_name]
        all_results['arms'][arm_name] = {
            'S': pqr['S'], 'S2E': pqr['S2E'], 'E2S': pqr['E2S'],
            'ci_lo': pqr['ci']['S_ci_lo'], 'ci_hi': pqr['ci']['S_ci_hi'],
            'best_epoch': pqr['best_epoch'],
        }
    for k, v in delta_results.items():
        all_results['delta_vs_p3_d0'][k] = {
            'delta_pp': v['delta_point_pp'],
            'ci_lo_pp': v['ci_lo_pp'],
            'ci_hi_pp': v['ci_hi_pp'],
            'declaration': v['declaration'],
            'S_arm': v['S_A_point'],
            'S_d0': v['S_B_point'],
        }

    output_path = os.path.join(args.output, 'p3_full_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nFull results saved to {output_path}")

    # ── Formal reading ───────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("FORMAL READING")
    logger.info(f"{'='*60}")
    logger.info("P3 tests whether a stronger speech encoder (WavLM-Large frozen)")
    logger.info("makes descriptor signal newly visible in the Speech<->EGG front.")
    logger.info("")
    n_above = sum(1 for d in delta_results.values() if d['delta_point_pp'] > 0)
    n_sig = sum(1 for d in delta_results.values()
                if d['delta_point_pp'] >= 2.0 and d['ci_lo_pp'] > 0)
    logger.info(f"  Arms above P3-D0: {n_above}/{len(delta_results)}")
    logger.info(f"  Arms significantly above (Δ>=2pp, CI>0): {n_sig}/{len(delta_results)}")
    if n_sig == 0:
        logger.info("")
        logger.info("  NULL CONFIRMED under foundation encoder regime.")
        logger.info("  Encoder confound eliminated. The null is about the")
        logger.info("  Speech<->EGG pair, not about encoder weakness.")
    else:
        logger.info("")
        logger.info("  Positive signal detected. Further investigation warranted.")


if __name__ == '__main__':
    main()
