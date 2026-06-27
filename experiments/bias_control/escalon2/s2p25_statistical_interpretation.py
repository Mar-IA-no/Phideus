#!/usr/bin/env python3
"""
S2-P2.5 Statistical Interpretation — Paired Grouped Bootstrap CI Delta

Applies the pre-registered analysis from PREDICCIONES_EPISTEMOLOGICAS_P25.md:
  - For each of 6 factorial arms, compute Delta = S_arm - S_d0 via
    paired_grouped_bootstrap_ci_delta()
  - Operative rule: A > B iff Δ >= 2pp AND CI excludes 0
  - Map results to prediction patterns P1-P6

Usage:
  python experiments/bias_control/escalon2/s2p25_statistical_interpretation.py \
      --lombard-dir data/lombard/FLombard \
      --segment-index data/lombard/segment_index.json \
      --f0-cache data/lombard/f0_cache_noise0.npz \
      --output data/lombard/p25_interpretation
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.bias_control.encoders.speech_egg_encoder import SpeechEGGEncoder
from src.bias_control.encoders.speech_egg_encoder_attn_bias import SpeechEGGEncoderAttnBias
from src.bias_control.encoders.speech_egg_encoder_xattn import SpeechEGGEncoderXAttn
from src.bias_control.encoders.projection import ProjectionHead, ConditionedProjectionHead
from src.bias_control.datasets.lombard_segments_aug import create_lombard_dataloaders_aug
from src.bias_control.vocal_descriptors import load_h_series_norm_stats

from experiments.bias_control.escalon2.eval_escalon2 import (
    extract_embeddings_lombard,
    evaluate_retrieval_lombard,
    paired_grouped_bootstrap_ci_delta,
    grouped_bootstrap_ci,
)
from experiments.bias_control.escalon2.train_escalon2_descriptors import DescriptorComputer
from experiments.bias_control.escalon2.train_escalon2_pca import extract_embeddings_pca

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ── Arm definitions ──────────────────────────────────────────────────────────

ARMS = {
    'd0': {
        'dir': 'data/lombard/d0_control',
        'encoder_class': 'base',
        'descriptor': None,
        'injection': None,
    },
    'v4lin_attnbias': {
        'dir': 'data/lombard/v4lin_attnbias_seed42',
        'encoder_class': 'attn_bias',
        'descriptor': 'v4_lin',
        'descriptor_dim': 4,
        'injection': 'attn_bias',
    },
    'v4lin_xattn': {
        'dir': 'data/lombard/v4lin_xattn_seed42',
        'encoder_class': 'xattn',
        'descriptor': 'v4_lin',
        'descriptor_dim': 4,
        'injection': 'xattn',
    },
    'hseries_attnbias': {
        'dir': 'data/lombard/hseries_attnbias_seed42',
        'encoder_class': 'attn_bias',
        'descriptor': 'h_series',
        'descriptor_dim': 8,
        'injection': 'attn_bias',
        'has_norm_stats': True,
    },
    'hseries_xattn': {
        'dir': 'data/lombard/hseries_xattn_seed42',
        'encoder_class': 'xattn',
        'descriptor': 'h_series',
        'descriptor_dim': 8,
        'injection': 'xattn',
        'has_norm_stats': True,
    },
    'a4_16k_attnbias': {
        'dir': 'data/lombard/a4_16k_attnbias_seed42',
        'encoder_class': 'attn_bias',
        'descriptor': 'a4_16k',
        'descriptor_dim': 8,
        'injection': 'attn_bias',
    },
    'a4_16k_xattn': {
        'dir': 'data/lombard/a4_16k_xattn_30ep_seed42',
        'encoder_class': 'xattn',
        'descriptor': 'a4_16k',
        'descriptor_dim': 8,
        'injection': 'xattn',
    },
    'v4lin_pca': {
        'dir': 'data/lombard/v4lin_pca_seed42',
        'encoder_class': 'pca',
        'descriptor': 'v4_lin',
        'descriptor_dim': 4,
        'injection': 'pca',
    },
    'hseries_pca': {
        'dir': 'data/lombard/hseries_pca_seed42',
        'encoder_class': 'pca',
        'descriptor': 'h_series',
        'descriptor_dim': 8,
        'injection': 'pca',
        'has_norm_stats': True,
    },
    'a4_16k_pca': {
        'dir': 'data/lombard/a4_16k_pca_seed42',
        'encoder_class': 'pca',
        'descriptor': 'a4_16k',
        'descriptor_dim': 8,
        'injection': 'pca',
    },
}


def load_model(arm_name, arm_cfg, device):
    """Load encoder models and projections from checkpoint."""
    ckpt_path = os.path.join(arm_cfg['dir'], 'best_model.pt')
    logger.info(f"Loading {arm_name} from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    enc_class = arm_cfg['encoder_class']

    if enc_class == 'base':
        speech_enc = SpeechEGGEncoder(output_dim=512, n_layers=4, n_heads=8)
        egg_enc = SpeechEGGEncoder(output_dim=512, n_layers=4, n_heads=8)
    elif enc_class == 'attn_bias':
        desc_dim = arm_cfg['descriptor_dim']
        speech_enc = SpeechEGGEncoderAttnBias(
            descriptor_dim=desc_dim, d_bias=16,
            output_dim=512, n_layers=4, n_heads=8,
        )
        egg_enc = SpeechEGGEncoderAttnBias(
            descriptor_dim=desc_dim, d_bias=16,
            output_dim=512, n_layers=4, n_heads=8,
        )
    elif enc_class == 'xattn':
        desc_dim = arm_cfg['descriptor_dim']
        speech_enc = SpeechEGGEncoderXAttn(
            descriptor_dim=desc_dim, n_xattn_heads=4,
            output_dim=512, n_layers=4, n_heads=8,
        )
        egg_enc = SpeechEGGEncoderXAttn(
            descriptor_dim=desc_dim, n_xattn_heads=4,
            output_dim=512, n_layers=4, n_heads=8,
        )
    elif enc_class == 'pca':
        # PCA/FiLM: base encoder + ConditionedProjectionHead
        speech_enc = SpeechEGGEncoder(output_dim=512, n_layers=4, n_heads=8)
        egg_enc = SpeechEGGEncoder(output_dim=512, n_layers=4, n_heads=8)
    else:
        raise ValueError(f"Unknown encoder class: {enc_class}")

    if enc_class == 'pca':
        desc_dim = arm_cfg['descriptor_dim']
        proj_speech = ConditionedProjectionHead(
            input_dim=512, hidden_dim=512, output_dim=256, cond_dim=desc_dim,
        )
        proj_egg = ConditionedProjectionHead(
            input_dim=512, hidden_dim=512, output_dim=256, cond_dim=desc_dim,
        )
    else:
        proj_speech = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=256)
        proj_egg = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=256)

    speech_enc.load_state_dict(ckpt['speech_enc'])
    egg_enc.load_state_dict(ckpt['egg_enc'])
    proj_speech.load_state_dict(ckpt['proj_speech'])
    proj_egg.load_state_dict(ckpt['proj_egg'])

    speech_enc.to(device).eval()
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

    return speech_enc, egg_enc, proj_speech, proj_egg, descriptor_fn, ckpt


def main():
    parser = argparse.ArgumentParser(description='S2-P2.5 Statistical Interpretation')
    parser.add_argument('--lombard-dir', type=str, required=True)
    parser.add_argument('--segment-index', type=str, required=True)
    parser.add_argument('--f0-cache', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Number of bootstrap iterations (default: 10000 for publication)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    os.makedirs(args.output, exist_ok=True)

    # ── Create test dataloader ───────────────────────────────────────────
    logger.info("Creating dataloaders...")
    _, _, test_loader = create_lombard_dataloaders_aug(
        lombard_dir=args.lombard_dir,
        segment_index_path=args.segment_index,
        f0_cache_path=args.f0_cache,
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

        speech_enc, egg_enc, proj_speech, proj_egg, descriptor_fn, ckpt = \
            load_model(arm_name, arm_cfg, device)

        logger.info("Extracting embeddings on test set...")
        if arm_cfg.get('injection') == 'pca':
            speech_embs, egg_embs = extract_embeddings_pca(
                speech_enc, egg_enc, proj_speech, proj_egg,
                test_loader, device=str(device),
                descriptor_computer=descriptor_fn,
            )
        else:
            speech_embs, egg_embs = extract_embeddings_lombard(
                speech_enc, egg_enc, proj_speech, proj_egg,
                test_loader, device=str(device),
                descriptor_fn=descriptor_fn,
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
        del speech_enc, egg_enc, proj_speech, proj_egg
        torch.cuda.empty_cache()

    # ── Pairwise delta comparisons vs D0 ─────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("PAIRWISE DELTA COMPARISONS vs D0")
    logger.info(f"{'='*60}")

    d0_pq = per_query_results['d0']['per_query']
    delta_results = {}

    for arm_name in ARMS:
        if arm_name == 'd0':
            continue

        arm_pq = per_query_results[arm_name]['per_query']
        delta = paired_grouped_bootstrap_ci_delta(
            arm_pq, d0_pq,
            alpha=0.05, n_bootstrap=args.n_bootstrap, seed=42, k=10,
        )

        delta_results[arm_name] = delta
        logger.info(
            f"  {arm_name} vs D0: Δ={delta['delta_point_pp']:+.1f}pp "
            f"CI=[{delta['ci_lo_pp']:+.1f}, {delta['ci_hi_pp']:+.1f}] "
            f"→ {delta['declaration']}"
        )

    # ── Summary table ────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY TABLE")
    logger.info(f"{'='*60}")
    logger.info(f"{'Arm':<22} {'S':>6} {'Δ vs D0':>8} {'CI_Δ':>18} {'Declaration':>12}")
    logger.info("-" * 70)

    d0_S = per_query_results['d0']['S']
    logger.info(f"{'d0 (baseline)':<22} {d0_S*100:>5.1f}% {'---':>8} {'---':>18} {'baseline':>12}")

    for arm_name in ['v4lin_attnbias', 'v4lin_xattn', 'v4lin_pca',
                     'hseries_attnbias', 'hseries_xattn', 'hseries_pca',
                     'a4_16k_attnbias', 'a4_16k_xattn', 'a4_16k_pca']:
        S = per_query_results[arm_name]['S']
        d = delta_results[arm_name]
        ci_str = f"[{d['ci_lo_pp']:+.1f}, {d['ci_hi_pp']:+.1f}]"
        logger.info(
            f"{arm_name:<22} {S*100:>5.1f}% {d['delta_point_pp']:>+7.1f}pp {ci_str:>18} {d['declaration']:>12}"
        )

    # ── Pairwise delta between descriptor families ───────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("INTER-DESCRIPTOR PAIRWISE DELTAS (within same mechanism)")
    logger.info(f"{'='*60}")

    # Within each mechanism: v4lin vs hseries, v4lin vs a4_16k, hseries vs a4_16k
    inter_deltas = {}
    for mech in ['attnbias', 'xattn', 'pca']:
        descs = [f'v4lin_{mech}', f'hseries_{mech}', f'a4_16k_{mech}']
        for i in range(len(descs)):
            for j in range(i+1, len(descs)):
                a, b = descs[i], descs[j]
                pq_a = per_query_results[a]['per_query']
                pq_b = per_query_results[b]['per_query']
                d = paired_grouped_bootstrap_ci_delta(pq_a, pq_b,
                    alpha=0.05, n_bootstrap=args.n_bootstrap, seed=42, k=10)
                key = f"{a}_vs_{b}"
                inter_deltas[key] = d
                logger.info(
                    f"  {a} vs {b}: Δ={d['delta_point_pp']:+.1f}pp "
                    f"CI=[{d['ci_lo_pp']:+.1f}, {d['ci_hi_pp']:+.1f}] "
                    f"→ {d['declaration']}"
                )

    # Within descriptor: all mechanism pairs (attn_bias, xattn, pca)
    logger.info(f"\n{'='*60}")
    logger.info("INTER-MECHANISM PAIRWISE DELTAS (within same descriptor)")
    logger.info(f"{'='*60}")

    for desc in ['v4lin', 'hseries', 'a4_16k']:
        mechs = [f'{desc}_attnbias', f'{desc}_xattn', f'{desc}_pca']
        for i in range(len(mechs)):
            for j in range(i+1, len(mechs)):
                a, b = mechs[i], mechs[j]
                pq_a = per_query_results[a]['per_query']
                pq_b = per_query_results[b]['per_query']
                d = paired_grouped_bootstrap_ci_delta(pq_a, pq_b,
                    alpha=0.05, n_bootstrap=args.n_bootstrap, seed=42, k=10)
                key = f"{a}_vs_{b}"
                inter_deltas[key] = d
                logger.info(
                    f"  {a} vs {b}: Δ={d['delta_point_pp']:+.1f}pp "
                    f"CI=[{d['ci_lo_pp']:+.1f}, {d['ci_hi_pp']:+.1f}] "
                    f"→ {d['declaration']}"
                )

    # ── Save all results ─────────────────────────────────────────────────
    all_results = {
        'arms': {},
        'delta_vs_d0': {},
        'inter_deltas': {},
        'n_bootstrap': args.n_bootstrap,
    }
    for arm_name in ARMS:
        pqr = per_query_results[arm_name]
        all_results['arms'][arm_name] = {
            'S': pqr['S'], 'S2E': pqr['S2E'], 'E2S': pqr['E2S'],
            'ci_lo': pqr['ci']['S_ci_lo'], 'ci_hi': pqr['ci']['S_ci_hi'],
            'best_epoch': pqr['best_epoch'],
        }
    for k, v in delta_results.items():
        all_results['delta_vs_d0'][k] = {
            'delta_pp': v['delta_point_pp'],
            'ci_lo_pp': v['ci_lo_pp'],
            'ci_hi_pp': v['ci_hi_pp'],
            'declaration': v['declaration'],
            'S_arm': v['S_A_point'],
            'S_d0': v['S_B_point'],
        }
    for k, v in inter_deltas.items():
        all_results['inter_deltas'][k] = {
            'delta_pp': v['delta_point_pp'],
            'ci_lo_pp': v['ci_lo_pp'],
            'ci_hi_pp': v['ci_hi_pp'],
            'declaration': v['declaration'],
        }

    output_path = os.path.join(args.output, 'p25_full_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nFull results saved to {output_path}")

    # ── Prediction matrix mapping ────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("PREDICTION MATRIX MAPPING")
    logger.info(f"{'='*60}")

    # Pre-registered patterns (from PREDICCIONES_EPISTEMOLOGICAS_P25.md):
    # P1: B > D0, C ≈ D0 → HIT core validated
    # P2: B > C > D0 → HIT + general signal boost
    # P3: All > D0 equally → mechanism > content (null for HIT)
    # P4: All ≈ D0 → attention mechanisms don't help
    # P5: C > B → A4 better than H-series → HIT refuted
    # P6: A(V4-lin) > B(H-series) → temporal ratios > harmonic structure

    # Classify: B = H-series, C = A4-16k, A = V4-lin
    # Use best mechanism for each descriptor (max S across all 3 mechanisms)
    best_mech = {}
    for desc in ['v4lin', 'hseries', 'a4_16k']:
        candidates = [f'{desc}_attnbias', f'{desc}_xattn', f'{desc}_pca']
        best_mech[desc] = max(candidates, key=lambda n: per_query_results[n]['S'])

    d0_S_val = per_query_results['d0']['S']

    # Get declarations for best mechanism of each descriptor vs D0
    decl_A = delta_results[best_mech['v4lin']]['declaration']
    decl_B = delta_results[best_mech['hseries']]['declaration']
    decl_C = delta_results[best_mech['a4_16k']]['declaration']

    logger.info(f"  Best mechanism per descriptor:")
    logger.info(f"    A (V4-lin):   {best_mech['v4lin']} → S={per_query_results[best_mech['v4lin']]['S']*100:.1f}%")
    logger.info(f"    B (H-series): {best_mech['hseries']} → S={per_query_results[best_mech['hseries']]['S']*100:.1f}%")
    logger.info(f"    C (A4-16k):   {best_mech['a4_16k']} → S={per_query_results[best_mech['a4_16k']]['S']*100:.1f}%")
    logger.info(f"    D0 baseline:  S={d0_S_val*100:.1f}%")

    logger.info(f"\n  Declarations vs D0:")
    logger.info(f"    A vs D0: {decl_A}")
    logger.info(f"    B vs D0: {decl_B}")
    logger.info(f"    C vs D0: {decl_C}")

    # B vs C comparison
    pq_b = per_query_results[best_mech['hseries']]['per_query']
    pq_c = per_query_results[best_mech['a4_16k']]['per_query']
    d_bc = paired_grouped_bootstrap_ci_delta(pq_b, pq_c,
        alpha=0.05, n_bootstrap=args.n_bootstrap, seed=42, k=10)
    decl_BC = d_bc['declaration']
    logger.info(f"    B vs C: {decl_BC} (Δ={d_bc['delta_point_pp']:+.1f}pp)")

    # A vs B comparison
    pq_a = per_query_results[best_mech['v4lin']]['per_query']
    d_ab = paired_grouped_bootstrap_ci_delta(pq_a, pq_b,
        alpha=0.05, n_bootstrap=args.n_bootstrap, seed=42, k=10)
    decl_AB = d_ab['declaration']
    logger.info(f"    A vs B: {decl_AB} (Δ={d_ab['delta_point_pp']:+.1f}pp)")

    # Pattern matching
    logger.info(f"\n  Pattern matching:")
    B_gt_D0 = decl_B in ("A > B",)  # B is model A in the delta call
    # Recalculate: in delta_results, arm is A, d0 is B
    B_gt_D0 = delta_results[best_mech['hseries']]['declaration'] == "A > B"
    C_eq_D0 = delta_results[best_mech['a4_16k']]['declaration'] == "A ≈ B"
    C_gt_D0 = delta_results[best_mech['a4_16k']]['declaration'] == "A > B"
    B_eq_D0 = delta_results[best_mech['hseries']]['declaration'] == "A ≈ B"
    A_gt_D0 = delta_results[best_mech['v4lin']]['declaration'] == "A > B"
    A_eq_D0 = delta_results[best_mech['v4lin']]['declaration'] == "A ≈ B"

    # Note: in paired_grouped_bootstrap_ci_delta(arm, d0), "A > B" means arm > d0

    patterns = []
    if B_gt_D0 and C_eq_D0:
        patterns.append("P1: B > D0, C ≈ D0 → HIT core validated")
    if B_gt_D0 and C_gt_D0:
        patterns.append("P2: B > D0 and C > D0 → HIT + general signal boost")
    if A_gt_D0 and B_gt_D0 and C_gt_D0:
        patterns.append("P3 candidate: All > D0 → check if equally (mechanism > content)")
    if A_eq_D0 and B_eq_D0 and C_eq_D0:
        patterns.append("P4: All ≈ D0 → attention mechanisms don't help Speech↔EGG")
    if d_bc['declaration'] == "B > A":  # C > B (C is model B in d_bc call)
        patterns.append("P5: C > B → A4 better than H-series → HIT refuted for this modality")
    if d_ab['declaration'] == "A > B":  # A > B
        patterns.append("P6: A(V4-lin) > B(H-series) → temporal ratios > harmonic structure")

    if not patterns:
        patterns.append("No clean pattern match — mixed results")

    for p in patterns:
        logger.info(f"    ✓ {p}")

    # Save prediction mapping
    all_results['prediction_mapping'] = {
        'best_mechanisms': {k: v for k, v in best_mech.items()},
        'declarations_vs_d0': {
            'A_v4lin': decl_A,
            'B_hseries': decl_B,
            'C_a4_16k': decl_C,
        },
        'B_vs_C': {
            'declaration': decl_BC,
            'delta_pp': d_bc['delta_point_pp'],
        },
        'A_vs_B': {
            'declaration': decl_AB,
            'delta_pp': d_ab['delta_point_pp'],
        },
        'matched_patterns': patterns,
    }

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nDone. Results at {output_path}")


if __name__ == '__main__':
    main()
