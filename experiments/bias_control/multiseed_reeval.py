#!/usr/bin/env python3
"""Multi-seed re-evaluation for foundation lock desempate (e25 vs e26).

Loads each checkpoint once, extracts embeddings once, then runs
structured pool evaluation with multiple seeds. Fast: ~5 min total.
"""

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bias_control.architectures.cross_modal_model import CrossModalModel
from src.bias_control.datasets.maestro_segments import MaestroSegmentDataset
from experiments.bias_control.evaluate_structured_pool import (
    extract_all_embeddings,
    build_segment_index,
    evaluate_with_precomputed_embeddings,
    analyze_hard_negatives_fast,
    PoolConfig,
)

SEEDS = [42, 123, 456, 789]
CHECKPOINTS = {
    'e25': 'data/bias_control_medium/training_outputs/bloqueA_runD-02/checkpoint_epoch25.pt',
    'e26': 'data/bias_control_medium/training_outputs/bloqueA_runD-02/checkpoint_epoch26.pt',
}
MAESTRO_DIR = 'data/maestro_v3/maestro-v3.0.0'
OUTPUT_FILE = 'data/bias_control_medium/training_outputs/bloqueA_runD-02/multiseed_reeval.json'


def evaluate_multi_seed(audio_embs, midi_embs, dataset, index, config, seeds):
    """Run structured pool eval with multiple seeds, return per-seed results."""
    results = {}
    for seed in seeds:
        t0 = time.time()
        a2m = evaluate_with_precomputed_embeddings(
            audio_embs, midi_embs, dataset, index, config,
            direction='a2m', seed=seed
        )
        m2a = evaluate_with_precomputed_embeddings(
            audio_embs, midi_embs, dataset, index, config,
            direction='m2a', seed=seed
        )
        hard = analyze_hard_negatives_fast(
            audio_embs, midi_embs, dataset, index,
            n_samples=500, seed=seed
        )

        S = min(a2m['mean_recall@10'], m2a['mean_recall@10'])
        gap = abs(a2m['mean_recall@10'] - m2a['mean_recall@10'])

        results[seed] = {
            'S': float(S),
            'a2m_r10': float(a2m['mean_recall@10']),
            'm2a_r10': float(m2a['mean_recall@10']),
            'hard_neg': float(hard['accuracy_vs_same_piece']),
            'gap': float(gap),
            'elapsed_s': round(time.time() - t0, 1),
        }
        print(f"  seed={seed}: S={S*100:.1f}%  A2M={a2m['mean_recall@10']*100:.1f}%  "
              f"M2A={m2a['mean_recall@10']*100:.1f}%  hard_neg={hard['accuracy_vs_same_piece']*100:.1f}%  "
              f"|gap|={gap*100:.1f}pp  ({results[seed]['elapsed_s']}s)")
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset once
    print("\nLoading validation dataset...")
    dataset = MaestroSegmentDataset(
        maestro_dir=MAESTRO_DIR,
        segment_len=4.0,
        hop=1.0,
        split='validation'
    )
    index = build_segment_index(dataset)
    print(f"Dataset: {len(dataset)} segments, {len(index['by_piece'])} pieces")

    config = PoolConfig(pool_size=256, n_hard_negatives=64, n_semi_hard_negatives=32, n_queries=500)

    all_results = {}

    for label, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING {label}: {ckpt_path}")
        print(f"{'='*60}")

        # Load model
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = CrossModalModel(audio_encoder='lite')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()

        # Extract embeddings once
        print("Extracting embeddings (one-time)...")
        torch.cuda.empty_cache()
        t0 = time.time()
        audio_embs, midi_embs = extract_all_embeddings(
            model, dataset, device, batch_size=64, num_workers=8
        )
        print(f"Embeddings extracted in {time.time()-t0:.1f}s")

        # Multi-seed eval
        print(f"\nRunning eval with {len(SEEDS)} seeds...")
        all_results[label] = evaluate_multi_seed(
            audio_embs, midi_embs, dataset, index, config, SEEDS
        )

        # Cleanup
        del model, checkpoint
        torch.cuda.empty_cache()

    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*60}\n")

    for label in CHECKPOINTS:
        res = all_results[label]
        S_vals = [r['S'] for r in res.values()]
        a2m_vals = [r['a2m_r10'] for r in res.values()]
        m2a_vals = [r['m2a_r10'] for r in res.values()]
        hard_vals = [r['hard_neg'] for r in res.values()]
        gap_vals = [r['gap'] for r in res.values()]

        print(f"{label}:")
        print(f"  S:        {np.mean(S_vals)*100:.1f}% +/- {np.std(S_vals)*100:.1f}%  "
              f"(range: {np.min(S_vals)*100:.1f}-{np.max(S_vals)*100:.1f}%)")
        print(f"  A2M:      {np.mean(a2m_vals)*100:.1f}% +/- {np.std(a2m_vals)*100:.1f}%")
        print(f"  M2A:      {np.mean(m2a_vals)*100:.1f}% +/- {np.std(m2a_vals)*100:.1f}%")
        print(f"  hard_neg: {np.mean(hard_vals)*100:.1f}% +/- {np.std(hard_vals)*100:.1f}%")
        print(f"  |gap|:    {np.mean(gap_vals)*100:.1f}pp +/- {np.std(gap_vals)*100:.1f}pp")
        print()

    # Delta
    e25 = all_results['e25']
    e26 = all_results['e26']
    dS = np.mean([r['S'] for r in e25.values()]) - np.mean([r['S'] for r in e26.values()])
    print(f"Delta (e25 - e26):")
    print(f"  S:        {dS*100:+.1f}pp")
    dA = np.mean([r['a2m_r10'] for r in e25.values()]) - np.mean([r['a2m_r10'] for r in e26.values()])
    print(f"  A2M:      {dA*100:+.1f}pp")
    dM = np.mean([r['m2a_r10'] for r in e25.values()]) - np.mean([r['m2a_r10'] for r in e26.values()])
    print(f"  M2A:      {dM*100:+.1f}pp")
    dH = np.mean([r['hard_neg'] for r in e25.values()]) - np.mean([r['hard_neg'] for r in e26.values()])
    print(f"  hard_neg: {dH*100:+.1f}pp")
    dG = np.mean([r['gap'] for r in e25.values()]) - np.mean([r['gap'] for r in e26.values()])
    print(f"  |gap|:    {dG*100:+.1f}pp")

    # Save
    output = {
        'seeds': SEEDS,
        'checkpoints': CHECKPOINTS,
        'results': all_results,
        'summary': {}
    }
    for label in CHECKPOINTS:
        res = all_results[label]
        output['summary'][label] = {
            'S_mean': float(np.mean([r['S'] for r in res.values()])),
            'S_std': float(np.std([r['S'] for r in res.values()])),
            'a2m_mean': float(np.mean([r['a2m_r10'] for r in res.values()])),
            'm2a_mean': float(np.mean([r['m2a_r10'] for r in res.values()])),
            'hard_neg_mean': float(np.mean([r['hard_neg'] for r in res.values()])),
            'gap_mean': float(np.mean([r['gap'] for r in res.values()])),
        }

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
