# Gate 3 DANN - Comparative Evaluation

**Date**: 2026-02-06 19:43
**Pool**: 256 candidates (64 hard + 32 semi-hard + 159 random + 1 positive)
**Queries**: 500, **Seed**: 42

## Retrieval Metrics (Structured Pool)

| Checkpoint | Description | R@1 a2m | R@5 a2m | R@10 a2m | R@1 m2a | R@5 m2a | R@10 m2a | MRR a2m | MRR m2a |
|-----------|-------------|---------|---------|----------|---------|---------|----------|---------|----------|
| gate2_ep45 | Gate 2 baseline (no DANN) | 4.4% | 20.8% | 34.4% | 5.2% | 24.6% | 37.6% | 0.138 | 0.158 |
| runA_best_ep7 | Run A (sin norm), best recall, epoch 7 | 6.0% | 17.2% | 27.8% | 6.0% | 22.6% | 35.4% | 0.132 | 0.148 |
| runB_ep5 | Run B (F.normalize), epoch 5, lambda~0.17 | 3.6% | 16.2% | 24.6% | 4.4% | 20.2% | 32.0% | 0.112 | 0.132 |
| runB_ep10 | Run B (F.normalize), epoch 10, lambda~0.33 | 5.4% | 18.2% | 29.8% | 4.2% | 23.2% | 34.6% | 0.130 | 0.140 |
| runC_best_ep4 | Run C (optimized), best recall, epoch 4 | 5.8% | 21.8% | 34.6% | 6.0% | 22.6% | 39.2% | 0.148 | 0.159 |
| runC_ep13 | Run C (optimized), epoch 13, lambda=0.80 | 6.2% | 20.0% | 32.2% | 6.0% | 24.6% | 38.0% | 0.144 | 0.163 |

## Improvement over Random

| Checkpoint | R@1 a2m (x) | R@5 a2m (x) | R@10 a2m (x) | R@1 m2a (x) | R@5 m2a (x) | R@10 m2a (x) |
|-----------|-------------|-------------|-------------|-------------|-------------|-------------|
| gate2_ep45 | 11.3x | 10.6x | 8.8x | 13.3x | 12.6x | 9.6x |
| runA_best_ep7 | 15.4x | 8.8x | 7.1x | 15.4x | 11.6x | 9.1x |
| runB_ep5 | 9.2x | 8.3x | 6.3x | 11.3x | 10.3x | 8.2x |
| runB_ep10 | 13.8x | 9.3x | 7.6x | 10.8x | 11.9x | 8.9x |
| runC_best_ep4 | 14.8x | 11.2x | 8.9x | 15.4x | 11.6x | 10.0x |
| runC_ep13 | 15.9x | 10.2x | 8.2x | 15.4x | 12.6x | 9.7x |

## Hard Negative Analysis

| Checkpoint | vs Same-Piece-Diff-Time | vs Random | Decision |
|-----------|------------------------|-----------|----------|
| gate2_ep45 | 80.4% | 87.0% | GO |
| runA_best_ep7 | 74.8% | 80.6% | GO |
| runB_ep5 | 70.4% | 72.4% | WEAK-GO |
| runB_ep10 | 73.6% | 78.2% | GO |
| runC_best_ep4 | 81.2% | 86.2% | GO |
| runC_ep13 | 76.6% | 80.8% | GO |

## Mean / Median Rank (lower is better, out of 256)

| Checkpoint | Mean a2m | Median a2m | Mean m2a | Median m2a |
|-----------|----------|-----------|----------|------------|
| gate2_ep45 | 37.4 | 18.0 | 31.6 | 16.0 |
| runA_best_ep7 | 55.7 | 30.5 | 45.1 | 20.0 |
| runB_ep5 | 61.8 | 33.0 | 55.3 | 23.0 |
| runB_ep10 | 55.1 | 28.0 | 50.2 | 20.5 |
| runC_best_ep4 | 39.6 | 19.0 | 36.0 | 16.0 |
| runC_ep13 | 48.2 | 22.0 | 41.4 | 16.0 |

## Configuration Notes

- **Run A**: No F.normalize before domain head. Linear lambda 0→1.
- **Run B**: F.normalize before domain head. Linear lambda 0→1.
- **Run C**: F.normalize + warmup_ramp_cap schedule (warmup=2000, ramp=6000, lambda_max=0.8). 3 LR groups.
- **All runs** start from Gate 2 checkpoint (epoch 45).
- **Evaluation**: Identical structured pool (256 candidates, 500 queries, seed 42) for all checkpoints.
