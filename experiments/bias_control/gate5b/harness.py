#!/usr/bin/env python3
"""
Gate 5B — Test Harness.

Common infrastructure for all Gate 5B scientific validation tests:
- Model loading via checkpoint_loader
- Dataset loading (MAESTRO validation)
- Embedding extraction (wraps evaluate_structured_pool)
- Standard result formatting and serialization

Usage from test scripts:
    from experiments.bias_control.gate5b.harness import (
        setup_gate5b_test, extract_embeddings, save_test_result, add_common_args,
    )

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    model, meta, dataset, index = setup_gate5b_test(args)
    audio_embs, midi_embs = extract_embeddings(model, dataset, args.device, meta['descriptor'])
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)

# Default results directory
GATE5B_RESULTS_DIR = Path('data/gate5b_results')


def add_common_args(parser: argparse.ArgumentParser):
    """Add standard Gate 5B CLI arguments to a parser."""
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: data/gate5b_results/<arm>/)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)


def _get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'


def setup_gate5b_test(
    args,
) -> Tuple[nn.Module, Dict[str, Any], Any, Dict]:
    """
    Load model + dataset + index for a Gate 5B test.

    Returns:
        (model, metadata, dataset, index)
    """
    from experiments.bias_control.gate5b.checkpoint_loader import (
        load_model_from_checkpoint,
    )
    from src.bias_control.datasets.maestro_segments import MaestroSegmentDataset
    from experiments.bias_control.evaluate_structured_pool import build_segment_index

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    logger.info(f"Loading model from {args.model}")
    model, meta = load_model_from_checkpoint(args.model, device=device)

    logger.info("Loading MAESTRO validation dataset...")
    dataset = MaestroSegmentDataset(
        maestro_dir=Path(args.maestro_dir),
        segment_len=4.0,
        hop=1.0,
        split='validation',
    )

    logger.info("Building segment index...")
    index = build_segment_index(dataset)
    logger.info(f"Index: {len(index['by_piece'])} pieces, {len(index['by_composer'])} composers")

    return model, meta, dataset, index


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    dataset,
    device: str,
    descriptor: str,
    num_workers: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract all embeddings using evaluate_structured_pool's infrastructure.

    Returns:
        (audio_embs, midi_embs): Both [N, 256] on CPU
    """
    from experiments.bias_control.gate5b.checkpoint_loader import get_eval_batch_size
    from experiments.bias_control.evaluate_structured_pool import extract_all_embeddings

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    batch_size = get_eval_batch_size(descriptor)

    logger.info(f"Extracting embeddings: batch_size={batch_size}, descriptor={descriptor}")
    start = time.time()
    audio_embs, midi_embs = extract_all_embeddings(
        model, dataset, dev,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    elapsed = time.time() - start
    logger.info(f"Extraction done in {elapsed:.1f}s — audio={audio_embs.shape}, midi={midi_embs.shape}")
    return audio_embs, midi_embs


def _embeddings_cache_path(output_dir: Path) -> Path:
    """Path for cached normal embeddings."""
    return output_dir / 'embeddings_normal.npz'


def save_embeddings(
    audio_embs: torch.Tensor,
    midi_embs: torch.Tensor,
    output_dir: Path,
):
    """Save normal embeddings to cache for reuse across tests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _embeddings_cache_path(output_dir)
    np.savez_compressed(
        path,
        audio_embs=audio_embs.cpu().numpy(),
        midi_embs=midi_embs.cpu().numpy(),
    )
    logger.info(f"Embeddings cached: {path} ({audio_embs.shape[0]} segments)")


def load_cached_embeddings(output_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load cached normal embeddings. Returns (audio_embs, midi_embs) as CPU tensors.
    Raises FileNotFoundError if cache doesn't exist.
    """
    path = _embeddings_cache_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"No cached embeddings at {path}")
    data = np.load(path)
    audio_embs = torch.from_numpy(data['audio_embs'])
    midi_embs = torch.from_numpy(data['midi_embs'])
    logger.info(f"Loaded cached embeddings: {path} ({audio_embs.shape[0]} segments)")
    return audio_embs, midi_embs


def get_normal_embeddings(
    model: nn.Module,
    dataset,
    device: str,
    descriptor: str,
    output_dir: Path,
    num_workers: int = 8,
    force_extract: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get normal embeddings — from cache if available, otherwise extract and cache.

    This is the main entry point for tests that need unmodified embeddings.
    """
    if not force_extract:
        try:
            return load_cached_embeddings(output_dir)
        except FileNotFoundError:
            pass

    audio_embs, midi_embs = extract_embeddings(
        model, dataset, device, descriptor, num_workers=num_workers,
    )
    save_embeddings(audio_embs, midi_embs, output_dir)
    return audio_embs, midi_embs


def get_output_dir(args, meta: Dict[str, Any]) -> Path:
    """Determine output directory for this arm's results."""
    if args.output:
        return Path(args.output)
    descriptor = meta['descriptor']
    # Map d0 -> D0 for directory name consistency
    arm_dir = 'D0' if descriptor == 'd0' else descriptor
    return GATE5B_RESULTS_DIR / arm_dir


def save_test_result(
    result: Dict[str, Any],
    output_dir: Path,
    test_name: str,
    meta: Dict[str, Any],
    seed: int = 42,
):
    """
    Save test result with standard metadata.

    Writes to output_dir/test_name.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{test_name}.json'

    result['_metadata'] = {
        'test_name': test_name,
        'descriptor': meta.get('descriptor'),
        'epoch': meta.get('epoch'),
        'best_S': meta.get('best_S'),
        'checkpoint': meta.get('checkpoint_path'),
        'seed': seed,
        'git_hash': _get_git_hash(),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Result saved: {output_path}")
    return output_path
