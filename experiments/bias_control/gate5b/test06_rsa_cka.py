#!/usr/bin/env python3
"""
Gate 5B — Test 6: RSA/CKA Between Layers.

Measures Representational Similarity Analysis (RSA) and Centered Kernel
Alignment (CKA) between all pairs of transformer layers across the audio
and MIDI encoders.

Method:
  1. Register forward hooks on all 8 transformer layers (4 audio + 4 MIDI)
  2. Forward pass ~500 segments through the model
  3. Collect activations: [N, T, D] per layer -> mean-pool over T -> [N, D]
  4. Compute pairwise RSA: Spearman correlation between RDMs
  5. Compute pairwise linear CKA
  6. Output: 8x8 RSA matrix, 8x8 CKA matrix

Interpretation:
  - High RSA/CKA between audio layer i and MIDI layer j means those layers
    form similar representational geometries — evidence of cross-modal alignment.
  - D0 (no descriptor) should show lower cross-encoder alignment than
    augmented models if descriptors truly bridge the modality gap.

Usage:
    python experiments/bias_control/gate5b/test06_rsa_cka.py \
        --model models/gate5b/d4a4/best_model.pt

    python experiments/bias_control/gate5b/test06_rsa_cka.py --all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default checkpoint paths
DEFAULT_CHECKPOINTS = {
    'D0': 'models/gate5b/D0/best_model.pt',
    'd4a4': 'models/gate5b/d4a4/best_model.pt',
    'a4r': 'models/gate5b/a4r/best_model.pt',
    'd4-a4r': 'models/gate5b/d4-a4r/best_model.pt',
}

# Layer names in canonical order
LAYER_NAMES = [
    'audio_0', 'audio_1', 'audio_2', 'audio_3',
    'midi_0', 'midi_1', 'midi_2', 'midi_3',
]

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning(
        "scipy not available — RSA computation will be skipped. "
        "Install with: pip install scipy"
    )


# ---------------------------------------------------------------------------
# RSA / CKA computation
# ---------------------------------------------------------------------------

def compute_rdm(X: torch.Tensor) -> torch.Tensor:
    """
    Compute Representational Dissimilarity Matrix.

    Args:
        X: [N, D] activation matrix

    Returns:
        [N, N] dissimilarity matrix (1 - cosine similarity)
    """
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    sim = X_norm @ X_norm.T  # [N, N]
    return 1 - sim  # dissimilarity


def compute_rsa(rdm1: torch.Tensor, rdm2: torch.Tensor) -> Tuple[float, float]:
    """
    RSA = Spearman correlation between upper triangles of two RDMs.

    Args:
        rdm1: [N, N] dissimilarity matrix
        rdm2: [N, N] dissimilarity matrix

    Returns:
        (correlation, p-value)
    """
    if not HAS_SCIPY:
        return float('nan'), float('nan')

    idx = torch.triu_indices(rdm1.size(0), rdm1.size(1), offset=1)
    v1 = rdm1[idx[0], idx[1]].cpu().numpy()
    v2 = rdm2[idx[0], idx[1]].cpu().numpy()
    corr, pval = spearmanr(v1, v2)
    return float(corr), float(pval)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA (Centered Kernel Alignment).

    Args:
        X: [N, D1] activation matrix
        Y: [N, D2] activation matrix

    Returns:
        CKA score in [0, 1]
    """
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    # HSIC via linear kernel
    # HSIC(X,Y) = ||Y^T X||_F^2 / (N-1)^2  (simplified for linear CKA)
    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    XtX = X @ X.T  # [N, N]
    YtY = Y @ Y.T  # [N, N]

    hsic_xy = (XtX * YtY).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()

    return float(hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8))


# ---------------------------------------------------------------------------
# Hook-based activation collection
# ---------------------------------------------------------------------------

def _get_encoders(model: nn.Module) -> Tuple[nn.Module, nn.Module]:
    """
    Retrieve audio_encoder and midi_encoder from any Gate 4.x model.

    All Gate 4.x models wrap a CrossModalModel as model.base_model,
    which contains .audio_encoder and .midi_encoder.
    """
    if hasattr(model, 'base_model'):
        base = model.base_model
    else:
        base = model

    audio_enc = base.audio_encoder
    midi_enc = base.midi_encoder
    return audio_enc, midi_enc


@torch.no_grad()
def collect_layer_activations(
    model: nn.Module,
    dataset,
    device: torch.device,
    n_samples: int = 500,
    batch_size: int = 32,
    num_workers: int = 8,
) -> Dict[str, torch.Tensor]:
    """
    Collect mean-pooled activations from all 8 transformer layers.

    Registers forward hooks on audio_encoder.transformer.layers[0..3]
    and midi_encoder.transformer.layers[0..3], runs forward passes,
    and returns mean-pooled activations per layer.

    Args:
        model: Gate 4.x model in eval mode
        dataset: MaestroSegmentDataset (validation)
        device: torch device
        n_samples: number of segments to process
        batch_size: batch size for forward passes
        num_workers: DataLoader workers

    Returns:
        Dict mapping layer name -> [N, D] tensor on CPU
            e.g. {'audio_0': [N, 1024], ..., 'midi_3': [N, 512]}
    """
    from src.bias_control.datasets.maestro_segments import collate_segments

    audio_enc, midi_enc = _get_encoders(model)

    # Storage for hooked activations (per-batch)
    hook_storage: Dict[str, List[torch.Tensor]] = {name: [] for name in LAYER_NAMES}

    # --- Register hooks ---
    hooks = []

    def _make_hook(layer_name: str):
        """Create a hook that stores the mean-pooled output of a transformer layer."""
        def hook_fn(module, input, output):
            # output from TransformerEncoderLayer: [B, T, D]
            # Mean-pool over T dimension -> [B, D]
            if isinstance(output, torch.Tensor):
                pooled = output.mean(dim=1)  # [B, D]
            else:
                # Shouldn't happen for standard TransformerEncoderLayer
                pooled = output[0].mean(dim=1)
            hook_storage[layer_name].append(pooled.cpu())
        return hook_fn

    # Audio transformer layers
    for i, layer in enumerate(audio_enc.transformer.layers):
        h = layer.register_forward_hook(_make_hook(f'audio_{i}'))
        hooks.append(h)

    # MIDI transformer layers
    for i, layer in enumerate(midi_enc.transformer.layers):
        h = layer.register_forward_hook(_make_hook(f'midi_{i}'))
        hooks.append(h)

    # --- Forward pass ---
    # Limit dataset to n_samples
    from torch.utils.data import Subset
    indices = list(range(min(n_samples, len(dataset))))
    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_segments,
        pin_memory=True,
        prefetch_factor=2,
    )

    model.eval()
    for batch in tqdm(loader, desc="Collecting layer activations"):
        audio = batch['audio'].to(device, non_blocking=True)
        midi_pitch = batch['midi_pitch'].to(device, non_blocking=True)
        midi_velocity = batch['midi_velocity'].to(device, non_blocking=True)
        midi_duration = batch['midi_duration'].to(device, non_blocking=True)
        midi_mask = batch['midi_mask'].to(device, non_blocking=True)

        # Forward pass triggers all hooks
        model(audio, midi_pitch, midi_velocity, midi_duration, midi_mask)

    # --- Remove hooks ---
    for h in hooks:
        h.remove()

    # --- Concatenate batches -> [N, D] ---
    activations = {}
    for name in LAYER_NAMES:
        if hook_storage[name]:
            activations[name] = torch.cat(hook_storage[name], dim=0)
            logger.info(f"  {name}: {activations[name].shape}")
        else:
            logger.warning(f"  {name}: NO activations collected!")
            activations[name] = torch.empty(0)

    return activations


# ---------------------------------------------------------------------------
# Main test logic
# ---------------------------------------------------------------------------

def compute_rsa_cka_matrices(
    activations: Dict[str, torch.Tensor],
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """
    Compute 8x8 RSA and CKA matrices from layer activations.

    Args:
        activations: Dict mapping layer name -> [N, D] tensor

    Returns:
        (rsa_matrix, rsa_pval_matrix, cka_matrix): each 8x8 list of lists
    """
    n_layers = len(LAYER_NAMES)
    rsa_matrix = [[0.0] * n_layers for _ in range(n_layers)]
    rsa_pval_matrix = [[0.0] * n_layers for _ in range(n_layers)]
    cka_matrix = [[0.0] * n_layers for _ in range(n_layers)]

    # Pre-compute RDMs for RSA
    rdms = {}
    if HAS_SCIPY:
        logger.info("Computing RDMs for RSA...")
        for name in LAYER_NAMES:
            X = activations[name]
            if X.numel() == 0:
                rdms[name] = None
            else:
                rdms[name] = compute_rdm(X)

    # Compute pairwise matrices
    logger.info("Computing pairwise RSA and CKA...")
    for i, name_i in enumerate(LAYER_NAMES):
        Xi = activations[name_i]
        for j, name_j in enumerate(LAYER_NAMES):
            Xj = activations[name_j]

            if Xi.numel() == 0 or Xj.numel() == 0:
                rsa_matrix[i][j] = float('nan')
                rsa_pval_matrix[i][j] = float('nan')
                cka_matrix[i][j] = float('nan')
                continue

            # CKA
            cka_matrix[i][j] = linear_cka(Xi, Xj)

            # RSA
            if HAS_SCIPY and rdms.get(name_i) is not None and rdms.get(name_j) is not None:
                corr, pval = compute_rsa(rdms[name_i], rdms[name_j])
                rsa_matrix[i][j] = corr
                rsa_pval_matrix[i][j] = pval
            else:
                rsa_matrix[i][j] = float('nan')
                rsa_pval_matrix[i][j] = float('nan')

    return rsa_matrix, rsa_pval_matrix, cka_matrix


def _summarize_cross_encoder(matrix: List[List[float]], label: str):
    """Log summary statistics for cross-encoder (audio vs MIDI) block."""
    cross_values = []
    for i in range(4):       # audio layers
        for j in range(4, 8):  # midi layers
            val = matrix[i][j]
            if not (val != val):  # not NaN
                cross_values.append(val)

    if cross_values:
        arr = np.array(cross_values)
        logger.info(
            f"  {label} cross-encoder (audio x midi): "
            f"mean={arr.mean():.4f}, max={arr.max():.4f}, min={arr.min():.4f}"
        )
    else:
        logger.info(f"  {label} cross-encoder: no valid values")


def evaluate_single(model_path, maestro_dir, device, num_workers, seed, n_samples):
    """Run RSA/CKA test on one checkpoint."""
    from experiments.bias_control.gate5b.harness import (
        setup_gate5b_test, get_output_dir, save_test_result,
    )
    from experiments.bias_control.gate5b.checkpoint_loader import get_eval_batch_size

    class Args:
        pass
    args = Args()
    args.model = model_path
    args.maestro_dir = maestro_dir
    args.device = device
    args.seed = seed
    args.num_workers = num_workers
    args.output = None

    model, meta, dataset, index = setup_gate5b_test(args)
    descriptor = meta['descriptor']
    batch_size = get_eval_batch_size(descriptor)

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(f"Collecting activations from {n_samples} samples...")
    activations = collect_layer_activations(
        model, dataset, dev,
        n_samples=n_samples,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    actual_n = activations[LAYER_NAMES[0]].shape[0] if activations[LAYER_NAMES[0]].numel() > 0 else 0
    logger.info(f"Collected {actual_n} samples across {len(LAYER_NAMES)} layers")

    logger.info("Computing RSA and CKA matrices...")
    rsa_matrix, rsa_pval_matrix, cka_matrix = compute_rsa_cka_matrices(activations)

    # Log summary
    _summarize_cross_encoder(cka_matrix, "CKA")
    if HAS_SCIPY:
        _summarize_cross_encoder(rsa_matrix, "RSA")

    result = {
        'rsa_matrix': rsa_matrix,
        'rsa_pval_matrix': rsa_pval_matrix,
        'cka_matrix': cka_matrix,
        'layer_names': LAYER_NAMES,
        'n_samples': actual_n,
        'has_scipy': HAS_SCIPY,
    }

    output_dir = get_output_dir(args, meta)
    save_test_result(result, output_dir, 'test06_rsa_cka', meta, seed=seed)

    return result, meta


def _format_matrix(matrix: List[List[float]], layer_names: List[str], label: str):
    """Pretty-print an 8x8 matrix."""
    logger.info(f"\n{label}:")
    header = f"{'':>10}" + "".join(f"{n:>10}" for n in layer_names)
    logger.info(header)
    logger.info("-" * (10 + 10 * len(layer_names)))
    for i, name in enumerate(layer_names):
        row = f"{name:>10}"
        for j in range(len(layer_names)):
            val = matrix[i][j]
            if val != val:  # NaN check
                row += f"{'NaN':>10}"
            else:
                row += f"{val:>10.4f}"
        logger.info(row)


def main():
    parser = argparse.ArgumentParser(description='Gate 5B — Test 6: RSA/CKA Between Layers')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--maestro-dir', type=str, default='data/maestro_v3/maestro-v3.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Number of segments to collect activations from')

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Provide --model PATH or --all")

    models = {}
    if args.all:
        models = dict(DEFAULT_CHECKPOINTS)
    else:
        models = {'custom': args.model}

    all_results = {}

    for arm, path in models.items():
        if not Path(path).exists():
            logger.warning(f"Checkpoint not found: {path} -- skipping {arm}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"RSA/CKA ANALYSIS: {arm}")
        logger.info(f"{'='*60}")

        start = time.time()
        result, meta = evaluate_single(
            path, args.maestro_dir, args.device,
            args.num_workers, args.seed, args.n_samples,
        )
        elapsed = time.time() - start

        all_results[arm] = {
            'descriptor': meta['descriptor'],
            'n_samples': result['n_samples'],
            'cka_matrix': result['cka_matrix'],
            'rsa_matrix': result['rsa_matrix'],
            'time_s': elapsed,
        }

        # Print matrices for this arm
        _format_matrix(result['cka_matrix'], LAYER_NAMES, f"CKA Matrix ({arm})")
        if HAS_SCIPY:
            _format_matrix(result['rsa_matrix'], LAYER_NAMES, f"RSA Matrix ({arm})")

        logger.info(f"\nCompleted {arm} in {elapsed:.1f}s")

    # Cross-arm summary
    if len(all_results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("CROSS-ENCODER ALIGNMENT SUMMARY")
        logger.info(f"{'='*60}")

        header = f"{'Arm':<12} {'CKA mean':>10} {'CKA max':>10}"
        if HAS_SCIPY:
            header += f" {'RSA mean':>10} {'RSA max':>10}"
        logger.info(header)
        logger.info("-" * len(header))

        for arm, data in all_results.items():
            # Cross-encoder block: audio rows (0-3) x midi cols (4-7)
            cka_cross = []
            rsa_cross = []
            for i in range(4):
                for j in range(4, 8):
                    cv = data['cka_matrix'][i][j]
                    rv = data['rsa_matrix'][i][j]
                    if cv == cv:  # not NaN
                        cka_cross.append(cv)
                    if rv == rv:
                        rsa_cross.append(rv)

            line = f"{arm:<12}"
            if cka_cross:
                line += f" {np.mean(cka_cross):>10.4f} {np.max(cka_cross):>10.4f}"
            else:
                line += f" {'N/A':>10} {'N/A':>10}"

            if HAS_SCIPY and rsa_cross:
                line += f" {np.mean(rsa_cross):>10.4f} {np.max(rsa_cross):>10.4f}"
            elif HAS_SCIPY:
                line += f" {'N/A':>10} {'N/A':>10}"

            logger.info(line)


if __name__ == '__main__':
    main()
